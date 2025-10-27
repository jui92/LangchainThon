# -*- coding: utf-8 -*-
################################################################################
# Job Helper Bot (Selenium ONLY for company summary + Speed-ups)
# - 회사 요약(회사명/소개/직무명 + 주요업무/자격요건/우대사항)을 Selenium 기반으로 수집
# - Wanted __NEXT_DATA__ 병합, FAST 모드, 적절한 클릭/스크롤 최적화
# - 기존 UI/LLM/뉴스/회사홈페이지/이력서-RAG/채점/팔로업 흐름 유지
################################################################################

import os, re, json, urllib.parse, time, io, random, shutil, tempfile, traceback
from typing import Optional, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
import html2text
import streamlit as st
import pandas as pd
import numpy as np

# ================== 기본 설정 ==================
st.set_page_config(page_title="Job Helper Bot", page_icon="🤖", layout="wide")
st.title("Job Helper Bot : 자소서 생성 / 모의 면접")

# ================== OpenAI ==================
try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop()

API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
if not API_KEY:
    API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password")
if not API_KEY:
    st.stop()
client = OpenAI(api_key=API_KEY)

with st.sidebar:
    st.subheader("모델 / 크롤링 설정")
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델(내부용)", ["text-embedding-3-small","text-embedding-3-large"], index=0)
    SELENIUM_TIMEOUT = st.slider("Selenium 대기(초)", 6, 30, 14)
    FAST_MODE = st.toggle("FAST 모드(클릭/스크롤 축소)", value=True)

# ================== 공통 유틸 ==================
def normalize_url(u: str) -> Optional[str]:
    if not u: return None
    u = u.strip()
    if not re.match(r"^https?://", u): u = "https://" + u
    parts = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

def html_to_text(html_str: str) -> str:
    conv = html2text.HTML2Text()
    conv.ignore_links = True
    conv.ignore_images = True
    conv.body_width = 0
    txt = conv.handle(html_str or "")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

# ================== (NEW) Selenium 로더 ==================
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

def _pick_chrome_binary() -> Optional[str]:
    cands = [
        os.getenv("CHROME_BIN"), os.getenv("GOOGLE_CHROME_BIN"),
        shutil.which("chromium"), shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        "/usr/bin/chromium","/usr/bin/chromium-browser",
        "/usr/bin/google-chrome","/usr/bin/google-chrome-stable",
    ]
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None

def _build_chrome(headless: bool = True):
    opts = ChromeOptions()
    if headless: opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1440,2400")
    opts.add_argument("--lang=ko-KR")
    binpath = _pick_chrome_binary()
    if binpath: opts.binary_location = binpath
    # Selenium Manager가 드라이버를 자동 준비
    driver = webdriver.Chrome(options=opts)
    return driver

def _click_by_text_candidates(driver, texts: List[str], per=10):
    for t in texts:
        try:
            xp1 = f"//*[normalize-space(text())='{t}']"
            xp2 = f"//*[contains(normalize-space(text()), '{t}')]"
            for xp in (xp1, xp2):
                els = driver.find_elements(By.XPATH, xp)
                for el in els[:per]:
                    try:
                        driver.execute_script("arguments[0].click();", el)
                        time.sleep(0.1 if FAST_MODE else 0.2)
                    except Exception:
                        continue
        except Exception:
            continue

def _click_many_css(driver, selectors: List[str], per=10):
    for sel in selectors:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            for el in els[:per]:
                try:
                    driver.execute_script("arguments[0].click();", el)
                    time.sleep(0.08 if FAST_MODE else 0.18)
                except Exception:
                    continue
        except Exception:
            continue

def _expand_wanted(driver):
    sel = [
        "[data-qa='btn-read-more']","[data-qa='job-header__more']",
        "button[aria-expanded='false']","[role='button'][class*='More']",
        "div[aria-expanded='false']",
    ]
    _click_many_css(driver, sel, per=(6 if FAST_MODE else 10))
    _click_by_text_candidates(driver, [
        "더보기","전체보기","상세보기","자세히","모두 보기",
        "주요업무","자격요건","우대사항","기업/팀 소개"
    ], per=(5 if FAST_MODE else 10))

def _expand_saramin(driver):
    sel = [".btn_more",".btnMore",".btn-detail",".btn_toggle",
           "[aria-expanded='false']","[role='button']","button[class*='more'], a[class*='more']"]
    _click_many_css(driver, sel, per=(5 if FAST_MODE else 10))
    _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","기업정보","상세정보"], per=(5 if FAST_MODE else 10))

def _expand_jobkorea(driver):
    sel = [".btnFold",".btnToggleRead",".btn_more",
           "[aria-expanded='false']","[role='button']","button[class*='More'], a[class*='More']"]
    _click_many_css(driver, sel, per=(5 if FAST_MODE else 10))
    _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","기업정보","상세보기"], per=(5 if FAST_MODE else 10))

def extract_wanted_from_next_html(html: str) -> str:
    """Wanted의 __NEXT_DATA__ JSON에서 채용 본문 키만 추출해 텍스트로 병합"""
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        tag = soup.select_one("script#__NEXT_DATA__")
        if not tag: return ""
        raw = (tag.string or tag.text or "").strip()
        data = json.loads(raw)
    except Exception:
        return ""
    key_whitelist = ["job","position","title","desc","description",
                     "responsibilit","duty","role","skill","stack",
                     "require","qualification","prefer","plus","nice"]
    def _walk(d, out):
        if isinstance(d, dict):
            for k, v in d.items():
                if any(t in str(k).lower() for t in key_whitelist):
                    if isinstance(v, str): out.append(v)
                    elif isinstance(v, list):
                        for it in v:
                            if isinstance(it, str): out.append(it)
                            elif isinstance(it, dict):
                                for _, subv in it.items():
                                    if isinstance(subv, str): out.append(subv)
                    elif isinstance(v, dict):
                        for _, subv in v.items():
                            if isinstance(subv, str): out.append(subv)
                _walk(v, out)
        elif isinstance(d, list):
            for it in d: _walk(it, out)
    bucket=[]; _walk(data, bucket)
    seen=set(); lines=[]
    for t in bucket:
        s=re.sub(r"\s+"," ", (t or "")).strip()
        if len(s)>2 and s not in seen:
            seen.add(s); lines.append(s)
    return "\n".join(lines[:900])

def selenium_get_html(url: str, timeout: int = 14) -> str:
    driver = _build_chrome(headless=True)
    try:
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        try:
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, "//*")))
        except TimeoutException:
            pass

        host = urllib.parse.urlsplit(url).netloc.lower()
        # 공통: 펼치기 버튼 터치
        _click_by_text_candidates(driver, ["더보기","상세보기","자세히 보기","전체보기","Read more","More"],
                                  per=(5 if FAST_MODE else 10))
        _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","Requirements","Responsibilities","Preferred"],
                                  per=(5 if FAST_MODE else 10))
        # 사이트별 추가 확장
        if "wanted.co.kr" in host: _expand_wanted(driver)
        if "saramin" in host:     _expand_saramin(driver)
        if "jobkorea" in host:    _expand_jobkorea(driver)

        # 짧은 스크롤 여러 번
        loops = 5 if FAST_MODE else 8
        for _ in range(loops):
            try:
                driver.execute_script("window.scrollBy(0, 1200);"); time.sleep(0.1 if FAST_MODE else 0.25)
            except Exception:
                break

        html = driver.page_source or ""
        # Wanted의 __NEXT_DATA__ 병합
        if "wanted.co.kr" in host:
            try:
                extra = extract_wanted_from_next_html(html)
                if extra:
                    html += "\n<div id='__WANTED_NEXT_EXTRACT__'>" + \
                            "".join([f"<p>{line}</p>" for line in extra.split("\n")]) + "</div>"
            except Exception:
                pass
        return html
    finally:
        try: driver.quit()
        except Exception: pass

def fetch_all_text_selenium(url: str) -> Tuple[str, Dict, Optional[str]]:
    url_n = normalize_url(url)
    if not url_n: return "", {"error":"invalid_url"}, None
    try:
        html = selenium_get_html(url_n, timeout=SELENIUM_TIMEOUT)
    except Exception as e:
        st.error(f"Selenium 로드 실패: {e}")
        st.code("".join(traceback.format_exc()))
        return "", {"source":"selenium_error","len":0,"url_final":url_n}, None
    if not html or len(html) < 200:
        return "", {"source":"selenium_failed","len":0,"url_final":url_n}, None
    txt = html_to_text(html)
    return txt, {"source":"selenium","len":len(txt),"url_final":url_n}, html

# ================== 메타/섹션 보조 추출 ==================
def extract_company_meta_from_html(html: Optional[str]) -> Dict[str,str]:
    meta = {"company_name":"","company_intro":"","job_title":""}
    if not html: return meta
    try:
        soup = BeautifulSoup(html, "html.parser")
        cand=[]
        og = soup.find("meta", {"property":"og:site_name"})
        if og and og.get("content"): cand.append(og["content"])
        app = soup.find("meta", {"name":"application-name"})
        if app and app.get("content"): cand.append(app["content"])
        if soup.title and soup.title.string: cand.append(soup.title.string)
        cand = [re.split(r"[\-\|\·\—]", c)[0].strip() for c in cand if c]
        cand = [c for c in cand if 2 <= len(c) <= 40]
        meta["company_name"] = cand[0] if cand else ""
        md = soup.find("meta", {"name":"description"}) or soup.find("meta", {"property":"og:description"})
        if md and md.get("content"):
            meta["company_intro"] = re.sub(r"\s+"," ", md["content"]).strip()[:500]
        jt = ""
        ogt = soup.find("meta", {"property":"og:title"})
        if ogt and ogt.get("content"): jt = ogt["content"]
        if not jt:
            h1 = soup.find("h1")
            if h1 and h1.get_text(): jt = h1.get_text(strip=True)
        if not jt:
            h2 = soup.find("h2")
            if h2 and h2.get_text(): jt = h2.get_text(strip=True)
        meta["job_title"] = re.sub(r"\s+"," ", jt).strip()[:120]
    except Exception:
        pass
    return meta

def rule_based_sections(raw_text: str) -> dict:
    txt = re.sub(r"\r", "", raw_text or "").strip()
    lines = [re.sub(r"\s+", " ", l).strip(" -•·▶▪️") for l in txt.split("\n")]
    lines = [l for l in lines if l]

    hdr_resp = re.compile(r"(주요\s*업무|담당\s*업무|Role|Responsibilities?)", re.I)
    hdr_qual = re.compile(r"(자격\s*요건|지원\s*자격|Requirements?|Qualifications?)", re.I)
    hdr_pref = re.compile(r"(우대\s*사항|우대|선호|Preferred|Nice\s*to\s*have|Plus)", re.I)

    bucket = None
    out = {"responsibilities": [], "qualifications": [], "preferences": []}

    def push(line, b):
        if line and len(line) > 1 and line not in out[b]:
            out[b].append(line[:180])

    for l in lines:
        if hdr_resp.search(l): bucket = "responsibilities"; continue
        if hdr_qual.search(l): bucket = "qualifications"; continue
        if hdr_pref.search(l): bucket = "preferences"; continue
        if bucket is None:
            if hdr_pref.search(l):
                bucket = "preferences"
            elif any(k in l.lower() for k in ["java","python","spark","airflow","kafka","ml","sql"]):
                bucket = "responsibilities"
            else:
                continue
        push(l, bucket)

    kw_pref = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
    remain_qual = []
    for q in out["qualifications"]:
        if kw_pref.search(q): out["preferences"].append(q)
        else: remain_qual.append(q)
    out["qualifications"] = remain_qual

    for k in out:
        seen=set(); clean=[]
        for s in out[k]:
            s = re.sub(r"\s+", " ", s).strip(" -•·▶▪️").strip()
            if s and s not in seen:
                seen.add(s); clean.append(s)
        out[k] = clean[:12]
    return out

# ================== LLM 정제 (채용 공고 → 구조 JSON) ==================
PROMPT_SYSTEM_STRUCT = ("너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
                        "입력 텍스트는 포털 광고 문구, UI잔재, 복수 직무가 섞여 있을 수 있다. "
                        "한국어로 간결하고 중복없이 정제하라.")

def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
    ctx = (raw_text or "").strip()
    if len(ctx) > 14000: ctx = ctx[:14000]

    user_msg = {"role": "user",
                "content": ("다음 채용 공고 원문을 구조화해줘.\n\n"
                            f"[힌트] 회사명 후보: {meta_hint.get('company_name','')}\n"
                            f"[힌트] 직무명 후보: {meta_hint.get('job_title','')}\n"
                            "--- 원문 시작 ---\n"
                            f"{ctx}\n"
                            "--- 원문 끝 ---\n\n"
                            "JSON으로만 답하고, 키는 반드시 아래만 포함:\n"
                            "{"
                            "\"company_name\": str, "
                            "\"company_intro\": str, "
                            "\"job_title\": str, "
                            "\"responsibilities\": [str], "
                            "\"qualifications\": [str], "           
                            "\"preferences\": [str]"               
                            "}\n"
                            "- '우대 사항(preferences)'은 역량/경험/지식 조건만 포함.\n"
                            "- 불릿/마커/이모지 제거, 문장 간결화, 중복 제거."),}

    try:
        resp = client.chat.completions.create(model=model, temperature=0.2,
                                              response_format={"type": "json_object"},
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg])
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        data = {"company_name": meta_hint.get("company_name",""),
                "company_intro": meta_hint.get("company_intro","원문이 정제되지 않았습니다."),
                "job_title": meta_hint.get("job_title",""),
                "responsibilities": [],
                "qualifications": [],
                "preferences": [],
                "error": str(e)}

    for k in ["responsibilities","qualifications","preferences"]:
        arr = data.get(k, [])
        if not isinstance(arr, list): arr = []
        clean_list=[]; seen=set()
        for it in arr:
            t = re.sub(r"\s+"," ", str(it)).strip(" -•·▶▪️").strip()
            if t and t not in seen:
                seen.add(t); clean_list.append(t[:180])
        data[k] = clean_list[:12]

    if len(data.get("preferences", [])) < 1:
        rb = rule_based_sections(ctx)
        if rb.get("preferences"):
            merged = data.get("preferences", []) + rb["preferences"]
            seen=set(); pref=[]
            for s in merged:
                s=s.strip()
                if s and s not in seen:
                    seen.add(s); pref.append(s)
            data["preferences"] = pref[:12]
        else:
            kw_pref = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
            remain=[]; moved=[]
            for q in data.get("qualifications", []):
                if kw_pref.search(q): moved.append(q)
                else: remain.append(q)
            if moved:
                data["preferences"] = moved[:12]
                data["qualifications"] = remain[:12]
    for k in ["company_name","company_intro","job_title"]:
        if k in data and isinstance(data[k], str):
            data[k] = re.sub(r"\s+"," ", data[k]).strip()
    return data

# ================== 파일 리더/임베딩/RAG 유틸 ==================
try:
    import pypdf
except Exception:
    pypdf = None

def read_pdf(data: bytes) -> str:
    if pypdf is None: return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
    except Exception:
        return ""

def read_docx(data: bytes) -> str:
    try:
        import docx2txt
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
            tmp.write(data); tmp.flush()
            text = docx2txt.process(tmp.name) or ""
            return text
    except Exception:
        return ""

def read_file_text(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith((".txt",".md")):
        for enc in ("utf-8","cp949","euc-kr"):
            try: return data.decode(enc)
            except Exception: continue
        return data.decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        return read_pdf(data)
    elif name.endswith(".docx"):
        return read_docx(data)
    return ""

def chunk(text: str, size: int = 600, overlap: int = 120) -> List[str]:
    t = re.sub(r"\s+"," ", text).strip()
    if not t: return []
    out, start = [], 0
    while start < len(t):
        end = min(len(t), start+size)
        out.append(t[start:end])
        if end == len(t): break
        start = max(0, end-overlap)
    return out

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=model_name, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs

def cosine_topk(matrix: np.ndarray, query_vec: np.ndarray, k: int = 4):
    if matrix.size == 0:
        return np.array([]), np.array([], dtype=int)
    qn = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn.T
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx

def retrieve_resume_chunks(query: str, k: int = 4):
    chs, embs = st.session_state.get("resume_chunks", []), st.session_state.get("resume_embeds", None)
    if not chs or embs is None:
        return []
    qv = embed_texts([query], EMBED_MODEL)
    scores, idxs = cosine_topk(embs, qv, k=k)
    return [(float(s), chs[int(i)]) for s, i in zip(scores, idxs)]

# ================== 회사 비전/인재상/뉴스 ==================
def http_get(url: str, timeout: int = 10) -> Optional[requests.Response]:
    try:
        r = requests.get(url,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
                                  "Accept-Language": "ko, en;q=0.9"}, timeout=timeout)
        if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
            return r
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_company_pages(home_url: str) -> Dict[str, List[str]]:
    out = {"vision": [], "talent": []}
    if not home_url: return out
    base = normalize_url(home_url)
    if not base: return out
    paths = ["","/","/about","/company","/about-us","/mission","/vision","/values","/culture","/careers","/talent","/people"]
    seen = set()
    for p in paths:
        url = (base.rstrip("/") + p) if p else base
        if url in seen: continue
        seen.add(url)
        r = http_get(url, timeout=8)
        if not r: continue
        soup = BeautifulSoup(r.text, "lxml")
        texts=[]
        for tag in soup.find_all(["h1","h2","h3","h4","p","li"]):
            try:
                t = tag.get_text(" ", strip=True)
            except Exception:
                continue
            if not t: continue
            t = re.sub(r"\s+"," ", t)
            if 6 <= len(t) <= 260:
                texts.append(t)
        for t in texts:
            low = t.lower()
            if any(k in low for k in ["talent","인재상","who we hire","people we"]):
                out["talent"].append(t)
            if any(k in low for k in ["비전","미션","핵심가치","가치","원칙","mission","vision","values","principle"]):
                out["vision"].append(t)
    for k in out:
        uniq=[]; s=set()
        for x in out[k]:
            x=x.strip()
            if x and x not in s:
                s.add(x); uniq.append(x[:200])
        out[k]=uniq[:12]
    return out

def _load_naver_keys():
    cid = os.getenv("NAVER_CLIENT_ID")
    csec = os.getenv("NAVER_CLIENT_SECRET")
    try:
        if hasattr(st, "secrets"):
            cid = cid or st.secrets.get("NAVER_CLIENT_ID", None)
            csec = csec or st.secrets.get("NAVER_CLIENT_SECRET", None)
    except Exception:
        pass
    return cid, csec

def naver_search_news(company: str, display: int = 5) -> List[Dict]:
    cid, csec = _load_naver_keys()
    if not (cid and csec): return []
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec}
    try:
        r = requests.get(url, headers=headers, params={"query": company, "display": display, "sort":"date"}, timeout=8)
        if r.status_code != 200: return []
        js=r.json()
        items=[]
        for it in js.get("items", []):
            title = re.sub(r"</?b>|&quot;|&apos;|&amp;|&lt;|&gt;", "", it.get("title","")).strip()
            items.append({"title": title, "link": it.get("link"), "pubDate": it.get("pubDate")})
        return items
    except Exception:
        return []

def google_news_rss(company: str, max_items: int = 5) -> List[Dict]:
    q = urllib.parse.quote(company)
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200: return []
        soup = BeautifulSoup(r.text, "xml")
        out=[]
        for it in soup.find_all("item")[:max_items]:
            out.append({"title": (it.title.get_text() if it.title else "").strip(),
                        "link": (it.link.get_text() if it.link else "").strip(),
                        "pubDate": (it.pubDate.get_text() if it.pubDate else "").strip()})
        return out
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=1200)
def fetch_latest_news(company: str, max_items: int = 5) -> List[Dict]:
    items = naver_search_news(company, display=max_items)
    if items: return items
    return google_news_rss(company, max_items=max_items)

# ================== 세션 상태 ==================
def _init_state():
    for k, v in {"clean_struct": None,
                 "resume_raw": "",
                 "resume_chunks": [],
                 "resume_embeds": None,
                 "current_question": "",
                 "answer_text": "",
                 "records": [],
                 "followups": [],
                 "selected_followup": "",
                 "followup_answer": "",
                 "last_result": None,
                 "last_followup_result": None,
                 "company_home": "",
                 "company_vision": [],
                 "company_talent": [],
                 "company_news": [],
                 "last_html": None }.items():
        if k not in st.session_state: st.session_state[k] = v
_init_state()

# ================== 1) 채용 공고 URL → 정제 (Selenium) ==================
st.header("1) 채용 공고 URL (Selenium 전용)")
url = st.text_input("채용 공고 상세 URL", placeholder="취업 포털 사이트의 URL을 입력하세요.")
st.text_input("회사 공식 홈페이지 URL (선택)", key="company_home", placeholder="회사 공식 홈페이지 URL을 입력하세요.")
company_name_override = st.text_input("회사명 수동 입력 (선택)", placeholder="예: 카카오헬스케어")

if st.button("원문 수집 → 정제 (Selenium ONLY)", type="primary"):
    if not url.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner("Selenium으로 원문 수집 중..."):
            raw, meta, html = fetch_all_text_selenium(url.strip())
            hint = extract_company_meta_from_html(html)
            st.session_state.last_html = html

        st.caption(f"수집 소스: {meta.get('source')} · 텍스트 길이: {meta.get('len')}")
        if not raw:
            st.error("원문 수집 실패(로그인/동적 렌더링/봇 차단 가능).")
        else:
            with st.spinner("LLM으로 정제 중..."):
                clean = llm_structurize(raw, hint, CHAT_MODEL)
            if not clean.get("preferences"):
                rb = rule_based_sections(raw)
                if rb.get("preferences"):
                    clean["preferences"] = rb["preferences"][:12]
            st.session_state.clean_struct = clean

            # 비전/인재상/뉴스 (동시 처리 + 캐시)
            with st.spinner("회사 비전/인재상/뉴스 확인 중..."):
                vis = []; tal = []; news_items = []
                def _do_pages():
                    if st.session_state.company_home.strip():
                        extra = fetch_company_pages(st.session_state.company_home.strip())
                        return extra.get("vision", []), extra.get("talent", [])
                    return [], []
                def _do_news():
                    cname = (company_name_override.strip() or
                             clean.get("company_name") or
                             hint.get("company_name") or "")
                    return fetch_latest_news(cname, max_items=5) if cname else []
                with ThreadPoolExecutor(max_workers=2) as ex:
                    fut_pages = ex.submit(_do_pages)
                    fut_news  = ex.submit(_do_news)
                    try: vis, tal = fut_pages.result()
                    except Exception: pass
                    try: news_items = fut_news.result()
                    except Exception: pass

                st.session_state.company_vision = vis
                st.session_state.company_talent = tal
                st.session_state.company_news = news_items

            st.success("정제 완료!")

# ================== 2) 회사 요약 ==================
st.header("2) 회사 요약")
clean = st.session_state.clean_struct
if clean:
    st.markdown(f"**회사명:** {clean.get('company_name','-')}")
    st.markdown(f"**간단한 회사 소개(요약):** {clean.get('company_intro','-')}")
    st.markdown(f"**모집 분야(직무명):** {clean.get('job_title','-')}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**주요 업무**")
        for b in clean.get("responsibilities", []): st.markdown(f"- {b}")
    with c2:
        st.markdown("**자격 요건**")
        for b in clean.get("qualifications", []): st.markdown(f"- {b}")
    with c3:
        st.markdown("**우대 사항**")
        prefs = clean.get("preferences", [])
        if prefs:
            for b in prefs: st.markdown(f"- {b}")
        else:
            st.caption("우대 사항이 명시되지 않았습니다.")
else:
    st.info("먼저 URL을 정제해 주세요.")

# ================== 회사 비전/인재상 & 최신 이슈 (있으면 표시) ==================
if st.session_state.company_vision or st.session_state.company_talent or st.session_state.company_news:
    st.divider()
    st.subheader("회사 비전/인재상 & 최신 이슈")
    colv, colt = st.columns(2)
    with colv:
        st.markdown("**비전/핵심가치 (스크래핑)**")
        for v in st.session_state.company_vision[:8]:
            st.markdown(f"- {v}")
        if not st.session_state.company_vision:
            st.caption("비전/핵심가치 정보를 찾지 못했습니다.")
    with colt:
        st.markdown("**인재상 (스크래핑)**")
        for t in st.session_state.company_talent[:8]:
            st.markdown(f"- {t}")
        if not st.session_state.company_talent:
            st.caption("인재상 정보를 찾지 못했습니다.")
    if st.session_state.company_news:
        st.markdown("**최신 뉴스(상위 3~5건)**")
        for n in st.session_state.company_news[:5]:
            st.markdown(f"- [{n.get('title','(제목 없음)')}]({n.get('link','#')})")

st.divider()

# ================== 3) 내 이력서/프로젝트 업로드 ==================
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader("여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True)
_RESUME_CHUNK = 500
_RESUME_OVLP  = 100

if st.button("이력서 인덱싱", type="secondary"):
    if not uploads:
        st.warning("파일을 업로드하세요.")
    else:
        all_text=[]
        for up in uploads:
            t = read_file_text(up)
            if t: all_text.append(t)
        resume_text = "\n\n".join(all_text)
        if not resume_text.strip():
            st.error("텍스트를 추출하지 못했습니다.")
        else:
            chunks = chunk(resume_text, size=_RESUME_CHUNK, overlap=_RESUME_OVLP)
            with st.spinner("이력서 벡터화 중..."):
                embeds = embed_texts(chunks, EMBED_MODEL)
            st.session_state.resume_raw = resume_text
            st.session_state.resume_chunks = chunks
            st.session_state.resume_embeds = embeds
            st.success(f"인덱싱 완료 (청크 {len(chunks)}개)")

st.divider()

# ================== 4) 이력서 기반 자소서 생성 ==================
st.header("4) 이력서 기반 자소서 생성")
topic = st.text_input("회사 요청 주제(선택)", placeholder="예: 직무 지원동기 / 협업 경험 / 문제해결 사례 등")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str) -> str:
    enrich = {"vision": st.session_state.company_vision[:6],
              "talent": st.session_state.company_talent[:6],
              "news": [n.get("title","") for n in st.session_state.company_news[:3]]}
    company = json.dumps({"clean":clean_struct, "extra":enrich}, ensure_ascii=False)
    resume_snippet = resume_text.strip()
    if len(resume_snippet) > 9000: resume_snippet = resume_snippet[:9000]
    system = ("너는 한국어 자기소개서 전문가다. 채용 공고의 회사/직무 요건과 후보자의 이력서를 참고해 "
              "회사 특화 자소서를 작성한다. 과장/허위 금지, 수치/지표/기간/임팩트 중심.")
    req = (f"회사 측 요청 주제는 '{topic_hint.strip()}' 이다. 이 주제를 중심으로 서술하라."
           if topic_hint and topic_hint.strip()
           else "특정 주제 요청이 없으므로 채용요건과 비전/인재상의 정합성을 강조하라.")
    user = (f"[회사/직무 요약(JSON)]\n{company}\n\n"
            f"[후보자 이력서(요약 가능)]\n{resume_snippet}\n\n"
            f"[작성 지시]\n- {req}\n- 분량 600~900자\n"
            "- 구성: 지원동기→역량/경험→성과/지표→입사 후 기여→마무리\n"
            "- 중복/미사여구 제거, 자연스러운 1인칭.")
    try:
        resp = client.chat.completions.create(model=model, temperature=0.4,
                                              messages=[{"role":"system","content":system},{"role":"user","content":user}])
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(자소서 생성 실패: {e})"

if st.button("자소서 생성", type="primary"):
    if not st.session_state.clean_struct:
        st.warning("먼저 회사 URL을 정제하세요.")
    elif not st.session_state.resume_raw.strip():
        st.warning("먼저 이력서를 업로드하고 '이력서 인덱싱'을 눌러주세요.")
    else:
        with st.spinner("자소서 생성 중..."):
            cover = build_cover_letter(st.session_state.clean_struct, st.session_state.resume_raw, topic, CHAT_MODEL)
        st.subheader("자소서 (생성 결과)")
        st.write(cover)
        st.download_button("자소서 TXT 다운로드", data=cover.encode("utf-8"),
                           file_name="cover_letter.txt", mime="text/plain")

st.divider()

# ================== 5) 질문 생성 & 답변 초안 (RAG 결합) ==================
st.header("5) 질문 생성 & 답변 초안 (RAG 결합)")
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)

cols_q = st.columns(2)
with cols_q[0]:
    if st.button("새 질문 받기", type="primary"):
        if not st.session_state.clean_struct:
            st.warning("먼저 회사 URL을 정제하세요.")
        else:
            def retrieve_resume_chunks_local(query: str, k: int = 4):
                chs, embs = st.session_state.get("resume_chunks", []), st.session_state.get("resume_embeds", None)
                if not chs or embs is None: return []
                qv = embed_texts([query], EMBED_MODEL)
                scores, idxs = cosine_topk(embs, qv, k=k)
                return [(float(s), chs[int(i)]) for s, i in zip(scores, idxs)]

            hits = retrieve_resume_chunks_local("핵심 프로젝트와 기술 스택 요약", k=4)
            resume_snips = [t for _, t in hits]
            resume_context = "\n".join([f"- {s[:350]}" for s in resume_snips])[:1200]
            ctx = json.dumps(st.session_state.clean_struct, ensure_ascii=False)
            user_msg = {"role": "user",
                        "content": (f"[회사/직무/요건]\n{ctx}\n\n"
                                    f"[지원자 이력서 요약(발췌)]\n{resume_context}\n\n"
                                    f"[요청]\n- 난이도/연차: {level}\n"
                                    f"- 중복 지양, 회사 요건과 이력서 교집합 및 공백영역 겨냥\n"
                                    f"- 한국어 면접 질문 1개만 한 줄로 출력")}
            try:
                resp = client.chat.completions.create(model=CHAT_MODEL, temperature=0.85,
                                                      messages=[{"role":"system","content":"면접 질문 생성기"}, user_msg])
                q = resp.choices[0].message.content.strip()
                q = re.sub(r"^\s*\d+[\).\s-]*","", q).split("\n")[0].strip()
            except Exception:
                q = ""
            if q:
                st.session_state.current_question = q
                st.session_state.answer_text = ""
                st.session_state.last_result = None
                st.session_state.followups = []
                st.session_state.selected_followup = ""
                st.session_state.followup_answer = ""
                st.success("질문 생성 완료!")
            else:
                st.error("질문 생성 실패")
with cols_q[1]:
    if st.button("RAG로 답변 초안 생성", type="secondary"):
        if not st.session_state.current_question:
            st.warning("먼저 질문을 생성하세요.")
        else:
            hits = retrieve_resume_chunks(st.session_state.current_question, k=4)
            resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
            ctx = json.dumps(st.session_state.clean_struct, ensure_ascii=False)
            user_msg = {"role": "user",
                        "content": (f"[회사/직무/채용요건]\n{ctx}\n\n"
                                    f"[지원자 이력서 발췌]\n{resume_text}\n\n"
                                    f"[면접 질문]\n{st.session_state.current_question}\n\n"
                                    "위 정보를 근거로 STAR 기반 한국어 답변 **초안**을 작성해줘.")}
            try:
                resp = client.chat.completions.create(model=CHAT_MODEL, temperature=0.5,
                                                      messages=[{"role":"system","content":"면접 답변 코치"}, user_msg])
                draft = resp.choices[0].message.content.strip()
            except Exception:
                draft = ""
            if draft:
                st.session_state.answer_text = draft
                st.success("초안 생성 완료!")
            else:
                st.error("초안 생성 실패")

st.text_area("질문", value=st.session_state.current_question, height=100)
st.text_area("나의 답변 (초안을 편집해 완성하세요)", height=200, key="answer_text")

# ================== 6) 채점 & 코칭 ==================
st.header("6) 채점 & 코칭")
if st.button("채점 & 코칭 실행", type="primary"):
    if not st.session_state.current_question:
        st.warning("먼저 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        ctx = json.dumps(st.session_state.clean_struct, ensure_ascii=False)
        hits = retrieve_resume_chunks(st.session_state.current_question + "\n" + st.session_state.answer_text[:800], k=4)
        resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
        user_msg = {"role":"user",
                    "content": (f"[회사/직무/채용요건]\n{ctx}\n\n"
                                f"[지원자 이력서 발췌]\n{resume_text}\n\n"
                                f"[면접 질문]\n{st.session_state.current_question}\n\n"
                                f"[지원자 답변]\n{st.session_state.answer_text}\n\n"
                                "다음 JSON 스키마로만 한국어 응답:\n"
                                "{"
                                "\"overall_score\": 0~100 정수,"
                                "\"criteria\": [{\"name\":\"문제정의\",\"score\":0~20,\"comment\":\"...\"},"
                                "{\"name\":\"데이터/지표\",\"score\":0~20,\"comment\":\"...\"},"
                                "{\"name\":\"실행력/주도성\",\"score\":0~20,\"comment\":\"...\"},"
                                "{\"name\":\"협업/커뮤니케이션\",\"score\":0~20,\"comment\":\"...\"},"
                                "{\"name\":\"고객가치\",\"score\":0~20,\"comment\":\"...\"}],"
                                "\"strengths\": [\"...\", \"...\"],"
                                "\"risks\": [\"...\", \"...\"],"
                                "\"improvements\": [\"...\", \"...\", \"...\"],"
                                "\"revised_answer\": \"STAR 구조로 간결히\""
                                "}") }
        try:
            resp = client.chat.completions.create(model=CHAT_MODEL, temperature=0.2,
                                                  response_format={"type":"json_object"},
                                                  messages=[{"role":"system","content":"엄격 채점관"}, user_msg])
            data = json.loads(resp.choices[0].message.content)
            crit_names = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]
            fixed=[]
            for name in crit_names:
                cand=None
                for it in data.get("criteria", []):
                    if str(it.get("name","")).strip()==name:
                        cand=it; break
                if not cand: cand={"name":name,"score":0,"comment":""}
                sc=int(cand.get("score",0)); sc=max(0,min(20,sc))
                cand["score"]=sc; cand["comment"]=str(cand.get("comment","")).strip()
                fixed.append(cand)
            total=sum(x["score"] for x in fixed)
            data["criteria"]=fixed; data["overall_score"]=total
        except Exception as e:
            data = {"overall_score":0,"criteria":[], "strengths":[], "risks":[], "improvements":[], "revised_answer":"", "error":str(e)}
        st.session_state.last_result = data
        st.success("채점/코칭 완료!")

st.divider()
st.subheader("피드백 결과")
last = st.session_state.last_result
if last:
    st.metric("총점(/100)", last.get("overall_score", 0))
    st.markdown("**기준별 점수 & 코멘트**")
    for it in last.get("criteria", []):
        st.markdown(f"- **{it.get('name','')}**: {it.get('score',0)}/20 — {it.get('comment','')}")
    if last.get("strengths"):
        st.markdown("**강점**"); 
        for s in last["strengths"]: st.markdown(f"- {s}")
    if last.get("risks"):
        st.markdown("**감점 요인/리스크**"); 
        for r in last["risks"]: st.markdown(f"- {r}")
    if last.get("improvements"):
        st.markdown("**개선 포인트**"); 
        for im in last["improvements"]: st.markdown(f"- {im}")
    if last.get("revised_answer"):
        st.markdown("**수정본 답변 (STAR)**"); 
        st.write(last["revised_answer"])
else:
    st.info("아직 채점 결과가 없습니다.")

st.divider()

# ================== 7) 팔로업 질문 · 답변 · 피드백 ==================
st.subheader("팔로업 질문 · 답변 · 피드백")
if last and not st.session_state.get("followups"):
    try:
        ctx = json.dumps({"company": st.session_state.clean_struct,
                          "vision": st.session_state.company_vision[:6],
                          "talent": st.session_state.company_talent[:6],
                          "news": [n.get("title","") for n in st.session_state.company_news[:3]]}, ensure_ascii=False)
        msg = {"role":"user",
               "content":(f"[회사/직무/요건/비전/이슈]\n{ctx}\n\n"
                          f"[지원자 답변]\n{st.session_state.answer_text}\n\n"
                          "면접관 관점에서 팔로업 질문 3개를 한 줄씩 한국어로 제안해줘. "
                          "기존 질문과 중복되지 않게, 지표/리스크/트레이드오프/의사결정 근거를 섞어줘.")}
        r = client.chat.completions.create(model=CHAT_MODEL, temperature=0.7,
                                           messages=[{"role":"system","content":"면접 팔로업 생성기"}, msg])
        followups = [re.sub(r'^\s*\d+[\).\s-]*','', l).strip()
                     for l in r.choices[0].message.content.splitlines() if l.strip()]
        st.session_state.followups = followups[:3]
    except Exception:
        st.session_state.followups = []

if last:
    if st.session_state.followups:
        st.markdown("**팔로업 질문 제안**")
        for i, f in enumerate(st.session_state.followups, 1):
            st.markdown(f"- ({i}) {f}")
        st.selectbox("채점 받을 팔로업 질문 선택", st.session_state.followups, index=0, key="selected_followup")
        st.text_area("팔로업 질문에 대한 나의 답변", height=160, key="followup_answer")
        if st.button("팔로업 채점 & 피드백", type="secondary"):
            fu_q   = st.session_state.get("selected_followup", "")
            fu_ans = st.session_state.get("followup_answer", "")
            if not fu_q:
                st.warning("팔로업 질문을 선택하세요.")
            elif not fu_ans.strip():
                st.warning("팔로업 답변을 작성하세요.")
            else:
                ctx = json.dumps(st.session_state.clean_struct, ensure_ascii=False)
                hits = retrieve_resume_chunks(fu_q + "\n" + fu_ans[:800], k=4)
                resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
                user_msg = {"role":"user",
                            "content": (f"[회사/직무/채용요건]\n{ctx}\n\n"
                                        f"[지원자 이력서 발췌]\n{resume_text}\n\n"
                                        f"[면접 질문]\n{fu_q}\n\n"
                                        f"[지원자 답변]\n{fu_ans}\n\n"
                                        "다음 JSON 스키마로만 한국어 응답:\n"
                                        "{"
                                        "\"overall_score\": 0~100 정수,"
                                        "\"criteria\": [{\"name\":\"문제정의\",\"score\":0~20,\"comment\":\"...\"},"
                                        "{\"name\":\"데이터/지표\",\"score\":0~20,\"comment\":\"...\"},"
                                        "{\"name\":\"실행력/주도성\",\"score\":0~20,\"comment\":\"...\"},"
                                        "{\"name\":\"협업/커뮤니케이션\",\"score\":0~20,\"comment\":\"...\"},"
                                        "{\"name\":\"고객가치\",\"score\":0~20,\"comment\":\"...\"}],"
                                        "\"strengths\": [\"...\"],\"risks\": [\"...\"],\"improvements\": [\"...\"],"
                                        "\"revised_answer\": \"STAR 구조로 간결히\""
                                        "}") }
                try:
                    r2 = client.chat.completions.create(model=CHAT_MODEL, temperature=0.2,
                                                        response_format={"type":"json_object"},
                                                        messages=[{"role":"system","content":"엄격 채점관"}, user_msg])
                    res_fu = json.loads(r2.choices[0].message.content)
                except Exception as e:
                    res_fu = {"overall_score":0,"criteria":[], "revised_answer":"", "error":str(e)}
                st.markdown("**팔로업 결과**")
                st.metric("총점(/100)", res_fu.get("overall_score", 0))
                for it in res_fu.get("criteria", []):
                    st.markdown(f"- **{it.get('name','')}**: {it.get('score',0)}/20 — {it.get('comment','')}")
                if res_fu.get("revised_answer",""):
                    st.markdown("**팔로업 수정본 (STAR)**")
                    st.write(res_fu["revised_answer"])
    else:
        st.caption("팔로업 질문은 메인 질문 채점 직후 자동 제안됩니다.")
