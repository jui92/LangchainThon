# -*- coding: utf-8 -*-
################################################################################
# Job Helper Bot (Selenium-enabled, domain-tuned, NameError-safe)
# - 자소서 생성 / 모의 면접
# - Selenium로 '더보기/상세/우대'를 펼쳐 우대사항까지 수집 (강제 사용 옵션 포함)
# - 정적 수집(requests/bs4, Jina proxy) 폴백
################################################################################

import os, re, json, urllib.parse, time, io, tempfile
from typing import Optional, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bs4 import BeautifulSoup
import html2text
import streamlit as st
import numpy as np
import pandas as pd

# ==== OpenAI ====
try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop()

# ==== (선택) Selenium ====
SELENIUM_AVAILABLE = True
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
    from webdriver_manager.chrome import ChromeDriverManager
except Exception:
    SELENIUM_AVAILABLE = False

# -----------------------------------------------------------------------------
# Streamlit 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Job Helper Bot (Selenium)", page_icon="🔎", layout="wide")
st.title("🔎 Job Helper Bot (Selenium) : 자소서 생성 / 모의 면접")

# -----------------------------------------------------------------------------
# OpenAI 키 입력/확보
# -----------------------------------------------------------------------------
API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
if not API_KEY:
    API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password")
if not API_KEY:
    st.stop()
client = OpenAI(api_key=API_KEY)

# -----------------------------------------------------------------------------
# Sidebar 옵션
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("모델 & 옵션")
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small","text-embedding-3-large"], index=0)
    ENABLE_COMPANY_ENRICH = st.checkbox("회사 비전/인재상/뉴스 수집", value=True)
    ENABLE_SELENIUM = st.checkbox(
        "Selenium(동적 수집) 사용",
        value=True if SELENIUM_AVAILABLE else False,
        help="‘더보기/상세/우대’ 버튼을 자동 클릭해 전체 본문 수집"
    )
    FORCE_SELENIUM = st.checkbox("Selenium만 사용(폴백 금지)", value=False)  # ★ 추가
    SELENIUM_TIMEOUT = st.slider("Selenium 대기(초)", 4, 20, 8)
    MAX_FETCH_PARALLEL = st.slider("병렬 수집 쓰레드", 2, 8, 4)

# -----------------------------------------------------------------------------
# HTTP 세션 & html2text
# -----------------------------------------------------------------------------
def _http_session():
    sess = requests.Session()
    retry = Retry(total=2, backoff_factor=0.3,
                  status_forcelist=[429,500,502,503,504],
                  allowed_methods=["GET","HEAD"])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    sess.mount("http://", adapter); sess.mount("https://", adapter)
    sess.headers.update({
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
        "Accept-Language":"ko, en;q=0.9"
    })
    return sess
HTTP = _http_session()

def _get_html2text():
    conv = html2text.HTML2Text()
    conv.ignore_links = True
    conv.ignore_images = True
    conv.body_width = 0
    return conv
HTML2TEXT = _get_html2text()

# -----------------------------------------------------------------------------
# 공통 유틸
# -----------------------------------------------------------------------------
def normalize_url(u: str) -> Optional[str]:
    if not u: return None
    u = u.strip()
    if not re.match(r"^https?://", u): u = "https://" + u
    parts = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

def clean_text(s: str, max_len: int = 16000) -> str:
    if not s: return ""
    s = re.sub(r"\r","", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s[:max_len] if len(s) > max_len else s

def http_get(url: str, timeout: int = 8) -> Optional[requests.Response]:
    try:
        r = HTTP.get(url, timeout=timeout)
        # 일반적으로 HTML만 허용
        if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
            return r
    except Exception:
        pass
    return None

# -----------------------------------------------------------------------------
# (1) 회사 비전/인재상/뉴스 수집 함수들  ※ UI에서 호출되므로 여기 ‘위쪽’에 선언
# -----------------------------------------------------------------------------
def fetch_company_pages(home_url: str) -> Dict[str, List[str]]:
    out = {"vision": [], "talent": []}
    base = normalize_url(home_url)
    if not base: return out
    paths = ["","/","/about","/company","/about-us","/mission","/vision","/values","/culture","/careers","/talent","/people"]
    urls, seen = [], set()
    for p in paths:
        u = (base.rstrip("/") + p) if p else base
        if u not in seen:
            seen.add(u); urls.append(u)

    texts_all=[]
    with ThreadPoolExecutor(max_workers=MAX_FETCH_PARALLEL) as ex:
        futs = {ex.submit(HTTP.get, u, 6): u for u in urls}
        for fu in as_completed(futs):
            r=None
            try: r = fu.result()
            except Exception: pass
            if not (r and r.status_code==200): continue
            soup = BeautifulSoup(r.text, "lxml")
            for tag in soup.find_all(["h1","h2","h3","h4","p","li"]):
                t = tag.get_text(" ", strip=True)
                if t and 6 <= len(t) <= 260:
                    texts_all.append(re.sub(r"\s+"," ", t))

    for t in texts_all:
        low = t.lower()
        if any(k in low for k in ["talent","인재상","인재","people we","who we hire"]):
            out["talent"].append(t)
        if any(k in low for k in ["비전","미션","핵심가치","가치","원칙","mission","vision","values","principle"]):
            out["vision"].append(t)

    for k in out:
        out[k] = list(dict.fromkeys(x.strip() for x in out[k]))[:12]
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
        r = HTTP.get(url, headers=headers, params={"query": company, "display": display, "sort":"date"}, timeout=6)
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
        r = HTTP.get(url, timeout=6)
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

def fetch_latest_news(company: str, max_items: int = 5) -> List[Dict]:
    items = naver_search_news(company, display=max_items)
    return items if items else google_news_rss(company, max_items=max_items)

# -----------------------------------------------------------------------------
# (2) Selenium 동적 수집 (선택)
# -----------------------------------------------------------------------------
def _build_chrome(headless: bool = True):
    if not SELENIUM_AVAILABLE:
        raise RuntimeError("Selenium 미설치")
    opts = ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1440,2400")
    opts.add_argument("--lang=ko-KR")
    # 헤드리스 탐지 회피 (best-effort)
    opts.add_argument("--disable-blink-features=AutomationControlled")
    try:
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=opts)
    except WebDriverException:
        driver = webdriver.Chrome(options=opts)
    # webdriver 특성 숨김 (가능한 환경에서만)
    try:
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })
    except Exception:
        pass
    return driver

def _click_by_text_candidates(driver, texts: List[str], timeout=4):
    for t in texts:
        try:
            xpath_exact = f"//*[normalize-space(text())='{t}']"
            xpath_contains = f"//*[contains(normalize-space(text()), '{t}')]"
            for xp in (xpath_exact, xpath_contains):
                els = driver.find_elements(By.XPATH, xp)
                for el in els[:8]:
                    try:
                        driver.execute_script("arguments[0].click();", el)
                        time.sleep(0.25)
                        return True
                    except Exception:
                        continue
        except Exception:
            continue
    return False

def _click_many(driver, css_list, limit_per_selector=8):
    for css in css_list:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, css)
            for el in els[:limit_per_selector]:
                try:
                    driver.execute_script("arguments[0].click();", el)
                    time.sleep(0.2)
                except Exception:
                    continue
        except Exception:
            continue

def _expand_wanted(driver):
    # 원티드: 상세 본문/섹션 토글
    selectors = [
        "[data-qa='btn-read-more']",
        "[aria-expanded='false']",
        "[role='button']",
        "button[class*='Read'], a[class*='Read']",
        "button[class*='More'], a[class*='More']",
    ]
    _click_many(driver, selectors)
    _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","기업정보"], timeout=4)

def _expand_saramin(driver):
    # 사람인: 접힘 토글과 더보기
    selectors = [
        ".btn_more", ".btnMore", ".btn-detail", ".btn_toggle",
        "[aria-expanded='false']", "[role='button']",
        "button[class*='more'], a[class*='more']"
    ]
    _click_many(driver, selectors)
    _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","기업정보","상세정보"], timeout=4)

def _expand_jobkorea(driver):
    # 잡코리아: 본문 접힘 토글 / 더보기
    selectors = [
        ".btnFold", ".btnToggleRead", ".btn_more",
        "[aria-expanded='false']", "[role='button']",
        "button[class*='More'], a[class*='More']"
    ]
    _click_many(driver, selectors)
    _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","기업정보","상세보기"], timeout=4)

def selenium_expand_then_get_html(url: str, timeout: int = 8) -> str:
    driver = _build_chrome(headless=True)
    try:
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        try:
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, "//*")))
        except TimeoutException:
            pass

        host = urllib.parse.urlsplit(url).netloc.lower()

        # 공통 클릭
        _click_by_text_candidates(driver, ["더보기","상세보기","자세히 보기","자세히","전체보기","펼치기","모두 보기","Read more","More"], timeout=timeout)
        _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","Requirements","Responsibilities","Preferred"], timeout=timeout)

        # 도메인 전용 확장
        if "wanted.co.kr" in host:
            _expand_wanted(driver)
        if "saramin.co.kr" in host or "saramin" in host:
            _expand_saramin(driver)
        if "jobkorea.co.kr" in host:
            _expand_jobkorea(driver)

        # 스크롤 다운(지연 로딩 방지)
        for _ in range(6):
            try:
                driver.execute_script("window.scrollBy(0, 1000);"); time.sleep(0.25)
            except Exception:
                break

        return driver.page_source or ""
    except Exception:
        return ""
    finally:
        try: driver.quit()
        except Exception: pass

# -----------------------------------------------------------------------------
# (3) 페이지 텍스트 추출 (Selenium → Jina → requests → BS4)
# -----------------------------------------------------------------------------
def html_to_text(html_str: str) -> str:
    txt = HTML2TEXT.handle(html_str or "")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return clean_text(txt)

def fetch_all_text(url: str, use_selenium: bool, timeout: int = 8, force_selenium: bool = False) -> Tuple[str, Dict, Optional[str]]:
    """
    force_selenium=True 이면, Selenium 실패 시 폴백하지 않고 즉시 실패 메타를 반환.
    """
    url = normalize_url(url)
    if not url:
        return "", {"error":"invalid_url"}, None

    # a) Selenium 먼저
    if use_selenium and SELENIUM_AVAILABLE:
        try:
            html_dyn = selenium_expand_then_get_html(url, timeout=timeout)
            if html_dyn and len(html_dyn) > 300:
                txt = html_to_text(html_dyn)
                return txt, {"source":"selenium","len":len(txt),"url_final":url}, html_dyn
            elif force_selenium:
                return "", {"source":"selenium_failed","len":0,"url_final":url}, None
        except Exception:
            if force_selenium:
                return "", {"source":"selenium_error","len":0,"url_final":url}, None
            # 폴백 허용이면 계속 진행

    # b) Jina proxy (content-type이 text/plain일 수 있으므로 직접 호출)
    try:
        parts = urllib.parse.urlsplit(url)
        prox = f"https://r.jina.ai/http://{parts.netloc}{parts.path}"
        if parts.query: prox += f"?{parts.query}"
        rj = HTTP.get(prox, timeout=timeout)  # content-type 제한 없이
        if rj and rj.status_code == 200 and rj.text:
            base_r = http_get(url, timeout=timeout)  # 원문 HTML (가능하면)
            soup_html = base_r.text if base_r else None
            return clean_text(rj.text), {"source":"jina","len":len(rj.text),"url_final":url}, soup_html
    except Exception:
        pass

    # c) 일반 GET → html2text
    r = http_get(url, timeout=timeout)
    if r:
        txt = html_to_text(r.text)
        return txt, {"source":"webbase","len":len(txt),"url_final":url}, r.text

    # d) BS4 fallback
    r2 = http_get(url, timeout=timeout)
    if r2:
        soup = BeautifulSoup(r2.text, "lxml")
        big=[]
        for sel in ("article","section","main","div","ul","ol"):
            for el in soup.select(sel):
                t = el.get_text(" ", strip=True)
                if t and len(t) > 300:
                    big.append(re.sub(r"\s+"," ", t))
        out = "\n\n".join(dict.fromkeys(big)) if big else soup.get_text(" ", strip=True)
        return clean_text(out), {"source":"bs4","len":len(out),"url_final":url}, r2.text

    return "", {"source":"none","len":0,"url_final":url}, None

# -----------------------------------------------------------------------------
# (4) 메타/정제/규칙 기반 보완
# -----------------------------------------------------------------------------
def extract_company_meta(soup_html: Optional[str]) -> Dict[str,str]:
    meta = {"company_name":"","company_intro":"","job_title":""}
    if not soup_html: return meta
    try:
        soup = BeautifulSoup(soup_html, "lxml")
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
    txt = clean_text(raw_text, 16000)
    lines = [re.sub(r"\s+"," ", l).strip(" -•·▶▪️") for l in txt.split("\n") if l.strip()]
    hdr_resp = re.compile(r"(주요\s*업무|담당\s*업무|Role|Responsibilities?)", re.I)
    hdr_qual = re.compile(r"(자격\s*요건|지원\s*자격|Requirements?|Qualifications?)", re.I)
    hdr_pref = re.compile(r"(우대\s*사항|우대|선호|Preferred|Nice\s*to\s*have|Plus)", re.I)
    bucket = None
    out = {"responsibilities": [], "qualifications": [], "preferences": []}

    def push(line, b):
        if line and len(line) > 1 and line not in out[b]:
            out[b].append(line[:180])

    for l in lines:
        if hdr_resp.search(l): bucket="responsibilities"; continue
        if hdr_qual.search(l): bucket="qualifications"; continue
        if hdr_pref.search(l): bucket="preferences"; continue
        if bucket is None:
            if hdr_pref.search(l): bucket="preferences"
            elif any(k in l.lower() for k in ["java","python","spark","airflow","kafka","ml","sql"]):
                bucket="responsibilities"
            else:
                continue
        push(l, bucket)

    kw_pref = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
    remain_qual=[]
    for q in out["qualifications"]:
        (out["preferences"] if kw_pref.search(q) else remain_qual).append(q)
    out["qualifications"]=remain_qual

    for k in out:
        out[k] = list(dict.fromkeys([re.sub(r"\s+"," ", s).strip(" -•·▶▪️").strip() for s in out[k]]))[:12]
    return out

PROMPT_SYSTEM_STRUCT = (
    "너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
    "입력 텍스트는 잡다한 광고/UI잔재가 섞여 있을 수 있다. 한국어로 간결하고 중복 없이 정제하라."
)

def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
    ctx = clean_text(raw_text, 14000)
    user_msg = {"role":"user","content":(
        "다음 채용 공고 원문을 구조화해줘.\n\n"
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
        "- '우대 사항(preferences)'은 표시가 있는 항목만 포함.\n"
        "- 불릿/이모지 제거, 간결화, 중복 제거."
    )}
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=900,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg]
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        data = {"company_name": meta_hint.get("company_name",""),
                "company_intro": meta_hint.get("company_intro","원문이 정제되지 않았습니다."),
                "job_title": meta_hint.get("job_title",""),
                "responsibilities": [], "qualifications": [], "preferences": [], "error": str(e)}

    for k in ["responsibilities","qualifications","preferences"]:
        arr = data.get(k, [])
        if not isinstance(arr, list): arr=[]
        clean=[]; seen=set()
        for it in arr:
            t = re.sub(r"\s+"," ", str(it)).strip(" -•·▶▪️").strip()
            if t and t not in seen:
                seen.add(t); clean.append(t[:180])
        data[k] = clean[:12]

    if len(data.get("preferences", [])) < 1:
        rb = rule_based_sections(ctx)
        if rb.get("preferences"):
            data["preferences"] = list(dict.fromkeys(data.get("preferences", []) + rb["preferences"]))[:12]
        else:
            kw_pref = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
            remain=[]; moved=[]
            for q in data.get("qualifications", []):
                (moved if kw_pref.search(q) else remain).append(q)
            data["preferences"]=moved[:12]; data["qualifications"]=remain[:12]

    for k in ["company_name","company_intro","job_title"]:
        if isinstance(data.get(k), str):
            data[k] = re.sub(r"\s+"," ", data[k]).strip()
    return data

# -----------------------------------------------------------------------------
# (5) 파일 리더 / 임베딩 / 검색 / LLM 질문·초안·채점
# -----------------------------------------------------------------------------
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
            return docx2txt.process(tmp.name) or ""
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
    t = re.sub(r"\s+"," ", text or "").strip()
    if not t: return []
    out, start = [], 0
    L = len(t)
    while start < L:
        end = min(L, start+size)
        out.append(t[start:end])
        if end == L: break
        start = max(0, end-overlap)
    return out

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=model_name, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0: return mat
    n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / n

def cosine_topk(matrix_n: np.ndarray, query_vec_n: np.ndarray, k: int = 4):
    if matrix_n.size == 0: return np.array([]), np.array([], dtype=int)
    sims = matrix_n @ query_vec_n.T
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx

def retrieve_resume_chunks(query: str, chunks: List[str], embeds_norm: np.ndarray, k: int = 4):
    if not chunks or embeds_norm is None or embeds_norm.size == 0:
        return []
    qv = embed_texts([query], EMBED_MODEL)
    qv_n = l2_normalize(qv)
    scores, idxs = cosine_topk(embeds_norm, qv_n, k=k)
    return [(float(s), chunks[int(i)]) for s, i in zip(scores, idxs)]

PROMPT_SYSTEM_Q = (
    "너는 채용담당자다. 회사/직무 맥락과 채용요건, 그리고 지원자의 이력서 요약을 함께 고려해 "
    "면접 질문을 한국어로 생성한다. 질문은 서로 겹치지 않게 다양화하고, 수치/지표/기간/규모/리스크/트레이드오프도 섞어라."
)
PROMPT_SYSTEM_DRAFT = (
    "너는 면접 답변 코치다. 회사/직무/채용요건과 지원자의 이력서 요약을 결합해 "
    "질문에 대한 답변 초안을 STAR(상황-과제-행동-성과)로 8~12문장, 한국어로 작성한다. "
    "가능하면 구체적인 지표/수치/기간/임팩트를 포함하라."
)
PROMPT_SYSTEM_SCORE_STRICT = (
    "너는 매우 엄격한 톱티어 면접 코치다. 아래 형식의 JSON만 출력하라. "
    "각 기준은 0~20 정수이며, 총점은 기준 합계(최대 100)와 반드시 일치해야 한다. "
    "과장/모호함/근거 부재/숫자 없는 주장/책임 회피 등을 강하게 감점하라."
)
CRITERIA = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def llm_generate_one_question_with_resume(clean: Dict, level: str, model: str,
                                          resume_chunks: List[str], resume_embeds_norm: np.ndarray) -> str:
    hits = retrieve_resume_chunks("핵심 프로젝트와 기술 스택 요약", resume_chunks, resume_embeds_norm, k=4)
    resume_context = "\n".join([f"- {t[:350]}" for _, t in hits])[:1200]
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {"role":"user","content":(
        f"[회사/직무/요건]\n{ctx}\n\n"
        f"[지원자 이력서 요약(발췌)]\n{resume_context}\n\n"
        f"[요청]\n- 난이도/연차: {level}\n"
        f"- 중복/유사도 지양, 교집합 또는 공백영역 겨냥\n"
        f"- 한국어 면접 질문 1개만 한 줄로 출력"
    )}
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.7, max_tokens=120,
            messages=[{"role":"system","content":PROMPT_SYSTEM_Q}, user_msg]
        )
        q = resp.choices[0].message.content.strip()
        q = re.sub(r"^\s*\d+[\).\s-]*","", q).split("\n")[0].strip()
        return q
    except Exception:
        return ""

def llm_draft_answer(clean: Dict, question: str, model: str,
                     resume_chunks: List[str], resume_embeds_norm: np.ndarray) -> str:
    hits = retrieve_resume_chunks(question, resume_chunks, resume_embeds_norm, k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {"role":"user","content":(
        f"[회사/직무/채용요건]\n{ctx}\n\n"
        f"[지원자 이력서 발췌]\n{resume_text}\n\n"
        f"[면접 질문]\n{question}\n\n"
        "위 정보를 근거로 STAR 기반 한국어 답변 **초안**을 작성해줘."
    )}
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.5, max_tokens=700,
            messages=[{"role":"system","content":PROMPT_SYSTEM_DRAFT}, user_msg]
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

def llm_score_and_coach_strict(clean: Dict, question: str, answer: str, model: str,
                               resume_chunks: List[str], resume_embeds_norm: np.ndarray) -> Dict:
    hits = retrieve_resume_chunks(question + "\n" + answer[:800], resume_chunks, resume_embeds_norm, k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {"role":"user","content":(
        f"[회사/직무/채용요건]\n{ctx}\n\n"
        f"[지원자 이력서 발췌]\n{resume_text}\n\n"
        f"[면접 질문]\n{question}\n\n"
        f"[지원자 답변]\n{answer}\n\n"
        "다음 JSON 스키마로만 한국어 응답:\n"
        "{"
        "\"overall_score\": 0~100 정수,"
        "\"criteria\": [{\"name\":\"문제정의\",\"score\":0~20,\"comment\":\"...\"},"
        "{\"name\":\"데이터/지표\",\"score\":0~20,\"comment\":\"...\"},"
        "{\"name\":\"실행력/주도성\",\"score\":0~20,\"comment\":\"...\"},"
        "{\"name\":\"협업/커뮤니케이션\",\"score\":0~20,\"comment\":\"...\"},"
        "{\"name\":\"고객가치\",\"score\":0~20,\"comment\":\"...\"}],"
        "\"strengths\": [\"...\"],"
        "\"risks\": [\"...\"],"
        "\"improvements\": [\"...\",\"...\",\"...\"],"
        "\"revised_answer\": \"STAR 구조로 간결히\""
        "}"
    )}
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=900,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_SCORE_STRICT}, user_msg]
        )
        data = json.loads(resp.choices[0].message.content)
        fixed=[]
        for name in CRITERIA:
            found=None
            for it in data.get("criteria", []):
                if str(it.get("name","")).strip()==name:
                    found=it; break
            if not found: found={"name":name,"score":0,"comment":""}
            sc = int(found.get("score",0)); sc=max(0,min(20,sc))
            found["score"]=sc; found["comment"]=str(found.get("comment","")).strip()
            fixed.append(found)
        total=sum(x["score"] for x in fixed)
        data["criteria"]=fixed
        data["overall_score"]=total
        for k in ["strengths","risks","improvements"]:
            arr=data.get(k,[]); 
            if not isinstance(arr,list): arr=[]
            data[k]=[str(x).strip() for x in arr if str(x).strip()][:5]
        data["revised_answer"]=str(data.get("revised_answer","")).strip()
        return data
    except Exception as e:
        return {"overall_score": 0,
                "criteria": [{"name": n, "score": 0, "comment": ""} for n in CRITERIA],
                "strengths": [], "risks": [], "improvements": [], "revised_answer": "",
                "error": str(e)}

# -----------------------------------------------------------------------------
# (6) 세션 상태
# -----------------------------------------------------------------------------
def _init_state():
    defaults = {
        "clean_struct": None,
        "resume_raw": "",
        "resume_chunks": [],
        "resume_embeds": None,
        "resume_embeds_norm": None,
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
        "company_news": []
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
_init_state()

# -----------------------------------------------------------------------------
# (7) UI: 1) 채용 공고 URL 입력 → 원문 수집·정제
# -----------------------------------------------------------------------------
st.header("1) 채용 공고 URL")
url = st.text_input("채용 공고 상세 URL", placeholder="취업 포털 사이트의 URL을 입력하세요.")
st.text_input("회사 공식 홈페이지 URL (선택)", key="company_home", placeholder="회사 공식 홈페이지 URL을 입력하세요.")

if st.button("원문 수집 → 정제", type="primary"):
    if not url.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner("원문 수집 중..."):
            raw, meta, soup_html = fetch_all_text(
                url.strip(),
                use_selenium=ENABLE_SELENIUM,
                timeout=SELENIUM_TIMEOUT,
                force_selenium=FORCE_SELENIUM,   # ★ 강제 사용 옵션 전달
            )
            hint = extract_company_meta(soup_html)

        # 수집 소스/길이 표시
        st.caption(f"수집 소스: {meta.get('source')} · 길이: {meta.get('len')}")

        if not raw:
            if meta.get("source") in ("selenium_failed","selenium_error"):
                st.error("Selenium 수집에 실패했습니다. (Chrome/크로미움 설치, 타임아웃/셀렉터 확인 필요)")
            else:
                st.error("원문을 가져오지 못했습니다. (로그인/동적 렌더링/봇 차단 가능)")
        else:
            with st.spinner("LLM으로 정제 중..."):
                clean = llm_structurize(raw, hint, CHAT_MODEL)

            # 규칙 기반 우대사항 보완
            if not clean.get("preferences"):
                rb = rule_based_sections(raw)
                if rb.get("preferences"):
                    clean["preferences"] = rb["preferences"][:12]

            st.session_state.clean_struct = clean

            # 간단 진단: 우대 키워드 등장량
            try:
                kw_cnt = sum(1 for x in re.findall(r"[가-힣A-Za-z]+", raw or "") if any(k in x for k in ["우대","우대사항","Preferred"]))
                st.caption(f"진단: 원문 내 '우대' 키워드 추정 등장 수 ≈ {kw_cnt}")
            except Exception:
                pass

            # 회사 비전/인재상/뉴스
            if ENABLE_COMPANY_ENRICH:
                with st.spinner("회사 비전/인재상/뉴스 수집 중..."):
                    vis, tal, news = [], [], []
                    tasks = []
                    with ThreadPoolExecutor(max_workers=3) as ex:
                        home = (st.session_state.company_home or "").strip()
                        if home:
                            tasks.append(("pages", ex.submit(fetch_company_pages, home)))
                        cname = clean.get("company_name") or hint.get("company_name") or ""
                        if cname:
                            tasks.append(("news", ex.submit(fetch_latest_news, cname, 5)))
                        for tag, fut in tasks:
                            try:
                                res = fut.result()
                                if tag=="pages":
                                    vis = res.get("vision", [])
                                    tal = res.get("talent", [])
                                else:
                                    news = res
                            except Exception:
                                pass
                    st.session_state.company_vision = vis
                    st.session_state.company_talent = tal
                    st.session_state.company_news = news
            else:
                st.session_state.company_vision = []
                st.session_state.company_talent = []
                st.session_state.company_news = []
            st.success("정제 완료!")

# -----------------------------------------------------------------------------
# (8) UI: 2) 회사 요약 섹션
# -----------------------------------------------------------------------------
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

# 회사 비전/인재상/뉴스 표시
if ENABLE_COMPANY_ENRICH and (st.session_state.company_vision or st.session_state.company_talent or st.session_state.company_news):
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

# -----------------------------------------------------------------------------
# (9) UI: 3) 이력서 업로드/인덱싱
# -----------------------------------------------------------------------------
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader("여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True)
_RESUME_CHUNK = 500; _RESUME_OVLP = 100

col_idx = st.columns(2)
with col_idx[0]:
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
                    embeds_norm = l2_normalize(embeds)
                st.session_state.resume_raw = resume_text
                st.session_state.resume_chunks = chunks
                st.session_state.resume_embeds = embeds
                st.session_state.resume_embeds_norm = embeds_norm
                st.success(f"인덱싱 완료 (청크 {len(chunks)}개)")

st.divider()

# -----------------------------------------------------------------------------
# (10) UI: 4) 자소서 생성
# -----------------------------------------------------------------------------
st.header("4) 이력서 기반 자소서 생성")
topic = st.text_input("회사 요청 주제(선택)", placeholder="예: 직무 지원동기 / 협업 경험 / 문제해결 사례 등")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str) -> str:
    enrich = {"vision": st.session_state.company_vision[:6],
              "talent": st.session_state.company_talent[:6],
              "news": [n.get("title","") for n in st.session_state.company_news[:3]]}
    company = json.dumps({"clean":clean_struct, "extra":enrich}, ensure_ascii=False)
    resume_snippet = resume_text.strip()[:9000]
    system = ("너는 한국어 자기소개서 전문가다. 채용 공고의 회사/직무 요건과 후보자의 이력서를 참고해 "
              "회사 특화 자소서를 작성한다. 과장/허위는 금지하고, 수치/지표/기간/임팩트 중심으로 구체화한다. "
              "회사의 비전/인재상/최근 이슈가 제공되면 자연스럽게 연결하라.")
    req = (f"회사 측 요청 주제는 '{topic_hint.strip()}' 이다. 이 주제를 중심으로 서술하라."
           if topic_hint and topic_hint.strip() else
           "특정 주제 요청이 없으므로, 채용 공고와 비전/인재상을 중심으로 지원동기와 직무적합성을 강조하라.")
    user = (f"[회사/직무 요약(JSON)]\n{company}\n\n"
            f"[후보자 이력서(요약 가능)]\n{resume_snippet}\n\n"
            f"[작성 지시]\n- {req}\n"
            "- 분량: 600~900자\n"
            "- 구성: 1) 지원 동기 2) 직무 관련 핵심 역량·경험 3) 성과/지표 4) 입사 후 기여 방안 5) 마무리\n"
            "- 자연스럽고 진정성 있는 1인칭 서술. 문장과 문단 가독성을 유지.\n"
            "- 불필요한 미사여구/중복/광고 문구 삭제.")
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.4, max_tokens=800,
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
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
        st.download_button("자소서 TXT 다운로드", data=cover.encode("utf-8"), file_name="cover_letter.txt", mime="text/plain")

st.divider()

# -----------------------------------------------------------------------------
# (11) UI: 5) 질문 생성 & 답변 초안
# -----------------------------------------------------------------------------
st.header("5) 질문 생성 & 답변 초안 (RAG 결합)")
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)

cols_q = st.columns(2)
with cols_q[0]:
    if st.button("새 질문 받기", type="primary"):
        if not st.session_state.clean_struct:
            st.warning("먼저 회사 URL을 정제하세요.")
        else:
            q = llm_generate_one_question_with_resume(
                st.session_state.clean_struct, level, CHAT_MODEL,
                st.session_state.resume_chunks, st.session_state.resume_embeds_norm
            )
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
            draft = llm_draft_answer(
                st.session_state.clean_struct, st.session_state.current_question, CHAT_MODEL,
                st.session_state.resume_chunks, st.session_state.resume_embeds_norm
            )
            if draft:
                st.session_state.answer_text = draft
                st.success("초안 생성 완료!")
            else:
                st.error("초안 생성 실패")

st.text_area("질문", value=st.session_state.current_question, height=100)
st.text_area("나의 답변 (초안을 편집해 완성하세요)", height=200, key="answer_text")

# -----------------------------------------------------------------------------
# (12) UI: 6) 채점 & 코칭
# -----------------------------------------------------------------------------
st.header("6) 채점 & 코칭")
if st.button("채점 & 코칭 실행", type="primary"):
    if not st.session_state.current_question:
        st.warning("먼저 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("채점/코칭 중..."):
            res = llm_score_and_coach_strict(
                st.session_state.clean_struct,
                st.session_state.current_question,
                st.session_state.answer_text,
                CHAT_MODEL,
                st.session_state.resume_chunks,
                st.session_state.resume_embeds_norm
            )
        st.session_state.last_result = res
        st.session_state.records.append({
            "question": st.session_state.current_question,
            "answer": st.session_state.answer_text,
            "overall": res.get("overall_score", 0),
            "criteria": res.get("criteria", []),
            "strengths": res.get("strengths", []),
            "risks": res.get("risks", []),
            "improvements": res.get("improvements", []),
            "revised_answer": res.get("revised_answer","")
        })
        st.success("채점/코칭 완료!")

# -----------------------------------------------------------------------------
# (13) UI: 7) 피드백 & 팔로업
# -----------------------------------------------------------------------------
st.header("7) 피드백 결과")
last = st.session_state.last_result
if last:
    left, right = st.columns([1,3])
    with left:
        st.metric("총점(/100)", last.get("overall_score", 0))
    with right:
        st.markdown("**기준별 점수 & 코멘트**")
        for it in last.get("criteria", []):
            st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
        if last.get("strengths"):
            st.markdown("**강점**")
            for s in last["strengths"]: st.markdown(f"- {s}")
        if last.get("risks"):
            st.markdown("**감점 요인/리스크**")
            for r in last["risks"]: st.markdown(f"- {r}")
        if last.get("improvements"):
            st.markdown("**개선 포인트**")
            for im in last["improvements"]: st.markdown(f"- {im}")
        if last.get("revised_answer"):
            st.markdown("**수정본 답변 (STAR)**")
            st.write(last["revised_answer"])
else:
    st.info("아직 채점 결과가 없습니다.")

st.divider()
st.subheader("팔로업 질문 · 답변 · 피드백")
if last and not st.session_state.followups:
    try:
        ctx = json.dumps({"company": st.session_state.clean_struct,
                          "vision": st.session_state.company_vision[:6],
                          "talent": st.session_state.company_talent[:6],
                          "news": [n.get("title","") for n in st.session_state.company_news[:3]]}, ensure_ascii=False)
        msg = {"role":"user","content":(
            f"[회사/직무/요건/비전/이슈]\n{ctx}\n\n"
            f"[지원자 답변]\n{st.session_state.answer_text}\n\n"
            "면접관 관점에서 팔로업 질문 3개를 한 줄씩 한국어로 제안해줘. "
            "기존 질문과 중복되지 않게, 지표/리스크/트레이드오프/의사결정 근거를 섞어줘."
        )}
        r = client.chat.completions.create(model=CHAT_MODEL, temperature=0.7, max_tokens=240,
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
                with st.spinner("팔로업 채점 중..."):
                    res_fu = llm_score_and_coach_strict(
                        st.session_state.clean_struct, fu_q, fu_ans, CHAT_MODEL,
                        st.session_state.resume_chunks, st.session_state.resume_embeds_norm
                    )
                st.markdown("**팔로업 결과**")
                st.metric("총점(/100)", res_fu.get("overall_score", 0))
                for it in res_fu.get("criteria", []):
                    st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
                if res_fu.get("revised_answer",""):
                    st.markdown("**팔로업 수정본 (STAR)**")
                    st.write(res_fu["revised_answer"])
    else:
        st.caption("팔로업 질문은 메인 질문 채점 직후 자동 제안됩니다.")
