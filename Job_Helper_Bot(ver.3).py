# -*- coding: utf-8 -*-
################################################################################
# Job Helper Bot (Selenium-ONLY)
# - 자소서 생성 / 모의 면접
# - 오직 Selenium으로만 페이지 로드/클릭/본문 추출 (폴백 없음)
################################################################################

import os, re, json, urllib.parse, time, io, tempfile
from typing import Optional, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# ==== Selenium (필수) ====
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import shutil, pathlib

# -----------------------------------------------------------------------------
# Streamlit 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Job Helper Bot (Selenium-ONLY)", page_icon="🔎", layout="wide")
st.title("🔎 Job Helper Bot (Selenium-ONLY) : 자소서 생성 / 모의 면접")

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
    ENABLE_COMPANY_ENRICH = st.checkbox("회사 비전/인재상/뉴스 수집(홈페이지만)", value=True)
    SELENIUM_TIMEOUT = st.slider("Selenium 대기(초)", 5, 25, 10)
    MAX_FETCH_PARALLEL = st.slider("병렬 수집 쓰레드", 2, 8, 4)

# -----------------------------------------------------------------------------
# html2text
# -----------------------------------------------------------------------------
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

def html_to_text(html_str: str) -> str:
    txt = HTML2TEXT.handle(html_str or "")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return clean_text(txt)

# -----------------------------------------------------------------------------
# (1) Selenium 빌더 & 확장 클릭(도메인 최적화 포함)
# -----------------------------------------------------------------------------
def _build_chrome(headless: bool = True):
    opts = ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1440,2400")
    opts.add_argument("--lang=ko-KR")
    opts.add_argument("--disable-blink-features=AutomationControlled")

    # ▶ 크롬 바이너리 자동 탐지 (Streamlit Cloud/서버 환경 대비)
    candidates = [
        os.getenv("GOOGLE_CHROME_BIN"),
        os.getenv("CHROME_BIN"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium-browser"),
        shutil.which("chromium"),
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            opts.binary_location = p
            break

    # ▶ webdriver-manager + Service 사용 (Selenium 4 권장 방식)
    try:
        driver_path = ChromeDriverManager().install()
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=opts)
    except WebDriverException:
        # 셀레니움 매니저(내장) 시도 — 로컬에 크롬이 있을 때만 작동
        driver = webdriver.Chrome(options=opts)

    # (가능한 환경에서) 헤드리스 탐지 회피
    try:
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })
    except Exception:
        pass
    return driver


def _click_by_text_candidates(driver, texts: List[str]):
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
                    except Exception:
                        continue
        except Exception:
            continue

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
    selectors = [
        "[data-qa='btn-read-more']",
        "[aria-expanded='false']",
        "[role='button']",
        "button[class*='Read'], a[class*='Read']",
        "button[class*='More'], a[class*='More']",
    ]
    _click_many(driver, selectors)
    _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","기업정보"])

def _expand_saramin(driver):
    selectors = [
        ".btn_more", ".btnMore", ".btn-detail", ".btn_toggle",
        "[aria-expanded='false']", "[role='button']",
        "button[class*='more'], a[class*='more']"
    ]
    _click_many(driver, selectors)
    _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","기업정보","상세정보"])

def _expand_jobkorea(driver):
    selectors = [
        ".btnFold", ".btnToggleRead", ".btn_more",
        "[aria-expanded='false']", "[role='button']",
        "button[class*='More'], a[class*='More']"
    ]
    _click_many(driver, selectors)
    _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","기업정보","상세보기"])

def selenium_only_get_html(url: str, timeout: int = 10) -> str:
    driver = _build_chrome(headless=True)
    try:
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        try:
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, "//*")))
        except TimeoutException:
            pass

        host = urllib.parse.urlsplit(url).netloc.lower()

        # 공통 “더보기/우대” 시도
        _click_by_text_candidates(driver, ["더보기","상세보기","자세히 보기","자세히","전체보기","펼치기","모두 보기","Read more","More"])
        _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","Requirements","Responsibilities","Preferred"])

        # 도메인 전용 확장
        if "wanted.co.kr" in host:
            _expand_wanted(driver)
        if "saramin.co.kr" in host or "saramin" in host:
            _expand_saramin(driver)
        if "jobkorea.co.kr" in host:
            _expand_jobkorea(driver)

        # 지연 로딩 방지 스크롤
        for _ in range(6):
            try:
                driver.execute_script("window.scrollBy(0, 1000);"); time.sleep(0.25)
            except Exception:
                break

        return driver.page_source or ""
    finally:
        try: driver.quit()
        except Exception: pass

# -----------------------------------------------------------------------------
# (2) 페이지 텍스트 추출 (Selenium-ONLY)
# -----------------------------------------------------------------------------
def fetch_all_text_selenium_only(url: str, timeout: int = 10) -> Tuple[str, Dict, Optional[str]]:
    url = normalize_url(url)
    if not url:
        return "", {"error":"invalid_url"}, None

    html_dyn = selenium_only_get_html(url, timeout=timeout)
    if not html_dyn or len(html_dyn) < 200:
        return "", {"source":"selenium_failed","len":0,"url_final":url}, None

    txt = html_to_text(html_dyn)
    return txt, {"source":"selenium","len":len(txt),"url_final":url}, html_dyn

# -----------------------------------------------------------------------------
# (3) 메타/정제/규칙 기반 보완
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

def clean_bullets(arr):
    clean=[]; seen=set()
    for it in arr:
        t = re.sub(r"\s+"," ", str(it)).strip(" -•·▶▪️").strip()
        if t and t not in seen:
            seen.add(t); clean.append(t[:180])
    return clean[:12]

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
        data[k] = clean_bullets(arr)

    if len(data.get("preferences", [])) < 1:
        rb = rule_based_sections(ctx)
        if rb.get("preferences"):
            data["preferences"] = clean_bullets(list(dict.fromkeys(data.get("preferences", []) + rb["preferences"])))
        else:
            kw_pref = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
            remain=[]; moved=[]
            for q in data.get("qualifications", []):
                (moved if kw_pref.search(q) else remain).append(q)
            data["preferences"]=clean_bullets(moved); data["qualifications"]=clean_bullets(remain)

    for k in ["company_name","company_intro","job_title"]:
        if isinstance(data.get(k), str):
            data[k] = re.sub(r"\s+"," ", data[k]).strip()
    return data

# -----------------------------------------------------------------------------
# (4) 파일/임베딩/RAG (이전과 동일)
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

def embed_texts(texts: List[str], client: OpenAI, model_name: str) -> np.ndarray:
    if not texts: return np.zeros((0, 1536), dtype=np.float32)
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

def retrieve_resume_chunks(query: str, chunks: List[str], embeds_norm: np.ndarray, client: OpenAI, model_name: str, k: int = 4):
    if not chunks or embeds_norm is None or embeds_norm.size == 0:
        return []
    qv = embed_texts([query], client, model_name)
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
    hits = retrieve_resume_chunks("핵심 프로젝트와 기술 스택 요약", resume_chunks, resume_embeds_norm, client, EMBED_MODEL, k=4)
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
    hits = retrieve_resume_chunks(question, resume_chunks, resume_embeds_norm, client, EMBED_MODEL, k=4)
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
    hits = retrieve_resume_chunks(question + "\n" + answer[:800], resume_chunks, resume_embeds_norm, client, EMBED_MODEL, k=4)
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
# (5) 세션 상태
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
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
_init_state()

# -----------------------------------------------------------------------------
# (6) UI: 1) 채용 공고 URL 입력 → (Selenium Only) 원문 수집·정제
# -----------------------------------------------------------------------------
st.header("1) 채용 공고 URL (Selenium 전용)")
url = st.text_input("채용 공고 상세 URL", placeholder="취업 포털/기업 채용 페이지 URL을 입력하세요.")
st.text_input("회사 공식 홈페이지 URL (선택)", key="company_home", placeholder="회사 공식 홈페이지 URL을 입력하세요.")

if st.button("원문 수집 → 정제 (Selenium ONLY)", type="primary"):
    if not url.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner("Selenium으로 원문 수집 중..."):
            raw, meta, soup_html = fetch_all_text_selenium_only(url.strip(), timeout=SELENIUM_TIMEOUT)
            hint = extract_company_meta(soup_html)

        st.caption(f"수집 소스: {meta.get('source')} · 길이: {meta.get('len')}")
        if not raw:
            st.error("Selenium 수집에 실패했습니다. (Chrome/Chromium 설치, 타임아웃/셀렉터 확인 필요)")
        else:
            with st.spinner("LLM으로 정제 중..."):
                clean = llm_structurize(raw, hint, CHAT_MODEL)
            if not clean.get("preferences"):
                rb = rule_based_sections(raw)
                if rb.get("preferences"): clean["preferences"] = clean_bullets(rb["preferences"])
            st.session_state.clean_struct = clean
            try:
                kw_cnt = len(re.findall(r"(우대|우대사항|Preferred)", raw, flags=re.I))
                st.caption(f"진단: 원문 내 '우대' 관련 키워드 추정 등장 수 ≈ {kw_cnt}")
            except Exception:
                pass
            st.success("정제 완료!")

# -----------------------------------------------------------------------------
# (7) UI: 2) 회사 요약
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

st.divider()

# -----------------------------------------------------------------------------
# (8) UI: 3) 이력서 업로드/인덱싱
# -----------------------------------------------------------------------------
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader("여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True)
_RESUME_CHUNK = 500; _RESUME_OVLP = 100

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
                embeds = embed_texts(chunks, client, EMBED_MODEL)
                embeds_norm = l2_normalize(embeds)
            st.session_state.resume_raw = resume_text
            st.session_state.resume_chunks = chunks
            st.session_state.resume_embeds = embeds
            st.session_state.resume_embeds_norm = embeds_norm
            st.success(f"인덱싱 완료 (청크 {len(chunks)}개)")

st.divider()

# -----------------------------------------------------------------------------
# (9) UI: 4) 자소서 생성
# -----------------------------------------------------------------------------
st.header("4) 이력서 기반 자소서 생성")
topic = st.text_input("회사 요청 주제(선택)", placeholder="예: 직무 지원동기 / 협업 경험 / 문제해결 사례 등")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str) -> str:
    company = json.dumps({"clean":clean_struct}, ensure_ascii=False)
    resume_snippet = resume_text.strip()[:9000]
    system = ("너는 한국어 자기소개서 전문가다. 채용 공고의 회사/직무 요건과 후보자의 이력서를 참고해 "
              "회사 특화 자소서를 작성한다. 과장/허위는 금지하고, 수치/지표/기간/임팩트 중심으로 구체화한다.")
    req = (f"회사 측 요청 주제는 '{topic_hint.strip()}' 이다. 이 주제를 중심으로 서술하라."
           if topic_hint and topic_hint.strip()
           else "특정 주제 요청이 없으므로, 채용 공고를 중심으로 지원동기와 직무적합성을 강조하라.")
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
# (10) UI: 5) 질문 생성 & 답변 초안
# -----------------------------------------------------------------------------
st.header("5) 질문 생성 & 답변 초안 (RAG 결합)")
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)

col1, col2 = st.columns(2)
with col1:
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
                st.success("질문 생성 완료!")
            else:
                st.error("질문 생성 실패")

with col2:
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
# (11) UI: 6) 채점 & 코칭
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
        st.success("채점/코칭 완료!")

st.divider()
st.subheader("피드백 결과")
last = st.session_state.last_result
if last:
    st.metric("총점(/100)", last.get("overall_score", 0))
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
