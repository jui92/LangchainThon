# -*- coding: utf-8 -*-
################################################################################################
# Job Helper Bot (Speed-Optimized)
# - Streamlit + OpenAI 기반 자소서/면접 지원 도우미
# - 성능 개선 사항:
#   * cache_data / cache_resource로 중복 호출 제거
#   * requests.Session + connection pool + 재시도
#   * 텍스트 정규화/하드컷으로 토큰 줄이기
#   * 뉴스/비전/인재상 수집 옵션화 및 병렬 처리
#   * 임베딩 벡터 캐시 + 정규화 캐시
#   * LLM 호출 최소화 및 파라미터 최적화
################################################################################################

# ===== 공통 임포트 =====
import os, re, json, urllib.parse, time, io, random, tempfile
from typing import Optional, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bs4 import BeautifulSoup
import html2text
import streamlit as st
import pandas as pd
import numpy as np

# ===== Streamlit 기본 설정 =====
st.set_page_config(page_title="Job Helper Bot (Fast)", page_icon="⚡", layout="wide")
st.title("⚡ Job Helper Bot (Fast) : 자소서 생성 / 모의 면접")

# ===== OpenAI 클라이언트 =====
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

# ===== 사이드바 옵션 =====
with st.sidebar:
    st.subheader("모델 & 옵션")
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델(내부용)", ["text-embedding-3-small","text-embedding-3-large"], index=0)
    ENABLE_COMPANY_ENRICH = st.checkbox("회사 비전/인재상/뉴스 수집", value=True)
    MAX_FETCH_PARALLEL = st.slider("병렬 수집 쓰레드", min_value=2, max_value=8, value=4, step=1)

# ===== HTTP 세션 (커넥션 풀/재시도) =====
# - 동일 호스트 반복 요청 시 성능 향상
@st.cache_resource(show_spinner=False)
def get_http_session():
    sess = requests.Session()
    retry = Retry(
        total=2, backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    # 일반 헤더 (ko 우선)
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
        "Accept-Language": "ko, en;q=0.9"
    })
    return sess

# ===== 유틸: URL 정규화 =====
def normalize_url(u: str) -> Optional[str]:
    if not u: return None
    u = u.strip()
    if not re.match(r"^https?://", u): u = "https://" + u
    parts = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

# ===== 유틸: 안전 텍스트 처리(하드컷/공백 정리/중복 제거) =====
def clean_text(s: str, max_len: int = 14000) -> str:
    if not s: return ""
    s = re.sub(r"\r", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len]
    return s

# ===== html → text 변환기 캐시 =====
@st.cache_resource(show_spinner=False)
def get_html2text():
    conv = html2text.HTML2Text()
    conv.ignore_links = True
    conv.ignore_images = True
    conv.body_width = 0
    return conv

# ===== HTTP GET 래퍼 =====
def http_get(url: str, timeout: int = 10) -> Optional[requests.Response]:
    try:
        sess = get_http_session()
        r = sess.get(url, timeout=timeout)
        if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
            return r
    except Exception:
        return None
    return None

# ===== 페이지 → 텍스트 추출 (Jina → Web → BS4 폴백) =====
@st.cache_data(show_spinner=False, ttl=60*30)  # 30분 캐시
def fetch_all_text(url: str) -> Tuple[str, Dict, Optional[str]]:
    url = normalize_url(url)
    if not url:
        return "", {"error":"invalid_url"}, None

    # 1) Jina proxy 우선
    try:
        parts = urllib.parse.urlsplit(url)
        prox = f"https://r.jina.ai/http://{parts.netloc}{parts.path}"
        if parts.query: prox += f"?{parts.query}"
        rj = http_get(prox, timeout=8)
        if rj and rj.text:
            soup_html = http_get(url, timeout=8).text if http_get(url, timeout=8) else None
            return clean_text(rj.text), {"source":"jina","len":len(rj.text),"url_final":url}, soup_html
    except Exception:
        pass

    # 2) 기본 HTML → markdown 텍스트
    r = http_get(url, timeout=8)
    if r:
        conv = get_html2text()
        txt = conv.handle(r.text)
        txt = re.sub(r"\n{3,}", "\n\n", txt or "").strip()
        return clean_text(txt), {"source":"webbase","len":len(txt),"url_final":url}, r.text

    # 3) BS4 fallback (대용량 텍스트)
    r2 = http_get(url, timeout=8)
    if r2:
        soup = BeautifulSoup(r2.text, "lxml")
        big=[]
        for sel in ("article","section","main","div","ul","ol"):
            for el in soup.select(sel):
                t = el.get_text(" ", strip=True)
                if t and len(t) > 300:
                    big.append(re.sub(r"\s+"," ", t))
        out = "\n\n".join(dict.fromkeys(big)) if big else soup.get_text(" ", strip=True)  # 중복 제거
        return clean_text(out), {"source":"bs4","len":len(out),"url_final":url}, r2.text

    return "", {"source":"none","len":0,"url_final":url}, None

# ===== soup HTML에서 메타 추출 =====
def extract_company_meta(soup_html: Optional[str]) -> Dict[str,str]:
    meta = {"company_name":"","company_intro":"","job_title":""}
    if not soup_html: return meta
    try:
        soup = BeautifulSoup(soup_html, "lxml")
        cand = []
        og = soup.find("meta", {"property":"og:site_name"})
        if og and og.get("content"): cand.append(og["content"])
        app = soup.find("meta", {"name":"application-name"})
        if app and app.get("content"): cand.append(app["content"])
        if soup.title and soup.title.string: cand.append(soup.title.string)
        # 첫 파편만
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

# ===== 키워드 기반 우대사항 보정 =====
def rule_based_sections(raw_text: str) -> dict:
    txt = clean_text(raw_text, 14000)
    lines = [re.sub(r"\s+", " ", l).strip(" -•·▶▪️") for l in txt.split("\n") if l.strip()]
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
            if hdr_pref.search(l):
                bucket="preferences"
            elif any(k in l.lower() for k in ["java","python","spark","airflow","kafka","ml","sql"]):
                bucket="responsibilities"
            else:
                continue
        push(l, bucket)

    # 자격요건 중 우대 키워드 이동
    kw_pref = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
    remain_qual=[]
    for q in out["qualifications"]:
        (out["preferences"] if kw_pref.search(q) else remain_qual).append(q)
    out["qualifications"]=remain_qual

    # 중복 제거
    for k in out:
        out[k] = list(dict.fromkeys([re.sub(r"\s+"," ", s).strip(" -•·▶▪️").strip() for s in out[k]]))[:12]
    return out

# ===== LLM: 공고 구조화 =====
PROMPT_SYSTEM_STRUCT = (
    "너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
    "입력 텍스트는 잡다한 광고/UI잔재가 섞여 있을 수 있다. 한국어로 간결하고 중복 없이 정제하라."
)
def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
    ctx = clean_text(raw_text, 14000)
    user_msg = {
        "role": "user",
        "content": (
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
        ),
    }
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=900,
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg]
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        data = {"company_name": meta_hint.get("company_name",""),
                "company_intro": meta_hint.get("company_intro","원문이 정제되지 않았습니다."),
                "job_title": meta_hint.get("job_title",""),
                "responsibilities": [], "qualifications": [], "preferences": [], "error": str(e)}

    # 리스트 클린
    for k in ["responsibilities","qualifications","preferences"]:
        arr = data.get(k, [])
        if not isinstance(arr, list): arr=[]
        clean=[]
        seen=set()
        for it in arr:
            t = re.sub(r"\s+"," ", str(it)).strip(" -•·▶▪️").strip()
            if t and t not in seen:
                seen.add(t); clean.append(t[:180])
        data[k] = clean[:12]

    # 프리퍼런스 보정
    if len(data.get("preferences", [])) < 1:
        rb = rule_based_sections(ctx)
        if rb.get("preferences"):
            merged = list(dict.fromkeys(data.get("preferences", []) + rb["preferences"]))[:12]
            data["preferences"] = merged
        else:
            kw_pref = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
            remain=[]; moved=[]
            for q in data.get("qualifications", []):
                (moved if kw_pref.search(q) else remain).append(q)
            data["preferences"] = moved[:12]
            data["qualifications"] = remain[:12]
    # 문자열 클린
    for k in ["company_name","company_intro","job_title"]:
        if isinstance(data.get(k), str):
            data[k] = re.sub(r"\s+"," ", data[k]).strip()
    return data

# ===== 파일 리더 =====
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

# ===== 청크/임베딩 =====
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

@st.cache_data(show_spinner=False, ttl=60*60)
def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=model_name, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs

@st.cache_data(show_spinner=False, ttl=60*60)
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

# ===== 회사 비전/인재상/뉴스 =====
def safe_get_text(el) -> str:
    try: return el.get_text(" ", strip=True)
    except Exception: return ""

def fetch_company_pages(home_url: str) -> Dict[str, List[str]]:
    out = {"vision": [], "talent": []}
    base = normalize_url(home_url)
    if not base: return out
    paths = ["","/","/about","/company","/about-us","/mission","/vision","/values","/culture","/careers","/talent","/people"]
    urls=[]
    seen=set()
    for p in paths:
        u = (base.rstrip("/") + p) if p else base
        if u not in seen:
            seen.add(u); urls.append(u)

    sess = get_http_session()
    texts_all=[]

    # 병렬 요청
    with ThreadPoolExecutor(max_workers=MAX_FETCH_PARALLEL) as ex:
        futures = {ex.submit(sess.get, u, timeout=6): u for u in urls}
        for fu in as_completed(futures):
            r=None
            try: r = fu.result()
            except Exception: pass
            if not (r and r.status_code==200): continue
            soup = BeautifulSoup(r.text, "lxml")
            for tag in soup.find_all(["h1","h2","h3","h4","p","li"]):
                t = safe_get_text(tag)
                if 6 <= len(t) <= 260:
                    texts_all.append(re.sub(r"\s+"," ", t))

    # 키워드 매칭/중복 제거
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
        r = get_http_session().get(url, headers=headers, params={"query": company, "display": display, "sort":"date"}, timeout=6)
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
        r = get_http_session().get(url, timeout=6)
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

# ===== 프롬프트(질문/초안/채점) =====
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
    user_msg = {"role": "user",
                "content": (f"[회사/직무/요건]\n{ctx}\n\n"
                            f"[지원자 이력서 요약(발췌)]\n{resume_context}\n\n"
                            f"[요청]\n- 난이도/연차: {level}\n"
                            f"- 중복/유사도 지양, 교집합 또는 공백영역을 겨냥\n"
                            f"- 한국어 면접 질문 1개만 한 줄로 출력")}
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
    user_msg = {"role":"user",
                "content": (f"[회사/직무/채용요건]\n{ctx}\n\n"
                            f"[지원자 이력서 발췌]\n{resume_text}\n\n"
                            f"[면접 질문]\n{question}\n\n"
                            "위 정보를 근거로 STAR 기반 한국어 답변 **초안**을 작성해줘.") }
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
    user_msg = {"role":"user",
                "content": (f"[회사/직무/채용요건]\n{ctx}\n\n"
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
                            "}") }
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=900,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_SCORE_STRICT}, user_msg]
        )
        data = json.loads(resp.choices[0].message.content)
        # 기준 보정
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

# ===== 세션 상태 초기화 =====
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

# ===== 1) 채용 공고 URL 입력 =====
st.header("1) 채용 공고 URL")
url = st.text_input("채용 공고 상세 URL", placeholder="취업 포털 사이트의 URL을 입력하세요.")
st.text_input("회사 공식 홈페이지 URL (선택)", key="company_home", placeholder="회사 공식 홈페이지 URL을 입력하세요.")

if st.button("원문 수집 → 정제", type="primary"):
    if not url.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner("원문 수집 중..."):
            raw, meta, soup_html = fetch_all_text(url.strip())
            hint = extract_company_meta(soup_html)

        if not raw:
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

            # 회사 비전/인재상/뉴스: 옵션에 따라 병렬 수집
            if ENABLE_COMPANY_ENRICH:
                with st.spinner("회사 비전/인재상/뉴스 수집 중..."):
                    vis, tal, news = [], [], []
                    tasks = []
                    with ThreadPoolExecutor(max_workers=3) as ex:
                        if st.session_state.company_home.strip():
                            tasks.append(("pages", ex.submit(fetch_company_pages, st.session_state.company_home.strip())))
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

# ===== 2) 회사 요약 표시 =====
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

# ===== 회사 비전/인재상/뉴스 =====
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

# ===== 3) 이력서 업로드/인덱싱 =====
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader("여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True)
_RESUME_CHUNK = 500
_RESUME_OVLP  = 100

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

# ===== 4) 자소서 생성 =====
st.header("4) 이력서 기반 자소서 생성")
topic = st.text_input("회사 요청 주제(선택)", placeholder="예: 직무 지원동기 / 협업 경험 / 문제해결 사례 등")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str) -> str:
    enrich = {"vision": st.session_state.company_vision[:6],
              "talent": st.session_state.company_talent[:6],
              "news": [n.get("title","") for n in st.session_state.company_news[:3]]}
    company = json.dumps({"clean":clean_struct, "extra":enrich}, ensure_ascii=False)
    resume_snippet = resume_text.strip()
    if len(resume_snippet) > 9000:
        resume_snippet = resume_snippet[:9000]
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
        st.download_button("자소서 TXT 다운로드", data=cover.encode("utf-8"),file_name="cover_letter.txt", mime="text/plain")

st.divider()

# ===== 5) 질문 생성 & 답변 초안 =====
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
ans = st.text_area("나의 답변 (초안을 편집해 완성하세요)", height=200, key="answer_text")

# ===== 6) 채점 & 코칭 =====
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

# ===== 7) 피드백 결과 =====
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

# ===== 8) 팔로업 질문/답변/피드백 =====
st.subheader("팔로업 질문 · 답변 · 피드백")
if last and not st.session_state.followups:
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
                st.session_state.last_followup_result = res_fu
                st.markdown("**팔로업 결과**")
                st.metric("총점(/100)", res_fu.get("overall_score", 0))
                for it in res_fu.get("criteria", []):
                    st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
                if res_fu.get("revised_answer",""):
                    st.markdown("**팔로업 수정본 (STAR)**")
                    st.write(res_fu["revised_answer"])
    else:
        st.caption("팔로업 질문은 메인 질문 채점 직후 자동 제안됩니다.")
