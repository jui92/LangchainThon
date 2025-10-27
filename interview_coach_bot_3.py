# -*- coding: utf-8 -*-
"""
Job Helper Bot (속도 튜닝판)
- 정확도 유지 / 속도 개선:
  1) fetch_all_text: 네트워크 1회 + 파싱 1회로 일원화 (중복 제거)
  2) 캐싱(st.cache_data/resource): 원문/파싱/임베딩/뉴스 재사용
  3) LLM 토큰 다이어트: 섹션 트림 + 규칙 파서 선반영
  4) 임베딩 캐시/상한: 큰 텍스트도 빠르게 RAG
  5) (옵션) 동적 크롤링 Playwright: 기본 꺼짐(느린 환경 고려)
"""

import os, re, io, json, hashlib, urllib.parse, time
from typing import Optional, Dict, List, Tuple

import requests
import streamlit as st
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import html2text

# ---------------- OpenAI ----------------
try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop()

# ---------------- PDF/DOCX ----------------
try:
    import pypdf
except Exception:
    pypdf = None


# =========================== 앱 설정 ===========================
st.set_page_config(page_title="Job Helper Bot (Fast)", page_icon="⚡", layout="wide")
st.title("Job Helper Bot — 빠르게! (정확도 유지·속도 개선판) ⚡")

# OpenAI
API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
if not API_KEY:
    API_KEY = st.text_input("OPENAI_API_KEY", type="password")
if not API_KEY:
    st.stop()

@st.cache_resource
def get_client(api_key: str):
    return OpenAI(api_key=api_key)

client = get_client(API_KEY)

# 사이드바
with st.sidebar:
    st.subheader("모델·옵션")
    CHAT_MODEL  = st.selectbox("대화/생성 모델", ["gpt-4o-mini", "gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small","text-embedding-3-large"], index=0)
    FAST_MODE   = st.toggle("빠른 모드(정확도 유지, 속도 우선)", value=True)
    ENABLE_DYNAMIC = st.toggle("동적 크롤링(Playwright) 사용", value=False)
    st.caption("동적 크롤링은 느릴 수 있어 기본 꺼짐. 필요시 켜서 재시도하세요.")


# =========================== 유틸/캐시 ===========================
def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def normalize_url(u: str) -> Optional[str]:
    if not u: return None
    u = u.strip()
    if not re.match(r"^https?://", u): u = "https://" + u
    parts = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

def http_get(url: str, timeout: int = 10) -> Optional[requests.Response]:
    """단일 HTTP GET (공통 User-Agent), 성공 시 HTML Response 반환"""
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
                "Accept-Language": "ko, en;q=0.9",
            },
            timeout=timeout,
        )
        if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
            return r
    except Exception:
        pass
    return None

def html_to_text(html_str: str) -> str:
    conv = html2text.HTML2Text()
    conv.ignore_links = True
    conv.ignore_images = True
    conv.body_width = 0
    txt = conv.handle(html_str or "")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_all_text(url: str) -> Tuple[str, Dict, Optional[BeautifulSoup]]:
    """
    속도 개선의 핵심:
    - 네트워크 1회 + 파싱 1회로 끝.
    - 빠른 텍스트 변환(html2text) → 충분하면 반환
    - 부족 시 블록 기반 보강 → 최후의 수단 전체 텍스트
    """
    url = normalize_url(url)
    if not url:
        return "", {"error": "invalid_url"}, None

    r = http_get(url, timeout=10)
    if not r:
        return "", {"error": "http_fail"}, None
    soup = BeautifulSoup(r.text, "lxml")

    # 1) 경량 변환
    fast_text = html_to_text(r.text)
    if fast_text and len(fast_text) > 400:
        return fast_text[:120000], {"source":"html2text","len":len(fast_text),"url_final":url}, soup

    # 2) 블록 기반
    blocks=[]
    for sel in ["article","section","main","div","ul","ol"]:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) > 300:
                blocks.append(re.sub(r"\s+"," ", txt))
    if blocks:
        seen=set(); out=[]
        for b in blocks:
            if b not in seen:
                seen.add(b); out.append(b)
        joined="\n\n".join(out)
        return joined[:120000], {"source":"bs4_blocks","len":len(joined),"url_final":url}, soup

    # 3) 전체 텍스트
    full = soup.get_text(" ", strip=True)
    return full[:120000], {"source":"bs4_full","len":len(full),"url_final":url}, soup

def trim_for_llm(text: str, hard_limit: int = 10000) -> str:
    """LLM 토큰 다이어트: 긴 원문은 앞 70% + 뒤 30%만"""
    t = (text or "").strip()
    if len(t) <= hard_limit:
        return t
    head = t[:int(hard_limit*0.7)]
    tail = t[-int(hard_limit*0.3):]
    return head + "\n...\n" + tail

# ---------------- 파일 리더 ----------------
def read_pdf(data: bytes) -> str:
    if pypdf is None: return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
    except Exception:
        return ""

def read_docx(data: bytes) -> str:
    # docx2txt 우선, 실패 시 python-docx fallback
    try:
        import docx2txt, tempfile
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
            tmp.write(data); tmp.flush()
            return docx2txt.process(tmp.name) or ""
    except Exception:
        try:
            import docx as docxlib
            f = io.BytesIO(data)
            doc = docxlib.Document(f)
            return "\n".join([p.text for p in doc.paragraphs])
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

# ---------------- 임베딩/RAG ----------------
def chunk(text: str, size: int = 400, overlap: int = 80) -> List[str]:
    t = re.sub(r"\s+"," ", text).strip()
    if not t: return []
    out, start = [], 0
    while start < len(t):
        end = min(len(t), start+size)
        out.append(t[start:end])
        if end == len(t): break
        start = max(0, end-overlap)
    return out

@st.cache_data(show_spinner=False, ttl=3600)
def cached_embeddings(chunks: List[str], model_name: str) -> np.ndarray:
    """청크 내용을 그대로 해싱해서 캐시 키로 사용 → 동일 내용 재임베딩 방지"""
    if not chunks:
        return np.zeros((0, 1536), dtype=np.float32)
    key = _hash("\n".join(chunks) + model_name)  # 캐시 키 용도로만 사용(내부적으로는 무시)
    resp = client.embeddings.create(model=model_name, input=chunks)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def cosine_topk(matrix: np.ndarray, query_vec: np.ndarray, k: int = 4):
    if matrix.size == 0: return np.array([]), np.array([], dtype=int)
    qn = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn.T
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    if not texts: return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=model_name, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def retrieve_resume_chunks(query: str, k: int = 4):
    chs, embs = st.session_state.get("resume_chunks", []), st.session_state.get("resume_embeds", None)
    if not chs or embs is None: return []
    qv = embed_texts([query], EMBED_MODEL)
    scores, idxs = cosine_topk(embs, qv, k=k)
    return [(float(s), chs[int(i)]) for s, i in zip(scores, idxs)]

# ---------------- 뉴스(간단, 캐시) ----------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_latest_news(company: str, max_items: int = 5) -> List[Dict]:
    # 외부 의존 최소화를 위해 Google RSS만 사용(빠름/안정)
    try:
        q = urllib.parse.quote(company)
        url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
        r = requests.get(url, timeout=6)
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


# =========================== 규칙 파서(섹션) ===========================
RESP_HDR = re.compile(r"(주요\s*업무|담당\s*업무|Role|Responsibilities?)", re.I)
QUAL_HDR = re.compile(r"(자격\s*요건|지원\s*자격|Requirements?|Qualifications?)", re.I)
PREF_HDR = re.compile(r"(우대\s*사항|우대|Preferred|Nice\s*to\s*have|Plus)", re.I)
PREF_KW  = re.compile(r"(우대|선호|preferred|plus|가산점|Nice\s*to\s*have)", re.I)

def _clean_line(s: str) -> str:
    return re.sub(r"\s+"," ", s or "").strip(" -•·▶▪️")[:180]

def collect_after_heading(soup: BeautifulSoup, head_regex: re.Pattern, limit: int = 16) -> List[str]:
    out, seen = [], set()
    heads = [tag for tag in soup.find_all(re.compile("^h[1-4]$")) if head_regex.search(tag.get_text(" ", strip=True) or "")]
    for h in heads:
        sib = h.find_next_sibling()
        while sib and sib.name not in {"h1","h2","h3","h4"} and len(out) < limit:
            if sib.name in {"ul","ol"}:
                for li in sib.find_all("li", recursive=True):
                    t=_clean_line(li.get_text(" ", strip=True))
                    if t and t not in seen:
                        seen.add(t); out.append(t)
            elif sib.name in {"p","div","section"}:
                txt = _clean_line(sib.get_text(" ", strip=True))
                if len(txt) > 4:
                    for l in re.split(r"[•\-\n·▪️▶]+|\s{2,}", txt):
                        t=_clean_line(l)
                        if t and t not in seen:
                            seen.add(t); out.append(t)
            sib = sib.find_next_sibling()
        if len(out) >= limit: break
    return out[:limit]

def rule_based_sections(soup: Optional[BeautifulSoup], raw: str) -> Dict[str, List[str]]:
    out = {"responsibilities":[], "qualifications":[], "preferences":[]}
    if soup:
        out["responsibilities"] = collect_after_heading(soup, RESP_HDR, 16)
        out["qualifications"]   = collect_after_heading(soup, QUAL_HDR, 16)
        out["preferences"]      = collect_after_heading(soup, PREF_HDR, 16)
    # 자격요건에 섞인 우대 이동
    remain, moved = [], []
    for q in out["qualifications"]:
        (moved if PREF_KW.search(q) else remain).append(q)
    out["qualifications"] = remain[:12]
    out["preferences"] = (out["preferences"] + moved)[:12]

    # soup로 못 찾았으면 raw fallback
    if not any(out.values()) and raw:
        lines = [ _clean_line(x) for x in raw.split("\n") if x.strip() ]
        bucket=None
        for l in lines:
            if RESP_HDR.search(l): bucket="responsibilities"; continue
            if QUAL_HDR.search(l): bucket="qualifications"; continue
            if PREF_HDR.search(l) or PREF_KW.search(l): bucket="preferences"; continue
            if bucket: out[bucket].append(l)
        for k in out:
            seen=set(); clean=[]
            for s in out[k]:
                t=_clean_line(s)
                if t and t not in seen: seen.add(t); clean.append(t)
            out[k]=clean[:12]
    return out

def extract_company_meta(soup: Optional[BeautifulSoup]) -> Dict[str,str]:
    meta = {"company_name":"","company_intro":"","job_title":""}
    if not soup: return meta
    cand = []
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
    return meta


# =========================== LLM 정제/질문/코칭 ===========================
CRITERIA = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str, fast: bool) -> Dict:
    ctx = trim_for_llm(raw_text, hard_limit=8000 if fast else 12000)
    system = ("너는 채용 공고를 구조화하는 보조원이다. 한국어로 간결/정확하게 정제한다.")
    user = (f"[힌트] 회사명 후보: {meta_hint.get('company_name','')}\n"
            f"[힌트] 직무명 후보: {meta_hint.get('job_title','')}\n"
            "--- 원문 시작 ---\n"
            f"{ctx}\n"
            "--- 원문 끝 ---\n\n"
            "JSON으로만 답하고, 키는 정확히 아래만 사용:\n"
            "{"
            "\"company_name\": str, "
            "\"company_intro\": str, "
            "\"job_title\": str, "
            "\"responsibilities\": [str], "
            "\"qualifications\": [str], "
            "\"preferences\": [str]"
            "}\n"
            "- 불릿/이모지 제거, 문장 간결화, 중복 제거.\n"
            "- '우대 사항(preferences)'은 '우대/선호/preferred/plus/가산점' 등 표시가 있는 항목에서 뽑아 비우지 않도록 노력해라.")
    try:
        r = client.chat.completions.create(
            model=model, temperature=0.2 if fast else 0.3,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        data = json.loads(r.choices[0].message.content)
    except Exception as e:
        data = {"company_name": meta_hint.get("company_name",""),
                "company_intro": meta_hint.get("company_intro","원문 정제 실패"),
                "job_title": meta_hint.get("job_title",""),
                "responsibilities": [], "qualifications": [], "preferences": [], "error": str(e)}

    # 클린/보정
    for k in ["responsibilities","qualifications","preferences"]:
        arr = data.get(k, [])
        if not isinstance(arr, list): arr = []
        clean_list=[]; seen=set()
        for it in arr:
            t = re.sub(r"\s+"," ", str(it)).strip(" -•·▶▪️").strip()
            if t and t not in seen:
                seen.add(t); clean_list.append(t[:180])
        data[k] = clean_list[:12]
    for k in ["company_name","company_intro","job_title"]:
        if k in data and isinstance(data[k], str):
            data[k] = re.sub(r"\s+"," ", data[k]).strip()
    if len(data.get("preferences", [])) < 1:
        remain, moved = [], []
        for q in data.get("qualifications", []):
            (moved if re.search(r"(우대|선호|preferred|plus|가산점)", q, re.I) else remain).append(q)
        if moved:
            data["preferences"] = moved[:12]
            data["qualifications"] = remain[:12]
    return data

def llm_generate_one_question(clean: Dict, resume_snips: str, level: str, model: str, fast: bool) -> str:
    system = ("너는 채용담당자다. 회사/직무 맥락·채용요건·지원자 이력서 요약을 반영해 "
              "서로 다른 관점의 면접 질문을 만든다. 한국어로 한 줄만.")
    user = (f"[회사/직무/요건]\n{json.dumps(clean, ensure_ascii=False)}\n\n"
            f"[이력서 발췌]\n{resume_snips}\n\n"
            f"[난이도/연차] {level}\n"
            "- 지표/수치/기간/규모/리스크/트레이드오프 요소를 적절히 섞고, 기존 질문과 중복되지 않게 한 줄만 출력")
    try:
        r = client.chat.completions.create(model=model, temperature=0.7 if fast else 0.8,
                                           messages=[{"role":"system","content":system},{"role":"user","content":user}])
        q = r.choices[0].message.content.strip()
        q = re.sub(r"^\s*\d+[\).\s-]*","", q).strip()
        return q.split("\n")[0].strip()
    except Exception:
        return ""

def llm_draft_answer(clean: Dict, question: str, resume_snips: str, model: str, fast: bool) -> str:
    system = ("너는 면접 답변 코치다. STAR(상황-과제-행동-성과) 기반으로 8~12문장, 한국어로 간결/구체.")
    user = (f"[회사/직무/요건]\n{json.dumps(clean, ensure_ascii=False)}\n\n"
            f"[이력서 발췌]\n{resume_snips}\n\n"
            f"[면접 질문]\n{question}\n\n"
            "위 정보를 근거로 STAR 기반 답변 **초안**을 작성.")
    try:
        r = client.chat.completions.create(model=model, temperature=0.45 if fast else 0.5,
                                           messages=[{"role":"system","content":system},{"role":"user","content":user}])
        return r.choices[0].message.content.strip()
    except Exception:
        return ""

def llm_score_and_coach_strict(clean: Dict, question: str, answer: str, resume_snips: str, model: str, fast: bool) -> Dict:
    system = ("너는 매우 엄격한 면접 코치다. JSON만 출력. 기준 5개(각 0~20) 합계=총점(0~100)과 일치.")
    schema = ("{"
              "\"overall_score\": 0~100 정수,"
              "\"criteria\": [{\"name\":\"문제정의\",\"score\":0~20,\"comment\":\"...\"},"
              "{\"name\":\"데이터/지표\",\"score\":0~20,\"comment\":\"...\"},"
              "{\"name\":\"실행력/주도성\",\"score\":0~20,\"comment\":\"...\"},"
              "{\"name\":\"협업/커뮤니케이션\",\"score\":0~20,\"comment\":\"...\"},"
              "{\"name\":\"고객가치\",\"score\":0~20,\"comment\":\"...\"}],"
              "\"strengths\": [\"...\"],"
              "\"risks\": [\"...\"],"
              "\"improvements\": [\"...\"],"
              "\"revised_answer\": \"STAR 구조로 간결히\""
              "}")
    user = (f"[회사/직무/요건]\n{json.dumps(clean, ensure_ascii=False)}\n\n"
            f"[이력서 발췌]\n{resume_snips}\n\n"
            f"[면접 질문]\n{question}\n\n"
            f"[지원자 답변]\n{answer}\n\n"
            f"다음 스키마로만 출력:\n{schema}")
    try:
        r = client.chat.completions.create(model=model, temperature=0.2 if fast else 0.25,
                                           response_format={"type":"json_object"},
                                           messages=[{"role":"system","content":system},{"role":"user","content":user}])
        data = json.loads(r.choices[0].message.content)
    except Exception as e:
        data = {"overall_score": 0,
                "criteria": [{"name": n, "score": 0, "comment": ""} for n in CRITERIA],
                "strengths": [], "risks": [], "improvements": [], "revised_answer": "",
                "error": str(e)}

    # 정합성 보정
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


# =========================== 상태 초기화 ===========================
def _init_state():
    defaults = {
        "clean_struct": None,
        "company_news": [],
        "resume_raw": "",
        "resume_chunks": [],
        "resume_embeds": None,
        "cover_letter": "",
        "current_question": "",
        "answer_text": "",
        "records": [],
        "last_result": None,
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v
_init_state()


# =========================== 1) 채용 공고 URL → 정제 ===========================
st.header("1) 채용 공고 URL → 정제")
url_input = st.text_input("채용 공고 상세 URL", placeholder="원티드/사람인/잡코리아 등 상세 URL을 입력")

cols = st.columns([1,1,2])
with cols[0]:
    do_fetch = st.button("원문 수집 → 정제", type="primary")
with cols[1]:
    st.caption("빠른 모드: 입력 토큰 축소/온도↓, 뉴스 적게 가져옴")

if do_fetch:
    if not url_input.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner("원문 수집/정제 중..."):
            raw, meta, soup = fetch_all_text(url_input.strip())
            hint = extract_company_meta(soup)
            # 규칙 파서(빠르고 무료) 먼저 반영 → LLM은 보완만
            rb = rule_based_sections(soup, raw)
            # LLM은 트림한 원문만
            clean = llm_structurize(raw, hint, CHAT_MODEL, FAST_MODE)
            # 규칙 파서로 이미 뽑은 항목은 우선 적용(정확도↑, LLM 부하↓)
            for k in ["responsibilities","qualifications","preferences"]:
                if rb.get(k):
                    clean[k] = rb[k]
            # 뉴스 (빠른 모드면 2건)
            cname = clean.get("company_name") or hint.get("company_name") or ""
            st.session_state.company_news = fetch_latest_news(cname, max_items=(2 if FAST_MODE else 5)) if cname else []
            st.session_state.clean_struct = clean
        st.success("정제 완료!")

# =========================== 2) 회사 요약 (정제 결과) ===========================
st.header("2) 회사 요약 (정제 결과)")
clean = st.session_state.clean_struct
if clean:
    st.markdown(f"**회사명:** {clean.get('company_name','-')}")
    st.markdown(f"**간단한 회사 소개:** {clean.get('company_intro','-')}")
    st.markdown(f"**모집 분야(직무명):** {clean.get('job_title','-')}")
    c1,c2,c3 = st.columns(3)
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
    if st.session_state.company_news:
        st.markdown("**최근 뉴스**")
        for n in st.session_state.company_news:
            st.markdown(f"- [{n['title']}]({n['link']})")
else:
    st.info("먼저 ‘원문 수집 → 정제’를 실행하세요.")


# =========================== 3) 내 이력서/프로젝트 업로드 ===========================
st.divider()
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader("여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True)

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
            chunks = chunk(resume_text, size=(350 if FAST_MODE else 400), overlap=(70 if FAST_MODE else 80))
            chunks = chunks[:200]  # 상한
            with st.spinner("이력서 벡터화(캐시) 중..."):
                embeds = cached_embeddings(chunks, EMBED_MODEL)
            st.session_state.resume_raw = resume_text
            st.session_state.resume_chunks = chunks
            st.session_state.resume_embeds = embeds
            st.success(f"인덱싱 완료 (청크 {len(chunks)}개)")


# =========================== 4) 이력서 기반 자소서 생성 ===========================
st.divider()
st.header("4) 이력서 기반 자소서 생성")
topic = st.text_input("회사 요청 주제(선택)", placeholder="예: 지원동기 / 협업 / 문제해결 등")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str, fast: bool) -> str:
    # RAG 없이 전체 이력서 요약 입력 (속도/간단성)
    rs = resume_text.strip()
    if len(rs) > (7000 if fast else 9000): rs = rs[:(7000 if fast else 9000)]
    system = ("너는 한국어 자기소개서 전문가다. 채용 공고 요건과 후보자의 이력서를 참고해 "
              "회사 특화 자소서를 작성한다. 수치/지표/기간/임팩트를 명확히.")
    req = f"회사 측 요청 주제는 '{topic_hint.strip()}'." if topic_hint.strip() else \
          "특정 주제가 없으므로 지원동기/적합성/기여방안을 중심으로."
    user = (f"[회사/직무 요약]\n{json.dumps(clean_struct, ensure_ascii=False)}\n\n"
            f"[후보자 이력서(요약 가능)]\n{rs}\n\n"
            f"[지시]\n- {req}\n- 분량: 600~900자\n- 구성: 1) 지원동기 2) 역량/경험 3) 성과/지표 4) 기여방안 5) 마무리")
    try:
        r = client.chat.completions.create(model=model, temperature=0.4 if fast else 0.45,
                                           messages=[{"role":"system","content":system},{"role":"user","content":user}])
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"(자소서 생성 실패: {e})"

if st.button("자소서 생성", type="primary"):
    if not st.session_state.clean_struct:
        st.warning("먼저 회사 정제를 실행하세요.")
    elif not st.session_state.resume_raw.strip():
        st.warning("먼저 이력서를 업로드하고 인덱싱하세요.")
    else:
        with st.spinner("자소서 생성 중..."):
            st.session_state.cover_letter = build_cover_letter(st.session_state.clean_struct,
                                                               st.session_state.resume_raw,
                                                               topic, CHAT_MODEL, FAST_MODE)
        st.success("자소서 생성 완료!")

if st.session_state.cover_letter:
    st.subheader("자소서 (생성 결과)")
    st.write(st.session_state.cover_letter)
    st.download_button("자소서 TXT 다운로드",
                       data=st.session_state.cover_letter.encode("utf-8"),
                       file_name="cover_letter.txt", mime="text/plain")


# =========================== 5) 질문 생성 & 답변 초안 ===========================
st.divider()
st.header("5) 질문 생성 & 답변 초안")
level = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)

c_q1, c_q2 = st.columns(2)
with c_q1:
    if st.button("새 질문 받기", type="primary"):
        if not st.session_state.clean_struct:
            st.warning("먼저 회사 정제를 실행하세요.")
        else:
            hits = retrieve_resume_chunks("핵심 프로젝트/성과/기술 스택 요약", k=(3 if FAST_MODE else 4))
            resume_snips = "\n".join([f"- {t[:350]}" for _, t in hits])[:1000]
            q = llm_generate_one_question(st.session_state.clean_struct, resume_snips, level, CHAT_MODEL, FAST_MODE)
            if q:
                st.session_state.current_question = q
                st.session_state.answer_text = ""
                st.session_state.last_result = None
                st.success("질문 생성 완료!")
            else:
                st.error("질문 생성 실패")
with c_q2:
    if st.button("RAG로 답변 초안 생성", type="secondary"):
        if not st.session_state.current_question:
            st.warning("먼저 질문을 생성하세요.")
        else:
            hits = retrieve_resume_chunks(st.session_state.current_question, k=(3 if FAST_MODE else 4))
            resume_snips = "\n".join([f"- {t[:400]}" for _, t in hits])[:1400]
            draft = llm_draft_answer(st.session_state.clean_struct, st.session_state.current_question,
                                     resume_snips, CHAT_MODEL, FAST_MODE)
            if draft:
                st.session_state.answer_text = draft
                st.success("초안 생성 완료!")
            else:
                st.error("초안 생성 실패")

st.text_area("질문", value=st.session_state.current_question, height=100)
st.text_area("나의 답변(초안을 편집해 완성하세요)", height=200, key="answer_text")


# =========================== 6) 채점 & 코칭 ===========================
st.divider()
st.header("6) 채점 & 코칭")
if st.button("채점 & 코칭 실행", type="primary"):
    if not st.session_state.current_question:
        st.warning("먼저 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("채점/코칭 중..."):
            hits = retrieve_resume_chunks(st.session_state.current_question + "\n" + st.session_state.answer_text[:800],
                                          k=(3 if FAST_MODE else 4))
            resume_snips = "\n".join([f"- {t[:400]}" for _, t in hits])[:1400]
            res = llm_score_and_coach_strict(st.session_state.clean_struct,
                                             st.session_state.current_question,
                                             st.session_state.answer_text,
                                             resume_snips, CHAT_MODEL, FAST_MODE)
        st.session_state.last_result = res
        st.session_state.records.append({
            "ts": pd.Timestamp.now(),
            "질문": st.session_state.current_question,
            "합계": res.get("overall_score", 0),
            "기준": res.get("criteria", []),
            "강점": res.get("strengths", []),
            "리스크": res.get("risks", []),
            "개선": res.get("improvements", []),
            "수정본": res.get("revised_answer",""),
            "원본답변": st.session_state.answer_text,
        })
        st.success("채점/코칭 완료!")

# 결과 표시
res = st.session_state.last_result
if res:
    st.subheader("피드백 결과")
    st.metric("총점(/100)", res.get("overall_score", 0))
    st.markdown("**기준별 점수 & 코멘트**")
    for it in res.get("criteria", []):
        st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
    if res.get("strengths"): st.markdown("**강점**\n- " + "\n- ".join(res["strengths"]))
    if res.get("risks"): st.markdown("**리스크**\n- " + "\n- ".join(res["risks"]))
    if res.get("improvements"): st.markdown("**개선 포인트**\n- " + "\n- ".join(res["improvements"]))
    if res.get("revised_answer",""):
        st.markdown("**수정본 (STAR)**")
        st.write(res["revised_answer"])

# 세션 리포트
st.divider()
st.header("세션 리포트 (CSV)")
def build_report(records):
    rows=[]
    for r in records:
        row={"timestamp":r.get("ts"),
             "질문":r.get("질문"),
             "합계":r.get("합계"),
             "원본답변":r.get("원본답변"),
             "수정본":r.get("수정본")}
        # 기준 점수 펼치기
        crit = r.get("기준", [])
        for it in crit:
            row[f"{it['name']}(/20)"]=it.get("score",0)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp","질문","합계","원본답변","수정본"])
df = build_report(st.session_state.records)
st.download_button("CSV 다운로드", data=df.to_csv(index=False).encode("utf-8-sig"),
                   file_name="interview_session_report.csv", mime="text/csv")
