# -*- coding: utf-8 -*-
"""
회사 맞춤 면접 코치 (전 방식 복구 + 사이트별 크롤러 + 자소서 유지 + 평가 확장 + 시각화 개선 + 수정본 제거)

요약:
1) 채용 공고 URL → 원문 수집(3단 폴백: Jina → 일반 HTML→ bs4) → LLM 구조화(JSON) → 규칙 파서 보정
   + 원티드/사람인/잡코리아 정밀 크롤러(경량): 헤더/리스트 휴리스틱으로 주요업무/자격요건/우대사항 직접 추출
2) 회사 요약: 구조화 결과 표시 (회사명/소개/직무/주요업무/자격요건/우대사항)
3) 이력서 업로드(pdf/txt/md/docx) → 내부 자동 RAG 인덱스
4) 자소서 생성: 결과를 session_state에 저장하여 다른 단계 후에도 사라지지 않음
5) 질문 생성 & 답변 초안(RAG 결합)
6) 채점 & 코칭(엄격) — '수정본(STAR)' 완전 제거, 평가항목 확장(총10개, 각0~10점, 합계100)
7) 팔로업 질문 & 답변 & 피드백 — 동일한 평가 스키마, '수정본' 없음
8) 피드백/시각화: 개별 답변 선택 시 궤적(레이더) 각각 표시, 미선택 시 평균 표시
9) 리포트/다운로드: 한글 컬럼(질문/합계 등)로 표준화

필수 패키지:
- openai, requests, beautifulsoup4, html2text, pypdf, docx2txt, lxml, numpy, pandas, plotly, streamlit
"""

import os, re, io, json, urllib.parse, tempfile
from typing import Optional, Tuple, Dict, List

import requests
from bs4 import BeautifulSoup
import html2text
import streamlit as st
import numpy as np
import pandas as pd

# Plotly (레이더 차트)
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ----------------------------- 스트림릿 페이지 설정 -----------------------------
st.set_page_config(page_title="회사 맞춤 면접 코치", page_icon="🎯", layout="wide")
st.title("회사 맞춤 면접 코치 · URL 정제 → 자소서 → 질문/RAG/채점 → 팔로업")

# ----------------------------- OpenAI 클라이언트 -----------------------------
try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` 패키지를 설치하세요. requirements.txt에 openai 추가.")
    st.stop()

API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
if not API_KEY:
    API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password")
if not API_KEY:
    st.stop()

client = OpenAI(api_key=API_KEY)

with st.sidebar:
    st.subheader("모델 설정")
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델(내부)", ["text-embedding-3-small","text-embedding-3-large"], index=0)

# ----------------------------- 공통 유틸 -----------------------------
def normalize_url(u: str) -> Optional[str]:
    """URL을 https:// 형태로 표준화하고, #fragment 제거."""
    if not u: return None
    u = u.strip()
    if not re.match(r"^https?://", u): u = "https://" + u
    parts = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

def http_get(url: str, timeout: int = 12) -> Optional[requests.Response]:
    """200 & text/html일 때만 반환. (로그인/동적페이지는 실패할 수 있음)"""
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124 Safari/537.36",
                "Accept-Language": "ko, en;q=0.9",
            },
            timeout=timeout,
        )
        if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
            return r
    except Exception:
        pass
    return None

# ----------------------------- 원문 수집(3단 폴백) -----------------------------
def fetch_jina_text(url: str, timeout: int = 15) -> str:
    """
    1) Jina Reader 프록시 (정적 스냅샷 비슷한 뷰) — 로그인/봇차단 우회에 유리할 때가 있음
    """
    try:
        parts = urllib.parse.urlsplit(url)
        prox = f"https://r.jina.ai/http://{parts.netloc}{parts.path}"
        if parts.query: prox += f"?{parts.query}"
        r = http_get(prox, timeout=timeout)
        return r.text.strip() if r else ""
    except Exception:
        return ""

def html_to_text(html_str: str) -> str:
    """2) 일반 HTML → 마크다운 텍스트 변환(링크/이미지 무시, 너비제한 해제)"""
    conv = html2text.HTML2Text()
    conv.ignore_links = True
    conv.ignore_images = True
    conv.body_width = 0
    txt = conv.handle(html_str)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def fetch_webbase_text(url: str) -> str:
    r = http_get(url, timeout=12)
    if not r: return ""
    return html_to_text(r.text)

def fetch_bs4_text(url: str) -> Tuple[str, Optional[BeautifulSoup]]:
    """
    3) bs4 DOM 파싱 → article/section/main/div/ul/ol 큰 덩어리 추출 → 합침
       soup은 메타(회사명/설명/제목) 추출에 사용
    """
    r = http_get(url, timeout=12)
    if not r: return "", None
    soup = BeautifulSoup(r.text, "lxml")
    blocks = []
    for sel in ["article","section","main","div","ul","ol"]:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) > 300:
                txt = re.sub(r"\s+"," ", txt)
                blocks.append(txt)
    if not blocks:
        return soup.get_text(" ", strip=True)[:120000], soup
    seen, out = set(), []
    for b in blocks:
        if b not in seen:
            seen.add(b); out.append(b)
    return ("\n\n".join(out)[:120000], soup)

def fetch_all_text(url: str):
    """Jina → 일반 HTML → bs4 순서로 시도. (텍스트, 메타정보, soup 반환)"""
    url = normalize_url(url)
    if not url: return "", {"error":"invalid_url"}, None
    jina = fetch_jina_text(url)
    if jina:
        _, soup = fetch_bs4_text(url)
        return jina, {"source":"jina","len":len(jina),"url_final":url}, soup
    web = fetch_webbase_text(url)
    if web:
        _, soup = fetch_bs4_text(url)
        return web, {"source":"webbase","len":len(web),"url_final":url}, soup
    bs, soup = fetch_bs4_text(url)
    return bs, {"source":"bs4","len":len(bs),"url_final":url}, soup

# ----------------------------- 메타 추출(회사명/소개/제목) -----------------------------
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

# ----------------------------- 사이트별 정밀 크롤러 (경량) -----------------------------
def _find_section_by_header(soup: BeautifulSoup, header_keywords, max_take=20) -> List[str]:
    """헤더 텍스트(주요업무/자격요건/우대사항 등)를 찾아 다음 형제 요소의 리스트/문단을 수집."""
    if not soup: return []
    # 헤더 후보: h1~h4, strong, b, span(粗)
    headers = soup.find_all(["h1","h2","h3","h4","strong","b","span"])
    out=[]
    for h in headers:
        ht = (h.get_text(" ", strip=True) or "").lower()
        if any(k in ht for k in header_keywords):
            # 형제 방향으로 ul/ol/li/p를 따라가며 bullet 수집
            sib = h
            take=[]
            steps=0
            while sib and steps<10:
                sib = sib.find_next_sibling()
                if not sib: break
                steps += 1
                txts=[]
                if sib.name in ["ul","ol"]:
                    for li in sib.find_all("li", recursive=False):
                        tv = li.get_text(" ", strip=True)
                        if tv: txts.append(tv)
                elif sib.name in ["p","div"]:
                    tv = sib.get_text(" ", strip=True)
                    if tv: txts.append(tv)
                for t in txts:
                    t = re.sub(r"\s+"," ", t).strip(" -•·▶▪️").strip()
                    if 3 <= len(t) <= 300:
                        out.append(t)
                if len(out) >= max_take: break
            if out: break
    # 중복 제거
    seen=set(); clean=[]
    for s in out:
        if s not in seen:
            seen.add(s); clean.append(s[:180])
    return clean

def parse_wanted(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """원티드: 헤더 기반 휴리스틱(주요업무/자격요건/우대사항)"""
    return {
        "responsibilities": _find_section_by_header(soup, ["주요 업무","담당 업무","responsibilities","what you will do"]),
        "qualifications":   _find_section_by_header(soup, ["자격 요건","지원 자격","requirements","qualifications","must have"]),
        "preferences":      _find_section_by_header(soup, ["우대 사항","우대","preferred","nice to have","plus"]),
    }

def parse_saramin(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """사람인: 유사 휴리스틱"""
    return {
        "responsibilities": _find_section_by_header(soup, ["담당업무","담당 업무","주요 업무","업무내용"]),
        "qualifications":   _find_section_by_header(soup, ["자격요건","자격 요건","지원자격","학력/경력","필수"]),
        "preferences":      _find_section_by_header(soup, ["우대사항","우대 사항","우대","가산점","우대조건"]),
    }

def parse_jobkorea(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """잡코리아: 유사 휴리스틱"""
    return {
        "responsibilities": _find_section_by_header(soup, ["담당업무","주요업무","업무내용"]),
        "qualifications":   _find_section_by_header(soup, ["자격요건","지원자격","필수"]),
        "preferences":      _find_section_by_header(soup, ["우대사항","우대","우대조건"]),
    }

def try_site_specific(url: str, soup: Optional[BeautifulSoup]) -> Dict[str, List[str]]:
    """
    도메인 인식 후 사이트별 파서 시도 → 일부라도 얻으면 반환, 아니면 빈 dict
    """
    if not soup: return {}
    dom = urllib.parse.urlsplit(url).netloc.lower()
    if "wanted.co.kr" in dom:
        return parse_wanted(soup)
    if "saramin.co.kr" in dom:
        return parse_saramin(soup)
    if "jobkorea.co.kr" in dom:
        return parse_jobkorea(soup)
    return {}

# ----------------------------- 규칙 파서(우대사항 보정/보완) -----------------------------
def rule_based_sections(raw_text: str) -> dict:
    """
    헤더/키워드 기반으로 responsibilities/qualifications/preferences를 최대한 채움.
    LLM 구조화가 빈약할 때 보완 역할.
    """
    txt = re.sub(r"\r", "", raw_text or "").strip()
    lines = [re.sub(r"\s+", " ", l).strip(" -•·▶▪️") for l in txt.split("\n")]
    lines = [l for l in lines if l]

    hdr_resp = re.compile(r"(주요\s*업무|담당\s*업무|Role|Responsibilities?)", re.I)
    hdr_qual = re.compile(r"(자격\s*요건|지원\s*자격|Requirements?|Qualifications?)", re.I)
    hdr_pref = re.compile(r"(우대\s*사항|우대|Preferred|Nice\s*to\s*have|Plus)", re.I)

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
            if re.search(r"(우대|preferred|nice to have|plus|가산점|있으면\s*좋음)", l, re.I):
                bucket = "preferences"
            elif any(k in l.lower() for k in ["java","python","spark","airflow","kafka","ml","sql","aws","docker","k8s"]):
                bucket = "responsibilities"
            else:
                continue
        push(l, bucket)

    # 자격요건에 섞인 우대 줄 이동
    kw_pref = re.compile(r"(우대|preferred|nice to have|plus|가산점|있으면\s*좋음)", re.I)
    remain_qual = []
    for q in out["qualifications"]:
        if kw_pref.search(q):
            out["preferences"].append(q)
        else:
            remain_qual.append(q)
    out["qualifications"] = remain_qual

    # 중복/길이 제한
    for k in out:
        seen=set(); clean=[]
        for s in out[k]:
            s = re.sub(r"\s+", " ", s).strip(" -•·▶▪️").strip()
            if s and s not in seen:
                seen.add(s); clean.append(s)
        out[k] = clean[:20]
    return out

# ----------------------------- LLM 구조화 (전 방식 복구) -----------------------------
PROMPT_SYSTEM_STRUCT = (
    "너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
    "입력 텍스트는 포털 광고 문구, UI잔재, 복수 직무가 섞여 있을 수 있다. "
    "한국어로 간결하고 중복없이 정제하라."
)

def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
    """
    원문 → JSON(회사명, 소개, 직무명, 주요업무, 자격요건, 우대사항)
    - LLM으로 1차 구조화
    - 사이트별 크롤러/규칙 파서로 보강 (특히 우대사항)
    """
    ctx = (raw_text or "").strip()
    if len(ctx) > 14000:
        ctx = ctx[:14000]

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
            "- '우대 사항(preferences)'은 비워두지 말고, 원문에서 '우대/선호/Preferred/Nice to have/Plus' 등 표시가 있는 항목을 그대로 담아라.\n"
            "- 불릿/마커/이모지 제거, 문장 간결화, 중복 제거."
        ),
    }

    data = {
        "company_name": meta_hint.get("company_name",""),
        "company_intro": meta_hint.get("company_intro",""),
        "job_title": meta_hint.get("job_title",""),
        "responsibilities": [],
        "qualifications": [],
        "preferences": [],
    }
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg],
        )
        data_llm = json.loads(resp.choices[0].message.content)
        # 병합(LLM 결과 우선, 힌트로 빈 값 채움)
        for k in data:
            if k in data_llm and data_llm[k]:
                data[k] = data_llm[k]
    except Exception:
        pass

    # 1차 클린업
    for k in ["responsibilities","qualifications","preferences"]:
        arr = data.get(k, [])
        if not isinstance(arr, list): arr = []
        clean=[]; seen=set()
        for it in arr:
            t = re.sub(r"\s+"," ", str(it)).strip(" -•·▶▪️").strip()
            if t and t not in seen:
                seen.add(t); clean.append(t[:200])
        data[k] = clean[:20]

    for k in ["company_name","company_intro","job_title"]:
        if k in data and isinstance(data[k], str):
            data[k] = re.sub(r"\s+"," ", data[k]).strip()

    return data

# ----------------------------- 파일 리더 (pdf/txt/md/docx) -----------------------------
try:
    import pypdf
except Exception:
    pypdf = None

def read_pdf(data: bytes) -> str:
    if pypdf is None:
        return ""
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

# ----------------------------- 간단 RAG (내부 자동 파라미터) -----------------------------
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

# ----------------------------- 세션 상태 초기화 -----------------------------
def _init_state():
    defaults = dict(
        clean_struct=None,
        raw_source_text="",
        site_sections=None,
        resume_raw="",
        resume_chunks=[],
        resume_embeds=None,
        cover_letter="",
        current_question="",
        answer_text="",
        records=[],  # [{질문, 합계, criteria:{기준명:점수}, 코멘트들...}]
        followups=[],  # 제안 리스트
        selected_followup="",
        followup_answer="",
        last_result=None,
        last_followup_result=None,
    )
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v
_init_state()

# ----------------------------- 1) 채용 공고 URL → 정제 -----------------------------
st.header("1) 채용 공고 URL → 정제")
job_url = st.text_input("채용 공고 상세 URL", placeholder="예: https://www.wanted.co.kr/wd/123456")
if st.button("원문 수집 → 정제", type="primary"):
    if not job_url.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner("원문 수집 중..."):
            raw, meta, soup = fetch_all_text(job_url.strip())
            st.session_state.raw_source_text = raw or ""
            site_parts = try_site_specific(job_url.strip(), soup)  # 사이트별 정밀 추출(가능한 경우)
        if not raw:
            st.error("원문을 가져오지 못했습니다. (로그인/동적 렌더링/봇 차단 가능)")
        else:
            # LLM 구조화 (전 방식)
            with st.spinner("LLM으로 구조화 중..."):
                hint = extract_company_meta(soup)
                clean = llm_structurize(raw, hint, CHAT_MODEL)

            # 사이트별 파싱 결과를 LLM 구조화에 보강(우선 적용)
            if site_parts:
                for k in ["responsibilities","qualifications","preferences"]:
                    lst = (site_parts.get(k) or []) + (clean.get(k) or [])
                    seen=set(); merged=[]
                    for x in lst:
                        x=x.strip()
                        if x and x not in seen:
                            seen.add(x); merged.append(x)
                    clean[k] = merged[:20]

            # 규칙 파서로 최종 보강(특히 우대사항)
            rb = rule_based_sections(raw)
            for k in ["responsibilities","qualifications","preferences"]:
                lst = (clean.get(k) or []) + (rb.get(k) or [])
                seen=set(); merged=[]
                for x in lst:
                    x=x.strip()
                    if x and x not in seen:
                        seen.add(x); merged.append(x)
                clean[k] = merged[:20]

            st.session_state.clean_struct = clean
            st.session_state.site_sections = site_parts or {}
            st.success("정제 완료!")

# ----------------------------- 2) 회사 요약 (정제 결과 표시) -----------------------------
st.header("2) 회사 요약 (정제 결과)")
clean = st.session_state.clean_struct
if clean:
    st.markdown(f"**회사명:** {clean.get('company_name','-')}")
    st.markdown(f"**간단한 회사 소개(요약):** {clean.get('company_intro','-')}")
    st.markdown(f"**모집 분야(직무명):** {clean.get('job_title','-')}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**주요 업무**")
        rs = clean.get("responsibilities", [])
        if rs:
            for b in rs: st.markdown(f"- {b}")
        else:
            st.caption("주요 업무 정보가 없습니다.")
    with c2:
        st.markdown("**자격 요건**")
        qs = clean.get("qualifications", [])
        if qs:
            for b in qs: st.markdown(f"- {b}")
        else:
            st.caption("자격 요건 정보가 없습니다.")
    with c3:
        st.markdown("**우대 사항**")
        ps = clean.get("preferences", [])
        if ps:
            for b in ps: st.markdown(f"- {b}")
        else:
            st.caption("우대 사항 정보가 없습니다.")
else:
    st.info("먼저 URL을 정제해 주세요.")

st.divider()

# ----------------------------- 3) 내 이력서/프로젝트 업로드 -----------------------------
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader("여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True)
if st.button("이력서 인덱싱(자동)", type="secondary"):
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
            chunks = chunk(resume_text, size=600, overlap=120)  # 내부 자동 파라미터
            with st.spinner("이력서 벡터화 중..."):
                embeds = embed_texts(chunks, EMBED_MODEL)
            st.session_state.resume_raw = resume_text
            st.session_state.resume_chunks = chunks
            st.session_state.resume_embeds = embeds
            st.success(f"인덱싱 완료 (청크 {len(chunks)}개)")

st.divider()

# ----------------------------- 4) 이력서 기반 자소서 생성 (세션 유지) -----------------------------
st.header("4) 이력서 기반 자소서 생성")
topic = st.text_input("회사 요청 주제(선택)", placeholder="예: 직무 지원동기 / 협업 경험 / 문제해결 사례 등")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str) -> str:
    """회사 요약 + 이력서 → 회사 특화 자소서(600~900자) — 결과는 세션에 영구 저장"""
    company = json.dumps(clean_struct or {}, ensure_ascii=False)
    resume_snippet = (resume_text or "").strip()
    if len(resume_snippet) > 9000: resume_snippet = resume_snippet[:9000]
    system = (
        "너는 한국어 자기소개서 전문가다. 채용 공고의 회사/직무 요건과 후보자의 이력서를 참고해 "
        "회사 특화 자소서를 작성한다. 과장/허위는 금지하고, 수치/지표/기간/임팩트 중심으로 구체화한다."
    )
    req = f"회사 측 요청 주제는 '{topic_hint.strip()}' 이다. 이 주제를 중심으로 서술하라." if topic_hint else \
          "특정 주제가 없으므로, 채용 공고와 직무적합성을 중심으로 지원동기와 핵심역량을 강조하라."
    user = (
        f"[회사/직무 요약(JSON)]\n{company}\n\n"
        f"[후보자 이력서]\n{resume_snippet}\n\n"
        f"[작성 지시]\n- {req}\n"
        "- 분량: 600~900자\n"
        "- 구성: 1) 지원 동기 2) 직무 관련 핵심 역량·경험 3) 성과/지표 4) 입사 후 기여 방안 5) 마무리\n"
        "- 불필요한 미사여구/중복/광고 문구 삭제."
    )
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.4,
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(자소서 생성 실패: {e})"

if st.button("자소서 생성", type="primary"):
    if not st.session_state.clean_struct:
        st.warning("먼저 회사 URL을 정제하세요.")
    elif not st.session_state.resume_raw.strip():
        st.warning("먼저 이력서를 업로드하고 '이력서 인덱싱(자동)'을 눌러주세요.")
    else:
        with st.spinner("자소서 생성 중..."):
            cover = build_cover_letter(st.session_state.clean_struct, st.session_state.resume_raw, topic, CHAT_MODEL)
        st.session_state.cover_letter = cover  # ✅ 세션에 저장 — 이후 단계에서도 유지
        st.success("자소서 생성 완료!")

# ✅ 자소서는 항상 화면에 유지해서, 이후 단계(질문/채점 등) 진행해도 사라지지 않음
if st.session_state.cover_letter:
    st.subheader("자소서 (생성 결과)")
    st.write(st.session_state.cover_letter)
    st.download_button("자소서 TXT 다운로드", data=st.session_state.cover_letter.encode("utf-8"),
                       file_name="cover_letter.txt", mime="text/plain")

st.divider()

# ----------------------------- 5) 질문 생성 & 답변 초안 (RAG 결합) -----------------------------
st.header("5) 질문 생성 & 답변 초안 (RAG 결합)")
level = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)

def llm_generate_one_question_with_resume(clean: Dict, level: str, model: str) -> str:
    hits = retrieve_resume_chunks("핵심 프로젝트와 기술 스택 요약", k=4)
    resume_context = "\n".join([f"- {t[:350]}" for _, t in hits])[:1200]
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {
        "role": "user",
        "content": (
            f"[회사/직무/요건]\n{ctx}\n\n"
            f"[지원자 이력서 요약(발췌)]\n{resume_context}\n\n"
            f"[요청]\n- 난이도/연차: {level}\n"
            f"- 중복/유사도 지양, 회사 요건과 이력서의 교집합 또는 공백영역 겨냥\n"
            f"- 한국어 면접 질문 1개만 한 줄로 출력"
        ),
    }
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.85,
            messages=[{"role":"system","content":"면접 질문 생성기"}, user_msg],
        )
        q = resp.choices[0].message.content.strip()
        q = re.sub(r"^\s*\d+[\).\s-]*","", q).strip()
        q = q.split("\n")[0].strip()
        return q
    except Exception:
        return ""

def llm_draft_answer(clean: Dict, question: str, model: str) -> str:
    hits = retrieve_resume_chunks(question, k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {
        "role":"user",
        "content": (
            f"[회사/직무/채용요건]\n{ctx}\n\n"
            f"[지원자 이력서 발췌]\n{resume_text}\n\n"
            f"[면접 질문]\n{question}\n\n"
            "STAR(상황-과제-행동-성과) 기반 한국어 답변 **초안**을 8~12문장으로 작성해줘. "
            "가능하면 지표/수치/기간/임팩트를 포함."
        )
    }
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.5,
            messages=[{"role":"system","content":"면접 답변 초안기"}, user_msg],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

cols_q = st.columns(2)
with cols_q[0]:
    if st.button("새 질문 받기", type="primary"):
        if not st.session_state.clean_struct:
            st.warning("먼저 회사 URL을 정제하세요.")
        else:
            q = llm_generate_one_question_with_resume(st.session_state.clean_struct, level, CHAT_MODEL)
            if q:
                st.session_state.current_question = q
                st.session_state.answer_text = ""   # 이전 답변 초기화
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
            draft = llm_draft_answer(st.session_state.clean_struct, st.session_state.current_question, CHAT_MODEL)
            if draft:
                st.session_state.answer_text = draft
                st.success("초안 생성 완료!")
            else:
                st.error("초안 생성 실패")

st.text_area("질문", value=st.session_state.current_question, height=100)
st.text_area("나의 답변 (초안을 편집해 완성하세요)", height=200, key="answer_text")

st.divider()

# ----------------------------- 6) 채점 & 코칭 (확장된 평가 항목, 수정본 제거) -----------------------------
st.header("6) 채점 & 코칭 (확장 기준, 엄격 모드)")

# 평가 항목 확장 (총 10개, 각 0~10점, 합계 100)
CRITERIA = [
    "문제정의", "데이터/지표", "실행력/주도성", "협업/커뮤니케이션", "고객가치",
    "문제해결", "리스크/품질", "비즈니스임팩트", "커뮤니케이션 명료성", "구조화/논리"
]

PROMPT_SYSTEM_SCORE_STRICT = (
    "너는 매우 엄격한 면접 코치다. 아래 JSON 스키마만 출력하라. "
    "각 기준은 0~10 정수이며, 총점(overall)은 기준 합계(최대 100)와 반드시 일치해야 한다. "
    "과장/모호함/근거 부재/숫자 없는 주장/책임 회피/모호한 주어 사용 등을 강하게 감점하라. "
    "각 기준에 대해 짧고 구체적인 코멘트를 제공하라. 수정본 답변은 제공하지 마라."
)

def llm_score_and_coach_strict(clean: Dict, question: str, answer: str, model: str) -> Dict:
    hits = retrieve_resume_chunks(question + "\n" + answer[:800], k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    ctx = json.dumps(clean, ensure_ascii=False)

    # ❗ 수정본 답변 제거: 스키마에서 제외
    schema = (
        "{"
        "\"overall\": 0~100 정수,"
        "\"criteria\": ["
        + ",".join([f"{{\"name\":\"{c}\",\"score\":0~10,\"comment\":\"...\"}}" for c in CRITERIA]) +
        "],"
        "\"strengths\": [\"...\",\"...\"],"
        "\"risks\": [\"...\",\"...\"],"
        "\"improvements\": [\"...\",\"...\",\"...\"]"
        "}"
    )

    user_msg = {
        "role":"user",
        "content": (
            f"[회사/직무/채용요건]\n{ctx}\n\n"
            f"[지원자 이력서 발췌]\n{resume_text}\n\n"
            f"[면접 질문]\n{question}\n\n"
            f"[지원자 답변]\n{answer}\n\n"
            f"다음 JSON 스키마로만 한국어 응답:\n{schema}"
        )
    }
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_SCORE_STRICT}, user_msg]
        )
        data = json.loads(resp.choices[0].message.content)
        crit = data.get("criteria", [])
        # 기준 보정/정렬
        fixed=[]
        # 들어온 기준을 딕트화
        got = {str(it.get("name","")).strip(): it for it in crit if isinstance(it, dict)}
        for name in CRITERIA:
            it = got.get(name, {"name":name, "score":0, "comment":""})
            sc = int(it.get("score",0)); sc=max(0,min(10, sc))
            fixed.append({"name":name, "score":sc, "comment":str(it.get("comment","")).strip()})
        total = sum(x["score"] for x in fixed)  # 0~100
        data = {
            "overall": int(total),
            "criteria": fixed,
            "strengths": [s for s in data.get("strengths", []) if str(s).strip()][:5],
            "risks": [s for s in data.get("risks", []) if str(s).strip()][:5],
            "improvements": [s for s in data.get("improvements", []) if str(s).strip()][:5],
        }
        return data
    except Exception as e:
        return {
            "overall": 0,
            "criteria": [{"name": n, "score": 0, "comment": ""} for n in CRITERIA],
            "strengths": [], "risks": [], "improvements": [],
            "error": str(e),
        }

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
                CHAT_MODEL
            )
        st.session_state.last_result = res
        # 기록용(질문/합계/개별 기준 점수)
        crit_map = {c["name"]: c["score"] for c in res.get("criteria", [])}
        st.session_state.records.append({
            "질문": st.session_state.current_question,
            "합계": res.get("overall", 0),
            **crit_map,
            "강점": res.get("strengths", []),
            "감점": res.get("risks", []),
            "개선": res.get("improvements", []),
        })
        st.success("채점/코칭 완료!")

# ----------------------------- 7) 팔로업 질문 & 답변 & 피드백 (수정본 제거) -----------------------------
st.header("7) 팔로업 질문 & 답변")
# 메인 피드백 존재 시, 팔로업 제안
if st.session_state.last_result and not st.session_state.followups:
    try:
        ctx = json.dumps(st.session_state.clean_struct or {}, ensure_ascii=False)
        msg = {
            "role":"user",
            "content":(
                f"[회사/직무/요건]\n{ctx}\n\n"
                f"[기존 질문]\n{st.session_state.current_question}\n\n"
                f"[기존 답변]\n{st.session_state.answer_text}\n\n"
                "면접관 관점에서 팔로업 질문 3개를 한 줄씩 한국어로 제안해줘. "
                "기존 질문과 중복되지 않게, 지표/리스크/의사결정 근거를 섞어줘."
            )
        }
        r = client.chat.completions.create(model=CHAT_MODEL, temperature=0.7,
                                           messages=[{"role":"system","content":"팔로업 질문 생성기"}, msg])
        followups = [re.sub(r'^\s*\d+[\).\s-]*','', l).strip()
                     for l in r.choices[0].message.content.splitlines() if l.strip()]
        st.session_state.followups = followups[:3]
    except Exception:
        st.session_state.followups = []

if st.session_state.followups:
    st.markdown("**팔로업 질문 제안**")
    for i, f in enumerate(st.session_state.followups, 1):
        st.markdown(f"- ({i}) {f}")
    st.selectbox("채점 받을 팔로업 질문 선택", st.session_state.followups, index=0, key="selected_followup")
    st.text_area("팔로업 질문에 대한 나의 답변", height=160, key="followup_answer")

    if st.button("팔로업 채점 & 피드백", type="secondary"):
        fu_q = st.session_state.get("selected_followup", "")
        fu_ans = st.session_state.get("followup_answer", "")
        if not fu_q:
            st.warning("팔로업 질문을 선택하세요.")
        elif not fu_ans.strip():
            st.warning("팔로업 답변을 작성하세요.")
        else:
            with st.spinner("팔로업 채점 중..."):
                res_fu = llm_score_and_coach_strict(
                    st.session_state.clean_struct, fu_q, fu_ans, CHAT_MODEL
                )
            st.session_state.last_followup_result = res_fu
            crit_map = {c["name"]: c["score"] for c in res_fu.get("criteria", [])}
            st.session_state.records.append({
                "질문": f"(팔로업) {fu_q}",
                "합계": res_fu.get("overall", 0),
                **crit_map,
                "강점": res_fu.get("strengths", []),
                "감점": res_fu.get("risks", []),
                "개선": res_fu.get("improvements", []),
            })
            st.success("팔로업 채점 완료!")

st.divider()

# ----------------------------- 8) 피드백 & 시각화 -----------------------------
st.header("8) 피드백")
last = st.session_state.last_result
if last:
    c1, c2 = st.columns([1,3])
    with c1:
        st.metric("합계(/100)", last.get("overall", 0))
    with c2:
        st.markdown("**기준별 점수 & 코멘트**")
        for it in last.get("criteria", []):
            st.markdown(f"- **{it['name']}**: {it['score']}/10 — {it.get('comment','')}")
        if last.get("strengths"):
            st.markdown("**강점**")
            for s in last["strengths"]: st.markdown(f"- {s}")
        if last.get("risks"):
            st.markdown("**감점 요인/리스크**")
            for r in last["risks"]: st.markdown(f"- {r}")
        if last.get("improvements"):
            st.markdown("**개선 포인트**")
            for im in last["improvements"]: st.markdown(f"- {im}")
else:
    st.info("아직 메인 채점 결과가 없습니다.")

# -------- 레이더 차트: 개별 답변 선택 시 각각 표시 / 미선택 시 평균만 --------
st.subheader("역량 레이더")
def records_to_df(records: List[Dict]) -> pd.DataFrame:
    rows=[]
    for rec in records:
        row = {"질문": rec.get("질문",""), "합계": rec.get("합계",0)}
        for c in CRITERIA:
            row[c] = rec.get(c, None)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["질문","합계"]+CRITERIA)

df = records_to_df(st.session_state.records)
if not df.empty:
    # 멀티 선택: 선택 시 해당 답변의 궤적 표시, 미선택 시 평균만 표시
    options = df["질문"].tolist()
    picked = st.multiselect("개별 답변 선택(미선택 시 평균만 표시)", options, default=[])
    # 평균
    avg = df[CRITERIA].mean(numeric_only=True)
    traces = []
    if PLOTLY_OK:
        fig = go.Figure()
        if not picked:
            r = avg.values.tolist()
            fig.add_trace(go.Scatterpolar(
                r=r + [r[0]], theta=CRITERIA + [CRITERIA[0]],
                fill='toself', name="평균"
            ))
        else:
            # 선택된 항목 각각 추가
            for label in picked:
                row = df[df["질문"]==label].iloc[0]
                r = [row[c] if pd.notnull(row[c]) else 0 for c in CRITERIA]
                fig.add_trace(go.Scatterpolar(
                    r=r + [r[0]], theta=CRITERIA + [CRITERIA[0]],
                    fill='toself', name=label
                ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])), showlegend=True, height=460)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Plotly 설치가 안 되어 있으면 막대 평균만
        st.bar_chart(avg)
    # 점수표 (한글 컬럼)
    st.dataframe(df, use_container_width=True)
else:
    st.caption("아직 평가 기록이 없습니다.")

st.divider()

# ----------------------------- 세션 리포트 (CSV) -----------------------------
st.subheader("세션 리포트 (CSV)")
def build_report(records: List[Dict]) -> pd.DataFrame:
    # 이미 레코드가 한글 컬럼이므로 그대로 CSV
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["질문","합계"]+CRITERIA+["강점","감점","개선"])

rep = build_report(st.session_state.records)
st.download_button("CSV 다운로드", data=rep.to_csv(index=False).encode("utf-8-sig"),
                   file_name="interview_session_report.csv", mime="text/csv")
