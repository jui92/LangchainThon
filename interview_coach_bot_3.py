# -*- coding: utf-8 -*-
"""
Company-tailored helper: 채용 URL → 정제 → 회사 요약 / 요건 → 이력서 업로드 → 자소서 생성
- 우대 사항 키: preferred_qualifications (aliases 흡수)
- 규칙 기반 보강 파서로 우대 항목 확실히 채움
"""

import os, re, io, json, urllib.parse
from typing import Optional, Tuple, Dict, List

import requests
from bs4 import BeautifulSoup
import html2text
import streamlit as st
import numpy as np
import pandas as pd

# ================== Streamlit 기본 ==================
st.set_page_config(page_title="Job Helper (우대사항 보강·별칭 통합)", page_icon="🧭", layout="wide")
st.title("Job Helper — 채용 URL 정제 · 회사 요약 · 이력서 업로드 · 자소서 생성")

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
    st.subheader("모델 설정")
    CHAT_MODEL  = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델(내부용)", ["text-embedding-3-small","text-embedding-3-large"], index=0)

# ================== HTTP 유틸 ==================
def normalize_url(u: str) -> Optional[str]:
    if not u: return None
    u = u.strip()
    if not re.match(r"^https?://", u): u = "https://" + u
    parts = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

def http_get(url: str, timeout: int = 12) -> Optional[requests.Response]:
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

# ================== 원문 수집 (Jina → Web → BS4) ==================
def fetch_jina_text(url: str, timeout: int = 15) -> str:
    """Jina Reader 프록시(정적 텍스트 스냅샷) 시도"""
    try:
        parts = urllib.parse.urlsplit(url)
        prox = f"https://r.jina.ai/http://{parts.netloc}{parts.path}"
        if parts.query: prox += f"?{parts.query}"
        r = http_get(prox, timeout=timeout)
        return r.text.strip() if r else ""
    except Exception:
        return ""

def html_to_text(html_str: str) -> str:
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
    r = http_get(url, timeout=12)
    if not r: return "", None
    soup = BeautifulSoup(r.text, "lxml")
    # 큰 블록 추출
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

# ================== 메타 추출(회사명/소개/직무명 힌트) ==================
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

# ================== 규칙 기반 보강 파서(우대사항 보장) ==================
def rule_based_sections(raw_text: str) -> dict:
    """
    원문에서 '주요 업무 / 자격 요건 / 우대 사항' 섹션을 최대한 복구.
    - 섹션 헤더 키워드 매칭
    - 불릿/특수문자 제거
    - 우대 키워드가 들어간 줄은 우대 사항으로 이동
    """
    txt = re.sub(r"\r", "", raw_text or "").strip()
    lines = [re.sub(r"\s+", " ", l).strip(" -•·▶▪️") for l in txt.split("\n")]
    lines = [l for l in lines if l]

    hdr_resp = re.compile(r"(주요\s*업무|담당\s*업무|Role|Responsibilities?)", re.I)
    hdr_qual = re.compile(r"(자격\s*요건|지원\s*자격|Requirements?|Qualifications?)", re.I)
    hdr_pref = re.compile(r"(우대\s*사항|우대|Preferred|Nice\s*to\s*have|Plus)", re.I)
    kw_pref  = re.compile(r"(우대|preferred|nice to have|plus|가산점|있으면\s*좋음)", re.I)

    out = {"responsibilities": [], "qualifications": [], "preferred_qualifications": []}
    bucket = None

    def push(line, key):
        t = re.sub(r"\s+", " ", line).strip(" -•·▶▪️").strip()
        if t and len(t) > 1 and t not in out[key]:
            out[key].append(t[:180])

    for l in lines:
        if hdr_resp.search(l):
            bucket = "responsibilities"; continue
        if hdr_qual.search(l):
            bucket = "qualifications"; continue
        if hdr_pref.search(l):
            bucket = "preferred_qualifications"; continue

        if bucket is None:
            if kw_pref.search(l):
                push(l, "preferred_qualifications")
            elif any(k in l.lower() for k in ["java","python","spark","airflow","kafka","ml","sql","aws","k8s","docker"]):
                push(l, "responsibilities")
            continue
        else:
            if bucket == "qualifications" and kw_pref.search(l):
                push(l, "preferred_qualifications")
            else:
                push(l, bucket)

    for k in out:
        seen=set(); cleaned=[]
        for s in out[k]:
            if s and s not in seen:
                seen.add(s); cleaned.append(s)
        out[k] = cleaned[:12]
    return out

# ================== 우대 별칭 흡수(표준키 preferred_qualifications) ==================
PREF_ALIASES = [
    "preferences", "preferred", "preferred_qualifications", "preferred_requirements",
    "preferred_skills", "nice_to_have", "nice_to_haves",
    "desirables", "desirable_qualifications", "plus", "bonus_qualifications", "good_to_have"
]

def unify_preferred_field(data: dict) -> dict:
    bucket = []
    for k in PREF_ALIASES:
        v = data.get(k, [])
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            bucket.extend(v)
    # clean & dedup
    seen, cleaned = set(), []
    for x in bucket:
        t = re.sub(r"\s+"," ", str(x)).strip(" -•·▶▪️").strip()
        if t and t not in seen:
            seen.add(t); cleaned.append(t[:180])
    data["preferred_qualifications"] = cleaned[:12]
    # 별칭키 제거(선택)
    for k in PREF_ALIASES:
        if k != "preferred_qualifications" and k in data:
            data.pop(k, None)
    return data

# ================== LLM 구조화(회사/직무/요건) ==================
PROMPT_SYSTEM_STRUCT = (
    "너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
    "입력 텍스트는 포털 광고 문구, UI잔재, 복수 직무가 섞여 있을 수 있다. "
    "한국어로 간결하고 중복없이 정제하라."
)

def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
    ctx = (raw_text or "").strip()
    if len(ctx) > 12000:
        ctx = ctx[:12000]

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
            "\"preferred_qualifications\": [str]"
            "}\n"
            "- 'preferred_qualifications'에는 원문 중 '우대/선호/Preferred/Nice to have/Plus' 등의 항목을 그대로 담을 것.\n"
            "- 불릿/마커/이모지 제거, 문장 간결화, 중복 제거."
        ),
    }

    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg],
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        data = {
            "company_name": meta_hint.get("company_name",""),
            "company_intro": meta_hint.get("company_intro","원문이 정제되지 않았습니다."),
            "job_title": meta_hint.get("job_title",""),
            "responsibilities": [],
            "qualifications": [],
            "preferred_qualifications": [],
            "error": str(e),
        }

    # 1) 1차 클린업
    for k in ["responsibilities","qualifications","preferred_qualifications"]:
        arr = data.get(k, [])
        if not isinstance(arr, list): arr = []
        cleaned=[]; seen=set()
        for it in arr:
            t = re.sub(r"\s+"," ", str(it)).strip(" -•·▶▪️").strip()
            if t and t not in seen:
                seen.add(t); cleaned.append(t[:180])
        data[k] = cleaned[:12]

    for k in ["company_name","company_intro","job_title"]:
        if k in data and isinstance(data[k], str):
            data[k] = re.sub(r"\s+"," ", data[k]).strip()

    # 2) 규칙 기반 보강
    rb = rule_based_sections(ctx)

    # 우대: 비었거나 적다면 보강
    if not data.get("preferred_qualifications"):
        data["preferred_qualifications"] = rb.get("preferred_qualifications", [])[:12]

    if not data.get("responsibilities"):
        data["responsibilities"] = rb.get("responsibilities", [])[:12]
    if not data.get("qualifications"):
        data["qualifications"] = rb.get("qualifications", [])[:12]

    # 3) 별칭 통합(안전망)
    data = unify_preferred_field(data)

    return data

# ================== 파일 리더 (PDF/TXT/MD/DOCX) ==================
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
        import docx2txt, tempfile
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

# ================== 임베딩 유틸 ==================
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

# ================== 세션 상태 ==================
def _init_state():
    for k, v in {
        "clean_struct": None,
        "resume_raw": "",
        "resume_chunks": [],
        "resume_embeds": None,
    }.items():
        if k not in st.session_state: st.session_state[k] = v
_init_state()

# ================== 1) 채용 공고 URL → 정제 ==================
st.header("1) 채용 공고 URL → 정제")
url = st.text_input("채용 공고 상세 URL", placeholder="예: https://www.wanted.co.kr/wd/123456")

if st.button("원문 수집 → 정제", type="primary"):
    if not url.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner("원문 수집 중..."):
            raw, meta, soup = fetch_all_text(url.strip())
            hint = extract_company_meta(soup)
        if not raw:
            st.error("원문을 가져오지 못했습니다. (로그인/동적 렌더링/봇 차단 가능)")
        else:
            with st.spinner("LLM으로 정제 중..."):
                clean = llm_structurize(raw, hint, CHAT_MODEL)
            st.session_state.clean_struct = clean
            st.success("정제 완료!")

# ================== 2) 회사 요약 (정제 결과) ==================
st.header("2) 회사 요약 (정제 결과)")
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
        prefs = clean.get("preferred_qualifications", [])
        if prefs:
            for b in prefs: st.markdown(f"- {b}")
        else:
            st.caption("우대 사항이 명시되지 않았습니다.")
else:
    st.info("먼저 URL을 정제해 주세요.")

st.divider()

# ================== 3) 내 이력서/프로젝트 업로드 ==================
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader(
    "이력서/프로젝트 파일 업로드 (PDF/TXT/MD/DOCX, 여러 개 가능)",
    type=["pdf","txt","md","docx"], accept_multiple_files=True
)

# 내부 고정 파라미터 (UI 비노출)
_RESUME_CHUNK = 600
_RESUME_OVLP  = 120

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
            chunks = chunk(resume_text, size=_RESUME_CHUNK, overlap=_RESUME_OVLP)
            with st.spinner("이력서 벡터화 중..."):
                embeds = embed_texts(chunks, EMBED_MODEL)
            st.session_state.resume_raw = resume_text
            st.session_state.resume_chunks = chunks
            st.session_state.resume_embeds = embeds
            st.success(f"인덱싱 완료 (청크 {len(chunks)}개)")

# ================== 4) 이력서 기반 자소서 생성 ==================
st.header("4) 이력서 기반 자소서 생성")
topic = st.text_input("회사 요청 주제(선택)", placeholder="예: 직무 지원동기 / 협업 경험 / 문제해결 사례 등")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str) -> str:
    company = json.dumps(clean_struct or {}, ensure_ascii=False)
    resume_snippet = resume_text.strip()
    if len(resume_snippet) > 9000:
        resume_snippet = resume_snippet[:9000]
    system = (
        "너는 한국어 자기소개서 전문가다. 채용 공고의 회사/직무 요건과 후보자의 이력서를 참고해 "
        "회사 특화 자소서를 작성한다. 과장/허위는 금지하고, 수치/지표/기간/임팩트 중심으로 구체화한다."
    )
    if topic_hint and topic_hint.strip():
        req = f"회사 측 요청 주제는 '{topic_hint.strip()}' 이다. 이 주제를 중심으로 서술하라."
    else:
        req = "특정 주제 요청이 없으므로, 채용 공고의 요건을 중심으로 지원동기와 직무적합성을 강조하라."
    user = (
        f"[회사/직무 요약(JSON)]\n{company}\n\n"
        f"[후보자 이력서(요약 가능)]\n{resume_snippet}\n\n"
        f"[작성 지시]\n- {req}\n"
        "- 분량: 600~900자\n"
        "- 구성: 1) 지원 동기 2) 직무 관련 핵심 역량·경험 3) 성과/지표 4) 입사 후 기여 방안 5) 마무리\n"
        "- 자연스럽고 진정성 있는 1인칭 서술. 문장과 문단 가독성을 유지.\n"
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
        st.subheader("자소서 (생성 결과)")
        st.write(cover)
        st.download_button(
            "자소서 TXT 다운로드",
            data=cover.encode("utf-8"),
            file_name="cover_letter.txt",
            mime="text/plain"
        )

st.caption("※ 우대 사항 키는 내부적으로 'preferred_qualifications'로 통일되며, preferences/nice_to_have 등 별칭도 자동 흡수됩니다.")
