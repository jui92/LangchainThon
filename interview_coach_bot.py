# -*- coding: utf-8 -*-
import os, re, json, urllib.parse, random, time, io
from typing import Optional, Tuple, Dict, List

import requests
from bs4 import BeautifulSoup
import html2text
import streamlit as st
import pandas as pd
import numpy as np

# ================== 기본 설정 ==================
st.set_page_config(page_title="회사 맞춤 면접 코치 (Step2: 질문/RAG/팔로업)", page_icon="🎯", layout="wide")
st.title("회사 맞춤 면접 코치 · URL 정제 → 자소서 → (Step2) 질문/RAG/팔로업/엄격 채점")

# ================== OpenAI ==================
try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop()

API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not API_KEY:
    API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password")
if not API_KEY:
    st.stop()
client = OpenAI(api_key=API_KEY)

with st.sidebar:
    st.subheader("모델 설정")
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0)
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

# ================== 메타/섹션 보조 추출 ==================
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

# ================== LLM 정제 (채용 공고 → 구조 JSON) ==================
PROMPT_SYSTEM_STRUCT = (
    "너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
    "입력 텍스트는 포털 광고 문구, UI잔재, 복수 직무가 섞여 있을 수 있다. "
    "한국어로 간결하고 중복없이 정제하라."
)

def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
    ctx = raw_text.strip()
    if len(ctx) > 9000:
        ctx = ctx[:9000]

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
            "}"
        ),
    }

    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg],
        )
        data = json.loads(resp.choices[0].message.content)
        # 후처리
        for k in ["responsibilities","qualifications","preferences"]:
            if not isinstance(data.get(k, []), list):
                data[k] = []
            clean = []
            seen = set()
            for it in data[k]:
                t = re.sub(r"\s+"," ", str(it)).strip(" -•·").strip()
                if t and t not in seen:
                    seen.add(t); clean.append(t)
            data[k] = clean[:12]
        for k in ["company_name","company_intro","job_title"]:
            if k in data and isinstance(data[k], str):
                data[k] = re.sub(r"\s+"," ", data[k]).strip()
        return data
    except Exception as e:
        return {
            "company_name": meta_hint.get("company_name",""),
            "company_intro": meta_hint.get("company_intro","원문이 정제되지 않았습니다."),
            "job_title": meta_hint.get("job_title",""),
            "responsibilities": [],
            "qualifications": [],
            "preferences": [],
            "error": str(e),
        }

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

# ================== 간단 청크/임베딩(내부: 숨김) ==================
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

# ================== 질문/초안/채점/팔로업 프롬프트 ==================
PROMPT_SYSTEM_Q = (
    "너는 채용담당자다. 회사/직무 맥락과 채용요건, 그리고 지원자의 이력서 요약을 함께 고려해 "
    "면접 질문을 한국어로 생성한다. 질문은 서로 형태·관점·키워드가 겹치지 않게 다양화하고, "
    "수치/지표/기간/규모/리스크/트레이드오프 등도 섞어라."
)

PROMPT_SYSTEM_DRAFT = (
    "너는 면접 답변 코치다. 회사/직무/채용요건과 지원자의 이력서 요약을 결합해 "
    "질문에 대한 답변 **초안**을 STAR(상황-과제-행동-성과)로 8~12문장, 한국어로 작성한다. "
    "가능하면 구체적인 지표/수치/기간/임팩트를 포함하라."
)

PROMPT_SYSTEM_SCORE_STRICT = (
    "너는 매우 엄격한 톱티어 면접 코치다. 아래 형식의 JSON만 출력하라. "
    "각 기준은 0~20 정수이며, 총점은 기준 합계(최대 100)와 반드시 일치해야 한다. "
    "과장/모호함/근거 부재/숫자 없는 주장/책임 회피/모호한 주어 사용 등을 강하게 감점하라. "
    "각 기준에 대해 짧지만 구체적 코멘트(강점/감점요인/개선포인트)를 제공하라."
)

CRITERIA = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def llm_generate_questions_with_resume(clean: Dict, level: str, model: str, num: int = 8) -> List[str]:
    # RAG에서 대표 스니펫 3~4개만 요약해 프롬프트에 주입
    resume_snips = []
    hits = retrieve_resume_chunks("핵심 프로젝트와 기술 스택 요약", k=4)
    resume_snips = [t for _, t in hits]
    resume_context = "\n".join([f"- {s[:350]}" for s in resume_snips])[:1200]

    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {
        "role": "user",
        "content": (
            f"[회사/직무/요건]\n{ctx}\n\n"
            f"[지원자 이력서 요약(발췌)]\n{resume_context}\n\n"
            f"[요청]\n- 난이도/연차: {level}\n"
            f"- 총 {num}개, 한 줄씩, 중복/유사도 최소화\n"
            f"- 회사 요건과 이력서의 교집합/공백영역을 모두 겨냥해 질문 생성"
        ),
    }
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.9,
            messages=[{"role":"system","content":PROMPT_SYSTEM_Q}, user_msg],
        )
        txt = resp.choices[0].message.content.strip()
        lines = [re.sub(r'^\s*\d+[\).\s-]*','', l).strip() for l in txt.splitlines() if l.strip()]
        lines = [l for l in lines if len(l.split()) > 2][:num]
        # 최근 질문과의 중복 간단 필터
        if "q_hist" in st.session_state:
            hist = st.session_state.q_hist[-12:]
            def sim(a,b):
                a_set=set(a.lower().split()); b_set=set(b.lower().split())
                inter=len(a_set&b_set); denom=max(1,len(a_set|b_set))
                return inter/denom
            uniq=[]
            for q in lines:
                if all(sim(q,h)<0.35 for h in hist):
                    uniq.append(q)
            if uniq: lines = uniq
        return lines[:num]
    except Exception:
        return []

def llm_draft_answer(clean: Dict, question: str, model: str) -> str:
    # RAG 근거 포함 초안
    hits = retrieve_resume_chunks(question, k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]

    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {
        "role": "user",
        "content": (
            f"[회사/직무/채용요건]\n{ctx}\n\n"
            f"[지원자 이력서 발췌]\n{resume_text}\n\n"
            f"[면접 질문]\n{question}\n\n"
            "위 정보를 근거로 STAR 기반 한국어 답변 **초안**을 작성해줘."
        )
    }
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.5,
            messages=[{"role":"system","content":PROMPT_SYSTEM_DRAFT}, user_msg],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

def llm_score_and_coach_strict(clean: Dict, question: str, answer: str, model: str) -> Dict:
    # 더 깐깐한 채점: 근거/지표 미제시 강력 감점
    hits = retrieve_resume_chunks(question + "\n" + answer[:800], k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]

    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {
        "role":"user",
        "content": (
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
            "\"strengths\": [\"...\", \"...\"],"
            "\"risks\": [\"...\", \"...\"],"
            "\"improvements\": [\"...\", \"...\", \"...\"],"
            "\"revised_answer\": \"STAR 구조로 간결히\""
            "}"
        )
    }
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_SCORE_STRICT}, user_msg]
        )
        data = json.loads(resp.choices[0].message.content)
        # 정합화
        crit = data.get("criteria", [])
        fixed=[]
        for name in CRITERIA:
            found=None
            for it in crit:
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
        return {
            "overall_score": 0,
            "criteria": [{"name": n, "score": 0, "comment": ""} for n in CRITERIA],
            "strengths": [],
            "risks": [],
            "improvements": [],
            "revised_answer": "",
            "error": str(e),
        }

# ================== 세션 상태 ==================
if "clean_struct" not in st.session_state:
    st.session_state.clean_struct = None
if "resume_raw" not in st.session_state:
    st.session_state.resume_raw = ""
if "resume_chunks" not in st.session_state:
    st.session_state.resume_chunks = []
if "resume_embeds" not in st.session_state:
    st.session_state.resume_embeds = None
if "q_hist" not in st.session_state:
    st.session_state.q_hist = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "answer_text" not in st.session_state:
    st.session_state.answer_text = ""
if "records" not in st.session_state:
    st.session_state.records = []
if "followups" not in st.session_state:
    st.session_state.followups = []
if "selected_followup" not in st.session_state:
    st.session_state.selected_followup = ""

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
        prefs = clean.get("preferences", [])
        if prefs:
            for b in prefs: st.markdown(f"- {b}")
        else:
            st.caption("우대 사항이 명시되지 않았습니다.")
else:
    st.info("먼저 URL을 정제해 주세요.")

st.divider()

# ================== 3) 내 이력서/프로젝트 업로드 ==================
st.header("3) 내 이력서/프로젝트 업로드 (PDF/TXT/MD/DOCX)")
uploads = st.file_uploader(
    "여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True
)
# 내부 기본 파라미터 (숨김)
_RESUME_CHUNK = 600
_RESUME_OVLP  = 120
_TOPK_RAG     = 4

cols_idx = st.columns(2)
with cols_idx[0]:
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
with cols_idx[1]:
    st.caption("※ 청크/오버랩/Top-K 파라미터는 내부 기본값으로 자동 처리합니다.")

st.divider()

# ================== 4) 자소서 생성 (Step1에서 추가된 기능 유지) ==================
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
        st.download_button("자소서 TXT 다운로드", data=cover.encode("utf-8"),
                           file_name="cover_letter.txt", mime="text/plain")

st.divider()

# ================== 5) 질문 생성 & 답변 초안 (RAG 결합, 내부 시드/개수 숨김) ==================
st.header("5) 질문 생성 & 답변 초안 (RAG 결합)")
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)

# 내부 파라미터 (숨김)
_Q_NUM  = 8
_SEED   = int(time.time()*1000000) % 2_147_483_647

cols_q = st.columns(2)
with cols_q[0]:
    if st.button("새 질문 받기", type="primary"):
        if not st.session_state.clean_struct:
            st.warning("먼저 회사 URL을 정제하세요.")
        else:
            qs = llm_generate_questions_with_resume(st.session_state.clean_struct, level, CHAT_MODEL, num=_Q_NUM)
            if qs:
                st.session_state.q_hist.extend(qs)
                # 다양성 위해 랜덤 1개 선택
                st.session_state.current_question = random.choice(qs)
                st.session_state.answer_text = ""  # 이전 답변 초기화
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
ans = st.text_area("나의 답변 (초안을 편집해 완성하세요)", height=200, key="answer_text")

# ================== 6) 채점 & 코칭 (엄격 모드) + 팔로업 준비 ==================
st.header("6) 채점 & 코칭 (엄격 모드)")
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

        # 팔로업 질문 3개 제안 저장
        try:
            ctx = json.dumps(st.session_state.clean_struct, ensure_ascii=False)
            msg = {
                "role":"user",
                "content":(
                    f"[회사/직무/요건]\n{ctx}\n\n"
                    f"[지원자 답변]\n{st.session_state.answer_text}\n\n"
                    "면접관 관점에서 팔로업 질문 3개를 한 줄씩 한국어로 제안해줘. "
                    "기존 질문과 중복되지 않게, 지표/리스크/트레이드오프/의사결정 근거를 섞어줘."
                )
            }
            r = client.chat.completions.create(model=CHAT_MODEL, temperature=0.7,
                                               messages=[{"role":"system","content":"면접 팔로업 생성기"}, msg])
            followups = [re.sub(r'^\s*\d+[\).\s-]*','', l).strip()
                         for l in r.choices[0].message.content.splitlines() if l.strip()]
            st.session_state.followups = followups[:3]
        except Exception:
            st.session_state.followups = []

# ================== 7) 피드백 결과 ==================
st.header("7) 피드백 결과")
if st.session_state.records:
    last = st.session_state.records[-1]
    left, right = st.columns([1,3])
    with left:
        st.metric("총점(/100)", last["overall"])
    with right:
        st.markdown("**기준별 점수 & 코멘트**")
        for it in last["criteria"]:
            st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
        if last["strengths"]:
            st.markdown("**강점**")
            for s in last["strengths"]: st.markdown(f"- {s}")
        if last["risks"]:
            st.markdown("**감점 요인/리스크**")
            for r in last["risks"]: st.markdown(f"- {r}")
        if last["improvements"]:
            st.markdown("**개선 포인트**")
            for im in last["improvements"]: st.markdown(f"- {im}")
        if last["revised_answer"]:
            st.markdown("**수정본 답변 (STAR)**")
            st.write(last["revised_answer"])
else:
    st.info("아직 채점 결과가 없습니다.")

st.divider()

# ================== 8) 팔로업 질문 ▶ 답변 ▶ 피드백 ==================
st.header("8) 팔로업 질문 ▶ 답변 ▶ 피드백")
if st.session_state.followups:
    st.markdown("**팔로업 질문 제안**")
    for i, f in enumerate(st.session_state.followups, 1):
        st.markdown(f"- ({i}) {f}")
    st.session_state.selected_followup = st.selectbox(
        "채점 받을 팔로업 질문 선택", st.session_state.followups, index=0
    )
    fu_ans = st.text_area("팔로업 질문에 대한 나의 답변", height=160, key="followup_answer")

    if st.button("팔로업 채점 & 피드백", type="secondary"):
        if not st.session_state.selected_followup:
            st.warning("팔로업 질문을 선택하세요.")
        elif not fu_ans.strip():
            st.warning("팔로업 답변을 작성하세요.")
        else:
            with st.spinner("팔로업 채점 중..."):
                res_fu = llm_score_and_coach_strict(
                    st.session_state.clean_struct,
                    st.session_state.selected_followup,
                    fu_ans, CHAT_MODEL
                )
            st.markdown("**팔로업 결과**")
            st.metric("총점(/100)", res_fu.get("overall_score", 0))
            for it in res_fu.get("criteria", []):
                st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
            if res_fu.get("revised_answer",""):
                st.markdown("**팔로업 수정본 (STAR)**")
                st.write(res_fu["revised_answer"])
else:
    st.caption("메인 질문을 채점하면 팔로업 질문이 자동으로 제안됩니다.")
