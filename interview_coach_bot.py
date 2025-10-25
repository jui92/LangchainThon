# -*- coding: utf-8 -*-
import os, re, json, urllib.parse, random, time, io, textwrap
from typing import Optional, Tuple, Dict, List

import requests
from bs4 import BeautifulSoup
import html2text
import streamlit as st
import pandas as pd
import numpy as np

# ================== 기본 설정 ==================
st.set_page_config(page_title="회사 맞춤 면접 코치 (RAG 확장판)", page_icon="🚀", layout="wide")
st.title("회사 맞춤 면접 코치 · 채용 URL → 정제 → RAG 초안 → 채점/코칭 → 레이더/세션")

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
    CHAT_MODEL = st.selectbox("대화/채점 모델", ["gpt-4o-mini","gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small","text-embedding-3-large"], index=0)

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

# ================== RAG: 이력서/프로젝트 업로드 ==================
try:
    import pypdf
except Exception:
    pypdf = None

def read_file_text(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith((".txt",".md")):
        for enc in ("utf-8","cp949","euc-kr"):
            try: return data.decode(enc)
            except Exception: continue
        return data.decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        if pypdf is None:
            return ""
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
        except Exception:
            return ""
    return ""

def chunk(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    t = re.sub(r"\s+"," ", text).strip()
    if not t: return []
    out, start = [], 0
    while start < len(t):
        end = min(len(t), start+size)
        out.append(t[start:end])
        if end == len(t): break
        start = max(0, end-overlap)
    return out

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
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
    store = st.session_state.get("rag_store", {})
    chs, embs = store.get("chunks", []), store.get("embeds")
    if not chs or embs is None:
        return []
    qv = embed_texts([query])
    scores, idxs = cosine_topk(embs, qv, k=k)
    return [(float(s), chs[int(i)]) for s, i in zip(scores, idxs)]

# ================== 질문/초안/채점 프롬프트 ==================
PROMPT_SYSTEM_Q = (
    "너는 채용담당자다. 회사/직무 맥락과 채용요건을 반영해 면접 질문을 한국어로 생성한다. "
    "질문은 서로 형태·관점·키워드가 겹치지 않게 다양화하고, 수치/지표/기간/규모/리스크 등도 섞어라."
)

PROMPT_SYSTEM_DRAFT = (
    "너는 면접 답변 코치다. 회사/직무/채용요건과 지원자의 이력서/프로젝트 요약을 결합해 "
    "질문에 대한 답변 **초안**을 STAR(상황-과제-행동-성과)로 8~12문장, 한국어로 작성한다. "
    "가능하면 구체적인 지표/수치/기간/임팩트를 포함하라."
)

PROMPT_SYSTEM_SCORE = (
    "너는 톱티어 면접 코치다. 아래 형식의 JSON만 출력하라. "
    "각 기준은 0~20 정수, 총점은 기준 합계(최대 100)와 반드시 일치해야 한다. "
    "각 기준에 대해 짧은 코멘트(강점/감점요인/개선포인트 포함)를 제공하라."
)

CRITERIA = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def llm_generate_questions(clean: Dict, q_type: str, level: str, model: str, num: int = 8, seed: int = 0) -> List[str]:
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {
        "role": "user",
        "content": (
            f"[회사/직무/요건]\n{ctx}\n\n"
            f"[요청]\n- 질문 유형: {q_type}\n- 난이도/연차: {level}\n"
            f"- 총 {num}개, 한 줄씩\n- 중복/유사도 최소화\n- 랜덤시드: {seed}"
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
        if "q_hist" in st.session_state:
            hist = st.session_state.q_hist[-10:]
            def sim(a,b):
                a_set=set(a.lower().split()); b_set=set(b.lower().split())
                inter=len(a_set&b_set); denom=max(1,len(a_set|b_set))
                return inter/denom
            uniq=[]
            for q in lines:
                if all(sim(q,h)<0.4 for h in hist):
                    uniq.append(q)
            if uniq: lines = uniq
        return lines[:num]
    except Exception:
        return []

def llm_draft_answer(clean: Dict, question: str, resume_snips: List[str], model: str) -> str:
    ctx = json.dumps(clean, ensure_ascii=False)
    resume_text = "\n".join([f"- {s[:400]}" for s in resume_snips])[:2000]
    user_msg = {
        "role": "user",
        "content": (
            f"[회사/직무/채용요건]\n{ctx}\n\n"
            f"[지원자 이력서/프로젝트 요약]\n{resume_text}\n\n"
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

def llm_score_and_coach(clean: Dict, question: str, answer: str, resume_snips: List[str], model: str) -> Dict:
    ctx = json.dumps(clean, ensure_ascii=False)
    resume_text = "\n".join([f"- {s[:400]}" for s in resume_snips])[:1600]
    user_msg = {
        "role":"user",
        "content": (
            f"[회사/직무/채용요건]\n{ctx}\n\n"
            f"[지원자 이력서/프로젝트 요약]\n{resume_text}\n\n"
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
            messages=[{"role":"system","content":PROMPT_SYSTEM_SCORE}, user_msg]
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
if "q_hist" not in st.session_state:
    st.session_state.q_hist = []
if "records" not in st.session_state:
    st.session_state.records = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "answer_text" not in st.session_state:
    st.session_state.answer_text = ""
if "rag_store" not in st.session_state:
    st.session_state.rag_store = {"chunks": [], "embeds": None}

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

# ================== 3) 내 이력서/프로젝트 업로드(RAG) ==================
st.header("3) 내 이력서/프로젝트 업로드 (RAG)")
docs = st.file_uploader("이력서/프로젝트 설명 파일 업로드 (PDF/TXT/MD, 여러 개 가능)", type=["pdf","txt","md"], accept_multiple_files=True)
rag_cols = st.columns(3)
with rag_cols[0]:
    chunk_size = st.number_input("청크 길이", value=900, min_value=300, max_value=2000, step=100)
with rag_cols[1]:
    chunk_overlap = st.number_input("오버랩", value=150, min_value=0, max_value=500, step=10)
with rag_cols[2]:
    top_k_rag = st.number_input("검색 상위 K", value=4, min_value=1, max_value=10, step=1)

if st.button("RAG 인덱싱", type="secondary"):
    if not docs:
        st.warning("파일을 업로드하세요.")
    else:
        chunks=[]
        for up in docs:
            t = read_file_text(up)
            if t:
                chunks += chunk(t, chunk_size, chunk_overlap)
        if not chunks:
            st.error("텍스트를 추출하지 못했습니다.")
        else:
            with st.spinner("임베딩 생성 중..."):
                vecs = embed_texts(chunks)
            st.session_state.rag_store["chunks"] = chunks
            st.session_state.rag_store["embeds"] = vecs
            st.success(f"인덱싱 완료 (청크 {len(chunks)}개)")

# ================== 4) 질문 생성 & 초안 ==================
st.header("4) 질문 생성 & 답변 초안")
q_type = st.selectbox("질문 유형", ["행동(STAR)","기술 심층","핵심가치 적합성","역질문"], index=0)
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)
seed   = st.number_input("랜덤시드", value=int(time.time())%1_000_000, step=1)
num    = st.slider("질문 개수", 4, 10, 8, 1)

cqa = st.columns(2)
with cqa[0]:
    if st.button("새 질문 받기", type="primary"):
        if not st.session_state.clean_struct:
            st.warning("먼저 URL을 정제하세요.")
        else:
            qs = llm_generate_questions(st.session_state.clean_struct, q_type, level, CHAT_MODEL, num=num, seed=int(seed))
            if qs:
                st.session_state.q_hist.extend(qs)
                st.session_state.current_question = random.choice(qs)
                st.session_state.answer_text = ""  # 이전 답변 초기화
                st.success("질문 생성 완료!")
            else:
                st.error("질문 생성 실패")
with cqa[1]:
    if st.button("RAG로 답변 초안 생성", type="secondary"):
        if not st.session_state.current_question:
            st.warning("먼저 질문을 생성하세요.")
        else:
            snips = []
            if st.session_state.rag_store.get("embeds") is not None:
                hits = retrieve_resume_chunks(st.session_state.current_question, k=top_k_rag)
                snips = [t for _, t in hits]
            draft = llm_draft_answer(st.session_state.clean_struct, st.session_state.current_question, snips, CHAT_MODEL)
            if draft:
                st.session_state.answer_text = draft
                st.success("초안 생성 완료!")
            else:
                st.error("초안 생성 실패(컨텍스트 부족 가능)")

st.text_area("질문", value=st.session_state.current_question, height=100)
ans = st.text_area("나의 답변 (초안을 편집해 완성하세요)", height=200, key="answer_text")

# ================== 5) 채점 & 코칭 (+ 팔로업 질문) ==================
st.header("5) 채점 & 코칭")
if st.button("채점 & 코칭 실행", type="primary"):
    if not st.session_state.current_question:
        st.warning("먼저 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        snips=[]
        if st.session_state.rag_store.get("embeds") is not None:
            hits = retrieve_resume_chunks(
                st.session_state.current_question + "\n" + st.session_state.answer_text[:800],
                k=top_k_rag
            )
            snips = [t for _, t in hits]
        with st.spinner("채점/코칭 중..."):
            res = llm_score_and_coach(st.session_state.clean_struct, st.session_state.current_question, st.session_state.answer_text, snips, CHAT_MODEL)
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

        # 팔로업 질문 3개 추가 제안
        try:
            ctx = json.dumps(st.session_state.clean_struct, ensure_ascii=False)
            msg = {
                "role":"user",
                "content":(
                    f"[회사/직무/요건]\n{ctx}\n\n"
                    f"[지원자 답변]\n{st.session_state.answer_text}\n\n"
                    "면접관 관점에서 팔로업 질문 3개를 한 줄씩 한국어로 제안해줘. "
                    "기존 질문과 중복되지 않게, 지표/리스크/트레이드오프를 섞어줘."
                )
            }
            r = client.chat.completions.create(model=CHAT_MODEL, temperature=0.7, messages=[{"role":"system","content":"면접 팔로업 생성기"}, msg])
            followups = [re.sub(r'^\s*\d+[\).\s-]*','', l).strip() for l in r.choices[0].message.content.splitlines() if l.strip()]
            st.markdown("**팔로업 질문 제안**")
            for f in followups[:3]:
                st.markdown(f"- {f}")
        except Exception:
            pass

# ================== 6) 피드백 결과 ==================
st.header("6) 피드백 결과")
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

# ================== 7) 역량 레이더 (누적+평균) ==================
st.header("7) 역량 레이더 (세션 누적)")
def build_comp_table(records):
    rows=[]
    for idx, r in enumerate(records, 1):
        crit = r.get("criteria", [])
        row={"#": idx, "question": r.get("question",""), "overall": r.get("overall",0)}
        cm = {c["name"]: c["score"] for c in crit if "name" in c}
        for k in CRITERIA:
            row[k] = cm.get(k, 0)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["#","question","overall"]+CRITERIA)

df = build_comp_table(st.session_state.records)
if not df.empty:
    avg = [float(df[k].mean()) for k in CRITERIA]
    cum = [int(df[k].sum()) for k in CRITERIA]
    try:
        import plotly.graph_objects as go
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=avg + [avg[0]], theta=CRITERIA + [CRITERIA[0]],
            fill='toself', name='평균(0~20)'
        ))
        radar.add_trace(go.Scatterpolar(
            r=cum + [cum[0]], theta=CRITERIA + [CRITERIA[0]],
            fill='toself', name='누적(합계)'
        ))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=420)
        st.plotly_chart(radar, use_container_width=True)
    except Exception:
        st.bar_chart(pd.DataFrame({"평균":avg,"누적":cum}, index=CRITERIA))
    st.markdown("**세션 표(질문별 기준 점수)**")
    st.dataframe(df, use_container_width=True)
else:
    st.caption("아직 누적 데이터가 없습니다. 질문 생성→답변→채점을 진행하세요.")

st.divider()

# ================== 8) 세션 저장/불러오기 ==================
st.header("8) 세션 저장/불러오기")
col_s = st.columns(2)
with col_s[0]:
    export = {
        "clean_struct": st.session_state.clean_struct,
        "q_hist": st.session_state.q_hist,
        "records": st.session_state.records,
    }
    st.download_button("세션 저장(JSON)", data=json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="interview_session.json", mime="application/json")
with col_s[1]:
    up = st.file_uploader("세션 불러오기(JSON)", type=["json"], accept_multiple_files=False, key="sess_up")
    if st.button("불러오기 실행", type="secondary"):
        if up is None:
            st.warning("JSON 파일을 올려주세요.")
        else:
            try:
                data = json.loads(up.read().decode("utf-8"))
                st.session_state.clean_struct = data.get("clean_struct")
                st.session_state.q_hist = data.get("q_hist", [])
                st.session_state.records = data.get("records", [])
                st.success("세션 불러오기 완료!")
            except Exception as e:
                st.error(f"불러오기 실패: {e}")

st.caption("총점은 기준(5×20) 합계와 항상 일치하도록 강제합니다. ‘새 질문 받기’ 클릭 시 답변란은 초기화됩니다.")
