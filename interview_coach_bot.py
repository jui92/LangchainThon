import os, io, re, json, textwrap, time, random
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Optional deps ----------
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import requests
from bs4 import BeautifulSoup

# (선택) WebBaseLoader – 실패해도 앱이 동작하도록 try-import
try:
    from langchain_community.document_loaders import WebBaseLoader
    HAS_WEBBASE = True
except Exception:
    HAS_WEBBASE = False

# -------------------------------------------------
# Page
# -------------------------------------------------
st.set_page_config(page_title="회사 특화 취업 준비 코치", page_icon="🎯", layout="wide")

# -------------------------------------------------
# Utils
# -------------------------------------------------
def _clean(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    return t

def read_pdf_to_text(data: bytes) -> str:
    if pypdf is None:
        return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = []
        for i in range(len(reader.pages)):
            pages.append(reader.pages[i].extract_text() or "")
        return "\n".join(pages)
    except Exception:
        return ""

def read_docx_to_text(data: bytes) -> str:
    """docx2txt는 파일 경로 기반이라 임시 파일 사용"""
    if docx2txt is None:
        return ""
    try:
        tmp = "/tmp/_upload.docx"
        with open(tmp, "wb") as f:
            f.write(data)
        txt = docx2txt.process(tmp) or ""
        return txt
    except Exception:
        return ""

def read_text_upload(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith((".txt",".md",".csv",".log")):
        for enc in ("utf-8","cp949","euc-kr","utf-16"):
            try:
                return data.decode(enc)
            except Exception:
                continue
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        return read_pdf_to_text(data)
    if name.endswith(".docx"):
        return read_docx_to_text(data)
    return ""

def get_api_key() -> Optional[str]:
    k = os.getenv("OPENAI_API_KEY")
    if k:
        return k
    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        return None

def chunk_text(text: str, size: int = 600, overlap: int = 120) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text: return []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        out.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return out

# -------------------------------------------------
# Raw page text fetchers
# -------------------------------------------------
def fetch_all_text_bs4(url: str, timeout: int = 12) -> str:
    """가능한 모든 텍스트(보이는 영역 위주). 동적 영역은 한계."""
    try:
        if not url.startswith("http"):
            url = "https://" + url
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200: return ""
        soup = BeautifulSoup(r.text, "html.parser")
        # 불필요한 script/style 제거
        for tag in soup(["script","style","noscript"]):
            tag.decompose()
        # aria-hidden 제외
        for tag in soup.find_all(attrs={"aria-hidden":"true"}):
            tag.decompose()
        txt = soup.get_text("\n")
        # 너무 긴 공백 정리
        txt = re.sub(r"\n{2,}", "\n", txt)
        txt = re.sub(r"[ \t]+", " ", txt)
        return txt.strip()
    except Exception:
        return ""

def fetch_all_text_webbase(url: str, timeout: int = 15) -> str:
    """WebBaseLoader가 있으면 사용 (내부적으로 newspaper/bs4 등 사용)"""
    if not HAS_WEBBASE:
        return ""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        body = "\n".join([d.page_content for d in docs if d and getattr(d,"page_content",None)])
        return body.strip()
    except Exception:
        return ""

def fetch_jobpage_text(url: str) -> Tuple[str, Dict[str,int], str]:
    """
    원문 텍스트, 사용한 렌즈별 길이, 최종 URL 반환
    우선 bs4 -> webbase 순으로 시도(둘 다 성공하면 더 긴 텍스트 선택)
    """
    urlf = url.strip()
    lens_count = {"bs4":0, "webbase":0}
    t1 = fetch_all_text_bs4(urlf)
    lens_count["bs4"] = len(t1)
    t2 = fetch_all_text_webbase(urlf)
    lens_count["webbase"] = len(t2)
    if len(t2) > len(t1):
        return t2, lens_count, urlf
    return t1, lens_count, urlf

# -------------------------------------------------
# OpenAI
# -------------------------------------------------
API_KEY = get_api_key()
if OpenAI is None:
    st.error("openai 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop()
if not API_KEY:
    st.error("OpenAI API 키가 필요합니다. (Secrets 또는 환경변수 OPENAI_API_KEY)")
    st.stop()

client = OpenAI(api_key=API_KEY, timeout=30.0)
CHAT_MODEL = "gpt-4o-mini"

def call_json_completion(prompt_sys: str, prompt_user: str, max_retries: int = 2) -> dict:
    """LLM에 JSON으로 파싱 강제. 실패시 재시도."""
    schema = {
        "type": "object",
        "properties": {
            "company_name": {"type":"string"},
            "company_intro": {"type":"string"},
            "role_title": {"type":"string"},
            "responsibilities": {"type":"array", "items":{"type":"string"}},
            "qualifications": {"type":"array", "items":{"type":"string"}},
            "preferences": {"type":"array", "items":{"type":"string"}}
        },
        "required": ["company_name","company_intro","role_title","responsibilities","qualifications","preferences"]
    }
    sys = prompt_sys + "\n\n반드시 위 스키마에 맞는 **JSON만** 출력하세요. 다른 텍스트 금지."
    for _ in range(max_retries+1):
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":prompt_user}]
        )
        txt = resp.choices[0].message.content.strip()
        # 코드블록 제거
        txt = re.sub(r"^```json\s*|\s*```$", "", txt, flags=re.S)
        try:
            data = json.loads(txt)
            # 필수키 누락 보정
            for k in ["responsibilities","qualifications","preferences"]:
                if not isinstance(data.get(k), list):
                    data[k] = [str(data.get(k,""))] if data.get(k) else []
            return data
        except Exception:
            continue
    return {}

# -------------------------------------------------
# Sidebar (디버그)
# -------------------------------------------------
with st.sidebar:
    st.title("⚙️ 설정")
    st.caption("필요 최소 설정만 노출합니다.")
    with st.expander("버전/상태 (디버그)"):
        ver_openai = None
        try:
            import openai as _op; ver_openai = getattr(_op,"__version__",None)
        except Exception: pass
        st.write({
            "openai_version": ver_openai,
            "HAS_WEBBASE": HAS_WEBBASE,
        })

# -------------------------------------------------
# 1) 채용 공고 URL → 원문 수집 · 정제
# -------------------------------------------------
st.header("1) 채용 공고 URL → 정제")

job_url = st.text_input("채용 공고 상세 URL", placeholder="https://...")

colb = st.columns([1,1,1])
with colb[0]:
    if st.button("원문 수집 → 정제", type="primary"):
        if not job_url.strip():
            st.warning("채용 공고 URL을 입력하세요.")
        else:
            with st.spinner("원문 수집 중 ..."):
                raw_text, lens, final_url = fetch_jobpage_text(job_url.strip())
                st.session_state["raw_job_text"] = raw_text
                st.session_state["raw_job_lens"] = lens
                st.session_state["raw_job_urlf"] = final_url

            if not st.session_state.get("raw_job_text"):
                st.warning("원문 텍스트를 가져오지 못했습니다. (로그인/동적렌더링/봇차단 가능)")

            # -------- LLM 정제 (요약/정형화) --------
            base = st.session_state.get("raw_job_text","")
            chunked = chunk_text(base, size=1600, overlap=150)
            # 너무 긴 경우 일부만 (과도한 토큰 방지)
            material = "\n\n".join(chunked[:4]) if chunked else base

            sys = (
                "너는 채용담당자다. 입력 텍스트는 채용 공고 원문이다. "
                "다음 항목으로 정확히 정리하라. 임의 생성 금지:\n"
                "- company_name: 회사명(없으면 사이트/브랜드명을 추출)\n"
                "- company_intro: 회사 소개(2~3문장)\n"
                "- role_title: 모집 분야/직무명(없으면 공고 제목에서 추출)\n"
                "- responsibilities: 주요업무 불릿 5~10개\n"
                "- qualifications: 자격요건 불릿 5~10개\n"
                "- preferences: 우대사항 불릿 3~10개 (없으면 빈 배열)\n"
            )
            user = f"[원문 일부]\n{material}\n\n[전체 길이] {len(base)}자"

            data = call_json_completion(sys, user)
            st.session_state["clean_struct"] = data

# -------------------------------------------------
# 2) 회사 요약 (정제 결과)
# -------------------------------------------------
st.header("2) 회사 요약 (정제 결과)")

cdata = st.session_state.get("clean_struct", {})
if cdata:
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f"**회사명:** {cdata.get('company_name','-')}")
    with c2: st.markdown(f"**모집 분야(직무명):** {cdata.get('role_title','-')}")
    with c3:
        if st.session_state.get("raw_job_urlf"):
            st.link_button("채용 공고 열기", st.session_state["raw_job_urlf"])

    st.markdown(f"**간단한 회사 소개(요약)**\n\n{cdata.get('company_intro','-')}")
    cc = st.columns(3)
    with cc[0]:
        st.subheader("주요 업무")
        for b in cdata.get("responsibilities",[]) or ["(없음)"]:
            st.markdown(f"- {b}")
    with cc[1]:
        st.subheader("자격 요건")
        for b in cdata.get("qualifications",[]) or ["(없음)"]:
            st.markdown(f"- {b}")
    with cc[2]:
        st.subheader("우대 사항")
        prefs = cdata.get("preferences", [])
        if not prefs:
            st.caption("우대 사항이 명시되지 않았습니다.")
        for b in prefs:
            st.markdown(f"- {b}")

    with st.expander("디버그: 공고 요약 상태"):
        st.json({
            "job_url": st.session_state.get("raw_job_urlf"),
            "lens": st.session_state.get("raw_job_lens"),
            "resp_cnt": len(cdata.get("responsibilities") or []),
            "qual_cnt": len(cdata.get("qualifications") or []),
            "pref_cnt": len(cdata.get("preferences") or []),
        })
else:
    st.info("상단에서 URL을 입력하고 ‘원문 수집 → 정제’를 먼저 실행하세요.")

# -------------------------------------------------
# 3) 이력서/프로젝트 업로드 → 내부 RAG 인덱싱(숨김)
# -------------------------------------------------
st.header("3) 내 이력서 / 프로젝트 업로드")
st.caption("pdf/txt/md/docx 파일을 업로드하면 내부적으로 자동 인덱싱됩니다. (옵션/숨김 파라미터 사용)")

if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = []

resume_files = st.file_uploader("이력서/포트폴리오 파일 (여러 개 가능)", type=["pdf","txt","md","docx"], accept_multiple_files=True)
if resume_files:
    with st.spinner("파일 인덱싱 중..."):
        added = 0
        for up in resume_files:
            raw = read_text_upload(up)
            if raw:
                # 내부적으로 작은 청크(이력서라 짧기 때문)
                chs = chunk_text(raw, size=400, overlap=80)
                st.session_state.rag_chunks.extend(chs)
                added += len(chs)
        st.success(f"추가 청크 {added}개")

# -------------------------------------------------
# 4) 질문 생성 & 답변 초안
# -------------------------------------------------
st.header("4) 질문 생성 · 답변 · 피드백")

# 내부 고정 파라미터(노출 제거)
NUM_QUESTIONS = 5
TEMPERATURE_Q = 0.9

if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = []

def summarize_resume(snippets: List[str], cap: int = 1200) -> str:
    if not snippets:
        return ""
    joined = " ".join(snippets)
    return joined[:cap]

def generate_questions(clean_struct: dict, resume_snippets: List[str]) -> List[str]:
    resume_sum = summarize_resume(resume_snippets)
    ctx = textwrap.dedent(f"""
    [회사명] {clean_struct.get('company_name','')}
    [직무] {clean_struct.get('role_title','')}
    [주요업무] {", ".join(clean_struct.get('responsibilities',[])[:6])}
    [자격요건] {", ".join(clean_struct.get('qualifications',[])[:6])}
    [우대사항] {", ".join(clean_struct.get('preferences',[])[:6])}
    [지원자 이력서 요약] {resume_sum or '(없음)'}
    """).strip()

    sys = (
        "너는 면접관이다. 회사/직무/요건과 지원자의 이력서를 반영하여 서로 관점이 다른 질문 5개를 생성하라. "
        "형태: 한 줄 질문. 중복/유사 금지. STAR 답변을 유도하도록 상황·지표·결정·리스크 등을 섞어라."
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=TEMPERATURE_Q,
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":ctx}]
    )
    raw = resp.choices[0].message.content.strip()
    qs = [re.sub(r'^\s*\d+\)\s*','',l).strip() for l in raw.splitlines() if len(l.strip())>0]
    if len(qs) > NUM_QUESTIONS:
        qs = qs[:NUM_QUESTIONS]
    return qs

qcols = st.columns([1,1,2])
with qcols[0]:
    if st.button("질문 생성", type="primary"):
        if not st.session_state.get("clean_struct"):
            st.warning("먼저 1)~2) 단계를 완료하세요.")
        else:
            with st.spinner("질문 생성 중..."):
                st.session_state.generated_questions = generate_questions(
                    st.session_state["clean_struct"],
                    st.session_state.get("rag_chunks", [])
                )
                # 새 질문 생성시 팔로업 입력 초기화
                st.session_state["selected_followup"] = ""
                st.session_state["followup_answer"] = ""
                st.session_state["last_followup_result"] = None

with qcols[1]:
    if st.button("질문 비우기"):
        st.session_state.generated_questions = []

st.write("**생성된 질문:**")
if st.session_state.generated_questions:
    for i,q in enumerate(st.session_state.generated_questions,1):
        st.markdown(f"{i}. {q}")
else:
    st.caption("아직 생성된 질문이 없습니다.")

# 답변 입력 & 채점
st.subheader("답변 입력")
answer = st.text_area("여기에 답변을 작성하세요 (STAR 권장: 상황-과제-행동-성과)", height=180, key="main_answer")

def llm_score_and_coach_strict(clean_struct: dict, question: str, answer: str, model: str) -> dict:
    """100점 만점 + 10개 항목(0~10→*10=100), 기준별 코멘트, 수정본(STAR)"""
    criteria = [
        "문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치",
        "시스템설계","트레이드오프","성능/비용","품질/신뢰성","리스크관리"
    ]
    ctx = textwrap.dedent(f"""
    [회사명] {clean_struct.get('company_name','')}
    [직무] {clean_struct.get('role_title','')}
    [주요업무] {", ".join(clean_struct.get('responsibilities',[])[:6])}
    [자격요건] {", ".join(clean_struct.get('qualifications',[])[:6])}
    [우대사항] {", ".join(clean_struct.get('preferences',[])[:6])}
    """).strip()
    sys = (
        f"너는 혹독하지만 공정한 면접 코치다. 아래 10개 기준에 대해 0~10점으로 채점하고, 각 기준별 코멘트를 1문장으로 제공하라.\n"
        f"- 기준: {', '.join(criteria)}\n"
        "총점은 기준 점수를 모두 합산해 10배수(=0~100)로 환산하라. "
        "마지막에 STAR(상황-과제-행동-성과) 형식의 ‘수정본 답변’을 제시하라."
    )
    user = f"[질문]\n{question}\n\n[답변]\n{answer}\n\n[회사/직무 컨텍스트]\n{ctx}"
    resp = client.chat.completions.create(
        model=model, temperature=0.2,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    content = resp.choices[0].message.content.strip()
    # 총점 추출
    m = re.search(r'(\d{1,3})\s*(?:/100|점|$)', content)
    overall = int(m.group(1)) if m else None
    # 기준별 파싱
    crit_scores = {}
    for line in content.splitlines():
        # 예) 문제정의: 8/10 — 코멘트....
        mm = re.match(r'\s*([가-힣A-Za-z/]+)\s*[:：]\s*(\d{1,2})\s*/\s*10', line)
        if mm:
            k = mm.group(1).strip()
            v = int(mm.group(2))
            crit_scores[k] = v*10  # 0~100환산
    # 코멘트 수집
    comments = []
    for c in criteria:
        m2 = re.search(rf"{re.escape(c)}\s*[:：].*", content)
        if m2:
            comments.append(m2.group(0))
    # 수정본
    revised = ""
    m3 = re.search(r"(수정본 답변[:：].*?$)", content, flags=re.S)
    if not m3:
        # 다른 형식 대비
        parts = content.split("\n")
        for i,ln in enumerate(parts):
            if "수정본" in ln and "답변" in ln:
                revised = "\n".join(parts[i+1:]).strip()
                break
    else:
        revised = m3.group(1)
    return {
        "overall_score": overall if overall is not None else sum(crit_scores.values())//10,
        "criteria_scores": crit_scores,
        "criteria_comment_lines": comments,
        "revised_answer": revised or ""
    }

# 질문 선택 & 채점
st.subheader("채점 & 코칭")
if st.session_state.generated_questions:
    choice = st.selectbox("채점할 질문 선택", st.session_state.generated_questions, index=0, key="selected_question_for_scoring")
    if st.button("채점 실행", type="primary"):
        if not st.session_state.get("main_answer","").strip():
            st.warning("답변을 입력하세요.")
        else:
            with st.spinner("채점 중 ..."):
                res = llm_score_and_coach_strict(st.session_state["clean_struct"], choice, st.session_state["main_answer"], CHAT_MODEL)
                st.session_state["last_score"] = res

# 결과 표시
st.subheader("피드백 결과")
last = st.session_state.get("last_score")
if last:
    lc1, lc2 = st.columns([1,3])
    with lc1: st.metric("총점(/100)", last.get("overall_score",0))
    with lc2:
        st.markdown("**기준별 근거(점수/감점/개선):**")
        for line in last.get("criteria_comment_lines",[]):
            st.markdown(f"- {line}")
        if last.get("revised_answer"):
            st.markdown("**수정본 답변(STAR)**")
            st.write(last["revised_answer"])

# -------------------------------------------------
# 5) 팔로업: 제안 → 선택 → 답변 → 피드백
# -------------------------------------------------
st.header("팔로업 질문 · 답변 · 피드백")

def propose_followups(clean_struct: dict, question: str, answer: str) -> List[str]:
    ctx = textwrap.dedent(f"""
    [회사] {clean_struct.get('company_name','')}
    [직무] {clean_struct.get('role_title','')}
    """)
    sys = "면접관으로서 위 답변을 더 깊게 검증하기 위한 팔로업 질문 3개를 생성하라. 한 줄씩."
    user = f"{ctx}\n[기존 질문]\n{question}\n\n[기존 답변]\n{answer}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.8,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    txt = resp.choices[0].message.content.strip()
    qs = [re.sub(r'^\s*\d+\)\s*','',l).strip() for l in txt.splitlines() if len(l.strip())>0]
    return qs[:3] if len(qs)>3 else qs

if "followups" not in st.session_state:
    st.session_state.followups = []

# 팔로업 제안
cols_fu = st.columns([1,1])
with cols_fu[0]:
    if st.button("팔로업 질문 제안"):
        if not st.session_state.get("selected_question_for_scoring") or not st.session_state.get("main_answer","").strip():
            st.warning("먼저 질문 선택과 답변 입력/채점을 진행하세요.")
        else:
            st.session_state.followups = propose_followups(
                st.session_state["clean_struct"],
                st.session_state["selected_question_for_scoring"],
                st.session_state["main_answer"]
            )

# 팔로업 선택 + 답변 입력 (위젯 key만 사용, 대입 금지)
st.write("**팔로업 질문 제안**")
if st.session_state.followups:
    for i,q in enumerate(st.session_state.followups,1):
        st.markdown(f"({i}) {q}")

st.selectbox(
    "채점 받을 팔로업 질문 선택",
    st.session_state.followups if st.session_state.followups else ["(팔로업 없음)"],
    index=0,
    key="selected_followup"
)

st.text_area(
    "팔로업 질문에 대한 나의 답변",
    height=160,
    key="followup_answer"
)

def score_followup(clean_struct: dict, fu_question: str, fu_answer: str) -> dict:
    # 기존 기준 축소(5개)로 빠르게
    criteria = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]
    sys = (
        f"아래 팔로업 답변을 0~100점으로 채점하고, 5개 기준(각 0~20) 점수와 한줄 코멘트를 제공하라. "
        f"기준: {', '.join(criteria)}. 마지막에 STAR 형식의 짧은 보완문단을 제시하라."
    )
    user = f"[팔로업 질문]\n{fu_question}\n\n[팔로업 답변]\n{fu_answer}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.2,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    txt = resp.choices[0].message.content.strip()
    m = re.search(r'(\d{1,3})\s*(?:/100|점|$)', txt)
    score = int(m.group(1)) if m else None
    # 기준 점수 파싱
    comp = []
    for line in txt.splitlines():
        mm = re.match(r'\s*([가-힣A-Za-z/]+)\s*[:：]\s*(\d{1,2})\s*/\s*20', line)
        if mm:
            comp.append((mm.group(1), int(mm.group(2))))
    m3 = re.search(r"(STAR.*?$)", txt, flags=re.S)
    rev = m3.group(1) if m3 else ""
    return {"overall": score, "comp": comp, "revised": rev, "raw": txt}

if st.button("팔로업 채점 & 피드백", type="secondary"):
    fu_q = st.session_state.get("selected_followup","")
    fu_ans = st.session_state.get("followup_answer","")
    if not fu_q or fu_q == "(팔로업 없음)":
        st.warning("팔로업 질문을 선택하세요.")
    elif not fu_ans.strip():
        st.warning("팔로업 답변을 작성하세요.")
    else:
        with st.spinner("팔로업 채점 중 ..."):
            res_fu = score_followup(st.session_state.get("clean_struct",{}), fu_q, fu_ans)
        st.markdown("**팔로업 결과**")
        st.metric("총점(/100)", res_fu.get("overall",0))
        if res_fu.get("comp"):
            st.markdown("**기준별 점수**")
            for k,v in res_fu["comp"]:
                st.markdown(f"- {k}: {v}/20")
        if res_fu.get("revised"):
            st.markdown("**보완 제안(STAR)**")
            st.write(res_fu["revised"])

# -------------------------------------------------
# 6) 역량 레이더 (세션 누적)
# -------------------------------------------------
st.header("역량 레이더 (세션 누적)")

# 히스토리 누적
if "history" not in st.session_state:
    st.session_state.history = []

# 채점 결과를 히스토리에 저장(버튼 직후 저장하도록 설계할 수도 있음)
if st.session_state.get("last_score") and st.session_state.get("selected_question_for_scoring"):
    # 중복 저장 방지 간단 처리: 최근 질문/답변 해시
    key_sig = st.session_state["selected_question_for_scoring"] + "::" + st.session_state.get("main_answer","")[:80]
    prev = st.session_state.history[-1]["sig"] if st.session_state.history else ""
    if prev != key_sig:
        st.session_state.history.append({
            "ts": pd.Timestamp.now(),
            "question": st.session_state["selected_question_for_scoring"],
            "answer": st.session_state.get("main_answer",""),
            "score": st.session_state["last_score"].get("overall_score",0),
            "criteria_scores": st.session_state["last_score"].get("criteria_scores",{}),
            "sig": key_sig
        })

competencies = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치",
                "시스템설계","트레이드오프","성능/비용","품질/신뢰성","리스크관리"]

def build_cdf(hist):
    rows = []
    for h in hist:
        row = {k: np.nan for k in competencies}
        for k,v in (h.get("criteria_scores") or {}).items():
            if k in row:
                row[k] = v//10  # 0~100 → 0~10 스케일로 표시 편의
        rows.append(row)
    return pd.DataFrame(rows) if rows else None

cdf = build_cdf(st.session_state.history)
if cdf is not None and not cdf.empty:
    # 평균
    avg = cdf.mean(skipna=True).fillna(0).tolist()
    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=avg+[avg[0]], theta=competencies+[competencies[0]],
                                      fill='toself', name="세션 평균"))
        # 최신 점수
        last_row = cdf.iloc[-1].fillna(0).tolist()
        fig.add_trace(go.Scatterpolar(r=last_row+[last_row[0]], theta=competencies+[competencies[0]],
                                      fill='toself', name="최신"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])), showlegend=True, height=420)
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cdf.fillna("-").assign(합계=cdf.fillna(0).sum(axis=1)), use_container_width=True)
    st.caption("파란색: 최신 / 초록색: 세션 평균. 표는 각 답변의 최신 점수(NA는 '-')와 세션 누적합·시도횟수를 보여줍니다.")
else:
    st.caption("아직 누적 점수가 없습니다. 위에서 채점까지 완료해 보세요.")
