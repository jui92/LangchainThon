# -*- coding: utf-8 -*-
import os, io, re, textwrap, time, random, urllib.parse
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import streamlit as st

# ---------- Optional deps ----------
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ---------- OpenAI SDK (>=1.x) ----------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ==========================
# Page config
# ==========================
st.set_page_config(page_title="회사 특화 면접 코치", page_icon="🎯", layout="wide")

# ==========================
# Helpers
# ==========================
def clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def snippet(t: str, n: int = 220) -> str:
    t = clean(t)
    return t if len(t) <= n else t[: n - 1] + "…"

def get_api_key() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("OPENAI_API_KEY", None)  # type: ignore
    except Exception:
        return None

def init_openai_client() -> Optional[OpenAI]:
    api_key = get_api_key()
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key, timeout=30.0)
    except Exception:
        return None

# ---------- fetch page text ----------
def fetch_url_text(url: str, timeout: int = 12) -> str:
    """Fetch visible text from a static HTML page."""
    try:
        if not url.startswith("http"):
            url = "https://" + url
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        # remove scripts/styles/nav
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text("\n")
        # collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()
    except Exception:
        return ""

# ---------- file readers ----------
def read_any_file(uploaded) -> str:
    name = uploaded.name.lower()
    raw = uploaded.read()

    if name.endswith((".txt", ".md")):
        for enc in ("utf-8", "cp949", "euc-kr"):
            try:
                return raw.decode(enc)
            except Exception:
                continue
        return raw.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        if pypdf is None:
            st.warning("PDF 파싱: pypdf 미설치")
            return ""
        try:
            reader = pypdf.PdfReader(io.BytesIO(raw))
            parts = []
            for i in range(len(reader.pages)):
                parts.append(reader.pages[i].extract_text() or "")
            return "\n".join(parts)
        except Exception as e:
            st.warning(f"PDF 파싱 실패: {e}")
            return ""

    if name.endswith(".docx"):
        if docx is None:
            st.warning("DOCX 파싱: python-docx 미설치")
            return ""
        try:
            f = io.BytesIO(raw)
            d = docx.Document(f)
            return "\n".join([p.text for p in d.paragraphs])
        except Exception as e:
            st.warning(f"DOCX 파싱 실패: {e}")
            return ""

    return ""

# ---------- LLM wrappers ----------
def llm_struct_from_job(client: OpenAI, model: str, url_text: str) -> Dict:
    """
    채용 공고 원문 텍스트를 구조화: 회사명, 회사소개(요약), 모집분야, 주요업무[], 자격요건[], 우대사항[]
    """
    sys = "너는 채용 공고를 구조화하는 보조자다. 한국어로만 답하라."
    prompt = f"""다음 채용 공고 원문에서 항목을 구조화해줘.
원문은 잡다한 문구(복지, 광고, 보상 등)도 포함될 수 있으니 '주요업무/자격요건/우대사항'과 직접 관련 없는 항목은 제외하고 깔끔하게 정리해.

[출력 JSON 스키마]
{{
  "company_name": "<문자열>",
  "company_intro": "<2~3문장 요약>",
  "role_title": "<직무명/모집분야>",
  "responsibilities": ["불릿", "..."],
  "qualifications": ["불릿", "..."],
  "preferences": ["불릿", "..."]
}}

[원문]
{url_text[:6000]}
"""
    try:
        r = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": prompt}]
        )
        txt = r.choices[0].message.content.strip()
        # JSON 추출 느슨하게
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return {}
        import json
        data = json.loads(m.group(0))
        # 보정
        for k in ["responsibilities", "qualifications", "preferences"]:
            v = data.get(k)
            if isinstance(v, str):
                # 줄바꿈/불릿 분해
                items = [clean(x) for x in re.split(r"[\n;•\-·▶▪️]+", v) if len(clean(x)) > 1]
                data[k] = items[:12]
            elif isinstance(v, list):
                data[k] = [clean(x) for x in v if len(clean(x)) > 1][:12]
            else:
                data[k] = []
        data["company_intro"] = snippet(data.get("company_intro", ""), 400)
        return data
    except Exception:
        return {}

def llm_generate_questions(client: OpenAI, model: str, ctx: str, n: int = 5) -> List[str]:
    sys = "너는 면접관이다. 회사/직무 맥락과 후보자의 이력서를 보고 면접 질문을 생성한다. 한국어로만 답해라."
    prompt = f"""[컨텍스트]
{ctx}

형식: 번호) 질문 한 줄
개수: {n}개
조건:
- 서로 관점, 키워드, 검증 포인트가 겹치지 않게 다양하게
- 지표/수치/규모/기간/트레이드오프가 드러나게
- 실무에서 실제로 검증하고 싶은 포인트를 반영
"""
    r = client.chat.completions.create(
        model=model, temperature=0.9,
        messages=[{"role":"system","content":sys},{"role":"user","content":prompt}]
    )
    raw = r.choices[0].message.content.strip()
    qs = [re.sub(r"^\s*\d+\)\s*","",line).strip() for line in raw.splitlines() if re.match(r"^\s*\d+\)", line)]
    if not qs:
        qs = [l.strip("- ").strip() for l in raw.splitlines() if len(l.strip())>0][:n]
    return qs[:n]

def parse_scores_from_text(txt: str) -> Tuple[Optional[int], Optional[List[int]]]:
    # overall
    overall = None
    m = re.search(r'(\d{1,3})\s*(?:/100|점)\b', txt)
    if m:
        overall = max(0, min(100, int(m.group(1))))
    if overall is None:
        m2 = re.search(r'\b(\d{1,2})/10\b', txt)
        if m2:
            overall = int(m2.group(1)) * 10

    # five criteria 0~20
    last_line = txt.splitlines()[-1]
    nums = re.findall(r'\b(\d{1,2})\b', last_line)
    if len(nums) < 5:
        nums = re.findall(r'\b(\d{1,2})\b', txt)
    comp = None
    if len(nums) >= 5:
        cand = [int(x) for x in nums[:5]]
        if all(0 <= x <= 5 for x in cand):
            cand = [x*4 for x in cand]
        elif all(0 <= x <= 10 for x in cand) and any(x>5 for x in cand):
            cand = [x*2 for x in cand]
        comp = [max(0, min(20, x)) for x in cand]
    return overall, comp

def llm_score_and_coach_strict(client: OpenAI, model: str, company_ctx: str,
                               question: str, answer: str) -> Dict:
    sys = """너는 깐깐한 면접 코치다. 한국어. 다음 형식을 철저히 지켜라:
1) 총점: 0~100 정수 1개
2) 기준별 근거(점수/감점/개선):
   - 문제정의(x/20): ...
   - 데이터/지표(x/20): ...
   - 실행력/주도성(x/20): ...
   - 협업/커뮤니케이션(x/20): ...
   - 고객가치(x/20): ...
3) 수정본 답변(STAR)
4) 역량 점수(쉼표로 5개만): a,b,c,d,e
"""
    user = f"""[회사/직무 컨텍스트]
{company_ctx}

[면접 질문]
{question}

[후보자 답변]
{answer}
"""
    r = client.chat.completions.create(
        model=model, temperature=0.3,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    content = r.choices[0].message.content.strip()
    overall, comp = parse_scores_from_text(content)
    # 섹션 파싱(간단)
    revised = ""
    m = re.search(r"수정본\s*답변.*?\n(.+)", content, re.S)
    if m:
        revised = m.group(1).strip()
    crit = []
    for key in ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]:
        m2 = re.search(rf"{key}\s*\((\d+)\s*/\s*20\)\s*:\s*(.+)", content)
        if m2:
            crit.append({"name": key, "score": int(m2.group(1)), "comment": m2.group(2)})
    return {
        "raw": content,
        "overall": overall if overall is not None else 0,
        "competencies": comp if comp else None,
        "revised": revised,
        "criteria": crit
    }

# ==========================
# Sidebar (settings)
# ==========================
with st.sidebar:
    st.title("⚙️ 설정")
    CHAT_MODEL = st.selectbox("챗 모델", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.caption("OpenAI 키는 환경변수/Secrets에서 자동 읽기")
client = init_openai_client()
if client is None:
    st.error("OpenAI 초기화 실패. OPENAI_API_KEY 설정 필요.")
    st.stop()

# ==========================
# Session init
# ==========================
for k, v in {
    "job_raw_text": "",
    "clean_struct": {},
    "questions": [],
    "current_question": "",
    "answer_text": "",
    "history": [],
    # 팔로업
    "followups": [],
    "selected_followup": "",
    "followup_answer": "",
    "last_followup_result": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# 1) 채용 공고 URL → 정제 (원문 수집 & 구조화)
# ============================================================
st.header("1) 채용 공고 URL → 정제")
job_url = st.text_input("채용 공고 상세 URL", placeholder="https://.../wd/xxxxx")
colb1, colb2 = st.columns([1, 3])
with colb1:
    if st.button("원문 수집 → 정제", type="primary"):
        if not job_url.strip():
            st.warning("채용 공고 URL을 입력하세요.")
        else:
            with st.spinner("원문 수집 중..."):
                raw = fetch_url_text(job_url.strip())
                st.session_state.job_raw_text = raw
            if not raw:
                st.warning("원문 텍스트를 가져오지 못했습니다. (로그인/JS 렌더링 등)")
            else:
                with st.spinner("구조화/정제 중..."):
                    data = llm_struct_from_job(client, CHAT_MODEL, raw)
                if not data:
                    st.warning("정제 결과가 부족합니다.")
                else:
                    data["job_url"] = job_url.strip()
                    st.session_state.clean_struct = data
with colb2:
    st.caption("팁: 로그인 페이지/동적 렌더링 사이트는 텍스트가 적을 수 있습니다.")

# ============================================================
# 2) 회사 요약 (정제 결과)
# ============================================================
st.header("2) 회사 요약 (정제 결과)")
c = st.session_state.clean_struct
if c:
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        st.markdown(f"**회사명:** {c.get('company_name','-')}")
        st.markdown(f"**모집 분야(직무명):** {c.get('role_title','-')}")
        st.markdown("**간단한 회사 소개(요약)**")
        st.write(c.get("company_intro","-"))

    with cc2:
        if c.get("job_url"):
            st.link_button("채용 공고 열기", c["job_url"])
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 주요 업무")
        if c.get("responsibilities"):
            for b in c["responsibilities"]:
                st.markdown(f"- {b}")
        else:
            st.caption("주요 업무가 비어있습니다.")
    with col2:
        st.markdown("### 자격 요건")
        if c.get("qualifications"):
            for b in c["qualifications"]:
                st.markdown(f"- {b}")
        else:
            st.caption("자격 요건이 비어있습니다.")
    with col3:
        st.markdown("### 우대 사항")
        if c.get("preferences"):
            for b in c["preferences"]:
                st.markdown(f"- {b}")
        else:
            st.caption("우대 사항이 비어있습니다.")
else:
    st.info("상단 URL을 수집/정제하면 이곳에 요약이 표시됩니다.")

# ============================================================
# 3) 내 이력서/프로젝트 업로드
# ============================================================
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader("PDF/TXT/MD/DOCX", type=["pdf","txt","md","docx"], accept_multiple_files=True)
resume_corpus = ""
if uploads:
    bodies=[]
    for f in uploads:
        txt = read_any_file(f)
        if txt: bodies.append(txt)
    resume_corpus = "\n\n".join(bodies)
    st.success(f"문서 {len(uploads)}개 로드 완료 (총 {len(resume_corpus)}자)")
else:
    st.caption("이력서를 올리면 이후 단계(자소서/질문)가 더 정교해집니다.")

# ============================================================
# 4) 이력서 기반 자소서 생성
# ============================================================
st.header("4) 이력서 기반 자소서 생성")
cl_topic = st.text_input("회사에서 요구한 자소서 주제(선택)", placeholder="예: 지원 동기 / 성장 스토리 / 문제 해결 사례 등")
if st.button("자소서 생성", type="secondary"):
    if not c:
        st.warning("먼저 1~2단계를 통해 회사 요약을 생성하세요.")
    else:
        sys = "너는 채용 담당자에게 어필할 자소서를 작성하는 보조자다. 한국어, 600~900자."
        ctx = textwrap.dedent(f"""
        [회사 요약]
        회사명: {c.get('company_name','')}
        직무: {c.get('role_title','')}
        주요업무: {', '.join(c.get('responsibilities',[])[:6])}
        자격요건: {', '.join(c.get('qualifications',[])[:6])}
        우대사항: {', '.join(c.get('preferences',[])[:6])}

        [후보자 이력서(요약)]
        {snippet(resume_corpus, 2000)}
        """)
        goal = cl_topic.strip() if cl_topic.strip() else "지원 회사/직무에 특화된 자기소개서"
        prompt = f"위 맥락을 반영하여 '{goal}'를 주제로, STAR 관점과 지표를 포함해 자연스럽게 작성."
        r = client.chat.completions.create(
            model=CHAT_MODEL, temperature=0.5,
            messages=[{"role":"system","content":sys},{"role":"user","content":ctx+"\n\n"+prompt}]
        )
        st.text_area("생성된 자소서", r.choices[0].message.content.strip(), height=280)

# ============================================================
# 5) 질문 생성 & 답변 초안
# ============================================================
st.header("5) 질문 생성 · 답변 초안")
colq1, colq2 = st.columns([1,1])
if colq1.button("질문 생성", type="primary"):
    if not c:
        st.warning("먼저 회사 요약을 만드세요.")
    else:
        ctx = textwrap.dedent(f"""
        회사명: {c.get('company_name','')}
        직무: {c.get('role_title','')}
        주요업무: {', '.join(c.get('responsibilities',[])[:6])}
        자격요건: {', '.join(c.get('qualifications',[])[:6])}
        우대사항: {', '.join(c.get('preferences',[])[:6])}
        후보자 이력서(요약): {snippet(resume_corpus, 1200)}
        """)
        qs = llm_generate_questions(client, CHAT_MODEL, ctx, n=5)
        st.session_state.questions = qs
        st.session_state.current_question = qs[0] if qs else ""
        # 팔로업 초기화
        st.session_state.followups = []
        st.session_state.selected_followup = ""
        st.session_state.followup_answer = ""
        st.session_state.last_followup_result = None

if colq2.button("질문 비우기", type="secondary"):
    st.session_state.questions = []
    st.session_state.current_question = ""
    st.session_state.answer_text = ""
    st.session_state.followups = []
    st.session_state.selected_followup = ""
    st.session_state.followup_answer = ""
    st.session_state.last_followup_result = None

if st.session_state.questions:
    st.markdown("**생성된 질문:**")
    for i, q in enumerate(st.session_state.questions, 1):
        st.markdown(f"{i}. {q}")

st.text_area("답변 입력\n(여기에 답변을 작성하세요: STAR 권장)", key="answer_text", height=160)

# ============================================================
# 6) 채점 & 코칭
# ============================================================
st.header("6) 채점 & 코칭")
# 채점 대상 선택
if st.session_state.questions:
    st.session_state.current_question = st.selectbox("채점 받을 질문 선택",
                                                     st.session_state.questions,
                                                     index=0, key="current_question_select")
if st.button("채점 실행", type="primary"):
    if not st.session_state.current_question:
        st.warning("먼저 질문을 생성하고 선택하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성하세요.")
    else:
        company_ctx = textwrap.dedent(f"""
        회사명: {c.get('company_name','')}
        직무: {c.get('role_title','')}
        주요업무: {', '.join(c.get('responsibilities',[])[:6])}
        자격요건: {', '.join(c.get('qualifications',[])[:6])}
        우대사항: {', '.join(c.get('preferences',[])[:6])}
        """)
        with st.spinner("채점/코칭 중..."):
            res = llm_score_and_coach_strict(client, CHAT_MODEL, company_ctx,
                                             st.session_state.current_question,
                                             st.session_state.answer_text)
        # 기록
        st.session_state.history.append({
            "ts": pd.Timestamp.now(),
            "question": st.session_state.current_question,
            "answer": st.session_state.answer_text,
            "result": res,
        })
        # 팔로업 제안 생성
        sys_fu = "너는 면접관이다. 아래 답변의 빈틈을 파고드는 팔로업 질문 3개를 제안해라. 한국어, 한 줄씩."
        fu_prompt = f"""[질문]
{st.session_state.current_question}

[답변]
{st.session_state.answer_text}

조건: 지표/수치/근거/리스크/의사결정 트레이드오프를 캐묻는 방향으로.
"""
        rfu = client.chat.completions.create(
            model=CHAT_MODEL, temperature=0.8,
            messages=[{"role":"system","content":sys_fu},{"role":"user","content":fu_prompt}]
        )
        lines = [re.sub(r"^\s*\d+\)\s*","",x).strip() for x in rfu.choices[0].message.content.strip().splitlines() if len(x.strip())>3]
        st.session_state.followups = lines[:3]

# 결과 출력
if st.session_state.history:
    last = st.session_state.history[-1]["result"]
    st.subheader("피드백 결과")
    mcol1, mcol2 = st.columns([1,3])
    with mcol1:
        st.metric("총점(/100)", last.get("overall", 0))
    with mcol2:
        st.markdown("**기준별 근거(점수/감점/개선):**")
        for it in last.get("criteria", []):
            st.markdown(f"- **{it['name']}({it['score']}/20)**: {it.get('comment','')}")
        if last.get("revised"):
            st.markdown("**수정본 답변(STAR)**")
            st.write(last["revised"])

# 누적 레이더
st.subheader("역량 레이더 (세션 누적)")
def comp_frame(hist):
    rows=[]
    for h in hist:
        comp = h["result"].get("competencies")
        if comp and len(comp)==5:
            rows.append(comp)
    if not rows: return None
    return pd.DataFrame(rows, columns=["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"])
cdf = comp_frame(st.session_state.history)
if cdf is not None:
    avg = cdf.mean().tolist()
    if PLOTLY_OK:
        fig = go.Figure()
        labels = list(cdf.columns)
        fig.add_trace(go.Scatterpolar(r=avg+[avg[0]], theta=labels+[labels[0]], fill='toself', name="세션 평균"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,20])), showlegend=True, height=420)
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.concat([cdf, pd.DataFrame({"합계": cdf.sum(axis=1)})], axis=1), use_container_width=True)
else:
    st.caption("아직 채점 결과가 없습니다.")

# ============================================================
# 7) 팔로업 질문 & 답변
# ============================================================
st.header("7) 팔로업 질문 · 답변")
if st.session_state.followups:
    st.markdown("**팔로업 질문 제안**")
    for i, q in enumerate(st.session_state.followups, 1):
        st.markdown(f"{i}) {q}")

    # ✅ 위젯들은 key만 사용하고, 세션에 '대입'하지 않음 (충돌 방지)
    st.selectbox("채점 받을 팔로업 질문 선택",
                 st.session_state.followups,
                 index=0,
                 key="selected_followup")

    st.text_area("팔로업 질문에 대한 나의 답변", height=160, key="followup_answer")

    if st.button("팔로업 채점 & 피드백", type="secondary"):
        fu_q   = st.session_state.get("selected_followup", "")
        fu_ans = st.session_state.get("followup_answer", "")
        if not fu_q:
            st.warning("팔로업 질문을 선택하세요.")
        elif not fu_ans.strip():
            st.warning("팔로업 답변을 작성하세요.")
        else:
            company_ctx = textwrap.dedent(f"""
            회사명: {c.get('company_name','')}
            직무: {c.get('role_title','')}
            주요업무: {', '.join(c.get('responsibilities',[])[:6])}
            자격요건: {', '.join(c.get('qualifications',[])[:6])}
            우대사항: {', '.join(c.get('preferences',[])[:6])}
            """)
            with st.spinner("팔로업 채점 중..."):
                res_fu = llm_score_and_coach_strict(client, CHAT_MODEL, company_ctx, fu_q, fu_ans)
            st.session_state.last_followup_result = res_fu

# ============================================================
# 8) 팔로업 피드백
# ============================================================
st.header("8) 팔로업 피드백")
fu = st.session_state.get("last_followup_result")
if fu:
    fc1, fc2 = st.columns([1,3])
    with fc1:
        st.metric("총점(/100)", fu.get("overall", 0))
    with fc2:
        st.markdown("**기준별 근거(점수/감점/개선):**")
        for it in fu.get("criteria", []):
            st.markdown(f"- **{it['name']}({it['score']}/20)**: {it.get('comment','')}")
        if fu.get("revised"):
            st.markdown("**수정본 답변(STAR)**")
            st.write(fu["revised"])
else:
    st.caption("위 7단계에서 팔로업 질문을 선택하고 답변을 작성한 뒤 '팔로업 채점 & 피드백'을 눌러주세요.")
