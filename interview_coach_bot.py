# filename: interview_coach_bot.py
import os, io, re, json, textwrap, time, difflib, urllib.parse, random
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st

# Optional deps
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# OpenAI
try:
    from openai import OpenAI
except Exception as e:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop()

# ------------------ helpers ------------------
def _clean(t:str)->str:
    return re.sub(r"\s+"," ", t or "").strip()

def _snippet(t:str, n:int=220)->str:
    t=_clean(t); return t if len(t)<=n else t[:n-1]+"…"

def load_api_key()->Optional[str]:
    k=os.getenv("OPENAI_API_KEY")
    if k: return k
    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        return None

def read_upload(f)->str:
    name=(f.name or "").lower()
    data=f.read()
    if name.endswith((".txt",".md",".csv",".log")):
        for enc in ("utf-8","cp949","euc-kr"):
            try: return data.decode(enc)
            except Exception: pass
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        if pypdf is None:
            return ""
        try:
            reader=pypdf.PdfReader(io.BytesIO(data))
            return "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
        except Exception:
            return ""
    if name.endswith(".docx"):
        # 매우 경량 파서 (python-docx 없이): 실패 시 안내
        try:
            import zipfile, xml.etree.ElementTree as ET
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                xml=z.read("word/document.xml")
            root=ET.fromstring(xml)
            ns={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            texts=[]
            for node in root.findall(".//w:t", ns):
                texts.append(node.text or "")
            return "\n".join(texts)
        except Exception:
            return ""
    return ""

def chunk_text(t:str, size:int=800, overlap:int=120)->List[str]:
    t=re.sub(r"\s+"," ", t or "").strip()
    if not t: return []
    out=[]; i=0
    while i<len(t):
        j=min(len(t), i+size)
        out.append(t[i:j])
        if j==len(t): break
        i=max(0, j-overlap)
    return out

# ------------------ OpenAI ------------------
API_KEY=load_api_key()
if not API_KEY:
    st.error("OpenAI API 키가 필요합니다. (Secrets 또는 환경변수 OPENAI_API_KEY)")
    st.stop()
client=OpenAI(api_key=API_KEY)

DEFAULT_MODEL="gpt-4o-mini"
EMBED_MODEL="text-embedding-3-small"

def embed(texts:List[str])->np.ndarray:
    if not texts: return np.zeros((0,1536), dtype=np.float32)
    r=client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in r.data], dtype=np.float32)

def cosine_topk(matrix:np.ndarray, query:np.ndarray, k:int=4):
    if matrix.size==0: return np.array([]), np.array([], dtype=int)
    qn=query/(np.linalg.norm(query,axis=1,keepdims=True)+1e-12)
    mn=matrix/(np.linalg.norm(matrix,axis=1,keepdims=True)+1e-12)
    sims=(mn@qn.T).reshape(-1)
    idx=np.argsort(-sims)[:k]
    return sims[idx], idx

# ------------------ Company from job URL (정제용 최소 형태) ------------------
SECTION_HEADS=[
    ("주요 업무", ["주요 업무","담당 업무","업무","What you will do","Responsibilities"]),
    ("자격 요건", ["자격 요건","지원 자격","Requirements","Qualifications","Must have"]),
    ("우대 사항", ["우대","우대 사항","Preferred","Nice to have"]),
]

def fetch_html(url:str)->BeautifulSoup|None:
    try:
        r=requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code!=200 or "text/html" not in r.headers.get("content-type",""): return None
        return BeautifulSoup(r.text, "html.parser")
    except Exception:
        return None

def extract_sections_from_job(soup:BeautifulSoup)->dict:
    out={"title":None,"responsibilities":[],"qualifications":[],"preference":[]}
    if soup is None: return out
    # title
    if soup.title and soup.title.string:
        out["title"]=_clean(soup.title.string)

    # JSON-LD 우선
    for s in soup.find_all("script", {"type":"application/ld+json"}):
        try:
            data=json.loads(s.string or "")
            lst=data if isinstance(data,list) else [data]
            for obj in lst:
                t=obj.get("@type")
                if (isinstance(t,list) and "JobPosting" in t) or t=="JobPosting":
                    desc=_clean(obj.get("description",""))
                    if desc:
                        # 라인 분리
                        bullets=[x.strip(" -•·▪︎▶") for x in re.split(r"[•\-\n•·▪︎▶]+", desc) if len(x.strip())>3]
                        # 대강 분류
                        for b in bullets:
                            low=b.lower()
                            if any(k in low for k in ["우대","preferred","nice to have"]):
                                out["preference"].append(b)
                            elif any(k in low for k in ["자격","require","qualif"]):
                                out["qualifications"].append(b)
                            else:
                                out["responsibilities"].append(b)
        except Exception:
            pass

    # 헤더 기반 추가 보강
    def harvest(keys):
        # 헤더부터 다음 헤더 직전까지 텍스트 수집
        for h in soup.find_all(re.compile("^h[1-4]$")):
            head=_clean(h.get_text())
            if not head: continue
            if any(k.lower() in head.lower() for k in keys):
                texts=[]
                sib=h.find_next_sibling()
                while sib and sib.name not in {"h1","h2","h3","h4"}:
                    if sib.name in {"p","li","ul","ol","div","section"}:
                        tx=_clean(sib.get_text(" "))
                        if len(tx)>5: texts.append(tx)
                    sib=sib.find_next_sibling()
                joined=" ".join(texts)
                if joined:
                    items=[x.strip(" -•·▪︎▶") for x in re.split(r"[•\-\n•·▪︎▶]+", joined) if len(x.strip())>3]
                    return items[:20]
        return []
    if not out["responsibilities"]:
        out["responsibilities"]=harvest(SECTION_HEADS[0][1])
    if not out["qualifications"]:
        out["qualifications"]=harvest(SECTION_HEADS[1][1])
    if not out["preference"]:
        out["preference"]=harvest(SECTION_HEADS[2][1])

    # 길이 제한
    out["responsibilities"]=[_snippet(x,140) for x in out["responsibilities"]][:12]
    out["qualifications"]=[_snippet(x,140) for x in out["qualifications"]][:12]
    out["preference"]=[_snippet(x,140) for x in out["preference"]][:12]
    return out

# ------------------ 페이지 설정 ------------------
st.set_page_config(page_title="지원 회사 특화 면접 코치", page_icon="🎯", layout="wide")

# ================== 1) 채용 공고 URL → 정제 ==================
st.header("1) 채용 공고 URL → 정제")
job_url=st.text_input("채용 공고 상세 URL", placeholder="https://... (원티드/사람인/잡코리아 등)")

company_state=st.session_state.setdefault("company_state", {})
if st.button("원문 수집 → 정제", type="primary"):
    soup=fetch_html(job_url) if job_url else None
    jp=extract_sections_from_job(soup) if soup else {"title":None,"responsibilities":[],"qualifications":[],"preference":[]}
    company_state["job_url"]=job_url or None
    company_state["role_title"]=jp.get("title")
    company_state["responsibilities"]=jp.get("responsibilities",[])
    company_state["qualifications"]=jp.get("qualifications",[])
    company_state["preference"]=jp.get("preference",[])
    st.success("정제 완료")

# ================== 2) 회사 요약 (정제 결과) ==================
st.header("2) 회사 요약 (정제 결과)")
c = company_state
colA,colB=st.columns([2,1])
with colA:
    st.markdown(f"**회사명:**  {_clean(c.get('company_name') or '') or '—'}")
with colB:
    st.markdown(f"**모집 분야(직무명):**  {_snippet(c.get('role_title') or '') or '—'}")

st.markdown("**간단한 회사 소개(요약)**")
st.write(_snippet(c.get("company_intro_site") or "채용 공고 상의 소개나 메타 설명이 없으면 공란일 수 있습니다.", 500))

col1,col2,col3=st.columns(3)
with col1:
    st.subheader("주요 업무")
    rs=c.get("responsibilities",[]) or ["(공고에서 추출된 주요 업무가 없습니다.)"]
    st.markdown("\n".join([f"- {x}" for x in rs]))
with col2:
    st.subheader("자격 요건")
    qs=c.get("qualifications",[]) or ["(공고에서 추출된 자격 요건이 없습니다.)"]
    st.markdown("\n".join([f"- {x}" for x in qs]))
with col3:
    st.subheader("우대 사항")
    ps=c.get("preference",[]) or ["(공고에서 추출된 우대 사항이 없습니다.)"]
    st.markdown("\n".join([f"- {x}" for x in ps]))

with st.expander("디버그: 공고 요약 상태"):
    lens={}
    if job_url:
        soup=fetch_html(job_url)
        if soup:
            # 단순 길이 지표
            bs4_len=len(_clean(soup.get_text(" ") or ""))
            lens["bs4"]=bs4_len
            lens["webbase"]=bs4_len  # 자리표시자
    st.json({
        "job_url": job_url,
        "lens": lens,
        "resp_cnt": len(c.get("responsibilities") or []),
        "qual_cnt": len(c.get("qualifications") or []),
        "pref_cnt": len(c.get("preference") or []),
    })

# ================== 3) 내 이력서/프로젝트 업로드 (RAG 인덱싱) ==================
st.header("3) 내 이력서/프로젝트 업로드")
uploaded=st.file_uploader("이력서/프로젝트 파일 추가 (PDF/TXT/MD/DOCX 가능, 여러 개 업로드)",
                          type=["pdf","txt","md","docx"], accept_multiple_files=True)

rag = st.session_state.setdefault("rag", {"chunks":[], "embeds":None})
if uploaded:
    texts=[]
    for f in uploaded:
        t=read_upload(f)
        if t: texts+=chunk_text(t, size=700, overlap=100)
    if texts:
        vec=embed(texts)
        if rag["embeds"] is None or rag["embeds"].size==0:
            rag["chunks"]=texts
            rag["embeds"]=vec
        else:
            rag["chunks"]+=texts
            rag["embeds"]=np.vstack([rag["embeds"], vec])
        st.success(f"문서 인덱싱: 청크 {len(texts)}개 추가. 총 {len(rag['chunks'])}개")

# ================== 공통: 질문 생성기 / 채점기 ==================
def build_company_ctx() -> str:
    return textwrap.dedent(f"""
    [모집 분야] {c.get('role_title') or ''}
    [주요 업무] {", ".join(c.get('responsibilities', [])[:6])}
    [자격 요건] {", ".join(c.get('qualifications', [])[:6])}
    [우대 사항] {", ".join(c.get('preference', [])[:6])}
    """).strip()

def retrieve_supports(query:str, k:int=4)->List[Tuple[float,str]]:
    if rag["embeds"] is None or not rag["chunks"]:
        return []
    qv=embed([query])
    scores, idxs = cosine_topk(rag["embeds"], qv, k=k)
    return [(float(s), rag["chunks"][int(i)]) for s,i in zip(scores, idxs)]

def gen_questions(model:str, ctx:str, n:int=5)->List[str]:
    sys = ("너는 한국 IT기업 면접관이다. 컨텍스트를 반영해 **행동(STAR), 기술 심층, 가치 적합성**이 섞이도록 "
           f"{n}개의 서로 다른 질문을 만들어라. 각 문장은 1줄.")
    user = f"[컨텍스트]\n{ctx}"
    r=client.chat.completions.create(model=model, temperature=0.9,
                                     messages=[{"role":"system","content":sys},
                                               {"role":"user","content":user}])
    raw=r.choices[0].message.content.strip()
    qs=[re.sub(r'^\s*\d+[.)]\s*','',x).strip() for x in raw.splitlines() if len(x.strip())>3]
    return qs[:n]

COMP_KEYS=["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def score_answer(model:str, question:str, answer:str, ctx:str, supports:List[Tuple[float,str]]):
    sup_txt="\n".join([f"- ({s:.2f}) {t[:400]}" for s,t in supports]) if supports else ""
    sys=( "너는 혹독하지만 공정한 면접 코치다. 다음 형식으로만 한국어로 답해라.\n"
         "1) 총점: 0~100 정수 1개\n"
         "2) 기준별 근거(점수/감점/개선): 각 항목에 대해 한 줄로 엄격하게\n"
         "3) 수정본 답변(STAR)\n"
         f"4) 역량 점수(0~20, 쉼표 5개): {', '.join(COMP_KEYS)}")
    user=(f"[질문]\n{question}\n\n[후보자 답변]\n{answer}\n\n[회사/공고 컨텍스트]\n{ctx}\n"
          f"[지원 문서 발췌]\n{sup_txt}")
    r=client.chat.completions.create(model=model, temperature=0.3,
                                     messages=[{"role":"system","content":sys},
                                               {"role":"user","content":user}])
    text=r.choices[0].message.content.strip()

    # 총점
    score=None
    m=re.search(r'(\d{1,3})\s*(?:/100|점|$)', text)
    if m: score=max(0,min(100,int(m.group(1))))
    # 역량
    nums=re.findall(r'\b(\d{1,2})\b', text.splitlines()[-1])
    comps=None
    if len(nums)>=5:
        cand=[int(x) for x in nums[:5]]
        if all(0<=x<=5 for x in cand): cand=[x*4 for x in cand]
        if all(0<=x<=10 for x in cand) and any(x>5 for x in cand): cand=[x*2 for x in cand]
        comps=[max(0,min(20,x)) for x in cand]
    return {"raw":text,"score":score,"competencies":comps}

# ================== 4) 질문 생성 · 답변 · 피드백 (예전 구조 복구) ==================
st.header("4) 질문 생성 · 답변 · 피드백")
MODEL=DEFAULT_MODEL

if "questions" not in st.session_state:
    st.session_state.questions=[]

colq1,colq2=st.columns([1,1])
with colq1:
    if st.button("질문 생성", type="primary"):
        ctx=build_company_ctx()
        st.session_state.questions=gen_questions(MODEL, ctx, n=5)
with colq2:
    if st.button("질문 비우기"):
        st.session_state.questions=[]

if st.session_state.questions:
    st.markdown("**생성된 질문:**")
    for i,q in enumerate(st.session_state.questions,1):
        st.markdown(f"{i}. {q}")

sel_q = st.selectbox("채점할 질문 선택", st.session_state.questions, index=0 if st.session_state.questions else None)
user_ans = st.text_area("답변 입력", height=160, key="main_answer")

if "history" not in st.session_state:
    st.session_state.history=[]

if st.button("채점 실행", type="primary"):
    if not sel_q or not user_ans.strip():
        st.warning("질문과 답변을 확인해 주세요.")
    else:
        ctx=build_company_ctx()
        supports=retrieve_supports(sel_q + "\n" + user_ans, k=4)
        res=score_answer(MODEL, sel_q, user_ans, ctx, supports)
        st.session_state.history.append({
            "ts": pd.Timestamp.now(),
            "question": sel_q,
            "answer": user_ans,
            "score": res["score"],
            "competencies": res["competencies"],
            "feedback": res["raw"],
            "supports": supports,
        })

# ----- 결과 표시 -----
st.subheader("피드백 결과")
if st.session_state.history:
    last=st.session_state.history[-1]
    c1,c2=st.columns([1,3])
    with c1:
        st.metric("총점(/100)", last.get("score","—"))
    with c2:
        st.markdown(last.get("feedback",""))

# ================== 5) 팔로업 질문 · 답변 · 피드백 (동일 구조) ==================
st.header("5) 팔로업 질문 · 답변 · 피드백")

# 팔로업 질문 제안 (자동)
if "followup_pool" not in st.session_state:
    st.session_state.followup_pool=[]
if st.button("팔로업 질문 제안 갱신"):
    # 최근 결과를 바탕으로 팔로업 3개 제안
    base_q = (st.session_state.history[-1]["question"] if st.session_state.history else "") or ""
    base_ans= (st.session_state.history[-1]["answer"] if st.session_state.history else "") or ""
    ctx=build_company_ctx()
    sys=("너는 까다로운 면접관이다. 아래 질문/답변을 바탕으로 **심층 팔로업 질문 3개**를 1줄씩 제안하라. 모호하면 안 된다.")
    user=f"[원질문]\n{base_q}\n\n[기존답변]\n{base_ans}\n\n[회사/공고 컨텍스트]\n{ctx}"
    r=client.chat.completions.create(model=MODEL, temperature=0.7,
                                     messages=[{"role":"system","content":sys},
                                               {"role":"user","content":user}])
    lines=[re.sub(r'^\s*\d+[.)]\s*','',x).strip() for x in r.choices[0].message.content.strip().splitlines() if len(x.strip())>3]
    st.session_state.followup_pool=lines[:3]
if st.session_state.followup_pool:
    st.markdown("**팔로업 질문 제안**")
    for i,q in enumerate(st.session_state.followup_pool,1):
        st.markdown(f"{i}) {q}")

followup_q = st.selectbox("채점할 팔로업 질문 선택", st.session_state.followup_pool, index=0 if st.session_state.followup_pool else None, key="followup_select")
followup_ans = st.text_area("팔로업 질문에 대한 나의 답변", height=140, key="followup_answer")

if "followup_history" not in st.session_state:
    st.session_state.followup_history=[]

if st.button("팔로업 채점 & 피드백", type="primary"):
    if not followup_q or not followup_ans.strip():
        st.warning("팔로업 질문과 답변을 입력해 주세요.")
    else:
        ctx=build_company_ctx()
        supports=retrieve_supports(followup_q + "\n" + followup_ans, k=4)
        res=score_answer(MODEL, followup_q, followup_ans, ctx, supports)
        st.session_state.followup_history.append({
            "ts": pd.Timestamp.now(),
            "question": followup_q,
            "answer": followup_ans,
            "score": res["score"],
            "competencies": res["competencies"],
            "feedback": res["raw"],
            "supports": supports,
        })

# 팔로업 결과
if st.session_state.followup_history:
    st.subheader("팔로업 결과")
    last=st.session_state.followup_history[-1]
    c1,c2=st.columns([1,3])
    with c1:
        st.metric("총점(/100)", last.get("score","—"))
    with c2:
        st.markdown(last.get("feedback",""))

# ================== 6) 역량 레이더 (세션 누적) ==================
st.header("6) 역량 레이더 (세션 누적)")
def to_df(hist):
    rows=[h["competencies"] for h in hist if h.get("competencies") and len(h["competencies"])==5]
    return pd.DataFrame(rows, columns=COMP_KEYS) if rows else None

main_df = to_df(st.session_state.history) or pd.DataFrame(columns=COMP_KEYS)
fup_df  = to_df(st.session_state.followup_history) or pd.DataFrame(columns=COMP_KEYS)

avg = (pd.concat([main_df, fup_df]).mean() if not pd.concat([main_df, fup_df]).empty else pd.Series([0]*5, index=COMP_KEYS)).values.tolist()
latest = (main_df.tail(1).values.tolist()[0] if len(main_df)>0 else [0]*5)

if PLOTLY_OK:
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(r=latest+[latest[0]], theta=COMP_KEYS+[COMP_KEYS[0]], fill='toself', name="최신(본 질문)"))
    fig.add_trace(go.Scatterpolar(r=avg+[avg[0]], theta=COMP_KEYS+[COMP_KEYS[0]], fill='toself', name="세션 평균"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,20])), showlegend=True, height=420)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.bar_chart(pd.DataFrame({"최신":latest,"평균":avg}, index=COMP_KEYS))

# 표로도 확인
tab1, tab2 = st.tabs(["본 질문 누적", "팔로업 누적"])
with tab1:
    if not main_df.empty:
        main_df["합계"]=main_df.sum(axis=1)
        st.dataframe(main_df, use_container_width=True)
    else:
        st.caption("본 질문 채점 결과가 아직 없습니다.")
with tab2:
    if not fup_df.empty:
        fup_df["합계"]=fup_df.sum(axis=1)
        st.dataframe(fup_df, use_container_width=True)
    else:
        st.caption("팔로업 채점 결과가 아직 없습니다.")

st.caption("※ ‘우대 사항’은 사이트별 마크업 차이로 누락 가능—별도 처리 예정.")
