# interview_coach_app.py
import os, io, re, json, textwrap, urllib.parse, difflib, random, time, tempfile
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ============== Optional deps ==============
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    import docx2txt
    DOCX_OK = True
except Exception:
    DOCX_OK = False

try:
    from bs4 import BeautifulSoup
except Exception:
    st.error("beautifulsoup4가 필요합니다. requirements.txt에 beautifulsoup4를 추가하세요.")
    st.stop()

try:
    import requests
except Exception:
    st.error("requests가 필요합니다. requirements.txt에 requests를 추가하세요.")
    st.stop()

# ============== OpenAI SDK (>=1.x) ==============
try:
    from openai import OpenAI
except Exception:
    st.error("openai 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop()

# ============== Page config ==============
st.set_page_config(page_title="회사 특화 가상 면접 코치", page_icon="🎯", layout="wide")

# ============== Helpers ==============
def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

def _snippetize(text: str, maxlen: int = 220) -> str:
    t = _clean_text(text)
    return t if len(t) <= maxlen else t[: maxlen - 1] + "…"

def chunk_text(text: str, size: int = 900, overlap: int = 150):
    text = re.sub(r"\s+", " ", text).strip()
    if not text: return []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        out.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return out

def _domain(u: str|None) -> str|None:
    if not u: return None
    try:
        if not u.startswith("http"): u = "https://" + u
        return urllib.parse.urlparse(u).netloc.lower().replace("www.","")
    except Exception:
        return None

# ============== Secrets / API Key ==============
def _secrets_file_exists() -> bool:
    candidates = [
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    return any(os.path.exists(p) for p in candidates)

def load_api_key_from_env_or_secrets() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if key: return key
    try:
        if _secrets_file_exists() or hasattr(st, "secrets"):
            return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        pass
    return None

# ============== File readers (.txt/.md/.pdf/.docx) ==============
def read_file_to_text(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith((".txt", ".md")):
        for enc in ("utf-8", "cp949", "euc-kr"):
            try: return data.decode(enc)
            except Exception: continue
        return data.decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        if pypdf is None:
            st.warning("pypdf가 필요합니다. requirements.txt에 pypdf 추가.")
            return ""
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
        except Exception as e:
            st.warning(f"PDF 파싱 실패({uploaded.name}): {e}")
            return ""
    elif name.endswith(".docx"):
        if not DOCX_OK:
            st.warning("docx2txt가 필요합니다. requirements.txt에 docx2txt 추가.")
            return ""
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tf:
                tf.write(data); tmp = tf.name
            return docx2txt.process(tmp) or ""
        except Exception as e:
            st.warning(f"DOCX 파싱 실패({uploaded.name}): {e}")
            return ""
        finally:
            if tmp:
                try: os.remove(tmp)
                except Exception: pass
    return ""

# ============== OpenAI client ==============
with st.sidebar:
    st.title("⚙️ 설정")
    API_KEY = load_api_key_from_env_or_secrets()
    if not API_KEY:
        st.info("환경변수/Secrets에서 키를 못 찾았습니다. 아래에 입력 후 엔터.")
        API_KEY = st.text_input("OPENAI_API_KEY", type="password")
    MODEL = st.selectbox("챗 모델", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small","text-embedding-3-large"], index=0)

    _openai_ver = None; _httpx_ver = None
    try:
        import openai as _openai_pkg; _openai_ver = getattr(_openai_pkg, "__version__", None)
    except Exception: pass
    try:
        import httpx as _httpx_pkg; _httpx_ver = getattr(_httpx_pkg, "__version__", None)
    except Exception: pass
    with st.expander("디버그: 시크릿/버전 상태"):
        st.write({
            "api_key_provided": bool(API_KEY),
            "openai_version": _openai_ver,
            "httpx_version": _httpx_ver,
        })

if not API_KEY:
    st.error("OpenAI API Key가 필요합니다.")
    st.stop()
try:
    client = OpenAI(api_key=API_KEY, timeout=30.0)
except Exception as e:
    st.error(f"OpenAI 초기화 오류: {e}"); st.stop()

# ============== Job posting parsing (HTML + LLM fallback) ==============
SECTION_KEYS = {
    "resp": ["주요 업무","담당 업무","업무","Responsibilities","What you will do","Role"],
    "qual": ["자격 요건","지원 자격","Requirements","Qualifications","Must have"],
    "pref": ["우대 사항","우대조건","Preferred","Nice to have","Plus","우대"]
}

def _extract_json_ld_job(soup: BeautifulSoup) -> Optional[dict]:
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
            seq = data if isinstance(data, list) else [data]
            for obj in seq:
                typ = obj.get("@type") if isinstance(obj, dict) else None
                if (isinstance(typ, list) and "JobPosting" in typ) or typ == "JobPosting":
                    return obj
        except Exception:
            continue
    return None

def pick_section(sections: Dict[str, str], keys: List[str]) -> Optional[str]:
    for head, body in sections.items():
        if any(kk.lower() in head.lower() for kk in keys):
            return body
    return None

def _split_bullets(txt: str) -> list:
    bullets = re.split(r"[•\-\n•·▪️▶︎●■□◆◇\r]+", txt)
    return [ _clean_text(b) for b in bullets if len(_clean_text(b)) > 2 ]

def llm_split_jobtext(raw_text: str, client, model: str) -> dict:
    """원문(raw_text)을 3섹션으로 정제: responsibilities / qualifications / preferred."""
    if not raw_text.strip():
        return {"responsibilities": [], "qualifications": [], "preferred": []}
    sys = ("너는 채용공고 정리 도우미다. 한국어 불릿으로 깔끔하게 나눠줘. "
           "출력은 JSON으로만, 키는 responsibilities/qualifications/preferred, 값은 문자열 배열. "
           "원문에 섹션 이름이 없어도 의미로 분류하고, 없으면 빈 배열로 남겨.")
    user = f"[채용공고 원문]\n{raw_text}"
    try:
        r = client.chat.completions.create(
            model=model, temperature=0.0,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        js = json.loads(r.choices[0].message.content)
        def _norm(lst):
            return [re.sub(r"\s+", " ", x).strip() for x in (lst or []) if len(re.sub(r'\s+',' ',x).strip())>1][:12]
        return {
            "responsibilities": _norm(js.get("responsibilities")),
            "qualifications":   _norm(js.get("qualifications")),
            "preferred":        _norm(js.get("preferred")),
        }
    except Exception:
        return {"responsibilities": [], "qualifications": [], "preferred": []}

def parse_job_posting(url: str) -> dict:
    out = {"title": None, "responsibilities": [], "qualifications": [], "preferred": [], "company_intro": None}
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""): return out
        soup = BeautifulSoup(r.text, "html.parser")

        # Title/meta
        jp = _extract_json_ld_job(soup)
        if jp:
            out["title"] = jp.get("title")
            desc = _clean_text(jp.get("description", ""))
            if desc:
                b = _split_bullets(desc)
                # 단순 분류
                for x in b:
                    low = x.lower()
                    if any(k in low for k in ["자격","요건","requirements","qualification","필수"]):
                        out["qualifications"].append(x)
                    elif any(k in low for k in ["우대","preferred","nice to have","plus"]):
                        out["preferred"].append(x)
                    else:
                        out["responsibilities"].append(x)

        # Headings scan
        sections = {}
        for h in soup.find_all(re.compile("^h[1-4]$")):
            head = _clean_text(h.get_text())
            if not head: continue
            nxt=[]; sib=h.find_next_sibling(); stop={"h1","h2","h3","h4"}
            while sib and sib.name not in stop:
                if sib.name in {"p","li","ul","ol","div"}:
                    txt=_clean_text(sib.get_text(" "))
                    if len(txt)>5: nxt.append(txt)
                sib=sib.find_next_sibling()
            if nxt: sections[head]=" ".join(nxt)

        resp = pick_section(sections, SECTION_KEYS["resp"])
        qual = pick_section(sections, SECTION_KEYS["qual"])
        pref = pick_section(sections, SECTION_KEYS["pref"])

        if resp and not out["responsibilities"]:
            out["responsibilities"]=_split_bullets(resp)[:12]
        if qual and not out["qualifications"]:
            out["qualifications"]=_split_bullets(qual)[:12]
        if pref and not out["preferred"]:
            out["preferred"]=_split_bullets(pref)[:12]

        meta_desc = soup.find("meta", {"name":"description"}) or soup.find("meta", {"property":"og:description"})
        if meta_desc and meta_desc.get("content"): out["company_intro"]=_snippetize(meta_desc["content"], 220)

        # ---------- LLM 폴백 (누락 보완) ----------
        if (not out["responsibilities"]) or (not out["qualifications"]) or (not out["preferred"]):
            full_text = _clean_text(soup.get_text(" "))
            split = llm_split_jobtext(full_text, client, MODEL)
            if not out["responsibilities"]: out["responsibilities"] = split["responsibilities"]
            if not out["qualifications"]:   out["qualifications"]   = split["qualifications"]
            if not out["preferred"]:        out["preferred"]        = split["preferred"]

        # 최종 다듬기
        out["responsibilities"] = [_snippetize(x, 140) for x in out["responsibilities"]][:12]
        out["qualifications"]   = [_snippetize(x, 140) for x in out["qualifications"]][:12]
        out["preferred"]        = [_snippetize(x, 140) for x in out["preferred"]][:12]
        return out

    except Exception:
        return out

# ============== Embedding / RAG (간단) ==============
def embed_texts(client: OpenAI, embed_model: str, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=embed_model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def cosine_topk(matrix: np.ndarray, query: np.ndarray, k: int = 4):
    if matrix.size == 0:
        return np.array([]), np.array([], dtype=int)
    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn.T
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx

# ============== Session init ==============
if "company" not in st.session_state:
    st.session_state.company = {
        "company_name": "(회사명 미설정)", "homepage": None, "values": [], "recent_projects": [],
        "company_intro_site": None, "role": "", "role_requirements": [], "role_qualifications": [],
        "role_preferred": [], "job_url": None, "news": []
    }
if "rag_store" not in st.session_state:
    st.session_state.rag_store = {"chunks": [], "embeds": None}
if "history" not in st.session_state:
    st.session_state.history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "answer_text" not in st.session_state:
    st.session_state.answer_text = ""

# ============== ① 회사/직무 입력 & 공고 정제 ==============
st.subheader("① 채용 공고 URL → 정제")
job_url_input = st.text_input("채용 공고 상세 URL", placeholder="https://...wanted.../wd/12345")
col_btn, col_blank = st.columns([1,5])
with col_btn:
    fetch_clicked = st.button("원문 수집 → 정제", type="primary", use_container_width=True)
if fetch_clicked:
    if not job_url_input.strip():
        st.warning("채용 공고 URL을 입력해 주세요.")
    else:
        with st.spinner("채용 공고를 수집/정제 중..."):
            parsed = parse_job_posting(job_url_input.strip())
            # 회사 상태 갱신 (회사명/직무는 URL만으로 알기 어려우므로 아래 UI에서 별도 입력 가능)
            st.session_state.company.update({
                "job_url": job_url_input.strip(),
                "role_requirements": parsed.get("responsibilities", []),
                "role_qualifications": parsed.get("qualifications", []),
                "role_preferred": parsed.get("preferred", []),
                "company_intro_site": parsed.get("company_intro"),
            })
        st.success("정제 완료!")

# 회사명/직무명 수동 입력(또는 뉴스/RAG에서 사용)
with st.expander("회사명/직무명 입력(선택)"):
    st.session_state.company["company_name"] = st.text_input("회사명", value=st.session_state.company.get("company_name",""))
    st.session_state.company["role"] = st.text_input("직무명", value=st.session_state.company.get("role",""))

# ============== ② 회사 요약 (정제 결과 표시) ==============
st.subheader("② 회사 요약 (정제 결과)")
c = st.session_state.company
cols = st.columns(3)
with cols[0]:
    st.markdown(f"**회사명:** {c.get('company_name')}")
with cols[1]:
    st.markdown(f"**모집 분야(직무명):** {c.get('role') or 'N/A'}")
with cols[2]:
    if c.get("job_url"): st.link_button("채용 공고 열기", c["job_url"])

st.markdown(f"**간단한 회사 소개(요약)**\n\n{c.get('company_intro_site') or '—'}")
st.divider()
colL, colM, colR = st.columns(3)
with colL:
    st.markdown("### 주요 업무")
    items = c.get("role_requirements", [])
    if items:
        st.markdown("\n".join([f"- {x}" for x in items]))
    else:
        st.caption("요약 가능한 주요업무가 없습니다.")
with colM:
    st.markdown("### 자격 요건")
    items = c.get("role_qualifications", [])
    if items:
        st.markdown("\n".join([f"- {x}" for x in items]))
    else:
        st.caption("요약 가능한 자격요건이 없습니다.")
with colR:
    st.markdown("### 우대 사항")
    items = c.get("role_preferred", [])
    if items:
        st.markdown("\n".join([f"- {x}" for x in items]))
    else:
        st.caption("요약 가능한 우대사항이 없습니다.")

# ============== ③ 질문 생성 ==============
st.subheader("③ 질문 생성")

# 질문 유형 복원
q_type = st.selectbox(
    "질문 유형",
    ["혼합", "행동(STAR)", "기술 심층", "핵심가치 적합성", "역질문"],
    index=0
)
TYPE_INSTRUCTIONS = {
    "혼합": "행동/기술/가치/역질문이 고르게 섞이되 서로 형태·관점이 다르게",
    "행동(STAR)": "과거 실무 사례를 끌어내도록 S(상황)-T(과제)-A(행동)-R(성과)를 유도",
    "기술 심층": "핵심 기술적 의사결정·트레이드오프·성능/비용/품질 지표를 파고드는 심층",
    "핵심가치 적합성": "핵심가치와 태도를 검증하는 상황기반 행동 질문",
    "역질문": "지원자가 회사를 평가할 수 있도록 통찰력 있는 역질문"
}
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"])
hint   = st.text_input("질문 생성 힌트(선택)", placeholder="예: 퍼널 전환/ 성능-비용 트레이드오프 / 품질 지표")

def build_ctx(company: dict) -> str:
    return textwrap.dedent(f"""
    [회사명] {company.get('company_name','')}
    [모집 분야] {company.get('role','')}
    [주요 업무] {", ".join(company.get('role_requirements', [])[:6])}
    [자격 요건] {", ".join(company.get('role_qualifications', [])[:6])}
    [우대 사항] {", ".join(company.get('role_preferred', [])[:6])}
    """).strip()

def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def pick_diverse(cands: list[str], hist: list[str], gamma: float = 0.35) -> str:
    if not cands: return ""
    if not hist:  return random.choice(cands)
    best=None; best_score=1e9
    for q in cands:
        sims=[_similarity(q,h) for h in hist] or [0.0]
        score=(sum(sims)/len(sims)) + gamma*np.std(sims)
        if score < best_score:
            best_score=score; best=q
    return best

if st.button("새 질문 받기", type="primary", use_container_width=True):
    st.session_state.answer_text = ""  # 이전 답변 초기화
    try:
        ctx = build_ctx(st.session_state.company)
        sys = f"""너는 '{c.get('company_name','')}'의 '{c.get('role','')}' 면접관이다.
회사/직무 컨텍스트와 채용공고(주요업무/자격/우대)를 반영하여 **{q_type}** 유형({TYPE_INSTRUCTIONS[q_type]})의 한국어 질문 **6개 후보**를 생성하라.
서로 형태·관점·키워드가 달라야 하며 난이도는 {level}.
지표/수치/기간/규모/리스크 요소를 적절히 섞어라.
포맷: 1) ... 2) ... 3) ... (한 줄씩)"""
        user = f"[컨텍스트]\n{ctx}\n[힌트]\n{hint or '없음'}"
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.8,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        raw = resp.choices[0].message.content.strip()
        cands = [re.sub(r'^\s*\d+\)\s*','',line).strip() for line in raw.splitlines() if re.match(r'^\s*\d+\)', line)]
        if not cands:
            cands = [l.strip("- ").strip() for l in raw.splitlines() if len(l.strip())>0][:6]
        hist_qs = [h["question"] for h in st.session_state.get("history", [])][-10:]
        selected = pick_diverse(cands, hist_qs)
        st.session_state.current_question = selected or (cands[0] if cands else "질문 생성 실패")
    except Exception as e:
        st.error(f"질문 생성 오류: {e}")

st.text_area("질문", height=110, value=st.session_state.get("current_question",""))

# ============== ④ 나의 답변 / 코칭(100점제) ==============
st.subheader("④ 나의 답변 / 코칭")
ans = st.text_area("여기에 답변을 작성하세요 (STAR 권장: 상황-과제-행동-성과)", height=180, key="answer_text")

def coach_answer(company: dict, question: str, answer: str) -> dict:
    comp = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]
    ctx = build_ctx(company)
    sys = ("너는 톱티어 면접 코치다. 한국어로 아래 형식에 맞춰 답하라:\n"
           "1) 총점: 0~100 정수 1개\n"
           "2) 기준별 근거(점수/감점/개선): 문제정의/데이터지표/실행력주도성/협업커뮤니케이션/고객가치\n"
           "3) 수정본 답변: STAR(상황-과제-행동-성과) 구조\n"
           "4) 역량 점수(각 0~20 정수): [문제정의, 데이터/지표, 실행력/주도성, 협업/커뮤니케이션, 고객가치]\n"
           "형식/숫자 범위 엄수.")
    user = f"[컨텍스트]\n{ctx}\n\n[면접 질문]\n{question}\n\n[후보자 답변]\n{answer}"
    r = client.chat.completions.create(model=MODEL, temperature=0.3,
                                       messages=[{"role":"system","content":sys},{"role":"user","content":user}])
    content = r.choices[0].message.content.strip()

    # 총점
    score = None
    m = re.search(r'(\d{1,3})\s*(?:/100|점|$)', content)
    if m: score = int(m.group(1))
    if score is None:
        m_any = re.search(r'\b(\d{1,3})\b', content)
        if m_any: score = max(0, min(100, int(m_any.group(1))))
    # 역량 5개
    line = content.splitlines()[-1]
    nums = re.findall(r'\b(\d{1,2})\b', line)
    if len(nums) < 5:
        nums = re.findall(r'\b(\d{1,2})\b', content)
    comp_scores = None
    if len(nums) >= 5:
        cand = [int(x) for x in nums[:5]]
        # 5점/10점 척도 보정
        if all(0 <= x <= 5 for x in cand): cand = [x * 4 for x in cand]
        if all(0 <= x <= 10 for x in cand) and any(x > 5 for x in cand): cand = [x * 2 for x in cand]
        comp_scores = [max(0, min(20, x)) for x in cand]

    return {"raw": content, "score": score, "competencies": comp_scores}

if st.button("채점 & 코칭", type="primary", use_container_width=True):
    if not st.session_state.get("current_question"):
        st.warning("먼저 '새 질문 받기'로 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("코칭 중..."):
            res = coach_answer(st.session_state.company, st.session_state["current_question"], st.session_state.answer_text)
            st.session_state.history.append({
                "ts": pd.Timestamp.now(),
                "question": st.session_state["current_question"],
                "user_answer": st.session_state.answer_text,
                "score": res.get("score"),
                "feedback": res.get("raw"),
                "competencies": res.get("competencies")
            })

# ============== 결과 표시 ==============
st.divider()
st.subheader("피드백 결과")
if st.session_state.history:
    last = st.session_state.history[-1]
    c1,c2 = st.columns([1,3])
    with c1: st.metric("총점(/100)", last.get("score","—"))
    with c2: st.markdown(last.get("feedback",""))
else:
    st.info("아직 결과가 없습니다.")

# ============== 레이더 (세션 누적) ==============
st.divider()
st.subheader("역량 레이더 (세션 누적)")
competencies = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def comp_df(hist):
    rows=[h["competencies"] for h in hist if h.get("competencies") and len(h["competencies"])==5]
    return pd.DataFrame(rows, columns=competencies) if rows else None

cdf = comp_df(st.session_state.history)
if cdf is not None and not cdf.empty:
    avg = cdf.mean().values.tolist()
    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=avg+[avg[0]], theta=competencies+[competencies[0]], fill='toself', name="세션 평균"
        ))
        last_row = cdf.iloc[-1].values.tolist()
        fig.add_trace(go.Scatterpolar(
            r=last_row+[last_row[0]], theta=competencies+[competencies[0]], fill='toself', name="최신"
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,20])), showlegend=True, height=420)
        st.plotly_chart(fig, use_container_width=True)
    cdf_show = cdf.copy()
    cdf_show["합계"] = cdf_show.sum(axis=1)
    st.dataframe(cdf_show, use_container_width=True)
else:
    st.caption("아직 역량 점수가 파싱된 코칭 결과가 없습니다.")

# ============== 팔로업 질문 · 답변 · 피드백 ==============
st.divider()
st.subheader("팔로업 질문 · 답변 · 피드백")

if "followup_suggestions" not in st.session_state:
    st.session_state.followup_suggestions = [
        "데이터 분석 과정에서 발견한 위험 요소는 무엇이었고, 이를 어떻게 관리했나요?",
        "고객 유지율을 높이기 위해 어떤 지표를 우선 개선하겠습니까? 이유는?",
        "대안 중 트레이드오프 선택 기준을 수치로 제시해 보세요."
    ]

st.selectbox("제안받은 팔로업 질문 선택", st.session_state.followup_suggestions, key="followup_pick")
st.text_area("팔로업 질문에 대한 나의 답변", key="followup_answer", height=140)

if st.button("팔로업 답변 피드백 받기", use_container_width=True):
    fq = st.session_state.get("followup_pick", "")
    fa = st.session_state.get("followup_answer", "").strip()
    if not fq or not fa:
        st.warning("팔로업 질문을 선택하고 답변을 입력해 주세요.")
    else:
        sys = ("너는 까다로운 면접관이다. 아래 팔로업 Q&A를 100점 만점으로 짧게 채점하고 "
               "감점요인/아쉬운점/개선 포인트를 불릿으로 제시한 뒤, 더 나은 예시 문장 3개를 제안하라.")
        user = f"[팔로업 질문]\n{fq}\n\n[후보자 답변]\n{fa}"
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0.2,
                messages=[{"role":"system","content":sys},{"role":"user","content":user}]
            )
            st.markdown(r.choices[0].message.content.strip())
        except Exception as e:
            st.error(f"팔로업 피드백 오류: {e}")

# ============== 파일 업로드(RAG 자료/이력서) ==============
st.divider()
st.subheader("이력서/회사 문서 업로드 (RAG 소스)")
docs = st.file_uploader("PDF/TXT/MD/DOCX 파일 업로드 (여러 파일 가능)", type=["txt","md","pdf","docx"], accept_multiple_files=True)
if docs:
    with st.spinner("문서 인덱싱 중..."):
        chunks=[]
        for up in docs:
            t = read_file_to_text(up)
            if t: chunks += chunk_text(t, 600, 120)  # 이력서 특화: 더 촘촘히
        if chunks:
            embs = embed_texts(client, EMBED_MODEL, chunks)
            st.session_state.rag_store["chunks"] += chunks
            if st.session_state.rag_store["embeds"] is None or st.session_state.rag_store["embeds"].size==0:
                st.session_state.rag_store["embeds"] = embs
            else:
                st.session_state.rag_store["embeds"] = np.vstack([st.session_state.rag_store["embeds"], embs])
            st.success(f"추가 청크 {len(chunks)}개")
