# app.py  —  (전 구성 복원 + 정보 수집 정확도 개선)
import os, io, re, json, textwrap, urllib.parse, difflib, random, time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# -------- Optional deps --------
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
    from bs4 import BeautifulSoup
except Exception:
    st.error("beautifulsoup4가 필요합니다. requirements.txt에 beautifulsoup4 추가")
    st.stop()

try:
    import requests
except Exception:
    st.error("requests가 필요합니다. requirements.txt에 requests 추가")
    st.stop()

# -------- OpenAI SDK (>=1.x) --------
try:
    from openai import OpenAI
except Exception:
    st.error("openai 패키지가 필요합니다. requirements.txt에 openai 추가")
    st.stop()

# -------- Page config --------
st.set_page_config(page_title="회사 특화 가상 면접 코치", page_icon="🎯", layout="wide")

# =====================================================================================
# 공통 유틸
# =====================================================================================
def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

def _snippet(text: str, n:int=220)->str:
    t=_clean_text(text)
    return t if len(t)<=n else t[:n-1]+"…"

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
    return ""

def _domain(u: str|None) -> str|None:
    if not u: return None
    try:
        if not u.startswith("http"): u = "https://" + u
        return urllib.parse.urlparse(u).netloc.lower().replace("www.","")
    except Exception:
        return None

# =====================================================================================
# 검색/수집: 홈페이지·커리어 링크 탐색 + 국내 포털 탐색 + HTML 파서
# =====================================================================================
CAREER_HINTS = ["careers","career","jobs","job","recruit","recruiting","join",
                "채용","인재","인재영입","입사지원","채용공고","커리어","recruitment","hire","hiring"]

JOB_SITES = ["wanted.co.kr","saramin.co.kr","jobkorea.co.kr","rocketpunch.com",
             "indeed.com","linkedin.com","recruit.navercorp.com","kakao.recruit"]

SEARCH_ENGINES = ["https://duckduckgo.com/html/?q={query}"]

SECTION_KEYS = {
    "resp": ["주요 업무","담당 업무","업무","Responsibilities","What you will do","Role","What you'll do"],
    "qual": ["자격 요건","지원 자격","Requirements","Qualifications","Must have"],
    "pref": ["우대 사항","우대조건","Preferred","Nice to have","Plus"]
}

def discover_job_from_homepage(homepage: str, limit: int = 6) -> list[str]:
    """홈페이지 내 a 링크와 예상경로에서 커리어 링크 후보를 수집"""
    out=[]; seen=set()
    try:
        if not homepage.startswith("http"): homepage = "https://" + homepage
        r = requests.get(homepage, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href=a["href"]; text=(a.get_text() or "").lower()
                if any(k in href.lower() or k in text for k in CAREER_HINTS):
                    url=urllib.parse.urljoin(homepage, href)
                    if url not in seen: seen.add(url); out.append(url)
                if len(out)>=limit: break
    except Exception:
        pass
    # 흔한 경로 가산
    for path in ["careers","recruit","jobs","career","채용","인재영입","recruitment","join"]:
        url = urllib.parse.urljoin(homepage.rstrip("/") + "/", path)
        if url not in out: out.append(url)
        if len(out)>=limit: break
    return out[:limit]

def portal_search(company_name: str, role: str, limit:int=6) -> list[str]:
    """국내 포털 우선 검색 (DuckDuckGo로 도메인 필터)"""
    urls=[]
    q_base = f"{company_name} {role} 채용" if role else f"{company_name} 채용"
    site_part = " OR ".join([f'site:{d}' for d in JOB_SITES])
    q = f'{q_base} ({site_part})'
    for engine in SEARCH_ENGINES:
        url = engine.format(query=urllib.parse.quote(q))
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code != 200: continue
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href=a["href"]
                if href.startswith("/l/?kh=-1&uddg="):
                    href = urllib.parse.unquote(href.split("/l/?kh=-1&uddg=")[-1])
                dom=_domain(href)
                if dom and any(d in dom for d in JOB_SITES):
                    if href not in urls: urls.append(href)
                if len(urls)>=limit: break
        except Exception:
            continue
    return urls[:limit]

def discover_job_urls(company: str, role: str, homepage: Optional[str], limit:int=6)->list[str]:
    urls=[]
    if homepage:
        urls += discover_job_from_homepage(homepage, limit=limit)
    if len(urls)<limit:
        urls += portal_search(company, role, limit=limit-len(urls))
    # 간단 중복 제거
    seen=set(); out=[]
    for u in urls:
        if u not in seen: seen.add(u); out.append(u)
        if len(out)>=limit: break
    return out

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

def _split_bullets(txt: str) -> list[str]:
    parts = re.split(r"[•\-\n•·▪️▶︎●■□◆◇\r\t]+", txt)
    out=[_clean_text(x) for x in parts if len(_clean_text(x))>2]
    # 불릿 과잉 문장 정리
    return [re.sub(r'^[\-\•\·\▶️\●\■\□\◆\◇\•\s]+','',x) for x in out]

def pick_section(sections: Dict[str,str], keys: List[str])->Optional[str]:
    for head, body in sections.items():
        if any(kk.lower() in head.lower() for kk in keys):
            return body
    return None

def parse_job_posting(url: str) -> dict:
    """주요업무/자격요건/우대사항을 **원문으로** 최대한 정확히 분리"""
    out = {"title": None, "responsibilities": [], "qualifications": [], "preferred": [], "company_intro": None}
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""): return out
        soup = BeautifulSoup(r.text, "html.parser")

        # JSON-LD 우선
        jp = _extract_json_ld_job(soup)
        if jp:
            out["title"] = jp.get("title")
            desc = _clean_text(jp.get("description", ""))
            if desc:
                for b in _split_bullets(desc):
                    low=b.lower()
                    if any(k in low for k in ["자격","요건","requirements","qualification","필수"]):
                        out["qualifications"].append(b)
                    elif any(k in low for k in ["우대","preferred","nice to have","plus"]):
                        out["preferred"].append(b)
                    else:
                        out["responsibilities"].append(b)

        # 헤더 섹션 스캔
        sections={}
        for h in soup.find_all(re.compile("^h[1-4]$")):
            head=_clean_text(h.get_text()); 
            if not head: continue
            cur=[]; sib=h.find_next_sibling(); stop={"h1","h2","h3","h4"}
            while sib and sib.name not in stop:
                if sib.name in {"p","li","ul","ol","div"}:
                    txt=_clean_text(sib.get_text(" "))
                    if len(txt)>5: cur.append(txt)
                sib=sib.find_next_sibling()
            if cur: sections[head]="\n".join(cur)

        resp = pick_section(sections, SECTION_KEYS["resp"])
        qual = pick_section(sections, SECTION_KEYS["qual"])
        pref = pick_section(sections, SECTION_KEYS["pref"])

        if resp and not out["responsibilities"]:
            out["responsibilities"]=_split_bullets(resp)[:24]
        if qual and not out["qualifications"]:
            out["qualifications"]=_split_bullets(qual)[:24]
        if pref and not out["preferred"]:
            out["preferred"]=_split_bullets(pref)[:24]

        # meta description → 회사 소개 후보
        meta = soup.find("meta", {"name":"description"}) or soup.find("meta", {"property":"og:description"})
        if meta and meta.get("content"): out["company_intro"]=_snippet(meta["content"], 240)

    except Exception:
        pass

    # 표시 안정화
    out["responsibilities"]=[_snippet(x,200) for x in out["responsibilities"]][:12]
    out["qualifications"]=[_snippet(x,200) for x in out["qualifications"]][:12]
    out["preferred"]=[_snippet(x,200) for x in out["preferred"]][:12]
    return out

# =====================================================================================
# OpenAI
# =====================================================================================
def load_api_key_from_env_or_secrets() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if key: return key
    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        return None

with st.sidebar:
    st.title("⚙️ 설정")
    API_KEY = load_api_key_from_env_or_secrets()
    if not API_KEY:
        st.info("환경변수/Secrets에서 키를 못 찾았습니다. 아래에 입력 후 엔터.")
        API_KEY = st.text_input("OPENAI_API_KEY", type="password")
    MODEL = st.selectbox("챗 모델", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small","text-embedding-3-large"], index=0)

if not API_KEY:
    st.error("OpenAI API Key가 필요합니다.")
    st.stop()
try:
    client = OpenAI(api_key=API_KEY, timeout=30.0)
except Exception as e:
    st.error(f"OpenAI 초기화 오류: {e}"); st.stop()

# =====================================================================================
# 상태 초기화
# =====================================================================================
if "company_state" not in st.session_state:
    st.session_state.company_state = {}
if "answer_text" not in st.session_state:
    st.session_state.answer_text = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# =====================================================================================
# ① 회사/직무 입력
# =====================================================================================
st.subheader("① 회사/직무 입력")
company_name_input = st.text_input("회사 이름", placeholder="예: 네이버 / Kakao / 삼성SDS")
role_title         = st.text_input("지원 직무명", placeholder="데이터 애널리스트 / ML 엔지니어 ...")
job_url_input      = st.text_input("채용 공고 URL(선택) — 없으면 자동 탐색")
homepage_input     = st.text_input("공식 홈페이지 URL(선택)", placeholder="https://...")

def build_company_obj(name: str, homepage: Optional[str], role: Optional[str], job_url: Optional[str]) -> dict:
    # 1) 공고 URL 결정
    discovered = [job_url] if job_url else discover_job_urls(name, role or "", homepage or None, limit=6)
    chosen = discovered[0] if discovered else None

    # 2) 공고 파싱
    jp = {"title": None, "responsibilities": [], "qualifications": [], "preferred": [], "company_intro": None}
    if chosen:
        jp = parse_job_posting(chosen)

    return {
        "company_name": name.strip() or "(회사명 미설정)",
        "homepage": homepage or None,
        "role": role or "",
        "role_requirements": jp["responsibilities"],
        "role_qualifications": jp["qualifications"],
        "role_preferred": jp["preferred"],
        "company_intro_site": jp["company_intro"],
        "job_url": chosen
    }

def generate_company_summary(c: dict) -> str:
    # 회사 소개만 요약(공고 섹션은 그대로)
    base = c.get("company_intro_site") or ""
    sys = ("너는 채용담당자다. 아래 원문을 바탕으로 **회사 소개 한 단락(2~3문장)**만 간결하게 요약하라. "
           "광고성 문구/과장표현 제거, 핵심 사실만 유지. 한국어.")
    try:
        r = client.chat.completions.create(
            model=MODEL, temperature=0.2,
            messages=[{"role":"system","content":sys},{"role":"user","content":base}]
        )
        intro = r.choices[0].message.content.strip()
    except Exception:
        intro = base or "회사 소개 정보가 충분하지 않습니다."
    return intro

# 빨간 버튼
if st.button("회사/직무 정보 불러오기", type="primary"):
    if not company_name_input.strip():
        st.warning("회사 이름을 입력해 주세요.")
    else:
        with st.spinner("회사/직무/공고를 수집 중..."):
            cobj = build_company_obj(company_name_input, homepage_input or None, role_title or None, job_url_input or None)
            summary_intro = generate_company_summary(cobj)
            cobj["company_intro_site"] = summary_intro  # 소개만 요약 결과로 교체
            st.session_state.company_state = {"company": cobj}
        st.success("회사 정보 갱신 완료")

company = st.session_state.get("company_state",{}).get("company", {
    "company_name": "(회사명 미설정)", "homepage": None, "role": "",
    "role_requirements": [], "role_qualifications": [], "role_preferred": [],
    "company_intro_site": None, "job_url": None
})

# =====================================================================================
# ② 회사 요약 (채용공고 기준)
# =====================================================================================
st.subheader("② 회사 요약 / 채용 요건")
cols = st.columns(3)
with cols[0]: st.markdown(f"**회사명:** {company.get('company_name')}")
with cols[1]: st.markdown(f"**모집 분야(직무명):** {company.get('role') or 'N/A'}")
with cols[2]:
    if company.get("job_url"): st.link_button("채용 공고 열기", company["job_url"])

st.markdown("**간단한 회사 소개(요약)**")
st.write(company.get("company_intro_site") or "—")

st.divider()
st.markdown("**모집 분야:** " + (company.get("role") or "—"))
colL, colM, colR = st.columns(3)
with colL:
    st.markdown("### 주요 업무(원문)")
    items = company.get("role_requirements", [])
    st.markdown("\n".join([f"- {x}" for x in items]) if items else "_공고에서 추출된 주요 업무가 없습니다._")
with colM:
    st.markdown("### 자격 요건(원문)")
    items = company.get("role_qualifications", [])
    st.markdown("\n".join([f"- {x}" for x in items]) if items else "_공고에서 추출된 자격 요건이 없습니다._")
with colR:
    st.markdown("### 우대 사항(원문)")
    items = company.get("role_preferred", [])
    st.markdown("\n".join([f"- {x}" for x in items]) if items else "_공고에서 추출된 우대 사항이 없습니다._")

# =====================================================================================
# ③ 질문 생성 (이전 구성)
# =====================================================================================
st.subheader("③ 질문 생성")
TYPE_INSTRUCTIONS = {
    "행동(STAR)": "과거 실무 사례를 끌어내도록 S(상황)-T(과제)-A(행동)-R(성과)를 유도하는 질문",
    "기술 심층": "핵심 기술적 의사결정·트레이드오프·성능/비용/품질 지표를 파고드는 심층 질문",
    "핵심가치 적합성": "핵심가치와 태도를 검증하는, 상황기반 행동을 유도하는 질문",
    "역질문": "지원자가 회사를 평가할 수 있도록 통찰력 있는 역질문"
}
q_type = st.selectbox("질문 유형", list(TYPE_INSTRUCTIONS.keys()))
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"])
hint   = st.text_input("질문 생성 힌트(선택)", placeholder="예: 전환 퍼널 / 모델 성능-비용 / 데이터 품질")

def build_ctx(c: dict) -> str:
    return textwrap.dedent(f"""
    [회사명] {c.get('company_name','')}
    [모집 분야] {c.get('role','')}
    [주요 업무] {", ".join(c.get('role_requirements', [])[:6])}
    [자격 요건] {", ".join(c.get('role_qualifications', [])[:6])}
    [우대 사항] {", ".join(c.get('role_preferred', [])[:6])}
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
        ctx = build_ctx(company)
        sys = f"""너는 '{company.get('company_name','')}'의 '{company.get('role','')}' 면접관이다.
회사/직무 컨텍스트와 채용공고(주요업무/자격/우대)를 반영하여 **{q_type}** 유형({TYPE_INSTRUCTIONS[q_type]})의 한국어 질문 **6개 후보**를 생성하라.
서로 형태·관점·키워드가 달라야 하며 난이도는 {level}.
지표/수치/기간/규모/리스크 요소를 적절히 섞어라.
포맷: 1) ... 2) ... 3) ... (한 줄씩)"""
        user = f"[컨텍스트]\n{ctx}\n[힌트]\n{hint or '없음'}"
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.85,
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

# =====================================================================================
# ④ 나의 답변 / 코칭 (100점제)
# =====================================================================================
st.subheader("④ 나의 답변 / 코칭")
ans = st.text_area("여기에 답변을 작성하세요 (STAR 권장: 상황-과제-행동-성과)", height=180, key="answer_text")

def coach_answer(company: dict, question: str, answer: str) -> dict:
    ctx = build_ctx(company)
    sys = ("너는 톱티어 면접 코치다. 한국어로 아래 형식에 맞춰 답하라:\n"
           "1) 총점: 0~100 정수 1개\n"
           "2) 강점: 2~3개 불릿\n"
           "3) 리스크: 2~3개 불릿\n"
           "4) 개선 포인트: 3개 불릿 (행동·지표·임팩트 중심)\n"
           "5) 수정본 답변: STAR(상황-과제-행동-성과) 구조로 간결하게\n"
           "6) 역량 점수(각 0~20 정수): [문제정의, 데이터/지표, 실행력/주도성, 협업/커뮤니케이션, 고객가치]\n"
           "형식/숫자 범위 엄수.")
    user = f"[컨텍스트]\n{ctx}\n\n[면접 질문]\n{question}\n\n[후보자 답변]\n{answer}"
    r = client.chat.completions.create(model=MODEL, temperature=0.35,
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
            res = coach_answer(company, st.session_state["current_question"], st.session_state.answer_text)
            st.session_state.history.append({
                "ts": pd.Timestamp.now(),
                "question": st.session_state["current_question"],
                "user_answer": st.session_state.answer_text,
                "score": res.get("score"),
                "feedback": res.get("raw"),
                "competencies": res.get("competencies")
            })

# =====================================================================================
# 결과/레이더/CSV (이전 구성 유지)
# =====================================================================================
st.divider()
st.subheader("피드백 결과")
if st.session_state.history:
    last = st.session_state.history[-1]
    c1,c2 = st.columns([1,3])
    with c1: st.metric("총점(/100)", last.get("score","—"))
    with c2: st.markdown(last.get("feedback",""))
else:
    st.info("아직 결과가 없습니다.")

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
        fig.add_trace(go.Scatterpolar(r=avg+[avg[0]], theta=competencies+[competencies[0]], fill='toself', name="세션 평균"))
        last_row = cdf.iloc[-1].values.tolist()
        fig.add_trace(go.Scatterpolar(r=last_row+[last_row[0]], theta=competencies+[competencies[0]], fill='toself', name="최신"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,20])), showlegend=True, height=420)
        st.plotly_chart(fig, use_container_width=True)
    cdf_show = cdf.copy()
    cdf_show["합계"] = cdf_show.sum(axis=1)
    st.dataframe(cdf_show, use_container_width=True)
else:
    st.caption("아직 역량 점수가 파싱된 코칭 결과가 없습니다.")

st.divider()
st.subheader("세션 리포트 (CSV)")
def build_report(hist):
    rows=[]
    for h in hist:
        row={"timestamp":h.get("ts"),"question":h.get("question"),"user_answer":h.get("user_answer"),
             "score":h.get("score"),"feedback_raw":h.get("feedback")}
        comps=h.get("competencies")
        if comps and len(comps)==5:
            for k,v in zip(competencies, comps): row[f"comp_{k}"]=v
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp","question","user_answer","score","feedback_raw"])
rep = build_report(st.session_state.history)
st.download_button("CSV 다운로드", data=rep.to_csv(index=False).encode("utf-8-sig"),
                   file_name="interview_session_report.csv", mime="text/csv")

st.caption("Tip) URL을 넣으면 정확도가 가장 높습니다. URL이 없으면 홈페이지→커리어→국내 포털 순으로 자동 탐색합니다.")
