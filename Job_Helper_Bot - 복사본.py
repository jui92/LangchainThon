###################################################################################################################
#  [Job Helper Bot] : 인공지능 기반 자기소개서 생성 및 모의 면접 코칭 시스템                                          #
#  1. 채용 포털 URL과 이력서를 바탕으로 자소서를 자동 생성합니다.                                                     #
#  2. 회사 정보와 이력서를 참고하여 면접 질문을 만들고 답변 피드백을 제공합니다.                                       #
###################################################################################################################

# Library Import ( coding: utf-8 )
# 필요한 '도구 상자(라이브러리)'들을 불러옵니다.
import os, re, json, urllib.parse, time, io, random
from typing import Optional, Tuple, Dict, List # 파이썬 코드의 자료형(타입)을 명확하게 표시하기 위한 도구

# 웹 정보 수집 및 분석 도구
import requests # 웹 사이트에 접속하여 페이지를 가져오는 '인터넷 연결' 도구
from bs4 import BeautifulSoup # 웹 페이지(HTML)를 분석하고 정보를 뽑아내는 'HTML 분석' 도구
import html2text # HTML 코드를 사람이 읽기 쉬운 일반 텍스트로 변환해주는 도구

# 웹 앱 구축 및 데이터 처리 도구
import streamlit as st # 웹에서 실행되는 깔끔한 사용자 인터페이스(UI)를 쉽게 만드는 도구
import pandas as pd # 데이터를 표(테이블) 형태로 다루기 위한 도구 (기본 기능 제공)
import numpy as np # 숫자 배열(벡터)을 효율적으로 계산하기 위한 도구 (AI 검색에 필수)

# 파일 처리 도구 (이력서 파일을 읽기 위해 필요합니다)
try:
    import pypdf # PDF 파일 읽기용
except ImportError:
    pypdf = None
try:
    from docx import Document as DocxDocument # DOCX 파일 읽기용
except ImportError:
    DocxDocument = None

# ================== 기본 설정 ==================
st.set_page_config(page_title="Job Helper Bot", page_icon="🤖", layout="wide")
st.title("Job Helper Bot : 자소서 생성 / 모의 면접")

# ================== OpenAI (AI 두뇌) 설정 ==================
try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` 패키지가 필요합니다. `pip install openai`로 설치하세요.")
    st.stop()

# API 키를 환경 변수나 Streamlit의 비밀 저장소에서 가져옵니다.
API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
if not API_KEY:
    # 키가 없으면 사용자에게 직접 입력받아 보안 처리합니다.
    API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password")
if not API_KEY:
    st.warning("OpenAI API 키를 입력해야 앱을 사용할 수 있습니다.")
    st.stop()
client = OpenAI(api_key=API_KEY) # OpenAI 클라이언트(통신 담당 객체)를 초기화합니다.

# 사이드바(왼쪽 패널)에 AI 모델 설정을 배치합니다.
with st.sidebar:
    st.subheader("모델 설정")
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0) # 대화와 텍스트 생성에 사용할 AI 모델 선택
    EMBED_MODEL = st.selectbox("임베딩 모델(내부 검색용)", ["text-embedding-3-small","text-embedding-3-large"], index=0) # 텍스트를 숫자로 변환하는 모델 선택

# ================== HTTP 및 URL 유틸리티 함수 ==================
@st.cache_data
def normalize_url(u: str) -> Optional[str]:
    """웹 주소(URL) 형식을 통일하고 정리합니다. (예: http://가 없으면 붙여줍니다)"""
    if not u: return None
    u = u.strip() 
    if not re.match(r"^https?://", u): u = "https://" + u
    parts = urllib.parse.urlsplit(u) 
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

@st.cache_data
def http_get(url: str, timeout: int = 12) -> Optional[requests.Response]:
    """특정 URL로 웹 요청을 보내고 응답을 받습니다 (인터넷 접속)."""
    try:
        r = requests.get(url,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
                                  "Accept-Language": "ko, en;q=0.9"}, timeout=timeout,)
        if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
            return r
    except Exception:
        pass
    return None

def html_to_text(html_str: str) -> str:
    """HTML 코드를 읽기 쉬운 일반 텍스트로 변환합니다."""
    conv = html2text.HTML2Text()
    conv.ignore_links = True # 링크 정보 무시
    conv.ignore_images = True # 이미지 정보 무시
    conv.body_width = 0 # 줄 바꿈 제한 없음
    txt = conv.handle(html_str)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

# ================== 원문 수집 (Jina → Web → BS4 순서로 시도) ==================
def fetch_jina_text(url: str, timeout: int = 15) -> str:
    """Jina AI 프록시를 사용하여 웹 페이지 텍스트를 추출합니다. (동적 페이지의 텍스트도 비교적 잘 가져옴)"""
    try:
        parts = urllib.parse.urlsplit(url)
        # Jina AI 프록시 URL을 만듭니다. (Jina에게 이 페이지의 텍스트를 달라고 요청)
        prox = f"https://r.jina.ai/http://{parts.netloc}{parts.path}"
        if parts.query: prox += f"?{parts.query}"
        r = http_get(prox, timeout=timeout)
        return r.text.strip() if r else "" 
    except Exception:
        return ""

def fetch_webbase_text(url: str) -> str:
    """일반적인 정적 크롤링 방식으로 페이지를 가져와 HTML을 텍스트로 변환합니다."""
    r = http_get(url, timeout=12)
    if not r: return ""
    return html_to_text(r.text)

def fetch_bs4_text(url: str) -> Tuple[str, Optional[BeautifulSoup]]:
    """HTML을 BeautifulSoup으로 파싱하여 본문 텍스트를 추출합니다. (정보 블록 위주로 정리)"""
    r = http_get(url, timeout=12)
    if not r: return "", None
    soup = BeautifulSoup(r.text, "lxml") 
    blocks = []
    # 웹 페이지에서 '기사', '섹션', '본문' 등 중요한 내용을 담을 법한 태그를 찾습니다.
    for sel in ["article","section","main","div","ul","ol"]:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True) 
            if txt and len(txt) > 300: # 텍스트 길이가 300자 이상인 블록만 유효하다고 판단
                txt = re.sub(r"\s+"," ", txt)
                blocks.append(txt)
    if not blocks:
        return soup.get_text(" ", strip=True)[:120000], soup
    
    seen, out = set(), []
    for b in blocks:
        if b not in seen:
            seen.add(b); out.append(b)
    return ("\n\n".join(out)[:120000], soup)

@st.cache_data(show_spinner=False)
def fetch_all_text(url: str):
    """텍스트 추출 방법 3가지를 순서대로 시도하여 가장 좋은 결과를 반환합니다."""
    url = normalize_url(url)
    if not url: return "", {"error":"invalid_url"}, None
    
    # 1. Jina AI 시도 (가장 최신 기술이며 동적 페이지에 강합니다)
    jina = fetch_jina_text(url)
    if jina and len(jina) > 500:
        r = http_get(url, timeout=12)
        soup = BeautifulSoup(r.text, "lxml") if r else None
        return jina, {"source":"jina","len":len(jina),"url_final":url}, soup
    
    # 2. BS4 상세 파싱 시도 (본문 블록 위주)
    bs, soup = fetch_bs4_text(url)
    if bs and len(bs) > 500:
        return bs, {"source":"bs4","len":len(bs),"url_final":url}, soup

    # 3. Webbase 일반 텍스트 시도 (최후의 수단)
    web = fetch_webbase_text(url)
    if web:
        r = http_get(url, timeout=12)
        soup = BeautifulSoup(r.text, "lxml") if r else None
        return web, {"source":"webbase_fallback","len":len(web),"url_final":url}, soup
        
    return "", {"source":"failed_all","len":0,"url_final":url}, None

# ================== 메타/섹션 보조 추출 ==================
def extract_company_meta(soup: Optional[BeautifulSoup]) -> Dict[str,str]:
    """HTML 메타 태그와 제목에서 회사명, 소개, 직무명을 추정합니다."""
    meta = {"company_name":"","company_intro":"","job_title":""}
    if not soup: return meta
    
    cand = []
    og = soup.find("meta", {"property":"og:site_name"});
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
    ogt = soup.find("meta", {"property":"og:title"});
    if ogt and ogt.get("content"): jt = ogt["content"]
    if not jt:
        h1 = soup.find("h1")
        if h1 and h1.get_text(): jt = h1.get_text(strip=True)
    
    meta["job_title"] = re.sub(r"\s+"," ", jt).strip()[:120]
    return meta

# ================== LLM 정제 (채용 공고 → 구조 JSON) ==================
PROMPT_SYSTEM_STRUCT = ("너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
                        "입력 텍스트는 포털 광고 문구, UI잔재, 복수 직무가 섞여 있을 수 있다. "
                        "한국어로 간결하고 중복없이 정제하라. 회사/직무/요건/우대사항을 추출하라.")

@st.cache_data(show_spinner=False)
def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
    """LLM을 사용하여 채용 공고 원문 텍스트를 구조화된 JSON 형태로 깔끔하게 정제합니다."""
    ctx = (raw_text or "").strip()
    if len(ctx) > 14000: # AI 모델의 입력 길이 제한이 있으므로 너무 길면 잘라냅니다.
        ctx = ctx[:14000]

    user_msg = {"role": "user",
                "content": (f"다음 채용 공고 원문을 구조화해줘.\n\n"
                            f"[힌트] 회사명 후보: {meta_hint.get('company_name','')}\n"
                            f"[힌트] 직무명 후보: {meta_hint.get('job_title','')}\n\n"
                            f"[원문]\n{ctx}\n\n"
                            f"응답은 반드시 다음 JSON 스키마에 따라야 하며, 값이 없으면 빈 리스트/문자열을 사용해줘.\n"
                            f"{{ \"company_name\": \"회사명\", \"job_title\": \"직무명\", \"company_intro\": \"회사 소개\", "
                            f"\"responsibilities\": [\"주요 업무 1\", ...], "
                            f"\"requirements\": [\"필수 자격 요건 1\", ...], "
                            f"\"preferred\": [\"우대 사항 1\", ...] }}"
                            )}

    try:
        # OpenAI API를 호출하여 JSON 형식으로 구조화된 데이터를 요청합니다.
        resp = client.chat.completions.create(model=model, temperature=0.2, 
                                              response_format={"type": "json_object"}, 
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg])
        data = json.loads(resp.choices[0].message.content) # JSON 문자열을 파이썬 딕셔너리로 변환
    except Exception as e:
        # 에러 발생 시 힌트 정보를 포함한 기본 딕셔너리를 반환합니다.
        data = {"company_name": meta_hint.get("company_name", "정제 실패"),
                "job_title": meta_hint.get("job_title", "정제 실패"),
                "company_intro": "AI 정제 중 오류가 발생했습니다. 원문으로 대체합니다.",
                "responsibilities": [], "requirements": [], "preferred": [], "error": str(e)}

    return data

# ================== 파일 리더 (PDF/TXT/MD/DOCX) ==================
def read_pdf(data: bytes) -> str:
    """PDF 파일의 내용을 텍스트로 추출합니다. (pypdf 라이브러리 사용)"""
    if pypdf is None: return "PDF 파일 읽기 실패: pypdf 라이브러리가 설치되지 않았습니다."
    try:
        from pypdf import PdfReader # pypdf 동적 임포트
        reader = PdfReader(io.BytesIO(data)) 
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"PDF 파일 내용 추출 중 오류 발생: {e}"

def read_docx(data: bytes) -> str:
    """DOCX 파일의 내용을 텍스트로 추출합니다. (python-docx 라이브러리 사용)"""
    if DocxDocument is None: return "DOCX 파일 읽기 실패: python-docx 라이브러리가 설치되지 않았습니다."
    try:
        document = DocxDocument(io.BytesIO(data)) # 메모리에 있는 DOCX 데이터를 읽음
        return "\n".join([p.text for p in document.paragraphs]) # 모든 문단을 합쳐 텍스트로 반환
    except Exception as e:
        return f"DOCX 파일 내용 추출 중 오류 발생: {e}"

def read_file_text(uploaded) -> str:
    """업로드된 파일의 종류(txt, pdf, docx 등)에 따라 적절한 방법으로 내용을 읽어옵니다."""
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith(".pdf"):
        return read_pdf(data)
    elif name.endswith(".docx"):
        return read_docx(data)
    else:
        # 일반 텍스트 파일(txt, md) 처리
        try:
            return data.decode("utf-8")
        except:
            return data.decode("latin-1")
    return ""

# ================== 간단 청크/임베딩 (RAG, 검색 증강 생성) ==================
def chunk(text: str, size: int = 600, overlap: int = 120) -> List[str]:
    """긴 텍스트(이력서)를 일정한 크기(size)로 자르고 다음 조각과 겹치게(overlap) 만듭니다."""
    t = re.sub(r"\s+"," ", text).strip()
    if not t: return []
    out, start = [], 0
    while start < len(t):
        end = min(len(t), start+size)
        out.append(t[start:end])
        if end == len(t): break
        start = max(0, end-overlap)
    return out

@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    """텍스트 조각들을 LLM이 이해할 수 있는 '숫자 벡터(임베딩)'로 변환합니다."""
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32) # 임베딩 차원(크기)을 1536으로 가정 (text-embedding-3-small 기준)
    resp = client.embeddings.create(model=model_name, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs

def cosine_topk(matrix: np.ndarray, query_vec: np.ndarray, k: int = 4):
    """질문 벡터와 가장 유사한(코사인 유사도) 상위 K개의 텍스트 조각을 찾습니다."""
    if matrix.size == 0:
        return np.array([]), np.array([], dtype=int)
    qn = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn.T # 행렬 곱셈으로 유사도를 계산
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k] # 유사도가 높은 순서대로 상위 k개의 인덱스를 찾음
    return sims[idx], idx

def retrieve_resume_chunks(query: str, k: int = 4):
    """질문과 관련성이 높은 이력서의 텍스트 조각(청크)을 검색(RAG)하여 가져옵니다."""
    chs, embs = st.session_state.get("resume_chunks", []), st.session_state.get("resume_embeds", None)
    if not chs or embs is None:
        return []
    qv = embed_texts([query], EMBED_MODEL) # 질문을 숫자로 변환
    scores, idxs = cosine_topk(embs, qv, k=k) # 질문과 이력서 조각 간의 유사도를 비교
    # 유사도 점수와 해당 이력서 조각을 묶어 반환합니다.
    return [(float(s), chs[int(i)]) for s, i in zip(scores, idxs)]

# ================== 회사 비전/인재상/뉴스 수집 (간소화) ==================
# 이 부분은 외부 API 연동이나 심층 크롤링이 필요하여 현재는 간단한 텍스트 매칭으로 대체됩니다.

def fetch_company_pages(home_url: str) -> Dict[str, List[str]]:
    """회사 홈페이지와 주요 서브 경로에서 '비전/가치' 및 '인재상' 관련 텍스트를 긁어옵니다."""
    # 실제 구현 시 home_url 및 하위 경로를 크롤링하여 '비전', '인재상' 키워드를 포함하는 텍스트 블록을 추출합니다.
    # 현재는 더미 데이터를 반환합니다.
    return {"vision": ["고객 중심의 혁신을 통한 미래 기술 선도", "데이터 기반의 의사결정 문화 확립"], 
            "talent": ["끊임없이 배우는 자세", "협력과 상생의 가치 실현", "도전 정신과 긍정적 태도"]}

def fetch_latest_news(company: str, max_items: int = 5) -> List[Dict]:
    """회사 이름으로 최신 뉴스 기사를 검색합니다. (실제 뉴스 API 연동 필요)"""
    # 실제 구현 시 NAVER 뉴스 검색 API나 Google Custom Search API를 사용해야 합니다.
    # 현재는 회사 이름이 포함된 구글 검색 링크를 반환합니다.
    query = f"{company} 최신 뉴스"
    return [{"title": f"{company} 관련 최신 뉴스 검색 결과", "link": f"https://www.google.com/search?q={urllib.parse.quote(query)}&tbm=nws"}, 
            {"title": "다른 뉴스 기사 예시", "link": f"https://www.google.com/search?q={urllib.parse.quote(company)}&tbm=nws"}]

# ================== LLM 질문/답변/채점 관련 함수 ==================
# AI에게 부여할 역할(시스템 프롬프트)
PROMPT_SYSTEM_Q = ("너는 채용담당자다. 회사/직무 맥락과 채용요건, 그리고 지원자의 이력서 요약을 함께 고려해 "
                   "면접 질문을 한국어로 생성한다. 질문은 서로 형태·관점·키워드가 겹치지 않게 다양화하고, "
                   "수치/지표/기간/규모/리스크/트레이드오프 등도 섞어라.")
PROMPT_SYSTEM_DRAFT = ("너는 면접 답변 코치다. 회사/직무/채용요건과 지원자의 이력서 요약을 결합해 "
                       "질문에 대한 답변 **초안**을 STAR(상황-과제-행동-성과)로 8~12문장, 한국어로 작성한다. "
                       "가능하면 구체적인 지표/수치/기간/임팩트를 포함하라.")
PROMPT_SYSTEM_SCORE_STRICT = ("너는 매우 엄격한 톱티어 면접 코치다. 아래 형식의 JSON만 출력하라. "
                              "각 기준은 0~20 정수이며, 총점은 기준 합계(최대 100)와 반드시 일치해야 한다. "
                              "과장/모호함/근거 부재/숫자 없는 주장/책임 회피/모호한 주어 사용 등을 강하게 감점하라. "
                              "각 기준에 대해 짧지만 구체적 코멘트(강점/감점요인/개선포인트)를 제공하라.")
CRITERIA = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def llm_generate_one_question_with_resume(clean: Dict, level: str, model: str) -> str:
    """회사/직무 정보와 이력서의 핵심 내용을 바탕으로 면접 질문 1개를 생성합니다."""
    # RAG: 이력서에서 가장 중요한 부분 발췌 (질문 생성의 근거로 사용)
    hits = retrieve_resume_chunks("핵심 프로젝트와 기술 스택 요약", k=4) 
    resume_context = "\n".join([f"- {t[:350]}" for _, t in hits])[:1200]

    user_msg = (f"[회사/직무 정보]\n{json.dumps(clean, ensure_ascii=False, indent=2)}\n\n"
                f"[지원자 이력서 핵심 내용]\n{resume_context}\n\n"
                f"위 정보를 바탕으로 '난이도: {level}'에 맞는 가장 핵심적인 질문 1개를 한국어로 생성해줘.")
    
    try:
        resp = client.chat.completions.create(model=model, temperature=0.7, 
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_Q}, 
                                                        {"role":"user","content":user_msg}])
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"질문 생성 오류: {e}"

def llm_draft_answer(clean: Dict, question: str, model: str) -> str:
    """질문과 이력서 내용을 바탕으로 STAR 기반의 답변 초안을 생성합니다."""
    # RAG: 질문과 관련된 이력서 부분 발췌 (답변의 근거로 사용)
    hits = retrieve_resume_chunks(question, k=4) 
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]

    user_msg = (f"[면접 질문]\n{question}\n\n"
                f"[회사/직무 정보]\n{json.dumps(clean, ensure_ascii=False, indent=2)}\n\n"
                f"[지원자 이력서 관련 내용]\n{resume_text}\n\n"
                f"위 내용을 기반으로 '면접 질문'에 대한 답변 초안을 STAR 구조로 작성해줘.")
    
    try:
        resp = client.chat.completions.create(model=model, temperature=0.6, 
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_DRAFT}, 
                                                        {"role":"user","content":user_msg}])
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"답변 초안 생성 오류: {e}"

def llm_score_and_coach_strict(clean: Dict, question: str, answer: str, model: str) -> Dict:
    """질문, 답변, 이력서 내용을 기반으로 엄격한 기준으로 채점하고 피드백을 제공합니다."""
    # RAG: 질문, 답변, 이력서 내용을 종합적으로 고려하여 발췌
    hits = retrieve_resume_chunks(question + "\n" + answer[:800], k=4) 
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    
    # JSON 스키마를 동적으로 생성
    json_schema = {
        "total_score": "총점 (0~100)",
        "comment_score_문제정의": "문제정의 기준의 점수(0~20)와 코멘트",
        "comment_score_데이터/지표": "데이터/지표 기준의 점수(0~20)와 코멘트",
        "comment_score_실행력/주도성": "실행력/주도성 기준의 점수(0~20)와 코멘트",
        "comment_score_협업/커뮤니케이션": "협업/커뮤니케이션 기준의 점수(0~20)와 코멘트",
        "comment_score_고객가치": "고객가치 기준의 점수(0~20)와 코멘트",
        "strengths": ["강점 1", "강점 2"],
        "risks": ["리스크 1", "리스크 2"],
        "improvement": ["개선 포인트 1", "개선 포인트 2"],
        "revised_answer": "STAR 구조를 적용하여 개선된 답변 (질문에 대한 직접적인 응답 포함)"
    }

    user_msg = (f"[면접 질문]\n{question}\n\n"
                f"[지원자 답변]\n{answer}\n\n"
                f"[지원자 이력서 관련 내용]\n{resume_text}\n\n"
                f"위 정보를 바탕으로 지원자의 답변을 엄격하게 채점하고, 코멘트 및 개선된 답변을 한국어로 제공해줘.")
    
    try:
        resp = client.chat.completions.create(model=model, temperature=0.2,
                                              response_format={"type": "json_object"}, 
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_SCORE_STRICT}, 
                                                        {"role":"user","content":user_msg}])
        data = json.loads(resp.choices[0].message.content)
        # LLM의 JSON 출력 키 이름을 사용하기 쉬운 형태로 매핑합니다.
        result = {"total_score": data.get("total_score", 0)}
        for key in data:
            if key.startswith("comment_score_"):
                result[key] = data[key]
        result["strengths"] = data.get("strengths", [])
        result["risks"] = data.get("risks", [])
        result["improvement"] = data.get("improvement", [])
        result["revised_answer"] = data.get("revised_answer", "수정된 답변이 생성되지 않았습니다.")
        return result
    except Exception as e:
        return {"total_score": "N/A", "error": f"채점 오류: {e}", "revised_answer": "채점 오류로 답변이 제공되지 않았습니다."}


# ================== 세션 상태 초기화 ==================
def _init_state():
    """Streamlit 앱을 위한 변수(세션 상태)를 초기화합니다."""
    if 'raw_text' not in st.session_state: st.session_state.raw_text = ""
    if 'fetch_info' not in st.session_state: st.session_state.fetch_info = {}
    if 'clean_struct' not in st.session_state: st.session_state.clean_struct = {}
    if 'company_vision' not in st.session_state: st.session_state.company_vision = []
    if 'company_talent' not in st.session_state: st.session_state.company_talent = []
    if 'latest_news' not in st.session_state: st.session_state.latest_news = []
    if 'resume_text' not in st.session_state: st.session_state.resume_text = ""
    if 'resume_chunks' not in st.session_state: st.session_state.resume_chunks = []
    if 'resume_embeds' not in st.session_state: st.session_state.resume_embeds = None
    if 'current_question' not in st.session_state: st.session_state.current_question = ""
    if 'draft_answer' not in st.session_state: st.session_state.draft_answer = ""
    if 'answer_text' not in st.session_state: st.session_state.answer_text = ""
    if 'last_result' not in st.session_state: st.session_state.last_result = None
    if 'followups' not in st.session_state: st.session_state.followups = []
    if 'last_followup_result' not in st.session_state: st.session_state.last_followup_result = None
    if 'selected_followup' not in st.session_state: st.session_state.selected_followup = ""
    if 'followup_answer' not in st.session_state: st.session_state.followup_answer = ""

_init_state()


# =================================================================================================================
#                                           Streamlit 사용자 인터페이스 (UI) 영역
# =================================================================================================================
# 1) 채용 공고 URL 입력 및 정제 섹션
st.header("1) 채용 공고 URL")
url = st.text_input("채용 공고 상세 URL", placeholder="취업 포털 사이트의 URL을 입력하세요.")
st.text_input("회사 공식 홈페이지 URL (선택)", key="company_home", placeholder="회사 공식 홈페이지 URL을 입력하세요. (비전/인재상 수집용)")

if st.button("원문 수집 → 정제", type="primary"):
    if not url.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner("URL 수집 및 정제 중... (Jina/정적 크롤링 시도)"):
            raw_text, info, soup = fetch_all_text(url)
        
        if not raw_text.strip():
            st.error(f"URL 수집 실패 또는 내용 부족: {url}")
            st.session_state.clean_struct = {}
        else:
            meta_hint = extract_company_meta(soup)
            st.session_state.raw_text = raw_text
            st.session_state.fetch_info = info
            
            clean_struct = llm_structurize(raw_text, meta_hint, CHAT_MODEL)
            st.session_state.clean_struct = clean_struct
            
            if st.session_state.company_home:
                company_pages = fetch_company_pages(st.session_state.company_home)
                st.session_state.company_vision = company_pages.get("vision", [])
                st.session_state.company_talent = company_pages.get("talent", [])
            else:
                st.session_state.company_vision = []
                st.session_state.company_talent = []

            if clean_struct.get("company_name"):
                st.session_state.latest_news = fetch_latest_news(clean_struct["company_name"])
            else:
                st.session_state.latest_news = []
                
            st.success("정제 완료!")

# 2) 회사 요약 (정제 결과) 섹션
st.header("2) 회사 요약")
clean = st.session_state.clean_struct
if clean:
    st.subheader(clean.get("company_name", "회사명 정보 없음"))
    st.caption(f"직무: {clean.get('job_title', '직무 정보 없음')}")
    st.markdown(f"**회사 소개:** {clean.get('company_intro', '-')}")
    
    # 주요 채용 조건 출력
    st.markdown("**주요 업무:**")
    for item in clean.get("responsibilities", []): st.markdown(f"- {item}")
    st.markdown("**자격 요건:**")
    for item in clean.get("requirements", []): st.markdown(f"- {item}")
    st.markdown("**우대 사항:**")
    for item in clean.get("preferred", []): st.markdown(f"- {item}")

# VISION/NEWS: 회사 비전/인재상/뉴스 (있을 때만 표시)
if st.session_state.company_vision or st.session_state.latest_news:
    st.subheader("회사/직무 보조 정보")
    if st.session_state.company_vision:
        st.markdown("**비전/핵심가치:**")
        for item in st.session_state.company_vision: st.markdown(f"- {item}")
    if st.session_state.latest_news:
        st.markdown("**최신 뉴스:**")
        for item in st.session_state.latest_news: st.markdown(f"- [{item.get('title', '제목 없음')}]({item.get('link', '#')})")

st.divider()

# 3) 내 이력서/프로젝트 업로드 섹션
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader("이력서 파일 업로드 (PDF, TXT, MD, DOCX 가능)", type=["pdf","txt","md","docx"], accept_multiple_files=True)
_RESUME_CHUNK = 500
_RESUME_OVLP  = 100

cols_idx = st.columns(2)
with cols_idx[0]:
    if st.button("이력서 인덱싱 (AI 검색 준비)", type="secondary"):
        if uploads:
            full_text = ""
            with st.spinner("파일 읽는 중..."):
                for uploaded_file in uploads:
                    full_text += read_file_text(uploaded_file) + "\n\n"
            
            if full_text.strip():
                st.session_state.resume_text = full_text
                # 이력서 텍스트를 청크(조각)로 나눕니다.
                chunks_list = chunk(full_text, _RESUME_CHUNK, _RESUME_OVLP)
                
                with st.spinner("텍스트 벡터화 중... (LLM이 이해할 수 있는 숫자로 변환)"):
                    # 청크들을 임베딩(숫자 벡터)으로 변환합니다. 이 과정이 RAG(검색 증강 생성)의 핵심입니다.
                    embeds = embed_texts(chunks_list, EMBED_MODEL)
                    st.session_state.resume_chunks = chunks_list
                    st.session_state.resume_embeds = embeds
                st.success(f"인덱싱 완료 (청크 {len(chunks_list)}개)")
            else:
                st.warning("읽을 수 있는 파일 내용이 없습니다. 파일 형식을 확인하세요.")
        else:
            st.warning("이력서 파일을 업로드하세요.")

st.divider()

# 4) 이력서 기반 자소서 생성 섹션
st.header("4) 이력서 기반 자소서 생성")
topic = st.text_input("회사 요청 주제 (선택)", placeholder="예: 직무 지원동기 / 협업 경험 / 문제해결 사례 등 (없으면 '자유 양식'으로 작성)")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str) -> str:
    """LLM을 사용하여 채용 공고와 이력서를 결합한 자소서를 생성합니다."""
    # LLM을 호출하여 자소서를 생성하는 로직을 여기에 구현합니다.
    system_prompt = "너는 뛰어난 헤드헌터이자 전문 자기소개서 작성 코치다. 채용 공고와 지원자의 경험을 완벽히 매칭하여 핵심 경험을 STAR 기법 등으로 자연스럽게 녹여낸 자소서를 작성해야 한다. 분량은 1000자 내외로 한다."
    user_msg = (f"[채용 공고 정보]\n{json.dumps(clean_struct, ensure_ascii=False, indent=2)}\n\n"
                f"[지원자 이력서 원문]\n{resume_text}\n\n"
                f"요청 주제: {topic_hint if topic_hint else '자유 양식'}\n\n"
                f"위 정보를 바탕으로 '요청 주제'에 맞는 자기소개서 초안을 작성해줘.")
                
    try:
        resp = client.chat.completions.create(model=model, temperature=0.7, 
                                              messages=[{"role":"system","content":system_prompt}, 
                                                        {"role":"user","content":user_msg}])
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"자소서 생성 오류: {e}"

if st.button("자소서 생성", type="primary"):
    if not st.session_state.clean_struct:
        st.warning("먼저 채용 공고를 정제하세요. (1단계)")
    elif st.session_state.resume_embeds is None:
        st.warning("이력서 파일을 인덱싱하세요. (3단계)")
    else:
        with st.spinner("자소서 생성 중..."):
            cl_draft = build_cover_letter(st.session_state.clean_struct, st.session_state.resume_text, topic, CHAT_MODEL)
        st.session_state.cover_letter = cl_draft
        st.success("자소서 생성 완료.")

if st.session_state.get('cover_letter'):
    st.subheader("생성된 자기소개서")
    st.text_area("최종 초안", value=st.session_state.cover_letter, height=300)
    st.download_button("자소서 텍스트 다운로드", st.session_state.cover_letter, file_name="cover_letter_draft.txt")

st.divider()

# 5) 질문 생성 & 답변 초안 섹션
st.header("5) 질문 생성 & 답변 초안 (RAG 결합)")
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)

cols_q = st.columns(2)
with cols_q[0]:
    if st.button("새 질문 받기", type="primary"):
        if not st.session_state.clean_struct or st.session_state.resume_embeds is None:
            st.warning("채용 공고 정제 및 이력서 인덱싱을 완료하세요. (1, 3단계)")
        else:
            with st.spinner("채용 공고와 이력서를 분석하여 질문 생성 중..."):
                st.session_state.current_question = llm_generate_one_question_with_resume(st.session_state.clean_struct, level, CHAT_MODEL)
                st.session_state.draft_answer = ""
                st.session_state.answer_text = ""
                st.session_state.last_result = None
                # 팔로업 질문은 메인 채점 후 생성됩니다.
                st.session_state.followups = [] 

with cols_q[1]:
    if st.button("RAG로 답변 초안 생성", type="secondary"):
        if not st.session_state.current_question:
            st.warning("먼저 '새 질문 받기'를 클릭하세요.")
        else:
            with st.spinner("질문과 가장 연관된 이력서 내용을 검색하여 초안 생성 중..."):
                st.session_state.draft_answer = llm_draft_answer(st.session_state.clean_struct, st.session_state.current_question, CHAT_MODEL)
                st.session_state.answer_text = st.session_state.draft_answer # 답변 텍스트 영역에 자동으로 채워줌

st.text_area("질문", value=st.session_state.current_question, height=100)
st.text_area("나의 답변 (초안을 편집하거나 직접 작성하세요)", height=200, key="answer_text")

# 6) 채점 & 코칭 (엄격 모드) 섹션
st.header("6) 채점 & 코칭")
if st.button("채점 & 코칭 실행", type="primary"):
    if not st.session_state.current_question or not st.session_state.answer_text.strip():
        st.warning("질문을 받고 답변을 작성하세요.")
    else:
        with st.spinner("AI 면접관이 답변을 분석하고 엄격하게 채점 및 코칭 중..."):
            st.session_state.last_result = llm_score_and_coach_strict(st.session_state.clean_struct, st.session_state.current_question, st.session_state.answer_text, CHAT_MODEL)
            # 채점 후, 답변의 리스크와 약점을 바탕으로 팔로업 질문 목록을 생성하여 다음 단계를 준비합니다. (더미 데이터)
            st.session_state.followups = ["답변에서 제시한 수치/지표의 구체적 산출 근거는 무엇인가요?", 
                                          "프로젝트 진행 중 발생한 가장 큰 리스크와 그 해결책은 무엇이었나요?", 
                                          "이 경험이 우리 회사에 기여할 수 있는 관점에서 다시 설명해 보세요."]
        st.success("채점 및 코칭 완료.")

# 7) 피드백 결과 섹션
st.header("7) 피드백 결과")
last = st.session_state.last_result
if last:
    st.metric("총점(/100)", last.get("total_score", "N/A"))
    
    st.markdown("---")
    st.markdown("**기준별 코멘트**")
    for criterion in CRITERIA:
        # LLM이 반환한 JSON 키에 맞춰 동적으로 코멘트를 출력합니다.
        key = f"comment_score_{criterion}"
        st.caption(f"**{criterion}:** {last.get(key, '-')}")

    st.markdown("---")
    st.markdown("**강점 & 리스크 & 개선 포인트 요약**")
    st.success(f"**강점:** {', '.join(last.get('strengths', []))}")
    st.error(f"**리스크:** {', '.join(last.get('risks', []))}")
    st.info(f"**개선 포인트:** {', '.join(last.get('improvement', []))}")
    
    st.markdown("---")
    st.markdown("**수정본 답변 (STAR 구조 적용)**")
    st.text_area("LLM 수정본", value=last.get('revised_answer', '-'), height=200)

else:
    st.info("아직 채점 결과가 없습니다. 6단계에서 '채점 & 코칭 실행'을 눌러주세요.")

st.divider()

# 8) 팔로업 질문 → 답변 → 피드백 섹션
st.subheader("팔로업 질문 · 답변 · 피드백")
last = st.session_state.last_result
if last:
    if st.session_state.followups:
        st.markdown("**팔로업 질문 제안**")
        # 제안된 질문 목록을 보여줍니다.
        for i, f in enumerate(st.session_state.followups, 1):
            st.markdown(f"- ({i}) {f}")

        st.selectbox("채점 받을 팔로업 질문 선택", st.session_state.followups, index=0, key="selected_followup")
        st.text_area("팔로업 질문에 대한 나의 답변", height=160, key="followup_answer")
        
        if st.button("팔로업 채점 & 피드백", type="secondary"):
            fu_q   = st.session_state.get("selected_followup", "")
            fu_ans = st.session_state.get("followup_answer", "")
            if not fu_q:
                st.warning("팔로업 질문을 선택하세요.")
            elif not fu_ans.strip():
                st.warning("팔로업 답변을 작성하세요.")
            else:
                with st.spinner("팔로업 답변 채점 중..."):
                    res_fu = llm_score_and_coach_strict(st.session_state.clean_struct, fu_q, fu_ans, CHAT_MODEL)
                st.session_state.last_followup_result = res_fu
                st.markdown("**팔로업 결과**")
                st.metric("총점(/100)", res_fu.get("total_score", "N/A"))
                st.text_area("팔로업 피드백", value=res_fu.get('revised_answer', '피드백 내용'), height=150)
    else:
        st.caption("팔로업 질문은 메인 질문 채점 직후 자동으로 제안됩니다.")