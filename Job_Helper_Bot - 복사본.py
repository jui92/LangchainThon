###################################################################################################################
#  [Job Helper Bot] : 인공지능 기반 자기소개서 생성 및 모의 면접 코칭 시스템                                          #
#  1. 채용 포털 사이트 URL로 조회한 회사 정보와 등록한 이력서를 바탕으로 자소서를 자동으로 생성해줍니다                  #
#  2. 채용 포털 사이트 URL / 기업 홈페이지 URL / 뉴스 기사 를 참고하여 모의 면접을 실시하고 답변에 대한 피드백을 해줍니다.#
###################################################################################################################

# Library Import ( coding: utf-8 )
import os, re, json, urllib.parse, time, io, random
from typing import Optional, Tuple, Dict, List  # 코드의 자료형(타입)을 명확히 정의하는 도구

# 웹 크롤링 및 데이터 분석 도구
import requests                 # 웹 서버에 접속하여 페이지를 가져오는 도구 (HTTP 통신)
from bs4 import BeautifulSoup   # 웹 페이지(HTML)를 분석하고 정보를 추출하는 도구
import html2text                # HTML 코드를 읽기 쉬운 일반 텍스트로 변환하는 도구

# 웹 애플리케이션 및 데이터 처리 도구
import streamlit as st      # 웹 사용자 인터페이스(UI)를 쉽게 구축하는 프레임워크
import pandas as pd         # 데이터프레임(표 형태 데이터) 처리 도구 (여기서는 사용되지 않음)
import numpy as np          # 숫자 배열(행렬) 계산 도구 (AI 임베딩 벡터 처리에 필수)

# 파일 처리 도구 (PDF, DOCX 등 이력서 파일을 읽기 위해 필요)
try:
    import pypdf # PDF 파일 읽기용 라이브러리
except Exception:
    pypdf = None
try:
    import docx2txt, tempfile # DOCX 파일 읽기용 라이브러리 (임시 파일 사용)
except Exception:
    pass

# ================== 기본 설정 ==================
st.set_page_config(page_title="Job Helper Bot", page_icon="🤖", layout="wide") # 웹 앱의 기본 설정
st.title("Job Helper Bot : 자소서 생성 / 모의 면접") # 앱 제목

# ================== OpenAI (AI 두뇌) 설정 ==================
try:
    from openai import OpenAI # OpenAI API를 사용하기 위한 클라이언트 모듈
except ImportError:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop() # 패키지 없으면 앱 실행 중단

# API 키 설정 (환경 변수 또는 Streamlit 비밀 저장소에서 가져옴)
API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
if not API_KEY:
    API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password") # 키가 없으면 사용자에게 직접 입력받음
if not API_KEY:
    st.stop() # 키 없으면 앱 실행 중단
client = OpenAI(api_key=API_KEY) # OpenAI 클라이언트 초기화

# 사이드바에 AI 모델 설정 배치
with st.sidebar:
    st.subheader("모델 설정")
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0) # 질문 생성 및 채점용 LLM
    EMBED_MODEL = st.selectbox("임베딩 모델(내부용)", ["text-embedding-3-small","text-embedding-3-large"], index=0) # 텍스트를 벡터로 변환하는 RAG용 모델

# ================== HTTP 유틸 (웹 연결 보조 함수) ==================
def normalize_url(u: str) -> Optional[str]:
    """URL 형식을 통일하고 정리합니다. (http:// 또는 https:// 가 없으면 추가)"""
    if not u: return None
    u = u.strip()
    if not re.match(r"^https?://", u): u = "https://" + u
    parts = urllib.parse.urlsplit(u)
    # URL에서 불필요한 fragment(#)를 제거하고 통일된 형태로 반환
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

def http_get(url: str, timeout: int = 12) -> Optional[requests.Response]:
    """특정 URL로 웹 요청을 보내고 응답을 받습니다."""
    try:
        r = requests.get(url,
                         # 봇이 아님을 알리는 User-Agent와 한국어 선호 헤더 추가
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
                                  "Accept-Language": "ko, en;q=0.9"}, timeout=timeout,)
        # 성공 코드(200)와 HTML 콘텐츠 타입인지 확인
        if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
            return r
    except Exception:
        pass
    return None

# ================== 원문 수집 (Jina → Web → BS4 순서로 시도) ==================
def fetch_jina_text(url: str, timeout: int = 15) -> str:
    """Jina AI 프록시를 사용하여 웹 페이지 텍스트를 추출합니다. (동적 렌더링 페이지에 유리)"""
    try:
        parts = urllib.parse.urlsplit(url)
        # Jina AI 리더 프록시 URL을 구성하여 Jina 서버를 통해 텍스트를 요청
        prox = f"https://r.jina.ai/http://{parts.netloc}{parts.path}"
        if parts.query: prox += f"?{parts.query}"
        r = http_get(prox, timeout=timeout)
        return r.text.strip() if r else "" 
    except Exception:
        return ""

def html_to_text(html_str: str) -> str:
    """HTML 코드를 일반 텍스트로 변환합니다."""
    conv = html2text.HTML2Text()
    conv.ignore_links = True              # 링크 정보 무시
    conv.ignore_images = True             # 이미지 정보 무시
    conv.body_width = 0                   # 줄 바꿈 제한 없음
    txt = conv.handle(html_str)
    txt = re.sub(r"\n{3,}", "\n\n", txt)  # 3줄 이상의 빈 줄을 2줄로 정리
    return txt.strip()

def fetch_webbase_text(url: str) -> str:
    """기본 크롤링으로 페이지 전체 HTML을 가져와 텍스트로 변환합니다. (정적 페이지용)"""
    r = http_get(url, timeout=12)
    if not r: return ""
    return html_to_text(r.text)

def fetch_bs4_text(url: str) -> Tuple[str, Optional[BeautifulSoup]]:
    """BeautifulSoup으로 파싱하여 본문 텍스트 블록 위주로 추출합니다. (정보 블록 위주)"""
    r = http_get(url, timeout=12)
    if not r: return "", None
    soup = BeautifulSoup(r.text, "lxml") # HTML 파싱
    blocks = []
    # 'article', 'section', 'main' 등 본문을 담을 법한 태그를 선택하여 텍스트 추출
    for sel in ["article","section","main","div","ul","ol"]:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True) 
            if txt and len(txt) > 300:      # 300자 이상인 블록만 유효하다고 판단
                txt = re.sub(r"\s+"," ", txt)
                blocks.append(txt)
    if not blocks:
        # 본문 블록이 없으면 페이지 전체 텍스트를 잘라서 반환
        return soup.get_text(" ", strip=True)[:120000], soup
    
    seen, out = set(), []
    for b in blocks:
        if b not in seen:
            seen.add(b); out.append(b)
    # 중복 제거된 블록들을 합쳐서 반환 (최대 12만 자)
    return ("\n\n".join(out)[:120000], soup)

def fetch_all_text(url: str):
    """3가지 텍스트 추출 방법(Jina, Webbase, BS4)을 순서대로 시도하여 가장 좋은 결과를 반환합니다."""
    url = normalize_url(url)
    if not url: return "", {"error":"invalid_url"}, None
    
    # 1. Jina AI 시도 (가장 우선)
    jina = fetch_jina_text(url)
    if jina:
        # Jina로 텍스트를 가져왔더라도, 메타 정보 추출을 위해 BS4를 한 번 더 시도
        _, soup = fetch_bs4_text(url) 
        return jina, {"source":"jina","len":len(jina),"url_final":url}, soup
    
    # 2. Webbase 일반 텍스트 시도 (Jina 실패 시)
    web = fetch_webbase_text(url)
    if web:
        _, soup = fetch_bs4_text(url)
        return web, {"source":"webbase","len":len(web),"url_final":url}, soup
    
    # 3. BS4 본문 블록 파싱 시도 (최후의 수단)
    bs, soup = fetch_bs4_text(url)
    return bs, {"source":"bs4","len":len(bs),"url_final":url}, soup

# ================== 메타/섹션 보조 추출 ==================
def extract_company_meta(soup: Optional[BeautifulSoup]) -> Dict[str,str]:
    """HTML 메타 태그와 제목에서 회사명, 소개, 직무명을 추정합니다."""
    meta = {"company_name":"","company_intro":"","job_title":""}
    if not soup: return meta
    
    # 회사명 후보 추출 (og:site_name, application-name, title)
    cand = []
    og = soup.find("meta", {"property":"og:site_name"})
    if og and og.get("content"): cand.append(og["content"])
    app = soup.find("meta", {"name":"application-name"})
    if app and app.get("content"): cand.append(app["content"])
    if soup.title and soup.title.string: cand.append(soup.title.string)
    
    # 후보들을 구분자(하이픈, 파이프 등) 기준으로 자르고 정리
    cand = [re.split(r"[\-\|\·\—]", c)[0].strip() for c in cand if c]
    cand = [c for c in cand if 2 <= len(c) <= 40]
    meta["company_name"] = cand[0] if cand else ""
    
    # 회사 소개(간단 설명) 추출 (og:description, description)
    md = soup.find("meta", {"name":"description"}) or soup.find("meta", {"property":"og:description"})
    if md and md.get("content"):
        meta["company_intro"] = re.sub(r"\s+"," ", md["content"]).strip()[:500]
        
    # 직무명 추출 (og:title, h1, h2 순서로 시도)
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

# ================== 규칙 파서 (LLM 보조용, 우대사항 등 정제) ==================
def rule_based_sections(raw_text: str) -> dict:
    """정규 표현식과 키워드를 사용하여 채용 공고 텍스트에서 섹션을 분리합니다."""
    txt = re.sub(r"\r", "", raw_text or "").strip()
    # 줄 바꿈 기준으로 분리 후, 불필요한 마커/공백 제거
    lines = [re.sub(r"\s+", " ", l).strip(" -•·▶▪️") for l in txt.split("\n")]
    lines = [l for l in lines if l]

    # 섹션 헤더를 식별하는 정규 표현식
    hdr_resp = re.compile(r"(주요\s*업무|담당\s*업무|Role|Responsibilities?)", re.I)
    hdr_qual = re.compile(r"(자격\s*요건|지원\s*자격|Requirements?|Qualifications?)", re.I)
    hdr_pref = re.compile(r"(우대\s*사항|우대|선호|Preferred|Nice\s*to\s*have|Plus)", re.I)

    bucket = None # 현재 처리 중인 섹션 (responsibilities, qualifications, preferences)
    out = {"responsibilities": [], "qualifications": [], "preferences": []}

    def push(line, b):
        """섹션 리스트에 라인을 추가하고 중복을 제거"""
        if line and len(line) > 1 and line not in out[b]:
            out[b].append(line[:180])

    for l in lines:
        if hdr_resp.search(l): bucket = "responsibilities"; continue # '주요 업무' 헤더 발견
        if hdr_qual.search(l): bucket = "qualifications"; continue   # '자격 요건' 헤더 발견
        if hdr_pref.search(l): bucket = "preferences"; continue      # '우대 사항' 헤더 발견
        
        if bucket is None:
            # 헤더 없이 시작된 경우, 키워드(기술 스택)를 보고 임시로 섹션을 추정
            if hdr_pref.search(l):
                bucket = "preferences"
            elif any(k in l.lower() for k in ["java","python","spark","airflow","kafka","ml","sql"]):
                bucket = "responsibilities"
            else:
                continue
        push(l, bucket)
    
    # 자격 요건에 '우대' 키워드가 포함된 경우, 우대 사항 섹션으로 이동 처리
    kw_pref = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
    remain_qual = []
    for q in out["qualifications"]:
        if kw_pref.search(q):
            out["preferences"].append(q)
        else:
            remain_qual.append(q)
    out["qualifications"] = remain_qual

    # 각 섹션의 중복 정리 및 최종 정리
    for k in out:
        seen=set(); clean=[]
        for s in out[k]:
            s = re.sub(r"\s+", " ", s).strip(" -•·▶▪️").strip()
            if s and s not in seen:
                seen.add(s); clean.append(s)
        out[k] = clean[:12] # 최대 12개 항목만 유지
    return out

# ================== LLM 정제 (채용 공고 → 구조 JSON) ==================
PROMPT_SYSTEM_STRUCT = ("너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
                        "입력 텍스트는 포털 광고 문구, UI잔재, 복수 직무가 섞여 있을 수 있다. "
                        "한국어로 간결하고 중복없이 정제하라.")

def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
    """LLM을 사용하여 채용 공고 원문 텍스트를 구조화된 JSON 형태로 정제합니다."""
    ctx = (raw_text or "").strip()
    if len(ctx) > 14000: # AI 모델의 입력 길이 제한 고려
        ctx = ctx[:14000]

    user_msg = {"role": "user",
                "content": ("다음 채용 공고 원문을 구조화해줘.\n\n"
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
                            "- '우대 사항(preferences)'은 비워두지 말고, 원문에서 '우대/선호/preferred/plus/가산점' 등 표시가 있는 항목을 그대로 담아라.\n"
                            "- 불릿/마커/이모지 제거, 문장 간결화, 중복 제거."),}

    try:
        # OpenAI API 호출: JSON 형식으로 응답 요청
        resp = client.chat.completions.create(model=model, temperature=0.2,
                                              response_format={"type": "json_object"},
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg])
        data = json.loads(resp.choices[0].message.content) # JSON 문자열 파싱
    except Exception as e:
        # 오류 발생 시 힌트 정보를 포함한 기본값 반환
        data = {"company_name": meta_hint.get("company_name",""),
                "company_intro": meta_hint.get("company_intro","원문이 정제되지 않았습니다."),
                "job_title": meta_hint.get("job_title",""),
                "responsibilities": [],
                "qualifications": [],
                "preferences": [],
                "error": str(e)}

    # LLM이 생성한 리스트 항목에 대한 후처리 (공백 제거, 중복 제거)
    for k in ["responsibilities","qualifications","preferences"]:
        arr = data.get(k, [])
        if not isinstance(arr, list): arr = []
        clean_list=[]; seen=set()
        for it in arr:
            t = re.sub(r"\s+"," ", str(it)).strip(" -•·▶▪️").strip()
            if t and t not in seen:
                seen.add(t); clean_list.append(t[:180])
        data[k] = clean_list[:12]

    # LLM이 생성한 문자열 항목에 대한 후처리
    for k in ["company_name","company_intro","job_title"]:
        if k in data and isinstance(data[k], str):
            data[k] = re.sub(r"\s+"," ", data[k]).strip()

    # 우대 사항이 부족할 경우, 규칙 기반 파서 결과를 보조적으로 사용
    if len(data.get("preferences", [])) < 1:
        rb = rule_based_sections(ctx)
        if rb.get("preferences"):
            merged = data.get("preferences", []) + rb["preferences"]
            seen=set(); pref=[]
            for s in merged:
                s=s.strip()
                if s and s not in seen:
                    seen.add(s); pref.append(s)
            data["preferences"] = pref[:12]
        # 최종적으로, 자격 요건에 우대 키워드가 남아있으면 우대 사항으로 이동
        if not data["preferences"]:
            kw_pref = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
            remain=[]; moved=[]
            for q in data.get("qualifications", []):
                if kw_pref.search(q): moved.append(q)
                else: remain.append(q)
            if moved:
                data["preferences"] = moved[:12]
                data["qualifications"] = remain[:12]
    return data

# ================== 파일 리더 (PDF/TXT/MD/DOCX) ==================
# PDF 리더 함수
def read_pdf(data: bytes) -> str:
    """PDF 파일의 내용을 텍스트로 추출합니다. (pypdf 라이브러리 사용)"""
    if pypdf is None: return ""
    try:
        # io.BytesIO를 사용하여 메모리 내의 바이트 데이터를 파일처럼 처리
        reader = pypdf.PdfReader(io.BytesIO(data)) 
        # 모든 페이지의 텍스트를 추출하여 결합
        return "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
    except Exception:
        return ""

# DOCX 리더 함수
def read_docx(data: bytes) -> str:
    """DOCX 파일의 내용을 텍스트로 추출합니다. (docx2txt 라이브러리 사용)"""
    try:
        import docx2txt, tempfile
        # docx2txt는 파일 경로가 필요하므로 임시 파일을 생성하여 데이터를 쓰고 처리
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
            tmp.write(data); tmp.flush()
            text = docx2txt.process(tmp.name) or ""
            return text
    except Exception:
        return ""

def read_file_text(uploaded) -> str:
    """업로드된 파일 종류에 따라 적절한 리더 함수를 호출하여 텍스트를 읽습니다."""
    name = uploaded.name.lower()
    data = uploaded.read() # 파일 내용을 바이트 형태로 읽기
    if name.endswith((".txt",".md")):
        # 텍스트 파일: 인코딩을 순차적으로 시도 (utf-8, cp949, euc-kr)
        for enc in ("utf-8","cp949","euc-kr"):
            try: return data.decode(enc)
            except Exception: continue
        return data.decode("utf-8", errors="ignore") # 최종적으로 무시하고 디코딩
    elif name.endswith(".pdf"):
        return read_pdf(data)
    elif name.endswith(".docx"):
        return read_docx(data)
    return ""

# ================== 간단 청크/임베딩 (RAG - 검색 증강 생성의 핵심) ==================
def chunk(text: str, size: int = 600, overlap: int = 120) -> List[str]:
    """긴 텍스트(이력서)를 일정한 크기(size)로 자르고 다음 조각과 겹치게(overlap) 만듭니다."""
    t = re.sub(r"\s+"," ", text).strip()
    if not t: return []
    out, start = [], 0
    while start < len(t):
        end = min(len(t), start+size)
        out.append(t[start:end])
        if end == len(t): break
        start = max(0, end-overlap) # 중복(overlap)을 적용하여 다음 청크 시작 위치 설정
    return out

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    """텍스트 조각들을 AI가 이해하는 '숫자 벡터(임베딩)'로 변환합니다."""
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32) # 임베딩 차원(크기) 1536 (text-embedding-3-small 기준)
    # OpenAI Embedding API 호출
    resp = client.embeddings.create(model=model_name, input=texts)
    # 결과를 NumPy 배열로 변환
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs

def cosine_topk(matrix: np.ndarray, query_vec: np.ndarray, k: int = 4):
    """질문 벡터와 가장 유사한(코사인 유사도) 상위 K개의 텍스트 조각 인덱스를 찾습니다."""
    if matrix.size == 0:
        return np.array([]), np.array([], dtype=int)
    # 벡터들을 정규화 (길이 1로 통일)
    qn = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn.T             # 행렬 곱셈: 코사인 유사도 계산
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]  # 유사도가 높은 순서대로 상위 k개의 인덱스를 찾음
    return sims[idx], idx

def retrieve_resume_chunks(query: str, k: int = 4):
    """질문과 관련성이 높은 이력서의 텍스트 조각(청크)을 검색하여 가져옵니다."""
    chs, embs = st.session_state.get("resume_chunks", []), st.session_state.get("resume_embeds", None)
    if not chs or embs is None:
        return []
    qv = embed_texts([query], EMBED_MODEL)    # 질문을 벡터로 변환
    scores, idxs = cosine_topk(embs, qv, k=k) # 이력서 청크들과 유사도 비교
    # 유사도 점수와 해당 이력서 청크를 묶어 반환
    return [(float(s), chs[int(i)]) for s, i in zip(scores, idxs)]

# ================== 회사 비전/인재상/뉴스 수집 ==================
VISION_KEYS = ["비전","미션","핵심가치","가치","원칙","문화","행동강령","Talent","인재상","Our Mission","Vision","Values"]

def safe_get_text(el) -> str:
    """BeautifulSoup 요소에서 안전하게 텍스트를 추출합니다."""
    try:
        return el.get_text(" ", strip=True)
    except Exception:
        return ""

def fetch_company_pages(home_url: str) -> Dict[str, List[str]]:
    """홈페이지와 주요 서브 경로에서 '비전/가치' 및 '인재상' 관련 텍스트를 스크래핑합니다."""
    out = {"vision": [], "talent": []}
    if not home_url: return out
    base = normalize_url(home_url)
    if not base: return out
    # 크롤링할 대표 서브 경로 목록
    paths = ["","/","/about","/company","/about-us","/mission","/vision","/values","/culture","/careers","/talent","/people"]
    seen = set()
    for p in paths:
        url = (base.rstrip("/") + p) if p else base
        if url in seen: continue
        seen.add(url)
        r = http_get(url, timeout=8)
        if not r: continue
        soup = BeautifulSoup(r.text, "lxml")
        texts=[]
        # 제목 태그와 문단 태그에서 텍스트 추출
        for tag in soup.find_all(["h1","h2","h3","h4","p","li"]):
            t = safe_get_text(tag)
            if not t: continue
            t = re.sub(r"\s+"," ", t)
            if 6 <= len(t) <= 260: # 적당한 길이의 텍스트만 필터링
                texts.append(t)
        # 키워드 매칭을 통해 vision/talent 섹션 분류
        for t in texts:
            if any(k.lower() in t.lower() for k in ["talent","인재상","인재","인재상은","people we","who we hire"]):
                out["talent"].append(t)
            if any(k.lower() in t.lower() for k in ["비전","미션","핵심가치","가치","원칙","mission","vision","values","principle"]):
                out["vision"].append(t)
    # 중복 제거 및 최종 정리
    for k in out:
        uniq=[]; s=set()
        for x in out[k]:
            x=x.strip()
            if x and x not in s:
                s.add(x); uniq.append(x[:200])
        out[k]=uniq[:12]
    return out

# 네이버/구글 뉴스 검색을 위한 보조 함수
def _load_naver_keys():
    """네이버 API 키(ID, Secret)를 환경 변수 또는 Streamlit Secret에서 로드합니다."""
    cid = os.getenv("NAVER_CLIENT_ID")
    csec = os.getenv("NAVER_CLIENT_SECRET")
    try:
        if hasattr(st, "secrets"):
            cid = cid or st.secrets.get("NAVER_CLIENT_ID", None)
            csec = csec or st.secrets.get("NAVER_CLIENT_SECRET", None)
    except Exception:
        pass
    return cid, csec

def naver_search_news(company: str, display: int = 5) -> List[Dict]:
    """네이버 검색 API를 사용하여 회사 관련 최신 뉴스 기사를 검색합니다."""
    cid, csec = _load_naver_keys()
    if not (cid and csec): return [] # 키가 없으면 검색 불가
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec}
    try:
        # 네이버 뉴스 API 호출: 정확도(sim) 대신 날짜(date) 최신순 정렬
        r = requests.get(url, headers=headers, params={"query": company, "display": display, "sort":"date"}, timeout=8)
        if r.status_code != 200: return []
        js=r.json()
        items=[]
        # HTML 태그 제거 및 정리
        for it in js.get("items", []):
            title = re.sub(r"</?b>|&quot;|&apos;|&amp;|&lt;|&gt;", "", it.get("title","")).strip()
            items.append({"title": title, "link": it.get("link"), "pubDate": it.get("pubDate")})
        return items
    except Exception:
        return []

def google_news_rss(company: str, max_items: int = 5) -> List[Dict]:
    """네이버 API 실패 시, Google News RSS 피드를 폴백(Fallback)으로 사용합니다."""
    q = urllib.parse.quote(company)
    # Google News RSS URL (한국어, 한국 지역)
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko" 
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200: return []
        soup = BeautifulSoup(r.text, "xml") # RSS는 XML 형식이므로 XML 파서 사용
        out=[]
        for it in soup.find_all("item")[:max_items]:
            out.append({"title": (it.title.get_text() if it.title else "").strip(),
                        "link": (it.link.get_text() if it.link else "").strip(),
                        "pubDate": (it.pubDate.get_text() if it.pubDate else "").strip()})
        return out
    except Exception:
        return []

def fetch_latest_news(company: str, max_items: int = 5) -> List[Dict]:
    """최신 뉴스 검색 (Naver 우선 → Google News 폴백)"""
    items = naver_search_news(company, display=max_items)
    if items: return items
    return google_news_rss(company, max_items=max_items)

# ================== LLM 질문/초안/채점 프롬프트 및 함수 ==================
# AI에게 부여할 역할(시스템 프롬프트) 정의
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
CRITERIA = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"] # 채점 기준

def llm_generate_one_question_with_resume(clean: Dict, level: str, model: str) -> str:
    """회사/직무 정보와 이력서 RAG 검색 결과를 바탕으로 면접 질문 1개를 생성합니다."""
    # RAG: 이력서에서 '핵심 프로젝트와 기술 스택' 관련 조각 4개 발췌
    hits = retrieve_resume_chunks("핵심 프로젝트와 기술 스택 요약", k=4)
    resume_snips = [t for _, t in hits]
    resume_context = "\n".join([f"- {s[:350]}" for s in resume_snips])[:1200]

    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {"role": "user",
                "content": (f"[회사/직무/요건]\n{ctx}\n\n"
                            f"[지원자 이력서 요약(발췌)]\n{resume_context}\n\n"
                            f"[요청]\n- 난이도/연차: {level}\n"
                            f"- 중복/유사도 지양, 회사 요건과 이력서의 교집합 또는 공백영역을 겨냥\n"
                            f"- 한국어 면접 질문 1개만 한 줄로 출력"),}
    try:
        # LLM 호출: 높은 온도(0.85)로 창의적인 질문 유도
        resp = client.chat.completions.create(model=model, temperature=0.85,
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_Q}, user_msg],)
        q = resp.choices[0].message.content.strip()
        # 답변 정리 (번호, 줄바꿈 제거)
        q = re.sub(r"^\s*\d+[\).\s-]*","", q).strip()
        q = q.split("\n")[0].strip()
        return q
    except Exception:
        return ""

def llm_draft_answer(clean: Dict, question: str, model: str) -> str:
    """질문과 이력서 RAG 검색 결과를 바탕으로 STAR 기반 답변 초안을 생성합니다."""
    # RAG: '면접 질문'과 관련된 이력서 조각 4개 발췌
    hits = retrieve_resume_chunks(question, k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {"role": "user",
                "content": (f"[회사/직무/채용요건]\n{ctx}\n\n"
                            f"[지원자 이력서 발췌]\n{resume_text}\n\n"
                            f"[면접 질문]\n{question}\n\n"
                            "위 정보를 근거로 STAR 기반 한국어 답변 **초안**을 작성해줘.")}
    try:
        # LLM 호출: 중간 온도(0.5)로 사실 기반의 논리적 답변 유도
        resp = client.chat.completions.create(model=model, temperature=0.5,
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_DRAFT}, user_msg],)
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

def llm_score_and_coach_strict(clean: Dict, question: str, answer: str, model: str) -> Dict:
    """질문, 답변, 이력서 RAG 검색 결과를 기반으로 엄격한 기준으로 채점하고 피드백을 제공합니다."""
    # RAG: 질문과 답변에 사용된 핵심 키워드 검색
    hits = retrieve_resume_chunks(question + "\n" + answer[:800], k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {"role":"user",
                "content": (f"[회사/직무/채용요건]\n{ctx}\n\n"
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
                            "}")}
    try:
        # LLM 호출: 낮은 온도(0.2)로 정밀하고 일관된 채점 요청
        resp = client.chat.completions.create(model=model, temperature=0.2, response_format={"type":"json_object"}, 
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_SCORE_STRICT}, user_msg])
        data = json.loads(resp.choices[0].message.content)
        
        # LLM이 출력한 JSON 데이터에 대한 후처리 및 유효성 검사
        crit = data.get("criteria", [])
        fixed=[]
        for name in CRITERIA:
            # LLM이 키 이름을 잘못 출력했을 경우를 대비하여 매칭 후 수정
            found=None
            for it in crit:
                if str(it.get("name","")).strip()==name:
                    found=it; break
            if not found: found={"name":name,"score":0,"comment":""}
            sc = int(found.get("score",0)); sc=max(0,min(20,sc)) # 점수 범위 0~20 제한
            found["score"]=sc; found["comment"]=str(found.get("comment","")).strip()
            fixed.append(found)
        total=sum(x["score"] for x in fixed) # 총점은 각 기준 점수의 합계로 재계산
        data["criteria"]=fixed
        data["overall_score"]=total
        
        # 리스트 항목들 정리
        for k in ["strengths","risks","improvements"]:
            arr=data.get(k,[]); 
            if not isinstance(arr,list): arr=[]
            data[k]=[str(x).strip() for x in arr if str(x).strip()][:5]
        data["revised_answer"]=str(data.get("revised_answer","")).strip()
        return data
    except Exception as e:
        # 채점 오류 발생 시 기본 오류 정보 반환
        return {"overall_score": 0,
                "criteria": [{"name": n, "score": 0, "comment": ""} for n in CRITERIA],
                "strengths": [], "risks": [], "improvements": [], "revised_answer": "",
                "error": str(e),}

# ================== 세션 상태 초기화 ==================
def _init_state():
    """Streamlit 앱을 위한 세션 상태(변수)를 초기화합니다."""
    for k, v in {"clean_struct": None,          # 정제된 채용 공고 정보 (JSON)
                 "resume_raw": "",              # 이력서 전체 텍스트
                 "resume_chunks": [],           # 이력서 텍스트 조각(청크) 리스트
                 "resume_embeds": None,         # 이력서 청크의 벡터(임베딩)
                 "current_question": "",        # 현재 면접 질문
                 "answer_text": "",             # 사용자가 작성한 답변
                 "records": [],                 # 면접 기록 저장 (히스토리)
                 "followups": [],               # 팔로업 질문 목록
                 "selected_followup": "",       # 선택된 팔로업 질문
                 "followup_answer": "",         # 팔로업 질문에 대한 답변
                 "last_result": None,           # 메인 질문의 채점 결과
                 "last_followup_result": None,  # 팔로업 질문의 채점 결과
                 "company_home": "",            # 회사 홈페이지 URL
                 "company_vision": [],          # 홈페이지에서 스크래핑한 비전/가치
                 "company_talent": [],          # 홈페이지에서 스크래핑한 인재상
                 "company_news": [] }.items():  # 검색된 최신 뉴스
        if k not in st.session_state: st.session_state[k] = v
_init_state()

# =================================================================================================================
#                                           Streamlit 사용자 인터페이스 (UI) 영역
# =================================================================================================================

# ================== 1) 채용 공고 URL → 정제 섹션 ==================
st.header("1) 채용 공고 URL")
url = st.text_input("채용 공고 상세 URL", placeholder="취업 포털 사이트의 URL을 입력하세요.")

# 회사 공식 홈페이지 URL (선택 입력 → 비전/인재상 수집에 사용)
st.text_input("회사 공식 홈페이지 URL (선택)", key="company_home", placeholder="회사 공식 홈페이지 URL을 입력하세요.")
if st.button("원문 수집 → 정제", type="primary"):
    if not url.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner("원문 수집 중..."):
            # URL에서 텍스트 추출 및 HTML 메타 정보 추출
            raw, meta, soup = fetch_all_text(url.strip())
            hint = extract_company_meta(soup)
        if not raw:
            st.error("원문을 가져오지 못했습니다. (로그인/동적 렌더링/봇 차단 가능)")
        else:
            with st.spinner("LLM으로 정제 중..."):
                # LLM을 사용하여 텍스트를 구조화된 JSON으로 변환
                clean = llm_structurize(raw, hint, CHAT_MODEL)
            
            # LLM이 우대사항을 잘 추출하지 못했을 경우, 규칙 기반 파서 결과를 보조적으로 추가
            if not clean.get("preferences"):
                rb = rule_based_sections(raw)
                if rb.get("preferences"):
                    clean["preferences"] = rb["preferences"][:12]
            st.session_state.clean_struct = clean

            # VISION/NEWS: 회사 비전/인재상/뉴스 수집
            with st.spinner("회사 비전/인재상/뉴스 확인 중..."):
                # 1) 홈페이지에서 비전/인재상 긁기 (선택입력 시)
                vis = []; tal = []
                if st.session_state.company_home.strip():
                    extra = fetch_company_pages(st.session_state.company_home.strip())
                    vis = extra.get("vision", [])
                    tal = extra.get("talent", [])
                # 2) 뉴스: 회사명 기반 검색 (Naver → Google News)
                cname = clean.get("company_name") or hint.get("company_name") or ""
                news_items = fetch_latest_news(cname, max_items=5) if cname else []

                st.session_state.company_vision = vis
                st.session_state.company_talent = tal
                st.session_state.company_news = news_items

            st.success("정제 완료!")

# ================== 2) 회사 요약 (정제 결과) 섹션 ==================
st.header("2) 회사 요약")
clean = st.session_state.clean_struct
if clean:
    st.markdown(f"**회사명:** {clean.get('company_name','-')}")
    st.markdown(f"**간단한 회사 소개(요약):** {clean.get('company_intro','-')}")
    st.markdown(f"**모집 분야(직무명):** {clean.get('job_title','-')}")
    
    # 주요 요건/우대 사항을 3단으로 출력
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

# VISION/NEWS: 회사 비전/인재상/뉴스 (수집된 경우에만 표시)
if st.session_state.company_vision or st.session_state.company_talent or st.session_state.company_news:
    st.divider()
    st.subheader("회사 비전/인재상 & 최신 이슈")  
    colv, colt = st.columns(2)
    with colv:
        st.markdown("**비전/핵심가치 (스크래핑)**")
        for v in st.session_state.company_vision[:8]:
            st.markdown(f"- {v}")
        if not st.session_state.company_vision:
            st.caption("비전/핵심가치 정보를 찾지 못했습니다.")
    with colt:
        st.markdown("**인재상 (스크래핑)**")
        for t in st.session_state.company_talent[:8]:
            st.markdown(f"- {t}")
        if not st.session_state.company_talent:
            st.caption("인재상 정보를 찾지 못했습니다.")

    if st.session_state.company_news:
        st.markdown("**최신 뉴스(상위 3~5건)**")
        for n in st.session_state.company_news[:5]:
            # 뉴스 제목과 링크를 마크다운 형식으로 출력
            st.markdown(f"- [{n.get('title','(제목 없음)')}]({n.get('link','#')})")

st.divider()

# ================== 3) 내 이력서/프로젝트 업로드 및 인덱싱 섹션 ==================
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader("여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True)
_RESUME_CHUNK = 500 # 텍스트 조각(청크)의 크기
_RESUME_OVLP  = 100 # 청크 간 겹치는 부분의 크기

cols_idx = st.columns(2)
with cols_idx[0]:
    if st.button("이력서 인덱싱", type="secondary"):
        if not uploads:
            st.warning("파일을 업로드하세요.")
        else:
            all_text=[]
            for up in uploads:
                t = read_file_text(up) # 파일에서 텍스트 추출
                if t: all_text.append(t)
            resume_text = "\n\n".join(all_text)
            if not resume_text.strip():
                st.error("텍스트를 추출하지 못했습니다.")
            else:
                chunks = chunk(resume_text, size=_RESUME_CHUNK, overlap=_RESUME_OVLP) # 텍스트 청크 분할
                with st.spinner("이력서 벡터화 중..."):
                    embeds = embed_texts(chunks, EMBED_MODEL) # 청크들을 AI 벡터로 변환
                st.session_state.resume_raw = resume_text
                st.session_state.resume_chunks = chunks
                st.session_state.resume_embeds = embeds       # 세션 상태에 저장 (RAG 검색에 사용)
                st.success(f"인덱싱 완료 (청크 {len(chunks)}개)")

st.divider()

# ================== 4) 이력서 기반 자소서 생성 섹션 ==================
st.header("4) 이력서 기반 자소서 생성")
topic = st.text_input("회사 요청 주제(선택)", placeholder="예: 직무 지원동기 / 협업 경험 / 문제해결 사례 등")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str) -> str:
    """LLM을 사용하여 채용 공고, 이력서, 회사 비전/뉴스 등을 결합한 자소서를 생성합니다."""
    # 회사 비전/인재상/뉴스도 프롬프트에 포함하여 반영하도록 함
    enrich = {"vision": st.session_state.company_vision[:6],
              "talent": st.session_state.company_talent[:6],
              "news": [n.get("title","") for n in st.session_state.company_news[:3]]}
    company = json.dumps({"clean":clean_struct, "extra":enrich}, ensure_ascii=False)
    resume_snippet = resume_text.strip()
    if len(resume_snippet) > 9000:
        resume_snippet = resume_snippet[:9000] # 이력서 길이를 제한
        
    system = ("너는 한국어 자기소개서 전문가다. 채용 공고의 회사/직무 요건과 후보자의 이력서를 참고해 "
              "회사 특화 자소서를 작성한다. 과장/허위는 금지하고, 수치/지표/기간/임팩트 중심으로 구체화한다. "
              "회사의 비전/인재상/최근 이슈가 제공되면 자연스럽게 연결하라.")
    
    if topic_hint and topic_hint.strip():
        req = f"회사 측 요청 주제는 '{topic_hint.strip()}' 이다. 이 주제를 중심으로 서술하라."
    else:
        req = "특정 주제 요청이 없으므로, 채용 공고와 비전/인재상을 중심으로 지원동기와 직무적합성을 강조하라."
        
    user = (f"[회사/직무 요약(JSON)]\n{company}\n\n"
            f"[후보자 이력서(요약 가능)]\n{resume_snippet}\n\n"
            f"[작성 지시]\n- {req}\n"
            "- 분량: 600~900자\n"
            "- 구성: 1) 지원 동기 2) 직무 관련 핵심 역량·경험 3) 성과/지표 4) 입사 후 기여 방안 5) 마무리\n"
            "- 자연스럽고 진정성 있는 1인칭 서술. 문장과 문단 가독성을 유지.\n"
            "- 불필요한 미사여구/중복/광고 문구 삭제.")
            
    try:
        # LLM 호출: 온도(0.4)로 일관성 유지하며 창의적 생성
        resp = client.chat.completions.create(model=model, temperature=0.4,
                                              messages=[{"role":"system","content":system},{"role":"user","content":user}])
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
        # 생성된 텍스트 다운로드 버튼 제공
        st.download_button("자소서 TXT 다운로드", data=cover.encode("utf-8"),file_name="cover_letter.txt", mime="text/plain")

st.divider()

# ================== 5) 질문 생성 & 답변 초안 섹션 ==================
st.header("5) 질문 생성 & 답변 초안 (RAG 결합)")
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0) # 질문 난이도 설정

cols_q = st.columns(2)
with cols_q[0]:
    if st.button("새 질문 받기", type="primary"):
        if not st.session_state.clean_struct:
            st.warning("먼저 회사 URL을 정제하세요.")
        else:
            # 질문 생성 함수 호출
            q = llm_generate_one_question_with_resume(st.session_state.clean_struct, level, CHAT_MODEL)
            if q:
                st.session_state.current_question = q
                # 새 질문 생성 시 이전 답변 및 채점 결과 초기화
                st.session_state.answer_text = ""
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
            # 답변 초안 생성 함수 호출
            draft = llm_draft_answer(st.session_state.clean_struct, st.session_state.current_question, CHAT_MODEL)
            if draft:
                st.session_state.answer_text = draft # 생성된 초안을 답변 입력창에 자동 입력
                st.success("초안 생성 완료!")
            else:
                st.error("초안 생성 실패")

st.text_area("질문", value=st.session_state.current_question, height=100)
# 사용자가 답변을 직접 수정하거나 작성하는 영역
ans = st.text_area("나의 답변 (초안을 편집해 완성하세요)", height=200, key="answer_text")

# ================== 6) 채점 & 코칭 (엄격 모드) 섹션 ==================
st.header("6) 채점 & 코칭")
if st.button("채점 & 코칭 실행", type="primary"):
    if not st.session_state.current_question:
        st.warning("먼저 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("채점/코칭 중..."):
            # 채점 및 코칭 함수 호출
            res = llm_score_and_coach_strict(st.session_state.clean_struct,
                                             st.session_state.current_question,
                                             st.session_state.answer_text,
                                             CHAT_MODEL)
        st.session_state.last_result = res
        # 면접 기록을 히스토리에 추가
        st.session_state.records.append({"question": st.session_state.current_question,
                                         "answer": st.session_state.answer_text, 
                                         "overall": res.get("overall_score", 0),
                                         "criteria": res.get("criteria", []),
                                         "strengths": res.get("strengths", []),
                                         "risks": res.get("risks", []),
                                         "improvements": res.get("improvements", []),
                                         "revised_answer": res.get("revised_answer","")})
        st.success("채점/코칭 완료!")

# ================== 7) 피드백 결과 섹션 ==================
st.header("7) 피드백 결과")
last = st.session_state.last_result
if last:
    left, right = st.columns([1,3])
    with left:
        st.metric("총점(/100)", last.get("overall_score", 0)) # 최종 점수 출력
    with right:
        st.markdown("**기준별 점수 & 코멘트**")
        for it in last.get("criteria", []):
            st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}") # 기준별 점수와 코멘트 출력
        
        # 강점, 리스크, 개선 포인트 출력
        if last.get("strengths"):
            st.markdown("**강점**")
            for s in last["strengths"]: st.markdown(f"- {s}")
        if last.get("risks"):
            st.markdown("**감점 요인/리스크**")
            for r in last["risks"]: st.markdown(f"- {r}")
        if last.get("improvements"):
            st.markdown("**개선 포인트**")
            for im in last["improvements"]: st.markdown(f"- {im}")
        
        # LLM이 STAR 구조로 개선한 답변 출력
        if last.get("revised_answer"):
            st.markdown("**수정본 답변 (STAR)**")
            st.write(last["revised_answer"])
else:
    st.info("아직 채점 결과가 없습니다.")

st.divider()

# ================== 8) 팔로업 질문 → 답변 → 팔로업 피드백 섹션 ==================
st.subheader("팔로업 질문 · 답변 · 피드백")
# 메인 피드백이 존재할 때만 팔로업 질문 생성
if last and not st.session_state.followups:
    try:
        # LLM에게 팔로업 질문 3개를 생성하도록 요청
        ctx = json.dumps({"company": st.session_state.clean_struct,
                          "vision": st.session_state.company_vision[:6],
                          "talent": st.session_state.company_talent[:6],
                          "news": [n.get("title","") for n in st.session_state.company_news[:3]]}, ensure_ascii=False)
        msg = {"role":"user",
               "content":(f"[회사/직무/요건/비전/이슈]\n{ctx}\n\n"
                          f"[지원자 답변]\n{st.session_state.answer_text}\n\n"
                          "면접관 관점에서 팔로업 질문 3개를 한 줄씩 한국어로 제안해줘. "
                          "기존 질문과 중복되지 않게, 지표/리스크/트레이드오프/의사결정 근거를 섞어줘.")}
        r = client.chat.completions.create(model=CHAT_MODEL, temperature=0.7,
                                           messages=[{"role":"system","content":"면접 팔로업 생성기"}, msg])
        # 응답에서 질문만 추출하여 정리
        followups = [re.sub(r'^\s*\d+[\).\s-]*','', l).strip()
                     for l in r.choices[0].message.content.splitlines() if l.strip()]
        st.session_state.followups = followups[:3]
    except Exception:
        st.session_state.followups = []

if last:
    if st.session_state.followups:
        st.markdown("**팔로업 질문 제안**")
        for i, f in enumerate(st.session_state.followups, 1):
            st.markdown(f"- ({i}) {f}")

        # 제안된 질문 중 하나를 선택
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
                with st.spinner("팔로업 채점 중..."):
                    # 팔로업 답변에 대한 채점 수행
                    res_fu = llm_score_and_coach_strict(st.session_state.clean_struct, fu_q, fu_ans, CHAT_MODEL)
                st.session_state.last_followup_result = res_fu
                
                # 팔로업 결과 출력
                st.markdown("**팔로업 결과**")
                st.metric("총점(/100)", res_fu.get("overall_score", 0))
                for it in res_fu.get("criteria", []):
                    st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
                if res_fu.get("revised_answer",""):
                    st.markdown("**팔로업 수정본 (STAR)**")
                    st.write(res_fu["revised_answer"])
    else:
        st.caption("팔로업 질문은 메인 질문 채점 직후 자동 제안됩니다.")