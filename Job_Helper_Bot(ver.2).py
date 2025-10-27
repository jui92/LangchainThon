###################################################################################################################
#  1. 채용 포털 사이트 URL로 조회한 회사 정보와 등록한 이력서를 바탕으로 자소서를 자동으로 생성해줍니다                  #
#  2. 채용 포털 사이트 URL / 기업 홈페이지 URL / 뉴스 기사 를 참고하여 모의 면접을 실시하고 답변에 대한 피드백을 해줍니다.#
#  [수정사항] 웹 크롤링을 Selenium(동적) 기반으로 변경하여 JavaScript 렌더링 콘텐츠도 수집 가능                         #
###################################################################################################################

# Library Import ( coding: utf-8 )
import os, re, json, urllib.parse, time, io, random
from typing import Optional, Tuple, Dict, List

# 웹 요청을 위한 라이브러리 (API/RSS용으로 일부 유지)
import requests
# HTML 파싱을 위한 라이브러리
from bs4 import BeautifulSoup
# HTML을 깨끗한 텍스트로 변환하기 위한 라이브러리
import html2text
# 웹 애플리케이션 프레임워크 (UI 구성)
import streamlit as st
# 데이터 처리를 위한 라이브러리
import pandas as pd
import numpy as np

# ================== SELENIUM Library ==================
try:
    # Selenium Library for dynamic scraping
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    # WebDriver 자동 관리를 위한 라이브러리 (Streamlit Cloud 환경에서는 필요에 따라 수동 설정 필요)
    from webdriver_manager.chrome import ChromeDriverManager 
except ImportError:
    st.error("`selenium` 및 `webdriver-manager` 패키지가 필요합니다. pip install을 확인하세요.")
    st.stop()


# ================== 기본 설정 ==================
# Streamlit 페이지 설정: 제목, 아이콘, 레이아웃을 'wide'로 설정
st.set_page_config(page_title="Job Helper Bot", page_icon="🤖", layout="wide")
# 애플리케이션의 메인 제목 표시
st.title("Job Helper Bot : 자소서 생성 / 모의 면접 (Selenium)")

# ================== OpenAI ==================
# OpenAI 라이브러리 임포트 시도
try:
    from openai import OpenAI
except ImportError:
    # 라이브러리가 설치되지 않은 경우 오류 메시지 출력 후 종료
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop()

# API 키를 환경 변수, Streamlit secrets, 또는 사용자 입력창에서 가져옴
API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
if not API_KEY:
    # API 키가 없는 경우 사용자에게 직접 입력 요청 (보안을 위해 password 타입으로)
    API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password")
if not API_KEY:
    # API 키 입력이 완료되지 않으면 앱 실행 중단
    st.stop()
# OpenAI 클라이언트 초기화
client = OpenAI(api_key=API_KEY)

# Streamlit 사이드바에 모델 설정 섹션 추가
with st.sidebar:
    st.subheader("모델 설정")
    # 대화/생성 모델 선택 박스 (사용자가 모델을 선택할 수 있도록 함)
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0)
    # 임베딩 모델 선택 박스 (RAG에 사용될 임베딩 모델 선택)
    EMBED_MODEL = st.selectbox("임베딩 모델(내부용)", ["text-embedding-3-small","text-embedding-3-large"], index=0)

# ================== SELENIUM WebDriver 초기화 ==================
@st.cache_resource
def get_webdriver():
    """Streamlit 환경에 적합한 Headless Chrome WebDriver를 초기화합니다."""
    options = Options()
    # Headless 모드 설정 (브라우저 창을 띄우지 않음)
    options.add_argument("--headless")
    # 봇 감지를 피하기 위한 User-Agent 설정
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    # 창 크기 설정
    options.add_argument("--window-size=1920,1080")
    # 메모리 사용량 최적화 및 안정화 옵션
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-features=NetworkService")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-logging")
    # 언어 설정
    options.add_argument("--lang=ko")

    try:
        # WebDriver Manager를 사용하여 크롬 드라이버 자동 설치 및 로드
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except Exception as e:
        # 드라이버 설치 실패 시 오류 메시지 출력 후 앱 중단
        st.error(f"Selenium WebDriver 초기화 실패: {e}. `webdriver-manager`와 Chrome 설치를 확인하세요.")
        st.stop()
        
    return driver

# 전역 드라이버 변수 초기화 (st.cache_resource로 캐시되어 앱이 재실행되어도 드라이버는 유지됨)
DRIVER = get_webdriver()


# ================== HTTP 유틸 (URL 정규화) ==================
def normalize_url(u: str) -> Optional[str]:
    """URL 문자열을 정규화하고 HTTPS 스키마를 추가합니다."""
    if not u: return None
    u = u.strip()
    # URL에 http 또는 https 스키마가 없으면 https://를 붙임
    if not re.match(r"^https?://", u): u = "https://" + u
    # URL을 파싱하여 쿼리나 프래그먼트 없이 기본 경로만 추출하여 반환
    parts = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


# ================== 동적 원문 수집 (SELENIUM 기반) ==================
def selenium_get_page_source(url: str, wait_selector: Optional[str] = None, timeout: int = 15) -> Optional[str]:
    """Selenium을 사용하여 URL에 접속하고, 동적 로딩을 기다린 후 페이지 소스를 반환합니다."""
    url = normalize_url(url)
    if not url: return None
    try:
        # 1. 페이지 로드
        DRIVER.get(url)
        
        # 2. 동적 콘텐츠가 로드될 때까지 기다림 (옵션)
        # 채용 공고나 회사 소개 페이지의 메인 콘텐츠를 나타내는 흔한 선택자를 기다립니다.
        # body 또는 주요 영역(article, section)이 로드될 때까지 기다립니다.
        wait_selector = wait_selector or "body, article, section, h1, .post, #contents"
        
        WebDriverWait(DRIVER, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
        )
        
        # 3. 최종 렌더링된 페이지 소스 반환
        return DRIVER.page_source
        
    except Exception as e:
        # 타임아웃 또는 기타 오류 발생 시
        print(f"Selenium get error for {url}: {e}")
        return None

def fetch_all_text(url: str):
    """최종 웹 콘텐츠 추출 함수: Selenium을 사용하여 렌더링된 HTML을 가져와 BS4로 파싱합니다."""
    url = normalize_url(url)
    if not url: return "", {"error":"invalid_url"}, None

    # 1. Selenium으로 페이지 소스 가져오기
    html_source = selenium_get_page_source(url)
        
    if not html_source:
        return "", {"source":"selenium_fail","error":"fail_to_get_source","url_final":url}, None

    # 2. BeautifulSoup으로 HTML 파싱 및 텍스트 추출
    soup = BeautifulSoup(html_source, "lxml")
    blocks = []
    # 기사, 섹션, 메인 등 주요 콘텐츠 태그를 순회하며 텍스트 추출 시도
    for sel in ["article","section","main","div","ul","ol"]:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True)
            # 길이가 300자 이상인 유의미한 텍스트 블록만 수집
            if txt and len(txt) > 300:
                txt = re.sub(r"\s+"," ", txt)
                blocks.append(txt)
    
    # 3. 텍스트 정리 및 반환
    if not blocks:
        # 유의미한 블록을 찾지 못했다면 전체 텍스트를 제한 길이(120000)만큼 반환
        raw_text = soup.get_text(" ", strip=True)[:120000]
    else:
        # 중복 블록 제거 후 텍스트를 합쳐서 반환
        seen, out = set(), []
        for b in blocks:
            if b not in seen:
                seen.add(b); out.append(b)
        raw_text = ("\n\n".join(out)[:120000])

    return raw_text, {"source":"selenium_bs4","len":len(raw_text),"url_final":url}, soup


# ================== 메타/섹션 보조 추출 ==================
def extract_company_meta(soup: Optional[BeautifulSoup]) -> Dict[str,str]:
    """HTML Soup 객체에서 회사명, 회사 소개, 직무명 후보를 추출합니다."""
    meta = {"company_name":"","company_intro":"","job_title":""}
    if not soup: return meta
    
    # 1. 회사명 후보 추출 (og:site_name, application-name, title 태그에서)
    cand = []
    og = soup.find("meta", {"property":"og:site_name"})
    if og and og.get("content"): cand.append(og["content"])
    app = soup.find("meta", {"name":"application-name"})
    if app and app.get("content"): cand.append(app["content"])
    if soup.title and soup.title.string: cand.append(soup.title.string)
    
    # 특수 문자(하이픈, 파이프 등)를 기준으로 회사명을 분리하고 정리
    cand = [re.split(r"[\-\|\·\—]", c)[0].strip() for c in cand if c]
    cand = [c for c in cand if 2 <= len(c) <= 40]
    meta["company_name"] = cand[0] if cand else ""
    
    # 2. 회사 소개/설명 추출 (description, og:description 메타 태그에서)
    md = soup.find("meta", {"name":"description"}) or soup.find("meta", {"property":"og:description"})
    if md and md.get("content"):
        meta["company_intro"] = re.sub(r"\s+"," ", md["content"]).strip()[:500]
        
    # 3. 직무명 추출 (og:title, H1, H2 태그에서)
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

# ================== 규칙 파서 ==================
def rule_based_sections(raw_text: str) -> dict:
    """정규 표현식을 기반으로 원문에서 주요 업무, 자격 요건, 우대 사항을 추출합니다. (LLM 보조용)"""
    txt = re.sub(r"\r", "", raw_text or "").strip()
    # 줄바꿈을 기준으로 분리하고 특수 문자 제거 및 공백 정리
    lines = [re.sub(r"\s+", " ", l).strip(" -•·▶▪️") for l in txt.split("\n")]
    lines = [l for l in lines if l]

    # 각 섹션의 헤더 정규 표현식 정의
    hdr_resp = re.compile(r"(주요\s*업무|담당\s*업무|Role|Responsibilities?)", re.I)
    hdr_qual = re.compile(r"(자격\s*요건|지원\s*자격|Requirements?|Qualifications?)", re.I)
    hdr_pref = re.compile(r"(우대\s*사항|우대|선호|Preferred|Nice\s*to\s*have|Plus)", re.I)

    bucket = None
    out = {"responsibilities": [], "qualifications": [], "preferences": []}

    def push(line, b):
        """추출된 항목을 버킷에 추가하고 중복을 방지합니다."""
        if line and len(line) > 1 and line not in out[b]:
            out[b].append(line[:180])

    # 각 줄을 순회하며 헤더를 기준으로 버킷(섹션)을 지정
    for l in lines:
        if hdr_resp.search(l): bucket = "responsibilities"; continue
        if hdr_qual.search(l): bucket = "qualifications"; continue
        if hdr_pref.search(l): bucket = "preferences"; continue
        
        # 헤더가 명확하지 않은 경우, 키워드로 섹션을 추정하여 다음 항목부터 해당 섹션으로 분류
        if bucket is None:
            if hdr_pref.search(l):
                bucket = "preferences"
            elif any(k in l.lower() for k in ["java","python","spark","airflow","kafka","ml","sql"]):
                bucket = "responsibilities" # 기술 스택이 언급되면 주요 업무로 간주
            else:
                continue # 관련 없는 항목은 무시
        push(l, bucket)
    
    # 2단계 정리: '자격 요건'에 포함된 우대 키워드 항목을 '우대 사항'으로 이동
    kw_pref = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
    remain_qual = []
    for q in out["qualifications"]:
        if kw_pref.search(q):
            out["preferences"].append(q)
        else:
            remain_qual.append(q)
    out["qualifications"] = remain_qual

    # 최종 정리: 각 섹션별 중복 제거 및 항목 길이 제한 (최대 12개)
    for k in out:
        seen=set(); clean=[]
        for s in out[k]:
            s = re.sub(r"\s+", " ", s).strip(" -•·▶▪️").strip()
            if s and s not in seen:
                seen.add(s); clean.append(s)
        out[k] = clean[:12]
    return out

# ================== LLM 정제 (채용 공고 → 구조 JSON) ==================
# LLM에게 부여할 시스템 역할 프롬프트
PROMPT_SYSTEM_STRUCT = ("너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
                        "입력 텍스트는 포털 광고 문구, UI잔재, 복수 직무가 섞여 있을 수 있다. "
                        "한국어로 간결하고 중복없이 정제하라.")

def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
    """LLM을 사용하여 채용 공고 원문을 JSON 구조로 정제합니다."""
    ctx = (raw_text or "").strip()
    if len(ctx) > 14000: # 컨텍스트 길이 제한
        ctx = ctx[:14000]

    # 사용자 메시지 구성: 원문과 메타 힌트를 제공하고 특정 JSON 스키마를 요청
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
        # LLM 호출: JSON 출력 형식 강제 (response_format)
        resp = client.chat.completions.create(model=model, temperature=0.2,
                                              response_format={"type": "json_object"},
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg])
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        # LLM 호출 실패 시, 힌트 메타데이터를 기반으로 기본 구조를 채워 반환
        data = {"company_name": meta_hint.get("company_name",""),
                "company_intro": meta_hint.get("company_intro","원문이 정제되지 않았습니다."),
                "job_title": meta_hint.get("job_title",""),
                "responsibilities": [],
                "qualifications": [],
                "preferences": [],
                "error": str(e)}

    # 후처리: LLM이 생성한 배열 항목들을 정리 (공백/특수문자 제거, 길이 제한, 중복 제거)
    for k in ["responsibilities","qualifications","preferences"]:
        arr = data.get(k, [])
        if not isinstance(arr, list): arr = []
        clean_list=[]; seen=set()
        for it in arr:
            t = re.sub(r"\s+"," ", str(it)).strip(" -•·▶▪️").strip()
            if t and t not in seen:
                seen.add(t); clean_list.append(t[:180])
        data[k] = clean_list[:12]

    # 후처리: 회사명/소개/직무명 문자열 정리
    for k in ["company_name","company_intro","job_title"]:
        if k in data and isinstance(data[k], str):
            data[k] = re.sub(r"\s+"," ", data[k]).strip()

    # 우대 사항(preferences)이 비어 있을 경우, Rule-Based Parser 결과를 병합 시도
    if len(data.get("preferences", [])) < 1:
        rb = rule_based_sections(ctx) # 규칙 기반 파서 재실행
        if rb.get("preferences"):
            merged = data.get("preferences", []) + rb["preferences"]
            seen=set(); pref=[]
            for s in merged:
                s=s.strip()
                if s and s not in seen:
                    seen.add(s); pref.append(s)
            data["preferences"] = pref[:12]
        # 최종적으로 우대 사항이 없으면, 자격 요건에서 '우대' 키워드가 포함된 항목을 이동
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
# PDF 읽기 라이브러리(pypdf) 동적 임포트 시도
try:
    import pypdf
except Exception:
    pypdf = None

def read_pdf(data: bytes) -> str:
    """PDF 파일의 내용을 텍스트로 추출합니다."""
    if pypdf is None: return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        # 페이지별 텍스트를 추출하여 합침
        return "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
    except Exception:
        return ""

def read_docx(data: bytes) -> str:
    """DOCX 파일의 내용을 텍스트로 추출합니다."""
    try:
        import docx2txt, tempfile
        # docx2txt 사용을 위해 임시 파일로 저장 후 처리
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
            tmp.write(data); tmp.flush()
            text = docx2txt.process(tmp.name) or ""
            return text
    except Exception:
        return ""

def read_file_text(uploaded) -> str:
    """업로드된 파일의 MIME 타입에 따라 적절한 리더를 사용하여 텍스트를 추출합니다."""
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith((".txt",".md")):
        # 텍스트 파일: 여러 인코딩(utf-8, cp949, euc-kr)으로 디코딩 시도
        for enc in ("utf-8","cp949","euc-kr"):
            try: return data.decode(enc)
            except Exception: continue
        return data.decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        return read_pdf(data)
    elif name.endswith(".docx"):
        return read_docx(data)
    return ""

# ================== 간단 청크/임베딩(내부 자동) ==================
def chunk(text: str, size: int = 600, overlap: int = 120) -> List[str]:
    """텍스트를 지정된 크기와 오버랩으로 청크(조각)로 분할합니다."""
    t = re.sub(r"\s+"," ", text).strip()
    if not t: return []
    out, start = [], 0
    # 텍스트 끝까지 반복하며 청크 생성
    while start < len(t):
        end = min(len(t), start+size)
        out.append(t[start:end])
        if end == len(t): break
        start = max(0, end-overlap) # 다음 청크 시작점을 오버랩만큼 뒤로 설정
    return out

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    """주어진 텍스트 리스트를 임베딩 벡터로 변환합니다."""
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32) # 빈 리스트 처리
    # OpenAI API를 호출하여 임베딩 생성
    resp = client.embeddings.create(model=model_name, input=texts)
    # NumPy 배열로 변환
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs

def cosine_topk(matrix: np.ndarray, query_vec: np.ndarray, k: int = 4):
    """코사인 유사도를 계산하여 가장 유사한 상위 K개 항목의 점수와 인덱스를 반환합니다."""
    if matrix.size == 0:
        return np.array([]), np.array([], dtype=int)
    # 쿼리 벡터와 행렬을 정규화 (길이 1로)
    qn = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    # 행렬 곱셈으로 코사인 유사도 계산
    sims = mn @ qn.T
    sims = sims.reshape(-1)
    # 유사도 점수가 높은 순서대로 상위 K개의 인덱스 추출
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx

def retrieve_resume_chunks(query: str, k: int = 4):
    """RAG (검색 증강 생성)를 위해 쿼리와 가장 유사한 이력서 청크를 검색합니다."""
    # 세션 상태에서 이력서 청크와 임베딩 벡터를 가져옴
    chs, embs = st.session_state.get("resume_chunks", []), st.session_state.get("resume_embeds", None)
    if not chs or embs is None:
        return []
    # 쿼리 문장 임베딩
    qv = embed_texts([query], EMBED_MODEL)
    # 상위 K개 유사 청크 검색
    scores, idxs = cosine_topk(embs, qv, k=k)
    # 점수와 청크 텍스트를 튜플 리스트로 반환
    return [(float(s), chs[int(i)]) for s, i in zip(scores, idxs)]

# ================== 회사 비전/인재상/뉴스 (SELENIUM 적용) ==================
VISION_KEYS = ["비전","미션","핵심가치","가치","원칙","문화","행동강령","Talent","인재상","Our Mission","Vision","Values"]

def safe_get_text(el) -> str:
    """BeautifulSoup 요소에서 안전하게 텍스트를 추출합니다."""
    try:
        return el.get_text(" ", strip=True)
    except Exception:
        return ""

def fetch_company_pages(home_url: str) -> Dict[str, List[str]]:
    """홈페이지 및 대표 서브경로에서 비전/인재상 후보 텍스트를 수집합니다. (Selenium 사용)"""
    out = {"vision": [], "talent": []}
    if not home_url: return out
    base = normalize_url(home_url)
    if not base: return out
    # 일반적으로 비전/인재상이 있는 서브 경로 리스트
    paths = ["","/","/about","/company","/about-us","/mission","/vision","/values","/culture","/careers","/talent","/people"]
    seen = set()
    for p in paths:
        url = (base.rstrip("/") + p) if p else base
        if url in seen: continue
        seen.add(url)
        
        # 1. Selenium으로 페이지 소스 가져오기
        html_source = selenium_get_page_source(url, timeout=8)
        if not html_source: continue
        
        soup = BeautifulSoup(html_source, "lxml") # HTML 파싱
        texts=[]
        # H 태그(제목)와 P, LI 태그에서 텍스트 추출
        for tag in soup.find_all(["h1","h2","h3","h4","p","li"]):
            t = safe_get_text(tag)
            if not t: continue
            t = re.sub(r"\s+"," ", t)
            if 6 <= len(t) <= 260:
                texts.append(t)
        # 키워드 매칭을 통해 vision 또는 talent로 분류
        for t in texts:
            if any(k.lower() in t.lower() for k in ["talent","인재상","인재","인재상은","people we","who we hire"]):
                out["talent"].append(t)
            if any(k.lower() in t.lower() for k in ["비전","미션","핵심가치","가치","원칙","mission","vision","values","principle"]):
                out["vision"].append(t)
    # 결과 정리/중복 제거/길이 제한
    for k in out:
        uniq=[]; s=set()
        for x in out[k]:
            x=x.strip()
            if x and x not in s:
                s.add(x); uniq.append(x[:200])
        out[k]=uniq[:12]
    return out

# NAVER NEWS → Google News RSS 폴백 (requests 사용)
def _load_naver_keys():
    """네이버 검색 API 키를 환경 변수 또는 secrets에서 로드합니다."""
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
    """네이버 뉴스 검색 API를 사용하여 회사 관련 최신 뉴스를 검색합니다. (requests 사용)"""
    cid, csec = _load_naver_keys()
    if not (cid and csec): return [] # API 키 없으면 반환
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec}
    try:
        # 쿼리: 회사명, 정렬: 날짜순, 표시 개수 제한
        r = requests.get(url, headers=headers, params={"query": company, "display": display, "sort":"date"}, timeout=8)
        if r.status_code != 200: return []
        js=r.json()
        items=[]
        for it in js.get("items", []):
            # HTML 태그 및 엔티티 제거 후 제목 정리
            title = re.sub(r"</?b>|&quot;|&apos;|&amp;|&lt;|&gt;", "", it.get("title","")).strip()
            items.append({"title": title, "link": it.get("link"), "pubDate": it.get("pubDate")})
        return items
    except Exception:
        return []

def google_news_rss(company: str, max_items: int = 5) -> List[Dict]:
    """네이버 API 실패 시, Google News RSS 피드를 폴백으로 사용합니다. (requests 사용)"""
    q = urllib.parse.quote(company)
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko" # 한국어, 한국 기준 RSS
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200: return []
        soup = BeautifulSoup(r.text, "xml") # XML 파싱
        out=[]
        # RSS <item> 태그에서 제목, 링크, 날짜 추출
        for it in soup.find_all("item")[:max_items]:
            out.append({"title": (it.title.get_text() if it.title else "").strip(),
                        "link": (it.link.get_text() if it.link else "").strip(),
                        "pubDate": (it.pubDate.get_text() if it.pubDate else "").strip()})
        return out
    except Exception:
        return []

def fetch_latest_news(company: str, max_items: int = 5) -> List[Dict]:
    """최신 뉴스 검색을 시도하고, 네이버 실패 시 Google로 폴백합니다."""
    items = naver_search_news(company, display=max_items)
    if items: return items
    return google_news_rss(company, max_items=max_items)

# ================== 질문/초안/채점/팔로업 프롬프트 ==================
# 면접 질문 생성 시스템 프롬프트
PROMPT_SYSTEM_Q = ("너는 채용담당자다. 회사/직무 맥락과 채용요건, 그리고 지원자의 이력서 요약을 함께 고려해 "
                   "면접 질문을 한국어로 생성한다. 질문은 서로 형태·관점·키워드가 겹치지 않게 다양화하고, "
                   "수치/지표/기간/규모/리스크/트레이드오프 등도 섞어라.")
# 답변 초안 생성 시스템 프롬프트 (STAR 기법 강조)
PROMPT_SYSTEM_DRAFT = ("너는 면접 답변 코치다. 회사/직무/채용요건과 지원자의 이력서 요약을 결합해 "
                       "질문에 대한 답변 **초안**을 STAR(상황-과제-행동-성과)로 8~12문장, 한국어로 작성한다. "
                       "가능하면 구체적인 지표/수치/기간/임팩트를 포함하라.")
# 채점 시스템 프롬프트 (엄격한 기준, JSON 출력 강제)
PROMPT_SYSTEM_SCORE_STRICT = ("너는 매우 엄격한 톱티어 면접 코치다. 아래 형식의 JSON만 출력하라. "
                              "각 기준은 0~20 정수이며, 총점은 기준 합계(최대 100)와 반드시 일치해야 한다. "
                              "과장/모호함/근거 부재/숫자 없는 주장/책임 회피/모호한 주어 사용 등을 강하게 감점하라. "
                              "각 기준에 대해 짧지만 구체적 코멘트(강점/감점요인/개선포인트)를 제공하라.")
# 채점 기준 항목
CRITERIA = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def llm_generate_one_question_with_resume(clean: Dict, level: str, model: str) -> str:
    """RAG를 활용하여 이력서 내용을 기반으로 면접 질문 하나를 생성합니다."""
    # RAG: 이력서에서 '핵심 프로젝트와 기술 스택' 관련 청크를 검색
    hits = retrieve_resume_chunks("핵심 프로젝트와 기술 스택 요약", k=4)
    resume_snips = [t for _, t in hits]
    resume_context = "\n".join([f"- {s[:350]}" for s in resume_snips])[:1200]

    ctx = json.dumps(clean, ensure_ascii=False)
    # 사용자 메시지: 회사 정보, 이력서 발췌 내용, 요청 사항 전달
    user_msg = {"role": "user",
                "content": (f"[회사/직무/요건]\n{ctx}\n\n"
                            f"[지원자 이력서 요약(발췌)]\n{resume_context}\n\n"
                            f"[요청]\n- 난이도/연차: {level}\n"
                            f"- 중복/유사도 지양, 회사 요건과 이력서의 교집합 또는 공백영역을 겨냥\n"
                            f"- 한국어 면접 질문 1개만 한 줄로 출력"),}
    try:
        # LLM 호출: 높은 temperature(0.85)로 창의적인 질문 유도
        resp = client.chat.completions.create(model=model, temperature=0.85,
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_Q}, user_msg],)
        q = resp.choices[0].message.content.strip()
        # 불필요한 번호나 줄바꿈 제거
        q = re.sub(r"^\s*\d+[\).\s-]*","", q).strip()
        q = q.split("\n")[0].strip()
        return q
    except Exception:
        return ""

def llm_draft_answer(clean: Dict, question: str, model: str) -> str:
    """RAG를 활용하여 질문에 대한 STAR 기반 답변 초안을 생성합니다."""
    # RAG: 질문과 가장 유사한 이력서 청크를 검색하여 컨텍스트로 제공
    hits = retrieve_resume_chunks(question, k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    ctx = json.dumps(clean, ensure_ascii=False)
    # 사용자 메시지: 회사 정보, 이력서 발췌, 질문 전달, STAR 기반 초안 요청
    user_msg = {"role": "user",
                "content": (f"[회사/직무/채용요건]\n{ctx}\n\n"
                            f"[지원자 이력서 발췌]\n{resume_text}\n\n"
                            f"[면접 질문]\n{question}\n\n"
                            "위 정보를 근거로 STAR 기반 한국어 답변 **초안**을 작성해줘.")}
    try:
        # LLM 호출: 중간 temperature(0.5)로 사실 기반의 논리적 답변 유도
        resp = client.chat.completions.create(model=model, temperature=0.5,
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_DRAFT}, user_msg],)
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

def llm_score_and_coach_strict(clean: Dict, question: str, answer: str, model: str) -> Dict:
    """지원자의 답변을 엄격한 기준으로 채점하고 상세 코칭을 JSON 형식으로 반환합니다."""
    # RAG: 질문과 답변에 관련된 이력서 청크를 검색하여 근거 확인에 활용
    hits = retrieve_resume_chunks(question + "\n" + answer[:800], k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    ctx = json.dumps(clean, ensure_ascii=False)
    
    # 사용자 메시지: 회사 정보, 이력서 발췌, 질문, 답변 제공, 특정 JSON 스키마 강제 요청
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
        # LLM 호출: 낮은 temperature(0.2)로 정해진 규칙에 따른 논리적이고 일관적인 채점 유도, JSON 출력 강제
        resp = client.chat.completions.create(model=model, temperature=0.2, response_format={"type":"json_object"}, 
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_SCORE_STRICT}, user_msg])
        data = json.loads(resp.choices[0].message.content)
        
        # 후처리: JSON 데이터의 무결성 및 형식 검사/수정
        crit = data.get("criteria", [])
        fixed=[]
        for name in CRITERIA:
            found=None
            for it in crit:
                if str(it.get("name","")).strip()==name: found=it; break
            if not found: found={"name":name,"score":0,"comment":""}
            # 점수 범위 강제 (0~20)
            sc = int(found.get("score",0)); sc=max(0,min(20,sc))
            found["score"]=sc; found["comment"]=str(found.get("comment","")).strip()
            fixed.append(found)
        total=sum(x["score"] for x in fixed) # 총점 재계산
        data["criteria"]=fixed
        data["overall_score"]=total
        
        # 강점, 리스크, 개선점 항목 정리
        for k in ["strengths","risks","improvements"]:
            arr=data.get(k,[]); 
            if not isinstance(arr,list): arr=[]
            data[k]=[str(x).strip() for x in arr if str(x).strip()][:5]
        data["revised_answer"]=str(data.get("revised_answer","")).strip()
        return data
    except Exception as e:
        # 오류 발생 시 기본 오류 응답 반환
        return {"overall_score": 0,
                "criteria": [{"name": n, "score": 0, "comment": ""} for n in CRITERIA],
                "strengths": [], "risks": [], "improvements": [], "revised_answer": "",
                "error": str(e),}

# ================== 세션 상태 ==================
def _init_state():
    """Streamlit 세션 상태(st.session_state) 변수들을 초기화합니다."""
    for k, v in {"clean_struct": None,         # LLM이 정제한 채용 공고 구조화 JSON
                 "resume_raw": "",             # 업로드된 이력서의 원문 텍스트
                 "resume_chunks": [],          # 이력서 원문이 분할된 청크 리스트
                 "resume_embeds": None,        # 이력서 청크의 임베딩 벡터 배열
                 "current_question": "",       # 현재 모의 면접 질문
                 "answer_text": "",            # 사용자가 입력/편집한 답변
                 "records": [],                # 면접 기록 (히스토리)
                 "followups": [],              # 팔로업 질문 제안 리스트
                 "selected_followup": "",      # 선택된 팔로업 질문
                 "followup_answer": "",        # 팔로업 질문에 대한 답변
                 "last_result": None,          # 메인 질문의 마지막 채점 결과
                 "last_followup_result": None, # 팔로업 질문의 마지막 채점 결과
                 "company_home": "",           # 회사 홈페이지 URL
                 "company_vision": [],         # 스크래핑된 회사 비전/가치
                 "company_talent": [],         # 스크래핑된 회사 인재상
                 "company_news": [] }.items(): # 검색된 회사 최신 뉴스
        if k not in st.session_state: st.session_state[k] = v
_init_state()

# ================== 1) 채용 공고 URL → 정제 ==================
st.header("1) 채용 공고 URL")
# 채용 공고 URL 입력 필드
url = st.text_input("채용 공고 상세 URL", placeholder="취업 포털 사이트의 URL을 입력하세요.")

# 회사 공식 홈페이지 URL 입력 (비전/인재상 스크래핑에 사용)
st.text_input("회사 공식 홈페이지 URL (선택)", key="company_home", placeholder="회사 공식 홈페이지 URL을 입력하세요.")
# 정제 시작 버튼
if st.button("원문 수집 → 정제 (Selenium 실행)", type="primary"):
    if not url.strip():
        st.warning("URL을 입력하세요.")
    else:
        # 1단계: 원문 수집 (Selenium 기반) 및 메타데이터 힌트 추출
        with st.spinner("Selenium으로 원문 수집 및 렌더링 중..."):
            # fetch_all_text 함수는 이제 Selenium을 사용하여 동적으로 콘텐츠를 가져옵니다.
            raw, meta, soup = fetch_all_text(url.strip()) 
            hint = extract_company_meta(soup)
        if not raw:
            st.error(f"원문을 가져오지 못했습니다. (Selenium 렌더링 실패: {meta.get('error', '알 수 없음')})")
        else:
            # 2단계: LLM을 사용한 구조화 정제
            with st.spinner("LLM으로 정제 중..."):
                clean = llm_structurize(raw, hint, CHAT_MODEL)
            # LLM 결과에 우대 사항이 없으면 규칙 기반 파서 결과를 병합하여 보완
            if not clean.get("preferences"):
                rb = rule_based_sections(raw)
                if rb.get("preferences"):
                    clean["preferences"] = rb["preferences"][:12]
            st.session_state.clean_struct = clean # 정제 결과 세션 저장

            # 3단계: 회사 비전/인재상/뉴스 추가 수집 (Selenium 사용)
            with st.spinner("회사 비전/인재상/뉴스 확인 중..."):
                # fetch_company_pages 함수도 Selenium을 사용하여 동적으로 콘텐츠를 가져옵니다.
                vis = []; tal = []
                if st.session_state.company_home.strip():
                    extra = fetch_company_pages(st.session_state.company_home.strip())
                    vis = extra.get("vision", [])
                    tal = extra.get("talent", [])
                # 최신 뉴스 검색 (requests/API 사용)
                cname = clean.get("company_name") or hint.get("company_name") or ""
                news_items = fetch_latest_news(cname, max_items=5) if cname else []

                st.session_state.company_vision = vis
                st.session_state.company_talent = tal
                st.session_state.company_news = news_items

            st.success("정제 완료!")

# ================== 2) 회사 요약 (정제 결과) ==================
st.header("2) 회사 요약")
clean = st.session_state.clean_struct
if clean:
    # 정제된 회사/직무 정보를 표시
    st.markdown(f"**회사명:** {clean.get('company_name','-')}")
    st.markdown(f"**간단한 회사 소개(요약):** {clean.get('company_intro','-')}")
    st.markdown(f"**모집 분야(직무명):** {clean.get('job_title','-')}")
    # 업무, 자격, 우대 사항을 3단 컬럼으로 분리하여 표시
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

    # 추가 정보 (비전/인재상/뉴스) 표시
    if st.session_state.company_vision or st.session_state.company_talent or st.session_state.company_news:
        st.divider()
        st.subheader("회사 비전/인재상 & 최신 이슈")  
        colv, colt = st.columns(2)
        with colv:
            st.markdown("**비전/핵심가치 (Selenium 스크래핑)**")
            for v in st.session_state.company_vision[:8]:
                st.markdown(f"- {v}")
            if not st.session_state.company_vision:
                st.caption("비전/핵심가치 정보를 찾지 못했습니다.")
        with colt:
            st.markdown("**인재상 (Selenium 스크래핑)**")
            for t in st.session_state.company_talent[:8]:
                st.markdown(f"- {t}")
            if not st.session_state.company_talent:
                st.caption("인재상 정보를 찾지 못했습니다.")

        if st.session_state.company_news:
            st.markdown("**최신 뉴스(상위 3~5건)**")
            for n in st.session_state.company_news[:5]:
                # 뉴스 제목과 링크를 마크다운으로 표시
                st.markdown(f"- [{n.get('title','(제목 없음)')}]({n.get('link','#')})")

else:
    st.info("먼저 URL을 정제해 주세요.")

st.divider()

# ================== 3) 내 이력서/프로젝트 업로드 (PDF/TXT/MD/DOCX) ==================
st.header("3) 내 이력서/프로젝트 업로드")
# 이력서 파일 업로드 위젯 (복수 파일 지원)
uploads = st.file_uploader("여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True)
# RAG 청크 크기 및 오버랩 설정
_RESUME_CHUNK = 500
_RESUME_OVLP  = 100

cols_idx = st.columns(2)
with cols_idx[0]:
    # 이력서 인덱싱 버튼 (텍스트 추출 및 임베딩 벡터 생성)
    if st.button("이력서 인덱싱", type="secondary"):
        if not uploads:
            st.warning("파일을 업로드하세요.")
        else:
            all_text=[]
            # 각 파일에서 텍스트 추출
            for up in uploads:
                t = read_file_text(up)
                if t: all_text.append(t)
            resume_text = "\n\n".join(all_text)
            if not resume_text.strip():
                st.error("텍스트를 추출하지 못했습니다.")
            else:
                # 텍스트를 청크로 분할
                chunks = chunk(resume_text, size=_RESUME_CHUNK, overlap=_RESUME_OVLP)
                # 청크를 임베딩 벡터로 변환 (RAG 준비)
                with st.spinner("이력서 벡터화 중..."):
                    embeds = embed_texts(chunks, EMBED_MODEL)
                # 세션 상태에 저장
                st.session_state.resume_raw = resume_text
                st.session_state.resume_chunks = chunks
                st.session_state.resume_embeds = embeds
                st.success(f"인덱싱 완료 (청크 {len(chunks)}개)")

st.divider()

# ================== 4) 이력서 기반 자소서 생성 ==================
st.header("4) 이력서 기반 자소서 생성")
# 자소서 주제 입력 필드 (선택 사항)
topic = st.text_input("회사 요청 주제(선택)", placeholder="예: 직무 지원동기 / 협업 경험 / 문제해결 사례 등")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str) -> str:
    """LLM을 사용하여 채용 공고, 회사 비전/뉴스, 이력서를 기반으로 자소서를 생성합니다."""
    # LLM 컨텍스트에 회사 비전, 인재상, 뉴스 정보를 추가
    enrich = {"vision": st.session_state.company_vision[:6],
              "talent": st.session_state.company_talent[:6],
              "news": [n.get("title","") for n in st.session_state.company_news[:3]]}
    company = json.dumps({"clean":clean_struct, "extra":enrich}, ensure_ascii=False)
    
    # 이력서 텍스트 길이 제한
    resume_snippet = resume_text.strip()
    if len(resume_snippet) > 9000:
        resume_snippet = resume_snippet[:9000]
        
    # 시스템 역할 프롬프트
    system = ("너는 한국어 자기소개서 전문가다. 채용 공고의 회사/직무 요건과 후보자의 이력서를 참고해 "
              "회사 특화 자소서를 작성한다. 과장/허위는 금지하고, 수치/지표/기간/임팩트 중심으로 구체화한다. "
              "회사의 비전/인재상/최근 이슈가 제공되면 자연스럽게 연결하라.")
              
    # 주제 요청에 따른 조건 분기
    if topic_hint and topic_hint.strip():
        req = f"회사 측 요청 주제는 '{topic_hint.strip()}' 이다. 이 주제를 중심으로 서술하라."
    else:
        req = "특정 주제 요청이 없으므로, 채용 공고와 비전/인재상을 중심으로 지원동기와 직무적합성을 강조하라."
        
    # 사용자 메시지 구성: 모든 정보를 포함하고 구체적인 형식 가이드라인 제시
    user = (f"[회사/직무 요약(JSON)]\n{company}\n\n"
            f"[후보자 이력서(요약 가능)]\n{resume_snippet}\n\n"
            f"[작성 지시]\n- {req}\n"
            "- 분량: 600~900자\n"
            "- 구성: 1) 지원 동기 2) 직무 관련 핵심 역량·경험 3) 성과/지표 4) 입사 후 기여 방안 5) 마무리\n"
            "- 자연스럽고 진정성 있는 1인칭 서술. 문장과 문단 가독성을 유지.\n"
            "- 불필요한 미사여구/중복/광고 문구 삭제.")
            
    try:
        # LLM 호출: 답변의 일관성과 창의성 사이에서 균형 (temperature 0.4)
        resp = client.chat.completions.create(model=model, temperature=0.4,
                                              messages=[{"role":"system","content":system},{"role":"user","content":user}])
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(자소서 생성 실패: {e})"

# 자소서 생성 버튼
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
        # 생성된 자소서를 다운로드할 수 있는 버튼 제공
        st.download_button("자소서 TXT 다운로드", data=cover.encode("utf-8"),file_name="cover_letter.txt", mime="text/plain")

st.divider()

# ================== 5) 질문 생성 & 답변 초안 (RAG 결합) ==================
st.header("5) 질문 생성 & 답변 초안 (RAG 결합)")
# 면접 질문 난이도/연차 선택
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)

cols_q = st.columns(2)
with cols_q[0]:
    # 새 질문 받기 버튼
    if st.button("새 질문 받기", type="primary"):
        if not st.session_state.clean_struct:
            st.warning("먼저 회사 URL을 정제하세요.")
        else:
            # 질문 생성 함수 호출
            q = llm_generate_one_question_with_resume(st.session_state.clean_struct, level, CHAT_MODEL)
            if q:
                # 새로운 질문/답변/결과로 세션 상태 초기화 및 질문 저장
                st.session_state.current_question = q
                st.session_state.answer_text = ""
                st.session_state.last_result = None
                st.session_state.followups = []
                st.session_state.selected_followup = ""
                st.session_state.followup_answer = ""
                st.success("질문 생성 완료!")
            else:
                st.error("질문 생성 실패")
with cols_q[1]:
    # RAG로 답변 초안 생성 버튼
    if st.button("RAG로 답변 초안 생성", type="secondary"):
        if not st.session_state.current_question:
            st.warning("먼저 질문을 생성하세요.")
        else:
            # 답변 초안 생성 함수 호출
            draft = llm_draft_answer(st.session_state.clean_struct, st.session_state.current_question, CHAT_MODEL)
            if draft:
                st.session_state.answer_text = draft # 초안을 답변 텍스트 영역에 자동 입력
                st.success("초안 생성 완료!")
            else:
                st.error("초안 생성 실패")

# 질문 텍스트 영역 (값은 세션 상태에서 가져옴)
st.text_area("질문", value=st.session_state.current_question, height=100)
# 답변 텍스트 영역 (key를 'answer_text'로 설정하여 사용자의 편집 내용을 세션 상태에 저장)
ans = st.text_area("나의 답변 (초안을 편집해 완성하세요)", height=200, key="answer_text")

# ================== 6) 채점 & 코칭 (엄격 모드) ==================
st.header("6) 채점 & 코칭")
# 채점 및 코칭 실행 버튼
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
        st.session_state.last_result = res # 최종 결과 저장
        # 면접 기록(records)에 현재 결과 추가
        st.session_state.records.append({"question": st.session_state.current_question,
                                         "answer": st.session_state.answer_text, 
                                         "overall": res.get("overall_score", 0),
                                         "criteria": res.get("criteria", []),
                                         "strengths": res.get("strengths", []),
                                         "risks": res.get("risks", []),
                                         "improvements": res.get("improvements", []),
                                         "revised_answer": res.get("revised_answer","")})
        st.success("채점/코칭 완료!")

# ================== 7) 피드백 결과 (아래에 팔로업 인라인 배치) ==================
st.header("7) 피드백 결과")
last = st.session_state.last_result
if last:
    # 총점과 상세 코멘트를 분리하여 표시
    left, right = st.columns([1,3])
    with left:
        st.metric("총점(/100)", last.get("overall_score", 0))
    with right:
        st.markdown("**기준별 점수 & 코멘트**")
        for it in last.get("criteria", []):
            st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
        # 강점, 감점 요인, 개선 포인트를 목록으로 표시
        if last.get("strengths"):
            st.markdown("**강점**")
            for s in last["strengths"]: st.markdown(f"- {s}")
        if last.get("risks"):
            st.markdown("**감점 요인/리스크**")
            for r in last["risks"]: st.markdown(f"- {r}")
        if last.get("improvements"):
            st.markdown("**개선 포인트**")
            for im in last["improvements"]: st.markdown(f"- {im}")
        if last.get("revised_answer"):
            st.markdown("**수정본 답변 (STAR)**")
            st.write(last["revised_answer"])
else:
    st.info("아직 채점 결과가 없습니다.")

st.divider()

# ================== 8) 팔로업 질문 → 답변 → 팔로업 피드백 ==================
st.subheader("팔로업 질문 · 답변 · 피드백")
# 메인 질문 채점 결과가 있고, 팔로업 질문이 아직 생성되지 않았다면 LLM에게 팔로업 질문 생성을 요청
if last and not st.session_state.followups:
    try:
        # 회사/직무/비전/이슈 등 컨텍스트 구성
        ctx = json.dumps({"company": st.session_state.clean_struct,
                          "vision": st.session_state.company_vision[:6],
                          "talent": st.session_state.company_talent[:6],
                          "news": [n.get("title","") for n in st.session_state.company_news[:3]]}, ensure_ascii=False)
        # 사용자 메시지: 답변 내용을 기반으로 면접관 관점에서 3개의 팔로업 질문 제안 요청
        msg = {"role":"user",
               "content":(f"[회사/직무/요건/비전/이슈]\n{ctx}\n\n"
                          f"[지원자 답변]\n{st.session_state.answer_text}\n\n"
                          "면접관 관점에서 팔로업 질문 3개를 한 줄씩 한국어로 제안해줘. "
                          "기존 질문과 중복되지 않게, 지표/리스크/트레이드오프/의사결정 근거를 섞어줘.")}
        r = client.chat.completions.create(model=CHAT_MODEL, temperature=0.7,
                                           messages=[{"role":"system","content":"면접 팔로업 생성기"}, msg])
        # 답변에서 줄별로 질문을 추출하고 번호/특수문자 제거
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

        # 팔로업 질문 선택 및 답변 입력 필드
        st.selectbox("채점 받을 팔로업 질문 선택", st.session_state.followups, index=0, key="selected_followup")
        st.text_area("팔로업 질문에 대한 나의 답변", height=160, key="followup_answer")
        
        # 팔로업 채점 버튼
        if st.button("팔로업 채점 & 피드백", type="secondary"):
            fu_q   = st.session_state.get("selected_followup", "")
            fu_ans = st.session_state.get("followup_answer", "")
            if not fu_q:
                st.warning("팔로업 질문을 선택하세요.")
            elif not fu_ans.strip():
                st.warning("팔로업 답변을 작성하세요.")
            else:
                with st.spinner("팔로업 채점 중..."):
                    # 팔로업 답변에 대한 채점 및 코칭 실행
                    res_fu = llm_score_and_coach_strict(st.session_state.clean_struct, fu_q, fu_ans, CHAT_MODEL)
                st.session_state.last_followup_result = res_fu # 팔로업 결과 저장
                
                # 팔로업 채점 결과 표시
                st.markdown("**팔로업 결과**")
                st.metric("총점(/100)", res_fu.get("overall_score", 0))
                for it in res_fu.get("criteria", []):
                    st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
                if res_fu.get("revised_answer",""):
                    st.markdown("**팔로업 수정본 (STAR)**")
                    st.write(res_fu["revised_answer"])
    else:
        st.caption("팔로업 질문은 메인 질문 채점 직후 자동 제안됩니다.")