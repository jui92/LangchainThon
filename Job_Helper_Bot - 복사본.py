###################################################################################################################
#  1. 채용 포털 사이트 URL로 조회한 회사 정보와 등록한 이력서를 바탕으로 자소서를 자동으로 생성해줍니다                  #
#  2. 채용 포털 사이트 URL / 기업 홈페이지 URL / 뉴스 기사 를 참고하여 모의 면접을 실시하고 답변에 대한 피드백을 해줍니다.#
###################################################################################################################

# Library Import ( coding: utf-8 )
# 필요한 '도구 상자(라이브러리)'들을 불러옵니다.
import os, re, json, urllib.parse, time, io, random
from typing import Optional, Tuple, Dict, List

import requests                  # 웹에서 정보를 가져오는(HTTP 통신) 라이브러리
from bs4 import BeautifulSoup    # 웹 페이지(HTML)를 분석하기 위한 라이브러리 (Beautiful Soup)
import html2text                 # HTML을 일반 텍스트로 변환하는 라이브러리
import streamlit as st           # 웹 앱을 쉽게 만들 수 있게 돕는 라이브러리
import pandas as pd              # 데이터 분석 및 처리를 위한 라이브러리
import numpy as np               # 숫자 배열(행렬) 계산을 위한 라이브러리

# ================== 동적 크롤링 (Selenium) 라이브러리 ==================
# 웹 브라우저를 흉내 내어 자바스크립트를 실행하는 데 필요한 도구들을 불러옵니다.
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    st.error("`selenium` 및 `webdriver-manager` 패키지가 필요합니다.")
    st.stop()


# ================== 기본 설정 ==================
st.set_page_config(page_title="Job Helper Bot", page_icon="🤖", layout="wide")
st.title("Job Helper Bot : 자소서 생성 / 모의 면접")

# ================== OpenAI 설정 ==================
try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop()

# API 키 설정
API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
if not API_KEY:
    API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password")
if not API_KEY:
    st.stop()
client = OpenAI(api_key=API_KEY)

# 사이드바에 모델 설정 항목 배치
with st.sidebar:
    st.subheader("모델 설정")
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델(내부용)", ["text-embedding-3-small","text-embedding-3-large"], index=0)


# ================== HTTP 및 URL 유틸 ==================
def normalize_url(u: str) -> Optional[str]:
    """웹 주소 형식을 통일하고 정리합니다."""
    if not u: return None
    u = u.strip()
    if not re.match(r"^https?://", u): u = "https://" + u
    parts = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

def http_get(url: str, timeout: int = 12) -> Optional[requests.Response]:
    """특정 URL로 웹 요청을 보내고 응답을 받습니다 (정적 크롤링 및 Jina 프록시용)."""
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
    conv.ignore_links = True
    conv.ignore_images = True
    conv.body_width = 0
    txt = conv.handle(html_str)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


# ================== Selenium 드라이버 설정 (동적 크롤링 준비) ==================
@st.cache_resource
def get_chrome_driver():
    """Selenium Chrome 드라이버를 설정하고 반환합니다."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except WebDriverException as e:
        st.error(f"Chrome 드라이버 초기화 실패. Selenium/WebDriver 설정 및 환경을 확인하세요: {e}")
        return None

_SELENIUM_DRIVER = get_chrome_driver()


# ================== 원문 수집 (Jina → Selenium → BS4 폴백) ==================
def fetch_jina_text(url: str, timeout: int = 15) -> str:
    """Jina AI 프록시를 사용하여 웹 페이지 텍스트를 추출 (동적 페이지에도 유리)."""
    try:
        parts = urllib.parse.urlsplit(url)
        prox = f"https://r.jina.ai/http://{parts.netloc}{parts.path}"
        if parts.query: prox += f"?{parts.query}"
        r = http_get(prox, timeout=timeout)
        return r.text.strip() if r else ""
    except Exception:
        return ""

def fetch_selenium_text(url: str) -> str:
    """Selenium을 사용하여 페이지 로드 및 '더보기' 클릭 후 텍스트를 추출합니다 (동적 크롤링)."""
    driver = _SELENIUM_DRIVER
    if not driver:
        return ""

    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # '더보기' 버튼 찾기 및 클릭
        buttons_to_click = [(By.XPATH, "//button[contains(text(), '더보기')]"),
                            (By.XPATH, "//button[contains(text(), '펼치기')]"),
                            (By.CSS_SELECTOR, "div.btn_detail_view button"),
                            (By.CSS_SELECTOR, "a.btn_more")]

        clicked = False
        for by, selector in buttons_to_click:
            try:
                more_button = WebDriverWait(driver, 1.5).until(
                    EC.element_to_be_clickable((by, selector))
                )
                if more_button.is_displayed():
                    more_button.click()
                    time.sleep(1.5)
                    clicked = True
                    break
            except (TimeoutException, NoSuchElementException):
                continue
            except Exception:
                continue
        
        if clicked:
             st.info("동적 크롤링: '더보기/펼치기' 버튼 클릭 성공.")

        final_html = driver.page_source
        return html_to_text(final_html)

    except TimeoutException:
        st.warning(f"동적 크롤링: 페이지 로딩 시간 초과: {url}")
        return html_to_text(driver.page_source if driver else "")
    except Exception as e:
        st.error(f"동적 크롤링 중 알 수 없는 오류 발생: {e}")
        return ""

def fetch_bs4_text(html_str: str) -> Tuple[str, Optional[BeautifulSoup]]:
    """HTML을 BeautifulSoup으로 파싱하여 본문 텍스트를 추출합니다 (주요 콘텐츠 블록 우선, 정적 크롤링 폴백용)."""
    if not html_str: return "", None
    soup = BeautifulSoup(html_str, "lxml")
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
    """텍스트 추출 방법 3가지를 순서대로 시도하여 가장 좋은 결과를 반환합니다."""
    url = normalize_url(url)
    if not url: return "", {"error":"invalid_url"}, None
    
    # 1. Jina AI 시도
    jina = fetch_jina_text(url)
    if jina:
        r = http_get(url, timeout=12)
        soup = BeautifulSoup(r.text, "lxml") if r else None
        return jina, {"source":"jina","len":len(jina),"url_final":url}, soup
    
    # 2. Selenium 동적 크롤링 시도
    if _SELENIUM_DRIVER:
        selenium_text = fetch_selenium_text(url)
        if selenium_text and len(selenium_text) > 500:
            return selenium_text, {"source":"selenium_dynamic","len":len(selenium_text),"url_final":url}, None

    # 3. 일반적인 정적 크롤링 시도
    r = http_get(url, timeout=12)
    if not r: return "", {"source":"failed_all","len":0,"url_final":url}, None
    
    bs_text, soup = fetch_bs4_text(r.text) 
    return bs_text, {"source":"bs4_fallback","len":len(bs_text),"url_final":url}, soup


# ================== 메타/섹션 보조 추출 ==================
def extract_company_meta(soup: Optional[BeautifulSoup]) -> Dict[str,str]:
    """HTML 메타 태그와 제목에서 회사명, 소개, 직무명을 추정합니다."""
    meta = {"company_name":"","company_intro":"","job_title":""}
    if not soup: return meta
    # 회사 이름 후보 찾기 (og:site_name, application-name, title 태그 순서)
    cand = []
    # (HTML 태그 분석 로직)
    meta["company_name"] = cand[0] if cand else ""
    # 회사 소개 (description 메타 태그)
    # (회사 소개 로직)
    # 직무명 (og:title, h1, h2 태그 순서)
    # (직무명 로직)
    return meta

# ================== 규칙 파서 ==================
def rule_based_sections(raw_text: str) -> dict:
    """텍스트에서 '주요 업무', '자격 요건', '우대 사항'과 같은 키워드를 기반으로 정보를 추출합니다."""
    # (정규표현식을 이용한 키워드 매칭 로직)
    return {}

# ================== LLM 정제 (채용 공고 → 구조 JSON) ==================
PROMPT_SYSTEM_STRUCT = ("너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
                        "입력 텍스트는 포털 광고 문구, UI잔재, 복수 직무가 섞여 있을 수 있다. "
                        "한국어로 간결하고 중복없이 정제하라.")

def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
    """LLM을 사용하여 채용 공고 원문 텍스트를 구조화된 JSON 형태로 깔끔하게 정제합니다."""
    ctx = (raw_text or "").strip()
    if len(ctx) > 14000:
        ctx = ctx[:14000]

    user_msg = {"role": "user",
                "content": ("다음 채용 공고 원문을 구조화해줘.\n\n"
                            "회사: {company_name}\n"
                            "직무: {job_title}\n\n"
                            "원문:\n{ctx}"
                            "\n\n결과는 다음 JSON 형식만 반환하라:\n"
                            "{{'company_name':'', 'company_intro':'', 'job_title':'', 'requirements':[], 'preferred':[], 'responsibilities':[], 'etc':[]}}"
                            ).format(company_name=meta_hint.get("company_name",""), job_title=meta_hint.get("job_title",""), ctx=ctx)
                }
    try:
        resp = client.chat.completions.create(model=model, temperature=0.2,
                                              response_format={"type": "json_object"},
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg])
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        data = {"company_name": meta_hint.get("company_name",""),
                "company_intro": meta_hint.get("company_intro",""),
                "job_title": meta_hint.get("job_title",""),
                "requirements": [], "preferred": [], "responsibilities": [], "etc": [],
                "error": str(e)}

    # 후처리 (리스트 정리)
    for key in ['requirements', 'preferred', 'responsibilities', 'etc']:
        if isinstance(data.get(key), list):
            data[key] = [item[:100] for item in data[key] if isinstance(item, str) and item.strip()][:10]
    return data

# ================== 파일 리더 (PDF/TXT/MD/DOCX) ==================
try:
    import pypdf
except Exception:
    pypdf = None
try:
    import docx
except Exception:
    docx = None

def read_pdf(data: bytes) -> str:
    """PDF 파일의 내용을 텍스트로 추출합니다."""
    # (PDF 읽기 로직)
    return "PDF 파일 내용 추출 실패 (pypdf 라이브러리 사용)"

def read_docx(data: bytes) -> str:
    """DOCX 파일의 내용을 텍스트로 추출합니다."""
    # (DOCX 읽기 로직)
    return "DOCX 파일 내용 추출 실패 (docx 라이브러리 사용)"

def read_file_text(uploaded) -> str:
    """업로드된 파일의 종류(txt, pdf, docx 등)에 따라 적절한 방법으로 내용을 읽어옵니다."""
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith(".pdf"):
        return read_pdf(data)
    elif name.endswith(".docx"):
        return read_docx(data)
    else:
        try:
            return data.decode("utf-8")
        except:
            return data.decode("latin-1")
    return ""

# ================== 간단 청크/임베딩(내부 자동) ==================
def chunk(text: str, size: int = 600, overlap: int = 120) -> List[str]:
    """긴 텍스트를 일정한 크기(size)로 자르고 다음 조각과 겹치게(overlap) 만듭니다."""
    # (청크 분할 로직)
    return ["청크 1", "청크 2"]

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    """텍스트 조각들을 LLM이 이해할 수 있는 숫자 벡터(임베딩)로 변환합니다."""
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=model_name, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs

def cosine_topk(matrix: np.ndarray, query_vec: np.ndarray, k: int = 4):
    """질문 벡터와 가장 유사한(코사인 유사도) 상위 K개의 텍스트 조각을 찾습니다."""
    # (유사도 계산 로직)
    return np.array([1.0]*k), np.array([0]*k)

def retrieve_resume_chunks(query: str, k: int = 4):
    """질문과 관련성이 높은 이력서의 텍스트 조각(청크)을 검색(RAG)하여 가져옵니다."""
    chs, embs = st.session_state.get("resume_chunks", []), st.session_state.get("resume_embeds", None)
    if not chs or embs is None:
        return []
    qv = embed_texts([query], EMBED_MODEL)
    scores, idxs = cosine_topk(embs, qv, k=k)
    return [(float(s), chs[int(i)]) for s, i in zip(scores, idxs)]

# ================== 회사 비전/인재상/뉴스 ==================
VISION_KEYS = ["비전","미션","핵심가치","가치","원칙","문화","행동강령","Talent","인재상","Our Mission","Vision","Values"]

def safe_get_text(el) -> str:
    """BeautifulSoup 요소에서 안전하게 텍스트를 추출합니다."""
    return el.get_text(" ", strip=True) if el else ""

def fetch_company_pages(home_url: str) -> Dict[str, List[str]]:
    """회사 홈페이지와 주요 서브 경로에서 '비전/가치' 및 '인재상' 관련 텍스트를 긁어옵니다."""
    # (스크래핑 로직)
    return {"vision": ["비전 내용"], "talent": ["인재상 내용"]}

def fetch_latest_news(company: str, max_items: int = 5) -> List[Dict]:
    """회사 이름으로 최신 뉴스 기사를 검색합니다."""
    # (뉴스 검색 로직)
    return [{"title": "뉴스 제목", "link": "#"}]

# ================== 질문/초안/채점/팔로업 프롬프트 ==================
PROMPT_SYSTEM_Q = ("너는 채용담당자다. 회사/직무 맥락과 채용요건, 그리고 지원자의 이력서 요약을 함께 고려해 "
                   "면접 질문을 한국어로 생성한다. 질문은 서로 형태·관점·키워드가 겹치지 않게 다양화하고, "
                   "수치/지표/기간/규모/리스크/트레이드오프 등도 섞어라.")
PROMPT_SYSTEM_DRAFT = ("너는 면접 답변 코치다. 회사/직무/채용요건과 지원자의 이력서 요약을 결합해 "
                       "질문에 대한 답변 **초안**을 STAR(상황-과제-행동-성과)로 8~12문장, 한국어로 작성한다. "
                       "가능하면 구체적인 지표/수치/기간/임팩트를 포함하라.")
PROMPT_SYSTEM_SCORE_STRICT = ("너는 매우 엄격한 톱티어 면접 코치다. 아래 형식의 JSON만 출력하라. "
                              "{{'total_score': 0, 'comment_score_problem': '', 'comment_score_data': '', 'comment_score_execution': '', 'comment_score_collaboration': '', 'comment_score_customer': '', 'risks': [], 'strengths': [], 'improvement': [], 'revised_answer': ''}} "
                              "각 기준에 대해 짧지만 구체적 코멘트(강점/감점요인/개선포인트)를 제공하라.")
CRITERIA = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def llm_generate_one_question_with_resume(clean: Dict, level: str, model: str) -> str:
    """회사/직무 정보와 이력서의 핵심 내용을 바탕으로 면접 질문 1개를 생성합니다."""
    hits = retrieve_resume_chunks("핵심 프로젝트와 기술 스택 요약", k=4)
    resume_snips = [t for _, t in hits]
    resume_context = "\n".join([f"- {s[:350]}" for s in resume_snips])[:1200]
    # (LLM 호출 로직)
    return "생성된 면접 질문입니다."

def llm_draft_answer(clean: Dict, question: str, model: str) -> str:
    """질문과 이력서 내용을 바탕으로 STAR 기반의 답변 초안을 생성합니다."""
    hits = retrieve_resume_chunks(question, k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    # (LLM 호출 로직)
    return "생성된 STAR 기반의 답변 초안입니다."

def llm_score_and_coach_strict(clean: Dict, question: str, answer: str, model: str) -> Dict:
    """질문, 답변, 이력서 내용을 기반으로 엄격한 기준으로 채점하고 피드백을 제공합니다."""
    hits = retrieve_resume_chunks(question + "\n" + answer[:800], k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    # (LLM 호출 로직)
    return {"total_score": 75, 
            "comment_score_problem": "피드백1",
            "comment_score_data": "피드백2",
            "comment_score_execution": "피드백3",
            "comment_score_collaboration": "피드백4",
            "comment_score_customer": "피드백5",
            "risks": ["리스크1"], "strengths": ["강점1"], "improvement": ["개선1"], 
            "revised_answer": "수정된 답변입니다."}

# ================== 세션 상태 ==================
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


# ================== 1) 채용 공고 URL → 정제 ==================
st.header("1) 채용 공고 URL")
url = st.text_input("채용 공고 상세 URL", placeholder="취업 포털 사이트의 URL을 입력하세요.")

st.text_input("회사 공식 홈페이지 URL (선택)", key="company_home", placeholder="회사 공식 홈페이지 URL을 입력하세요.")
if st.button("원문 수집 → 정제", type="primary"):
    if not url.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner(f"URL 수집 및 정제 중... (최대 30초 소요, 현재 모드: {_SELENIUM_DRIVER.name if _SELENIUM_DRIVER else 'Static Fallback'})"):
            raw_text, info, soup = fetch_all_text(url)
        
        if not raw_text.strip():
            st.error(f"URL 수집 실패: {url}")
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

# ================== 2) 회사 요약 (정제 결과) ==================
st.header("2) 회사 요약")
clean = st.session_state.clean_struct
if clean:
    st.subheader(clean.get("company_name", "회사명 정보 없음"))
    st.caption(f"직무: {clean.get('job_title', '직무 정보 없음')}")
    st.markdown(f"**회사 소개:** {clean.get('company_intro', '-')}")
    
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

# ================== 3) 내 이력서/프로젝트 업로드 (PDF/TXT/MD/DOCX) ==================
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader("여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True)
_RESUME_CHUNK = 500
_RESUME_OVLP  = 100

cols_idx = st.columns(2)
with cols_idx[0]:
    if st.button("이력서 인덱싱", type="secondary"):
        if uploads:
            full_text = ""
            with st.spinner("파일 읽는 중..."):
                for uploaded_file in uploads:
                    full_text += read_file_text(uploaded_file) + "\n\n"
            
            if full_text.strip():
                st.session_state.resume_text = full_text
                chunks_list = chunk(full_text, _RESUME_CHUNK, _RESUME_OVLP)
                
                with st.spinner("텍스트 벡터화 중..."):
                    embeds = embed_texts(chunks_list, EMBED_MODEL)
                    st.session_state.resume_chunks = chunks_list
                    st.session_state.resume_embeds = embeds
                st.success(f"인덱싱 완료 (청크 {len(chunks_list)}개)")
            else:
                st.warning("읽을 수 있는 파일 내용이 없습니다.")
        else:
            st.warning("이력서 파일을 업로드하세요.")

st.divider()

# ================== 4) 이력서 기반 자소서 생성 ==================
st.header("4) 이력서 기반 자소서 생성")
topic = st.text_input("회사 요청 주제(선택)", placeholder="예: 직무 지원동기 / 협업 경험 / 문제해결 사례 등")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str) -> str:
    system = ("너는 한국어 자기소개서 전문가다. 채용 공고의 회사/직무 요건과 후보자의 이력서를 참고해 "
              "회사 특화 자소서를 작성한다. 과장/허위는 금지하고, 수치/지표/기간/임팩트 중심으로 구체화한다. "
              "회사의 비전/인재상/최근 이슈가 제공되면 자연스럽게 연결하라.")
    user_prompt = (f"## 채용 공고 요약\n{json.dumps(clean_struct, ensure_ascii=False, indent=2)}\n\n"
                   f"## 내 이력서 내용\n{resume_text}\n\n"
                   f"## 요청 주제\n{topic_hint}\n\n"
                   "주제에 맞춰 500자 내외로 자소서를 작성하고, 작성 후에는 반드시 최종본만 반환하라.")
    
    resp = client.chat.completions.create(model=model, temperature=0.7,
                                          messages=[{"role":"system","content":system}, {"role":"user", "content": user_prompt}])
    return resp.choices[0].message.content.strip()

if st.button("자소서 생성", type="primary"):
    if not st.session_state.clean_struct:
        st.warning("먼저 채용 공고를 정제하세요.")
    elif not st.session_state.resume_text:
        st.warning("이력서 파일을 인덱싱하세요.")
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

# ================== 5) 질문 생성 & 답변 초안 (RAG 결합) ==================
st.header("5) 질문 생성 & 답변 초안 (RAG 결합)")
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)

cols_q = st.columns(2)
with cols_q[0]:
    if st.button("새 질문 받기", type="primary"):
        if not st.session_state.clean_struct or not st.session_state.resume_embeds is not None:
            st.warning("채용 공고 정제 및 이력서 인덱싱을 완료하세요.")
        else:
            with st.spinner("질문 생성 중..."):
                st.session_state.current_question = llm_generate_one_question_with_resume(st.session_state.clean_struct, level, CHAT_MODEL)
                st.session_state.draft_answer = ""
                st.session_state.answer_text = ""
                st.session_state.last_result = None
                st.session_state.followups = []

with cols_q[1]:
    if st.button("RAG로 답변 초안 생성", type="secondary"):
        if not st.session_state.current_question:
            st.warning("먼저 '새 질문 받기'를 클릭하세요.")
        else:
            with st.spinner("답변 초안 생성 중..."):
                st.session_state.draft_answer = llm_draft_answer(st.session_state.clean_struct, st.session_state.current_question, CHAT_MODEL)
                st.session_state.answer_text = st.session_state.draft_answer

st.text_area("질문", value=st.session_state.current_question, height=100)
ans = st.text_area("나의 답변 (초안을 편집해 완성하세요)", height=200, key="answer_text")

# ================== 6) 채점 & 코칭 (엄격 모드) ==================
st.header("6) 채점 & 코칭")
if st.button("채점 & 코칭 실행", type="primary"):
    if not st.session_state.current_question or not st.session_state.answer_text.strip():
        st.warning("질문을 받고 답변을 작성하세요.")
    else:
        with st.spinner("답변 채점 및 코칭 중..."):
            st.session_state.last_result = llm_score_and_coach_strict(st.session_state.clean_struct, st.session_state.current_question, st.session_state.answer_text, CHAT_MODEL)
            st.session_state.followups = ["팔로업 질문 1", "팔로업 질문 2", "팔로업 질문 3"]

# ================== 7) 피드백 결과 (아래에 팔로업 인라인 배치) ==================
st.header("7) 피드백 결과")
last = st.session_state.last_result
if last:
    st.metric("총점(/100)", last.get("total_score", "N/A"))
    
    st.markdown("---")
    st.markdown("**기준별 코멘트**")
    for criterion in CRITERIA:
        key = f"comment_score_{criterion.split('/')[0].lower()}"
        st.caption(f"**{criterion}:** {last.get(key, '-')}")

    st.markdown("---")
    st.markdown("**강점 & 리스크 & 개선 포인트**")
    st.success(f"**강점:** {', '.join(last.get('strengths', []))}")
    st.error(f"**리스크:** {', '.join(last.get('risks', []))}")
    st.info(f"**개선 포인트:** {', '.join(last.get('improvement', []))}")
    
    st.markdown("---")
    st.markdown("**수정본 답변 (STAR 적용)**")
    st.text_area("LLM 수정본", value=last.get('revised_answer', '-'), height=200)

else:
    st.info("아직 채점 결과가 없습니다.")

st.divider()

# ================== 8) 팔로업 질문 → 답변 → 팔로업 피드백 ==================
st.subheader("팔로업 질문 · 답변 · 피드백")
last = st.session_state.last_result
if last:
    if st.session_state.followups:
        st.markdown("**팔로업 질문 제안**")
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
                with st.spinner("팔로업 채점 중..."):
                    res_fu = llm_score_and_coach_strict(st.session_state.clean_struct, fu_q, fu_ans, CHAT_MODEL)
                st.session_state.last_followup_result = res_fu
                st.markdown("**팔로업 결과**")
                st.metric("총점(/100)", res_fu.get("total_score", "N/A"))
                st.text_area("팔로업 피드백", value=res_fu.get('revised_answer', '-'), height=150)
    else:
        st.caption("팔로업 질문은 메인 질문 채점 직후 자동 제안됩니다.")