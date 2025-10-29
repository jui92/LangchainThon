import os, re, io, json, time, shutil, urllib.parse, tempfile, traceback
from typing import Optional, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import numpy as np
import pandas as pd
import requests
import html2text
from bs4 import BeautifulSoup

# OpenAI
try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop()

API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
if not API_KEY:
    API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password")
if not API_KEY:
    st.stop()
client = OpenAI(api_key=API_KEY)

with st.sidebar:
    st.subheader("모델 / 크롤링 옵션")
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small","text-embedding-3-large"], index=0)
    SELENIUM_TIMEOUT = st.slider("Selenium 대기(초)", 6, 30, 14)
    FAST_MODE = st.toggle("Fast 모드(빠르게)", value=True)

# -----------------------------------------------------------------------------
# html2text
# -----------------------------------------------------------------------------
def _get_html2text():
    conv = html2text.HTML2Text()
    conv.ignore_links = True
    conv.ignore_images = True
    conv.body_width = 0
    return conv
HTML2TEXT = _get_html2text()

def html_to_text(html_str: str) -> str:
    txt = HTML2TEXT.handle(html_str or "")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return re.sub(r"\s+", " ", txt).strip()

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def normalize_url(u: str) -> Optional[str]:
    if not u: return None
    u = u.strip()
    if not re.match(r"^https?://", u): u = "https://" + u
    parts = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

def clean_text(s: str, max_len: int = 16000) -> str:
    if not s: return ""
    s = re.sub(r"\r", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len] if len(s) > max_len else s

# -----------------------------------------------------------------------------
# Selenium driver
# -----------------------------------------------------------------------------
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

def _pick_chrome_binary() -> Optional[str]:
    cands = [
        os.getenv("CHROME_BIN"), os.getenv("GOOGLE_CHROME_BIN"),
        shutil.which("chromium"), shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        "/usr/bin/chromium","/usr/bin/chromium-browser",
        "/usr/bin/google-chrome","/usr/bin/google-chrome-stable",
    ]
    for p in cands:
        if p and os.path.exists(p): return p
    return None

def _build_chrome(headless: bool = True):
    opts = ChromeOptions()
    if headless: opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1440,2400")
    opts.add_argument("--lang=ko-KR")
    binpath = _pick_chrome_binary()
    if binpath: opts.binary_location = binpath
    # 드라이버 초기화 시 Selenium Manager가 자동으로 설치/관리하도록 함
    driver = webdriver.Chrome(options=opts)
    return driver

# -----------------------------------------------------------------------------
# Domain expand helpers (채용 포털용 확장 버튼들)
# -----------------------------------------------------------------------------
def _click_by_text_candidates(driver, texts: List[str], per=12):
    for t in texts:
        try:
            xp1 = f"//*[normalize-space(text())='{t}']"
            xp2 = f"//*[contains(normalize-space(text()), '{t}')]"
            for xp in (xp1, xp2):
                els = driver.find_elements(By.XPATH, xp)
                for el in els[:per]:
                    try:
                        driver.execute_script("arguments[0].click();", el)
                        time.sleep(0.12 if FAST_MODE else 0.25)
                    except Exception:
                        continue
        except Exception:
            continue

def _click_many_css(driver, selectors: List[str], per=12):
    for sel in selectors:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            for el in els[:per]:
                try:
                    driver.execute_script("arguments[0].click();", el)
                    time.sleep(0.1 if FAST_MODE else 0.2)
                except Exception:
                    continue
        except Exception:
            continue

def _expand_wanted(driver):
    sel = [
        "[data-qa='btn-read-more']","[data-qa='job-header__more']",
        "button[aria-expanded='false']","[role='button'][class*='More']",
        "div[aria-expanded='false']",
    ]
    _click_many_css(driver, sel, per=(8 if FAST_MODE else 12))
    _click_by_text_candidates(driver, [
        "더보기","전체보기","자세히","상세보기","모두 보기",
        "주요업무","자격요건","우대사항","기업/팀 소개",
        "나중에 하기","닫기","확인"
    ], per=(6 if FAST_MODE else 12))

def _expand_saramin(driver):
    sel = [".btn_more",".btnMore",".btn-detail",".btn_toggle",
           "[aria-expanded='false']","[role='button']","button[class*='more'], a[class*='more']"]
    _click_many_css(driver, sel, per=(6 if FAST_MODE else 12))
    _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","기업정보","상세정보"], per=(6 if FAST_MODE else 12))

def _expand_jobkorea(driver):
    sel = [".btnFold",".btnToggleRead",".btn_more",
           "[aria-expanded='false']","[role='button']","button[class*='More'], a[class*='More']"]
    _click_many_css(driver, sel, per=(6 if FAST_MODE else 12))
    _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","기업정보","상세보기"], per=(6 if FAST_MODE else 12))

# -----------------------------------------------------------------------------
# Wanted __NEXT_DATA__ → text
# -----------------------------------------------------------------------------
def extract_wanted_from_next_html(html: str) -> str:
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        tag = soup.select_one("script#__NEXT_DATA__")
        if not tag: return ""
        raw = (tag.string or tag.text or "").strip()
        data = json.loads(raw)
    except Exception:
        return ""
    key_whitelist = [
        "job","position","title","desc","description",
        "responsibilit","duty","role","skill","stack",
        "require","qualification","prefer","plus","nice"
    ]
    def _safe(x): return re.sub(r"\s+"," ", (x or "")).strip()
    def _walk(d, out):
        if isinstance(d, dict):
            for k, v in d.items():
                if any(t in str(k).lower() for t in key_whitelist):
                    if isinstance(v, str): out.append(v)
                    elif isinstance(v, list):
                        for it in v:
                            if isinstance(it, str): out.append(it)
                            elif isinstance(it, dict):
                                for _, subv in it.items():
                                    if isinstance(subv, str): out.append(subv)
                    elif isinstance(v, dict):
                        for _, subv in v.items():
                            if isinstance(subv, str): out.append(subv)
                _walk(v, out)
        elif isinstance(d, list):
            for it in d: _walk(it, out)
    bucket=[]; _walk(data, bucket)
    seen=set(); lines=[]
    for t in bucket:
        s=_safe(t)
        if len(s)>2 and s not in seen:
            seen.add(s); lines.append(s)
    return "\n".join(lines[:900])

# -----------------------------------------------------------------------------
# Selenium fetch (DOM + NEXT_DATA)
# -----------------------------------------------------------------------------
def selenium_get_html(url: str, timeout: int = 14) -> str:
    driver = _build_chrome(headless=True)
    try:
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        try:
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, "//*")))
        except TimeoutException:
            pass

        host = urllib.parse.urlsplit(url).netloc.lower()
        _click_by_text_candidates(driver, ["더보기","상세보기","자세히 보기","전체보기","Read more","More"],
                                  per=(6 if FAST_MODE else 10))
        _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","Requirements","Responsibilities","Preferred"],
                                  per=(6 if FAST_MODE else 10))

        if "wanted.co.kr" in host: _expand_wanted(driver)
        if "saramin" in host:     _expand_saramin(driver)
        if "jobkorea" in host:    _expand_jobkorea(driver)

        loops = 5 if FAST_MODE else 8
        for _ in range(loops):
            try:
                # 스크롤을 통해 동적 로딩되는 콘텐츠 로드 시도
                driver.execute_script("window.scrollBy(0, 1200);"); time.sleep(0.12 if FAST_MODE else 0.25)
            except Exception:
                break

        html = driver.page_source or ""
        if "wanted.co.kr" in host:
            try:
                txt_next = extract_wanted_from_next_html(html)
                if txt_next:
                    html += "\n<div id='__WANTED_NEXT_EXTRACT__'>" + \
                            "".join([f"<p>{line}</p>" for line in txt_next.split("\n")]) + "</div>"
            except Exception:
                pass
        return html
    finally:
        try: driver.quit()
        except Exception: pass

# -----------------------------------------------------------------------------
# MODIFIED: Robust fetch function (requests primary, selenium fallback)
# -----------------------------------------------------------------------------
def fetch_all_text_selenium(url: str, timeout: int = 14) -> Tuple[str, Dict, Optional[str]]:
    url_n = normalize_url(url)
    if not url_n: return "", {"error":"invalid_url"}, None
    
    # --- 1. Requests (Fast/Static) Attempt ---
    # 일반적인 기업 홈페이지의 경우 requests가 빠르고 안정적입니다.
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        # 빠른 응답을 위해 짧은 타임아웃 설정 (5초)
        response = requests.get(url_n, headers=headers, timeout=5) 
        response.raise_for_status() 
        html_req = response.text

        soup_req = BeautifulSoup(html_req, "html.parser")
        # 스크립트, 스타일, 네비게이션 등 불필요한 요소 제거 (정적 페이지 크롤링 품질 향상)
        for tag in soup_req(["script", "style", "header", "footer", "nav", ".menu", "#menu", "[class*='sidebar']"]):
            tag.decompose()
        
        # main, article 등 주요 콘텐츠 영역을 찾아서 텍스트 추출 시도
        main_content_candidates = soup_req.select("main, .main, #main, article, .content, #content, [role='main'], body")
        
        text_to_convert = html_req
        if main_content_candidates:
            # 가장 긴 텍스트를 가진 콘텐츠 후보 선택
            best_candidate = max(main_content_candidates, key=lambda tag: len(tag.get_text()))
            if best_candidate:
                text_to_convert = best_candidate.prettify()

        txt_req = html_to_text(text_to_convert)

        # 텍스트 길이가 300자 이상이면 유의미한 정보로 판단하고 즉시 반환 (requests 성공)
        if len(txt_req) > 300: 
            return txt_req, {"source":"requests","len":len(txt_req),"url_final":url_n}, html_req

    except requests.exceptions.RequestException:
        # requests 실패 시 Selenium으로 자연스럽게 폴백
        pass
    except Exception:
        # 기타 오류 발생 시에도 Selenium으로 폴백
        pass

    # --- 2. Selenium (Slow/Dynamic) Fallback (기존 로직) ---
    # requests가 실패했거나, 동적 로딩이 필요한 채용 공고 페이지일 경우 Selenium 시도
    try:
        html_sel = selenium_get_html(url_n, timeout=timeout)
    except Exception as e:
        # Selenium 로드 자체가 실패한 경우에만 오류 메시지 표시
        st.error(f"Selenium 로드 실패: {e}")
        st.code("".join(traceback.format_exc()))
        return "", {"source":"selenium_error","len":0,"url_final":url_n}, None

    if not html_sel or len(html_sel) < 200:
        return "", {"source":"selenium_failed","len":0,"url_final":url_n}, None

    txt_sel = html_to_text(html_sel)
    return txt_sel, {"source":"selenium","len":len(txt_sel),"url_final":url_n}, html_sel

# -----------------------------------------------------------------------------
# Meta & rule-based sections (ONLY 3 buckets)
# -----------------------------------------------------------------------------
def extract_company_meta_from_html(html: Optional[str]) -> Dict[str, str]:
    meta = {"company_name": "", "company_intro": "", "job_title": ""}
    if not html: return meta
    try:
        soup = BeautifulSoup(html, "html.parser")
        cand=[]
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
        jt=""
        ogt = soup.find("meta", {"property":"og:title"})
        if ogt and ogt.get("content"): jt = ogt["content"]
        if not jt:
            h1 = soup.find("h1")
            if h1 and h1.get_text(): jt = h1.get_text(strip=True)
        if not jt:
            h2 = soup.find("h2")
            if h2 and h2.get_text(): jt = h2.get_text(strip=True)
        meta["job_title"] = re.sub(r"\s+"," ", jt).strip()[:120]
    except Exception:
        pass
    return meta

def rule_based_sections(raw_text: str) -> dict:
    txt = clean_text(raw_text, 16000)
    lines = [re.sub(r"\s+"," ", l).strip(" -•·▶▪️") for l in txt.split("\n") if l.strip()]
    
    # 정규표현식 정의
    hdr_resp = re.compile(r"(주요\s*업무|담당\s*업무|Role|Responsibilities?)", re.I)
    hdr_qual = re.compile(r"(자격\s*요건|지원\s*자격|Requirements?|Qualifications?)", re.I)
    hdr_pref = re.compile(r"(우대\s*사항|우대|선호|Preferred|Nice\s*to\s*have|Plus)", re.I)
    kw_pref = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
    
    out = {"responsibilities": [], "qualifications": [], "preferences": []}
    bucket=None
    
    def push(line,b):
        if line and len(line)>1 and line not in out[b]:
            out[b].append(line[:180])

    # 라인을 돌면서 섹션을 분류
    for l in lines:
        if hdr_resp.search(l):
            bucket="responsibilities"; continue
        if hdr_qual.search(l):
            bucket="qualifications"; continue
        if hdr_pref.search(l):
            bucket="preferences"; continue
        
        # 헤더가 없는 경우: 키워드로 분류 시도
        if bucket is None:
            low = l.lower()
            if hdr_pref.search(l):
                bucket = "preferences"
            elif any(k in low for k in ["java","python","spring","kotlin","react","next","kafka","sql","ml","cloud","aws","gcp"]):
                bucket = "responsibilities"
            else:
                continue

        push(l,bucket)

    # 자격요건에 있지만 우대사항 키워드가 포함된 문구는 우대사항으로 이동
    remain=[]
    for q in out["qualifications"]:
        (out["preferences"] if kw_pref.search(q) else remain).append(q)
    out["qualifications"]=remain

    # 중복 제거 및 길이 제한
    for k in out:
        seen=set(); clean=[]
        for s in out[k]:
            s=re.sub(r"\s+"," ", s).strip(" -•·▶▪️").strip()
            if s and s not in seen:
                seen.add(s); clean.append(s)
        out[k]=clean[:14] # 각 섹션별 최대 14개 항목

    return out

# -----------------------------------------------------------------------------
# LLM structure / Q&A / scoring
# -----------------------------------------------------------------------------
PROMPT_SYSTEM_STRUCT = ("너는 채용 공고를 깔끔하게 구조화하는 보조원이다. 한국어로 간결하고 중복없이 정제하라.")
PROMPT_FORMAT_STRUCT = """
--- JSON 응답 형식 (이 형식 그대로만 응답하세요) ---
{
  "company_name": "회사명",
  "job_title": "직무/포지션 명",
  "responsibilities": ["핵심 주요 업무 1", "핵심 주요 업무 2", ...],
  "qualifications": ["필수 자격 요건 1", "필수 자격 요건 2", ...],
  "preferences": ["우대 사항 1", "우대 사항 2", ...],
  "company_intro": "회사 소개 (100자 내로 간결하게)"
}
---
"""
def llm_structurize(raw_text: str, meta_hint: Dict[str, str], model: str) -> Dict:
    ctx = clean_text(raw_text, 14000)
    user_msg = {"role": "user", "content": (
        "다음 채용 공고 원문을 구조화해줘.\n\n"
        f"[힌트] 회사명 후보: {meta_hint.get('company_name', '')}, 직무 후보: {meta_hint.get('job_title', '')}\n"
        f"[회사 소개 힌트]: {meta_hint.get('company_intro', '')}\n\n"
        f"[채용 공고 원문]\n{ctx}"
    )}
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM_STRUCT + PROMPT_FORMAT_STRUCT},
                user_msg
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(res.choices[0].message.content)
        return data
    except Exception as e:
        st.warning(f"구조화 실패 (LLM 오류): {e}")
        return {}

def structurize_and_refine(raw_text: str, meta_hint: Dict[str, str], model: str) -> Dict:
    # 1. Rule-based extraction (Fast fallback)
    rule_sections = rule_based_sections(raw_text)
    
    # 2. LLM structure (Accurate)
    llm_data = llm_structurize(raw_text, meta_hint, model)

    # 3. Merge & Refine
    final_data = {
        "company_name": (llm_data.get("company_name") or meta_hint.get("company_name", "")).strip()[:80],
        "job_title": (llm_data.get("job_title") or meta_hint.get("job_title", "")).strip()[:120],
        "company_intro": (llm_data.get("company_intro") or meta_hint.get("company_intro", "")).strip()[:500],
    }
    
    # LLM 결과가 있으면 사용, 없으면 Rule-based 결과 사용
    def _merge_list(llm_list, rule_list):
        if llm_list and isinstance(llm_list, list) and len("".join(llm_list)) > 20:
            return llm_list[:20]
        return rule_list[:20]

    final_data["responsibilities"] = _merge_list(llm_data.get("responsibilities"), rule_sections["responsibilities"])
    final_data["qualifications"] = _merge_list(llm_data.get("qualifications"), rule_sections["qualifications"])
    final_data["preferences"] = _merge_list(llm_data.get("preferences"), rule_sections["preferences"])

    # Final validation
    if not final_data.get("company_name"): final_data["company_name"] = "회사명_미확인"
    if not final_data.get("job_title"): final_data["job_title"] = "직무명_미확인"
    if not final_data.get("company_intro"): final_data["company_intro"] = "회사_소개_미확인"
    if not final_data.get("responsibilities"): final_data["responsibilities"] = ["주요 업무 미확인"]
    
    return final_data

# -----------------------------------------------------------------------------
# Document Chunking / Embedding
# -----------------------------------------------------------------------------
def chunk_text(text: str, max_len=1800, min_len=100) -> List[str]:
    # 간단한 단락 기반 청킹
    if not text: return []
    text = clean_text(text, 65536)
    chunks = []
    current_chunk = ""
    for line in text.split("\n\n"):
        if len(current_chunk) + len(line) + 2 < max_len:
            current_chunk += line + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    # 너무 짧은 청크 병합 시도
    final_chunks = []
    temp_chunk = ""
    for chunk in chunks:
        if len(temp_chunk) + len(chunk) + 2 < max_len:
            temp_chunk += chunk + "\n\n"
        else:
            if temp_chunk:
                final_chunks.append(temp_chunk.strip())
            temp_chunk = chunk + "\n\n"
    if temp_chunk:
        final_chunks.append(temp_chunk.strip())

    return [c for c in final_chunks if len(c) > min_len]

def get_text_embedding(text_list: List[str], model: str) -> List[List[float]]:
    try:
        text_list = [t.replace("\n", " ") for t in text_list]
        response = client.embeddings.create(
            input=text_list,
            model=model
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        st.error(f"임베딩 생성 실패: {e}")
        return [[]] * len(text_list)

# -----------------------------------------------------------------------------
# RAG (Retrieval)
# -----------------------------------------------------------------------------
def _find_best_chunks_by_embed(
    query_embed: List[float], 
    doc_embeds: List[List[float]], 
    doc_chunks: List[str], 
    top_k: int = 4
) -> List[str]:
    if not query_embed or not doc_embeds: return []
    query_vec = np.array(query_embed)
    doc_vecs = np.array(doc_embeds)
    
    # 코사인 유사도 계산 (벡터 정규화 가정)
    # np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # 정규화된 벡터이므로 단순히 내적(dot product)이 유사도입니다.
    similarities = np.dot(doc_vecs, query_vec)
    
    # 유사도가 높은 순서로 인덱스 정렬
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Top-K 청크 선택
    selected_chunks = []
    for i in sorted_indices:
        if len(selected_chunks) >= top_k: break
        if similarities[i] > 0.6: # 최소 유사도 컷오프
            selected_chunks.append(doc_chunks[i])
            
    return selected_chunks[:top_k]

def retrieve_context(
    query: str, 
    resume_chunks: List[str], 
    resume_embeds: List[List[float]], 
    job_desc_chunks: List[str], 
    job_desc_embeds: List[List[float]], 
    embed_model: str
) -> Tuple[str, str]:
    # 쿼리 임베딩
    query_embed = get_text_embedding([query], embed_model)[0]
    if not query_embed:
        return "이력서 내용을 찾을 수 없음", "채용 공고 내용을 찾을 수 없음"
    
    # 이력서에서 관련 청크 검색
    resume_context_list = _find_best_chunks_by_embed(query_embed, resume_embeds, resume_chunks, top_k=2)
    resume_context = "--- 이력서 관련 내용 ---\n" + "\n\n".join(resume_context_list)
    
    # 채용 공고에서 관련 청크 검색 (RAG-Job-Description은 필요하지 않을 수 있지만, 포함)
    job_context_list = _find_best_chunks_by_embed(query_embed, job_desc_embeds, job_desc_chunks, top_k=2)
    job_context = "--- 채용 공고 관련 내용 ---\n" + "\n\n".join(job_context_list)
    
    return resume_context, job_context

# -----------------------------------------------------------------------------
# LLM Generation
# -----------------------------------------------------------------------------

PROMPT_SYSTEM_JD_CV = (
    "너는 최고의 커리어 코치이자 자기소개서 작성 전문가이다. "
    "사용자가 제공한 채용 공고 정보와 이력서 내용을 바탕으로, 해당 회사와 직무에 완벽하게 맞춰진 지원 동기와 역량을 강조하는 자기소개서 초안을 작성해줘야 한다. "
    "결과물은 **마크다운 형식의 깔끔한 한국어 에세이 형태**여야 하며, **구조화된 채용 공고의 각 핵심 요소(주요 업무, 자격 요건, 우대 사항)를 자신의 경험으로 연결**하여 설득력을 높여야 한다."
)
PROMPT_SYSTEM_INTERVIEW = (
    "너는 최고의 면접관이자 커리어 코치이다. "
    "제시된 채용 공고 구조화 정보와 이력서(혹은 자기소개서) 내용을 바탕으로, 지원자의 **면접 질문을 생성**하고, **답변에 대한 전문적인 피드백을 제공**해야 한다."
    "**질문은 채용 공고와 이력서 내용을 융합**하여 지원자의 **핵심 역량과 직무 적합성을 깊이 있게 검증**할 수 있도록 구체적이고 심층적이어야 한다."
)

# 1. 자기소개서 생성
def llm_generate_cover_letter(
    job_struct: Dict, resume_text: str, model: str
) -> str:
    jd = json.dumps(job_struct, ensure_ascii=False, indent=2)
    
    user_msg = {"role": "user", "content": (
        "다음 채용 공고 구조화 정보와 이력서 원문을 참고하여 지원 동기 및 역량 중심의 자기소개서(에세이)를 작성해줘.\n\n"
        "[채용 공고 구조화 정보]\n" + jd + "\n\n"
        "[이력서 원문]\n" + resume_text
    )}
    
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM_JD_CV},
                user_msg
            ],
            temperature=0.6 # 창의성을 위해 온도를 약간 높임
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"자기소개서 생성 실패: {e}"

# 2. 면접 질문 생성
PROMPT_FORMAT_QUESTION = """
--- JSON 응답 형식 (이 형식 그대로만 응답하세요) ---
{
  "questions": [
    {
      "type": "jd_based",
      "question": "지원한 직무의 주요 업무(예: 대규모 데이터 처리 시스템 구축)와 관련하여 본인의 경험을 구체적으로 설명해주세요.",
      "relevance": "주요 업무/자격 요건/우대 사항 중 가장 관련 있는 핵심 요소"
    },
    {
      "type": "resume_based",
      "question": "이력서에 언급된 '프로젝트 X'에서 발생했던 가장 큰 기술적 어려움은 무엇이었고, 이를 어떻게 해결했는지 설명해 주십시오.",
      "relevance": "이력서의 특정 경험/기술"
    },
    // 최소 5개 이상의 질문을 생성합니다.
  ],
  "followup_questions": ["질문1에 대한 팔로업 질문", "질문2에 대한 팔로업 질문", ...],
}
---
"""
def llm_generate_questions(
    job_struct: Dict, resume_text: str, model: str
) -> Dict:
    jd = json.dumps(job_struct, ensure_ascii=False, indent=2)
    
    user_msg = {"role": "user", "content": (
        "다음 채용 공고 구조화 정보와 이력서 원문을 참고하여, 면접 질문 (최소 5개) 및 각 질문에 대한 팔로업 질문 (최소 3개)을 JSON 형식으로 생성해줘.\n\n"
        "[채용 공고 구조화 정보]\n" + jd + "\n\n"
        "[이력서 원문]\n" + resume_text
    )}
    
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM_INTERVIEW + PROMPT_FORMAT_QUESTION},
                user_msg
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        st.warning(f"질문 생성 실패: {e}")
        return {"questions": [], "followup_questions": []}

# 3. 면접 답변 채점 및 피드백
PROMPT_FORMAT_SCORE = """
--- JSON 응답 형식 (이 형식 그대로만 응답하세요) ---
{
  "overall_score": 85, // 0~100점 사이의 점수
  "feedback_kr": "답변에 대한 구체적이고 상세한 한국어 피드백",
  "next_followup": ["질문1에 대한 추가 팔로업 질문", "질문2에 대한 추가 팔로업 질문"],
  "rag_used": {
    "resume": "답변 평가에 사용된 이력서의 주요 인용구",
    "job_desc": "답변 평가에 사용된 채용 공고의 주요 인용구"
  }
}
---
"""
def llm_score_and_coach_strict(
    job_struct: Dict,
    question: str,
    answer: str,
    model: str,
    resume_chunks: List[str],
    resume_embeds: List[List[float]],
    job_desc_chunks: List[str], # RAG에 사용될 Job Description 청크
    job_desc_embeds: List[List[float]],
    embed_model: str = EMBED_MODEL
) -> Dict:
    
    # RAG를 통해 관련 이력서 내용과 JD 내용을 가져옴
    res_ctx, jd_ctx = retrieve_context(
        f"{question}에 대한 답변: {answer}", 
        resume_chunks, resume_embeds, job_desc_chunks, job_desc_embeds, embed_model
    )

    jd = json.dumps(job_struct, ensure_ascii=False, indent=2)
    
    user_msg = {"role": "user", "content": (
        "아래 채용 공고, 질문, 답변, 이력서 관련 내용을 참고하여 답변을 100점 만점으로 채점하고, 상세한 피드백을 JSON 형식으로 제공해줘.\n"
        "피드백은 질문의 의도를 얼마나 잘 파악하고, 직무 적합성과 깊이 있는 전문성을 보여줬는지에 초점을 맞춰.\n\n"
        f"[채용 공고 구조화 정보]\n{jd}\n\n"
        f"[질문]\n{question}\n\n"
        f"[답변]\n{answer}\n\n"
        f"[RAG 컨텍스트]\n{res_ctx}\n\n{jd_ctx}"
    )}
    
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM_INTERVIEW + "\n" + PROMPT_FORMAT_SCORE},
                user_msg
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(res.choices[0].message.content)
        # 점수와 피드백 유효성 검사 및 기본값 설정
        data["overall_score"] = int(data.get("overall_score", 0))
        data["feedback_kr"] = data.get("feedback_kr", "피드백을 생성하지 못했습니다.")
        data["next_followup"] = data.get("next_followup", [])
        return data
    except Exception as e:
        st.warning(f"채점 및 피드백 실패: {e}")
        return {"overall_score": 0, "feedback_kr": "시스템 오류로 채점 및 피드백에 실패했습니다.", "next_followup": [], "rag_used": {}}

# -----------------------------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------------------------
# ... (Streamlit UI/Logic - The full logic is not provided, but the parts that were)

# st.session_state 초기화
if "clean_struct" not in st.session_state: st.session_state.clean_struct = {}
if "raw_text" not in st.session_state: st.session_state.raw_text = ""
if "resume_text" not in st.session_state: st.session_state.resume_text = ""
if "resume_chunks" not in st.session_state: st.session_state.resume_chunks = []
if "resume_embeds" not in st.session_state: st.session_state.resume_embeds = []
if "job_desc_chunks" not in st.session_state: st.session_state.job_desc_chunks = []
if "job_desc_embeds" not in st.session_state: st.session_state.job_desc_embeds = []
if "interview_questions" not in st.session_state: st.session_state.interview_questions = []
if "followups" not in st.session_state: st.session_state.followups = []

st.title("취업 도우미 봇 🤖")
st.markdown("채용 공고 URL과 기업 홈페이지 URL을 입력하면, 맞춤형 자소서 생성 및 모의 면접을 제공합니다.")

# -----------------------------------------------------------------------------
# Input URLs and Fetching
# -----------------------------------------------------------------------------
job_url = st.text_input("**1. 채용 공고 URL**", key="job_url")
company_url = st.text_input("**2. 기업 홈페이지 URL** (정보 보강용)", key="company_url")

def fetch_and_process(job_u, company_u):
    all_raw_text = ""
    job_html, company_html = None, None
    meta_hint = {}
    
    st.subheader("크롤링 결과")
    
    # 1. 채용 공고 크롤링
    with st.spinner(f"채용 공고 크롤링 중 ({job_u})..."):
        # 수정된 fetch_all_text_selenium 함수 사용
        job_txt, job_meta, job_html = fetch_all_text_selenium(job_u, SELENIUM_TIMEOUT)
        if job_txt:
            all_raw_text += job_txt
            meta_hint.update(extract_company_meta_from_html(job_html))
            st.success(f"✅ 채용 공고 크롤링 성공: {len(job_txt)}자")
        else:
            st.warning("⚠️ 채용 공고 크롤링 실패 또는 내용 부족")
            
    # 2. 기업 홈페이지 크롤링 (보강용)
    if company_u and job_txt: # 채용 공고 크롤링 성공 시에만 시도
        with st.spinner(f"기업 홈페이지 크롤링 중 ({company_u})..."):
            # 수정된 fetch_all_text_selenium 함수 사용 (requests 우선 적용)
            company_txt, company_meta, company_html = fetch_all_text_selenium(company_u, SELENIUM_TIMEOUT)
            if company_txt and len(company_txt)>len(job_txt)*0.1: # 유의미한 길이인 경우만
                all_raw_text += "\n\n" + company_txt
                # 기업 메타 정보 보강
                company_meta_data = extract_company_meta_from_html(company_html)
                if not meta_hint.get("company_name") and company_meta_data.get("company_name"):
                     meta_hint["company_name"] = company_meta_data["company_name"]
                if not meta_hint.get("company_intro") and company_meta_data.get("company_intro"):
                     meta_hint["company_intro"] = company_meta_data["company_intro"]
                st.success(f"✅ 기업 홈페이지 크롤링 성공: {len(company_txt)}자 (정보 보강 완료)")
            else:
                 st.info("ℹ️ 기업 홈페이지 크롤링 실패 또는 보강할 내용 부족")

    st.session_state.raw_text = all_raw_text
    
    if not all_raw_text.strip():
        st.error("❌ 크롤링된 내용이 없습니다. URL을 다시 확인하거나 Selenium 옵션을 조정해 보세요.")
        return

    st.markdown("---")
    
    # 3. 정보 구조화
    with st.spinner("LLM을 이용한 정보 구조화 및 정제 중..."):
        clean_struct = structurize_and_refine(all_raw_text, meta_hint, CHAT_MODEL)
        st.session_state.clean_struct = clean_struct
        
        # JD 청크 생성 및 임베딩
        jd_chunks = chunk_text(all_raw_text, max_len=1800)
        jd_embeds = get_text_embedding(jd_chunks, EMBED_MODEL)
        st.session_state.job_desc_chunks = jd_chunks
        st.session_state.job_desc_embeds = jd_embeds
        
        st.success("✅ 정보 구조화 및 임베딩 완료")

# '정보 조회 및 구조화' 버튼
if st.button("🚀 정보 조회 및 구조화 시작", type="primary"):
    if not job_url:
        st.warning("채용 공고 URL을 입력해주세요.")
    else:
        fetch_and_process(job_url, company_url)

# 구조화된 정보 표시
if st.session_state.clean_struct:
    struct = st.session_state.clean_struct
    
    st.markdown("---")
    st.subheader("💡 구조화된 채용 정보")
    col1, col2 = st.columns(2)
    col1.metric("회사명", struct.get("company_name", "-"))
    col2.metric("직무", struct.get("job_title", "-"))
    st.info(f"**회사 소개:** {struct.get('company_intro', '-')}")
    
    st.markdown("#### 핵심 직무 요건")
    st.columns(3)[0].markdown("##### 주요 업무")
    for item in struct.get("responsibilities", []): st.caption(f"- {item}")
    st.columns(3)[1].markdown("##### 필수 자격")
    for item in struct.get("qualifications", []): st.caption(f"- {item}")
    st.columns(3)[2].markdown("##### 우대 사항")
    for item in struct.get("preferences", []): st.caption(f"- {item}")

# -----------------------------------------------------------------------------
# Resume Upload and Processing
# -----------------------------------------------------------------------------
if st.session_state.clean_struct:
    st.markdown("---")
    st.subheader("📝 이력서 등록 및 자소서 생성")
    
    uploaded_file = st.file_uploader("이력서 파일 등록 (TXT, PDF, DOCX 등 텍스트 추출 가능한 파일)", type=["txt", "pdf", "docx"])
    
    if uploaded_file:
        try:
            # 파일 형식에 따른 텍스트 추출 로직 (여기에 실제 로직 필요: 예. PyPDF2, docx 라이브러리)
            # 현재는 TXT 파일만 지원한다고 가정
            if uploaded_file.type == "text/plain":
                 st.session_state.resume_text = uploaded_file.read().decode("utf-8")
                 st.success("✅ 이력서(TXT) 내용 로드 완료")
            else:
                 st.warning("⚠️ 현재는 TXT 파일만 텍스트 추출을 지원합니다. 실제 서비스 시에는 PDF/DOCX 파싱 로직이 필요합니다.")
                 st.session_state.resume_text = ""
        except Exception as e:
            st.error(f"이력서 파일 처리 오류: {e}")
            st.session_state.resume_text = ""
            
    if st.session_state.resume_text:
        st.text_area("**이력서 원문 (편집 가능)**", value=st.session_state.resume_text, height=300, key="current_resume_text")
        
        if st.button("🌟 이력서 기반 맞춤 자소서 생성", type="secondary"):
            if not st.session_state.current_resume_text.strip():
                st.warning("이력서 내용을 입력해주세요.")
            else:
                st.session_state.resume_text = st.session_state.current_resume_text # 업데이트
                
                # RAG를 위한 이력서 청크/임베딩 생성
                with st.spinner("이력서 분석 및 임베딩 중..."):
                    resume_chunks = chunk_text(st.session_state.resume_text, max_len=1800)
                    resume_embeds = get_text_embedding(resume_chunks, EMBED_MODEL)
                    st.session_state.resume_chunks = resume_chunks
                    st.session_state.resume_embeds = resume_embeds
                    st.success("✅ 이력서 분석 완료")
                
                with st.spinner("AI 자기소개서 생성 중..."):
                    cover_letter = llm_generate_cover_letter(
                        st.session_state.clean_struct, st.session_state.resume_text, CHAT_MODEL
                    )
                    st.session_state.cover_letter = cover_letter
                
                st.markdown("#### 🎉 생성된 맞춤 자기소개서 초안")
                st.markdown(st.session_state.cover_letter)

# -----------------------------------------------------------------------------
# Interview Preparation
# -----------------------------------------------------------------------------
if st.session_state.get("cover_letter") or (st.session_state.resume_text and st.session_state.clean_struct):
    st.markdown("---")
    st.subheader("🎤 맞춤형 면접 대비")
    
    interview_source = st.session_state.get("cover_letter") or st.session_state.resume_text
    
    if not st.session_state.interview_questions:
        if st.button("🔍 맞춤 면접 질문 생성", type="primary"):
            if not st.session_state.resume_chunks:
                st.error("이력서 분석(임베딩)이 필요합니다. '이력서 기반 맞춤 자소서 생성' 버튼을 먼저 눌러주세요.")
            else:
                with st.spinner("맞춤 면접 질문을 생성 중입니다..."):
                    q_data = llm_generate_questions(
                        st.session_state.clean_struct, interview_source, CHAT_MODEL
                    )
                    st.session_state.interview_questions = q_data.get("questions", [])
                    st.session_state.followups = q_data.get("followup_questions", [])
                    st.success("✅ 면접 질문 생성 완료")
    
    if st.session_state.interview_questions:
        st.markdown("#### 면접 질문 목록")
        
        for i, q_item in enumerate(st.session_state.interview_questions, 1):
            q = q_item["question"]
            rel = q_item["relevance"]
            
            st.markdown(f"**({i}) {q}**")
            st.caption(f"관련 항목: {rel}")
            st.text_area(f"질문 {i}에 대한 나의 답변", key=f"answer_{i}", height=100)
            
            # 답변 채점/피드백 버튼
            if st.button(f"질문 {i} 채점 & 피드백 받기", key=f"score_btn_{i}", type="secondary"):
                answer = st.session_state.get(f"answer_{i}", "").strip()
                if not answer:
                    st.warning("답변을 작성해 주세요.")
                else:
                    with st.spinner(f"질문 {i} 답변 채점 중..."):
                        res_score = llm_score_and_coach_strict(
                            st.session_state.clean_struct, q, answer, CHAT_MODEL,
                            st.session_state.resume_chunks, st.session_state.resume_embeds,
                            st.session_state.job_desc_chunks, st.session_state.job_desc_embeds
                        )
                        st.session_state[f"feedback_{i}"] = res_score
                        st.session_state.followups.extend(res_score.get("next_followup", []))
                    
                    st.experimental_rerun() # 피드백 표시를 위해 새로고침

            # 피드백 표시
            feedback = st.session_state.get(f"feedback_{i}")
            if feedback:
                st.markdown(f"**[피드백] 총점: {feedback.get('overall_score', 0)}/100**")
                st.markdown(feedback.get("feedback_kr", "피드백 내용 없음"))
                if feedback.get("next_followup"):
                     st.info(f"💡 추가 팔로업 질문: {', '.join(feedback['next_followup'])}")

        st.markdown("---")
        st.markdown("#### 🔄 팔로업 질문 연습")
        st.info("이전에 생성된 또는 채점에서 추가된 팔로업 질문으로 심층 연습을 할 수 있습니다.")
        
        # 중복 제거
        st.session_state.followups = list(set([f for f in st.session_state.followups if f])) 

        if st.session_state.followups:
            selected_followup = st.selectbox("채점 받을 팔로업 질문 선택", st.session_state.followups, index=0, key="selected_followup")
            st.text_area("팔로업 질문에 대한 나의 답변", key="followup_answer", height=160)
            
            if st.button("팔로업 채점 & 피드백", type="secondary"):
                fu_q = st.session_state.get("selected_followup", ""); fu_ans = st.session_state.get("followup_answer", "")
                if not fu_q: st.warning("팔로업 질문을 선택하세요.")
                elif not fu_ans.strip(): st.warning("팔로업 답변을 작성하세요.")
                else:
                    with st.spinner("팔로업 채점 중..."):
                        res_fu = llm_score_and_coach_strict(
                            st.session_state.clean_struct, fu_q, fu_ans, CHAT_MODEL,
                            st.session_state.resume_chunks, st.session_state.resume_embeds,
                            st.session_state.job_desc_chunks, st.session_state.job_desc_embeds
                        )
                    st.markdown("---")
                    st.markdown(f"**[팔로업 결과] 질문: {fu_q}**")
                    st.metric("총점(/100)", res_fu.get("overall_score", 0))
                    st.markdown(res_fu.get("feedback_kr", "피드백 내용 없음"))
                    
                    # 추가 팔로업 질문이 있으면 세션에 추가
                    if res_fu.get("next_followup"):
                         st.info(f"💡 추가 팔로업 질문: {', '.join(res_fu['next_followup'])}")
                         st.session_state.followups.extend(res_fu["next_followup"])

        else:
            st.info("현재 추가 팔로업 질문이 없습니다. 기본 질문에 대한 채점을 진행하시면 새로운 질문이 추가될 수 있습니다.")