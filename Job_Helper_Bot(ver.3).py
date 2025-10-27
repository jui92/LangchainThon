# -*- coding: utf-8 -*-
################################################################################
# Job Helper Bot (Selenium-ONLY + Speed-up + Follow-up restored)
# - 채용 포털 URL + (선택) 기업 홈페이지 URL + 최신 뉴스 → 모의면접/자소서
# - Selenium 전용 수집(원티드 Next.js __NEXT_DATA__ 병합), 규칙 파서 보강
# - 속도 개선: Fast 모드, 동시 처리(ThreadPoolExecutor), 캐시(st.cache_data)
# - 팔로업 질문 · 답변 · 피드백 UI 복원
################################################################################

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

# Selenium (Selenium Manager 사용)
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# (선택) 문서 파서
try:
    import pypdf
except Exception:
    pypdf = None
try:
    import docx2txt
except Exception:
    docx2txt = None

# -----------------------------------------------------------------------------
# App/Sidebar
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Job Helper Bot", page_icon="🤖", layout="wide")
st.title("Job Helper Bot : 자소서 생성 / 모의 면접")

API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
if not API_KEY:
    API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password", help="OpenAI API 키가 필요합니다.")
if not API_KEY:
    st.stop()
client = OpenAI(api_key=API_KEY)

with st.sidebar:
    st.subheader("모델 / 크롤링 옵션")
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small","text-embedding-3-large"], index=0)
    SELENIUM_TIMEOUT = st.slider("Selenium 대기(초)", 6, 30, 14)
    FAST_MODE = st.toggle("Fast 모드(확장 최소화, 더 빠름)", value=True)
    st.caption("Fast 모드: 클릭/스크롤 시도 횟수를 줄여 속도를 높입니다.")

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
    driver = webdriver.Chrome(options=opts)  # Selenium Manager 자동
    return driver

# -----------------------------------------------------------------------------
# Domain expand helpers
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
                        time.sleep(0.15 if FAST_MODE else 0.25)
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
                    time.sleep(0.12 if FAST_MODE else 0.2)
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
    _click_many_css(driver, sel, per=(5 if FAST_MODE else 12))
    _click_by_text_candidates(driver, [
        "더보기","전체보기","자세히","상세보기","모두 보기",
        "주요업무","자격요건","우대사항","기업/팀 소개","혜택 및 복지",
        "나중에 하기","닫기","확인"
    ])

def _expand_saramin(driver):
    sel = [".btn_more",".btnMore",".btn-detail",".btn_toggle",
           "[aria-expanded='false']","[role='button']","button[class*='more'], a[class*='more']"]
    _click_many_css(driver, sel, per=(4 if FAST_MODE else 12))
    _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","기업정보","상세정보"])

def _expand_jobkorea(driver):
    sel = [".btnFold",".btnToggleRead",".btn_more",
           "[aria-expanded='false']","[role='button']","button[class*='More'], a[class*='More']"]
    _click_many_css(driver, sel, per=(4 if FAST_MODE else 12))
    _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","기업정보","상세보기"])

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
        "require","qualification",
        "prefer","plus","nice",
        "benefit","welfare","perk"
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
    return "\n".join(lines[:600])

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
        # light expand first
        _click_by_text_candidates(driver, ["더보기","상세보기","자세히 보기","전체보기","Read more","More"], per=(3 if FAST_MODE else 8))
        _click_by_text_candidates(driver, ["우대","우대사항","자격요건","주요업무","Requirements","Responsibilities","Preferred"], per=(3 if FAST_MODE else 8))

        if "wanted.co.kr" in host: _expand_wanted(driver)
        if "saramin" in host:     _expand_saramin(driver)
        if "jobkorea" in host:    _expand_jobkorea(driver)

        # scrolls
        loops = 3 if FAST_MODE else 7
        for _ in range(loops):
            try:
                driver.execute_script("window.scrollBy(0, 1200);"); time.sleep(0.15 if FAST_MODE else 0.25)
            except Exception:
                break

        html = driver.page_source or ""
        # Wanted extra
        if "wanted.co.kr" in host:
            try:
                txt_next = extract_wanted_from_next_html(html)
                if txt_next:
                    html += "\n<!-- TEXT_MERGE_SPLIT -->\n" + "\n".join([f"<p>{line}</p>" for line in txt_next.split("\n")])
            except Exception:
                pass
        return html
    finally:
        try: driver.quit()
        except Exception: pass

def fetch_all_text_selenium(url: str, timeout: int = 14) -> Tuple[str, Dict, Optional[str]]:
    url_n = normalize_url(url)
    if not url_n: return "", {"error":"invalid_url"}, None
    try:
        html = selenium_get_html(url_n, timeout=timeout)
    except Exception as e:
        st.error(f"Selenium 로드 실패: {e}")
        st.code("".join(traceback.format_exc()))
        return "", {"source":"selenium_error","len":0,"url_final":url_n}, None
    if not html or len(html) < 200:
        return "", {"source":"selenium_failed","len":0,"url_final":url_n}, None
    txt = html_to_text(html)
    return txt, {"source":"selenium","len":len(txt),"url_final":url_n}, html

# -----------------------------------------------------------------------------
# Meta & rule-based sections
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

    hdr_resp = re.compile(r"(주요\s*업무|담당\s*업무|Role|Responsibilities?)", re.I)
    hdr_qual = re.compile(r"(자격\s*요건|지원\s*자격|Requirements?|Qualifications?)", re.I)
    hdr_pref = re.compile(r"(우대\s*사항|우대|선호|Preferred|Nice\s*to\s*have|Plus)", re.I)
    out = {"responsibilities": [], "qualifications": [], "preferences": []}
    bucket=None

    def push(line,b):
        if line and len(line)>1 and line not in out[b]:
            out[b].append(line[:180])

    for l in lines:
        if hdr_resp.search(l): bucket="responsibilities"; continue
        if hdr_qual.search(l): bucket="qualifications"; continue
        if hdr_pref.search(l): bucket="preferences"; continue
        if bucket is None:
            if hdr_pref.search(l): bucket="preferences"
            elif any(k in l.lower() for k in ["java","python","spring","kafka","ml","sql"]):
                bucket="responsibilities"
            else: continue
        push(l,bucket)

    kw_pref = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
    remain=[]
    for q in out["qualifications"]:
        (out["preferences"] if kw_pref.search(q) else remain).append(q)
    out["qualifications"]=remain

    for k in out:
        seen=set(); clean=[]
        for s in out[k]:
            s=re.sub(r"\s+"," ", s).strip(" -•·▶▪️").strip()
            if s and s not in seen:
                seen.add(s); clean.append(s)
        out[k]=clean[:14]
    return out

# -----------------------------------------------------------------------------
# LLM structure / Q&A / scoring
# -----------------------------------------------------------------------------
PROMPT_SYSTEM_STRUCT = ("너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
                        "입력 텍스트는 광고/UX잔재/복수 직무가 섞여 있을 수 있다. "
                        "한국어로 간결하고 중복없이 정제하라.")

def llm_structurize(raw_text: str, meta_hint: Dict[str, str], model: str) -> Dict:
    ctx = clean_text(raw_text, 14000)
    user_msg = {"role": "user", "content": (
        "다음 채용 공고 원문을 구조화해줘.\n\n"
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
        "- '우대 사항(preferences)'은 원문 표기가 있는 항목만 담되, 비워두지 않도록 규칙 파서로 보강.\n"
        "- 불릿/이모지 제거, 문장 간결화, 중복 제거."
    )}
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=900,
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg]
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        data = {"company_name": meta_hint.get("company_name",""),
                "company_intro": meta_hint.get("company_intro","원문 정제 실패"),
                "job_title": meta_hint.get("job_title",""),
                "responsibilities": [], "qualifications": [], "preferences": [], "error": str(e)}

    for k in ["responsibilities","qualifications","preferences"]:
        arr = data.get(k, []); 
        if not isinstance(arr, list): arr=[]
        clean_list=[]; seen=set()
        for it in arr:
            t = re.sub(r"\s+"," ", str(it)).strip(" -•·▶▪️").strip()
            if t and t not in seen:
                seen.add(t); clean_list.append(t[:180])
        data[k] = clean_list[:14]

    if len(data.get("preferences", [])) < 1:
        rb = rule_based_sections(ctx)
        if rb.get("preferences"):
            merged = data.get("preferences", []) + rb["preferences"]
            data["preferences"] = list(dict.fromkeys(merged))[:14]
        else:
            kw_pref = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
            remain=[]; moved=[]
            for q in data.get("qualifications", []):
                (moved if kw_pref.search(q) else remain).append(q)
            data["preferences"]=moved[:14]; data["qualifications"]=remain[:14]

    for k in ["company_name","company_intro","job_title"]:
        if isinstance(data.get(k), str): data[k]=re.sub(r"\s+"," ", data[k]).strip()
    return data

def chunk(text: str, size: int = 600, overlap: int = 120) -> List[str]:
    t = re.sub(r"\s+"," ", text or "").strip()
    if not t: return []
    out, start = [], 0; L=len(t)
    while start < L:
        end = min(L, start+size); out.append(t[start:end])
        if end==L: break
        start = max(0, end-overlap)
    return out

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    if not texts: return np.zeros((0,1536), dtype=np.float32)
    resp = client.embeddings.create(model=model_name, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def cosine_topk(matrix: np.ndarray, query_vec: np.ndarray, k: int = 4):
    if matrix.size==0: return np.array([]), np.array([], dtype=int)
    qn = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn.T; sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx

def retrieve_resume_chunks(query: str, chunks: List[str], embeds: np.ndarray, k: int = 4):
    if not chunks or embeds is None or embeds.size==0: return []
    qv = embed_texts([query], EMBED_MODEL)
    scores, idxs = cosine_topk(embeds, qv, k=k)
    return [(float(s), chunks[int(i)]) for s, i in zip(scores, idxs)]

PROMPT_SYSTEM_Q = ("너는 채용담당자다. 회사/직무 맥락과 채용요건, 지원자 이력서를 함께 고려해 "
                   "서로 겹치지 않는 고품질 한국어 면접 질문을 만든다. 수치/지표/기간/규모/리스크/트레이드오프를 섞어라.")
PROMPT_SYSTEM_DRAFT = ("너는 면접 답변 코치다. 회사/직무/채용요건과 지원자의 이력서 요약을 결합해 "
                       "STAR(상황-과제-행동-성과)로 8~12문장 답변 **초안**을 한국어로 작성한다. "
                       "가능하면 지표/수치/기간/임팩트를 포함하라.")
PROMPT_SYSTEM_SCORE_STRICT = ("너는 매우 엄격한 톱티어 면접 코치다. 아래 형식의 JSON만 출력하라. "
                              "각 기준은 0~20 정수이며 총점은 합계(100)와 일치해야 한다.")
CRITERIA = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def llm_generate_one_question_with_resume(clean: Dict, level: str, model: str,
                                          resume_chunks: List[str], resume_embeds: np.ndarray) -> str:
    hits = retrieve_resume_chunks("핵심 프로젝트와 기술 스택 요약", resume_chunks, resume_embeds, k=4)
    resume_context = "\n".join([f"- {t[:350]}" for _, t in hits])[:1200]
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {"role":"user","content":(
        f"[회사/직무/요건]\n{ctx}\n\n"
        f"[지원자 이력서 요약(발췌)]\n{resume_context}\n\n"
        f"[요청]\n- 난이도/연차: {level}\n- 중복/유사도 지양\n- 한국어 면접 질문 1개만 한 줄로 출력")}
    try:
        r = client.chat.completions.create(model=model, temperature=0.8, max_tokens=120,
                                           messages=[{"role":"system","content":PROMPT_SYSTEM_Q}, user_msg])
        q = r.choices[0].message.content.strip()
        q = re.sub(r"^\s*\d+[\).\s-]*","", q).split("\n")[0].strip()
        return q
    except Exception:
        return ""

def llm_draft_answer(clean: Dict, question: str, model: str,
                     resume_chunks: List[str], resume_embeds: np.ndarray) -> str:
    hits = retrieve_resume_chunks(question, resume_chunks, resume_embeds, k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {"role":"user","content":(
        f"[회사/직무/채용요건]\n{ctx}\n\n[지원자 이력서 발췌]\n{resume_text}\n\n[면접 질문]\n{question}\n\n"
        "위 정보를 근거로 STAR 기반 한국어 답변 **초안**을 작성해줘.")}
    try:
        r = client.chat.completions.create(model=model, temperature=0.5, max_tokens=700,
                                           messages=[{"role":"system","content":PROMPT_SYSTEM_DRAFT}, user_msg])
        return r.choices[0].message.content.strip()
    except Exception:
        return ""

def llm_score_and_coach_strict(clean: Dict, question: str, answer: str, model: str,
                               resume_chunks: List[str], resume_embeds: np.ndarray) -> Dict:
    hits = retrieve_resume_chunks(question + "\n" + answer[:800], resume_chunks, resume_embeds, k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {"role":"user","content":(
        f"[회사/직무/채용요건]\n{ctx}\n\n[지원자 이력서 발췌]\n{resume_text}\n\n"
        f"[면접 질문]\n{question}\n\n[지원자 답변]\n{answer}\n\n"
        "다음 JSON 스키마로만 한국어 응답:\n"
        "{"
        "\"overall_score\": 0~100 정수,"
        "\"criteria\": [{\"name\":\"문제정의\",\"score\":0~20,\"comment\":\"...\"},"
        "{\"name\":\"데이터/지표\",\"score\":0~20,\"comment\":\"...\"},"
        "{\"name\":\"실행력/주도성\",\"score\":0~20,\"comment\":\"...\"},"
        "{\"name\":\"협업/커뮤니케이션\",\"score\":0~20,\"comment\":\"...\"},"
        "{\"name\":\"고객가치\",\"score\":0~20,\"comment\":\"...\"}],"
        "\"strengths\": [\"...\"],\"risks\": [\"...\"],\"improvements\": [\"...\",\"...\",\"...\"],"
        "\"revised_answer\": \"STAR 구조로 간결히\""
        "}"
    )}
    try:
        r = client.chat.completions.create(model=model, temperature=0.2, max_tokens=900,
                                           response_format={"type":"json_object"},
                                           messages=[{"role":"system","content":PROMPT_SYSTEM_SCORE_STRICT}, user_msg])
        data = json.loads(r.choices[0].message.content)
        fixed=[]
        for name in CRITERIA:
            found=None
            for it in data.get("criteria", []):
                if str(it.get("name","")).strip()==name: found=it; break
            if not found: found={"name":name,"score":0,"comment":""}
            sc=int(found.get("score",0)); sc=max(0,min(20,sc))
            found["score"]=sc; found["comment"]=str(found.get("comment","")).strip()
            fixed.append(found)
        total=sum(x["score"] for x in fixed)
        data["criteria"]=fixed; data["overall_score"]=total
        for k in ["strengths","risks","improvements"]:
            arr=data.get(k,[]); 
            if not isinstance(arr,list): arr=[]
            data[k]=[str(x).strip() for x in arr if str(x).strip()][:5]
        data["revised_answer"]=str(data.get("revised_answer","")).strip()
        return data
    except Exception as e:
        return {"overall_score":0,"criteria":[{"name":n,"score":0,"comment":""} for n in CRITERIA],
                "strengths": [],"risks": [],"improvements": [],"revised_answer":"", "error":str(e)}

# -----------------------------------------------------------------------------
# Company pages / news (캐시 + 동시 처리)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _http_get(url: str, timeout: int = 8) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
            "Accept-Language":"ko, en;q=0.9"
        })
        if r.status_code==200: return r.text
    except Exception:
        pass
    return ""

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_company_pages(home_url: str) -> Dict[str, List[str]]:
    out = {"vision": [], "talent": []}
    base = normalize_url(home_url or "")
    if not base: return out
    paths = ["","/","/about","/company","/about-us","/mission","/vision","/values","/culture","/careers","/talent","/people"]
    seen=set()
    for p in paths:
        url = (base.rstrip("/") + p) if p else base
        if url in seen: continue
        seen.add(url)
        html = _http_get(url, timeout=6)
        if not html: continue
        soup = BeautifulSoup(html, "lxml")
        texts=[]
        for tag in soup.find_all(["h1","h2","h3","h4","p","li"]):
            t = tag.get_text(" ", strip=True)
            if not t: continue
            t = re.sub(r"\s+"," ", t)
            if 6 <= len(t) <= 260: texts.append(t)
        for t in texts:
            low=t.lower()
            if any(k in low for k in ["talent","인재상","who we hire","people we"]):
                out["talent"].append(t)
            if any(k in low for k in ["비전","미션","핵심가치","가치","원칙","mission","vision","values","principle"]):
                out["vision"].append(t)
    for k in out:
        uniq=[]; s=set()
        for x in out[k]:
            x=x.strip()
            if x and x not in s:
                s.add(x); uniq.append(x[:200])
        out[k]=uniq[:12]
    return out

@st.cache_data(show_spinner=False, ttl=1200)
def google_news_rss(company: str, max_items: int = 5) -> List[Dict]:
    if not company: return []
    q = urllib.parse.quote(company)
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        r = requests.get(url, timeout=6)
        if r.status_code != 200: return []
        soup = BeautifulSoup(r.text, "xml")
        out=[]
        for it in soup.find_all("item")[:max_items]:
            out.append({"title": (it.title.get_text() if it.title else "").strip(),
                        "link": (it.link.get_text() if it.link else "").strip(),
                        "pubDate": (it.pubDate.get_text() if it.pubDate else "").strip()})
        return out
    except Exception:
        return []

# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
def _init_state():
    defaults = {
        "clean_struct": None,
        "resume_raw": "",
        "resume_chunks": [],
        "resume_embeds": None,
        "current_question": "",
        "answer_text": "",
        "last_result": None,
        "followups": [],
        "selected_followup": "",
        "followup_answer": "",
        "company_home": "",
        "company_vision": [],
        "company_talent": [],
        "company_news": [],
        "last_html": None,
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v
_init_state()

# -----------------------------------------------------------------------------
# UI 1) 채용 공고 URL + (선택) 기업 홈페이지 URL
# -----------------------------------------------------------------------------
st.header("1) 채용 공고 URL (Selenium 전용)")
job_url = st.text_input("채용 공고 상세 URL", placeholder="원티드/사람인/잡코리아/기업 채용 페이지 URL")
st.text_input("회사 공식 홈페이지 URL (선택)", key="company_home", placeholder="회사 공식 홈페이지 URL을 입력하세요.")

if st.button("원문 수집 → 정제 (Selenium ONLY)", type="primary"):
    if not job_url.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner("Selenium으로 원문 수집 중..."):
            raw, meta, html = fetch_all_text_selenium(job_url.strip(), timeout=SELENIUM_TIMEOUT)
            hint = extract_company_meta_from_html(html)
            st.session_state.last_html = html

        st.caption(f"수집 소스: {meta.get('source')} · 텍스트 길이: {meta.get('len')}")
        if not raw:
            st.error("수집 실패(로그인/동적 렌더링/봇 차단 가능).")
        else:
            # 동시 처리: LLM 정제 + (홈페이지/뉴스)
            with st.spinner("정제 및 부가정보 수집 중..."):
                tasks=[]
                with ThreadPoolExecutor(max_workers=3) as ex:
                    tasks.append(("clean", ex.submit(llm_structurize, raw, hint, CHAT_MODEL)))
                    if st.session_state.company_home.strip():
                        tasks.append(("pages", ex.submit(fetch_company_pages, st.session_state.company_home.strip())))
                    cname = hint.get("company_name","")
                    tasks.append(("news", ex.submit(google_news_rss, cname, 5)))

                    clean=None; vis=[]; tal=[]; news=[]
                    for name, fut in tasks:
                        try:
                            res = fut.result()
                            if name=="clean": clean=res
                            elif name=="pages":
                                vis = res.get("vision", []); tal = res.get("talent", [])
                            elif name=="news": news = res or []
                        except Exception:
                            continue
                # 규칙 파서 보강
                if clean and not clean.get("preferences"):
                    rb = rule_based_sections(raw)
                    if rb.get("preferences"): clean["preferences"]=rb["preferences"]
                st.session_state.clean_struct = clean
                st.session_state.company_vision = vis
                st.session_state.company_talent = tal
                st.session_state.company_news = news
            st.success("정제 완료!")

# -----------------------------------------------------------------------------
# UI 2) 회사 요약
# -----------------------------------------------------------------------------
st.header("2) 회사 요약")
clean = st.session_state.clean_struct
if clean:
    st.markdown(f"**회사명:** {clean.get('company_name','-')}")
    st.markdown(f"**간단한 회사 소개(요약):** {clean.get('company_intro','-')}")
    st.markdown(f"**모집 분야(직무명):** {clean.get('job_title','-')}")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**주요 업무**");   [st.markdown(f"- {b}") for b in clean.get("responsibilities", [])]
    with c2:
        st.markdown("**자격 요건**");   [st.markdown(f"- {b}") for b in clean.get("qualifications", [])]
    with c3:
        st.markdown("**우대 사항**")
        prefs = clean.get("preferences", [])
        if prefs: [st.markdown(f"- {b}") for b in prefs]
        else: st.caption("우대 사항이 명시되지 않았습니다.")
else:
    st.info("먼저 URL을 정제해 주세요.")

# 비전/인재상/뉴스
if st.session_state.company_vision or st.session_state.company_talent or st.session_state.company_news:
    st.divider()
    st.subheader("회사 비전/인재상 & 최신 이슈")
    vcol, tcol = st.columns(2)
    with vcol:
        st.markdown("**비전/핵심가치**")
        for v in st.session_state.company_vision[:8]: st.markdown(f"- {v}")
        if not st.session_state.company_vision: st.caption("비전/핵심가치를 찾지 못했습니다.")
    with tcol:
        st.markdown("**인재상**")
        for t in st.session_state.company_talent[:8]: st.markdown(f"- {t}")
        if not st.session_state.company_talent: st.caption("인재상 정보를 찾지 못했습니다.")
    if st.session_state.company_news:
        st.markdown("**최신 뉴스(상위 3~5)**")
        for n in st.session_state.company_news[:5]:
            st.markdown(f"- [{n.get('title','(제목 없음)')}]({n.get('link','#')})")

st.divider()

# -----------------------------------------------------------------------------
# UI 3) 이력서 업로드/인덱싱
# -----------------------------------------------------------------------------
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader("여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True)
_RESUME_CHUNK=500; _RESUME_OVLP=100

def read_pdf(data: bytes) -> str:
    if pypdf is None: return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
    except Exception: return ""

def read_docx_file(data: bytes) -> str:
    if docx2txt is None: return ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
            tmp.write(data); tmp.flush()
            return docx2txt.process(tmp.name) or ""
    except Exception: return ""

def read_file_text(uploaded) -> str:
    name = uploaded.name.lower(); data = uploaded.read()
    if name.endswith((".txt",".md")):
        for enc in ("utf-8","cp949","euc-kr"):
            try: return data.decode(enc)
            except Exception: continue
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):  return read_pdf(data)
    if name.endswith(".docx"): return read_docx_file(data)
    return ""

if st.button("이력서 인덱싱", type="secondary"):
    if not uploads: st.warning("파일을 업로드하세요.")
    else:
        all_text=[]
        for up in uploads:
            t = read_file_text(up)
            if t: all_text.append(t)
        resume_text = "\n\n".join(all_text)
        if not resume_text.strip(): st.error("텍스트 추출 실패")
        else:
            chunks = chunk(resume_text, size=_RESUME_CHUNK, overlap=_RESUME_OVLP)
            with st.spinner("이력서 벡터화 중..."):
                embeds = embed_texts(chunks, EMBED_MODEL)
            st.session_state.resume_raw = resume_text
            st.session_state.resume_chunks = chunks
            st.session_state.resume_embeds = embeds
            st.success(f"인덱싱 완료 (청크 {len(chunks)}개)")

st.divider()

# -----------------------------------------------------------------------------
# UI 4) 자소서 생성
# -----------------------------------------------------------------------------
st.header("4) 이력서 기반 자소서 생성")
topic = st.text_input("회사 요청 주제(선택)", placeholder="예: 지원동기 / 협업 경험 / 문제해결 사례 등")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str) -> str:
    enrich = {"vision": st.session_state.company_vision[:6],
              "talent": st.session_state.company_talent[:6],
              "news": [n.get("title","") for n in st.session_state.company_news[:3]]}
    company = json.dumps({"clean":clean_struct, "extra":enrich}, ensure_ascii=False)
    resume_snip = resume_text.strip()[:9000]
    system = ("너는 한국어 자기소개서 전문가다. 회사/직무 요건과 후보자의 이력서를 참고해 "
              "회사 특화 자소서를 작성한다. 과장/허위 금지, 수치/지표/기간/임팩트 중심.")
    req = (f"회사 측 요청 주제는 '{topic_hint.strip()}' 이다. 이 주제를 중심으로 서술." 
           if topic_hint and topic_hint.strip()
           else "특정 주제가 없으므로 채용요건, 비전/인재상과의 정합성을 강조.")
    user = (f"[회사/직무 요약(JSON)]\n{company}\n\n[후보자 이력서]\n{resume_snip}\n\n"
            f"[작성 지시]\n- {req}\n- 분량 600~900자\n"
            "- 구성: 지원동기→역량/경험→성과/지표→입사 후 기여→마무리\n"
            "- 중복/미사여구 제거, 자연스러운 1인칭.")
    try:
        r = client.chat.completions.create(model=model, temperature=0.4, max_tokens=800,
                                           messages=[{"role":"system","content":system},{"role":"user","content":user}])
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"(자소서 생성 실패: {e})"

if st.button("자소서 생성", type="primary"):
    if not st.session_state.clean_struct:
        st.warning("먼저 회사 URL을 정제하세요.")
    elif not st.session_state.resume_raw.strip():
        st.warning("먼저 이력서를 업로드하고 '이력서 인덱싱'을 눌러주세요.")
    else:
        with st.spinner("자소서 생성 중..."):
            cover = build_cover_letter(st.session_state.clean_struct, st.session_state.resume_raw, topic, CHAT_MODEL)
        st.subheader("자소서 (생성 결과)"); st.write(cover)
        st.download_button("자소서 TXT 다운로드", data=cover.encode("utf-8"),
                           file_name="cover_letter.txt", mime="text/plain")

st.divider()

# -----------------------------------------------------------------------------
# UI 5) 질문 생성 & 답변 초안
# -----------------------------------------------------------------------------
st.header("5) 질문 생성 & 답변 초안 (RAG 결합)")
level = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)

c1, c2 = st.columns(2)
with c1:
    if st.button("새 질문 받기", type="primary"):
        if not st.session_state.clean_struct:
            st.warning("먼저 회사 URL을 정제하세요.")
        else:
            q = llm_generate_one_question_with_resume(
                st.session_state.clean_struct, level, CHAT_MODEL,
                st.session_state.resume_chunks, st.session_state.resume_embeds
            )
            if q:
                st.session_state.current_question = q
                st.session_state.answer_text = ""
                st.session_state.last_result = None
                st.session_state.followups = []
                st.session_state.selected_followup = ""
                st.session_state.followup_answer = ""
                st.success("질문 생성 완료!")
            else:
                st.error("질문 생성 실패")
with c2:
    if st.button("RAG로 답변 초안 생성", type="secondary"):
        if not st.session_state.current_question:
            st.warning("먼저 질문을 생성하세요.")
        else:
            draft = llm_draft_answer(
                st.session_state.clean_struct, st.session_state.current_question, CHAT_MODEL,
                st.session_state.resume_chunks, st.session_state.resume_embeds
            )
            if draft:
                st.session_state.answer_text = draft
                st.success("초안 생성 완료!")
            else:
                st.error("초안 생성 실패")

st.text_area("질문", value=st.session_state.current_question, height=100)
st.text_area("나의 답변 (초안을 편집해 완성)", key="answer_text", height=200)

# -----------------------------------------------------------------------------
# UI 6) 채점 & 코칭
# -----------------------------------------------------------------------------
st.header("6) 채점 & 코칭")
if st.button("채점 & 코칭 실행", type="primary"):
    if not st.session_state.current_question:
        st.warning("먼저 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("채점/코칭 중..."):
            res = llm_score_and_coach_strict(
                st.session_state.clean_struct,
                st.session_state.current_question,
                st.session_state.answer_text,
                CHAT_MODEL,
                st.session_state.resume_chunks,
                st.session_state.resume_embeds
            )
        st.session_state.last_result = res
        st.success("채점/코칭 완료!")

st.divider()
st.subheader("피드백 결과")
last = st.session_state.last_result
if last:
    st.metric("총점(/100)", last.get("overall_score", 0))
    st.markdown("**기준별 점수 & 코멘트**")
    for it in last.get("criteria", []):
        st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
    if last.get("strengths"):
        st.markdown("**강점**"); [st.markdown(f"- {s}") for s in last["strengths"]]
    if last.get("risks"):
        st.markdown("**감점 요인/리스크**"); [st.markdown(f"- {r}") for r in last["risks"]]
    if last.get("improvements"):
        st.markdown("**개선 포인트**"); [st.markdown(f"- {im}") for im in last["improvements"]]
    if last.get("revised_answer"):
        st.markdown("**수정본 답변(STAR)**"); st.write(last["revised_answer"])
else:
    st.info("아직 채점 결과가 없습니다.")

st.divider()

# -----------------------------------------------------------------------------
# UI 7) 팔로업 질문 · 답변 · 피드백 (복원)
# -----------------------------------------------------------------------------
st.subheader("팔로업 질문 · 답변 · 피드백")

# 메인 피드백 직후 자동 제안
if last and not st.session_state.followups:
    try:
        ctx = json.dumps({"company": st.session_state.clean_struct,
                          "vision": st.session_state.company_vision[:6],
                          "talent": st.session_state.company_talent[:6],
                          "news": [n.get("title","") for n in st.session_state.company_news[:3]]},
                         ensure_ascii=False)
        msg = {"role":"user","content":(
            f"[회사/직무/요건/비전/이슈]\n{ctx}\n\n[지원자 답변]\n{st.session_state.answer_text}\n\n"
            "면접관 관점에서 팔로업 질문 3개를 한 줄씩 한국어로 제안해줘. "
            "기존 질문과 중복되지 않게, 지표/리스크/트레이드오프/의사결정 근거를 섞어줘.")}
        r = client.chat.completions.create(model=CHAT_MODEL, temperature=0.7,
                                           messages=[{"role":"system","content":"면접 팔로업 생성기"}, msg])
        followups = [re.sub(r'^\s*\d+[\).\s-]*','', l).strip()
                     for l in r.choices[0].message.content.splitlines() if l.strip()]
        st.session_state.followups = followups[:3]
    except Exception:
        st.session_state.followups = []

if last:
    if st.session_state.followups:
        st.markdown("**팔로업 질문 제안**")
        for i, f in enumerate(st.session_state.followups, 1): st.markdown(f"- ({i}) {f}")
        st.selectbox("채점 받을 팔로업 질문 선택", st.session_state.followups, index=0, key="selected_followup")
        st.text_area("팔로업 질문에 대한 나의 답변", key="followup_answer", height=160)
        if st.button("팔로업 채점 & 피드백", type="secondary"):
            fu_q = st.session_state.get("selected_followup",""); fu_ans = st.session_state.get("followup_answer","")
            if not fu_q: st.warning("팔로업 질문을 선택하세요.")
            elif not fu_ans.strip(): st.warning("팔로업 답변을 작성하세요.")
            else:
                with st.spinner("팔로업 채점 중..."):
                    res_fu = llm_score_and_coach_strict(
                        st.session_state.clean_struct, fu_q, fu_ans, CHAT_MODEL,
                        st.session_state.resume_chunks, st.session_state.resume_embeds
                    )
                st.markdown("**팔로업 결과**")
                st.metric("총점(/100)", res_fu.get("overall_score", 0))
                for it in res_fu.get("criteria", []):
                    st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
                if res_fu.get("revised_answer",""):
                    st.markdown("**팔로업 수정본(STAR)**"); st.write(res_fu["revised_answer"])
    else:
        st.caption("팔로업 질문은 메인 질문 채점 직후 자동 제안됩니다.")
