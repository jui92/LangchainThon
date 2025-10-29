# -*- coding: utf-8 -*-
################################################################################
# Job Helper Bot - FULL (Precision Tuned: section scoring, noise stripping)
# 실행: streamlit run Job_Helper_Bot_full_precise.py
################################################################################
import os, re, io, json, time, shutil, tempfile, urllib.parse
from typing import List, Dict, Tuple, Optional
from collections import Counter

import streamlit as st
import numpy as np
import requests
import html2text
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET

# ======================= OpenAI ===========================
try:
    from openai import OpenAI
except ImportError:
    st.error("pip install openai 로 설치 후 다시 실행하세요."); st.stop()
API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st,"secrets") else None)
if not API_KEY: API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password")
if not API_KEY: st.stop()
client = OpenAI(api_key=API_KEY)

# ======================= Sidebar =========================
with st.sidebar:
    st.subheader("모델 / 크롤링 옵션")
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small","text-embedding-3-large"], index=0)
    SELENIUM_TIMEOUT = st.slider("Selenium 대기(초)", 8, 30, 16)
    MAX_PAGES = st.slider("회사 홈페이지 최대 페이지 수", 6, 30, 14)
    PREFER_DYNAMIC = st.toggle("회사 홈페이지: 동적 우선", value=True)
    FAST_MODE = st.toggle("FAST 모드(스크롤 최소화)", value=True)

# ======================= Utils ===========================
def _h2t():
    conv = html2text.HTML2Text(); conv.ignore_links=True; conv.ignore_images=True; conv.body_width=0
    return conv
H2T = _h2t()

def html_to_text(html_str: str) -> str:
    txt = H2T.handle(html_str or ""); txt = re.sub(r"\n{3,}", "\n\n", txt)
    return re.sub(r"\s+"," ", txt).strip()

def clean_text(s: str, max_len: int = 16000) -> str:
    if not s: return ""
    s = re.sub(r"\r","",s); s = re.sub(r"\s+"," ",s).strip()
    return s[:max_len] if len(s)>max_len else s

def normalize_url(u: str) -> Optional[str]:
    if not u: return None
    u=u.strip();  u = u if re.match(r"^https?://", u) else "https://"+u
    parts = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

def soup_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html or "", "html.parser")

def parse_sitemap_locations(xml_text: str) -> List[str]:
    out=[]
    if not xml_text: return out
    try:
        root = ET.fromstring(xml_text); ns={}
        if root.tag.startswith("{"):
            uri = root.tag.split("}")[0].strip("{"); ns={"ns":uri}
            locs = root.findall(".//ns:loc", ns)
        else:
            locs = root.findall(".//loc")
        for loc in locs:
            if loc is not None and loc.text: out.append(loc.text.strip())
    except Exception:
        out = re.findall(r"<loc>\s*([^<>\s]+)\s*</loc>", xml_text, flags=re.I)
    return out

# ======================= Selenium ========================
HAS_SELENIUM = True
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except Exception:
    HAS_SELENIUM = False
    st.error("Selenium 필요: pip install selenium"); st.stop()

def _pick_chrome_binary()->Optional[str]:
    cands=[os.getenv("CHROME_BIN"),os.getenv("GOOGLE_CHROME_BIN"),
           shutil.which("chromium"),shutil.which("chromium-browser"),shutil.which("google-chrome"),
           "/usr/bin/chromium","/usr/bin/chromium-browser","/usr/bin/google-chrome-stable"]
    return next((p for p in cands if p and os.path.exists(p)), None)

def build_driver(headless: bool=True):
    opts=ChromeOptions()
    if headless: opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox"); opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu"); opts.add_argument("--window-size=1440,2400"); opts.add_argument("--lang=ko-KR")
    p=_pick_chrome_binary()
    if p: opts.binary_location=p
    return webdriver.Chrome(options=opts)

def wait_dom(d, t): 
    try: WebDriverWait(d, t).until(EC.presence_of_element_located((By.XPATH,"//*")))
    except Exception: pass

def smart_scroll(d, loops=8):
    loops = 3 if FAST_MODE else loops
    for _ in range(loops):
        try: d.execute_script("window.scrollBy(0, document.body.scrollHeight*0.7);"); time.sleep(0.12 if FAST_MODE else 0.3)
        except Exception: break

def click_more_generic(d):
    for t in ["더보기","전체보기","상세보기","자세히 보기","더 보기","Read more","More","보기"]:
        for xp in (f"//*[normalize-space(text())='{t}']", f"//*[contains(normalize-space(text()), '{t}')]"):
            try:
                for el in d.find_elements(By.XPATH, xp)[:8]:
                    try: d.execute_script("arguments[0].click();", el); time.sleep(0.08 if FAST_MODE else 0.18)
                    except Exception: pass
            except Exception: pass

def expand_portal_specific(d, host):
    sels=[]
    if "wanted.co.kr" in host: sels=["[data-qa='btn-read-more']","[data-qa='job-header__more']","button[aria-expanded='false']"]
    if "saramin" in host:     sels+=[".btn_more",".btn-detail",".btn_toggle"]
    if "jobkorea" in host:    sels+=[".btnFold",".btnToggleRead",".btn_more"]
    for sel in sels:
        try:
            for el in d.find_elements(By.CSS_SELECTOR, sel)[:10]:
                try: d.execute_script("arguments[0].click();", el); time.sleep(0.08 if FAST_MODE else 0.18)
                except Exception: pass
        except Exception: pass

def get_html_dynamic(url: str, timeout: int) -> str:
    url = normalize_url(url) or url
    d = build_driver(headless=True)
    try:
        d.set_page_load_timeout(timeout); d.get(url); wait_dom(d, timeout)
        host = urllib.parse.urlsplit(url).netloc.lower()
        expand_portal_specific(d, host); click_more_generic(d); smart_scroll(d, 10)
        return d.page_source or ""
    finally:
        try: d.quit()
        except Exception: pass

# ================= 채용공고 수집/정제 =====================
def extract_wanted_next(html: str) -> str:
    try:
        soup = soup_html(html); tag = soup.select_one("script#__NEXT_DATA__")
        if not tag: return ""; data = json.loads((tag.string or tag.text or "").strip())
    except Exception: return ""
    keys = ["job","position","title","desc","description","responsibilit","duty","role","skill","require","qualification","prefer","plus","nice"]
    def walk(d, out):
        if isinstance(d, dict):
            for k,v in d.items():
                if any(t in str(k).lower() for t in keys):
                    if isinstance(v,str): out.append(v)
                    elif isinstance(v,list):
                        for it in v: walk(it,out)
                    elif isinstance(v,dict):
                        for subv in v.values(): walk(subv,out)
                walk(v,out)
        elif isinstance(d,list):
            for it in d: walk(it,out)
    bucket=[]; walk(data,bucket)
    seen=set(); lines=[]
    for t in bucket:
        s=re.sub(r"\s+"," ", (t or "")).strip()
        if len(s)>2 and s not in seen: seen.add(s); lines.append(s)
    return "\n".join(lines[:900])

def fetch_job_text_dynamic(url: str, timeout: int)->Tuple[str, Dict, Optional[str]]:
    html = get_html_dynamic(url, timeout=timeout)
    if not html or len(html)<200: return "", {"source":"selenium_failed","len":0,"url_final":url}, None
    if "wanted.co.kr" in urllib.parse.urlsplit(url).netloc.lower():
        try:
            extra = extract_wanted_next(html)
            if extra: html += "\n<div id='__WANTED_NEXT_EXTRACT__'>" + "".join(f"<p>{l}</p>" for l in extra.split("\n")) + "</div>"
        except Exception: pass
    txt = html_to_text(html)
    return txt, {"source":"selenium","len":len(txt),"url_final":url}, html

# ================== 규칙 파서(업무/자격/우대) ==============
def rule_based_sections(raw_text: str)->dict:
    txt = clean_text(raw_text, 16000)
    lines = [re.sub(r"\s+"," ", l).strip(" -•·▶▪️") for l in txt.split("\n") if l.strip()]
    hdr_resp = re.compile(r"(주요\s*업무|담당\s*업무|Role|Responsibilities?)", re.I)
    hdr_qual = re.compile(r"(자격\s*요건|지원\s*자격|Requirements?|Qualifications?)", re.I)
    hdr_pref = re.compile(r"(우대\s*사항|우대|선호|Preferred|Nice\s*to\s*have|Plus)", re.I)
    out={"responsibilities":[], "qualifications":[], "preferences":[]}
    bucket=None
    def push(line,b): 
        if line and len(line)>1 and line not in out[b]: out[b].append(line[:180])
    for l in lines:
        if hdr_resp.search(l): bucket="responsibilities"; continue
        if hdr_qual.search(l): bucket="qualifications"; continue
        if hdr_pref.search(l): bucket="preferences"; continue
        if bucket is None:
            low=l.lower()
            if hdr_pref.search(l): bucket="preferences"
            elif any(k in low for k in ["java","python","spring","kotlin","react","next","kafka","sql","ml","cloud","aws","gcp"]):
                bucket="responsibilities"
            else: continue
        push(l,bucket)
    # ‘우대’ 섞임 정리
    kw_pref=re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
    remain=[]; 
    for q in out["qualifications"]:
        (out["preferences"] if kw_pref.search(q) else remain).append(q)
    out["qualifications"]=remain
    for k in out:
        seen=set(); clean=[]
        for s in out[k]:
            s=re.sub(r"\s+"," ", s).strip(" -•·▶▪️").strip()
            if s and s not in seen: seen.add(s); clean.append(s)
        out[k]=clean[:14]
    return out

# =================== LLM 구조화 ===========================
PROMPT_SYSTEM_STRUCT = "너는 채용 공고를 한국어로 간결하고 중복 없이 구조화하는 보조원이다."
def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str)->Dict:
    ctx=clean_text(raw_text,14000)
    user={"role":"user","content":(
        "다음 채용 공고 원문을 구조화해줘.\n\n"
        f"[힌트] 회사명 후보: {meta_hint.get('company_name','')}\n"
        f"[힌트] 직무명 후보: {meta_hint.get('job_title','')}\n"
        "--- 원문 시작 ---\n"+ctx+"\n--- 원문 끝 ---\n\n"
        "JSON으로만 답:\n"
        "{\"company_name\":str,\"company_intro\":str,\"job_title\":str,"
        "\"responsibilities\":[str],\"qualifications\":[str],\"preferences\":[str]}"
    )}
    try:
        r = client.chat.completions.create(model=model, temperature=0.2, max_tokens=900,
                                           response_format={"type":"json_object"},
                                           messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user])
        data=json.loads(r.choices[0].message.content)
    except Exception as e:
        data={"company_name":meta_hint.get("company_name",""),
              "company_intro":meta_hint.get("company_intro","원문 정제 실패"),
              "job_title":meta_hint.get("job_title",""),
              "responsibilities":[], "qualifications":[], "preferences":[], "error":str(e)}
    for k in ["responsibilities","qualifications","preferences"]:
        arr=data.get(k,[]); 
        if not isinstance(arr,list): arr=[]
        clean_list=[]; seen=set()
        for it in arr:
            t=re.sub(r"\s+"," ", str(it)).strip(" -•·▶▪️").strip()
            if t and t not in seen: seen.add(t); clean_list.append(t[:180])
        data[k]=clean_list[:14]
    if len(data.get("preferences",[]))<1:
        rb=rule_based_sections(ctx); 
        if rb.get("preferences"): data["preferences"] = list(dict.fromkeys(data["preferences"]+rb["preferences"]))[:14]
    for k in ["company_name","company_intro","job_title"]:
        if isinstance(data.get(k),str): data[k]=re.sub(r"\s+"," ", data[k]).strip()
    return data

def extract_company_meta_from_html(html: Optional[str])->Dict[str,str]:
    meta={"company_name":"","company_intro":"","job_title":""}
    if not html: return meta
    try:
        soup=soup_html(html); cand=[]
        og=soup.find("meta",{"property":"og:site_name"})
        if og and og.get("content"): cand.append(og["content"])
        app=soup.find("meta",{"name":"application-name"})
        if app and app.get("content"): cand.append(app["content"])
        if soup.title and soup.title.string: cand.append(soup.title.string)
        cand=[re.split(r"[\-\|\·\—]",c)[0].strip() for c in cand if c]
        cand=[c for c in cand if 2<=len(c)<=40]; meta["company_name"]= (cand[0] if cand else "")
        md=soup.find("meta",{"name":"description"}) or soup.find("meta",{"property":"og:description"})
        if md and md.get("content"): meta["company_intro"]=re.sub(r"\s+"," ", md["content"]).strip()[:500]
        jt=""; ogt=soup.find("meta",{"property":"og:title"})
        if ogt and ogt.get("content"): jt=ogt["content"]
        if not jt:
            for hx in ["h1","h2"]:
                h=soup.find(hx)
                if h and h.get_text(): jt=h.get_text(strip=True); break
        meta["job_title"]=re.sub(r"\s+"," ", jt).strip()[:120]
    except Exception: pass
    return meta

# ============ 회사 홈페이지 정밀 추출(핵심 개선 구간) =========
VISION_KEYS = ["비전","미션","핵심가치","가치","철학","원칙","소개","문화",
               "purpose","mission","vision","values","principle","philosophy",
               "culture","esg","sustainability","about","company","who we are",
               "what we do","our story","brand","story","core value","core values"]
TALENT_KEYS = ["인재","인재상","채용철학","people","talent","who we hire",
               "what we look for","team","careers","recruit","채용","our people"]

NEG_PATH = ["news","press","ir","blog","notice","media","pr","events","careers","recruit","job"]
PORTAL_DOMAINS={"wanted.co.kr","saramin.co.kr","jobkorea.co.kr","linkedin.com","kr.indeed.com","rocketpunch.com"}
DEFAULT_TIMEOUT=10; MAX_DEPTH=1; MIN_HTML_LEN=1200
USER_AGENT=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36 JobHelperBot/Harvester/1.1")

def _domain(host:str)->str:
    parts=host.split(".");  return ".".join(parts[-2:]) if len(parts)>=2 else host

def _http_get(url:str, timeout:int=DEFAULT_TIMEOUT)->str:
    try:
        r=requests.get(url, timeout=timeout, headers={"User-Agent":USER_AGENT,"Accept-Language":"ko, en;q=0.9"}, allow_redirects=True)
        if r.status_code==200 and r.text: return r.text
    except Exception: pass
    return ""

def strip_noise(soup: BeautifulSoup):
    for tag in soup(["script","style","noscript","template","svg","iframe"]): tag.decompose()
    for sel in ["header","nav","footer","aside","form"]: 
        for t in soup.select(sel): t.decompose()
    # class/id 기반 잡음
    noisy = ["breadcrumb","breadcrumbs","sitemap","menu","gnb","lnb","footer","cookie","consent","popup","modal","subscribe"]
    for n in noisy:
        for t in soup.select(f".{n}, #{n}"): 
            try: t.decompose()
            except Exception: pass

def is_core_sentence(s:str)->bool:
    s=s.strip()
    if len(s)<12: return False
    bad = {"about","vision","core value","core values","values","people","team","careers","recruit"}
    low=s.lower()
    if low in bad: return False
    return True

def lang_filter(s: str)->bool:
    s=s.strip()
    # 한국어 또는 핵심영문 키워드 포함
    return bool(re.search(r"[가-힣]", s)) or bool(re.search(r"\b(vision|mission|values?|talent|people|culture)\b", s, re.I))

def jaccard(a: str, b: str)->float:
    A=set(a.lower().split()); B=set(b.lower().split()); 
    if not A or not B: return 0.0
    return len(A&B)/max(1,len(A|B))

def dedup_lines(lines: List[str], thr: float=0.95)->List[str]:
    out=[]
    for s in lines:
        if not any(jaccard(s, t)>=thr for t in out):
            out.append(s)
    return out

def score_url_title(url: str, title: str)->int:
    score=0; L=url.lower()
    if any(k in L for k in ["about","company","mission","vision","values","culture","talent","people","our-story","who-we-are","what-we-do","brand"]): score+=3
    if any(n in L for n in NEG_PATH): score-=3
    t=title.lower()
    if any(k in t for k in ["mission","vision","values","인재상","people","talent","핵심가치","비전","문화"]): score+=4
    # 도메인 보정
    host=urllib.parse.urlsplit(url).netloc.lower()
    if "kakaohealthcare.com" in host and any(k in L for k in ["/about","/company","/our","/story","/mission","/vision","/values"]): score+=2
    return score

def extract_sections(html: str)->Tuple[List[str], List[str]]:
    """헤딩-섹션 추출: (vision_like, talent_like)"""
    soup=soup_html(html); strip_noise(soup)
    # title
    ttl = soup.title.get_text(strip=True) if soup.title else ""
    # 헤딩 목록
    heads = soup.find_all(re.compile(r"h[1-3]"))
    chunks=[]
    for i,h in enumerate(heads):
        txt=h.get_text(" ", strip=True)
        sect=[txt]
        # 다음 헤딩 전까지 형제들 수집
        for sib in h.find_next_siblings():
            if re.match(r"h[1-3]", sib.name or "", re.I): break
            t = sib.get_text(" ", strip=True)
            if t: sect.append(t)
        block= " ".join(sect)
        if len(block)>30: chunks.append(block)
    # 보조: p/li 스캔
    if not chunks:
        for tag in soup.find_all(["p","li"]):
            t=tag.get_text(" ", strip=True)
            if t: chunks.append(t)

    def classify(lines: List[str], keys: List[str])->List[str]:
        out=[]
        for ln in lines:
            if any(k.lower() in ln.lower() for k in [*keys, "core value","core values"]):
                out.append(ln)
        return out

    # 후보 집계
    vision_raw = classify(chunks, VISION_KEYS)
    talent_raw = classify(chunks, TALENT_KEYS)

    # 문장화 + 필터
    def explode_and_filter(blocks: List[str])->List[str]:
        sents=[]
        for b in blocks:
            # 구분자 기준 분할
            parts = re.split(r"[•\u2022\-\–\·\•]|(?<=[.!?])\s+", b)
            for p in parts:
                p = re.sub(r"\s+"," ", p).strip(" -•·▶▪️")
                if is_core_sentence(p) and lang_filter(p):
                    sents.append(p[:220])
        return dedup_lines(sents, 0.93)[:30]

    return explode_and_filter(vision_raw), explode_and_filter(talent_raw)

def collect_nav_links(html: str, base_url: str)->List[str]:
    soup=soup_html(html); cands=[]
    for a in soup.find_all("a", href=True):
        txt=(a.get_text(" ", strip=True) or "").lower(); href=a["href"].strip()
        url=urllib.parse.urljoin(base_url, href); path=urllib.parse.urlsplit(url).path.lower()
        if any(b in path for b in NEG_PATH):  # 뉴스/IR/채용 등은 제외
            continue
        if any(k in txt for k in [k.lower() for k in VISION_KEYS+TALENT_KEYS]) or \
           any(k in path for k in ["about","company","mission","vision","values","culture","people","talent","who-we-are","our-story","what-we-do","brand","/ko","/kr"]):
            cands.append(url)
    # dedup
    seen=set(); out=[]
    for u in cands:
        if u not in seen:
            seen.add(u); out.append(u)
    return out[:20]

def read_sitemap(home: str)->List[str]:
    base=(normalize_url(home) or "").rstrip("/")
    xml=_http_get(base+"/sitemap.xml", timeout=6)
    if not xml: return []
    urls=parse_sitemap_locations(xml)
    keep=[]
    for u in urls:
        L=u.lower()
        if any(k in L for k in ["about","company","mission","vision","values","culture","talent","people","our-story","who-we-are","brand","/ko","/kr"]) and not any(n in L for n in NEG_PATH):
            keep.append(u)
    return keep[:25]

def extract_title(html: str)->str:
    s=soup_html(html)
    return s.title.get_text(strip=True) if s.title else ""

def company_home_harvest(home_url: str, prefer_dynamic: bool, timeout: int, max_pages: int)->Dict[str,List[str]]:
    out={"vision":[], "talent":[]}
    base=normalize_url(home_url or "")
    if not base: return out
    roots=[base.rstrip("/")]
    for loc in ("/kr","/ko","/ko-kr"): roots.append(base.rstrip("/")+loc)
    for p in ("/about","/company","/about-us","/mission","/vision","/values","/culture","/people",
              "/careers","/talent","/esg","/sustainability","/who-we-are","/what-we-do","/our-story","/brand"):
        roots.append(base.rstrip("/")+p)
    roots += read_sitemap(base)

    visited=set(); queue=[(u,0) for u in roots]; pages=0

    while queue and pages<max_pages:
        url,depth = queue.pop(0)
        if url in visited: continue
        visited.add(url); pages+=1

        # 정/동적 폴백
        html = ""
        if prefer_dynamic:
            html = get_html_dynamic(url, timeout) or ""
            if not html or len(html)<200: html = _http_get(url, timeout)
        else:
            html = _http_get(url, timeout)
            if not html or len(html)<MIN_HTML_LEN: html = get_html_dynamic(url, timeout)
        if not html: continue

        # 페이지 스코어: 타이틀/URL 기반
        title = extract_title(html)
        page_score = score_url_title(url, title)
        if page_score < 0:   # 뉴스/IR 등은 스킵
            continue

        # 섹션 추출
        v_lines, t_lines = extract_sections(html)
        if v_lines: out["vision"] += v_lines
        if t_lines: out["talent"] += t_lines

        # depth=1까지 확장 (스코어 양수인 링크만)
        if depth<MAX_DEPTH:
            for nxt in collect_nav_links(html, url):
                if nxt not in visited:
                    queue.append((nxt, depth+1))

    # 후처리: 중복/유사도 제거 + 길이 제한
    for k in out:
        lines=[re.sub(r"\s+"," ", s).strip() for s in out[k] if is_core_sentence(s)]
        out[k]=dedup_lines(lines, 0.93)[:30]
    return out

def infer_home_from_job(job_html: str, job_url: str, company_name: str)->List[str]:
    soup=soup_html(job_html); anchors=soup.find_all("a", href=True)
    cands=[]; cmp_l=(company_name or "").lower()
    for a in anchors:
        href=a["href"].strip(); txt=(a.get_text(" ", strip=True) or "").lower()
        url=urllib.parse.urljoin(job_url, href); host=urllib.parse.urlsplit(url).netloc.lower()
        if not host or _domain(host) in PORTAL_DOMAINS: continue
        score=0
        if cmp_l and cmp_l in txt: score+=2
        if any(k in txt for k in [k.lower() for k in VISION_KEYS+TALENT_KEYS]): score+=2
        if url.endswith("/") or url.count("/")<=3: score+=1
        # 경로 페널티
        if any(bad in url.lower() for bad in NEG_PATH): score-=2
        cands.append((url, score))
    scored=Counter()
    for it in cands: scored[it[0]] += it[1]
    candidates=[u for u,_ in scored.most_common()]
    roots=[]; seen=set()
    for u in candidates:
        pu=urllib.parse.urlsplit(u); root=f"{pu.scheme}://{pu.netloc}/"
        if root not in seen: seen.add(root); roots.append(root)
    return roots[:5] or candidates[:5]

# ===================== 뉴스(RSS) ==========================
@st.cache_data(show_spinner=False, ttl=1200)
def google_news_rss_multi(company: str, max_items:int=5)->List[Dict]:
    def rss(q):
        url=f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=ko&gl=KR&ceid=KR:ko"
        try:
            r=requests.get(url, timeout=6)
            if r.status_code!=200: return []
            soup=soup_html(r.text)  # 간단 추출
            items=soup.find_all("item"); out=[]
            for it in items[:max_items]:
                title=it.find("title").get_text(strip=True) if it.find("title") else ""
                link =it.find("link").get_text(strip=True) if it.find("link") else ""
                pub  =it.find("pubdate").get_text(strip=True) if it.find("pubdate") else ""
                out.append({"title":title,"link":link,"pubDate":pub})
            return out
        except Exception: return []
    for q in [company, f"\"{company}\" 회사", f"\"{company}\" 채용"]:
        res=rss(q)
        if res: return res
    return []

# ======================= 임베딩/RAG =======================
def chunk(text: str, size:int=600, overlap:int=120)->List[str]:
    t=re.sub(r"\s+"," ", text or "").strip(); out=[]; start=0; L=len(t)
    while start<L: end=min(L,start+size); out.append(t[start:end]); start = max(0, end-overlap); 
    return out

def embed_texts(texts: List[str], model_name: str)->np.ndarray:
    if not texts: return np.zeros((0,1536), dtype=np.float32)
    resp=client.embeddings.create(model=model_name, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def cosine_topk(M: np.ndarray, q: np.ndarray, k:int=4):
    if M.size==0: return np.array([]), np.array([], dtype=int)
    qn=q/(np.linalg.norm(q, axis=1, keepdims=True)+1e-12); mn=M/(np.linalg.norm(M, axis=1, keepdims=True)+1e-12)
    sims=mn@qn.T; sims=sims.reshape(-1); idx=np.argsort(-sims)[:k]; return sims[idx], idx

def retrieve_resume_chunks(query: str, chunks: List[str], embeds: np.ndarray, k:int=4):
    if not chunks or embeds is None or embeds.size==0: return []
    qv=embed_texts([query], EMBED_MODEL); scores, idxs=cosine_topk(embeds, qv, k=k)
    return [(float(s), chunks[int(i)]) for s,i in zip(scores, idxs)]

PROMPT_SYSTEM_Q="너는 채용담당자다. 회사/직무/요건과 이력서를 고려해 좋은 한국어 질문을 만든다."
PROMPT_SYSTEM_DRAFT="너는 면접 코치다. STAR로 8~12문장 한국어 답변 초안을 작성한다."
CRITERIA=["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def llm_generate_one_question_with_resume(clean: Dict, level: str, model: str, resume_chunks: List[str], resume_embeds: np.ndarray)->str:
    hits=retrieve_resume_chunks("핵심 프로젝트와 기술 스택 요약", resume_chunks, resume_embeds, k=4)
    resume="\n".join([f"- {t[:350]}" for _,t in hits])[:1200]
    ctx=json.dumps(clean, ensure_ascii=False)
    user={"role":"user","content":f"[회사/직무/요건]\n{ctx}\n\n[이력서 발췌]\n{resume}\n\n[요청]\n- 난이도/연차: {level}\n- 한국어 면접 질문 1개만 한 줄로 출력"}
    try:
        r=client.chat.completions.create(model=model, temperature=0.8, max_tokens=120, messages=[{"role":"system","content":PROMPT_SYSTEM_Q}, user])
        q=r.choices[0].message.content.strip()
        return re.sub(r"^\s*\d+[\).\s-]*","", q).split("\n")[0].strip()
    except Exception: return ""

def llm_draft_answer(clean: Dict, question: str, model: str, resume_chunks: List[str], resume_embeds: np.ndarray)->str:
    hits=retrieve_resume_chunks(question, resume_chunks, resume_embeds, k=4)
    resume="\n".join([f"- {t[:400]}" for _,t in hits])[:1600]
    ctx=json.dumps(clean, ensure_ascii=False)
    user={"role":"user","content":f"[회사/직무/채용요건]\n{ctx}\n\n[이력서 발췌]\n{resume}\n\n[면접 질문]\n{question}\n\nSTAR 기반 한국어 답변 **초안**을 작성해줘."}
    try:
        r=client.chat.completions.create(model=model, temperature=0.5, max_tokens=700, messages=[{"role":"system","content":PROMPT_SYSTEM_DRAFT}, user])
        return r.choices[0].message.content.strip()
    except Exception: return ""

def llm_score_and_coach_strict(clean: Dict, question: str, answer: str, model: str, resume_chunks: List[str], resume_embeds: np.ndarray)->Dict:
    hits=retrieve_resume_chunks(question+"\n"+answer[:800], resume_chunks, resume_embeds, k=4)
    resume="\n".join([f"- {t[:400]}" for _,t in hits])[:1600]
    ctx=json.dumps(clean, ensure_ascii=False)
    user={"role":"user","content":(
        f"[회사/직무/채용요건]\n{ctx}\n\n[이력서 발췌]\n{resume}\n\n[면접 질문]\n{question}\n\n[지원자 답변]\n{answer}\n\n"
        "JSON만 출력:\n"
        "{\"overall_score\":0-100,"
        "\"criteria\":[{\"name\":\"문제정의\",\"score\":0-20,\"comment\":\"...\"},"
        "{\"name\":\"데이터/지표\",\"score\":0-20,\"comment\":\"...\"},"
        "{\"name\":\"실행력/주도성\",\"score\":0-20,\"comment\":\"...\"},"
        "{\"name\":\"협업/커뮤니케이션\",\"score\":0-20,\"comment\":\"...\"},"
        "{\"name\":\"고객가치\",\"score\":0-20,\"comment\":\"...\"}],"
        "\"strengths\":[\"...\"],\"risks\":[\"...\"],\"improvements\":[\"...\",\"...\"],"
        "\"revised_answer\":\"STAR 재작성\"}"
    )}
    try:
        r=client.chat.completions.create(model=model, temperature=0.2, max_tokens=900, response_format={"type":"json_object"},
                                         messages=[{"role":"system","content":"면접 코치(엄격 모드)"}, user])
        data=json.loads(r.choices[0].message.content)
        # 정규화
        fixed=[]
        for name in CRITERIA:
            cur=next((it for it in data.get("criteria",[]) if str(it.get("name","")).strip()==name), {"name":name,"score":0,"comment":""})
            cur["score"]=int(max(0,min(20,int(cur.get("score",0)))))
            cur["comment"]=str(cur.get("comment","")).strip(); fixed.append(cur)
        data["criteria"]=fixed; data["overall_score"]=sum(x["score"] for x in fixed)
        for k in ["strengths","risks","improvements"]:
            arr=data.get(k,[]); 
            if not isinstance(arr,list): arr=[]
            data[k]=[str(x).strip() for x in arr if str(x).strip()][:5]
        data["revised_answer"]=str(data.get("revised_answer","")).strip()
        return data
    except Exception as e:
        return {"overall_score":0,"criteria":[{"name":n,"score":0,"comment":""} for n in CRITERIA],
                "strengths": [],"risks": [],"improvements": [],"revised_answer":"", "error":str(e)}

# ======================= 상태/UI ==========================
def _init_state():
    for k,v in dict(clean=None,last_html=None,company_vision=[],company_talent=[],company_news=[],auto_home_used="",
                    resume_chunks=[],resume_embeds=None,resume_raw="",current_question="",answer_text="",last_result=None).items():
        if k not in st.session_state: st.session_state[k]=v
_init_state()

st.header("1) 채용 공고 URL (Selenium 전용)")
job_url = st.text_input("채용 공고 상세 URL", placeholder="원티드/사람인/잡코리아/기업 채용 페이지 URL")
st.text_input("회사 공식 홈페이지 URL (선택)", key="company_home", placeholder="회사 공식 홈페이지 URL(미입력 시 자동 추론)")

if st.button("원문 수집 → 정제", type="primary"):
    if not job_url.strip(): st.warning("채용 공고 URL을 입력하세요.")
    else:
        with st.spinner("채용 공고 로딩/정제 중..."):
            raw, meta, html = fetch_job_text_dynamic(job_url.strip(), timeout=SELENIUM_TIMEOUT)
            hint=extract_company_meta_from_html(html); st.session_state.last_html=html
            if not raw: st.error("채용 공고 수집 실패")
            else:
                st.caption(f"소스: {meta.get('source')} · 텍스트 길이: {meta.get('len')}")
                st.session_state.clean = llm_structurize(raw, hint, CHAT_MODEL)
        if st.session_state.clean:
            with st.spinner("회사 홈페이지(비전/인재상) 수집 중..."):
                vision, talent, used = [], [], ""
                home_input = st.session_state.company_home.strip()
                auto_candidates = infer_home_from_job(html or "", meta.get("url_final","") or job_url,
                                                      st.session_state.clean.get("company_name",""))
                for candidate in ([home_input] if home_input else []) + auto_candidates:
                    if not candidate: continue
                    pages = company_home_harvest(candidate, PREFER_DYNAMIC, SELENIUM_TIMEOUT, MAX_PAGES)
                    if pages.get("vision") or pages.get("talent"):
                        vision = pages.get("vision", []); talent = pages.get("talent", []); used = candidate if not home_input else home_input; break
                st.session_state.company_vision = vision; st.session_state.company_talent = talent; st.session_state.auto_home_used = used
            with st.spinner("뉴스 수집 중..."):
                cname = st.session_state.clean.get("company_name","") or ""
                st.session_state.company_news = google_news_rss_multi(cname, 5)
            st.success("정제 완료!")

# 회사 요약
st.header("2) 회사 요약")
clean = st.session_state.clean
if clean:
    st.markdown(f"**회사명:** {clean.get('company_name','-')}")
    st.markdown(f"**간단한 회사 소개(요약):** {clean.get('company_intro','-')}")
    st.markdown(f"**모집 분야(직무명):** {clean.get('job_title','-')}")
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown("**주요 업무**"); [st.markdown(f"- {b}") for b in clean.get("responsibilities",[])]
    with c2: st.markdown("**자격 요건**"); [st.markdown(f"- {b}") for b in clean.get("qualifications",[])]
    with c3:
        st.markdown("**우대 사항**")
        prefs=clean.get("preferences",[])
        if prefs: [st.markdown(f"- {b}") for b in prefs]
        else: st.caption("명시된 우대 사항이 없습니다.")
else:
    st.info("먼저 URL을 정제해 주세요.")

# 비전/인재상/뉴스
st.divider(); st.subheader("회사 비전/인재상 & 최신 이슈")
vcol,tcol=st.columns(2)
with vcol:
    st.markdown("**비전/핵심가치**")
    if st.session_state.company_vision: [st.markdown(f"- {v}") for v in st.session_state.company_vision[:10]]
    else: st.caption("비전/핵심가치를 찾지 못했습니다.")
with tcol:
    st.markdown("**인재상**")
    if st.session_state.company_talent: [st.markdown(f"- {t}") for t in st.session_state.company_talent[:10]]
    else: st.caption("인재상 정보를 찾지 못했습니다.")
if st.session_state.auto_home_used: st.caption(f"회사 홈페이지 사용: {st.session_state.auto_home_used}")

st.markdown("**최신 뉴스(상위 3~5)**")
if st.session_state.company_news: 
    for n in st.session_state.company_news[:5]: st.markdown(f"- [{n.get('title','(제목 없음)')}]({n.get('link','#')})")
else: st.caption("뉴스 결과가 없습니다.")

# 이력서 인덱싱 (PDF/DOCX/TXT)
st.divider(); st.header("3) 내 이력서/프로젝트 업로드")
uploads=st.file_uploader("여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True)
_RESUME_CHUNK=500; _RESUME_OVLP=100
try: import pypdf
except Exception: pypdf=None
try: import docx2txt
except Exception: docx2txt=None

def read_pdf(data: bytes)->str:
    if pypdf is None: return ""
    try:
        r=pypdf.PdfReader(io.BytesIO(data))
        return "\n\n".join([(r.pages[i].extract_text() or "") for i in range(len(r.pages))])
    except Exception: return ""
def read_docx(data: bytes)->str:
    if docx2txt is None: return ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
            tmp.write(data); tmp.flush(); return docx2txt.process(tmp.name) or ""
    except Exception: return ""
def read_uploaded(up)->str:
    name=up.name.lower(); data=up.read()
    if name.endswith((".txt",".md")):
        for enc in ("utf-8","cp949","euc-kr"):
            try: return data.decode(enc)
            except Exception: continue
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):  return read_pdf(data)
    if name.endswith(".docx"): return read_docx(data)
    return ""

if st.button("이력서 인덱싱", type="secondary"):
    if not uploads: st.warning("파일을 업로드하세요.")
    else:
        texts=[]
        for up in uploads:
            t=read_uploaded(up)
            if t: texts.append(t)
        resume_text="\n\n".join(texts)
        if not resume_text.strip(): st.error("텍스트 추출 실패")
        else:
            chunks=chunk(resume_text, size=_RESUME_CHUNK, overlap=_RESUME_OVLP)
            with st.spinner("이력서 벡터화 중..."):
                embeds=embed_texts(chunks, EMBED_MODEL)
            st.session_state.resume_raw=resume_text; st.session_state.resume_chunks=chunks; st.session_state.resume_embeds=embeds
            st.success(f"인덱싱 완료 (청크 {len(chunks)}개)")

# 질문/초안/채점
st.divider(); st.header("4) 질문 생성 & 답변 초안 (RAG)")
level=st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)
c1,c2=st.columns(2)
with c1:
    if st.button("새 질문", type="primary"):
        if not st.session_state.clean: st.warning("먼저 회사 URL을 정제하세요.")
        else:
            q=llm_generate_one_question_with_resume(st.session_state.clean, level, CHAT_MODEL, st.session_state.resume_chunks, st.session_state.resume_embeds)
            if q: st.session_state.current_question=q; st.session_state.answer_text=""; st.session_state.last_result=None; st.success("질문 생성 완료!")
            else: st.error("질문 생성 실패")
with c2:
    if st.button("RAG로 답변 초안", type="secondary"):
        if not st.session_state.current_question: st.warning("먼저 질문을 생성하세요.")
        else:
            draft=llm_draft_answer(st.session_state.clean, st.session_state.current_question, CHAT_MODEL, st.session_state.resume_chunks, st.session_state.resume_embeds)
            if draft: st.session_state.answer_text=draft; st.success("초안 생성 완료!")
            else: st.error("초안 생성 실패")

st.text_area("질문", value=st.session_state.current_question, height=90)
st.text_area("나의 답변 (초안을 편집해 완성)", key="answer_text", height=200)

st.header("5) 채점 & 코칭")
if st.button("채점 실행", type="primary"):
    if not st.session_state.current_question: st.warning("먼저 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip(): st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("채점/코칭 중..."):
            res=llm_score_and_coach_strict(st.session_state.clean, st.session_state.current_question, st.session_state.answer_text,
                                           CHAT_MODEL, st.session_state.resume_chunks, st.session_state.resume_embeds)
        st.session_state.last_result=res; st.success("완료!")

st.divider(); st.subheader("피드백 결과")
last=st.session_state.last_result
if last:
    st.metric("총점(/100)", last.get("overall_score",0))
    st.markdown("**기준별 점수 & 코멘트**")
    for it in last.get("criteria", []):
        st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
    if last.get("strengths"): st.markdown("**강점**"); [st.markdown(f"- {s}") for s in last["strengths"]]
    if last.get("risks"): st.markdown("**감점 요인/리스크**"); [st.markdown(f"- {r}") for r in last["risks"]]
    if last.get("improvements"): st.markdown("**개선 포인트**"); [st.markdown(f"- {x}") for x in last["improvements"]]
    if last.get("revised_answer"): st.markdown("**수정본(STAR)**"); st.write(last["revised_answer"])
else:
    st.caption("아직 채점 결과가 없습니다.")
