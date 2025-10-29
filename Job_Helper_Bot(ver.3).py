# -*- coding: utf-8 -*-
################################################################################
# Job Helper Bot - FULL (Static + Dynamic Crawl, lxml-free, Company Home Booster, RAG Coach)
# 실행: streamlit run Job_Helper_Bot_full_fixed.py
################################################################################
import os, re, io, json, time, shutil, tempfile, urllib.parse, traceback
from typing import List, Dict, Tuple, Optional
from collections import Counter

import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup   # 파서는 soup_html()로 통일
import html2text
from xml.etree import ElementTree as ET

# ======================= OpenAI ===========================
try:
    from openai import OpenAI
except ImportError:
    st.error("pip install openai 로 설치 후 다시 실행하세요.")
    st.stop()

API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
if not API_KEY:
    API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password")
if not API_KEY:
    st.stop()
client = OpenAI(api_key=API_KEY)

# ======================= UI Options =======================
with st.sidebar:
    st.subheader("모델 / 크롤링 옵션")
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small","text-embedding-3-large"], index=0)
    SELENIUM_TIMEOUT = st.slider("Selenium 대기(초)", 8, 30, 16)
    MAX_PAGES = st.slider("회사 홈페이지 최대 페이지 수", 6, 30, 14)
    PREFER_DYNAMIC = st.toggle("회사 홈페이지: 동적 우선", value=True)
    FAST_MODE = st.toggle("FAST 모드(스크롤/대기 최소화)", value=True)

# ======================= html2text ========================
def _h2t():
    conv = html2text.HTML2Text()
    conv.ignore_links = True
    conv.ignore_images = True
    conv.body_width = 0
    return conv
H2T = _h2t()

def html_to_text(html_str: str) -> str:
    txt = H2T.handle(html_str or "")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return re.sub(r"\s+", " ", txt).strip()

def clean_text(s: str, max_len: int = 16000) -> str:
    if not s: return ""
    s = re.sub(r"\r", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len] if len(s) > max_len else s

def normalize_url(u: str) -> Optional[str]:
    if not u: return None
    u = u.strip()
    if not re.match(r"^https?://", u): u = "https://" + u
    parts = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

# =================== Parser-safe helpers ==================
def soup_html(html: str) -> BeautifulSoup:
    """lxml 미설치 환경에서도 안전한 HTML 파싱 (내장 html.parser 사용)"""
    return BeautifulSoup(html or "", "html.parser")

def parse_sitemap_locations(xml_text: str) -> List[str]:
    """sitemap.xml에서 <loc> 텍스트 리스트 추출 (ElementTree 사용)"""
    out = []
    if not xml_text:
        return out
    try:
        root = ET.fromstring(xml_text)
        ns = {}
        if root.tag.startswith("{"):
            uri = root.tag.split("}")[0].strip("{")
            ns = {"ns": uri}
            loc_tags = root.findall(".//ns:loc", ns)
        else:
            loc_tags = root.findall(".//loc")
        for loc in loc_tags:
            if loc is not None and loc.text:
                out.append(loc.text.strip())
    except Exception:
        # 파싱 실패 시 정규식 폴백
        out = re.findall(r"<loc>\s*([^<>\s]+)\s*</loc>", xml_text, flags=re.I)
    return out

# ======================= Selenium =========================
HAS_SELENIUM = True
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException
except Exception:
    HAS_SELENIUM = False
    st.error("Selenium이 필요합니다. pip install selenium && 로컬 크롬 설치 확인")
    st.stop()

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

def build_driver(headless: bool=True):
    opts = ChromeOptions()
    if headless: opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1440,2400")
    opts.add_argument("--lang=ko-KR")
    binpath = _pick_chrome_binary()
    if binpath: opts.binary_location = binpath
    return webdriver.Chrome(options=opts)   # Selenium Manager auto-download

def wait_dom(driver, timeout: int):
    try:
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, "//*")))
    except TimeoutException:
        pass

def smart_scroll(driver, loops: int = 8):
    loops = 3 if FAST_MODE else loops
    for _ in range(loops):
        try:
            driver.execute_script("window.scrollBy(0, document.body.scrollHeight*0.7);")
            time.sleep(0.12 if FAST_MODE else 0.3)
        except Exception:
            break

def click_more_generic(driver):
    texts = ["더보기","전체보기","상세보기","자세히 보기","더 보기","Read more","More","보기"]
    for t in texts:
        try:
            xp1 = f"//*[normalize-space(text())='{t}']"
            xp2 = f"//*[contains(normalize-space(text()), '{t}')]"
            for xp in (xp1, xp2):
                for el in driver.find_elements(By.XPATH, xp)[:8]:
                    try:
                        driver.execute_script("arguments[0].click();", el)
                        time.sleep(0.08 if FAST_MODE else 0.18)
                    except Exception:
                        continue
        except Exception:
            continue

def expand_portal_specific(driver, host: str):
    if "wanted.co.kr" in host:
        sels = [
            "[data-qa='btn-read-more']","[data-qa='job-header__more']",
            "button[aria-expanded='false']","[role='button'][class*='More']",
        ]
    elif "saramin" in host:
        sels = [".btn_more",".btnMore",".btn-detail",".btn_toggle","button[class*='more'], a[class*='more']"]
    elif "jobkorea" in host:
        sels = [".btnFold",".btnToggleRead",".btn_more","button[class*='More'], a[class*='More']"]
    else:
        sels = []
    for sel in sels:
        try:
            for el in driver.find_elements(By.CSS_SELECTOR, sel)[:10]:
                try:
                    driver.execute_script("arguments[0].click();", el)
                    time.sleep(0.08 if FAST_MODE else 0.18)
                except Exception:
                    continue
        except Exception:
            continue

def get_html_dynamic(url: str, timeout: int) -> str:
    url = normalize_url(url) or url
    driver = build_driver(headless=True)
    try:
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        wait_dom(driver, timeout)
        host = urllib.parse.urlsplit(url).netloc.lower()
        expand_portal_specific(driver, host)
        click_more_generic(driver)
        smart_scroll(driver, loops=10)
        return driver.page_source or ""
    finally:
        try: driver.quit()
        except Exception: pass

# =================== 채용공고 → 텍스트 ====================
def extract_wanted_next(html: str) -> str:
    try:
        soup = soup_html(html)
        tag = soup.select_one("script#__NEXT_DATA__")
        if not tag: return ""
        data = json.loads((tag.string or tag.text or "").strip())
    except Exception:
        return ""
    key_whitelist = ["job","position","title","desc","description","responsibilit","duty","role","skill","require","qualification","prefer","plus","nice"]
    def _walk(d, out):
        if isinstance(d, dict):
            for k, v in d.items():
                if any(t in str(k).lower() for t in key_whitelist):
                    if isinstance(v, str): out.append(v)
                    elif isinstance(v, list):
                        for it in v:
                            if isinstance(it, str): out.append(it)
                            elif isinstance(it, dict):
                                for subv in it.values():
                                    if isinstance(subv, str): out.append(subv)
                    elif isinstance(v, dict):
                        for subv in v.values():
                            if isinstance(subv, str): out.append(subv)
                _walk(v, out)
        elif isinstance(d, list):
            for it in d: _walk(it, out)
    bucket=[]; _walk(data, bucket)
    seen=set(); lines=[]
    for t in bucket:
        s=re.sub(r"\s+"," ", (t or "")).strip()
        if len(s)>2 and s not in seen:
            seen.add(s); lines.append(s)
    return "\n".join(lines[:900])

def fetch_job_text_dynamic(url: str, timeout: int) -> Tuple[str, Dict, Optional[str]]:
    html = get_html_dynamic(url, timeout=timeout)
    if not html or len(html) < 200:
        return "", {"source":"selenium_failed","len":0,"url_final":url}, None
    if "wanted.co.kr" in (urllib.parse.urlsplit(url).netloc.lower()):
        try:
            nxt = extract_wanted_next(html)
            if nxt:
                html += "\n<div id='__WANTED_NEXT_EXTRACT__'>" + "".join(f"<p>{l}</p>" for l in nxt.split("\n")) + "</div>"
        except Exception:
            pass
    txt = html_to_text(html)
    return txt, {"source":"selenium","len":len(txt),"url_final":url}, html

# ============== 규칙 파서 (업무/자격/우대) ==================
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
            low = l.lower()
            if hdr_pref.search(l): bucket = "preferences"
            elif any(k in low for k in ["java","python","spring","kotlin","react","next","kafka","sql","ml","cloud","aws","gcp"]):
                bucket = "responsibilities"
            else:
                continue
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

# =================== LLM 구조화 ===========================
PROMPT_SYSTEM_STRUCT = "너는 채용 공고를 한국어로 간결하고 중복 없이 구조화하는 보조원이다."
def llm_structurize(raw_text: str, meta_hint: Dict[str, str], model: str) -> Dict:
    ctx = clean_text(raw_text, 14000)
    user_msg = {"role":"user","content":(
        "다음 채용 공고 원문을 구조화해줘.\n\n"
        f"[힌트] 회사명 후보: {meta_hint.get('company_name','')}\n"
        f"[힌트] 직무명 후보: {meta_hint.get('job_title','')}\n"
        "--- 원문 시작 ---\n"
        f"{ctx}\n--- 원문 끝 ---\n\n"
        "JSON으로만 답:\n"
        "{"
        "\"company_name\": str, "
        "\"company_intro\": str, "
        "\"job_title\": str, "
        "\"responsibilities\": [str], "
        "\"qualifications\": [str], "
        "\"preferences\": [str]"
        "}"
    )}
    try:
        r = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=900,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg]
        )
        data = json.loads(r.choices[0].message.content)
    except Exception as e:
        data = {"company_name": meta_hint.get("company_name",""),
                "company_intro": meta_hint.get("company_intro","원문 정제 실패"),
                "job_title": meta_hint.get("job_title",""),
                "responsibilities": [], "qualifications": [], "preferences": [], "error": str(e)}
    for k in ["responsibilities","qualifications","preferences"]:
        arr = data.get(k, [])
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
    for k in ["company_name","company_intro","job_title"]:
        if isinstance(data.get(k), str): data[k]=re.sub(r"\s+"," ", data[k]).strip()
    return data

def extract_company_meta_from_html(html: Optional[str]) -> Dict[str, str]:
    meta = {"company_name":"","company_intro":"","job_title":""}
    if not html: return meta
    try:
        soup = soup_html(html)
        cand=[]
        og = soup.find("meta", {"property":"og:site_name"})
        if og and og.get("content"): cand.append(og["content"])
        app = soup.find("meta", {"name":"application-name"})
        if app and app.get("content"): cand.append(app["content"])
        if soup.title and soup.title.string: cand.append(soup.title.string)
        cand = [re.split(r"[\-\|\·\—]", c)[0].strip() for c in cand if c]
        cand = [c for c in cand if 2 <= len(c) <= 40]
        meta["company_name"] = (cand[0] if cand else "")
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

# ============ 회사 홈페이지(정적+동적 폴백) =================
VISION_KEYS = [
    "비전","미션","핵심가치","가치","철학","원칙","소개","연혁","문화",
    "purpose","mission","vision","values","value","principle","philosophy",
    "culture","esg","sustainability","about","company","who we are","what we do",
    "our story","brand","story"
]
TALENT_KEYS = [
    "인재","인재상","채용철학","인사제도","복리후생","people","talent",
    "who we hire","what we look for","team","careers","recruit","채용"
]
PORTAL_DOMAINS = {"wanted.co.kr","saramin.co.kr","jobkorea.co.kr","linkedin.com","kr.indeed.com","rocketpunch.com"}
DEFAULT_TIMEOUT = 10
MAX_PAGES_DEFAULT = 14
MAX_DEPTH = 1
MIN_HTML_LEN = 1200
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36 JobHelperBot/Harvester/1.0")

def _domain(host: str) -> str:
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host

def _http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT, "Accept-Language":"ko, en;q=0.9"}, allow_redirects=True)
        if r.status_code == 200 and r.text:
            return r.text
    except Exception:
        pass
    return ""

def extract_texts(html: str) -> List[str]:
    soup = soup_html(html)
    texts=[]
    for tag in soup.find_all(["h1","h2","h3","h4","p","li"]):
        t = tag.get_text(" ", strip=True)
        if not t: continue
        t = re.sub(r"\s+"," ", t)
        if 6 <= len(t) <= 260:
            texts.append(t)
    return texts

def parse_jsonld_and_meta(html: str) -> List[str]:
    soup = soup_html(html)
    out=[]
    for sel in ["meta[name='description']","meta[property='og:description']","meta[property='og:title']"]:
        tag = soup.select_one(sel)
        if tag and tag.get("content"):
            out.append(tag["content"])
    for tag in soup.select("script[type='application/ld+json']"):
        try:
            data = json.loads(tag.text)
        except Exception:
            continue
        nodes = data if isinstance(data, list) else [data]
        for node in nodes:
            if not isinstance(node, dict): continue
            if str(node.get("@type","")).lower() in ("organization","website","webpage","aboutpage"):
                for k in ("description","mission","value","headline","about","name"):
                    v = node.get(k)
                    if isinstance(v,str) and 6 <= len(v) <= 260:
                        out.append(v)
    uniq=set(); res=[]
    for t in out:
        t = re.sub(r"\s+"," ", t).strip()
        if t and t not in uniq:
            uniq.add(t); res.append(t)
    return res

def collect_nav_links(html: str, base_url: str) -> List[str]:
    soup = soup_html(html)
    cands=[]
    for a in soup.find_all("a", href=True):
        txt  = (a.get_text(" ", strip=True) or "").lower()
        href = a["href"].strip()
        hlow = href.lower()
        if any(k in txt for k in [k.lower() for k in VISION_KEYS + TALENT_KEYS]) or \
           any(k in hlow for k in [
               "about","company","mission","vision","values","culture",
               "people","talent","careers","recruit","esg","sustainability",
               "/kr","/ko","/ko-kr","who-we-are","what-we-do","our-story","brand"
           ]):
            cands.append(urllib.parse.urljoin(base_url, href))
    uniq=set(); out=[]
    for u in cands:
        if u not in uniq:
            uniq.add(u); out.append(u)
    return out[:20]

def read_sitemap(home: str) -> List[str]:
    base = (normalize_url(home) or "").rstrip("/")
    site = base + "/sitemap.xml"
    xml = _http_get(site, timeout=6)
    if not xml:
        return []
    urls = parse_sitemap_locations(xml)
    keep=[]
    for u in urls:
        ul = u.lower()
        if any(k in ul for k in ["about","company","mission","vision","values","culture","talent","people","careers","esg","sustainability","/ko","/kr","ko-kr"]):
            keep.append(u)
    return keep[:25]

def company_home_harvest(home_url: str, prefer_dynamic: bool, timeout: int, max_pages: int) -> Dict[str, List[str]]:
    out = {"vision": [], "talent": []}
    base = normalize_url(home_url or "")
    if not base: return out
    roots = [base.rstrip("/")]
    for loc in ("/kr","/ko","/ko-kr"): roots.append(base.rstrip("/") + loc)
    for p in ("/about","/company","/about-us","/mission","/vision","/values",
              "/culture","/careers","/talent","/people","/esg","/sustainability",
              "/who-we-are","/what-we-do","/our-story","/brand"):
        roots.append(base.rstrip("/") + p)
    roots += read_sitemap(base)
    visited=set()
    queue = [(u,0) for u in roots]
    pages=0
    def classify_add(texts: List[str]):
        for t in texts:
            low = t.lower()
            if any(k in low for k in [k.lower() for k in TALENT_KEYS]): out["talent"].append(t)
            if any(k in low for k in [k.lower() for k in VISION_KEYS]): out["vision"].append(t)
    while queue and pages < max_pages:
        url, depth = queue.pop(0)
        if url in visited: continue
        visited.add(url); pages+=1
        html = ""
        if prefer_dynamic:
            html = get_html_dynamic(url, timeout=timeout) or ""
            if (not html) or len(html) < 200:
                html = _http_get(url, timeout=timeout)
        else:
            html = _http_get(url, timeout=timeout)
            if (not html) or len(html) < MIN_HTML_LEN:
                html = get_html_dynamic(url, timeout=timeout)
        if not html: continue
        classify_add(extract_texts(html))
        classify_add(parse_jsonld_and_meta(html))
        if depth < MAX_DEPTH and url in roots:
            for nxt in collect_nav_links(html, url):
                if nxt not in visited: queue.append((nxt, depth+1))
    for k in out:
        uniq=set(); res=[]
        for x in out[k]:
            x = re.sub(r"\s+"," ", x).strip()
            if x and x not in uniq:
                uniq.add(x); res.append(x[:200])
        out[k]=res[:30]
    return out

def infer_home_from_job(job_html: str, job_url: str, company_name: str) -> List[str]:
    soup = soup_html(job_html)
    anchors = soup.find_all("a", href=True)
    cands=[]
    cmp_l = (company_name or "").lower()
    for a in anchors:
        href = a["href"].strip()
        txt  = (a.get_text(" ", strip=True) or "").lower()
        url  = urllib.parse.urljoin(job_url, href)
        host = urllib.parse.urlsplit(url).netloc.lower()
        if not host or _domain(host) in PORTAL_DOMAINS:
            continue
        score = 0
        if cmp_l and cmp_l in txt: score += 2
        if any(k in txt for k in [k.lower() for k in VISION_KEYS+TALENT_KEYS]): score += 2
        if url.endswith("/") or url.count("/") <= 3: score += 1
        cands.append((url, score))
    scored = Counter()
    for it in cands: scored[it[0]] += it[1]
    candidates = [u for u,_ in scored.most_common()]
    roots=[]; seen=set()
    for u in candidates:
        pu = urllib.parse.urlsplit(u)
        root = f"{pu.scheme}://{pu.netloc}/"
        if root not in seen:
            seen.add(root); roots.append(root)
    return roots[:5] or candidates[:5]

# ===================== 뉴스 (RSS) =========================
@st.cache_data(show_spinner=False, ttl=1200)
def google_news_rss_multi(company: str, max_items: int = 5) -> List[Dict]:
    def rss(q):
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=ko&gl=KR&ceid=KR:ko"
        try:
            r = requests.get(url, timeout=6)
            if r.status_code != 200: return []
            soup = soup_html(r.text)  # RSS는 xml이지만 제목/링크만 단순 추출하므로 html 파서로 충분
            items = soup.find_all("item")
            out=[]
            for it in items[:max_items]:
                title = it.find("title").get_text(strip=True) if it.find("title") else ""
                link  = it.find("link").get_text(strip=True) if it.find("link") else ""
                pub   = it.find("pubdate").get_text(strip=True) if it.find("pubdate") else ""
                out.append({"title": title, "link": link, "pubDate": pub})
            return out
        except Exception:
            return []
    for q in [company, f"\"{company}\" 회사", f"\"{company}\" 채용"]:
        res = rss(q)
        if res: return res
    return []

# ======================= 임베딩/RAG =======================
def chunk(text: str, size: int = 600, overlap: int = 120) -> List[str]:
    t = re.sub(r"\s+"," ", text or "").strip()
    out=[]; start=0; L=len(t)
    while start < L:
        end=min(L,start+size); out.append(t[start:end])
        if end==L: break
        start=max(0,end-overlap)
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

PROMPT_SYSTEM_Q = "너는 채용담당자다. 회사/직무/요건과 이력서를 고려해 좋은 한국어 질문을 만든다."
PROMPT_SYSTEM_DRAFT = "너는 면접 코치다. STAR로 8~12문장 한국어 답변 초안을 작성한다."
CRITERIA = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def llm_generate_one_question_with_resume(clean: Dict, level: str, model: str,
                                          resume_chunks: List[str], resume_embeds: np.ndarray) -> str:
    hits = retrieve_resume_chunks("핵심 프로젝트와 기술 스택 요약", resume_chunks, resume_embeds, k=4)
    resume_context = "\n".join([f"- {t[:350]}" for _, t in hits])[:1200]
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {"role":"user","content":(
        f"[회사/직무/요건]\n{ctx}\n\n[이력서 발췌]\n{resume_context}\n\n"
        f"[요청]\n- 난이도/연차: {level}\n- 한국어 면접 질문 1개만 한 줄로 출력")}
    try:
        r = client.chat.completions.create(model=model, temperature=0.8, max_tokens=120,
                                           messages=[{"role":"system","content":PROMPT_SYSTEM_Q}, user_msg])
        q = r.choices[0].message.content.strip()
        return re.sub(r"^\s*\d+[\).\s-]*","", q).split("\n")[0].strip()
    except Exception:
        return ""

def llm_draft_answer(clean: Dict, question: str, model: str,
                     resume_chunks: List[str], resume_embeds: np.ndarray) -> str:
    hits = retrieve_resume_chunks(question, resume_chunks, resume_embeds, k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {"role":"user","content":(
        f"[회사/직무/채용요건]\n{ctx}\n\n[이력서 발췌]\n{resume_text}\n\n[면접 질문]\n{question}\n\n"
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
        f"[회사/직무/채용요건]\n{ctx}\n\n[이력서 발췌]\n{resume_text}\n\n"
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
                                           messages=[{"role":"system","content":"면접 코치(엄격 모드)"}, user_msg])
        data = json.loads(r.choices[0].message.content)
        fixed=[]
        for name in CRITERIA:
            cur=None
            for it in data.get("criteria", []):
                if str(it.get("name","")).strip()==name: cur=it; break
            if not cur: cur={"name":name,"score":0,"comment":""}
            sc=int(cur.get("score",0)); sc=max(0,min(20,sc))
            cur["score"]=sc; cur["comment"]=str(cur.get("comment","")).strip()
            fixed.append(cur)
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

# ======================= 상태 =============================
def _init_state():
    defaults = dict(
        clean=None, last_html=None,
        company_vision=[], company_talent=[], company_news=[],
        auto_home_used="",
        resume_chunks=[], resume_embeds=None, resume_raw="",
        current_question="", answer_text="", last_result=None,
        followups=[], selected_followup="", followup_answer="",
    )
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v
_init_state()

# ======================= UI 1: 수집 =======================
st.header("1) 채용 공고 URL (Selenium 전용)")
job_url = st.text_input("채용 공고 상세 URL", placeholder="원티드/사람인/잡코리아/기업 채용 페이지 URL")
st.text_input("회사 공식 홈페이지 URL (선택)", key="company_home", placeholder="회사 공식 홈페이지 URL(미입력 시 자동 추론)")

if st.button("원문 수집 → 정제", type="primary"):
    if not job_url.strip():
        st.warning("채용 공고 URL을 입력하세요.")
    else:
        with st.spinner("채용 공고 로딩/정제 중..."):
            raw, meta, html = fetch_job_text_dynamic(job_url.strip(), timeout=SELENIUM_TIMEOUT)
            hint = extract_company_meta_from_html(html)
            st.session_state.last_html = html
            if not raw:
                st.error("채용 공고 수집 실패")
            else:
                st.caption(f"소스: {meta.get('source')} · 텍스트 길이: {meta.get('len')}")
                st.session_state.clean = llm_structurize(raw, hint, CHAT_MODEL)

        if st.session_state.clean:
            with st.spinner("회사 홈페이지(비전/인재상) 수집 중..."):
                vision, talent, used = [], [], ""
                home_input = st.session_state.company_home.strip()

                auto_candidates = infer_home_from_job(
                    job_html=html or "", job_url=meta.get("url_final","") or job_url,
                    company_name=st.session_state.clean.get("company_name","")
                )

                for candidate in ([home_input] if home_input else []) + auto_candidates:
                    if not candidate: continue
                    pages = company_home_harvest(candidate, prefer_dynamic=PREFER_DYNAMIC,
                                                 timeout=SELENIUM_TIMEOUT, max_pages=MAX_PAGES)
                    if pages.get("vision") or pages.get("talent"):
                        vision = pages.get("vision", [])
                        talent = pages.get("talent", [])
                        used = candidate if not home_input else home_input
                        break

                st.session_state.company_vision = vision
                st.session_state.company_talent = talent
                st.session_state.auto_home_used = used

            with st.spinner("뉴스 수집 중..."):
                cname = st.session_state.clean.get("company_name","") or ""
                st.session_state.company_news = google_news_rss_multi(cname, 5)

            st.success("정제 완료!")

# ======================= UI 2: 회사 요약 ===================
st.header("2) 회사 요약")
clean = st.session_state.clean
if clean:
    st.markdown(f"**회사명:** {clean.get('company_name','-')}")
    st.markdown(f"**간단한 회사 소개(요약):** {clean.get('company_intro','-')}")
    st.markdown(f"**모집 분야(직무명):** {clean.get('job_title','-')}")
    c1,c2,c3 = st.columns(3)
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
            st.caption("명시된 우대 사항이 없습니다.")
else:
    st.info("먼저 URL을 정제해 주세요.")

# =================== UI 2.5: 비전/인재상/뉴스 =============
st.divider()
st.subheader("회사 비전/인재상 & 최신 이슈")
vcol, tcol = st.columns(2)
with vcol:
    st.markdown("**비전/핵심가치**")
    if st.session_state.company_vision:
        for v in st.session_state.company_vision[:8]: st.markdown(f"- {v}")
    else:
        st.caption("비전/핵심가치를 찾지 못했습니다.")
with tcol:
    st.markdown("**인재상**")
    if st.session_state.company_talent:
        for t in st.session_state.company_talent[:8]: st.markdown(f"- {t}")
    else:
        st.caption("인재상 정보를 찾지 못했습니다.")
if st.session_state.auto_home_used:
    st.caption(f"회사 홈페이지 사용: {st.session_state.auto_home_used}")

st.markdown("**최신 뉴스(상위 3~5)**")
if st.session_state.company_news:
    for n in st.session_state.company_news[:5]:
        st.markdown(f"- [{n.get('title','(제목 없음)')}]({n.get('link','#')})")
else:
    st.caption("뉴스 결과가 없습니다.")

st.divider()

# ======================= UI 3: 이력서 인덱싱 ===============
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader("여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True)
_RESUME_CHUNK=500; _RESUME_OVLP=100
try:
    import pypdf
except Exception:
    pypdf = None
try:
    import docx2txt
except Exception:
    docx2txt = None

def read_pdf(data: bytes) -> str:
    if pypdf is None: return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
    except Exception: return ""

def read_docx(data: bytes) -> str:
    if docx2txt is None: return ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
            tmp.write(data); tmp.flush()
            return docx2txt.process(tmp.name) or ""
    except Exception: return ""

def read_uploaded(up) -> str:
    name = up.name.lower(); data = up.read()
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
            t = read_uploaded(up)
            if t: texts.append(t)
        resume_text = "\n\n".join(texts)
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

# ======================= UI 4~7: Q/A & 코칭 ===============
st.header("4) 질문 생성 & 답변 초안 (RAG)")
level = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)

c1, c2 = st.columns(2)
with c1:
    if st.button("새 질문", type="primary"):
        if not st.session_state.clean:
            st.warning("먼저 회사 URL을 정제하세요.")
        else:
            q = llm_generate_one_question_with_resume(
                st.session_state.clean, level, CHAT_MODEL,
                st.session_state.resume_chunks, st.session_state.resume_embeds
            )
            if q:
                st.session_state.current_question = q
                st.session_state.answer_text = ""
                st.session_state.last_result = None
                st.session_state.followups = []
                st.success("질문 생성 완료!")
            else:
                st.error("질문 생성 실패")
with c2:
    if st.button("RAG로 답변 초안", type="secondary"):
        if not st.session_state.current_question:
            st.warning("먼저 질문을 생성하세요.")
        else:
            draft = llm_draft_answer(
                st.session_state.clean, st.session_state.current_question, CHAT_MODEL,
                st.session_state.resume_chunks, st.session_state.resume_embeds
            )
            if draft:
                st.session_state.answer_text = draft
                st.success("초안 생성 완료!")
            else:
                st.error("초안 생성 실패")

st.text_area("질문", value=st.session_state.current_question, height=90)
st.text_area("나의 답변 (초안을 편집해 완성)", key="answer_text", height=200)

st.header("5) 채점 & 코칭")
if st.button("채점 실행", type="primary"):
    if not st.session_state.current_question:
        st.warning("먼저 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("채점/코칭 중..."):
            res = llm_score_and_coach_strict(
                st.session_state.clean,
                st.session_state.current_question,
                st.session_state.answer_text,
                CHAT_MODEL,
                st.session_state.resume_chunks,
                st.session_state.resume_embeds
            )
        st.session_state.last_result = res
        st.success("완료!")

st.divider()
st.subheader("피드백 결과")
last = st.session_state.last_result
if last:
    st.metric("총점(/100)", last.get("overall_score", 0))
    st.markdown("**기준별 점수 & 코멘트**")
    for it in last.get("criteria", []):
        st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
    if last.get("strengths"):
        st.markdown("**강점**");  [st.markdown(f"- {s}") for s in last["strengths"]]
    if last.get("risks"):
        st.markdown("**감점 요인/리스크**"); [st.markdown(f"- {r}") for r in last["risks"]]
    if last.get("improvements"):
        st.markdown("**개선 포인트**"); [st.markdown(f"- {x}") for x in last["improvements"]]
    if last.get("revised_answer"):
        st.markdown("**수정본(STAR)**"); st.write(last["revised_answer"])
else:
    st.caption("아직 채점 결과가 없습니다.")
