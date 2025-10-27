# -*- coding: utf-8 -*-
"""
Job Helper Bot (동적 크롤링 + 우대 사항까지 확보)
- 1) 채용 공고 URL → 정제(정적 수집 → 포털 정밀 파서 → (부족 시) 동적 크롤링으로 '더보기' 클릭 → JSON 상태 파싱 → LLM 보강)
- 2) 회사 요약(정제 결과)
- 3) 이력서/프로젝트 업로드 (pdf/txt/md/docx) + 인덱싱
- 4) 이력서 기반 자소서 생성 (세션 상태에 저장되어 사라지지 않음)
- 5) 질문 생성 & 답변 초안 (이력서 RAG 반영)
- 6) 채점 & 코칭 (총점=기준 다섯 항목 합계, 일관성 유지)
- 7) 팔로업 질문 & 답변
- 8) 팔로업 피드백
"""

import os, re, io, json, time, random, urllib.parse
from typing import Optional, Dict, List, Tuple

import requests
import streamlit as st
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import html2text

# ---------------- OpenAI ----------------
try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop()

# ---------------- PDF/DOCX ----------------
try:
    import pypdf
except Exception:
    pypdf = None

# ---------------- 페이지 설정 ----------------
st.set_page_config(page_title="Job Helper Bot", page_icon="🧭", layout="wide")
st.title("Job Helper Bot — 회사 정제/자소서/질문/채점/팔로업 (동적 크롤링 지원)")

# ---------------- OpenAI 키/모델 ----------------
API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
if not API_KEY:
    API_KEY = st.text_input("OPENAI_API_KEY", type="password")
if not API_KEY:
    st.stop()
client = OpenAI(api_key=API_KEY)

with st.sidebar:
    st.subheader("모델/옵션")
    CHAT_MODEL  = st.selectbox("대화/생성 모델", ["gpt-4o-mini", "gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small","text-embedding-3-large"], index=0)
    ENABLE_DYNAMIC = st.toggle("동적 크롤링(Playwright) 사용", value=True)
    st.caption("첫 실행 전 1회: `python -m playwright install chromium`")

# ---------------- 유틸 ----------------
def normalize_url(u: str) -> Optional[str]:
    if not u: return None
    u = u.strip()
    if not re.match(r"^https?://", u): u = "https://" + u
    parts = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

def http_get(url: str, timeout: int = 12) -> Optional[requests.Response]:
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
                "Accept-Language": "ko, en;q=0.9",
            },
            timeout=timeout,
        )
        if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
            return r
    except Exception:
        pass
    return None

def html_to_text(html_str: str) -> str:
    conv = html2text.HTML2Text()
    conv.ignore_links = True
    conv.ignore_images = True
    conv.body_width = 0
    txt = conv.handle(html_str or "")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

# ---------------- 정적 수집 ----------------
def fetch_jina_text(url: str, timeout: int = 15) -> str:
    """Jina proxy로 간단 텍스트 수집(일부 사이트 차단 가능)"""
    try:
        parts = urllib.parse.urlsplit(url)
        prox = f"https://r.jina.ai/http://{parts.netloc}{parts.path}"
        if parts.query: prox += f"?{parts.query}"
        r = http_get(prox, timeout=timeout)
        return r.text.strip() if r else ""
    except Exception:
        return ""

def fetch_web_text(url: str) -> str:
    r = http_get(url, timeout=12)
    if not r: return ""
    return html_to_text(r.text)

def fetch_bs4_text(url: str) -> Tuple[str, Optional[BeautifulSoup], Optional[str]]:
    r = http_get(url, timeout=12)
    if not r: return "", None, None
    soup = BeautifulSoup(r.text, "lxml")
    return soup.get_text(" ", strip=True)[:120000], soup, r.text

def fetch_all_text_static(url: str):
    url = normalize_url(url)
    if not url: return "", {"error":"invalid_url"}, None, None
    # 1) Jina
    jina = fetch_jina_text(url)
    if jina:
        _, soup, html = fetch_bs4_text(url)
        return jina, {"source":"jina","len":len(jina),"url":url}, soup, html
    # 2) 직접 텍스트 변환
    web = fetch_web_text(url)
    if web:
        _, soup, html = fetch_bs4_text(url)
        return web, {"source":"web","len":len(web),"url":url}, soup, html
    # 3) 최소 보장
    bs, soup, html = fetch_bs4_text(url)
    return bs, {"source":"bs4","len":len(bs),"url":url}, soup, html

# ---------------- 동적 수집: '더보기' 클릭 강화판 ----------------
def fetch_dynamic_html(url: str, site_hint: str | None = None,
                       max_rounds: int = 5, wait_ms: int = 700) -> str:
    """Playwright로 접속 → '더보기/펼치기/상세' 후보를 여러 방식으로 클릭 → DOM 변화 없으면 종료 → 최종 HTML 반환"""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return ""

    def _candidate_selectors(site: str | None):
        common = [
            'button:has-text("더보기")',
            'button:has-text("더 보기")',
            'button:has-text("전체 보기")',
            'button:has-text("상세 보기")',
            'button:has-text("펼치기")',
            '[role="button"]:has-text("더보기")',
            '[data-testid*="more"]',
            '[data-cy*="more"]',
            '[aria-label*="더보기"]',
            '.more, .btn-more, .see-more, .read-more, .expand, .toggle',
            '[class*="fold"] button, [class*="fold"] a',
            '[class*="collapsed"] button, [class*="collapsed"] a',
        ]
        site = (site or "").lower()
        if "wanted" in site:
            common += [
                'button:has-text("상세요강 더보기")',
                '[data-accordion-button]',
            ]
        if "saramin" in site:
            common += [
                '#btnReadMore, .btn_open, .btn_more, .recruit_more',
                '.wrap_jview .btn_toggle, .wrap_jview .btn_more',
            ]
        if "jobkorea" in site:
            common += [
                '.btnMore, .foldBtn, .lyBtnMore',
                '.tbDetail .btn, .tbSupport .btn',
            ]
        return list(dict.fromkeys(common))

    html = ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            ctx = browser.new_context(locale="ko-KR")
            page = ctx.new_page()
            page.set_default_timeout(15000)
            page.goto(url, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=12000)
            except Exception:
                pass

            last_height = -1
            selectors = _candidate_selectors(site_hint)

            for _round in range(max_rounds):
                clicked_any = False

                # 텍스트 기반 클릭
                for text in ["더보기", "더 보기", "전체 보기", "상세 보기", "펼치기", "접기"]:
                    try:
                        loc = page.get_by_text(text, exact=False)
                        count = loc.count()
                        for i in range(min(count, 6)):
                            try:
                                el = loc.nth(i)
                                el.scroll_into_view_if_needed()
                                el.click()
                                page.wait_for_timeout(wait_ms)
                                clicked_any = True
                            except Exception:
                                continue
                    except Exception:
                        pass

                # 셀렉터 기반 클릭
                for sel in selectors:
                    try:
                        loc = page.locator(sel)
                        count = loc.count()
                        if count == 0: 
                            continue
                        for i in range(min(count, 6)):
                            try:
                                el = loc.nth(i)
                                el.scroll_into_view_if_needed()
                                el.click()
                                page.wait_for_timeout(wait_ms)
                                clicked_any = True
                            except Exception:
                                continue
                    except Exception:
                        continue

                try:
                    cur_height = page.evaluate("()=>document.body.scrollHeight")
                except Exception:
                    cur_height = -1

                if not clicked_any and (cur_height == last_height):
                    break
                last_height = cur_height

            html = page.content()
            ctx.close()
            browser.close()
    except Exception:
        html = ""
    return html

# ---------------- JSON 상태(__NEXT_DATA__ 등) 파싱 ----------------
def extract_json_blobs_from_soup(soup: BeautifulSoup) -> list:
    blobs = []
    for s in soup.find_all("script"):
        text = (s.string or s.get_text() or "").strip()
        if not text:
            continue
        if text.startswith("{") or text.startswith("["):
            try:
                blobs.append(json.loads(text))
            except Exception:
                pass
    return blobs

def pick_sections_from_json_blobs(blobs: list) -> dict:
    out = {"responsibilities": [], "qualifications": [], "preferences": []}
    seen = {k:set() for k in out}

    def _push(k, v):
        t = re.sub(r"\s+"," ", str(v)).strip(" -•·▶▪️").strip()
        if t and t not in seen[k]:
            seen[k].add(t); out[k].append(t[:180])

    pref_kw = re.compile(r"(우대|선호|preferred|plus|Nice\s*to\s*have|가산점)", re.I)
    qual_kw = re.compile(r"(자격|요건|qualification|requirement)", re.I)
    resp_kw = re.compile(r"(업무|담당|role|responsibilit)", re.I)

    def walk(o):
        if isinstance(o, dict):
            for k,v in o.items():
                kl = str(k).lower()
                if isinstance(v, list):
                    if pref_kw.search(kl):
                        for it in v: _push("preferences", it)
                    elif qual_kw.search(kl):
                        for it in v: _push("qualifications", it)
                    elif resp_kw.search(kl):
                        for it in v: _push("responsibilities", it)
                elif isinstance(v, str):
                    if pref_kw.search(kl):
                        for it in re.split(r"[•\-\n·▪️▶]+", v): _push("preferences", it)
                    elif qual_kw.search(kl):
                        for it in re.split(r"[•\-\n·▪️▶]+", v): _push("qualifications", it)
                    elif resp_kw.search(kl):
                        for it in re.split(r"[•\-\n·▪️▶]+", v): _push("responsibilities", it)
                walk(v)
        elif isinstance(o, list):
            for it in o: walk(it)

    for b in blobs: walk(b)

    # 자격요건에 섞인 우대 키워드 이동
    moved, remain = [], []
    pref_kw2 = re.compile(r"(우대|선호|preferred|plus|Nice\s*to\s*have|가산점)", re.I)
    for q in out["qualifications"]:
        if pref_kw2.search(q): moved.append(q)
        else: remain.append(q)
    out["qualifications"] = remain
    out["preferences"] = (out["preferences"] + moved)[:12]
    for k in out: out[k] = out[k][:12]
    return out

# ---------------- 포털 정밀 파서 ----------------
RESP_HDR = re.compile(r"(주요\s*업무|담당\s*업무|Role|Responsibilities?)", re.I)
QUAL_HDR = re.compile(r"(자격\s*요건|지원\s*자격|Requirements?|Qualifications?)", re.I)
PREF_HDR = re.compile(r"(우대\s*사항|우대|Preferred|Nice\s*to\s*have|Plus)", re.I)
PREF_KW  = re.compile(r"(우대|선호|preferred|plus|가산점|Nice\s*to\s*have)", re.I)

def _clean_line(s: str) -> str:
    return re.sub(r"\s+"," ", s or "").strip(" -•·▶▪️")[:180]

def _push_unique(bucket: List[str], text: str, seen: set):
    t = _clean_line(text)
    if t and t not in seen:
        seen.add(t); bucket.append(t)

def collect_after_heading(soup: BeautifulSoup, head_regex: re.Pattern, limit: int = 16) -> List[str]:
    out, seen = [], set()
    heads = [tag for tag in soup.find_all(re.compile("^h[1-4]$")) if head_regex.search(tag.get_text(" ", strip=True) or "")]
    for h in heads:
        sib = h.find_next_sibling()
        while sib and sib.name not in {"h1","h2","h3","h4"} and len(out) < limit:
            if sib.name in {"ul","ol"}:
                for li in sib.find_all("li", recursive=True):
                    _push_unique(out, li.get_text(" ", strip=True), seen)
                    if len(out) >= limit: break
            elif sib.name in {"p","div","section"}:
                txt = sib.get_text(" ", strip=True)
                if len(txt) > 4:
                    for l in re.split(r"[•\-\n·▪️▶]+|\s{2,}", txt):
                        _push_unique(out, l, seen)
                        if len(out) >= limit: break
            sib = sib.find_next_sibling()
        if len(out) >= limit: break
    return out[:limit]

def parse_wanted(soup: BeautifulSoup) -> Dict[str, List[str]]:
    res  = collect_after_heading(soup, RESP_HDR, 16)
    qual = collect_after_heading(soup, QUAL_HDR, 16)
    pref = collect_after_heading(soup, PREF_HDR, 16)
    # 자격요건에 섞인 우대 이동
    remain, moved = [], []
    for q in qual:
        (moved if PREF_KW.search(q) else remain).append(q)
    pref += moved
    return {"responsibilities": res[:12], "qualifications": remain[:12], "preferences": pref[:12]}

def parse_saramin(soup: BeautifulSoup) -> Dict[str, List[str]]:
    out = {"responsibilities": [], "qualifications": [], "preferences": []}
    out["responsibilities"] += collect_after_heading(soup, RESP_HDR, 16)
    out["qualifications"]   += collect_after_heading(soup, QUAL_HDR, 16)
    out["preferences"]      += collect_after_heading(soup, PREF_HDR, 16)
    # DL 구조 보조
    for dl in soup.find_all("dl"):
        for dt in dl.find_all("dt", recursive=False):
            title = dt.get_text(" ", strip=True) or ""
            dd = dt.find_next_sibling("dd")
            if not dd: continue
            text = dd.get_text(" ", strip=True)
            if not text: continue
            lines = re.split(r"[•\-\n·▪️▶]+|\s{2,}", text)
            if RESP_HDR.search(title):
                out["responsibilities"] += lines
            elif QUAL_HDR.search(title):
                out["qualifications"] += lines
            elif PREF_HDR.search(title) or PREF_KW.search(title):
                out["preferences"] += lines
    # 자격요건에 섞인 우대 이동
    remain, moved = [], []
    for q in out["qualifications"]:
        (moved if PREF_KW.search(q) else remain).append(q)
    out["qualifications"] = remain
    out["preferences"] += moved

    # 클린
    for k in out:
        seen=set(); clean=[]
        for s in out[k]:
            s=_clean_line(s)
            if s and s not in seen:
                seen.add(s); clean.append(s)
        out[k]=clean[:12]
    return out

def parse_jobkorea(soup: BeautifulSoup) -> Dict[str, List[str]]:
    res  = collect_after_heading(soup, re.compile(r"(상세\s*요강|주요\s*업무|담당\s*업무|Responsibilities?)", re.I), 16)
    qual = collect_after_heading(soup, re.compile(r"(지원\s*자격|자격\s*요건|Requirements?|Qualifications?)", re.I), 16)
    pref = collect_after_heading(soup, re.compile(r"(우대\s*사항|우대|Preferred|Plus|Nice\s*to\s*have)", re.I), 16)
    # 이동
    remain, moved = [], []
    for q in qual:
        (moved if PREF_KW.search(q) else remain).append(q)
    return {"responsibilities": res[:12], "qualifications": remain[:12], "preferences": (pref+moved)[:12]}

def parse_portal_specific(url: str, soup: Optional[BeautifulSoup], raw_text: str) -> Dict[str, List[str]]:
    out = {"responsibilities":[], "qualifications":[], "preferences":[]}
    if not soup:
        # 최소 라인 스캔
        lines = [ _clean_line(x) for x in (raw_text or "").split("\n") if x.strip() ]
        bucket = None
        for l in lines:
            if RESP_HDR.search(l): bucket="responsibilities"; continue
            if QUAL_HDR.search(l): bucket="qualifications"; continue
            if PREF_HDR.search(l) or PREF_KW.search(l): bucket="preferences"; continue
            if bucket: out[bucket].append(l)
        for k in out:
            seen=set(); clean=[]
            for s in out[k]:
                s=_clean_line(s)
                if s and s not in seen: seen.add(s); clean.append(s)
            out[k]=clean[:12]
        return out

    host = urllib.parse.urlsplit(normalize_url(url) or "").netloc.lower()
    if "wanted.co.kr" in host:   out = parse_wanted(soup)
    elif "saramin.co.kr" in host: out = parse_saramin(soup)
    elif "jobkorea.co.kr" in host: out = parse_jobkorea(soup)
    else:
        out["responsibilities"] = collect_after_heading(soup, RESP_HDR, 16)
        out["qualifications"]   = collect_after_heading(soup, QUAL_HDR, 16)
        out["preferences"]      = collect_after_heading(soup, PREF_HDR, 16)

    # 자격요건 내 우대 키워드 보정
    remain, moved = [], []
    for q in out.get("qualifications", []):
        (moved if PREF_KW.search(q) else remain).append(q)
    out["qualifications"] = remain
    out["preferences"] = (out.get("preferences", []) + moved)[:12]

    # 중복 정리
    for k in out:
        seen=set(); clean=[]
        for s in out[k]:
            s=_clean_line(s)
            if s and s not in seen:
                seen.add(s); clean.append(s)
        out[k]=clean[:12]
    return out

# ---------------- 메타 추출 & LLM 정제 ----------------
def extract_company_meta(soup: Optional[BeautifulSoup]) -> Dict[str,str]:
    meta = {"company_name":"","company_intro":"","job_title":""}
    if not soup: return meta
    cand = []
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

PROMPT_SYSTEM_STRUCT = ("너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
                        "입력 텍스트는 포털 광고 문구, UI잔재, 복수 직무가 섞여 있을 수 있다. "
                        "한국어로 간결하고 중복없이 정제하라.")

def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
    ctx = (raw_text or "").strip()
    if len(ctx) > 14000: ctx = ctx[:14000]
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
                            "- 불릿/마커/이모지 제거, 문장 간결화, 중복 제거.")}

    try:
        resp = client.chat.completions.create(model=model, temperature=0.2,
                                              response_format={"type":"json_object"},
                                              messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg])
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        data = {"company_name": meta_hint.get("company_name",""),
                "company_intro": meta_hint.get("company_intro","원문이 정제되지 않았습니다."),
                "job_title": meta_hint.get("job_title",""),
                "responsibilities": [],
                "qualifications": [],
                "preferences": [],
                "error": str(e)}

    # 클린
    for k in ["responsibilities","qualifications","preferences"]:
        arr = data.get(k, [])
        if not isinstance(arr, list): arr = []
        clean_list=[]; seen=set()
        for it in arr:
            t = re.sub(r"\s+"," ", str(it)).strip(" -•·▶▪️").strip()
            if t and t not in seen:
                seen.add(t); clean_list.append(t[:180])
        data[k] = clean_list[:12]
    for k in ["company_name","company_intro","job_title"]:
        if k in data and isinstance(data[k], str):
            data[k] = re.sub(r"\s+"," ", data[k]).strip()

    # 우대 보정: 자격요건에 섞인 우대 이동
    if len(data.get("preferences", [])) < 1:
        pref_kw = re.compile(r"(우대|선호|preferred|plus|Nice\s*to\s*have|가산점)", re.I)
        remain, moved = [], []
        for q in data.get("qualifications", []):
            (moved if pref_kw.search(q) else remain).append(q)
        if moved:
            data["preferences"] = moved[:12]
            data["qualifications"] = remain[:12]
    return data

# ---------------- 파일 리더 ----------------
def read_pdf(data: bytes) -> str:
    if pypdf is None: return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
    except Exception:
        return ""

def read_docx(data: bytes) -> str:
    # docx2txt 우선, 실패 시 python-docx fallback
    try:
        import docx2txt, tempfile
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
            tmp.write(data); tmp.flush()
            return docx2txt.process(tmp.name) or ""
    except Exception:
        try:
            import docx as docxlib
            f = io.BytesIO(data)
            doc = docxlib.Document(f)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception:
            return ""

def read_file_text(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith((".txt",".md")):
        for enc in ("utf-8","cp949","euc-kr"):
            try: return data.decode(enc)
            except Exception: continue
        return data.decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        return read_pdf(data)
    elif name.endswith(".docx"):
        return read_docx(data)
    return ""

# ---------------- 임베딩/RAG ----------------
def chunk(text: str, size: int = 600, overlap: int = 120) -> List[str]:
    t = re.sub(r"\s+"," ", text).strip()
    if not t: return []
    out, start = [], 0
    while start < len(t):
        end = min(len(t), start+size)
        out.append(t[start:end])
        if end == len(t): break
        start = max(0, end-overlap)
    return out

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    if not texts: return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=model_name, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def cosine_topk(matrix: np.ndarray, query_vec: np.ndarray, k: int = 4):
    if matrix.size == 0: return np.array([]), np.array([], dtype=int)
    qn = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn.T
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx

def retrieve_resume_chunks(query: str, k: int = 4):
    chs, embs = st.session_state.get("resume_chunks", []), st.session_state.get("resume_embeds", None)
    if not chs or embs is None: return []
    qv = embed_texts([query], EMBED_MODEL)
    scores, idxs = cosine_topk(embs, qv, k=k)
    return [(float(s), chs[int(i)]) for s, i in zip(scores, idxs)]

# ---------------- 뉴스 (선택) ----------------
def _load_naver_keys():
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
    cid, csec = _load_naver_keys()
    if not (cid and csec): return []
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec}
    try:
        r = requests.get(url, headers=headers, params={"query": company, "display": display, "sort":"date"}, timeout=8)
        if r.status_code != 200: return []
        js=r.json()
        items=[]
        for it in js.get("items", []):
            title = re.sub(r"</?b>|&quot;|&apos;|&amp;|&lt|&gt;", "", it.get("title","")).strip()
            items.append({"title": title, "link": it.get("link"), "pubDate": it.get("pubDate")})
        return items
    except Exception:
        return []

def google_news_rss(company: str, max_items: int = 5) -> List[Dict]:
    q = urllib.parse.quote(company)
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        r = requests.get(url, timeout=8)
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

def fetch_latest_news(company: str, max_items: int = 5) -> List[Dict]:
    items = naver_search_news(company, display=max_items)
    if items: return items
    return google_news_rss(company, max_items=max_items)

# ---------------- 질문/초안/채점/팔로업 LLM ----------------
CRITERIA = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

PROMPT_SYSTEM_Q = ("너는 채용담당자다. 회사/직무 맥락과 채용요건, 그리고 지원자의 이력서 요약을 함께 고려해 "
                   "서로 다른 관점의 한국어 면접 질문을 생성한다. 수치/지표/기간/규모/리스크/트레이드오프 등을 섞어라.")
PROMPT_SYSTEM_DRAFT = ("너는 면접 답변 코치다. 회사/직무/채용요건과 지원자의 이력서 요약을 결합해 "
                       "STAR(상황-과제-행동-성과) 기반 답변 **초안**을 8~12문장으로 한국어로 작성한다.")
PROMPT_SYSTEM_SCORE = ("너는 매우 엄격한 면접 코치다. 아래 형식의 JSON만 출력하라. "
                       "각 기준은 0~20 정수이며, 총점은 합계(최대 100)와 반드시 일치해야 한다. "
                       "각 기준에 대해 짧고 구체적인 코멘트를 제공하라.")

def llm_generate_one_question_with_resume(clean: Dict, level: str, model: str) -> str:
    hits = retrieve_resume_chunks("핵심 프로젝트와 기술 스택 요약", k=4)
    resume_snips = [t for _, t in hits]
    resume_context = "\n".join([f"- {s[:350]}" for s in resume_snips])[:1200]
    ctx = json.dumps(clean, ensure_ascii=False)
    user = {"role":"user",
            "content": (f"[회사/직무/요건]\n{ctx}\n\n"
                        f"[지원자 이력서 발췌]\n{resume_context}\n\n"
                        f"[요청]\n- 난이도/연차: {level}\n- 한국어 면접 질문 1개만 한 줄로 출력")}
    try:
        r = client.chat.completions.create(model=model, temperature=0.8,
                                           messages=[{"role":"system","content":PROMPT_SYSTEM_Q}, user])
        q = r.choices[0].message.content.strip()
        q = re.sub(r"^\s*\d+[\).\s-]*","", q).strip()
        return q.split("\n")[0].strip()
    except Exception:
        return ""

def llm_draft_answer(clean: Dict, question: str, model: str) -> str:
    hits = retrieve_resume_chunks(question, k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    ctx = json.dumps(clean, ensure_ascii=False)
    user = {"role":"user",
            "content": (f"[회사/직무/채용요건]\n{ctx}\n\n"
                        f"[지원자 이력서 발췌]\n{resume_text}\n\n"
                        f"[면접 질문]\n{question}\n\n"
                        "위 정보를 근거로 STAR 기반 한국어 답변 **초안**을 작성해줘.")}
    try:
        r = client.chat.completions.create(model=model, temperature=0.5,
                                           messages=[{"role":"system","content":PROMPT_SYSTEM_DRAFT}, user])
        return r.choices[0].message.content.strip()
    except Exception:
        return ""

def llm_score_and_coach_strict(clean: Dict, question: str, answer: str, model: str) -> Dict:
    hits = retrieve_resume_chunks(question + "\n" + answer[:800], k=4)
    resume_text = "\n".join([f"- {t[:400]}" for _, t in hits])[:1600]
    ctx = json.dumps(clean, ensure_ascii=False)
    user = {"role":"user",
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
                        "\"strengths\": [\"...\"],"
                        "\"risks\": [\"...\"],"
                        "\"improvements\": [\"...\"],"
                        "\"revised_answer\": \"STAR 구조로 간결히\""
                        "}")}

    try:
        r = client.chat.completions.create(model=model, temperature=0.2, response_format={"type":"json_object"},
                                           messages=[{"role":"system","content":PROMPT_SYSTEM_SCORE}, user])
        data = json.loads(r.choices[0].message.content)
    except Exception as e:
        return {"overall_score": 0,
                "criteria": [{"name": n, "score": 0, "comment": ""} for n in CRITERIA],
                "strengths": [], "risks": [], "improvements": [], "revised_answer": "",
                "error": str(e)}

    # 정합성 보정
    fixed=[]
    for name in CRITERIA:
        found=None
        for it in data.get("criteria", []):
            if str(it.get("name","")).strip()==name:
                found=it; break
        if not found: found={"name":name,"score":0,"comment":""}
        sc = int(found.get("score",0)); sc=max(0,min(20,sc))
        found["score"]=sc; found["comment"]=str(found.get("comment","")).strip()
        fixed.append(found)
    total=sum(x["score"] for x in fixed)
    data["criteria"]=fixed
    data["overall_score"]=total
    for k in ["strengths","risks","improvements"]:
        arr=data.get(k,[]); 
        if not isinstance(arr,list): arr=[]
        data[k]=[str(x).strip() for x in arr if str(x).strip()][:5]
    data["revised_answer"]=str(data.get("revised_answer","")).strip()
    return data

# ---------------- 세션 상태 ----------------
def _init_state():
    defaults = {
        "clean_struct": None,
        "company_news": [],
        "resume_raw": "",
        "resume_chunks": [],
        "resume_embeds": None,
        "cover_letter": "",
        "current_question": "",
        "answer_text": "",
        "records": [],
        "last_result": None,
        "followups": [],
        "selected_followup": "",
        "followup_answer": "",
        "last_followup_result": None,
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v
_init_state()

# =============================================================================
# 1) 채용 공고 URL → 정제
# =============================================================================
st.header("1) 채용 공고 URL → 정제")
url_input = st.text_input("채용 공고 상세 URL", placeholder="원티드/사람인/잡코리아 등 상세 URL을 입력")
if st.button("원문 수집 → 정제", type="primary"):
    if not url_input.strip():
        st.warning("URL을 입력하세요.")
    else:
        url = url_input.strip()
        # 1) 정적 수집
        with st.spinner("정적 수집 중..."):
            raw, meta, soup, html_raw = fetch_all_text_static(url)
            hint = extract_company_meta(soup) if soup else {"company_name":"","company_intro":"","job_title":""}

        # 2) 1차 정밀 파싱
        site_struct = parse_portal_specific(url, soup, raw)
        ok_cnt = sum(len(site_struct.get(k, [])) for k in ["responsibilities","qualifications","preferences"])

        # 3) 부족하면 동적 크롤링 + JSON 상태 파싱 → 재파싱
        if ENABLE_DYNAMIC and ok_cnt < 3:
            host = urllib.parse.urlsplit(url).netloc.lower()
            with st.spinner("동적 수집(상세 정보 더보기 클릭) 중..."):
                dyn_html = fetch_dynamic_html(url, site_hint=host, max_rounds=6, wait_ms=700)
            if dyn_html:
                soup_dyn = BeautifulSoup(dyn_html, "lxml")
                raw_dyn = html_to_text(dyn_html)
                # JSON 블롭에서 섹션 먼저 시도
                blobs = extract_json_blobs_from_soup(soup_dyn)
                from_json = pick_sections_from_json_blobs(blobs)
                # DOM 정밀 파서도 병행
                site_struct_dyn = parse_portal_specific(url, soup_dyn, raw_dyn)
                # 병합(풍부한 항목 우선)
                merged = {"responsibilities": [], "qualifications": [], "preferences": []}
                for key in merged:
                    cand = (from_json.get(key, []) or []) + (site_struct_dyn.get(key, []) or []) + (site_struct.get(key, []) or [])
                    seen=set(); tmp=[]
                    for s in cand:
                        s = re.sub(r"\s+"," ", s).strip(" -•·▶▪️").strip()
                        if s and s not in seen:
                            seen.add(s); tmp.append(s[:180])
                    merged[key] = tmp[:12]
                site_struct = merged
                ok_cnt = sum(len(site_struct.get(k, [])) for k in ["responsibilities","qualifications","preferences"])

        # 4) 그래도 비면 LLM 정제 보완
        if ok_cnt < 3:
            with st.spinner("LLM 정제 보완 중..."):
                clean = llm_structurize(raw, hint, CHAT_MODEL)
            for k in ["responsibilities","qualifications","preferences"]:
                if site_struct.get(k):
                    clean[k] = site_struct[k]
        else:
            clean = {
                "company_name": hint.get("company_name",""),
                "company_intro": hint.get("company_intro",""),
                "job_title": hint.get("job_title",""),
                "responsibilities": site_struct.get("responsibilities",[]),
                "qualifications":   site_struct.get("qualifications",[]),
                "preferences":      site_struct.get("preferences",[]),
            }

        st.session_state.clean_struct = clean
        with st.spinner("최신 뉴스 확인 중..."):
            cname = clean.get("company_name") or hint.get("company_name") or ""
            st.session_state.company_news = fetch_latest_news(cname, max_items=5) if cname else []
        st.success("정제 완료!")

# =============================================================================
# 2) 회사 요약 (정제 결과)
# =============================================================================
st.header("2) 회사 요약 (정제 결과)")
clean = st.session_state.clean_struct
if clean:
    st.markdown(f"**회사명:** {clean.get('company_name','-')}")
    st.markdown(f"**간단한 회사 소개:** {clean.get('company_intro','-')}")
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
            st.caption("우대 사항이 명시되지 않았습니다.")
else:
    st.info("먼저 '원문 수집 → 정제'를 실행하세요.")

# =============================================================================
# 3) 내 이력서/프로젝트 업로드
# =============================================================================
st.divider()
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader("여러 개 업로드 가능", type=["pdf","txt","md","docx"], accept_multiple_files=True)

if st.button("이력서 인덱싱", type="secondary"):
    if not uploads:
        st.warning("파일을 업로드하세요.")
    else:
        all_text=[]
        for up in uploads:
            t = read_file_text(up)
            if t: all_text.append(t)
        resume_text = "\n\n".join(all_text)
        if not resume_text.strip():
            st.error("텍스트를 추출하지 못했습니다.")
        else:
            chunks = chunk(resume_text, size=600, overlap=120)
            with st.spinner("이력서 벡터화 중..."):
                embeds = embed_texts(chunks, EMBED_MODEL)
            st.session_state.resume_raw = resume_text
            st.session_state.resume_chunks = chunks
            st.session_state.resume_embeds = embeds
            st.success(f"인덱싱 완료 (청크 {len(chunks)}개)")

# =============================================================================
# 4) 이력서 기반 자소서 생성 (세션에 저장)
# =============================================================================
st.divider()
st.header("4) 이력서 기반 자소서 생성")
topic = st.text_input("회사 요청 주제(선택)", placeholder="예: 직무 지원동기 / 협업 경험 / 문제해결 사례 등")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str) -> str:
    enrich = {"news": [n.get("title","") for n in st.session_state.company_news[:3]]}
    company = json.dumps({"clean":clean_struct, "extra":enrich}, ensure_ascii=False)
    resume_snippet = resume_text.strip()
    if len(resume_snippet) > 9000: resume_snippet = resume_snippet[:9000]
    system = ("너는 한국어 자기소개서 전문가다. 채용 공고의 회사/직무 요건과 후보자의 이력서를 참고해 "
              "회사 특화 자소서를 작성한다. 수치/지표/기간/임팩트 중심으로 구체화한다.")
    req = f"회사 측 요청 주제는 '{topic_hint.strip()}' 이다." if topic_hint.strip() else \
          "특정 주제 요청이 없으므로, 채용 공고를 중심으로 지원동기와 직무적합성을 강조하라."
    user = (f"[회사/직무 요약(JSON)]\n{company}\n\n"
            f"[후보자 이력서(요약 가능)]\n{resume_snippet}\n\n"
            f"[작성 지시]\n- {req}\n"
            "- 분량: 600~900자\n"
            "- 구성: 1) 지원 동기 2) 직무 관련 핵심 역량·경험 3) 성과/지표 4) 입사 후 기여 방안 5) 마무리\n"
            "- 자연스럽고 진정성 있는 1인칭 서술. 불필요한 미사여구/중복 문구 삭제.")
    try:
        r = client.chat.completions.create(model=model, temperature=0.4,
                                           messages=[{"role":"system","content":system},{"role":"user","content":user}])
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"(자소서 생성 실패: {e})"

if st.button("자소서 생성", type="primary"):
    if not st.session_state.clean_struct:
        st.warning("먼저 회사 정제를 실행하세요.")
    elif not st.session_state.resume_raw.strip():
        st.warning("먼저 이력서를 업로드하고 인덱싱하세요.")
    else:
        with st.spinner("자소서 생성 중..."):
            cover = build_cover_letter(st.session_state.clean_struct, st.session_state.resume_raw, topic, CHAT_MODEL)
        st.session_state.cover_letter = cover
        st.success("자소서 생성 완료!")

if st.session_state.cover_letter:
    st.subheader("자소서 (생성 결과)")
    st.write(st.session_state.cover_letter)
    st.download_button("자소서 TXT 다운로드", data=st.session_state.cover_letter.encode("utf-8"),
                       file_name="cover_letter.txt", mime="text/plain")

# =============================================================================
# 5) 질문 생성 & 답변 초안
# =============================================================================
st.divider()
st.header("5) 질문 생성 & 답변 초안")
level = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)

c_q1, c_q2 = st.columns(2)
with c_q1:
    if st.button("새 질문 받기", type="primary"):
        if not st.session_state.clean_struct:
            st.warning("먼저 회사 정제를 실행하세요.")
        else:
            q = llm_generate_one_question_with_resume(st.session_state.clean_struct, level, CHAT_MODEL)
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
with c_q2:
    if st.button("RAG로 답변 초안 생성", type="secondary"):
        if not st.session_state.current_question:
            st.warning("먼저 질문을 생성하세요.")
        else:
            draft = llm_draft_answer(st.session_state.clean_struct, st.session_state.current_question, CHAT_MODEL)
            if draft:
                st.session_state.answer_text = draft
                st.success("초안 생성 완료!")
            else:
                st.error("초안 생성 실패")

st.text_area("질문", value=st.session_state.current_question, height=100)
st.text_area("나의 답변(초안을 편집해 완성하세요)", height=200, key="answer_text")

# =============================================================================
# 6) 채점 & 코칭
# =============================================================================
st.divider()
st.header("6) 채점 & 코칭")
if st.button("채점 & 코칭 실행", type="primary"):
    if not st.session_state.current_question:
        st.warning("먼저 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("채점/코칭 중..."):
            res = llm_score_and_coach_strict(st.session_state.clean_struct,
                                             st.session_state.current_question,
                                             st.session_state.answer_text,
                                             CHAT_MODEL)
        st.session_state.last_result = res
        st.session_state.records.append({
            "question": st.session_state.current_question,
            "answer": st.session_state.answer_text,
            "overall": res.get("overall_score", 0),
            "criteria": res.get("criteria", []),
            "strengths": res.get("strengths", []),
            "risks": res.get("risks", []),
            "improvements": res.get("improvements", []),
            "revised_answer": res.get("revised_answer","")
        })
        st.success("채점/코칭 완료!")

# =============================================================================
# 7) 팔로업 질문 & 답변
# =============================================================================
st.divider()
st.header("7) 팔로업 질문 & 답변")
last = st.session_state.last_result
if last and not st.session_state.followups:
    try:
        ctx = json.dumps({"company": st.session_state.clean_struct,
                          "news": [n.get("title","") for n in st.session_state.company_news[:3]]}, ensure_ascii=False)
        msg = {"role":"user",
               "content":(f"[회사/직무/요건/이슈]\n{ctx}\n\n"
                          f"[지원자 답변]\n{st.session_state.answer_text}\n\n"
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
        for i, f in enumerate(st.session_state.followups, 1):
            st.markdown(f"- ({i}) {f}")
        st.selectbox("채점 받을 팔로업 질문 선택", st.session_state.followups, index=0, key="selected_followup")
        st.text_area("팔로업 질문에 대한 나의 답변", height=160, key="followup_answer")
    else:
        st.caption("팔로업 질문은 메인 질문 채점 직후 자동 제안됩니다.")

# =============================================================================
# 8) 팔로업 피드백
# =============================================================================
st.divider()
st.header("8) 팔로업 피드백")
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
        st.success("팔로업 채점 완료!")

res_fu = st.session_state.last_followup_result
if res_fu:
    st.markdown("**팔로업 결과**")
    st.metric("총점(/100)", res_fu.get("overall_score", 0))
    for it in res_fu.get("criteria", []):
        st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
    if res_fu.get("revised_answer",""):
        st.markdown("**팔로업 수정본 (STAR)**")
        st.write(res_fu["revised_answer"])
