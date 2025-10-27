import os, shutil, streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

st.title("Selenium Mini Check")

cands = [
    os.getenv("CHROME_BIN"), os.getenv("GOOGLE_CHROME_BIN"),
    shutil.which("chromium"), shutil.which("chromium-browser"),
    shutil.which("google-chrome"), "/usr/bin/chromium", "/usr/bin/google-chrome"
]
binpath = next((c for c in cands if c and os.path.exists(c)), None)
st.write("Picked binary:", binpath)

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--disable-gpu")
opts.add_argument("--window-size=1200,900")
if binpath: opts.binary_location = binpath

try:
    driver = webdriver.Chrome(options=opts)   # Selenium Manager 사용
    driver.get("https://example.com")
    st.success("Selenium OK. Title: " + driver.title)
    driver.quit()
except Exception as e:
    st.error("Selenium failed: " + str(e))
