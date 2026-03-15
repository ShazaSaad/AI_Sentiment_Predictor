import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
import xml.etree.ElementTree as ET


# ================================
# CONFIG
# ================================

API_KEY = "bz.IC2EJ47677Z4C2TYZPM3D6D22HFVJBAL"

NEWS_URL = "https://api.benzinga.com/api/v2/news"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9"
}

OUTPUT_PATH = "data/raw_news/benzinga_news_raw.csv"


# ================================
# FETCH FULL ARTICLE HTML
# ================================

def fetch_article_html(url):
    """
    Download the HTML of a Benzinga article page.
    """
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.text


# ================================
# EXTRACT ARTICLE BODY + IMAGES
# ================================

def extract_benzinga_body(html_text):
    """
    Extract the article text and image URLs from Benzinga JSON-LD metadata.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    scripts = soup.find_all("script", type="application/ld+json")

    for script in scripts:
        try:
            data = json.loads(script.string)

            if isinstance(data, dict) and "articleBody" in data:

                body = BeautifulSoup(
                    data.get("articleBody"), "html.parser"
                ).get_text(separator="\n", strip=True)

                return body

        except Exception:
            continue

    return None, []


# ================================
# MAIN DATA COLLECTION FUNCTION
# ================================

def fetch_benzinga_news(page_size=30):

    if not API_KEY:
        raise ValueError("BENZINGA_API_KEY environment variable not set")

    params = {
        "token": API_KEY,
        "pageSize": page_size
    }

    print("Fetching news from Benzinga API...")

    response = requests.get(NEWS_URL, params=params)

    if response.status_code != 200:
        raise Exception("Failed to fetch Benzinga API")

    root = ET.fromstring(response.text)

    rows = []

    for item in root.findall(".//item"):

        title = item.findtext("title")
        article_url = item.findtext("url")

        if not title or not article_url:
            continue

        body = None
        images = []

        try:
            html_text = fetch_article_html(article_url)
            body = extract_benzinga_body(html_text)

        except Exception:
            print(f"Failed to fetch article body: {article_url}")

        rows.append({
            "id": item.findtext("id"),
            "author": item.findtext("author"),
            "created": item.findtext("created"),
            "updated": item.findtext("updated"),
            "title": title,
            "body": body,
            "url": article_url
        })

    df = pd.DataFrame(rows)

    os.makedirs("data/raw_news", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} articles to {OUTPUT_PATH}")


# ================================
# RUN SCRIPT
# ================================

if __name__ == "__main__":
    fetch_benzinga_news()