import requests
from bs4 import BeautifulSoup
import time
import json

BASE = "https://walkingdead.fandom.com/api.php"

def get_category_members(category, limit=500):
    members = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": limit,
        "format": "json"
    }
    while True:
        r = requests.get(BASE, params=params)
        data = r.json()
        members.extend([m["title"] for m in data["query"]["categorymembers"]])
        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
        else:
            break
    return members

def get_page_text(title):
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text"
    }
    r = requests.get(BASE, params=params)
    data = r.json()
    if "error" in data:
        return None
    html = data["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")

    paragraphs = soup.find_all("p")
    text = "\n".join(p.get_text() for p in paragraphs)
    return text

# Get all character pages, then keep only TV show pages
all_pages = get_category_members("Characters", limit=500)
pages = [p for p in all_pages if "(TV Series)" in p or "(TV Universe)" in p]

# Filter out category pages and user sandbox pages
pages = [
    p for p in pages
    if not p.startswith("Category:") and not p.startswith("User:")
]

print(f"Filtered down to {len(pages)} TV pages out of {len(all_pages)}")

articles = {}

for title in pages:
    text = get_page_text(title)
    if text:
        articles[title] = text
        print(f"Scraped: {title} ({len(text)} chars)")
    time.sleep(1)

# Save everything to disk so we don't have to re-scrape
with open("twd_articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(f"Saved {len(articles)} articles to twd_articles.json")