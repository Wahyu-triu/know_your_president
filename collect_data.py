"""
Scraper: Pidato Presiden - Sekretariat Negara RI
Source: https://www.setneg.go.id/listcontent/listberita/pidato_presiden

Scrapes pages 1-5 (10 entries per page) and outputs:
  page | date | title | link | content

Usage:
    pip install requests beautifulsoup4
    python setneg_pidato_scraper.py

Output: pidato_presiden.csv
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
import re

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL  = "https://www.setneg.go.id"
LIST_URL  = BASE_URL + "/listcontent/listberita/pidato_presiden/{offset}"

# Page 1 = offset 0, Page 2 = offset 10, ..., Page 5 = offset 40
PAGES = {1: 0, 2: 10, 3: 20, 4: 30, 5: 40}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
    "Referer": BASE_URL + "/",
}

DAYS_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

DELAY_LIST    = 1.5   # seconds between list pages
DELAY_CONTENT = 1.0   # seconds between individual speech pages

OUTPUT_CSV = "pidato_presiden.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def extract_date(text: str) -> str:
    """Return the first line that starts with an Indonesian day name."""
    for line in text.splitlines():
        line = line.strip()
        if any(line.startswith(d) for d in DAYS_ID):
            return line
    return ""


def scrape_list_page(page_num: int, offset: int) -> list[dict]:
    """Scrape one listing page and return a list of entry dicts."""
    url = LIST_URL.format(offset=offset)
    print(f"[Page {page_num}] Fetching list: {url}")
    soup = get_soup(url)

    entries = []

    # Each speech entry is an <h4> containing an <a> link
    for h4 in soup.find_all("h4"):
        a_tag = h4.find("a", href=True)
        if not a_tag:
            continue

        title = a_tag.get_text(strip=True)
        href  = a_tag["href"]
        link  = href if href.startswith("http") else BASE_URL + href

        # Date lives in the next sibling element after the <h4>
        date    = ""
        snippet = ""
        sibling = h4.find_next_sibling()
        if sibling:
            raw = sibling.get_text("\n", strip=True)
            date    = extract_date(raw)
            # Snippet = everything after the date line (location + opening words)
            snippet = raw.replace(date, "").strip()
            snippet = re.sub(r"\s+", " ", snippet)[:300]

        entries.append({
            "page":    page_num,
            "date":    date,
            "title":   title,
            "link":    link,
            "snippet": snippet,
            "content": "",   # filled in next step
        })

    print(f"  → Found {len(entries)} entries")
    return entries


def scrape_content(entry: dict) -> str:
    """Fetch the full speech page and extract the body text."""
    url = entry["link"]
    print(f"  Fetching content: {url}")
    try:
        soup = get_soup(url)

        # The speech body is usually inside a <div class="content-detail">
        # or the main content area — try multiple selectors
        body = (
            soup.find("div", class_="content-detail")
            or soup.find("div", id="content-detail")
            or soup.find("div", class_="detail-content")
            or soup.find("article")
        )

        if body:
            # Remove nav / header / footer clutter inside the div
            for tag in body.find_all(["nav", "script", "style", "footer"]):
                tag.decompose()
            text = body.get_text("\n", strip=True)
        else:
            # Fallback: grab all <p> tags
            paragraphs = soup.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paragraphs)

        # Collapse excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    except Exception as exc:
        print(f"    !! Error fetching content: {exc}")
        return f"[Error: {exc}]"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_entries = []

    # Step 1 – collect all list entries across pages 1–5
    for page_num, offset in PAGES.items():
        entries = scrape_list_page(page_num, offset)
        all_entries.extend(entries)
        time.sleep(DELAY_LIST)

    print(f"\nTotal entries collected: {len(all_entries)}")

    # Step 2 – fetch full content for each entry
    print("\nFetching speech content …")
    for i, entry in enumerate(all_entries, 1):
        print(f"[{i}/{len(all_entries)}]", end=" ")
        entry["content"] = scrape_content(entry)
        time.sleep(DELAY_CONTENT)

    # Step 3 – write CSV
    fields = ["page", "date", "title", "link", "snippet", "content"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_entries)

    print(f"\n✓ Saved {len(all_entries)} rows to '{OUTPUT_CSV}'")

    # Step 4 – print a preview table
    print("\n" + "=" * 80)
    print(f"{'Page':<6} {'Date':<30} {'Title'[:50]:<52}")
    print("=" * 80)
    for e in all_entries:
        page  = str(e["page"])
        date  = e["date"][:30] if e["date"] else "(no date)"
        title = e["title"][:50] if e["title"] else "(no title)"
        print(f"{page:<6} {date:<30} {title:<52}")

    print("=" * 80)
    print(f"\nDone. Full content saved in '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()