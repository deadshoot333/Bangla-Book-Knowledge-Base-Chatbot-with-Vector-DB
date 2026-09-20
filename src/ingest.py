"""
Book ingestion.

Primary path: crawl the book's Bengali Wikisource index page, discover every
chapter/sub-page link, fetch each one, and pull out the chapter text plus
metadata (book name, chapter name, section name, source URL).

Fallback path: if no Wikisource pages can be found (e.g. offline, or the
book was only provided as a PDF), extract the same information from a local
PDF, splitting on heading-like lines to approximate chapter/section
boundaries.

Output: one JSON file per chapter in `data/raw/`, each shaped like:

{
  "book_name": "বিশ্বের উপাদান",
  "chapter_name": "প্রথম অধ্যায়: ভূমিকা",
  "section_name": null,
  "source_url": "https://bn.wikisource.org/wiki/বিশ্বের_উপাদান/প্রথম_অধ্যায়",
  "text": "... raw chapter text ..."
}

Run:
    python -m src.ingest
"""

import json
import re
import sys
import time
from urllib.parse import urljoin

import requests
# from bs4 import BeautifulSoup

from src.config import BOOK_NAME, DATA_RAW_DIR

HEADERS = {"User-Agent": "bangla-rag-chatbot/1.0 (educational course project)"}

# Wikisource boilerplate we don't want to treat as book content.
IGNORE_LINK_PATTERNS = re.compile(
    r"(বিশেষ:|আলোচনা:|সাহায্য:|টেমপ্লেট:|বিষয়শ্রেণী:|উইকিসংকলন:|ব্যবহারকারী:"
    r"|action=edit|Special:|Talk:|Help:|Template:|Category:|File:|চিত্র:)"
)


def _slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^\w\-\u0980-\u09FF]", "", text)
    return text[:max_len] or "chapter"


def get_chapter_links(index_url: str) -> list[dict]:
    """Fetch the index/table-of-contents page and return every chapter/
    sub-page link found in the main content area, in document order."""
    resp = requests.get(index_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    content = soup.select_one("#mw-content-text .mw-parser-output") or soup
    links = []
    seen = set()
    for a in content.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/wiki/") or IGNORE_LINK_PATTERNS.search(href):
            continue
        full_url = urljoin(index_url, href)
        if full_url in seen or full_url == index_url:
            continue
        seen.add(full_url)
        links.append({"chapter_name": a.get_text(strip=True), "url": full_url})
    return links


def clean_wikisource_html(soup: BeautifulSoup) -> BeautifulSoup:
    """Strip navigation, edit-section links, reference/footnote scaffolding
    and other non-book chrome from a Wikisource content block."""
    for selector in [
        ".mw-editsection", ".navbox", ".noprint", ".reflist",
        "sup.reference", "table", ".thumb", ".mw-references-wrap",
        "#toc", ".hatnote",
    ]:
        for tag in soup.select(selector):
            tag.decompose()
    return soup


def fetch_chapter(url: str, chapter_name: str) -> dict:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    content = soup.select_one("#mw-content-text .mw-parser-output")
    if content is None:
        return {"chapter_name": chapter_name, "source_url": url, "sections": []}

    content = clean_wikisource_html(content)

    # Split the chapter into sections using h2/h3/h4 headings, so each
    # section keeps its own name for citation purposes.
    sections = []
    current_heading = None
    current_paras = []

    def flush():
        if current_paras:
            sections.append({
                "section_name": current_heading,
                "text": "\n".join(p for p in current_paras if p.strip()),
            })

    for el in content.find_all(["h2", "h3", "h4", "p", "li"]):
        if el.name in ("h2", "h3", "h4"):
            flush()
            current_heading = el.get_text(strip=True) or current_heading
            current_paras = []
        else:
            txt = el.get_text(" ", strip=True)
            if txt:
                current_paras.append(txt)
    flush()

    if not sections:
        # No headings at all — treat the whole page as one section.
        full_text = content.get_text("\n", strip=True)
        sections = [{"section_name": None, "text": full_text}]

    return {"chapter_name": chapter_name, "source_url": url, "sections": sections}


def ingest_from_wikisource(index_url: str) -> int:
    print(f"Fetching index page: {index_url}")
    try:
        links = get_chapter_links(index_url)
    except requests.RequestException as exc:
        print(f"  Could not reach Wikisource index page ({exc}).")
        return 0

    if not links:
        print("  No chapter links found on the index page.")
        return 0

    print(f"  Found {len(links)} candidate chapter/sub-page links.")
    saved = 0
    for i, link in enumerate(links, 1):
        try:
            chapter = fetch_chapter(link["url"], link["chapter_name"])
        except requests.RequestException as exc:
            print(f"  [{i}/{len(links)}] skip {link['url']} ({exc})")
            continue

        for j, sec in enumerate(chapter["sections"]):
            if not sec["text"].strip():
                continue
            record = {
                "book_name": BOOK_NAME,
                "chapter_name": chapter["chapter_name"],
                "section_name": sec["section_name"],
                "source_url": chapter["source_url"],
                "text": sec["text"],
            }
            fname = f"{i:03d}_{j:02d}_{_slugify(chapter['chapter_name'])}.json"
            (DATA_RAW_DIR / fname).write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            saved += 1

        print(f"  [{i}/{len(links)}] saved chapter: {chapter['chapter_name']}")
        time.sleep(0.5)  # be polite to Wikisource

    return saved


def ingest_from_pdf(pdf_path: str) -> int:
    """Fallback ingestion from a local PDF copy of the book. Chapters are
    approximated by detecting short, heading-like lines (e.g. lines that
    start with 'অধ্যায়' or are short and end without sentence punctuation)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  pypdf is not installed; cannot fall back to PDF ingestion.")
        return 0

    from pathlib import Path
    if not Path(pdf_path).exists():
        print(f"  No PDF found at {pdf_path} either. Nothing to ingest.")
        return 0

    print(f"Falling back to local PDF: {pdf_path}")
    reader = PdfReader(pdf_path)

    heading_re = re.compile(r"^(অধ্যায়|পরিচ্ছেদ|ভাগ)[\s:.\u0964-]*")
    chapter_name = "ভূমিকা"
    buffer = []
    saved = 0
    chapter_idx = 0

    def flush(idx):
        nonlocal saved
        text = "\n".join(buffer).strip()
        if not text:
            return
        record = {
            "book_name": BOOK_NAME,
            "chapter_name": chapter_name,
            "section_name": None,
            "source_url": f"local_pdf::{pdf_path}#chapter={idx}",
            "text": text,
        }
        fname = f"{idx:03d}_00_{_slugify(chapter_name)}.json"
        (DATA_RAW_DIR / fname).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        saved += 1

    for page in reader.pages:
        page_text = page.extract_text() or ""
        for line in page_text.split("\n"):
            stripped = line.strip()
            if heading_re.match(stripped) or (
                0 < len(stripped) < 40 and not stripped.endswith(("।", ".", ","))
                and stripped == stripped.strip()
                and buffer  # avoid treating page 1's title as a false split
            ):
                flush(chapter_idx)
                chapter_idx += 1
                chapter_name = stripped
                buffer = []
            else:
                buffer.append(line)
    flush(chapter_idx)

    return saved


def main():
    # count = ingest_from_wikisource(WIKISOURCE_INDEX_URL)
    # if count == 0:
    count = ingest_from_pdf(r"O:\New folder\bangla-rag-chatbot\bangla-rag-chatbot\bangla_book.pdf")

    if count == 0:
        print(
            "\nNo content was ingested. Set WIKISOURCE_INDEX_URL in src/config.py "
            "(or a .env file) to the real Bengali Wikisource index page for "
            "'বিশ্বের উপাদান', or set LOCAL_PDF_PATH to a local copy of the book."
        )
        sys.exit(1)

    print(f"\nDone. Saved {count} chapter/section records to {DATA_RAW_DIR}")


if __name__ == "__main__":
    main()
