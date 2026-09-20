"""
Central configuration for the Bangla RAG Knowledge Base Chatbot.

Edit the values in this file (or override them via a `.env` file — see
`.env.example`) before running the ingestion / indexing / app scripts.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CHROMA_DIR_A = ROOT_DIR / "chroma_db" / "strategy_a"   # fixed-size chunking
CHROMA_DIR_B = ROOT_DIR / "chroma_db" / "strategy_b"   # paragraph/semantic chunking

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, CHROMA_DIR_A, CHROMA_DIR_B):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Book / source configuration
# ---------------------------------------------------------------------------
BOOK_NAME = "বিশ্বের উপাদান"

# The Bengali Wikisource "index"/table-of-contents page for the book.
# Replace this with the real URL for your copy of the book before running
# ingest.py. It should be a page that links out to every chapter/sub-page.
# Example shape: "https://bn.wikisource.org/wiki/বিশ্বের_উপাদান"
WIKISOURCE_INDEX_URL = os.getenv(
    "WIKISOURCE_INDEX_URL",
    "https://bn.wikisource.org/wiki/বিশ্বের_উপাদান",
)

# Optional local fallback: if you were given a clean PDF of the book instead
# of (or in addition to) a Wikisource copy, point this at it and ingest.py
# will use it when the Wikisource crawl finds no pages.
LOCAL_PDF_PATH = os.getenv("LOCAL_PDF_PATH", str(ROOT_DIR / "বিশ্বের_উপাদান.pdf"))

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# multilingual-e5-base supports Bengali (and 100 other languages) and needs
# "query: " / "passage: " prefixes on the input text (E5 convention).
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# ---------------------------------------------------------------------------
# Chunking strategies (compared against each other in evaluate.py)
# ---------------------------------------------------------------------------
# Strategy A: fixed-size character chunking with overlap.
CHUNK_SIZE_A = 800
CHUNK_OVERLAP_A = 120

# Strategy B: smaller fixed-size chunks, more overlap (denser retrieval
# granularity) — used as the second strategy for the hit-rate comparison.
CHUNK_SIZE_B = 400
CHUNK_OVERLAP_B = 80

# ---------------------------------------------------------------------------
# LLM (Gemini)
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVER_TOP_K = 4
