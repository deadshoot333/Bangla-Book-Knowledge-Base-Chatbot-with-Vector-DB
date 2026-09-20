# বিশ্বের উপাদান — Knowledge Base Chatbot (RAG)

A Retrieval-Augmented Generation chatbot that answers questions **only**
from the Bangla book *বিশ্বের উপাদান*, with mandatory chapter/section
citations and an explicit "not found in the book" fallback when the answer
isn't in the source text.

> **Before you run this**: `src/config.py` has a placeholder
> `WIKISOURCE_INDEX_URL`. Replace it with the real Bengali Wikisource
> index/table-of-contents page for your copy of the book (or set
> `LOCAL_PDF_PATH` to a local PDF copy) — see [Book Information](#book-information).

---

## 1. Book Information

- **Title:** বিশ্বের উপাদান
- **Bengali Wikisource link:** `https://bn.wikisource.org/wiki/বিশ্বের_উপাদান`
  *(placeholder — update `WIKISOURCE_INDEX_URL` in `src/config.py` or your
  `.env` with the exact URL your course provided; Wikisource page slugs are
  case- and spelling-sensitive)*
- **Description:** A Bangla-language book covering the constituent elements
  of the world/earth. The chatbot in this repo treats it as a closed corpus:
  every answer must be traceable to a specific chapter or section of this
  book, and the bot must decline to answer anything the book doesn't cover.

## 2. Setup & Running Instructions

### Required Python version
Python 3.10+

### Installation
```bash
git clone <this-repo-url>
cd bangla-rag-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure
```bash
cp .env.example .env
# then edit .env:
#   GOOGLE_API_KEY=<your Gemini API key>
#   WIKISOURCE_INDEX_URL=<the real book index page, if different>
```

### Build the knowledge base
```bash
python -m src.ingest          # crawl every chapter (or fall back to local PDF)
python -m src.preprocess      # clean the raw text
python -m src.chunk_and_index # chunk, embed, and write both Chroma collections
```

### Run the chatbot
```bash
python app.py
```
Gradio will print a local URL (and optionally a public share link) — open
it, type a question in Bangla, and read the answer plus its cited
chapter/section.

### Run the retrieval evaluation (chunking strategy comparison)
```bash
python -m src.evaluate
```

## 3. Technical Details

| Component | Choice |
|---|---|
| Embedding model | `intfloat/multilingual-e5-base` |
| Vector database | Chroma (local, persisted to disk) |
| Chunking (Strategy A, default) | size = 800 chars, overlap = 120 chars |
| Chunking (Strategy B, comparison) | size = 400 chars, overlap = 80 chars |
| Retriever | Chroma similarity search, top-k = 4 |
| LLM | Gemini (`gemini-2.0-flash` via `langchain-google-genai`) |
| UI | Gradio `ChatInterface` |

### Why `multilingual-e5-base`?
The book and every user question are in Bangla, so an English-only
embedding model (e.g. plain `all-MiniLM-L6-v2`) would place Bangla text in
an embedding space it was never trained to represent well, producing poor
similarity search. `multilingual-e5-base`:
- Is trained across 100+ languages, including Bengali, on a mix of
  parallel and monolingual data, so semantically similar Bangla sentences
  land close together in vector space even when they use different
  vocabulary.
- Uses the E5 "instruction prefix" convention (`"query: "` /
  `"passage: "`), which this project applies automatically in
  `src/chunk_and_index.py`'s `E5Embeddings` wrapper — this measurably
  improves retrieval quality over feeding raw text straight into the model.
- Is small enough (278M parameters) to embed a single book and run
  retrieval on commodity hardware without a GPU, unlike the larger
  `multilingual-e5-large`.

### Chunk size & overlap
- **Strategy A (800 / 120):** chosen as the default because Bangla
  sentences tend to run long, and 800 characters (roughly 120–160 Bangla
  words) is usually enough to hold a complete idea/paragraph without
  splitting it mid-thought — important for the "cite the passage that
  actually answers the question" requirement. 120-character overlap keeps
  the sentence bridging two chunks from being cut off in both.
- **Strategy B (400 / 80):** a denser configuration used to test whether
  smaller, more targeted chunks retrieve short factual answers more
  precisely (see the comparison in [§5](#5-comparison-of-two-approaches)).
- Both splitters use Bangla-aware separators, preferring to break on the
  Bangla sentence-ending "।" and on paragraph boundaries before falling
  back to a hard character cut (`src/chunk_and_index.py:BANGLA_SEPARATORS`).

### Preprocessing approach
`src/preprocess.py` applies, per chapter/section record:
1. Unicode NFC normalization (Bangla has combining vowel signs/conjuncts
   that can be represented by more than one equivalent byte sequence;
   normalizing avoids silently-duplicated near-identical chunks).
2. Stripping Wikisource chrome: `[সম্পাদনা]` / `[edit]` edit-section
   markers, footnote markers like `[12]`, bare page-number lines.
3. Whitespace normalization that collapses runs of spaces/blank lines
   without merging separate paragraphs into one.
4. (Ingestion-time, in `src/ingest.py`) removal of Wikisource
   navigation/reference/table markup before any text is even saved, so
   `preprocess.py` only has to deal with inline noise.

### Metadata preserved per chunk
Every chunk carries:
- `book_name` — always "বিশ্বের উপাদান"
- `chapter_name` — the chapter/page title as it appeared on Wikisource (or
  the detected heading, in the PDF fallback path)
- `section_name` — the nearest `h2/h3/h4` heading within the chapter, if any
- `source_url` — the exact Wikisource page URL the chunk came from (or a
  `local_pdf::path#chapter=N` reference in the PDF fallback path)

### Retriever configuration
`Chroma.as_retriever(search_kwargs={"k": 4})` — plain similarity search
over the E5 embeddings, top-4 chunks per query (`RETRIEVER_TOP_K` in
`src/config.py`).

### LLM & citation/refusal behavior
`src/rag_chain.py` builds a strict system prompt (in Bangla) that
instructs Gemini to:
1. Answer only from the retrieved context blocks.
2. Always close the answer with `(উৎস: <chapter name>)`.
3. Reply with exactly *"এই তথ্য বইটিতে পাওয়া যায়নি।"* ("This information
   was not found in the book.") when the retrieved context doesn't contain
   the answer, instead of guessing.

## 4. RAG Pipeline

```
Wikisource (book index page)
        │  crawl every chapter/sub-page link
        ▼
   src/ingest.py            → data/raw/*.json   (chapter_name, section_name, source_url, text)
        │
        ▼
   src/preprocess.py        → data/processed/*.json   (cleaned text, same metadata)
        │
        ▼
   src/chunk_and_index.py   → chunk (2 strategies) → embed (multilingual-e5) → Chroma (2 collections)
        │
        ▼
   src/rag_chain.py
        │  user question
        ▼
   query embedding (E5, "query: " prefix)
        ▼
   Chroma similarity search (top-k)
        ▼
   retrieved context blocks (with chapter/section labels)
        ▼
   Gemini LLM (strict "book-only + cite + refuse" prompt)
        ▼
   final answer + chapter/section citation  →  app.py (Gradio UI)
```

## 5. Comparison of Two Approaches

**Compared:** Chunking Strategy A (800/120) vs. Chunking Strategy B
(400/80) — see [§3](#technical-details). Both use the same embedding model
and the same 10 test questions, so chunking is the only variable.

**How the test was performed** (`src/evaluate.py`):
1. Each of the two strategies is built into its own Chroma collection
   (`chroma_db/strategy_a`, `chroma_db/strategy_b`) by `chunk_and_index.py`.
2. For every question in `test_questions.json`, the script retrieves the
   top-4 chunks from each collection.
3. A question counts as a **hit** for a strategy if the `chapter_name` of
   at least one retrieved chunk matches that question's labeled
   `expected_chapter`.
4. `hit_rate = hits / total_questions`, reported per strategy.

**Results:** *(fill in after running `python -m src.evaluate` against your
real, fully-ingested book — the table below is the format to report in)*

| Approach | Hit Rate |
|---|---|
| Chunking Strategy A (800/120) | __%  |
| Chunking Strategy B (400/80) | __%  |

**How to interpret:** larger chunks (A) tend to win on questions that need
surrounding context to disambiguate, while smaller chunks (B) tend to win
on narrow factual lookups because irrelevant surrounding sentences don't
dilute the chunk's embedding. Report which one actually retrieved the
correct chapter more often for *your* 10 questions, and give a one- or
two-sentence hypothesis for why, based on the questions that flipped
between strategies (printed as `MISS` lines by `evaluate.py`).

## 6. Repository Structure

```
bangla-rag-chatbot/
├── app.py                     # Gradio chat UI
├── requirements.txt
├── .env.example
├── test_questions.md          # human-readable test set
├── test_questions.json        # machine-readable test set (used by evaluate.py)
├── notebooks/
│   └── bangla_rag_chatbot.ipynb   # notebook equivalent of the full pipeline
├── src/
│   ├── config.py               # all paths/models/hyperparameters
│   ├── ingest.py                # Wikisource crawler (+ local PDF fallback)
│   ├── preprocess.py            # text cleaning
│   ├── chunk_and_index.py       # chunking + embeddings + Chroma indexing
│   ├── rag_chain.py             # retriever + Gemini + citation/refusal logic
│   └── evaluate.py              # hit-rate comparison of the two chunking strategies
└── data/
    ├── raw/                     # ingest.py output
    └── processed/                # preprocess.py output
```
