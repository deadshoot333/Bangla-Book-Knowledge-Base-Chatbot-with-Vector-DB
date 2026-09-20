"""
Chunking, embedding, and vector-store indexing.

Builds TWO Chroma collections from the same cleaned source data, using two
different chunking strategies, so evaluate.py can compare their retrieval
hit-rates:

  Strategy A: larger chunks (CHUNK_SIZE_A / CHUNK_OVERLAP_A) — fewer, more
              context-rich chunks per chapter.
  Strategy B: smaller chunks (CHUNK_SIZE_B / CHUNK_OVERLAP_B) — more, more
              narrowly-focused chunks, which can improve precision on
              short factual questions at the cost of context.

Both use the same multilingual-e5 embedding model. E5 models expect a
"passage: " prefix on indexed text and a "query: " prefix on the query text
at retrieval time — see EMBEDDING NOTE below.

Run:
    python -m src.chunk_and_index
"""

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    CHROMA_DIR_A,
    CHROMA_DIR_B,
    CHUNK_OVERLAP_A,
    CHUNK_OVERLAP_B,
    CHUNK_SIZE_A,
    CHUNK_SIZE_B,
    DATA_PROCESSED_DIR,
    EMBEDDING_MODEL_NAME,
)

# Bangla-aware separators: prefer splitting on the Bangla sentence-ending
# danda ("।"), then paragraph/line breaks, then spaces, before falling back
# to a hard character cut. This keeps chunks from breaking mid-sentence
# whenever the chunk_size budget allows it.
BANGLA_SEPARATORS = ["\n\n", "\n", "।", "; ", " ", ""]


# ---------------------------------------------------------------------------
# EMBEDDING NOTE
# ---------------------------------------------------------------------------
# intfloat/multilingual-e5-base was trained with instruction-style prefixes:
#   - documents/passages to be indexed should be prefixed "passage: "
#   - user queries at search time should be prefixed "query: "
# Skipping these prefixes still "works" but noticeably hurts retrieval
# quality, so this wrapper applies them automatically.
class E5Embeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")


def load_processed_documents() -> list[Document]:
    docs = []
    for path in sorted(DATA_PROCESSED_DIR.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        docs.append(
            Document(
                page_content=record["text"],
                metadata={
                    "book_name": record["book_name"],
                    "chapter_name": record["chapter_name"],
                    "section_name": record.get("section_name") or "",
                    "source_url": record["source_url"],
                },
            )
        )
    return docs


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=BANGLA_SEPARATORS,
    )
    return splitter.split_documents(docs)


def build_index(docs: list[Document], persist_dir, embeddings) -> Chroma:
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )


def main():
    source_docs = load_processed_documents()
    if not source_docs:
        print(
            "No processed documents found. Run `python -m src.ingest` and "
            "`python -m src.preprocess` first."
        )
        return

    print(f"Loaded {len(source_docs)} cleaned chapter/section documents.")
    embeddings = E5Embeddings(model_name=EMBEDDING_MODEL_NAME)

    chunks_a = split_documents(source_docs, CHUNK_SIZE_A, CHUNK_OVERLAP_A)
    print(f"Strategy A: {len(chunks_a)} chunks "
          f"(chunk_size={CHUNK_SIZE_A}, overlap={CHUNK_OVERLAP_A})")
    build_index(chunks_a, CHROMA_DIR_A, embeddings)
    print(f"  -> indexed into {CHROMA_DIR_A}")

    chunks_b = split_documents(source_docs, CHUNK_SIZE_B, CHUNK_OVERLAP_B)
    print(f"Strategy B: {len(chunks_b)} chunks "
          f"(chunk_size={CHUNK_SIZE_B}, overlap={CHUNK_OVERLAP_B})")
    build_index(chunks_b, CHROMA_DIR_B, embeddings)
    print(f"  -> indexed into {CHROMA_DIR_B}")

    print("\nIndexing complete. The chatbot (app.py) uses Strategy A by "
          "default — see src/rag_chain.py to switch.")


if __name__ == "__main__":
    main()
