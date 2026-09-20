"""
Compares the two chunking strategies (Strategy A vs Strategy B, see
src/config.py and src/chunk_and_index.py) using a simple hit-rate metric:

    For each labeled test question, retrieve the top-K chunks from a given
    strategy's Chroma collection. Count it as a "hit" if any retrieved
    chunk's `chapter_name` matches the question's expected chapter.

    hit_rate = hits / total_questions

This requires `test_questions.json` (see the repo root — generated from
test_questions.md) with an `expected_chapter` field per question, since we
are checking whether the *correct source* was retrieved, not grading the
final generated answer.

Run:
    python -m src.evaluate
"""

import json
from pathlib import Path

from src.chunk_and_index import E5Embeddings
from src.config import CHROMA_DIR_A, CHROMA_DIR_B, EMBEDDING_MODEL_NAME, RETRIEVER_TOP_K, ROOT_DIR
from langchain_chroma import Chroma

TEST_QUESTIONS_PATH = ROOT_DIR / "test_questions.json"


def load_test_questions() -> list[dict]:
    if not TEST_QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{TEST_QUESTIONS_PATH} not found. Create it with entries like:\n"
            '[{"question": "...", "expected_chapter": "..."}, ...]\n'
            "(see test_questions.md for the human-readable version)."
        )
    return json.loads(TEST_QUESTIONS_PATH.read_text(encoding="utf-8"))


def hit_rate_for_strategy(persist_dir: Path, questions: list[dict], embeddings, k: int) -> float:
    vectordb = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    hits = 0
    for item in questions:
        docs = retriever.invoke(item["question"])
        retrieved_chapters = {d.metadata.get("chapter_name", "") for d in docs}
        if item["expected_chapter"] in retrieved_chapters:
            hits += 1
        else:
            print(f"  MISS: {item['question']!r} "
                  f"(expected {item['expected_chapter']!r}, got {retrieved_chapters})")

    return hits / len(questions) if questions else 0.0


def main():
    questions = load_test_questions()
    embeddings = E5Embeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"Evaluating {len(questions)} test questions, k={RETRIEVER_TOP_K}\n")

    print("Strategy A (larger chunks):")
    rate_a = hit_rate_for_strategy(CHROMA_DIR_A, questions, embeddings, RETRIEVER_TOP_K)

    print("\nStrategy B (smaller chunks):")
    rate_b = hit_rate_for_strategy(CHROMA_DIR_B, questions, embeddings, RETRIEVER_TOP_K)

    print("\n| Approach              | Hit Rate |")
    print("|------------------------|----------|")
    print(f"| Chunking Strategy A    | {rate_a * 100:.0f}%      |")
    print(f"| Chunking Strategy B    | {rate_b * 100:.0f}%      |")


if __name__ == "__main__":
    main()
