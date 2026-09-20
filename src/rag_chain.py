"""
The RAG pipeline itself:

    question -> query embedding -> Chroma similarity search -> context
    -> Gemini (via LangChain) -> answer + citation, OR an explicit
    "not found in the book" response.

Exposes `answer_question(question)` for use by app.py and evaluate.py.

Run directly for a quick command-line smoke test:
    python -m src.rag_chain "প্রশ্ন এখানে লিখুন"
"""

import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.chunk_and_index import E5Embeddings
from src.config import (
    CHROMA_DIR_A,
    EMBEDDING_MODEL_NAME,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    RETRIEVER_TOP_K,
)
from langchain_chroma import Chroma

SYSTEM_PROMPT = """তুমি একটি বই-ভিত্তিক প্রশ্নোত্তর সহকারী। তোমার একমাত্র জ্ঞানের উৎস হলো
নিচে দেওয়া "প্রসঙ্গ" (Context) অংশ, যা "{book_name}" বই থেকে নেওয়া হয়েছে।

কঠোর নিয়মাবলী:
1. শুধুমাত্র নিচের প্রসঙ্গে যা আছে তার ভিত্তিতে উত্তর দাও। বইয়ের বাইরের কোনো
   তথ্য, অনুমান বা সাধারণ জ্ঞান ব্যবহার করবে না।
2. প্রতিটি উত্তরের শেষে অবশ্যই সংশ্লিষ্ট অধ্যায়/অনুচ্ছেদের নাম উল্লেখ করে
   উৎস দেখাও, এই ফরম্যাটে: "(উৎস: <অধ্যায়ের নাম>)"।
3. যদি প্রসঙ্গে প্রশ্নের উত্তর না থাকে, তাহলে স্পষ্টভাবে বাংলায় বলো:
   "এই তথ্য বইটিতে পাওয়া যায়নি।" — এবং কোনো অনুমানভিত্তিক উত্তর দেবে না।
4. উত্তর সংক্ষিপ্ত, স্পষ্ট এবং প্রাসঙ্গিক রাখো।
"""

USER_PROMPT = """প্রসঙ্গ:
{context}

প্রশ্ন: {question}

উপরের নিয়ম মেনে বাংলায় উত্তর দাও।"""

_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
)


def _format_context(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        chapter = d.metadata.get("chapter_name", "অজানা অধ্যায়")
        section = d.metadata.get("section_name") or ""
        label = f"{chapter}" + (f" — {section}" if section else "")
        blocks.append(f"[খণ্ড {i} | {label}]\n{d.page_content}")
    return "\n\n".join(blocks)


def get_retriever(persist_dir=CHROMA_DIR_A, k: int = RETRIEVER_TOP_K):
    embeddings = E5Embeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": k})


def get_llm():
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add "
            "your Gemini API key."
        )
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.1
    )


_retriever = None
_llm = None
_chain = None


def _lazy_init():
    global _retriever, _llm, _chain
    if _retriever is None:
        _retriever = get_retriever()
    if _llm is None:
        _llm = get_llm()
    if _chain is None:
        _chain = _prompt | _llm | StrOutputParser()


def answer_question(question: str) -> dict:
    """Run the full pipeline for one question.

    Returns:
        {
          "answer": str,
          "sources": [{"chapter_name": ..., "section_name": ..., "source_url": ...}, ...],
          "found_in_book": bool,
        }
    """
    _lazy_init()

    docs = _retriever.invoke(question)
    context = _format_context(docs)

    answer = _chain.invoke(
        {"book_name": "বিশ্বের উপাদান", "context": context, "question": question}
    )

    sources = [
        {
            "chapter_name": d.metadata.get("chapter_name"),
            "section_name": d.metadata.get("section_name") or None,
            "source_url": d.metadata.get("source_url"),
        }
        for d in docs
    ]

    return {
        "answer": answer.strip(),
        "sources": sources,
        "found_in_book": "পাওয়া যায়নি" not in answer,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "এই বইটি কী নিয়ে লেখা?"
    result = answer_question(q)
    print("প্রশ্ন:", q)
    print("\nউত্তর:", result["answer"])
    print("\nসম্ভাব্য উৎস অংশসমূহ:")
    for s in result["sources"]:
        print(" -", s["chapter_name"], "/", s["section_name"], "->", s["source_url"])
