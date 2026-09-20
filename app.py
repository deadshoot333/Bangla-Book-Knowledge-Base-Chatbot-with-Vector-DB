"""
Gradio chat interface for the বিশ্বের উপাদান Knowledge Base Chatbot.

Run:
    python app.py

Requires the vector index to already be built:
    python -m src.ingest
    python -m src.preprocess
    python -m src.chunk_and_index
"""

import gradio as gr

from src.rag_chain import answer_question


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return "_কোনো উৎস পাওয়া যায়নি।_"
    lines = []
    for s in sources:
        label = s["chapter_name"] or "অজানা অধ্যায়"
        if s.get("section_name"):
            label += f" — {s['section_name']}"
        lines.append(f"- **{label}**  \n  {s['source_url']}")
    return "\n".join(lines)


def chat_fn(message: str, history):
    result = answer_question(message)
    sources_md = format_sources(result["sources"])
    reply = f"{result['answer']}\n\n---\n**সংশ্লিষ্ট উৎস (Retrieved sources):**\n{sources_md}"
    return reply


demo = gr.ChatInterface(
    fn=chat_fn,
    title="বিশ্বের উপাদান — Knowledge Base Chatbot",
    description=(
        "এই চ্যাটবট শুধুমাত্র \"বিশ্বের উপাদান\" বই থেকে প্রশ্নের উত্তর দেয়, "
        "এবং প্রতিটি উত্তরের সাথে সংশ্লিষ্ট অধ্যায়/অনুচ্ছেদের উৎস উল্লেখ করে। "
        "বইয়ে না থাকা তথ্যের জন্য এটি স্পষ্টভাবে জানিয়ে দেয় যে উত্তর পাওয়া যায়নি।"
    ),
    examples=[
        "এই বইটি কী নিয়ে লেখা?",
        "প্রথম অধ্যায়ে কী আলোচনা করা হয়েছে?",
    ],
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()
