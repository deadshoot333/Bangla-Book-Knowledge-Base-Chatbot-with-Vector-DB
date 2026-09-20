# Test Questions

> **Template — fill in with your book's real content.**
> These 10 questions are placeholders showing the required shape (question,
> expected answer, expected source chapter, and one deliberately
> out-of-book question to test the refusal behavior). After you run
> `src/ingest.py` against the real Wikisource pages for বিশ্বের উপাদান,
> replace `expected_answer` / `expected_chapter` below — and in the
> matching `test_questions.json` used by `src/evaluate.py` — with the
> actual chapter names and correct answers from your ingested data.

| # | Question (বাংলা) | Expected Answer (summary) | Expected Source Chapter |
|---|---|---|---|
| 1 | বইটির নাম কী? | বিশ্বের উপাদান | (title/cover page) |
| 2 | প্রথম অধ্যায়ে প্রধানত কী নিয়ে আলোচনা করা হয়েছে? | *[replace after ingestion]* | *[chapter 1 name]* |
| 3 | দ্বিতীয় অধ্যায়ে কোন বিষয়টি ব্যাখ্যা করা হয়েছে? | *[replace after ingestion]* | *[chapter 2 name]* |
| 4 | বইয়ে উল্লেখিত প্রধান বিষয়বস্তুগুলো কী কী? | *[replace after ingestion]* | *[relevant chapter(s)]* |
| 5 | *[a specific factual question about a concept covered in the book]* | *[replace]* | *[chapter name]* |
| 6 | *[a specific factual question about a different chapter]* | *[replace]* | *[chapter name]* |
| 7 | *[a "definition" style question — e.g. "X বলতে কী বোঝায়?"]* | *[replace]* | *[chapter name]* |
| 8 | *[a comparison/relationship question across two ideas in the book]* | *[replace]* | *[chapter name]* |
| 9 | *[a question about the last chapter / conclusion]* | *[replace]* | *[chapter name]* |
| 10 | বইয়ে কি বাংলাদেশের অর্থনীতি নিয়ে বিস্তারিত আলোচনা আছে? *(deliberately out-of-scope)* | "এই তথ্য বইটিতে পাওয়া যায়নি।" | — (tests the refusal path) |

See `test_questions.json` for the machine-readable version consumed by
`src/evaluate.py` (only `question` and `expected_chapter` are needed there).
