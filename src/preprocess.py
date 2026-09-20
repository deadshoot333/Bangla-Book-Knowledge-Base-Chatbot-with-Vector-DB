"""
Cleaning and preprocessing for raw ingested chapter/section text before
chunking. Kept as small, composable functions so the choices are easy to
document in the README and easy to unit-test.
"""

import json
import re
import unicodedata

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR

# Wikisource / OCR noise patterns worth stripping.
_EDIT_MARKER_RE = re.compile(r"\[সম্পাদনা\]|\[edit\]", re.IGNORECASE)
_FOOTNOTE_RE = re.compile(r"\[\d+\]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)


def clean_text(text: str) -> str:
    """Normalize and strip a single chapter/section's raw text."""
    # Unicode-normalize so visually-identical Bangla glyph sequences compare
    # and embed consistently (important for a script with many combining
    # vowel signs / conjuncts).
    text = unicodedata.normalize("NFC", text)

    text = _EDIT_MARKER_RE.sub("", text)
    text = _FOOTNOTE_RE.sub("", text)
    text = _PAGE_NUM_RE.sub("", text)

    # Normalize whitespace without collapsing paragraph breaks.
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(ln for ln in lines if ln)

    return text.strip()


def preprocess_all() -> int:
    """Read every raw JSON record, clean its text, and write it back out to
    data/processed/ with the same metadata."""
    count = 0
    for src_path in sorted(DATA_RAW_DIR.glob("*.json")):
        record = json.loads(src_path.read_text(encoding="utf-8"))
        cleaned = clean_text(record["text"])
        if not cleaned:
            continue
        record["text"] = cleaned
        out_path = DATA_PROCESSED_DIR / src_path.name
        out_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        count += 1
    return count


if __name__ == "__main__":
    n = preprocess_all()
    print(f"Cleaned {n} records -> {DATA_PROCESSED_DIR}")
