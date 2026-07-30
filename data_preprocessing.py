"""
data_preprocessing.py
======================
Loads employee_docs.csv, cleans the text, standardizes column names,
and outputs cleaned_docs.csv ready for ML training and index building.

Run: python data_preprocessing.py
"""

import os
import re
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "employee_docs.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "cleaned_docs.csv")

VALID_CATEGORIES = {"HR", "IT", "Finance"}


def clean_text(text: str) -> str:
    """Basic text cleaning: strip whitespace, collapse spaces, remove control chars."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"[\r\n\t]+", " ", text)          # collapse newlines/tabs
    text = re.sub(r" {2,}", " ", text)               # collapse multiple spaces
    text = re.sub(r"[^\x20-\x7E]", "", text)        # remove non-ASCII
    return text


def standardize_category(cat: str) -> str:
    """Map category variations to canonical values."""
    mapping = {
        "human resources": "HR",
        "h.r.": "HR",
        "information technology": "IT",
        "info tech": "IT",
        "finance": "Finance",
        "financial": "Finance",
        "accounting": "Finance",
    }
    cat_clean = cat.strip()
    return mapping.get(cat_clean.lower(), cat_clean)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        print("        Run generate_sample_data.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"[INFO] Loaded {len(df)} rows from {INPUT_FILE}")

    # ── Standardize column names ──────────────────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {}
    if "content" in df.columns and "text" not in df.columns:
        rename_map["content"] = "text"
    if "doc_id" in df.columns and "id" not in df.columns:
        rename_map["doc_id"] = "id"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    required_cols = {"id", "title", "category", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        return

    # ── Clean text fields ──────────────────────────────────────────────────────
    df["text"]     = df["text"].apply(clean_text)
    df["title"]    = df["title"].apply(clean_text)
    df["category"] = df["category"].apply(lambda x: standardize_category(str(x)))

    # ── Add snippet column (first 200 chars of text) ──────────────────────────
    if "snippet" not in df.columns:
        df["snippet"] = df["text"].str[:200]

    # ── Filter invalid rows ───────────────────────────────────────────────────
    original_len = len(df)
    df = df[df["text"].str.len() > 10]
    df = df[df["category"].isin(VALID_CATEGORIES)]
    dropped = original_len - len(df)
    if dropped:
        print(f"[WARN] Dropped {dropped} invalid rows.")

    # ── Deduplicate by id ──────────────────────────────────────────────────────
    df.drop_duplicates(subset=["id"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Output ─────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"[OK] Saved {len(df)} cleaned documents -> {OUTPUT_FILE}")
    print(f"\nCategory distribution:")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
