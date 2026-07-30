"""
build_local_index.py
====================
Builds a local semantic search index (local_search_index.pkl) from
cleaned_docs.csv using Sentence-Transformers embeddings.

Run: python build_local_index.py
(Run data_preprocessing.py first to generate cleaned_docs.csv)
"""

import os
import pickle
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "cleaned_docs.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "local_search_index.pkl")

# Sentence-Transformers model to use for embeddings
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        print("        Run generate_sample_data.py then data_preprocessing.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"[INFO] Loaded {len(df)} documents from {INPUT_FILE}")

    if df.empty:
        print("[ERROR] No documents to index.")
        return

    documents = []
    for _, row in df.iterrows():
        title   = str(row.get("title", ""))
        text    = str(row.get("text", ""))
        snippet = str(row.get("snippet", text[:200]))
        documents.append({
            "id":       str(row.get("id", "")),
            "title":    title,
            "text":     text,
            "category": str(row.get("category", "General")),
            "snippet":  snippet,
        })

    texts = [f"{doc['title']} - {doc['text']}" if doc['title'] else doc['text'] for doc in documents]

    # ── Load model and encode ──────────────────────────────────────────────────
    print(f"[INFO] Loading SentenceTransformer model: {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)

    print(f"[INFO] Encoding {len(texts)} documents … (this may take a moment)")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"[INFO] Encoding complete in {elapsed:.1f}s. Shape: {embeddings.shape}")

    # ── Save index ─────────────────────────────────────────────────────────────
    index_data = {
        "model_name": MODEL_NAME,
        "documents":  documents,
        "embeddings": embeddings,
    }
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(index_data, f)

    print(f"\n[OK] Local search index saved -> {OUTPUT_FILE}")
    print(f"     Documents indexed: {len(documents)}")
    print(f"     Embedding dims:    {embeddings.shape[1]}")
    print(f"\nYou can now start the server with: python server.py")


if __name__ == "__main__":
    main()
