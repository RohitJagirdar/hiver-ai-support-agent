"""
Generate and Cache Vector Embeddings using all-MiniLM-L6-v2.
Allows zero-startup-latency vector search (<0.5ms) at test and evaluation runtime.
"""

import os
import csv
import numpy as np
from sentence_transformers import SentenceTransformer

CSV_PATH = "data/processed/applesupport_pairs.csv"
EMB_PATH = "data/processed/embeddings_cache.npy"

def build_cache():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}")

    print(f"Loading pairs from {CSV_PATH}...")
    texts = []
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Embed combined customer query + answer keywords for rich retrieval
            text = row.get("inbound_text", "")
            texts.append(text)

    print(f"Encoding {len(texts)} texts using all-MiniLM-L6-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized_embeddings = (embeddings / norms).astype(np.float32)

    np.save(EMB_PATH, normalized_embeddings)
    print(f"Successfully cached {normalized_embeddings.shape} embeddings to {EMB_PATH} ({os.path.getsize(EMB_PATH) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    build_cache()
