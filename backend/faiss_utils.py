import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# --- Embedding model ---
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# --- FAISS index and stored summaries files ---
FAISS_FILE = "faiss_index.pkl"
TEXTS_FILE = "stored_texts.pkl"

# --- Load or initialize FAISS index ---
if os.path.exists(FAISS_FILE):
    with open(FAISS_FILE, "rb") as f:
        index = pickle.load(f)
else:
    index = faiss.IndexFlatL2(EMBEDDING_MODEL.get_sentence_embedding_dimension())

# --- Load or initialize stored summaries ---
if os.path.exists(TEXTS_FILE):
    with open(TEXTS_FILE, "rb") as f:
        stored_texts = pickle.load(f)
else:
    stored_texts = []

# --- Store summary in FAISS ---
def store_in_faiss(text: str):
    text = text.strip()
    if not text:
        return
    # Avoid duplicates
    if text in stored_texts:
        return
    # Embed and add to FAISS
    embedding = EMBEDDING_MODEL.encode([text])
    index.add(np.array(embedding).astype(np.float32))
    stored_texts.append(text)

    # Persist to disk
    with open(FAISS_FILE, "wb") as f:
        pickle.dump(index, f)
    with open(TEXTS_FILE, "wb") as f:
        pickle.dump(stored_texts, f)

# --- Semantic Search ---
def semantic_search(query: str, top_k: int = 3):
    query = query.strip()
    if not query or not stored_texts:
        return []

    query_vec = EMBEDDING_MODEL.encode([query])
    D, I = index.search(np.array(query_vec).astype(np.float32), top_k)

    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx != -1:
            results.append({
                "summary": stored_texts[idx],
                "score": float(dist)
            })
    return results
