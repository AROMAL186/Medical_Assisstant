import faiss
import numpy as np
import pickle
import os

# Paths
EMBEDDINGS_PATH = "data/embeddings/pubmed_chunk_embeddings.npy"
TEXTS_PATH = "data/embeddings/pubmed_texts.pkl"
INDEX_DIR = "faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")

# Load embeddings and texts
print("🔹 Loading embeddings and texts...")
embeddings = np.load(EMBEDDINGS_PATH).astype('float32')

with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)

# Sanity check
assert len(embeddings) == len(texts), "Embeddings and texts count mismatch!"

# Create FAISS index
print("🔹 Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the index
os.makedirs(INDEX_DIR, exist_ok=True)
faiss.write_index(index, INDEX_PATH)

print(f"✅ FAISS index created and saved at: {INDEX_PATH}")
print(f"📄 Total documents indexed: {len(texts)}")
