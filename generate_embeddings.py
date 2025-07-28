import os
import torch
from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Load chunked dataset
DATASET_PATH = "data/chunked/chunked_dataset"
dataset = load_from_disk(DATASET_PATH)

# Load sentence embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Batch embedding
BATCH_SIZE = 256
all_embeddings = []

print("Generating embeddings...")
for i in range(0, len(dataset), BATCH_SIZE):
    batch = dataset[i : i + BATCH_SIZE]["text"]
    with torch.no_grad():
        embeddings = embedding_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    all_embeddings.extend(embeddings)

# Convert to numpy array
all_embeddings = np.array(all_embeddings)

# Save embeddings
os.makedirs("data/embeddings", exist_ok=True)
np.save("data/embeddings/pubmed_chunk_embeddings.npy", all_embeddings)

# Save associated texts (for retrieval)
with open("data/embeddings/pubmed_texts.pkl", "wb") as f:
    pickle.dump(dataset["text"], f)

print(f"✅ Saved {len(all_embeddings)} embeddings to 'data/embeddings/'.")
