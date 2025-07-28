# chunk_documents.py

import os
from datasets import Dataset

# Configuration
RAW_DOCS_PATH = "data/parsed/pubmed_texts.txt"
CHUNKED_DATASET_PATH = "data/chunked/chunked_dataset"
CHUNK_SIZE = 300  # number of characters
CHUNK_OVERLAP = 50

def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks

def chunk_documents(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        raw_docs = raw_text.split("\n\n---\n\n")

    all_chunks = []
    for doc in raw_docs:
        for chunk in chunk_text(doc, CHUNK_SIZE, CHUNK_OVERLAP):
            all_chunks.append({"text": chunk})
    
    return Dataset.from_list(all_chunks)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(CHUNKED_DATASET_PATH), exist_ok=True)
    dataset = chunk_documents(RAW_DOCS_PATH)
    print(f"Generated {len(dataset)} text chunks.")
    dataset.save_to_disk(CHUNKED_DATASET_PATH)
