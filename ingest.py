import os
import glob

def load_pubmed_data(data_dir="data/pubmed_data"):
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    documents = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(content)
    return documents

if __name__ == "__main__":
    docs = load_pubmed_data()
    print(f"Loaded {len(docs)} documents.")

