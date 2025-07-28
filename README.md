# 🔍 Retrieval-Augmented Generation (RAG) with Mistral + FAISS + PubMed

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using:

* 🧠 [**Mistral**](https://huggingface.co/mistralai/Mistral-7B-v0.1) as the **language model** (generator)
* 📦 [**FAISS**](https://github.com/facebookresearch/faiss) for **vector-based document retrieval**
* 📄 **PubMed** articles or any custom documents as external knowledge
* 🔗 Retrieval + Generation fused for factual, context-grounded responses

> **Author:** Aromal Joseph
> 🚀 AIOps Engineer | LLM + RAG Enthusiast
> 📫 [LinkedIn](https://www.linkedin.com/in/aromaljoseph)

---

## 📚 What is RAG?

**Retrieval-Augmented Generation (RAG)** enhances LLM responses by integrating **external document retrieval** into the generation process. Instead of relying solely on model parameters (parametric memory), RAG retrieves relevant text passages (non-parametric memory) and uses them as dynamic context during generation.

| Memory Type        | Description                          |
| ------------------ | ------------------------------------ |
| **Parametric**     | Model's internal knowledge (Mistral) |
| **Non-Parametric** | Retrieved documents (FAISS + PubMed) |

---

## 💡 Use Cases

* Biomedical Q\&A using PubMed abstracts
* Chatbots with custom knowledge bases
* Legal/financial assistants with grounded outputs
* Research support tools

---

## 🧱 Architecture Overview

```
+------------+     +---------------------+     +------------------------+
|  User Query| --> | Embed & Search FAISS| --> | Retrieve Top-k Chunks  |
+------------+     +---------------------+     +------------------------+
                                                            |
                                                            v
                                            +-------------------------------+
                                            | Mistral (Generator)           |
                                            | Prompt = Query + Retrieved    |
                                            +-------------------------------+
                                                            |
                                                            v
                                                +--------------------+
                                                | Final Answer Text  |
                                                +--------------------+
```

---

## 🧾 Dataset – PubMed Abstracts

You can use PubMed data (bio/medical literature) as your knowledge base.

### 🔽 To download sample PubMed abstracts:

```bash
wget https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed23n0001.xml.gz
gunzip pubmed23n0001.xml.gz
```

Then parse it with your custom XML parser or a library like `biopython`.

---

## ⚙️ Step-by-Step Pipeline

### 1. Preprocess Documents

* Chunk `.txt`/`.csv` or PubMed abstracts into smaller passages (100–300 tokens).
* Clean stopwords, special chars.

### 2. Embed Chunks

* Use `sentence-transformers` for dense embeddings (e.g. `all-MiniLM-L6-v2`).

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(text_chunks)
```

### 3. Create FAISS Index

```python
import faiss
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings))
```

Optional: Save the FAISS index for reuse:

```python
faiss.write_index(index, "faiss_index/index.faiss")
```

### 4. Query Processing

```python
query_embedding = model.encode([query])
_, top_k_indices = index.search(np.array(query_embedding), k=5)
retrieved_docs = [text_chunks[i] for i in top_k_indices[0]]
```

### 5. Prompt Construction

```python
prompt = "Context:\n" + "\n".join(retrieved_docs) + f"\n\nQuestion: {query}\nAnswer:"
```

### 6. Answer Generation using Mistral

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 🏗️ Project Structure

```
rag_mistral/
├── data/                   # Raw documents or PubMed XMLs
├── embeddings/             # Saved embedding files
├── faiss_index/            # Serialized FAISS index
├── retriever.py            # Handles retrieval from FAISS
├── generator.py            # Loads Mistral and generates output
├── preprocess.py           # Cleaning, chunking, tokenizing
├── main.py                 # Glue code: pipeline from query to answer
├── requirements.txt        # All Python dependencies
└── README.md               # You are here
```

---

## 🧪 Example Command

```bash
python main.py --query "What causes type 1 diabetes?"
```

---

## 📦 Installation

### ✅ Python Dependencies

```bash
pip install faiss-cpu
pip install transformers
pip install sentence-transformers
pip install beautifulsoup4
pip install lxml
pip install tqdm
```

> If using Mistral locally, install PyTorch (with GPU):
> [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## ✍️ Sample Code Snippet (main.py)

```python
from retriever import get_top_docs
from generator import generate_answer

query = "What is gene therapy?"
top_docs = get_top_docs(query)
answer = generate_answer(query, top_docs)

print("Answer:", answer)
```

---

## 🛠 Future Improvements

* [ ] Add BM25 retriever for hybrid search
* [ ] Optimize chunking strategy
* [ ] Add Streamlit/Gradio interface
* [ ] PubMed parser for full ingestion
* [ ] Caching and batching for speed

---

## 🧑‍💻 Author

**Aromal Joseph**
AIOps Engineer | LLM Developer | Research + Product
🔗 [LinkedIn](https://www.linkedin.com/in/aromaljoseph)

---

## 📜 License

MIT License — use, share, modify freely.

---

## 🙌 Acknowledgements

* [FAISS](https://github.com/facebookresearch/faiss) – Facebook AI
* [Hugging Face](https://huggingface.co) – Transformers & Mistral
* [NCBI PubMed](https://pubmed.ncbi.nlm.nih.gov/) – Biomedical open data
