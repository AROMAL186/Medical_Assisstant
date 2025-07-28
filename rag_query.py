import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Load FAISS index and embedded texts
index = faiss.read_index("faiss_index/index.faiss")
with open("data/embeddings/pubmed_texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Load sentence embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load local Mistral .gguf model
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    use_mlock=True,
    verbose=False
)

def rag_query(query, top_k=5):
    print(f"\n💬 You: {query}")

    # Convert query to embedding and search
    query_embedding = embedder.encode([query]).astype("float32")
    _, indices = index.search(query_embedding, top_k)

    # Retrieve top relevant chunks
    retrieved_chunks = [texts[int(i)] for i in indices[0]]
    context = "\n".join(retrieved_chunks)

    # Friendly prompt formatting
    prompt = f"""You are a friendly and helpful assistant.

Using the following context, answer the user's question clearly and naturally.

Context:
{context}

User's Question:
{query}

Your Response:"""

    # Generate response from model
    response = llm(prompt, max_tokens=256, stop=["</s>", "User:", "Question:"])
    print("\n🤖 Bot:", response["choices"][0]["text"].strip())

# Run as chatbot
if __name__ == "__main__":
    print("👋 Hello! I'm your AI assistant. Ask me anything or type 'exit' to quit.")
    while True:
        user_query = input("\n🧑 You: ")
        if user_query.strip().lower() == "exit":
            print("👋 Goodbye! Have a great day.")
            break
        rag_query(user_query)
