# 🧠 NeuroIndex

**NeuroIndex** is a hybrid **vector + semantic graph memory system** for embeddings.

It combines:
- ⚡ RAM-based LRU cache (working memory)
- 🔎 FAISS vector search (similarity)
- 🕸️ Semantic graph traversal (associative recall)
- 💾 Persistent SQLite storage (long-term memory)

Designed for **AI memory**, **RAG systems**, **chatbots**, and **semantic search pipelines**.

---

## ✨ Why NeuroIndex?

Most vector databases only answer:
> “What is similar?”

NeuroIndex also answers:
> “What is related?”

This makes it ideal for:
- Conversational AI memory
- Document understanding
- Knowledge graphs + embeddings
- Long-running agents
- Offline / local-first AI systems

---


## 🚀 Quick Start
from neuroindex import NeuroIndex
import numpy as np

#### Create index
ni = NeuroIndex(
    path="./memory",   # persistent storage folder
    dim=112            # embedding dimension
)

#### Dummy embedding function
def embed(text: str):
    return np.random.rand(112).astype("float32")

#### Add documents
ni.add_document("Neural networks use embeddings", embed("doc1"))
ni.add_document("FAISS enables fast vector search", embed("doc2"))
ni.add_document("Graphs capture semantic relationships", embed("doc3"))

#### Search by vector
query_vec = embed("search")
results = ni.search(query_vector=query_vec, k=3)

for r in results:
    print(r.source, r.similarity, r.text)


#### Search directly with text (recommended)

results = ni.search_text(
    "What is semantic memory?",
    embed_fn=embed,
    k=3
)

for r in results:
    print(r.text)

ni.get_stats()


## 📦 Installation
```bash
pip install neuroindex

