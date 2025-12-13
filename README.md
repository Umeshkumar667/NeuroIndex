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

## 📦 Installation

```bash
pip install neuroindex
