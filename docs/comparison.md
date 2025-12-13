# NeuroIndex vs FAISS vs Vector Databases

## FAISS
- Vector similarity only
- No persistence (by default)
- No semantic relationships
- Excellent low-level library

## Typical Vector Databases
- Cloud-oriented
- Focus on ANN search
- Limited relationship modeling
- Often expensive / external dependency

## NeuroIndex
- Local-first, offline-capable
- Combines cache + vectors + graph
- Persistent SQLite storage
- Relationship-aware retrieval
- Model-agnostic embeddings

### Summary

| Feature              | FAISS | Vector DB | NeuroIndex |
|---------------------|-------|-----------|-------------|
| Vector Search       | ✅    | ✅        | ✅         |
| Semantic Graph      | ❌    | ❌        | ✅         |
| Persistence         | ❌    | ✅        | ✅         |
| Offline-first       | ⚠️    | ❌        | ✅         |
| Model agnostic      | ✅    | ⚠️        | ✅         |
