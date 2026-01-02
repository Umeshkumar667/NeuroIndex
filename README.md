# 🧠 NeuroIndex

[![Tests](https://github.com/Umeshkumar667/NeuroIndex/actions/workflows/tests.yml/badge.svg)](https://github.com/Umeshkumar667/NeuroIndex/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/neuroindex.svg)](https://badge.fury.io/py/neuroindex)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**NeuroIndex** is a **production-ready hybrid vector + semantic graph memory system** for AI applications.

It combines:
- ⚡ **FAISS vector search** for fast similarity matching
- 🕸️ **Semantic graph traversal** for relationship-aware retrieval
- 💾 **LRU cache** for frequently accessed items
- 🔒 **SQLite persistence** for durability
- 📊 **Built-in metrics** for observability

Designed for **RAG systems**, **AI agents**, **chatbots**, and **semantic search pipelines**.

## ✨ Why NeuroIndex?

Most vector databases only answer:
> "What is similar?"

NeuroIndex also answers:
> "What is related?"

This makes it ideal for:
- 🤖 Conversational AI memory
- 📚 Document understanding & retrieval
- 🧠 Knowledge graphs + embeddings
- 🔄 Long-running AI agents
- 📴 Offline / local-first AI systems

## 🏭 Production Readiness

### ✅ Guarantees
- **Deterministic results** - Same query, same results
- **Persistent storage** - SQLite + FAISS index saved to disk
- **Thread-safe** - Concurrent read operations supported
- **Graceful failures** - Clear exceptions, no silent errors
- **No data loss** - All writes are durable

### ⚠️ Current Limitations
- Single-writer (concurrent writes require external locking)
- CPU-only (via faiss-cpu)
- Single-node (not distributed)

### 📊 Benchmarks

| Documents | Insert Time | Search (k=10) | Memory |
|-----------|-------------|---------------|--------|
| 1,000     | 0.8s        | 2ms           | 45MB   |
| 10,000    | 8.5s        | 8ms           | 380MB  |
| 100,000   | 92s         | 25ms          | 3.2GB  |

*Benchmarked on Apple M1, 16GB RAM, dim=384*

### 🔄 API Stability

We follow [Semantic Versioning](https://semver.org/). These APIs are stable:
- `NeuroIndex.add_document()`
- `NeuroIndex.search()`
- `NeuroIndex.search_text()`
- `NeuroIndex.get_document()`
- `NeuroIndex.delete_document()`
- `NeuroIndex.update_document()`
- `NeuroIndex.get_stats()`
- `NeuroIndex.close()`

## 🧩 System Architecture

```
TEXT / DATA
    ↓
[Embedding Model]  ← OpenAI / HuggingFace / Cohere / Local
    ↓
[NeuroIndex]       ← Memory + Retrieval
    ↓
[LLM / App / Agent / API]
```

## 🔌 Integration Patterns

NeuroIndex can be used as:
- Memory layer for RAG pipelines
- Long-term memory for chatbots
- Knowledge base for document search
- Experience memory for AI agents
- Offline semantic retrieval system

It does **not** depend on any specific model, framework, or cloud provider.

## 📦 Installation

```bash
pip install neuroindex
```

For development:
```bash
pip install neuroindex[dev]
```

## 🚀 Quick Start

```python
from neuroindex import NeuroIndex
import numpy as np

# Create index (use context manager for automatic cleanup)
with NeuroIndex(path="./memory", dim=384) as ni:
    
    # Your embedding function (use any model)
    def embed(text: str) -> np.ndarray:
        # Replace with your actual embedding model
        return np.random.rand(384).astype("float32")
    
    # Add documents
    ni.add_document(
        text="Neural networks use embeddings for representation",
        vector=embed("neural networks"),
        metadata={"source": "textbook", "chapter": 1}
    )
    
    ni.add_document(
        text="FAISS enables fast vector similarity search",
        vector=embed("vector search"),
        metadata={"source": "documentation"}
    )
    
    # Search by vector
    query_vec = embed("how do neural networks work?")
    results = ni.search(query_vector=query_vec, k=5)
    
    for r in results:
        print(f"[{r.source}] {r.similarity:.3f}: {r.text}")
    
    # Or search with text directly
    results = ni.search_text(
        "What is semantic memory?",
        embed_fn=embed,
        k=3
    )
    
    # Get statistics
    print(ni.get_stats())
    # {'version': '0.2.0', 'total_documents': 2, 'faiss_vectors': 2, ...}
    
    # Get performance metrics
    print(ni.get_metrics())
    # {'uptime_seconds': 1.23, 'cache': {'hits': 5, 'misses': 2, ...}, ...}
```

## 📖 API Reference

### NeuroIndex

```python
class NeuroIndex:
    def __init__(
        self,
        path: str = './neuroindex_data',  # Storage directory
        dim: int = 384,                    # Embedding dimension
        cache_size: int = 10000,           # LRU cache size
        similarity_threshold: float = 0.7, # Graph edge threshold
        log_level: str = "INFO"            # Logging level
    ): ...
    
    def add_document(
        self,
        text: str,                         # Document text
        vector: np.ndarray,                # Embedding vector
        metadata: dict = None              # Optional metadata
    ) -> str:                              # Returns node_id
        """Add a document to the index."""
    
    def search(
        self,
        query_vector: np.ndarray,          # Query embedding
        k: int = 10,                       # Number of results
        use_graph: bool = True,            # Use graph traversal
        use_cache: bool = True,            # Check cache first
        min_similarity: float = 0.0        # Minimum similarity
    ) -> List[SearchResult]:
        """Search for similar documents."""
    
    def search_text(
        self,
        text: str,                         # Query text
        embed_fn: Callable,                # Embedding function
        k: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """Search using raw text."""
    
    def get_document(self, node_id: str) -> Optional[dict]:
        """Get a document by ID."""
    
    def delete_document(self, node_id: str) -> bool:
        """Delete a document."""
    
    def update_document(
        self,
        node_id: str,
        text: str = None,
        vector: np.ndarray = None,
        metadata: dict = None
    ) -> bool:
        """Update an existing document."""
    
    def get_stats(self) -> dict:
        """Get index statistics."""
    
    def get_metrics(self) -> dict:
        """Get performance metrics."""
    
    def rebuild_index(self) -> None:
        """Rebuild FAISS index from storage."""
    
    def clear(self) -> None:
        """Clear all data (WARNING: destructive!)."""
    
    def close(self) -> None:
        """Close and save all data."""
```

### SearchResult

```python
@dataclass
class SearchResult:
    node_id: str          # Unique document ID
    text: str             # Document text
    similarity: float     # Cosine similarity (0-1)
    metadata: dict        # User metadata
    source: str           # 'cache', 'faiss', or 'graph'
```

### Exceptions

```python
from neuroindex import (
    NeuroIndexError,        # Base exception
    DimensionMismatchError, # Wrong embedding dimension
    StorageError,           # Database/file errors
    IndexCorruptedError,    # Corrupted index
    DocumentNotFoundError,  # Document doesn't exist
    InvalidInputError,      # Invalid input parameters
)
```

## 🔧 Advanced Usage

### Custom Similarity Threshold

```python
# Only create graph edges for very similar documents
ni = NeuroIndex(path="./memory", dim=384, similarity_threshold=0.9)
```

### Filtered Search

```python
# Only return results above a similarity threshold
results = ni.search(query_vec, k=10, min_similarity=0.8)
```

### Disable Graph/Cache for Pure FAISS Search

```python
# Use only FAISS for search (fastest)
results = ni.search(query_vec, k=10, use_graph=False, use_cache=False)
```

### Performance Monitoring

```python
# Get detailed metrics
metrics = ni.get_metrics()
print(f"Cache hit rate: {metrics['cache']['hit_rate']:.2%}")
print(f"Avg search time: {metrics['operations']['search']['avg_ms']:.2f}ms")
```

### Rebuild Index After Many Deletions

```python
# Compact the FAISS index
ni.rebuild_index()
```

## 🆚 Comparison

| Feature              | FAISS | Chroma | Pinecone | NeuroIndex |
|---------------------|-------|--------|----------|------------|
| Vector Search       | ✅    | ✅     | ✅       | ✅         |
| Semantic Graph      | ❌    | ❌     | ❌       | ✅         |
| Local/Offline       | ✅    | ✅     | ❌       | ✅         |
| Persistence         | ❌    | ✅     | ✅       | ✅         |
| Built-in Cache      | ❌    | ❌     | ❌       | ✅         |
| Metrics/Observability| ❌   | ❌     | ✅       | ✅         |
| Zero Dependencies*  | ✅    | ❌     | ❌       | ✅         |

*Only numpy, networkx, and faiss-cpu

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=neuroindex --cov-report=html

# Run benchmarks
python benchmarks/run_benchmarks.py
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- **GitHub**: https://github.com/Umeshkumar667/NeuroIndex
- **PyPI**: https://pypi.org/project/neuroindex/
- **Issues**: https://github.com/Umeshkumar667/NeuroIndex/issues

---

Made with ❤️ by [Umeshkumar Pal](https://github.com/Umeshkumar667)
