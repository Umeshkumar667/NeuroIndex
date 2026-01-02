"""
NeuroIndex - Production-Ready Hybrid Vector + Semantic Graph Memory System

A high-performance memory system for AI applications combining:
- FAISS vector search for fast similarity matching
- Semantic graph traversal for relationship-aware retrieval
- LRU cache for frequently accessed items
- SQLite persistence for durability

Example:
    >>> from neuroindex import NeuroIndex
    >>> import numpy as np
    >>>
    >>> with NeuroIndex(path="./memory", dim=384) as ni:
    ...     embedding = np.random.rand(384).astype('float32')
    ...     node_id = ni.add_document("Hello world", embedding)
    ...     results = ni.search(embedding, k=5)
    ...     print(results)

Author: Umeshkumar Pal
License: MIT
Repository: https://github.com/Umeshkumar667/NeuroIndex
"""

from .core import NeuroIndex, SearchResult
from .exceptions import (
    ConcurrencyError,
    DimensionMismatchError,
    DocumentNotFoundError,
    IndexCorruptedError,
    InvalidInputError,
    NeuroIndexError,
    StorageError,
)
from .metrics import MetricsCollector

__version__ = "0.2.0"
__author__ = "Umeshkumar Pal"
__license__ = "MIT"

__all__ = [
    # Main classes
    "NeuroIndex",
    "SearchResult",
    "MetricsCollector",
    # Exceptions
    "NeuroIndexError",
    "DimensionMismatchError",
    "StorageError",
    "IndexCorruptedError",
    "DocumentNotFoundError",
    "InvalidInputError",
    "ConcurrencyError",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]
