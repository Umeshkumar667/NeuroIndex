"""
NeuroIndex Metrics & Observability

Provides production-grade metrics tracking for monitoring and optimization.
"""

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class OperationMetrics:
    """Tracks metrics for a single operation type."""

    latencies: List[float] = field(default_factory=list)
    count: int = 0
    errors: int = 0

    def record(self, latency: float, success: bool = True):
        self.latencies.append(latency)
        self.count += 1
        if not success:
            self.errors += 1

    def summary(self) -> Dict[str, Any]:
        if not self.latencies:
            return {
                "count": 0,
                "errors": 0,
                "avg_ms": 0,
                "p50_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0,
            }

        arr = np.array(self.latencies) * 1000  # Convert to ms
        return {
            "count": self.count,
            "errors": self.errors,
            "avg_ms": round(float(np.mean(arr)), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "min_ms": round(float(np.min(arr)), 3),
            "max_ms": round(float(np.max(arr)), 3),
        }


class MetricsCollector:
    """
    Thread-safe metrics collector for NeuroIndex operations.

    Usage:
        metrics = MetricsCollector()

        with metrics.measure("search"):
            # do search operation
            pass

        print(metrics.summary())
    """

    def __init__(self):
        self._lock = Lock()
        self._operations: Dict[str, OperationMetrics] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._graph_traversals = 0
        self._faiss_searches = 0
        self._start_time = time.time()

    def _get_or_create(self, operation: str) -> OperationMetrics:
        if operation not in self._operations:
            self._operations[operation] = OperationMetrics()
        return self._operations[operation]

    def record(self, operation: str, latency: float, success: bool = True):
        """Record a single operation."""
        with self._lock:
            self._get_or_create(operation).record(latency, success)

    def record_cache_hit(self):
        """Record a cache hit."""
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss."""
        with self._lock:
            self._cache_misses += 1

    def record_graph_traversal(self):
        """Record a graph traversal."""
        with self._lock:
            self._graph_traversals += 1

    def record_faiss_search(self):
        """Record a FAISS search."""
        with self._lock:
            self._faiss_searches += 1

    class _MeasureContext:
        """Context manager for measuring operation latency."""

        def __init__(self, collector: "MetricsCollector", operation: str):
            self.collector = collector
            self.operation = operation
            self.start_time: Optional[float] = None
            self.success = True

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            latency = time.time() - self.start_time
            self.success = exc_type is None
            self.collector.record(self.operation, latency, self.success)
            return False  # Don't suppress exceptions

    def measure(self, operation: str) -> _MeasureContext:
        """
        Context manager to measure operation latency.

        Usage:
            with metrics.measure("search"):
                results = index.search(query)
        """
        return self._MeasureContext(self, operation)

    def summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        with self._lock:
            uptime = time.time() - self._start_time
            total_cache = self._cache_hits + self._cache_misses

            result = {
                "uptime_seconds": round(uptime, 2),
                "cache": {
                    "hits": self._cache_hits,
                    "misses": self._cache_misses,
                    "hit_rate": round(self._cache_hits / max(1, total_cache), 4),
                },
                "graph_traversals": self._graph_traversals,
                "faiss_searches": self._faiss_searches,
                "operations": {},
            }

            for op_name, op_metrics in self._operations.items():
                result["operations"][op_name] = op_metrics.summary()

            return result

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._operations.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._graph_traversals = 0
            self._faiss_searches = 0
            self._start_time = time.time()
