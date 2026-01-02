"""
Performance benchmarks for NeuroIndex.

Run with: pytest tests/performance/ -v --benchmark-only
Or standalone: python tests/performance/test_benchmark.py
"""

import pytest
import numpy as np
import time
import tempfile
import shutil
from typing import Dict, Any


class TestInsertionPerformance:
    """Benchmark insertion performance."""
    
    @pytest.mark.parametrize("n_docs", [100, 1000])
    def test_insertion_speed(self, temp_path, n_docs):
        """Benchmark document insertion."""
        from neuroindex import NeuroIndex
        
        ni = NeuroIndex(path=temp_path, dim=384, log_level="WARNING")
        
        vectors = np.random.rand(n_docs, 384).astype('float32')
        texts = [f"Document number {i} with sample content" for i in range(n_docs)]
        
        start = time.time()
        for text, vec in zip(texts, vectors):
            ni.add_document(text, vec)
        duration = time.time() - start
        
        docs_per_sec = n_docs / duration
        
        print(f"\n  Inserted {n_docs} docs in {duration:.2f}s ({docs_per_sec:.0f} docs/sec)")
        
        # Basic performance expectations
        assert duration < n_docs * 0.1  # At least 10 docs/sec
        
        ni.close()


class TestSearchPerformance:
    """Benchmark search performance."""
    
    def test_search_latency(self, temp_path):
        """Benchmark search latency."""
        from neuroindex import NeuroIndex
        
        ni = NeuroIndex(path=temp_path, dim=384, log_level="WARNING")
        
        # Add documents
        n_docs = 1000
        for i in range(n_docs):
            vec = np.random.rand(384).astype('float32')
            ni.add_document(f"Doc {i}", vec)
        
        # Warm up
        query = np.random.rand(384).astype('float32')
        ni.search(query, k=10)
        
        # Benchmark
        n_queries = 100
        latencies = []
        
        for _ in range(n_queries):
            query = np.random.rand(384).astype('float32')
            start = time.time()
            ni.search(query, k=10)
            latencies.append(time.time() - start)
        
        avg_latency = np.mean(latencies) * 1000
        p99_latency = np.percentile(latencies, 99) * 1000
        
        print(f"\n  Search latency (n={n_docs}): avg={avg_latency:.2f}ms, p99={p99_latency:.2f}ms")
        
        # Basic latency expectations
        assert avg_latency < 100  # Less than 100ms average
        
        ni.close()
    
    @pytest.mark.parametrize("k", [1, 10, 50])
    def test_search_k_scaling(self, temp_path, k):
        """Test how search scales with k."""
        from neuroindex import NeuroIndex
        
        ni = NeuroIndex(path=temp_path, dim=384, log_level="WARNING")
        
        # Add documents
        for i in range(500):
            vec = np.random.rand(384).astype('float32')
            ni.add_document(f"Doc {i}", vec)
        
        # Benchmark
        n_queries = 50
        latencies = []
        
        for _ in range(n_queries):
            query = np.random.rand(384).astype('float32')
            start = time.time()
            ni.search(query, k=k)
            latencies.append(time.time() - start)
        
        avg_latency = np.mean(latencies) * 1000
        
        print(f"\n  Search k={k}: avg={avg_latency:.2f}ms")
        
        ni.close()


class TestMemoryUsage:
    """Benchmark memory usage."""
    
    def test_memory_efficiency(self, temp_path):
        """Test memory doesn't grow unbounded."""
        import sys
        from neuroindex import NeuroIndex
        
        ni = NeuroIndex(path=temp_path, dim=384, cache_size=100, log_level="WARNING")
        
        # Add more docs than cache size
        for i in range(500):
            vec = np.random.rand(384).astype('float32')
            ni.add_document(f"Doc {i}", vec)
        
        stats = ni.get_stats()
        
        # Cache should not exceed max size
        assert stats['cache_size'] <= 100
        
        ni.close()


def run_full_benchmark():
    """Run comprehensive benchmarks and print results."""
    from neuroindex import NeuroIndex
    
    print("\n" + "=" * 60)
    print("NeuroIndex Performance Benchmarks")
    print("=" * 60)
    
    results: Dict[str, Any] = {}
    
    for n_docs in [1000, 5000, 10000]:
        temp_path = tempfile.mkdtemp()
        
        try:
            ni = NeuroIndex(path=temp_path, dim=384, log_level="WARNING")
            
            # Insertion benchmark
            vectors = np.random.rand(n_docs, 384).astype('float32')
            
            start = time.time()
            for i in range(n_docs):
                ni.add_document(f"Document {i}", vectors[i])
            insert_time = time.time() - start
            
            # Search benchmark
            n_queries = 100
            latencies = []
            
            for _ in range(n_queries):
                query = np.random.rand(384).astype('float32')
                start = time.time()
                ni.search(query, k=10)
                latencies.append(time.time() - start)
            
            results[n_docs] = {
                'insert_time': insert_time,
                'insert_rate': n_docs / insert_time,
                'avg_search_ms': np.mean(latencies) * 1000,
                'p99_search_ms': np.percentile(latencies, 99) * 1000,
            }
            
            ni.close()
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)
    
    # Print results
    print("\n📊 Results:")
    print("-" * 60)
    print(f"{'Docs':>10} | {'Insert (s)':>12} | {'Rate (d/s)':>12} | {'Avg (ms)':>10} | {'P99 (ms)':>10}")
    print("-" * 60)
    
    for n_docs, data in results.items():
        print(f"{n_docs:>10,} | {data['insert_time']:>12.2f} | {data['insert_rate']:>12.0f} | {data['avg_search_ms']:>10.2f} | {data['p99_search_ms']:>10.2f}")
    
    print("-" * 60)
    print("\n✅ Benchmarks complete!")
    
    return results


if __name__ == "__main__":
    run_full_benchmark()

