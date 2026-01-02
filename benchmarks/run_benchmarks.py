#!/usr/bin/env python3
"""
NeuroIndex Performance Benchmarks

Comprehensive benchmarks for measuring NeuroIndex performance.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --docs 10000 50000 100000
    python benchmarks/run_benchmarks.py --output results.json
"""

import argparse
import json
import time
import tempfile
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import numpy as np

# Add parent directory to path for importing neuroindex
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuroindex import NeuroIndex


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def benchmark_insertion(n_docs: int, dim: int = 384) -> Dict[str, Any]:
    """Benchmark document insertion."""
    temp_path = tempfile.mkdtemp()
    
    try:
        ni = NeuroIndex(path=temp_path, dim=dim, log_level="WARNING")
        
        # Pre-generate data
        vectors = np.random.rand(n_docs, dim).astype('float32')
        texts = [f"Document {i}: This is sample text content for benchmarking purposes." for i in range(n_docs)]
        
        mem_before = get_memory_mb()
        
        start = time.time()
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            ni.add_document(text, vec, metadata={'index': i})
        duration = time.time() - start
        
        mem_after = get_memory_mb()
        
        result = {
            'n_docs': n_docs,
            'dimension': dim,
            'duration_sec': round(duration, 3),
            'docs_per_sec': round(n_docs / duration, 1),
            'ms_per_doc': round(duration / n_docs * 1000, 3),
            'memory_mb': round(mem_after - mem_before, 1) if mem_before > 0 else None,
        }
        
        ni.close()
        return result
        
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def benchmark_search(n_docs: int, n_queries: int = 100, k: int = 10, dim: int = 384) -> Dict[str, Any]:
    """Benchmark search operations."""
    temp_path = tempfile.mkdtemp()
    
    try:
        ni = NeuroIndex(path=temp_path, dim=dim, log_level="WARNING")
        
        # Insert documents
        print(f"    Inserting {n_docs:,} documents...")
        for i in range(n_docs):
            vec = np.random.rand(dim).astype('float32')
            ni.add_document(f"Document {i}", vec)
        
        # Warm up
        for _ in range(10):
            query = np.random.rand(dim).astype('float32')
            ni.search(query, k=k)
        
        # Benchmark
        print(f"    Running {n_queries} search queries...")
        latencies = []
        
        for _ in range(n_queries):
            query = np.random.rand(dim).astype('float32')
            start = time.time()
            results = ni.search(query, k=k)
            latencies.append(time.time() - start)
        
        latencies_ms = np.array(latencies) * 1000
        
        result = {
            'n_docs': n_docs,
            'n_queries': n_queries,
            'k': k,
            'dimension': dim,
            'avg_latency_ms': round(float(np.mean(latencies_ms)), 3),
            'p50_latency_ms': round(float(np.percentile(latencies_ms, 50)), 3),
            'p95_latency_ms': round(float(np.percentile(latencies_ms, 95)), 3),
            'p99_latency_ms': round(float(np.percentile(latencies_ms, 99)), 3),
            'min_latency_ms': round(float(np.min(latencies_ms)), 3),
            'max_latency_ms': round(float(np.max(latencies_ms)), 3),
            'qps': round(n_queries / sum(latencies), 1),
        }
        
        ni.close()
        return result
        
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def benchmark_persistence(n_docs: int, dim: int = 384) -> Dict[str, Any]:
    """Benchmark persistence (save/load)."""
    temp_path = tempfile.mkdtemp()
    
    try:
        # Create and populate index
        ni = NeuroIndex(path=temp_path, dim=dim, log_level="WARNING")
        
        for i in range(n_docs):
            vec = np.random.rand(dim).astype('float32')
            ni.add_document(f"Document {i}", vec)
        
        # Close (saves data)
        start = time.time()
        ni.close()
        save_time = time.time() - start
        
        # Reload
        start = time.time()
        ni2 = NeuroIndex(path=temp_path, dim=dim, log_level="WARNING")
        load_time = time.time() - start
        
        # Verify
        stats = ni2.get_stats()
        ni2.close()
        
        return {
            'n_docs': n_docs,
            'save_time_sec': round(save_time, 3),
            'load_time_sec': round(load_time, 3),
            'verified_docs': stats['total_documents'],
        }
        
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def run_benchmarks(doc_counts: List[int], output_file: str = None) -> Dict[str, Any]:
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("NeuroIndex Performance Benchmarks")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Document counts: {doc_counts}")
    print()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'version': '0.2.0',
        'insertion': [],
        'search': [],
        'persistence': [],
    }
    
    # Insertion benchmarks
    print("[INSERT] Insertion Benchmarks")
    print("-" * 50)
    
    for n_docs in doc_counts:
        print(f"  Testing {n_docs:,} documents...")
        result = benchmark_insertion(n_docs)
        results['insertion'].append(result)
        print(f"    OK: {result['duration_sec']:.2f}s ({result['docs_per_sec']:.0f} docs/sec)")
    
    print()
    
    # Search benchmarks
    print("\n[SEARCH] Search Benchmarks")
    print("-" * 50)
    
    for n_docs in doc_counts:
        print(f"  Testing {n_docs:,} documents...")
        result = benchmark_search(n_docs)
        results['search'].append(result)
        print(f"    OK: avg={result['avg_latency_ms']:.2f}ms, p99={result['p99_latency_ms']:.2f}ms, {result['qps']:.0f} QPS")
    
    print()
    
    # Persistence benchmarks
    print("\n[PERSIST] Persistence Benchmarks")
    print("-" * 50)
    
    for n_docs in doc_counts:
        print(f"  Testing {n_docs:,} documents...")
        result = benchmark_persistence(n_docs)
        results['persistence'].append(result)
        print(f"    OK: save={result['save_time_sec']:.2f}s, load={result['load_time_sec']:.2f}s")
    
    print()
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Docs':>10} | {'Insert (s)':>12} | {'Docs/sec':>10} | {'Search (ms)':>12} | {'QPS':>8}")
    print("-" * 70)
    
    for i, n_docs in enumerate(doc_counts):
        insert = results['insertion'][i]
        search = results['search'][i]
        print(f"{n_docs:>10,} | {insert['duration_sec']:>12.2f} | {insert['docs_per_sec']:>10.0f} | {search['avg_latency_ms']:>12.2f} | {search['qps']:>8.0f}")
    
    print("-" * 70)
    print()
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[OK] Results saved to {output_file}")
    
    print("[OK] Benchmarks complete!")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run NeuroIndex benchmarks')
    parser.add_argument('--docs', nargs='+', type=int, default=[1000, 5000, 10000],
                        help='Document counts to benchmark')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    run_benchmarks(args.docs, args.output)


if __name__ == "__main__":
    main()

