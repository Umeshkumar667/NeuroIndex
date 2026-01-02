"""
Integration tests for concurrent operations.
"""

import pytest
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from neuroindex import NeuroIndex


class TestConcurrentWrites:
    """Test concurrent write operations."""
    
    def test_concurrent_adds(self, ni):
        """Multiple threads adding documents should work."""
        errors = []
        node_ids = []
        lock = threading.Lock()
        
        def add_docs(worker_id: int, count: int):
            try:
                for i in range(count):
                    vec = np.random.rand(128).astype('float32')
                    node_id = ni.add_document(f"Worker {worker_id} Doc {i}", vec)
                    with lock:
                        node_ids.append(node_id)
            except Exception as e:
                with lock:
                    errors.append(e)
        
        threads = []
        num_workers = 5
        docs_per_worker = 20
        
        for i in range(num_workers):
            t = threading.Thread(target=add_docs, args=(i, docs_per_worker))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(node_ids) == num_workers * docs_per_worker
        
        # Verify all documents are searchable
        stats = ni.get_stats()
        assert stats['total_documents'] == num_workers * docs_per_worker
    
    def test_concurrent_adds_with_executor(self, ni):
        """Test with ThreadPoolExecutor."""
        def add_doc(idx: int):
            vec = np.random.rand(128).astype('float32')
            return ni.add_document(f"Doc {idx}", vec)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_doc, i) for i in range(100)]
            node_ids = [f.result() for f in as_completed(futures)]
        
        assert len(node_ids) == 100
        assert ni.get_stats()['total_documents'] == 100


class TestConcurrentReads:
    """Test concurrent read operations."""
    
    def test_concurrent_searches(self, ni_with_docs):
        """Multiple threads searching should work."""
        errors = []
        results_count = []
        lock = threading.Lock()
        
        def search_docs(worker_id: int, count: int):
            try:
                for _ in range(count):
                    vec = np.random.rand(128).astype('float32')
                    results = ni_with_docs.search(vec, k=3)
                    with lock:
                        results_count.append(len(results))
            except Exception as e:
                with lock:
                    errors.append(e)
        
        threads = []
        num_workers = 10
        searches_per_worker = 20
        
        for i in range(num_workers):
            t = threading.Thread(target=search_docs, args=(i, searches_per_worker))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results_count) == num_workers * searches_per_worker
    
    def test_concurrent_gets(self, ni):
        """Multiple threads getting documents should work."""
        # First, add some documents
        node_ids = []
        for i in range(10):
            vec = np.random.rand(128).astype('float32')
            node_id = ni.add_document(f"Doc {i}", vec)
            node_ids.append(node_id)
        
        errors = []
        docs_found = []
        lock = threading.Lock()
        
        def get_docs(count: int):
            try:
                for _ in range(count):
                    for node_id in node_ids:
                        doc = ni.get_document(node_id)
                        if doc:
                            with lock:
                                docs_found.append(doc['id'])
            except Exception as e:
                with lock:
                    errors.append(e)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=get_docs, args=(10,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0


class TestMixedConcurrency:
    """Test mixed concurrent operations."""
    
    def test_concurrent_read_write(self, ni):
        """Concurrent reads and writes should work."""
        stop_event = threading.Event()
        errors = []
        lock = threading.Lock()
        
        def writer(worker_id: int):
            try:
                count = 0
                while not stop_event.is_set() and count < 50:
                    vec = np.random.rand(128).astype('float32')
                    ni.add_document(f"Writer {worker_id} Doc {count}", vec)
                    count += 1
            except Exception as e:
                with lock:
                    errors.append(('writer', e))
        
        def reader(worker_id: int):
            try:
                count = 0
                while not stop_event.is_set() and count < 100:
                    vec = np.random.rand(128).astype('float32')
                    ni.search(vec, k=5)
                    count += 1
            except Exception as e:
                with lock:
                    errors.append(('reader', e))
        
        # Start writers and readers
        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=writer, args=(i,)))
        for i in range(5):
            threads.append(threading.Thread(target=reader, args=(i,)))
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join(timeout=30)
        
        stop_event.set()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
    
    def test_concurrent_update_read(self, ni):
        """Concurrent updates and reads should work."""
        # Add initial documents
        node_ids = []
        for i in range(10):
            vec = np.random.rand(128).astype('float32')
            node_id = ni.add_document(f"Doc {i}", vec)
            node_ids.append(node_id)
        
        errors = []
        lock = threading.Lock()
        
        def updater():
            try:
                for _ in range(20):
                    for node_id in node_ids:
                        ni.update_document(node_id, metadata={'updated': True})
            except Exception as e:
                with lock:
                    errors.append(('updater', e))
        
        def reader():
            try:
                for _ in range(50):
                    for node_id in node_ids:
                        ni.get_document(node_id)
            except Exception as e:
                with lock:
                    errors.append(('reader', e))
        
        threads = [
            threading.Thread(target=updater),
            threading.Thread(target=updater),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"

