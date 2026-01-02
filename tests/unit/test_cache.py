"""
Unit tests for NeuroCache (LRU cache).
"""

import pytest
import numpy as np
from neuroindex.core import NeuroCache


class TestNeuroCacheBasics:
    """Basic cache operations."""
    
    def test_add_and_get(self):
        """Test adding and retrieving from cache."""
        cache = NeuroCache(max_size=10)
        
        node = {
            'id': 'test_id',
            'text': 'test text',
            'vector': np.random.rand(128).astype('float32'),
            'metadata': {'key': 'value'}
        }
        
        cache.add('test_id', node)
        result = cache.get('test_id')
        
        assert result is not None
        assert result['id'] == 'test_id'
        assert result['text'] == 'test text'
    
    def test_get_nonexistent_returns_none(self):
        """Getting nonexistent key should return None."""
        cache = NeuroCache(max_size=10)
        assert cache.get('nonexistent') is None
    
    def test_remove(self):
        """Test removing from cache."""
        cache = NeuroCache(max_size=10)
        
        node = {'id': 'test_id', 'text': 'test', 'vector': np.zeros(128), 'metadata': {}}
        cache.add('test_id', node)
        
        assert cache.remove('test_id') is True
        assert cache.get('test_id') is None
    
    def test_remove_nonexistent_returns_false(self):
        """Removing nonexistent key should return False."""
        cache = NeuroCache(max_size=10)
        assert cache.remove('nonexistent') is False
    
    def test_len(self):
        """Test cache length."""
        cache = NeuroCache(max_size=10)
        
        for i in range(5):
            node = {'id': f'id_{i}', 'text': 'test', 'vector': np.zeros(128), 'metadata': {}}
            cache.add(f'id_{i}', node)
        
        assert len(cache) == 5


class TestNeuroCacheLRU:
    """LRU eviction behavior."""
    
    def test_lru_eviction(self):
        """Oldest items should be evicted when cache is full."""
        cache = NeuroCache(max_size=3)
        
        for i in range(5):
            node = {'id': f'id_{i}', 'text': 'test', 'vector': np.zeros(128), 'metadata': {}}
            cache.add(f'id_{i}', node)
        
        # First two should be evicted
        assert cache.get('id_0') is None
        assert cache.get('id_1') is None
        
        # Last three should still be there
        assert cache.get('id_2') is not None
        assert cache.get('id_3') is not None
        assert cache.get('id_4') is not None
    
    def test_access_updates_order(self):
        """Accessing an item should move it to the end."""
        cache = NeuroCache(max_size=3)
        
        for i in range(3):
            node = {'id': f'id_{i}', 'text': 'test', 'vector': np.zeros(128), 'metadata': {}}
            cache.add(f'id_{i}', node)
        
        # Access id_0 to move it to the end
        cache.get('id_0')
        
        # Add new item, should evict id_1 (now oldest)
        node = {'id': 'id_3', 'text': 'test', 'vector': np.zeros(128), 'metadata': {}}
        cache.add('id_3', node)
        
        assert cache.get('id_0') is not None  # Was accessed, should survive
        assert cache.get('id_1') is None  # Should be evicted
    
    def test_update_existing_reorders(self):
        """Updating existing item should move it to end."""
        cache = NeuroCache(max_size=3)
        
        for i in range(3):
            node = {'id': f'id_{i}', 'text': 'test', 'vector': np.zeros(128), 'metadata': {}}
            cache.add(f'id_{i}', node)
        
        # Update id_0
        node = {'id': 'id_0', 'text': 'updated', 'vector': np.zeros(128), 'metadata': {}}
        cache.add('id_0', node)
        
        # Add new item, should evict id_1
        node = {'id': 'id_3', 'text': 'test', 'vector': np.zeros(128), 'metadata': {}}
        cache.add('id_3', node)
        
        result = cache.get('id_0')
        assert result is not None
        assert result['text'] == 'updated'


class TestNeuroCacheSearch:
    """Cache search functionality."""
    
    def test_search_returns_sorted_by_similarity(self):
        """Search should return results sorted by similarity."""
        cache = NeuroCache(max_size=10)
        
        # Add documents with different vectors
        for i in range(5):
            vec = np.zeros(128, dtype='float32')
            vec[0] = i * 0.2  # Different first component
            node = {'id': f'id_{i}', 'text': f'text_{i}', 'vector': vec, 'metadata': {}}
            cache.add(f'id_{i}', node)
        
        # Query with vector similar to id_4
        query = np.zeros(128, dtype='float32')
        query[0] = 0.8
        
        results = cache.search(query, k=3)
        
        assert len(results) == 3
        # Results should be sorted by similarity (highest first)
        assert results[0].similarity >= results[1].similarity
        assert results[1].similarity >= results[2].similarity
    
    def test_search_empty_cache(self):
        """Search on empty cache should return empty list."""
        cache = NeuroCache(max_size=10)
        query = np.random.rand(128).astype('float32')
        results = cache.search(query, k=5)
        assert results == []
    
    def test_search_k_larger_than_cache(self):
        """Search with k larger than cache size should return all items."""
        cache = NeuroCache(max_size=10)
        
        for i in range(3):
            vec = np.random.rand(128).astype('float32')
            node = {'id': f'id_{i}', 'text': f'text_{i}', 'vector': vec, 'metadata': {}}
            cache.add(f'id_{i}', node)
        
        query = np.random.rand(128).astype('float32')
        results = cache.search(query, k=10)
        
        assert len(results) == 3


class TestNeuroCacheClear:
    """Cache clearing functionality."""
    
    def test_clear_empties_cache(self):
        """Clear should remove all items."""
        cache = NeuroCache(max_size=10)
        
        for i in range(5):
            node = {'id': f'id_{i}', 'text': 'test', 'vector': np.zeros(128), 'metadata': {}}
            cache.add(f'id_{i}', node)
        
        cache.clear()
        
        assert len(cache) == 0
        for i in range(5):
            assert cache.get(f'id_{i}') is None

