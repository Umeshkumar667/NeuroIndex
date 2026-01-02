"""
Unit tests for FAISSIndexManager.
"""

import pytest
import numpy as np
from neuroindex.core import FAISSIndexManager


class TestFAISSBasics:
    """Basic FAISS index operations."""
    
    def test_add_and_search(self, temp_path):
        """Test adding vectors and searching."""
        faiss_idx = FAISSIndexManager(temp_path, dim=128)
        
        vec = np.random.rand(128).astype('float32')
        faiss_idx.add('test_id', vec)
        
        results = faiss_idx.search(vec, k=1)
        
        assert len(results) == 1
        assert results[0][0] == 'test_id'
        assert results[0][1] > 0.99  # Should be very similar (same vector)
    
    def test_search_empty_index(self, temp_path):
        """Searching empty index should return empty list."""
        faiss_idx = FAISSIndexManager(temp_path, dim=128)
        
        query = np.random.rand(128).astype('float32')
        results = faiss_idx.search(query, k=5)
        
        assert results == []
    
    def test_size_property(self, temp_path):
        """Test index size tracking."""
        faiss_idx = FAISSIndexManager(temp_path, dim=128)
        
        assert faiss_idx.size == 0
        
        for i in range(10):
            vec = np.random.rand(128).astype('float32')
            faiss_idx.add(f'id_{i}', vec)
        
        assert faiss_idx.size == 10
    
    def test_remove(self, temp_path):
        """Test removing from index."""
        faiss_idx = FAISSIndexManager(temp_path, dim=128)
        
        vec = np.random.rand(128).astype('float32')
        faiss_idx.add('test_id', vec)
        
        assert faiss_idx.remove('test_id') is True
        
        # Should not find after removal
        results = faiss_idx.search(vec, k=1)
        assert len(results) == 0 or results[0][0] != 'test_id'
    
    def test_remove_nonexistent(self, temp_path):
        """Removing nonexistent should return False."""
        faiss_idx = FAISSIndexManager(temp_path, dim=128)
        assert faiss_idx.remove('nonexistent') is False


class TestFAISSSearch:
    """FAISS search functionality."""
    
    def test_search_returns_most_similar(self, temp_path):
        """Search should return most similar vectors first."""
        faiss_idx = FAISSIndexManager(temp_path, dim=128)
        
        # Add distinct vectors that will have clear similarity differences
        # Use non-zero vectors to avoid normalization issues
        base = np.ones(128, dtype='float32')
        
        for i in range(5):
            vec = base.copy()
            vec[0] = 1.0 + i * 0.5  # Vary first component
            faiss_idx.add(f'id_{i}', vec)
        
        # Query with high first component - should match id_4 best
        query = base.copy()
        query[0] = 3.0
        
        results = faiss_idx.search(query, k=3)
        
        assert len(results) == 3
        # Results should be ordered by similarity (highest first)
        assert results[0][1] >= results[1][1]
        assert results[1][1] >= results[2][1]
    
    def test_search_k_larger_than_index(self, temp_path):
        """Search with k > index size should return all."""
        faiss_idx = FAISSIndexManager(temp_path, dim=128)
        
        for i in range(5):
            vec = np.random.rand(128).astype('float32')
            faiss_idx.add(f'id_{i}', vec)
        
        query = np.random.rand(128).astype('float32')
        results = faiss_idx.search(query, k=100)
        
        assert len(results) == 5
    
    def test_similarity_is_cosine(self, temp_path):
        """Verify similarity is cosine-based."""
        faiss_idx = FAISSIndexManager(temp_path, dim=128)
        
        # Add a vector
        vec = np.ones(128, dtype='float32')
        faiss_idx.add('ones', vec)
        
        # Search with same direction (should be ~1.0)
        results = faiss_idx.search(vec, k=1)
        assert abs(results[0][1] - 1.0) < 0.01
        
        # Search with opposite direction (should be ~-1.0 or low)
        opposite = -vec
        results = faiss_idx.search(opposite, k=1)
        assert results[0][1] < 0.1


class TestFAISSPersistence:
    """FAISS persistence functionality."""
    
    def test_save_and_load(self, temp_path):
        """Index should persist after save/load."""
        # Create and save
        faiss_idx1 = FAISSIndexManager(temp_path, dim=128)
        
        vecs = []
        for i in range(10):
            vec = np.random.rand(128).astype('float32')
            faiss_idx1.add(f'id_{i}', vec)
            vecs.append(vec)
        
        faiss_idx1.save()
        
        # Load in new instance
        faiss_idx2 = FAISSIndexManager(temp_path, dim=128)
        
        assert faiss_idx2.size == 10
        
        # Search should work
        results = faiss_idx2.search(vecs[0], k=1)
        assert results[0][0] == 'id_0'
    
    def test_mappings_persist(self, temp_path):
        """ID mappings should persist correctly."""
        faiss_idx1 = FAISSIndexManager(temp_path, dim=128)
        
        vec = np.random.rand(128).astype('float32')
        faiss_idx1.add('unique_id_12345', vec)
        faiss_idx1.save()
        
        faiss_idx2 = FAISSIndexManager(temp_path, dim=128)
        results = faiss_idx2.search(vec, k=1)
        
        assert results[0][0] == 'unique_id_12345'


class TestFAISSRebuild:
    """FAISS index rebuild functionality."""
    
    def test_rebuild_clears_old_data(self, temp_path):
        """Rebuild should start fresh."""
        faiss_idx = FAISSIndexManager(temp_path, dim=128)
        
        # Add some vectors
        for i in range(10):
            vec = np.random.rand(128).astype('float32')
            faiss_idx.add(f'old_{i}', vec)
        
        # Rebuild with new vectors
        new_vectors = [
            ('new_0', np.random.rand(128).astype('float32')),
            ('new_1', np.random.rand(128).astype('float32')),
        ]
        faiss_idx.rebuild(new_vectors)
        
        assert faiss_idx.size == 2
        
        # Old IDs should not be found
        query = np.random.rand(128).astype('float32')
        results = faiss_idx.search(query, k=10)
        ids = [r[0] for r in results]
        
        assert all(id.startswith('new_') for id in ids)

