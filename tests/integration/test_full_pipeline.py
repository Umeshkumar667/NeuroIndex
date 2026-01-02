"""
Integration tests for the full NeuroIndex pipeline.
"""

import pytest
import numpy as np
from neuroindex import NeuroIndex, SearchResult


class TestAddAndSearch:
    """Test add and search integration."""
    
    def test_add_single_and_search(self, ni, sample_vector):
        """Add one document and search for it."""
        node_id = ni.add_document("Test document", sample_vector)
        
        results = ni.search(sample_vector, k=1)
        
        assert len(results) == 1
        assert results[0].node_id == node_id
        assert results[0].text == "Test document"
        assert results[0].similarity > 0.99
    
    def test_add_multiple_and_search(self, ni):
        """Add multiple documents and search."""
        docs = []
        for i in range(10):
            vec = np.random.rand(128).astype('float32')
            node_id = ni.add_document(f"Document {i}", vec, metadata={'index': i})
            docs.append((node_id, vec))
        
        # Search for each document
        for node_id, vec in docs:
            results = ni.search(vec, k=1)
            assert len(results) >= 1
            assert results[0].node_id == node_id
    
    def test_search_returns_correct_order(self, ni):
        """Results should be ordered by similarity."""
        base_vec = np.ones(128, dtype='float32')
        
        # Add documents with varying similarity to base
        for i in range(5):
            vec = base_vec * (1 - i * 0.1)  # Decreasing similarity
            ni.add_document(f"Doc {i}", vec)
        
        results = ni.search(base_vec, k=5)
        
        # Verify descending similarity order
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity
    
    def test_search_with_min_similarity(self, ni):
        """min_similarity should filter results."""
        base = np.ones(128, dtype='float32')
        orthogonal = np.zeros(128, dtype='float32')
        orthogonal[0] = 1.0
        
        ni.add_document("Similar", base)
        ni.add_document("Different", orthogonal)
        
        results = ni.search(base, k=10, min_similarity=0.9)
        
        # Should only return the similar document
        assert all(r.similarity >= 0.9 for r in results)


class TestSearchText:
    """Test text-based search."""
    
    def test_search_text_basic(self, ni, embed_fn):
        """Test search_text helper."""
        vec = embed_fn("hello world")
        ni.add_document("hello world", vec)
        
        results = ni.search_text("hello world", embed_fn=embed_fn, k=1)
        
        assert len(results) == 1
        assert "hello" in results[0].text.lower()


class TestGetDocument:
    """Test document retrieval."""
    
    def test_get_existing_document(self, ni, sample_vector):
        """Get an existing document by ID."""
        node_id = ni.add_document("Test doc", sample_vector, metadata={'key': 'value'})
        
        doc = ni.get_document(node_id)
        
        assert doc is not None
        assert doc['id'] == node_id
        assert doc['text'] == "Test doc"
        assert doc['metadata'] == {'key': 'value'}
    
    def test_get_nonexistent_document(self, ni):
        """Get a nonexistent document returns None."""
        doc = ni.get_document("nonexistent_id")
        assert doc is None


class TestDeleteDocument:
    """Test document deletion."""
    
    def test_delete_existing(self, ni, sample_vector):
        """Delete an existing document."""
        node_id = ni.add_document("To delete", sample_vector)
        
        result = ni.delete_document(node_id)
        
        assert result is True
        assert ni.get_document(node_id) is None
    
    def test_delete_nonexistent(self, ni):
        """Delete nonexistent returns False."""
        result = ni.delete_document("nonexistent")
        assert result is False
    
    def test_deleted_not_in_search(self, ni, sample_vector):
        """Deleted documents should not appear in search."""
        node_id = ni.add_document("Delete me", sample_vector)
        ni.delete_document(node_id)
        
        results = ni.search(sample_vector, k=10)
        result_ids = [r.node_id for r in results]
        
        assert node_id not in result_ids


class TestUpdateDocument:
    """Test document updates."""
    
    def test_update_text(self, ni, sample_vector):
        """Update document text."""
        node_id = ni.add_document("Original text", sample_vector)
        
        result = ni.update_document(node_id, text="Updated text")
        
        assert result is True
        doc = ni.get_document(node_id)
        assert doc['text'] == "Updated text"
    
    def test_update_metadata(self, ni, sample_vector):
        """Update document metadata."""
        node_id = ni.add_document("Test", sample_vector, metadata={'old': 'data'})
        
        ni.update_document(node_id, metadata={'new': 'data'})
        
        doc = ni.get_document(node_id)
        assert doc['metadata'] == {'new': 'data'}
    
    def test_update_vector(self, ni):
        """Update document vector."""
        old_vec = np.zeros(128, dtype='float32')
        new_vec = np.ones(128, dtype='float32')
        
        node_id = ni.add_document("Test", old_vec)
        ni.update_document(node_id, vector=new_vec)
        
        # Search with new vector should find it
        results = ni.search(new_vec, k=1)
        assert len(results) >= 1
        assert results[0].node_id == node_id
    
    def test_update_nonexistent(self, ni):
        """Update nonexistent returns False."""
        result = ni.update_document("nonexistent", text="new")
        assert result is False


class TestDuplicateHandling:
    """Test duplicate document handling."""
    
    def test_duplicate_returns_same_id(self, ni, sample_vector):
        """Adding duplicate should return existing ID."""
        id1 = ni.add_document("Same text", sample_vector)
        id2 = ni.add_document("Same text", sample_vector)
        
        assert id1 == id2
    
    def test_duplicate_not_added_twice(self, ni, sample_vector):
        """Duplicate should not increase document count."""
        ni.add_document("Same text", sample_vector)
        count_before = ni.get_stats()['total_documents']
        
        ni.add_document("Same text", sample_vector)
        count_after = ni.get_stats()['total_documents']
        
        assert count_before == count_after


class TestStats:
    """Test statistics."""
    
    def test_stats_structure(self, ni):
        """Stats should have expected structure."""
        stats = ni.get_stats()
        
        assert 'version' in stats
        assert 'total_documents' in stats
        assert 'faiss_vectors' in stats
        assert 'cache_size' in stats
        assert 'graph_nodes' in stats
        assert 'graph_edges' in stats
        assert 'dimension' in stats
        assert 'path' in stats
    
    def test_stats_values(self, ni, sample_vector):
        """Stats should reflect actual state."""
        for i in range(5):
            vec = np.random.rand(128).astype('float32')
            ni.add_document(f"Doc {i}", vec)
        
        stats = ni.get_stats()
        
        assert stats['total_documents'] == 5
        assert stats['faiss_vectors'] == 5
        assert stats['dimension'] == 128


class TestMetrics:
    """Test metrics collection."""
    
    def test_metrics_structure(self, ni):
        """Metrics should have expected structure."""
        metrics = ni.get_metrics()
        
        assert 'uptime_seconds' in metrics
        assert 'cache' in metrics
        assert 'operations' in metrics
    
    def test_metrics_track_operations(self, ni, sample_vector):
        """Metrics should track operations."""
        ni.add_document("Test", sample_vector)
        ni.search(sample_vector, k=1)
        
        metrics = ni.get_metrics()
        
        assert 'add_document' in metrics['operations']
        assert 'search' in metrics['operations']
        assert metrics['operations']['add_document']['count'] >= 1
        assert metrics['operations']['search']['count'] >= 1


class TestContextManager:
    """Test context manager usage."""
    
    def test_context_manager(self, temp_path, sample_vector):
        """Context manager should properly close."""
        with NeuroIndex(path=temp_path, dim=128, log_level="WARNING") as ni:
            ni.add_document("Test", sample_vector)
            stats = ni.get_stats()
        
        assert stats['total_documents'] == 1


class TestRepr:
    """Test string representation."""
    
    def test_repr(self, ni):
        """Repr should be informative."""
        repr_str = repr(ni)
        
        assert 'NeuroIndex' in repr_str
        assert 'dim=128' in repr_str

