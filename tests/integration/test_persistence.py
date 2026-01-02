"""
Integration tests for data persistence.
"""

import pytest
import numpy as np
from neuroindex import NeuroIndex


class TestDataPersistence:
    """Test that data persists across restarts."""
    
    def test_documents_persist(self, temp_path):
        """Documents should persist after close/reopen."""
        vec = np.random.rand(128).astype('float32')
        
        # First session
        ni1 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        node_id = ni1.add_document("Persistent doc", vec, metadata={'key': 'value'})
        ni1.close()
        
        # Second session
        ni2 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        doc = ni2.get_document(node_id)
        
        assert doc is not None
        assert doc['text'] == "Persistent doc"
        assert doc['metadata'] == {'key': 'value'}
        np.testing.assert_array_almost_equal(doc['vector'], vec)
        
        ni2.close()
    
    def test_faiss_index_persists(self, temp_path):
        """FAISS index should persist and work after restart."""
        vec = np.random.rand(128).astype('float32')
        
        # First session
        ni1 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        node_id = ni1.add_document("FAISS test", vec)
        ni1.close()
        
        # Second session
        ni2 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        results = ni2.search(vec, k=1)
        
        assert len(results) >= 1
        assert results[0].node_id == node_id
        
        ni2.close()
    
    def test_graph_persists(self, temp_path):
        """Semantic graph should persist after restart."""
        vecs = [np.random.rand(128).astype('float32') for _ in range(5)]
        
        # First session - add similar docs to create edges
        ni1 = NeuroIndex(path=temp_path, dim=128, similarity_threshold=0.5, log_level="WARNING")
        base = vecs[0]
        for i, vec in enumerate(vecs):
            ni1.add_document(f"Doc {i}", base + vec * 0.01)  # Similar vectors
        
        stats1 = ni1.get_stats()
        ni1.close()
        
        # Second session
        ni2 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        stats2 = ni2.get_stats()
        
        assert stats2['graph_nodes'] == stats1['graph_nodes']
        assert stats2['graph_edges'] == stats1['graph_edges']
        
        ni2.close()
    
    def test_multiple_documents_persist(self, temp_path):
        """Multiple documents should all persist."""
        # First session
        ni1 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        
        node_ids = []
        for i in range(20):
            vec = np.random.rand(128).astype('float32')
            node_id = ni1.add_document(f"Document {i}", vec)
            node_ids.append(node_id)
        
        ni1.close()
        
        # Second session
        ni2 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        
        assert ni2.get_stats()['total_documents'] == 20
        
        for node_id in node_ids:
            doc = ni2.get_document(node_id)
            assert doc is not None
        
        ni2.close()


class TestDeletePersistence:
    """Test that deletions persist."""
    
    def test_delete_persists(self, temp_path):
        """Deleted documents should stay deleted after restart."""
        vec = np.random.rand(128).astype('float32')
        
        # First session - add and delete
        ni1 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        node_id = ni1.add_document("To delete", vec)
        ni1.delete_document(node_id)
        ni1.close()
        
        # Second session - verify deleted
        ni2 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        doc = ni2.get_document(node_id)
        
        assert doc is None
        
        ni2.close()


class TestUpdatePersistence:
    """Test that updates persist."""
    
    def test_text_update_persists(self, temp_path):
        """Text updates should persist."""
        vec = np.random.rand(128).astype('float32')
        
        # First session
        ni1 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        node_id = ni1.add_document("Original", vec)
        ni1.update_document(node_id, text="Updated")
        ni1.close()
        
        # Second session
        ni2 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        doc = ni2.get_document(node_id)
        
        assert doc['text'] == "Updated"
        
        ni2.close()


class TestIndexRebuild:
    """Test index rebuilding."""
    
    def test_rebuild_from_storage(self, temp_path):
        """Should be able to rebuild index from storage."""
        vecs = []
        
        # First session - add documents
        ni1 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        for i in range(10):
            vec = np.random.rand(128).astype('float32')
            ni1.add_document(f"Doc {i}", vec)
            vecs.append(vec)
        ni1.close()
        
        # Second session - rebuild index
        ni2 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        ni2.rebuild_index()
        
        # Search should still work
        results = ni2.search(vecs[0], k=1)
        assert len(results) >= 1
        
        ni2.close()


class TestClearPersistence:
    """Test clear operation."""
    
    def test_clear_removes_all(self, temp_path):
        """Clear should remove all data."""
        # First session - add documents
        ni1 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        for i in range(10):
            vec = np.random.rand(128).astype('float32')
            ni1.add_document(f"Doc {i}", vec)
        
        ni1.clear()
        ni1.close()
        
        # Second session - verify empty
        ni2 = NeuroIndex(path=temp_path, dim=128, log_level="WARNING")
        
        assert ni2.get_stats()['total_documents'] == 0
        assert ni2.get_stats()['faiss_vectors'] == 0
        
        ni2.close()

