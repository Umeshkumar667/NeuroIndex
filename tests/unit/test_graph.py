"""
Unit tests for SemanticGraph.
"""

import pytest
import numpy as np
from neuroindex.core import SemanticGraph


class TestGraphBasics:
    """Basic graph operations."""
    
    def test_add_node(self, temp_path):
        """Test adding a node."""
        graph = SemanticGraph(temp_path)
        
        vec = np.random.rand(128).astype('float32')
        graph.add_node('test_id', vec, {'key': 'value'})
        
        assert graph.node_count() == 1
    
    def test_remove_node(self, temp_path):
        """Test removing a node."""
        graph = SemanticGraph(temp_path)
        
        vec = np.random.rand(128).astype('float32')
        graph.add_node('test_id', vec, {})
        
        assert graph.remove_node('test_id') is True
        assert graph.node_count() == 0
    
    def test_remove_nonexistent(self, temp_path):
        """Removing nonexistent should return False."""
        graph = SemanticGraph(temp_path)
        assert graph.remove_node('nonexistent') is False
    
    def test_edge_count(self, temp_path):
        """Test edge counting."""
        graph = SemanticGraph(temp_path, similarity_threshold=0.5)
        
        # Add similar vectors (should create edges)
        base = np.random.rand(128).astype('float32')
        graph.add_node('node_0', base, {})
        graph.add_node('node_1', base + 0.01, {})  # Very similar
        
        assert graph.edge_count() >= 1


class TestGraphEdgeCreation:
    """Edge creation based on similarity."""
    
    def test_similar_nodes_connected(self, temp_path):
        """Similar nodes should be connected."""
        graph = SemanticGraph(temp_path, similarity_threshold=0.9)
        
        vec = np.ones(128, dtype='float32')
        graph.add_node('node_1', vec, {})
        graph.add_node('node_2', vec * 0.99, {})  # Very similar
        
        neighbors = graph.get_neighbors('node_1')
        neighbor_ids = [n[0] for n in neighbors]
        
        assert 'node_2' in neighbor_ids
    
    def test_dissimilar_nodes_not_connected(self, temp_path):
        """Dissimilar nodes should not be connected."""
        graph = SemanticGraph(temp_path, similarity_threshold=0.9)
        
        vec1 = np.zeros(128, dtype='float32')
        vec1[0] = 1.0
        
        vec2 = np.zeros(128, dtype='float32')
        vec2[1] = 1.0  # Orthogonal
        
        graph.add_node('node_1', vec1, {})
        graph.add_node('node_2', vec2, {})
        
        neighbors = graph.get_neighbors('node_1')
        neighbor_ids = [n[0] for n in neighbors]
        
        assert 'node_2' not in neighbor_ids
    
    def test_max_edges_limit(self, temp_path):
        """Should not exceed max_edges."""
        graph = SemanticGraph(temp_path, similarity_threshold=0.0, max_edges=3)
        
        # Add many similar nodes
        base = np.random.rand(128).astype('float32')
        for i in range(10):
            graph.add_node(f'node_{i}', base + i * 0.001, {})
        
        # Last node should have at most max_edges neighbors
        neighbors = graph.get_neighbors('node_9')
        assert len(neighbors) <= 3


class TestGraphTraversal:
    """Graph traversal search."""
    
    def test_traversal_finds_connected_nodes(self, temp_path):
        """Traversal should find connected nodes."""
        graph = SemanticGraph(temp_path, similarity_threshold=0.5)
        
        # Create a chain of similar nodes
        base = np.random.rand(128).astype('float32')
        for i in range(5):
            graph.add_node(f'node_{i}', base + i * 0.01, {})
        
        results = graph.search_by_traversal(base, k=5)
        
        assert len(results) > 0
    
    def test_traversal_returns_empty_for_no_matches(self, temp_path):
        """Traversal should return empty if no matches."""
        graph = SemanticGraph(temp_path, similarity_threshold=0.99)
        
        vec1 = np.zeros(128, dtype='float32')
        vec1[0] = 1.0
        graph.add_node('node_1', vec1, {})
        
        # Query with orthogonal vector
        query = np.zeros(128, dtype='float32')
        query[1] = 1.0
        
        results = graph.search_by_traversal(query, k=5)
        assert len(results) == 0


class TestGraphPersistence:
    """Graph persistence."""
    
    def test_save_and_load(self, temp_path):
        """Graph should persist after save/load."""
        # Create and save
        graph1 = SemanticGraph(temp_path)
        
        for i in range(5):
            vec = np.random.rand(128).astype('float32')
            graph1.add_node(f'node_{i}', vec, {'index': i})
        
        graph1.save()
        
        # Load in new instance
        graph2 = SemanticGraph(temp_path)
        
        assert graph2.node_count() == 5
    
    def test_edges_persist(self, temp_path):
        """Edges should persist correctly."""
        # Create graph with edges
        graph1 = SemanticGraph(temp_path, similarity_threshold=0.5)
        
        base = np.random.rand(128).astype('float32')
        graph1.add_node('node_0', base, {})
        graph1.add_node('node_1', base + 0.01, {})
        
        edge_count = graph1.edge_count()
        graph1.save()
        
        # Load and verify
        graph2 = SemanticGraph(temp_path)
        assert graph2.edge_count() == edge_count

