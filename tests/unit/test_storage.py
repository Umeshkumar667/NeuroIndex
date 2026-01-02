"""
Unit tests for PersistentStorage (SQLite).
"""

import pytest
import numpy as np
import time
from neuroindex.core import PersistentStorage
from neuroindex.exceptions import StorageError


class TestStorageBasics:
    """Basic storage operations."""
    
    def test_add_and_get(self, temp_path):
        """Test adding and retrieving a node."""
        storage = PersistentStorage(temp_path)
        
        node = {
            'id': 'test_id',
            'text': 'test text',
            'vector': np.random.rand(128).astype('float32'),
            'metadata': {'key': 'value'},
            'access_count': 0,
            'last_accessed': time.time(),
            'creation_time': time.time(),
            'importance_score': 1.0
        }
        
        storage.add_node(node)
        result = storage.get_node('test_id')
        
        assert result is not None
        assert result['id'] == 'test_id'
        assert result['text'] == 'test text'
        assert result['metadata'] == {'key': 'value'}
        np.testing.assert_array_almost_equal(result['vector'], node['vector'])
        
        storage.close()
    
    def test_get_nonexistent_returns_none(self, temp_path):
        """Getting nonexistent node should return None."""
        storage = PersistentStorage(temp_path)
        assert storage.get_node('nonexistent') is None
        storage.close()
    
    def test_delete_node(self, temp_path):
        """Test deleting a node."""
        storage = PersistentStorage(temp_path)
        
        node = {
            'id': 'test_id',
            'text': 'test',
            'vector': np.zeros(128, dtype='float32'),
            'metadata': {}
        }
        storage.add_node(node)
        
        assert storage.delete_node('test_id') is True
        assert storage.get_node('test_id') is None
        
        storage.close()
    
    def test_delete_nonexistent_returns_false(self, temp_path):
        """Deleting nonexistent node should return False."""
        storage = PersistentStorage(temp_path)
        assert storage.delete_node('nonexistent') is False
        storage.close()
    
    def test_node_count(self, temp_path):
        """Test node counting."""
        storage = PersistentStorage(temp_path)
        
        for i in range(10):
            node = {
                'id': f'id_{i}',
                'text': 'test',
                'vector': np.zeros(128, dtype='float32'),
                'metadata': {}
            }
            storage.add_node(node)
        
        assert storage.get_node_count() == 10
        
        storage.close()


class TestStorageUpdate:
    """Storage update operations."""
    
    def test_update_text(self, temp_path):
        """Test updating node text."""
        storage = PersistentStorage(temp_path)
        
        node = {
            'id': 'test_id',
            'text': 'original',
            'vector': np.zeros(128, dtype='float32'),
            'metadata': {}
        }
        storage.add_node(node)
        
        storage.update_node('test_id', {'text': 'updated'})
        result = storage.get_node('test_id')
        
        assert result['text'] == 'updated'
        
        storage.close()
    
    def test_update_vector(self, temp_path):
        """Test updating node vector."""
        storage = PersistentStorage(temp_path)
        
        original_vec = np.zeros(128, dtype='float32')
        new_vec = np.ones(128, dtype='float32')
        
        node = {
            'id': 'test_id',
            'text': 'test',
            'vector': original_vec,
            'metadata': {}
        }
        storage.add_node(node)
        
        storage.update_node('test_id', {'vector': new_vec})
        result = storage.get_node('test_id')
        
        np.testing.assert_array_equal(result['vector'], new_vec)
        
        storage.close()
    
    def test_update_metadata(self, temp_path):
        """Test updating node metadata."""
        storage = PersistentStorage(temp_path)
        
        node = {
            'id': 'test_id',
            'text': 'test',
            'vector': np.zeros(128, dtype='float32'),
            'metadata': {'old': 'value'}
        }
        storage.add_node(node)
        
        storage.update_node('test_id', {'metadata': {'new': 'data'}})
        result = storage.get_node('test_id')
        
        assert result['metadata'] == {'new': 'data'}
        
        storage.close()
    
    def test_update_access(self, temp_path):
        """Test access count update."""
        storage = PersistentStorage(temp_path)
        
        node = {
            'id': 'test_id',
            'text': 'test',
            'vector': np.zeros(128, dtype='float32'),
            'metadata': {},
            'access_count': 0
        }
        storage.add_node(node)
        
        storage.update_access('test_id')
        storage.update_access('test_id')
        
        result = storage.get_node('test_id')
        assert result['access_count'] == 2
        
        storage.close()
    
    def test_update_nonexistent_returns_false(self, temp_path):
        """Updating nonexistent node should return False."""
        storage = PersistentStorage(temp_path)
        assert storage.update_node('nonexistent', {'text': 'new'}) is False
        storage.close()


class TestStorageIteration:
    """Storage iteration functionality."""
    
    def test_iterate_all(self, temp_path):
        """Test iterating over all nodes."""
        storage = PersistentStorage(temp_path)
        
        for i in range(25):
            node = {
                'id': f'id_{i}',
                'text': f'text_{i}',
                'vector': np.random.rand(128).astype('float32'),
                'metadata': {'index': i}
            }
            storage.add_node(node)
        
        nodes = list(storage.iterate_all())
        assert len(nodes) == 25
        
        ids = {n['id'] for n in nodes}
        for i in range(25):
            assert f'id_{i}' in ids
        
        storage.close()
    
    def test_iterate_empty(self, temp_path):
        """Iterating empty storage should yield nothing."""
        storage = PersistentStorage(temp_path)
        nodes = list(storage.iterate_all())
        assert nodes == []
        storage.close()


class TestStoragePersistence:
    """Storage persistence across restarts."""
    
    def test_data_survives_restart(self, temp_path):
        """Data should persist when storage is reopened."""
        # First session
        storage1 = PersistentStorage(temp_path)
        vec = np.random.rand(128).astype('float32')
        
        node = {
            'id': 'persist_test',
            'text': 'persistent data',
            'vector': vec,
            'metadata': {'important': True}
        }
        storage1.add_node(node)
        storage1.close()
        
        # Second session
        storage2 = PersistentStorage(temp_path)
        result = storage2.get_node('persist_test')
        
        assert result is not None
        assert result['text'] == 'persistent data'
        assert result['metadata'] == {'important': True}
        np.testing.assert_array_almost_equal(result['vector'], vec)
        
        storage2.close()


class TestStorageSpecialCases:
    """Edge cases and special scenarios."""
    
    def test_unicode_text(self, temp_path):
        """Storage should handle unicode text."""
        storage = PersistentStorage(temp_path)
        
        node = {
            'id': 'unicode_test',
            'text': '你好世界 🌍 مرحبا',
            'vector': np.zeros(128, dtype='float32'),
            'metadata': {}
        }
        storage.add_node(node)
        
        result = storage.get_node('unicode_test')
        assert result['text'] == '你好世界 🌍 مرحبا'
        
        storage.close()
    
    def test_large_metadata(self, temp_path):
        """Storage should handle large metadata."""
        storage = PersistentStorage(temp_path)
        
        large_metadata = {f'key_{i}': f'value_{i}' * 100 for i in range(100)}
        
        node = {
            'id': 'large_meta',
            'text': 'test',
            'vector': np.zeros(128, dtype='float32'),
            'metadata': large_metadata
        }
        storage.add_node(node)
        
        result = storage.get_node('large_meta')
        assert result['metadata'] == large_metadata
        
        storage.close()
    
    def test_replace_existing(self, temp_path):
        """Adding with same ID should replace existing."""
        storage = PersistentStorage(temp_path)
        
        node1 = {
            'id': 'same_id',
            'text': 'original',
            'vector': np.zeros(128, dtype='float32'),
            'metadata': {}
        }
        storage.add_node(node1)
        
        node2 = {
            'id': 'same_id',
            'text': 'replaced',
            'vector': np.ones(128, dtype='float32'),
            'metadata': {}
        }
        storage.add_node(node2)
        
        result = storage.get_node('same_id')
        assert result['text'] == 'replaced'
        assert storage.get_node_count() == 1
        
        storage.close()

