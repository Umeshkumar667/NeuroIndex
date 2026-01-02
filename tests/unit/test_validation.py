"""
Unit tests for input validation in NeuroIndex.
"""

import pytest
import numpy as np
from neuroindex import NeuroIndex, DimensionMismatchError, InvalidInputError


class TestDimensionValidation:
    """Tests for embedding dimension validation."""
    
    def test_correct_dimension_accepted(self, ni, sample_vector):
        """Correct dimension should work."""
        node_id = ni.add_document("test", sample_vector)
        assert node_id is not None
        assert len(node_id) == 16
    
    def test_wrong_dimension_raises(self, ni):
        """Wrong dimension should raise DimensionMismatchError."""
        wrong_dim = np.random.rand(64).astype('float32')  # Expected 128
        
        with pytest.raises(DimensionMismatchError) as exc_info:
            ni.add_document("test", wrong_dim)
        
        assert exc_info.value.expected == 128
        assert exc_info.value.got == 64
    
    def test_wrong_dimension_in_search(self, ni, sample_vector):
        """Wrong dimension in search should raise."""
        ni.add_document("test", sample_vector)
        
        wrong_dim = np.random.rand(256).astype('float32')
        with pytest.raises(DimensionMismatchError):
            ni.search(wrong_dim)
    
    def test_list_input_converted(self, ni):
        """List input should be converted to numpy array."""
        vec_list = [0.1] * 128
        node_id = ni.add_document("test", vec_list)
        assert node_id is not None


class TestTextValidation:
    """Tests for text input validation."""
    
    def test_valid_text_accepted(self, ni, sample_vector):
        """Valid text should work."""
        node_id = ni.add_document("Hello world", sample_vector)
        assert node_id is not None
    
    def test_empty_text_raises(self, ni, sample_vector):
        """Empty text should raise InvalidInputError."""
        with pytest.raises(InvalidInputError):
            ni.add_document("", sample_vector)
    
    def test_whitespace_only_raises(self, ni, sample_vector):
        """Whitespace-only text should raise InvalidInputError."""
        with pytest.raises(InvalidInputError):
            ni.add_document("   \n\t  ", sample_vector)
    
    def test_text_stripped(self, ni, sample_vector):
        """Text should be stripped of leading/trailing whitespace."""
        node_id = ni.add_document("  hello world  ", sample_vector)
        doc = ni.get_document(node_id)
        assert doc['text'] == "hello world"
    
    def test_non_string_raises(self, ni, sample_vector):
        """Non-string text should raise InvalidInputError."""
        with pytest.raises(InvalidInputError):
            ni.add_document(123, sample_vector)
        
        with pytest.raises(InvalidInputError):
            ni.add_document(None, sample_vector)


class TestVectorValidation:
    """Tests for vector value validation."""
    
    def test_nan_vector_raises(self, ni):
        """Vector with NaN should raise InvalidInputError."""
        vec = np.array([np.nan] * 128, dtype='float32')
        
        with pytest.raises(InvalidInputError) as exc_info:
            ni.add_document("test", vec)
        
        assert "NaN" in str(exc_info.value)
    
    def test_inf_vector_raises(self, ni):
        """Vector with Inf should raise InvalidInputError."""
        vec = np.array([np.inf] * 128, dtype='float32')
        
        with pytest.raises(InvalidInputError) as exc_info:
            ni.add_document("test", vec)
        
        assert "Inf" in str(exc_info.value)
    
    def test_negative_inf_raises(self, ni):
        """Vector with -Inf should raise InvalidInputError."""
        vec = np.array([-np.inf] * 128, dtype='float32')
        
        with pytest.raises(InvalidInputError):
            ni.add_document("test", vec)
    
    def test_mixed_nan_inf_raises(self, ni):
        """Vector with mixed NaN and Inf should raise."""
        vec = np.random.rand(128).astype('float32')
        vec[0] = np.nan
        vec[1] = np.inf
        
        with pytest.raises(InvalidInputError):
            ni.add_document("test", vec)


class TestParameterValidation:
    """Tests for parameter validation."""
    
    def test_negative_dimension_raises(self, temp_path):
        """Negative dimension should raise."""
        with pytest.raises(InvalidInputError):
            NeuroIndex(path=temp_path, dim=-1)
    
    def test_zero_dimension_raises(self, temp_path):
        """Zero dimension should raise."""
        with pytest.raises(InvalidInputError):
            NeuroIndex(path=temp_path, dim=0)
    
    def test_negative_cache_size_raises(self, temp_path):
        """Negative cache size should raise."""
        with pytest.raises(InvalidInputError):
            NeuroIndex(path=temp_path, dim=128, cache_size=-1)
    
    def test_invalid_similarity_threshold_raises(self, temp_path):
        """Similarity threshold outside [0,1] should raise."""
        with pytest.raises(InvalidInputError):
            NeuroIndex(path=temp_path, dim=128, similarity_threshold=1.5)
        
        with pytest.raises(InvalidInputError):
            NeuroIndex(path=temp_path, dim=128, similarity_threshold=-0.5)
    
    def test_negative_k_raises(self, ni, sample_vector):
        """Negative k in search should raise."""
        ni.add_document("test", sample_vector)
        
        with pytest.raises(InvalidInputError):
            ni.search(sample_vector, k=-1)
    
    def test_zero_k_raises(self, ni, sample_vector):
        """Zero k in search should raise."""
        ni.add_document("test", sample_vector)
        
        with pytest.raises(InvalidInputError):
            ni.search(sample_vector, k=0)

