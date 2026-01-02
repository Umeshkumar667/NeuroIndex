"""
Pytest configuration and shared fixtures for NeuroIndex tests.
"""

import os
import shutil
import tempfile
import pytest
import numpy as np


@pytest.fixture(scope="function")
def temp_path():
    """Create a temporary directory for test storage."""
    path = tempfile.mkdtemp(prefix="neuroindex_test_")
    yield path
    # Cleanup
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="function")
def ni(temp_path):
    """Create a NeuroIndex instance for testing."""
    from neuroindex import NeuroIndex
    
    index = NeuroIndex(path=temp_path, dim=128, cache_size=100, log_level="WARNING")
    yield index
    index.close()


@pytest.fixture(scope="function")
def ni_with_docs(ni):
    """NeuroIndex with some pre-loaded documents."""
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "Neural networks are inspired by the human brain",
        "Vector databases store embeddings for similarity search",
    ]
    
    for i, text in enumerate(texts):
        vec = np.random.rand(128).astype('float32')
        ni.add_document(text, vec, metadata={"index": i})
    
    return ni


@pytest.fixture
def sample_vector():
    """Generate a random sample vector."""
    return np.random.rand(128).astype('float32')


@pytest.fixture
def sample_vectors():
    """Generate multiple random sample vectors."""
    return np.random.rand(100, 128).astype('float32')


@pytest.fixture
def embed_fn():
    """Simple deterministic embedding function for testing."""
    def _embed(text: str) -> np.ndarray:
        # Create deterministic embedding from text hash
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(128).astype('float32')
    return _embed

