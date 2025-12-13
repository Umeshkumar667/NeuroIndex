import os
import shutil
import numpy as np
import pytest
from neuroindex import NeuroIndex

TEST_PATH = "./test_ni_store"
VECTOR_DIM = 112

@pytest.fixture(scope="module")
def ni():
    if os.path.exists(TEST_PATH):
        shutil.rmtree(TEST_PATH)
    ni_instance = NeuroIndex(path=TEST_PATH, dim=VECTOR_DIM, cache_size=100)
    yield ni_instance
    ni_instance.close()
    if os.path.exists(TEST_PATH):
        shutil.rmtree(TEST_PATH)

def test_add_and_search(ni):
    vec = np.random.rand(VECTOR_DIM).astype("float32")
    node_id = ni.add_document(text="Test document", vector=vec)
    results = ni.search(query_vector=vec, k=1)
    assert len(results) > 0
    assert results[0].node_id == node_id

def test_stats(ni):
    stats = ni.get_stats()
    assert "total_documents" in stats
    assert stats["cache_size"] <= 100
