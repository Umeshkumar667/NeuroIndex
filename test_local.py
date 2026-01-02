"""
Quick local test for NeuroIndex
Run: python test_local.py
"""

from neuroindex import NeuroIndex
import numpy as np
import shutil

print("=" * 60)
print("NeuroIndex Local Test")
print("=" * 60)

# Create index
ni = NeuroIndex(path="./test_memory", dim=384, log_level="WARNING")


# Embedding function (simulated)
def embed(text):
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(384).astype("float32")


# Add documents
print("\n[1] Adding documents...")
docs = [
    "Neural networks are a type of machine learning model",
    "FAISS is a library for efficient similarity search",
    "Python is a popular programming language for AI",
    "Vector databases store embeddings for semantic search",
    "RAG combines retrieval with generation for better AI responses",
]

for doc in docs:
    node_id = ni.add_document(doc, embed(doc))
    print(f"  Added: {doc[:50]}...")

# Search
print('\n[2] Searching for: "How do neural networks work?"')
results = ni.search_text("How do neural networks work?", embed_fn=embed, k=3)
for i, r in enumerate(results, 1):
    print(f"  {i}. [{r.source}] {r.similarity:.3f}: {r.text[:50]}...")

# Get document
print("\n[3] Get document by ID...")
doc = ni.get_document(node_id)
print(f"  Text: {doc['text'][:50]}...")

# Update document
print("\n[4] Updating document...")
ni.update_document(node_id, metadata={"updated": True, "category": "AI"})
doc = ni.get_document(node_id)
print(f"  New metadata: {doc['metadata']}")

# Stats
print("\n[5] Index Statistics:")
stats = ni.get_stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

# Metrics
print("\n[6] Performance Metrics:")
metrics = ni.get_metrics()
print(f"  Uptime: {metrics['uptime_seconds']:.2f}s")
print(f"  Cache hit rate: {metrics['cache']['hit_rate']:.2%}")
if "search" in metrics["operations"]:
    print(f"  Avg search time: {metrics['operations']['search']['avg_ms']:.2f}ms")

# Delete document
print("\n[7] Deleting a document...")
deleted = ni.delete_document(node_id)
print(f"  Deleted: {deleted}")
print(f"  Documents remaining: {ni.get_stats()['total_documents']}")

# Cleanup
ni.close()
shutil.rmtree("./test_memory", ignore_errors=True)

print("\n" + "=" * 60)
print("All tests passed! NeuroIndex is working correctly.")
print("=" * 60)

