"""
NeuroIndex - Adaptive Concept-Graph Memory System
Hybrid Vector + Graph Memory System
"""

import os
import time
import pickle
import hashlib
import sqlite3
import threading
import queue
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
import faiss
import networkx as nx

# ---------------------------
# Search Result Dataclass
# ---------------------------
@dataclass
class SearchResult:
    node_id: str
    text: str
    similarity: float
    metadata: Dict[str, Any]
    source: str  # 'cache', 'vector', 'graph'

# ---------------------------
# Bloom Filter for duplicates
# ---------------------------
class BloomFilter:
    def __init__(self, capacity: int = 1000000, error_rate: float = 0.1):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_array_size = int(-capacity * np.log(error_rate) / (np.log(2) ** 2))
        self.hash_count = int(self.bit_array_size * np.log(2) / capacity)
        self.bit_array = np.zeros(self.bit_array_size, dtype=bool)

    def _hash(self, item: str, seed: int) -> int:
        return hash(f"{item}_{seed}") % self.bit_array_size

    def add(self, item: str):
        for i in range(self.hash_count):
            self.bit_array[self._hash(item, i)] = True

    def contains(self, item: str) -> bool:
        return all(self.bit_array[self._hash(item, i)] for i in range(self.hash_count))

# ---------------------------
# RAM Cache
# ---------------------------
class NeuroCache:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: Dict[str, Dict] = {}
        self.access_order: List[str] = []
        self.lock = threading.Lock()

    def add(self, node_id: str, node: Dict):
        with self.lock:
            if node_id in self.cache:
                self.access_order.remove(node_id)
            elif len(self.cache) >= self.max_size:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]

            self.cache[node_id] = node
            self.access_order.append(node_id)

    def get(self, node_id: str) -> Optional[Dict]:
        with self.lock:
            if node_id in self.cache:
                self.access_order.remove(node_id)
                self.access_order.append(node_id)
                return self.cache[node_id]
        return None

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[SearchResult]:
        results = []
        with self.lock:
            for node_id, node in self.cache.items():
                node_vector = node['vector']
                similarity = np.dot(query_vector, node_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(node_vector)
                )
                results.append(SearchResult(
                    node_id=node_id,
                    text=node['text'],
                    similarity=similarity,
                    metadata=node['metadata'],
                    source='cache'
                ))
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]

# ---------------------------
# Semantic Graph
# ---------------------------
class SemanticGraph:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.graph_file = os.path.join(storage_path, "semantic_graph.pkl")
        self.graph = nx.Graph()
        self.load()

    def add_node(self, node_id: str, vector: np.ndarray, metadata: Dict):
        self.graph.add_node(node_id, vector=vector, **metadata)
        similar_nodes = self._find_similar_nodes(vector, threshold=0.7, max_edges=10)
        for sid, sim in similar_nodes:
            if sid != node_id:
                self.graph.add_edge(node_id, sid, weight=sim)

    def _find_similar_nodes(self, query_vector: np.ndarray,
                            threshold: float = 0.7,
                            max_edges: int = 10) -> List[Tuple[str, float]]:
        sims = []
        for nid in self.graph.nodes():
            node_data = self.graph.nodes[nid]
            if 'vector' in node_data:
                sim = np.dot(query_vector, node_data['vector']) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(node_data['vector'])
                )
                if sim >= threshold:
                    sims.append((nid, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:max_edges]

    def get_neighbors(self, node_id: str) -> List[Tuple[str, float]]:
        if node_id not in self.graph:
            return []
        neighbors = []
        for nid in self.graph.neighbors(node_id):
            weight = self.graph.edges[node_id, nid].get('weight', 0.5)
            neighbors.append((nid, weight))
        return neighbors

    def search_by_traversal(self, query_vector: np.ndarray, k: int = 10) -> List[str]:
        start_nodes = self._find_similar_nodes(query_vector, threshold=0.5, max_edges=5)
        if not start_nodes:
            return []
        visited, candidates = set(), []
        for start_id, sim in start_nodes:
            if start_id in visited:
                continue
            queue_ = [(start_id, sim)]
            while queue_ and len(candidates) < k*2:
                nid, sim_val = queue_.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                candidates.append((nid, sim_val))
                for neighbor, weight in self.get_neighbors(nid):
                    if neighbor not in visited:
                        queue_.append((neighbor, sim_val*weight*0.9))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in candidates[:k]]

    def save(self):
        os.makedirs(self.storage_path, exist_ok=True)
        with open(self.graph_file, 'wb') as f:
            pickle.dump(self.graph, f)

    def load(self):
        if os.path.exists(self.graph_file):
            try:
                with open(self.graph_file, 'rb') as f:
                    self.graph = pickle.load(f)
            except Exception:
                self.graph = nx.Graph()

# ---------------------------
# Persistent Storage
# ---------------------------
class PersistentStorage:
    def __init__(self, path: str):
        self.db_path = os.path.join(path, "nodes.db")
        os.makedirs(path, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                vector BLOB,
                text TEXT,
                metadata BLOB,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                creation_time REAL,
                importance_score REAL DEFAULT 1.0
            )
        ''')
        conn.commit()
        conn.close()

    def add_node(self, node: Dict):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO nodes
            (id, vector, text, metadata, access_count, last_accessed, creation_time, importance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            node['id'],
            pickle.dumps(node['vector']),
            node['text'],
            pickle.dumps(node['metadata']),
            node.get('access_count', 0),
            node.get('last_accessed', time.time()),
            node.get('creation_time', time.time()),
            node.get('importance_score', 1.0)
        ))
        conn.commit()
        conn.close()

    def get_node(self, node_id: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM nodes WHERE id=?', (node_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return {
                'id': row[0],
                'vector': pickle.loads(row[1]),
                'text': row[2],
                'metadata': pickle.loads(row[3]),
                'access_count': row[4],
                'last_accessed': row[5],
                'creation_time': row[6],
                'importance_score': row[7]
            }
        return None

    def update_access(self, node_id: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            UPDATE nodes SET access_count=access_count+1, last_accessed=?
            WHERE id=?
        ''', (time.time(), node_id))
        conn.commit()
        conn.close()

    def get_node_count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM nodes')
        count = c.fetchone()[0]
        conn.close()
        return count

    def iterate_vectors(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT id, vector FROM nodes')
        while True:
            rows = c.fetchmany(1000)
            if not rows:
                break
            for nid, vec_blob in rows:
                yield nid, pickle.loads(vec_blob)
        conn.close()

# ---------------------------
# NeuroIndex Main Class
# ---------------------------
class NeuroIndex:
    def __init__(self, path: str = './neuroindex_data', dim: int = 112, cache_size: int = 10000):
        self.path = path
        self.dim = dim
        self.cache = NeuroCache(max_size=cache_size)
        self.graph = SemanticGraph(path)
        self.storage = PersistentStorage(path)
        self.bloom = BloomFilter()
        self.vector_index = faiss.IndexFlatL2(dim)
        self.index_trained = False
        self.node_id_to_idx = {}
        self.idx_to_node_id = {}
        self.update_queue = queue.Queue()
        self.running = True
        self.bg_thread = threading.Thread(target=self._bg_worker, daemon=True)
        self.bg_thread.start()

    # ---------------------------
    # Background worker
    # ---------------------------
    def _bg_worker(self):
        while self.running:
            try:
                task = self.update_queue.get(timeout=1.0)
                if task[0] == 'save_graph':
                    self.graph.save()
                self.update_queue.task_done()
            except queue.Empty:
                continue

    # ---------------------------
    # Add document
    # ---------------------------
    def add_document(self, text: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> str:
        node_id = hashlib.sha256(f"{text}_{vector.tobytes()}".encode()).hexdigest()[:16]
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if self.bloom.contains(text_hash):
            existing = self.storage.get_node(node_id)
            if existing:
                return node_id
        self.bloom.add(text_hash)
        node = {
            'id': node_id,
            'text': text,
            'vector': vector.astype("float32"),
            'metadata': metadata or {},
            'access_count': 0,
            'last_accessed': time.time(),
            'creation_time': time.time(),
            'importance_score': 1.0
        }
        self.storage.add_node(node)
        self.graph.add_node(node_id, vector, node['metadata'])
        self.cache.add(node_id, node)
        self.update_queue.put(('save_graph',))
        return node_id

    # ---------------------------
    # Search
    # ---------------------------
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[SearchResult]:
        results = []
        results.extend(self.cache.search(query_vector, k=k//2))
        graph_ids = self.graph.search_by_traversal(query_vector, k=k)
        for nid in graph_ids:
            node = self.storage.get_node(nid)
            if node:
                sim = np.dot(query_vector, node['vector']) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(node['vector'])
                )
                results.append(SearchResult(
                    node_id=nid,
                    text=node['text'],
                    similarity=sim,
                    metadata=node['metadata'],
                    source='graph'
                ))
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]
    def search_text(self, text: str, embed_fn, k: int = 5):
        """
        Search using raw text instead of a precomputed vector.

        Parameters
        ----------
        text : str
            Input query text.
        embed_fn : Callable[[str], np.ndarray]
            Function that converts text -> embedding vector.
        k : int
            Number of results to return.

        Returns
        -------
        List[SearchResult]
        """
        vector = embed_fn(text)
        return self.search(query_vector=vector, k=k)
    
    def get_stats(self):
        return {
            'total_documents': self.storage.get_node_count(),
            'cache_size': len(self.cache.cache),
            'graph_nodes': self.graph.graph.number_of_nodes(),
            'graph_edges': self.graph.graph.number_of_edges()
        }

    def close(self):
        self.running = False
        self.bg_thread.join(timeout=5)
        self.graph.save()
