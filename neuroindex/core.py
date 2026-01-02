"""
NeuroIndex - Production-Ready Hybrid Vector + Semantic Graph Memory System

A high-performance memory system combining:
- FAISS vector search for fast similarity matching
- Semantic graph traversal for relationship-aware retrieval  
- LRU cache for frequently accessed items
- SQLite persistence for durability

Author: Umeshkumar Pal
License: MIT
"""

import os
import time
import pickle
import hashlib
import sqlite3
import logging
import threading
import queue
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import faiss
import networkx as nx

from .exceptions import (
    NeuroIndexError,
    DimensionMismatchError,
    StorageError,
    IndexCorruptedError,
    DocumentNotFoundError,
    InvalidInputError,
)
from .metrics import MetricsCollector

# Configure module logger
logger = logging.getLogger(__name__)


# ---------------------------
# Search Result Dataclass
# ---------------------------
@dataclass
class SearchResult:
    """
    Represents a single search result.
    
    Attributes:
        node_id: Unique identifier for the document
        text: The document text content
        similarity: Cosine similarity score (0-1)
        metadata: User-provided metadata dict
        source: Where the result came from ('cache', 'faiss', 'graph')
    """
    node_id: str
    text: str
    similarity: float
    metadata: Dict[str, Any]
    source: str  # 'cache', 'faiss', 'graph'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.node_id,
            "text": self.text,
            "similarity": self.similarity,
            "metadata": self.metadata,
            "source": self.source,
        }


# ---------------------------
# Bloom Filter for duplicates
# ---------------------------
class BloomFilter:
    """
    Probabilistic data structure for fast duplicate detection.
    
    Used to quickly check if a document might already exist before
    doing expensive database lookups.
    """
    
    def __init__(self, capacity: int = 1000000, error_rate: float = 0.01):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_array_size = int(-capacity * np.log(error_rate) / (np.log(2) ** 2))
        self.hash_count = max(1, int(self.bit_array_size * np.log(2) / capacity))
        self.bit_array = np.zeros(self.bit_array_size, dtype=bool)
        self._lock = threading.Lock()
    
    def _hash(self, item: str, seed: int) -> int:
        """Generate hash for item with given seed."""
        return int(hashlib.md5(f"{item}_{seed}".encode()).hexdigest(), 16) % self.bit_array_size
    
    def add(self, item: str) -> None:
        """Add item to the filter."""
        with self._lock:
            for i in range(self.hash_count):
                self.bit_array[self._hash(item, i)] = True
    
    def contains(self, item: str) -> bool:
        """Check if item might be in the filter (may have false positives)."""
        with self._lock:
            return all(self.bit_array[self._hash(item, i)] for i in range(self.hash_count))
    
    def clear(self) -> None:
        """Clear all items from the filter."""
        with self._lock:
            self.bit_array.fill(False)


# ---------------------------
# RAM Cache (LRU)
# ---------------------------
class NeuroCache:
    """
    Thread-safe LRU cache for frequently accessed documents.
    
    Provides O(1) access to hot documents without database lookups.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: Dict[str, Dict] = {}
        self.access_order: List[str] = []
        self._lock = threading.RLock()
    
    def add(self, node_id: str, node: Dict) -> None:
        """Add or update a node in the cache."""
        with self._lock:
            if node_id in self.cache:
                self.access_order.remove(node_id)
            elif len(self.cache) >= self.max_size:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            
            self.cache[node_id] = node
            self.access_order.append(node_id)
    
    def get(self, node_id: str) -> Optional[Dict]:
        """Get a node from cache, updating access order."""
        with self._lock:
            if node_id in self.cache:
                self.access_order.remove(node_id)
                self.access_order.append(node_id)
                return self.cache[node_id]
        return None
    
    def remove(self, node_id: str) -> bool:
        """Remove a node from the cache."""
        with self._lock:
            if node_id in self.cache:
                del self.cache[node_id]
                self.access_order.remove(node_id)
                return True
        return False
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[SearchResult]:
        """Search cache for similar documents."""
        results = []
        query_norm = np.linalg.norm(query_vector)
        
        if query_norm == 0:
            return results
        
        with self._lock:
            for node_id, node in self.cache.items():
                node_vector = node['vector']
                node_norm = np.linalg.norm(node_vector)
                
                if node_norm == 0:
                    continue
                
                similarity = float(np.dot(query_vector, node_vector) / (query_norm * node_norm))
                results.append(SearchResult(
                    node_id=node_id,
                    text=node['text'],
                    similarity=similarity,
                    metadata=node['metadata'],
                    source='cache'
                ))
        
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
    
    def __len__(self) -> int:
        with self._lock:
            return len(self.cache)


# ---------------------------
# Semantic Graph
# ---------------------------
class SemanticGraph:
    """
    Graph structure for relationship-aware retrieval.
    
    Maintains semantic connections between documents based on
    embedding similarity, enabling associative recall.
    """
    
    def __init__(self, storage_path: str, similarity_threshold: float = 0.7, max_edges: int = 10):
        self.storage_path = storage_path
        self.graph_file = os.path.join(storage_path, "semantic_graph.pkl")
        self.similarity_threshold = similarity_threshold
        self.max_edges = max_edges
        self.graph = nx.Graph()
        self._lock = threading.RLock()
        self.load()
    
    def add_node(self, node_id: str, vector: np.ndarray, metadata: Dict) -> None:
        """Add a node and create edges to similar existing nodes."""
        with self._lock:
            self.graph.add_node(node_id, vector=vector, **metadata)
            similar_nodes = self._find_similar_nodes(vector)
            
            for sid, sim in similar_nodes:
                if sid != node_id:
                    self.graph.add_edge(node_id, sid, weight=sim)
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        with self._lock:
            if node_id in self.graph:
                self.graph.remove_node(node_id)
                return True
        return False
    
    def _find_similar_nodes(self, query_vector: np.ndarray) -> List[Tuple[str, float]]:
        """Find nodes similar to query vector above threshold."""
        sims = []
        query_norm = np.linalg.norm(query_vector)
        
        if query_norm == 0:
            return sims
        
        for nid in self.graph.nodes():
            node_data = self.graph.nodes[nid]
            if 'vector' in node_data:
                node_vector = node_data['vector']
                node_norm = np.linalg.norm(node_vector)
                
                if node_norm == 0:
                    continue
                
                sim = float(np.dot(query_vector, node_vector) / (query_norm * node_norm))
                if sim >= self.similarity_threshold:
                    sims.append((nid, sim))
        
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:self.max_edges]
    
    def get_neighbors(self, node_id: str) -> List[Tuple[str, float]]:
        """Get neighbors of a node with edge weights."""
        with self._lock:
            if node_id not in self.graph:
                return []
            
            neighbors = []
            for nid in self.graph.neighbors(node_id):
                weight = self.graph.edges[node_id, nid].get('weight', 0.5)
                neighbors.append((nid, weight))
            return neighbors
    
    def search_by_traversal(self, query_vector: np.ndarray, k: int = 10) -> List[str]:
        """
        Search by traversing the graph from similar entry points.
        
        Uses BFS-style traversal weighted by similarity.
        """
        with self._lock:
            start_nodes = self._find_similar_nodes(query_vector)
            if not start_nodes:
                return []
            
            visited = set()
            candidates = []
            
            for start_id, sim in start_nodes:
                if start_id in visited:
                    continue
                
                queue_ = [(start_id, sim)]
                while queue_ and len(candidates) < k * 2:
                    nid, sim_val = queue_.pop(0)
                    if nid in visited:
                        continue
                    
                    visited.add(nid)
                    candidates.append((nid, sim_val))
                    
                    for neighbor, weight in self.get_neighbors(nid):
                        if neighbor not in visited:
                            queue_.append((neighbor, sim_val * weight * 0.9))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [nid for nid, _ in candidates[:k]]
    
    def save(self) -> None:
        """Persist graph to disk."""
        with self._lock:
            os.makedirs(self.storage_path, exist_ok=True)
            try:
                with open(self.graph_file, 'wb') as f:
                    pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.debug(f"Saved semantic graph with {self.graph.number_of_nodes()} nodes")
            except Exception as e:
                logger.error(f"Failed to save semantic graph: {e}")
                raise StorageError(f"Failed to save semantic graph: {e}")
    
    def load(self) -> None:
        """Load graph from disk."""
        with self._lock:
            if os.path.exists(self.graph_file):
                try:
                    with open(self.graph_file, 'rb') as f:
                        self.graph = pickle.load(f)
                    logger.debug(f"Loaded semantic graph with {self.graph.number_of_nodes()} nodes")
                except Exception as e:
                    logger.warning(f"Failed to load semantic graph, starting fresh: {e}")
                    self.graph = nx.Graph()
            else:
                self.graph = nx.Graph()
    
    def node_count(self) -> int:
        with self._lock:
            return self.graph.number_of_nodes()
    
    def edge_count(self) -> int:
        with self._lock:
            return self.graph.number_of_edges()


# ---------------------------
# Persistent Storage (SQLite)
# ---------------------------
class PersistentStorage:
    """
    SQLite-based persistent storage for documents.
    
    Provides durable storage with proper connection handling
    and transaction management.
    """
    
    def __init__(self, path: str):
        self.path = path
        self.db_path = os.path.join(path, "nodes.db")
        self._local = threading.local()
        os.makedirs(path, exist_ok=True)
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, timeout=30.0)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        c = conn.cursor()
        
        # Main nodes table
        c.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                text TEXT NOT NULL,
                metadata BLOB,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                creation_time REAL,
                importance_score REAL DEFAULT 1.0
            )
        ''')
        
        # Metadata table for versioning
        c.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        # Insert version if not exists
        c.execute('''
            INSERT OR IGNORE INTO metadata (key, value) VALUES ('version', '0.2.0')
        ''')
        
        # Create index for faster lookups
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_nodes_creation_time ON nodes(creation_time)
        ''')
        
        conn.commit()
        logger.debug("Database initialized successfully")
    
    def add_node(self, node: Dict) -> None:
        """Add or update a node in storage."""
        conn = self._get_connection()
        c = conn.cursor()
        
        try:
            c.execute('''
                INSERT OR REPLACE INTO nodes
                (id, vector, text, metadata, access_count, last_accessed, creation_time, importance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                node['id'],
                pickle.dumps(node['vector'], protocol=pickle.HIGHEST_PROTOCOL),
                node['text'],
                pickle.dumps(node.get('metadata', {}), protocol=pickle.HIGHEST_PROTOCOL),
                node.get('access_count', 0),
                node.get('last_accessed', time.time()),
                node.get('creation_time', time.time()),
                node.get('importance_score', 1.0)
            ))
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise StorageError(f"Failed to add node: {e}")
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get a node by ID."""
        conn = self._get_connection()
        c = conn.cursor()
        
        c.execute('SELECT * FROM nodes WHERE id=?', (node_id,))
        row = c.fetchone()
        
        if row:
            return {
                'id': row[0],
                'vector': pickle.loads(row[1]),
                'text': row[2],
                'metadata': pickle.loads(row[3]) if row[3] else {},
                'access_count': row[4],
                'last_accessed': row[5],
                'creation_time': row[6],
                'importance_score': row[7]
            }
        return None
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node by ID."""
        conn = self._get_connection()
        c = conn.cursor()
        
        try:
            c.execute('DELETE FROM nodes WHERE id=?', (node_id,))
            conn.commit()
            return c.rowcount > 0
        except sqlite3.Error as e:
            conn.rollback()
            raise StorageError(f"Failed to delete node: {e}")
    
    def update_node(self, node_id: str, updates: Dict) -> bool:
        """Update specific fields of a node."""
        conn = self._get_connection()
        c = conn.cursor()
        
        # Build dynamic update query
        set_clauses = []
        values = []
        
        if 'text' in updates:
            set_clauses.append('text=?')
            values.append(updates['text'])
        
        if 'vector' in updates:
            set_clauses.append('vector=?')
            values.append(pickle.dumps(updates['vector'], protocol=pickle.HIGHEST_PROTOCOL))
        
        if 'metadata' in updates:
            set_clauses.append('metadata=?')
            values.append(pickle.dumps(updates['metadata'], protocol=pickle.HIGHEST_PROTOCOL))
        
        if 'importance_score' in updates:
            set_clauses.append('importance_score=?')
            values.append(updates['importance_score'])
        
        if not set_clauses:
            return False
        
        values.append(node_id)
        query = f"UPDATE nodes SET {', '.join(set_clauses)} WHERE id=?"
        
        try:
            c.execute(query, values)
            conn.commit()
            return c.rowcount > 0
        except sqlite3.Error as e:
            conn.rollback()
            raise StorageError(f"Failed to update node: {e}")
    
    def update_access(self, node_id: str) -> None:
        """Update access count and timestamp for a node."""
        conn = self._get_connection()
        c = conn.cursor()
        
        try:
            c.execute('''
                UPDATE nodes SET access_count=access_count+1, last_accessed=?
                WHERE id=?
            ''', (time.time(), node_id))
            conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"Failed to update access stats: {e}")
    
    def get_node_count(self) -> int:
        """Get total number of nodes."""
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM nodes')
        return c.fetchone()[0]
    
    def iterate_all(self, batch_size: int = 1000):
        """Iterate over all nodes in batches."""
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT id, vector, text, metadata FROM nodes')
        
        while True:
            rows = c.fetchmany(batch_size)
            if not rows:
                break
            
            for row in rows:
                yield {
                    'id': row[0],
                    'vector': pickle.loads(row[1]),
                    'text': row[2],
                    'metadata': pickle.loads(row[3]) if row[3] else {}
                }
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# ---------------------------
# FAISS Index Manager
# ---------------------------
class FAISSIndexManager:
    """
    Manages FAISS index for fast vector similarity search.
    
    Handles index creation, persistence, and rebuilding.
    """
    
    def __init__(self, path: str, dim: int, use_gpu: bool = False):
        self.path = path
        self.dim = dim
        self.index_file = os.path.join(path, "faiss.index")
        self.mapping_file = os.path.join(path, "faiss_mapping.pkl")
        self.use_gpu = use_gpu
        
        self._lock = threading.RLock()
        self.index: Optional[faiss.Index] = None
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self._next_idx = 0
        
        self._load_or_create()
    
    def _load_or_create(self) -> None:
        """Load existing index or create new one."""
        with self._lock:
            if os.path.exists(self.index_file) and os.path.exists(self.mapping_file):
                try:
                    self.index = faiss.read_index(self.index_file)
                    with open(self.mapping_file, 'rb') as f:
                        data = pickle.load(f)
                        self.id_to_idx = data['id_to_idx']
                        self.idx_to_id = data['idx_to_id']
                        self._next_idx = data['next_idx']
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                except Exception as e:
                    logger.warning(f"Failed to load FAISS index: {e}. Creating new index.")
                    self._create_new_index()
            else:
                self._create_new_index()
    
    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        # Use IndexFlatIP for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_to_idx = {}
        self.idx_to_id = {}
        self._next_idx = 0
        logger.debug(f"Created new FAISS index with dim={self.dim}")
    
    def add(self, node_id: str, vector: np.ndarray) -> None:
        """Add a vector to the index."""
        with self._lock:
            # Normalize vector for cosine similarity
            vector = vector.astype(np.float32)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            # Add to FAISS
            self.index.add(vector.reshape(1, -1))
            
            # Update mappings
            idx = self._next_idx
            self.id_to_idx[node_id] = idx
            self.idx_to_id[idx] = node_id
            self._next_idx += 1
    
    def remove(self, node_id: str) -> bool:
        """
        Mark a vector as removed (FAISS doesn't support true deletion).
        
        For now, we just remove from mappings. Full rebuild needed for cleanup.
        """
        with self._lock:
            if node_id in self.id_to_idx:
                idx = self.id_to_idx[node_id]
                del self.id_to_idx[node_id]
                del self.idx_to_id[idx]
                return True
        return False
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Returns list of (node_id, similarity) tuples.
        """
        with self._lock:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query
            query_vector = query_vector.astype(np.float32)
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
            
            # Search (k might be larger than ntotal)
            actual_k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_vector.reshape(1, -1), actual_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx in self.idx_to_id:
                    node_id = self.idx_to_id[idx]
                    similarity = float(distances[0][i])  # Already cosine similarity
                    results.append((node_id, similarity))
            
            return results
    
    def save(self) -> None:
        """Persist index to disk."""
        with self._lock:
            os.makedirs(self.path, exist_ok=True)
            
            try:
                faiss.write_index(self.index, self.index_file)
                with open(self.mapping_file, 'wb') as f:
                    pickle.dump({
                        'id_to_idx': self.id_to_idx,
                        'idx_to_id': self.idx_to_id,
                        'next_idx': self._next_idx
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.debug(f"Saved FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                raise StorageError(f"Failed to save FAISS index: {e}")
    
    def rebuild(self, vectors: List[Tuple[str, np.ndarray]]) -> None:
        """Rebuild index from scratch with given vectors."""
        with self._lock:
            self._create_new_index()
            
            for node_id, vector in vectors:
                self.add(node_id, vector)
            
            logger.info(f"Rebuilt FAISS index with {len(vectors)} vectors")
    
    @property
    def size(self) -> int:
        """Get number of vectors in index."""
        with self._lock:
            return self.index.ntotal if self.index else 0


# ---------------------------
# NeuroIndex Main Class
# ---------------------------
class NeuroIndex:
    """
    Production-ready hybrid vector + semantic graph memory system.
    
    Combines fast FAISS search, semantic graph traversal, and LRU caching
    for optimal retrieval performance.
    
    Args:
        path: Directory for persistent storage
        dim: Embedding dimension (must match your embeddings)
        cache_size: Maximum items in LRU cache
        similarity_threshold: Minimum similarity for graph edges (0-1)
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    
    Example:
        >>> from neuroindex import NeuroIndex
        >>> import numpy as np
        >>> 
        >>> # Initialize
        >>> ni = NeuroIndex(path="./memory", dim=384)
        >>> 
        >>> # Add documents
        >>> embedding = np.random.rand(384).astype('float32')
        >>> node_id = ni.add_document("Hello world", embedding)
        >>> 
        >>> # Search
        >>> results = ni.search(embedding, k=5)
        >>> for r in results:
        ...     print(f"{r.text}: {r.similarity:.3f}")
        >>> 
        >>> # Always close when done
        >>> ni.close()
    """
    
    VERSION = "0.2.0"
    
    def __init__(
        self,
        path: str = './neuroindex_data',
        dim: int = 384,
        cache_size: int = 10000,
        similarity_threshold: float = 0.7,
        log_level: str = "INFO"
    ):
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"{__name__}.NeuroIndex")
        
        # Validate inputs
        if dim <= 0:
            raise InvalidInputError(f"Dimension must be positive, got {dim}")
        
        if cache_size < 0:
            raise InvalidInputError(f"Cache size must be non-negative, got {cache_size}")
        
        if not 0 <= similarity_threshold <= 1:
            raise InvalidInputError(f"Similarity threshold must be in [0, 1], got {similarity_threshold}")
        
        self.path = Path(path)
        self.dim = dim
        self._lock = threading.RLock()
        
        # Initialize components
        self.cache = NeuroCache(max_size=cache_size)
        self.graph = SemanticGraph(str(self.path), similarity_threshold=similarity_threshold)
        self.storage = PersistentStorage(str(self.path))
        self.faiss_index = FAISSIndexManager(str(self.path), dim)
        self.bloom = BloomFilter()
        self.metrics = MetricsCollector()
        
        # Background worker
        self.update_queue: queue.Queue = queue.Queue()
        self._running = True
        self._bg_thread = threading.Thread(target=self._bg_worker, daemon=True)
        self._bg_thread.start()
        
        # Rebuild bloom filter from storage
        self._rebuild_bloom_filter()
        
        self.logger.info(
            f"NeuroIndex v{self.VERSION} initialized at {path} "
            f"(dim={dim}, docs={self.storage.get_node_count()})"
        )
    
    def _rebuild_bloom_filter(self) -> None:
        """Rebuild bloom filter from existing documents."""
        count = 0
        for node in self.storage.iterate_all():
            text_hash = hashlib.md5(node['text'].encode()).hexdigest()
            self.bloom.add(text_hash)
            count += 1
        
        if count > 0:
            self.logger.debug(f"Rebuilt bloom filter with {count} documents")
    
    def _bg_worker(self) -> None:
        """Background worker for deferred operations."""
        while self._running:
            try:
                task = self.update_queue.get(timeout=1.0)
                if task[0] == 'save':
                    self.graph.save()
                    self.faiss_index.save()
                self.update_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Background worker error: {e}")
    
    def _validate_vector(self, vector: Union[np.ndarray, list]) -> np.ndarray:
        """Validate and normalize input vector."""
        # Convert to numpy if needed
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)
        
        if not isinstance(vector, np.ndarray):
            raise InvalidInputError(f"Vector must be numpy array or list, got {type(vector)}")
        
        # Ensure float32
        vector = vector.astype(np.float32)
        
        # Check dimension
        if vector.shape[0] != self.dim:
            raise DimensionMismatchError(self.dim, vector.shape[0])
        
        # Check for NaN/Inf
        if np.isnan(vector).any():
            raise InvalidInputError("Vector contains NaN values")
        
        if np.isinf(vector).any():
            raise InvalidInputError("Vector contains Inf values")
        
        return vector
    
    def _validate_text(self, text: str) -> str:
        """Validate input text."""
        if not isinstance(text, str):
            raise InvalidInputError(f"Text must be string, got {type(text)}")
        
        text = text.strip()
        if not text:
            raise InvalidInputError("Text cannot be empty")
        
        return text
    
    def add_document(
        self,
        text: str,
        vector: Union[np.ndarray, list],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a document to the index.
        
        Args:
            text: Document text content
            vector: Embedding vector (must match index dimension)
            metadata: Optional metadata dictionary
        
        Returns:
            node_id: Unique identifier for the document
        
        Raises:
            DimensionMismatchError: If vector dimension doesn't match index
            InvalidInputError: If text is empty or vector contains invalid values
        """
        with self.metrics.measure("add_document"):
            # Validate inputs
            text = self._validate_text(text)
            vector = self._validate_vector(vector)
            metadata = metadata or {}
            
            # Generate node ID
            node_id = hashlib.sha256(f"{text}_{vector.tobytes()}".encode()).hexdigest()[:16]
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Check for duplicates
            if self.bloom.contains(text_hash):
                existing = self.storage.get_node(node_id)
                if existing:
                    self.logger.debug(f"Document already exists: {node_id}")
                    return node_id
            
            self.bloom.add(text_hash)
            
            # Create node
            node = {
                'id': node_id,
                'text': text,
                'vector': vector,
                'metadata': metadata,
                'access_count': 0,
                'last_accessed': time.time(),
                'creation_time': time.time(),
                'importance_score': 1.0
            }
            
            with self._lock:
                # Add to all stores
                self.storage.add_node(node)
                self.faiss_index.add(node_id, vector)
                self.graph.add_node(node_id, vector, metadata)
                self.cache.add(node_id, node)
            
            # Queue background save
            self.update_queue.put(('save',))
            
            self.logger.debug(f"Added document: {node_id}")
            return node_id
    
    def search(
        self,
        query_vector: Union[np.ndarray, list],
        k: int = 10,
        use_graph: bool = True,
        use_cache: bool = True,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            use_graph: Whether to use graph traversal
            use_cache: Whether to check cache first
            min_similarity: Minimum similarity threshold (0-1)
        
        Returns:
            List of SearchResult objects, sorted by similarity (highest first)
        
        Raises:
            DimensionMismatchError: If query dimension doesn't match index
        """
        with self.metrics.measure("search"):
            query_vector = self._validate_vector(query_vector)
            
            if k <= 0:
                raise InvalidInputError(f"k must be positive, got {k}")
            
            results_dict: Dict[str, SearchResult] = {}
            
            # 1. Check cache first
            if use_cache and len(self.cache) > 0:
                cache_results = self.cache.search(query_vector, k=k)
                for r in cache_results:
                    if r.similarity >= min_similarity:
                        results_dict[r.node_id] = r
                        self.metrics.record_cache_hit()
            
            # 2. FAISS search
            faiss_results = self.faiss_index.search(query_vector, k=k * 2)
            self.metrics.record_faiss_search()
            
            for node_id, similarity in faiss_results:
                if similarity >= min_similarity and node_id not in results_dict:
                    node = self.storage.get_node(node_id)
                    if node:
                        results_dict[node_id] = SearchResult(
                            node_id=node_id,
                            text=node['text'],
                            similarity=similarity,
                            metadata=node['metadata'],
                            source='faiss'
                        )
                        # Add to cache
                        self.cache.add(node_id, node)
                        self.storage.update_access(node_id)
            
            # 3. Graph traversal for relationship-aware results
            if use_graph:
                graph_ids = self.graph.search_by_traversal(query_vector, k=k)
                self.metrics.record_graph_traversal()
                
                for nid in graph_ids:
                    if nid not in results_dict:
                        node = self.storage.get_node(nid)
                        if node:
                            # Calculate similarity
                            node_vector = node['vector']
                            norm_q = np.linalg.norm(query_vector)
                            norm_n = np.linalg.norm(node_vector)
                            
                            if norm_q > 0 and norm_n > 0:
                                similarity = float(np.dot(query_vector, node_vector) / (norm_q * norm_n))
                                
                                if similarity >= min_similarity:
                                    results_dict[nid] = SearchResult(
                                        node_id=nid,
                                        text=node['text'],
                                        similarity=similarity,
                                        metadata=node['metadata'],
                                        source='graph'
                                    )
            
            # Sort by similarity and return top k
            results = list(results_dict.values())
            results.sort(key=lambda x: x.similarity, reverse=True)
            
            return results[:k]
    
    def search_text(
        self,
        text: str,
        embed_fn: Callable[[str], np.ndarray],
        k: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search using raw text with an embedding function.
        
        Args:
            text: Query text
            embed_fn: Function that converts text to embedding vector
            k: Number of results to return
            **kwargs: Additional arguments passed to search()
        
        Returns:
            List of SearchResult objects
        
        Example:
            >>> def embed(text):
            ...     return model.encode(text)
            >>> results = ni.search_text("What is AI?", embed_fn=embed)
        """
        vector = embed_fn(text)
        return self.search(query_vector=vector, k=k, **kwargs)
    
    def get_document(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            node_id: Document identifier
        
        Returns:
            Document dict or None if not found
        """
        # Check cache first
        cached = self.cache.get(node_id)
        if cached:
            self.metrics.record_cache_hit()
            return cached
        
        self.metrics.record_cache_miss()
        
        # Check storage
        node = self.storage.get_node(node_id)
        if node:
            self.cache.add(node_id, node)
            return node
        
        return None
    
    def delete_document(self, node_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            node_id: Document identifier
        
        Returns:
            True if document was deleted, False if not found
        
        Raises:
            DocumentNotFoundError: If document doesn't exist (optional strict mode)
        """
        with self.metrics.measure("delete_document"):
            with self._lock:
                # Remove from all stores
                deleted = self.storage.delete_node(node_id)
                
                if deleted:
                    self.cache.remove(node_id)
                    self.faiss_index.remove(node_id)
                    self.graph.remove_node(node_id)
                    self.update_queue.put(('save',))
                    self.logger.debug(f"Deleted document: {node_id}")
                
                return deleted
    
    def update_document(
        self,
        node_id: str,
        text: Optional[str] = None,
        vector: Optional[Union[np.ndarray, list]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing document.
        
        Args:
            node_id: Document identifier
            text: New text (optional)
            vector: New embedding vector (optional)
            metadata: New metadata (optional)
        
        Returns:
            True if updated, False if document not found
        
        Raises:
            DimensionMismatchError: If new vector dimension doesn't match
        """
        with self.metrics.measure("update_document"):
            # Validate inputs
            updates = {}
            
            if text is not None:
                updates['text'] = self._validate_text(text)
            
            if vector is not None:
                updates['vector'] = self._validate_vector(vector)
            
            if metadata is not None:
                updates['metadata'] = metadata
            
            if not updates:
                return False
            
            with self._lock:
                # Get existing node
                existing = self.storage.get_node(node_id)
                if not existing:
                    return False
                
                # Update storage
                success = self.storage.update_node(node_id, updates)
                
                if success:
                    # Update cache
                    cached = self.cache.get(node_id)
                    if cached:
                        cached.update(updates)
                        self.cache.add(node_id, cached)
                    
                    # If vector changed, rebuild FAISS entry
                    if 'vector' in updates:
                        self.faiss_index.remove(node_id)
                        self.faiss_index.add(node_id, updates['vector'])
                        
                        # Update graph node
                        self.graph.remove_node(node_id)
                        self.graph.add_node(node_id, updates['vector'], updates.get('metadata', existing['metadata']))
                    
                    self.update_queue.put(('save',))
                    self.logger.debug(f"Updated document: {node_id}")
                
                return success
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with statistics about the index
        """
        return {
            'version': self.VERSION,
            'total_documents': self.storage.get_node_count(),
            'faiss_vectors': self.faiss_index.size,
            'cache_size': len(self.cache),
            'graph_nodes': self.graph.node_count(),
            'graph_edges': self.graph.edge_count(),
            'dimension': self.dim,
            'path': str(self.path),
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.metrics.summary()
    
    def rebuild_index(self) -> None:
        """
        Rebuild FAISS index from storage.
        
        Useful after many deletions or when index is corrupted.
        """
        self.logger.info("Rebuilding FAISS index...")
        
        vectors = []
        for node in self.storage.iterate_all():
            vectors.append((node['id'], node['vector']))
        
        self.faiss_index.rebuild(vectors)
        self.faiss_index.save()
        
        self.logger.info(f"Rebuilt FAISS index with {len(vectors)} vectors")
    
    def clear(self) -> None:
        """
        Clear all data from the index.
        
        WARNING: This permanently deletes all documents!
        """
        self.logger.warning("Clearing all data from index...")
        
        import shutil
        
        with self._lock:
            self.cache.clear()
            self.bloom.clear()
            
            # Close existing connections first
            self.storage.close()
            
            # Recreate fresh index (without deleting files - just reinitialize)
            # This avoids file locking issues on Windows
            self.faiss_index._create_new_index()
            self.faiss_index.save()
            
            self.graph.graph.clear()
            self.graph.save()
            
            # Delete and recreate storage
            if self.path.exists():
                try:
                    shutil.rmtree(self.path)
                except (PermissionError, OSError) as e:
                    # On Windows, files might be locked. Just delete contents.
                    self.logger.debug(f"Could not delete directory, recreating: {e}")
            
            self.storage = PersistentStorage(str(self.path))
            self.faiss_index = FAISSIndexManager(str(self.path), self.dim)
            self.graph = SemanticGraph(str(self.path))
        
        self.logger.info("Index cleared")
    
    def close(self) -> None:
        """
        Close the index and release resources.
        
        Always call this when done using the index!
        """
        self.logger.info("Closing NeuroIndex...")
        
        self._running = False
        self._bg_thread.join(timeout=5)
        
        # Save all data
        self.graph.save()
        self.faiss_index.save()
        self.storage.close()
        
        self.logger.info("NeuroIndex closed successfully")
    
    def __enter__(self) -> 'NeuroIndex':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:
        return f"NeuroIndex(path='{self.path}', dim={self.dim}, docs={self.storage.get_node_count()})"
