"""
NeuroIndex Custom Exceptions

Provides clear, actionable error messages for production use.
"""


class NeuroIndexError(Exception):
    """Base exception for all NeuroIndex errors."""

    pass


class DimensionMismatchError(NeuroIndexError):
    """Raised when embedding dimension doesn't match the index dimension."""

    def __init__(self, expected: int, got: int):
        self.expected = expected
        self.got = got
        super().__init__(
            f"Embedding dimension mismatch: expected {expected}, got {got}. "
            f"Ensure all embeddings have the same dimension as the index."
        )


class StorageError(NeuroIndexError):
    """Raised when storage operations fail (SQLite, file I/O)."""

    pass


class IndexCorruptedError(NeuroIndexError):
    """Raised when the index is corrupted and cannot be loaded."""

    def __init__(self, path: str, original_error: Exception = None):
        self.path = path
        self.original_error = original_error
        msg = f"Index at '{path}' is corrupted."
        if original_error:
            msg += f" Original error: {original_error}"
        msg += " Consider rebuilding the index from source data."
        super().__init__(msg)


class DocumentNotFoundError(NeuroIndexError):
    """Raised when a requested document doesn't exist."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        super().__init__(f"Document with ID '{node_id}' not found.")


class InvalidInputError(NeuroIndexError):
    """Raised when input validation fails."""

    pass


class ConcurrencyError(NeuroIndexError):
    """Raised when concurrent operations conflict."""

    pass
