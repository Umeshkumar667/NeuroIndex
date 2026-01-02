# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-02

### Added
- **FAISS Integration**: FAISS index is now actually used for vector search (was previously unused)
- **FAISS Persistence**: Index is saved to disk and loaded on restart
- **Delete Operation**: `delete_document(node_id)` to remove documents
- **Update Operation**: `update_document(node_id, text=, vector=, metadata=)` to modify documents
- **Custom Exceptions**: Clear, actionable exceptions (`DimensionMismatchError`, `StorageError`, etc.)
- **Input Validation**: Comprehensive validation for all inputs (dimensions, NaN/Inf, empty text)
- **Metrics Collection**: Built-in performance metrics via `get_metrics()`
- **Logging**: Configurable logging with `log_level` parameter
- **Context Manager**: `with NeuroIndex(...) as ni:` for automatic cleanup
- **Thread Safety**: Improved locking for concurrent operations
- **CI/CD**: GitHub Actions for testing on Python 3.9-3.12, multiple OS
- **Benchmarks**: Performance benchmarking scripts
- **Comprehensive Tests**: 80%+ test coverage with unit, integration, and performance tests

### Changed
- **API Stability**: Documented stable APIs following SemVer
- **Search Results**: Now include source information ('cache', 'faiss', 'graph')
- **Default Dimension**: Changed from 112 to 384 (more common embedding size)
- **Error Handling**: All errors now raise specific exceptions instead of failing silently

### Fixed
- FAISS index was created but never used for search
- Graph persistence was unreliable
- Cache eviction could cause issues under heavy load
- SQLite connections weren't properly pooled

### Security
- Added input sanitization for text content
- Improved pickle handling warnings in documentation

## [0.1.2] - 2025-12-15

### Added
- `search_text()` helper for text-based queries
- Improved public API ergonomics

### Fixed
- Test stability improvements

## [0.1.1] - 2025-12-10

### Added
- Initial PyPI release
- Basic documentation

## [0.1.0] - 2025-12-01

### Added
- Initial release
- Hybrid cache + vector + graph memory system
- SQLite persistence
- Basic search functionality
