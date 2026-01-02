# Contributing to NeuroIndex

First off, thank you for considering contributing to NeuroIndex! 🎉

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

When creating a bug report, include:
- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs actual behavior
- **Environment info**: Python version, OS, NeuroIndex version
- **Code samples** if applicable

```markdown
### Bug Report

**Description**: Brief description of the bug

**Steps to Reproduce**:
1. Create index with `NeuroIndex(dim=384)`
2. Add document with `add_document(...)`
3. Search returns incorrect results

**Expected**: Should return similar documents
**Actual**: Returns empty list

**Environment**:
- Python: 3.11
- OS: Ubuntu 22.04
- NeuroIndex: 0.2.0
```

### 💡 Suggesting Features

Feature requests are welcome! Please provide:
- Clear description of the feature
- Use case / motivation
- Possible implementation approach (optional)

### 🔧 Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Ensure all tests pass**: `pytest tests/ -v`
6. **Format code**: `black neuroindex/` and `isort neuroindex/`
7. **Commit changes**: `git commit -m "Add my feature"`
8. **Push to branch**: `git push origin feature/my-feature`
9. **Open Pull Request**

## Development Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/NeuroIndex.git
cd NeuroIndex

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=neuroindex --cov-report=html

# Run specific test file
pytest tests/unit/test_validation.py -v

# Run specific test
pytest tests/unit/test_validation.py::TestDimensionValidation::test_correct_dimension_accepted -v
```

### Code Formatting

We use `black` for formatting and `isort` for imports:

```bash
# Format code
black neuroindex/
isort neuroindex/

# Check formatting without changes
black --check neuroindex/
isort --check-only neuroindex/
```

### Type Checking

```bash
mypy neuroindex/ --ignore-missing-imports
```

## Project Structure

```
neuroindex/
├── __init__.py        # Public API exports
├── core.py            # Main NeuroIndex class
├── exceptions.py      # Custom exceptions
└── metrics.py         # Metrics collection

tests/
├── conftest.py        # Shared fixtures
├── unit/              # Unit tests
├── integration/       # Integration tests
└── performance/       # Benchmarks

benchmarks/
└── run_benchmarks.py  # Performance benchmarks
```

## Coding Guidelines

### Style

- Follow PEP 8
- Use type hints for all public functions
- Maximum line length: 100 characters
- Use descriptive variable names

### Docstrings

Use Google-style docstrings:

```python
def add_document(self, text: str, vector: np.ndarray) -> str:
    """Add a document to the index.
    
    Args:
        text: The document text content.
        vector: Embedding vector (must match index dimension).
    
    Returns:
        Unique node ID for the document.
    
    Raises:
        DimensionMismatchError: If vector dimension doesn't match.
        InvalidInputError: If text is empty.
    
    Example:
        >>> ni = NeuroIndex(dim=384)
        >>> node_id = ni.add_document("Hello world", embedding)
    """
```

### Tests

- All new features need tests
- Aim for 80%+ coverage
- Use descriptive test names
- Test edge cases and error conditions

```python
def test_add_document_with_empty_text_raises_error(self, ni):
    """Empty text should raise InvalidInputError."""
    vec = np.random.rand(128).astype('float32')
    with pytest.raises(InvalidInputError):
        ni.add_document("", vec)
```

### Commits

- Use clear, descriptive commit messages
- Reference issues when applicable: `Fix #123: Handle empty vectors`
- Keep commits focused on a single change

## Release Process

1. Update version in `pyproject.toml` and `neuroindex/__init__.py`
2. Update `CHANGELOG.md`
3. Create PR with version bump
4. After merge, create GitHub Release with tag `v0.X.X`
5. GitHub Actions will publish to PyPI

## Questions?

- Open an issue for questions about contributing
- Tag with `question` label

Thank you for contributing! 🙏

