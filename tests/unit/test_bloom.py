"""
Unit tests for BloomFilter.
"""

import pytest
from neuroindex.core import BloomFilter


class TestBloomFilterBasics:
    """Basic bloom filter operations."""
    
    def test_add_and_contains(self):
        """Test adding and checking membership."""
        bloom = BloomFilter(capacity=1000, error_rate=0.01)
        
        bloom.add("test_item")
        assert bloom.contains("test_item") is True
    
    def test_not_contains(self):
        """Non-added items should (mostly) not be contained."""
        bloom = BloomFilter(capacity=1000, error_rate=0.01)
        
        bloom.add("item_1")
        
        # Non-added item should not be contained (probabilistic)
        assert bloom.contains("completely_different_item") is False
    
    def test_multiple_items(self):
        """Test with multiple items."""
        bloom = BloomFilter(capacity=1000, error_rate=0.01)
        
        items = [f"item_{i}" for i in range(100)]
        for item in items:
            bloom.add(item)
        
        # All added items should be found
        for item in items:
            assert bloom.contains(item) is True
    
    def test_clear(self):
        """Test clearing the filter."""
        bloom = BloomFilter(capacity=1000, error_rate=0.01)
        
        bloom.add("test_item")
        assert bloom.contains("test_item") is True
        
        bloom.clear()
        assert bloom.contains("test_item") is False


class TestBloomFilterFalsePositives:
    """False positive rate testing."""
    
    def test_false_positive_rate(self):
        """False positive rate should be within expected bounds."""
        bloom = BloomFilter(capacity=10000, error_rate=0.01)
        
        # Add half the items
        for i in range(5000):
            bloom.add(f"added_{i}")
        
        # Check false positives on non-added items
        false_positives = 0
        test_count = 5000
        
        for i in range(test_count):
            if bloom.contains(f"not_added_{i}"):
                false_positives += 1
        
        false_positive_rate = false_positives / test_count
        
        # Should be close to error_rate (with some tolerance)
        # Allow up to 3x the specified error rate
        assert false_positive_rate < 0.03, f"False positive rate {false_positive_rate} too high"


class TestBloomFilterEdgeCases:
    """Edge cases for bloom filter."""
    
    def test_empty_string(self):
        """Should handle empty string."""
        bloom = BloomFilter()
        
        bloom.add("")
        assert bloom.contains("") is True
    
    def test_unicode(self):
        """Should handle unicode strings."""
        bloom = BloomFilter()
        
        bloom.add("你好世界")
        assert bloom.contains("你好世界") is True
        assert bloom.contains("hello") is False
    
    def test_very_long_string(self):
        """Should handle very long strings."""
        bloom = BloomFilter()
        
        long_string = "a" * 10000
        bloom.add(long_string)
        assert bloom.contains(long_string) is True

