"""
Tests for enhanced features of the IMDB collector.
"""
import pytest
from pathlib import Path
import json
import time
from src.data.imdb_collector import DataCollector

class TestEnhancedCollector:
    @pytest.fixture
    def collector(self, tmp_path):
        """Create a collector instance with temporary directory"""
        return DataCollector(data_dir=str(tmp_path))

    def test_directory_creation(self, collector, tmp_path):
        """Test that required directories are created"""
        assert (tmp_path / "raw").exists()
        assert (tmp_path / "processed").exists()

    def test_metadata_collection(self, collector):
        """Test movie metadata collection"""
        movie_id = "0111161"  # The Shawshank Redemption
        metadata = collector.get_movie_metadata(movie_id)
        
        # Basic structure checks
        assert metadata
        assert metadata["movie_id"] == movie_id
        assert "title" in metadata
        assert "year" in metadata
        assert "rating" in metadata
        assert "genres" in metadata
        assert "cast" in metadata
        assert len(metadata["cast"]) <= 5  # Should only have top 5 cast members

    def test_review_collection(self, collector):
        """Test review collection with limits"""
        movie_id = "0111161"
        max_reviews = 3
        reviews = collector.get_movie_reviews(movie_id, max_reviews=max_reviews)
        
        assert len(reviews) <= max_reviews
        if reviews:
            review = reviews[0]
            assert "rating" in review
            assert "title" in review
            assert "content" in review
            assert "date" in review
            assert "helpful_votes" in review

    def test_rate_limiting(self, collector):
        """Test rate limiting between requests"""
        movie_id = "0111161"
        start_time = time.time()
        
        # Make two quick requests
        collector.get_movie_reviews(movie_id, max_reviews=2)
        collector.get_movie_reviews(movie_id, max_reviews=2)
        
        # Check if rate limiting worked
        elapsed = time.time() - start_time
        assert elapsed >= 1.0, "Rate limiting should enforce delays between requests"

    def test_save_data_formats(self, collector, sample_movie_data, tmp_path):
        """Test saving data in different formats"""
        # Test JSON saving
        json_file = "test.json"
        collector.save_data(sample_movie_data, json_file)
        json_path = tmp_path / "raw" / json_file
        assert json_path.exists()
        
        # Test CSV saving
        csv_file = "test.csv"
        collector.save_data([sample_movie_data], csv_file)
        csv_path = tmp_path / "raw" / csv_file
        assert csv_path.exists()

    def test_error_handling(self, collector):
        """Test handling of invalid movie IDs"""
        invalid_id = "9999999"
        # Should return empty dict/list for invalid IDs
        assert collector.get_movie_metadata(invalid_id) == {}
        assert collector.get_movie_reviews(invalid_id) == []