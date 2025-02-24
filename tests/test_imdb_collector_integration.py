"""
Integration tests for the IMDB data collector.
"""
import pytest
from pathlib import Path
import tempfile
import json
from src.data.imdb_collector import DataCollector

@pytest.fixture
def temp_collector():
    """Create a collector instance with a temporary directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        collector = DataCollector(data_dir=temp_dir)  # Changed from output_path to data_dir
        yield collector

def test_metadata_collection(temp_collector):
    """Test collecting movie metadata"""
    movie_id = "0111161"  # The Shawshank Redemption
    metadata = temp_collector.get_movie_metadata(movie_id)
    
    # Check basic metadata structure
    assert metadata["movie_id"] == f"tt{movie_id}"
    assert "title" in metadata
    assert "year" in metadata
    assert "rating" in metadata
    assert "genres" in metadata
    assert isinstance(metadata["genres"], list)
    assert "cast" in metadata
    assert isinstance(metadata["cast"], list)
    assert len(metadata["cast"]) <= 5  # Should only have top 5 cast members

def test_review_collection(temp_collector):
    """Test collecting movie reviews"""
    movie_id = "0111161"
    max_reviews = 2
    reviews = temp_collector.get_movie_reviews(movie_id, max_reviews=max_reviews)
    
    # Check review count and structure
    assert len(reviews) <= max_reviews
    if reviews:
        review = reviews[0]
        assert "rating" in review
        assert "date" in review
        assert "helpful" in review
        assert "title" in review
        assert "content" in review
        assert "user_id" in review
        
        # Check helpful votes structure
        assert isinstance(review["helpful"], dict)
        assert "up" in review["helpful"]
        assert "total" in review["helpful"]

def test_full_movie_data_collection(temp_collector):
    """Test collecting complete movie data"""
    movie_id = "0111161"
    movie_data = temp_collector.collect_movie_data(movie_id, max_reviews=2)
    
    # Check combined data structure
    assert "movie_id" in movie_data
    assert "title" in movie_data
    assert "reviews" in movie_data
    assert isinstance(movie_data["reviews"], list)

def test_data_saving(temp_collector):
    """Test saving data in different formats"""
    test_data = {
        "movie_id": "tt0111161",
        "title": "Test Movie",
        "year": 2024,
        "reviews": [
            {
                "rating": "10",
                "title": "Great movie",
                "content": "Test review content"
            }
        ]
    }
    
    # Test JSON saving
    temp_collector.save_data(test_data, "test_movie.json")
    json_path = Path(temp_collector.raw_dir) / "test_movie.json"
    assert json_path.exists()
    
    # Verify JSON content
    with open(json_path, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
        assert saved_data["movie_id"] == test_data["movie_id"]
        assert saved_data["title"] == test_data["title"]

def test_review_processing(temp_collector):
    """Test review processing functionality"""
    sample_reviews = [
        {
            "rating": "8",
            "date": "21 February 2024",
            "helpful": {"up": 50, "total": 100},
            "title": "Good movie",
            "content": "Test review content",
            "user_id": "ur123456"
        }
    ]
    
    processed = temp_collector.process_reviews(sample_reviews)
    assert len(processed) == 1
    
    processed_review = processed[0]
    assert processed_review["rating"] == 8.0
    assert "helpful_ratio" in processed_review
    assert processed_review["helpful_ratio"] == 0.5
    assert processed_review["content_length"] == len(sample_reviews[0]["content"])