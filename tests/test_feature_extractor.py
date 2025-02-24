"""
Integration tests for the feature extractor.
"""
import pytest
from pathlib import Path
import tempfile
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.feature_extractor import FeatureExtractor

@pytest.fixture
def sample_reviews():
    """Create sample processed reviews for testing"""
    return [
        {
            "rating": 10.0,
            "date": datetime.now().isoformat(),
            "helpful_ratio": 0.8,
            "title": "Absolutely amazing film, loved every minute!",
            "content": "This movie was fantastic. The acting was superb and the storyline kept me engaged throughout. I would highly recommend it to anyone who enjoys this genre.",
            "content_length": 150,
            "user_id": "user1"
        },
        {
            "rating": 9.0,
            "date": (datetime.now() - timedelta(days=5)).isoformat(),
            "helpful_ratio": 0.7,
            "title": "Great movie with excellent performances",
            "content": "I really enjoyed this film. The character development was excellent and the cinematography was beautiful. Would watch again.",
            "content_length": 120,
            "user_id": "user2"
        },
        {
            "rating": 3.0,
            "date": (datetime.now() - timedelta(days=2)).isoformat(),
            "helpful_ratio": 0.3,
            "title": "Disappointing and boring",
            "content": "I was really disappointed by this movie. The plot was predictable and the acting was mediocre at best. Not worth the time.",
            "content_length": 110,
            "user_id": "user3"
        },
        {
            "rating": 10.0,
            "date": (datetime.now() - timedelta(hours=1)).isoformat(),
            "helpful_ratio": 0.1,
            "title": "AMAZING MUST SEE!!!!!",
            "content": "WOW!!! BEST MOVIE EVER!!!! THE ACTING WAS AMAZING!!!!! I LOVED IT SO MUCH!!!! EVERYONE MUST SEE THIS FILM!!!!!!",
            "content_length": 100,
            "user_id": "user4"
        },
        {
            "rating": 10.0,
            "date": datetime.now().isoformat(),
            "helpful_ratio": 0.0,
            "title": "Great movie!!",
            "content": "This movie was fantastic. The acting was superb and the storyline kept me engaged throughout. Would highly recommend.",
            "content_length": 95,
            "user_id": "user5"
        }
    ]

@pytest.fixture
def temp_extractor(sample_reviews):
    """Create a feature extractor with temporary directory and sample data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        processed_dir = temp_dir_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample processed reviews file
        movie_id = "tt0111161"  # The Shawshank Redemption
        with open(processed_dir / f"{movie_id}_processed.json", 'w', encoding='utf-8') as f:
            json.dump(sample_reviews, f)
        
        extractor = FeatureExtractor(data_dir=temp_dir)
        yield extractor

def test_load_processed_reviews(temp_extractor):
    """Test loading processed reviews"""
    reviews = temp_extractor.load_processed_reviews("tt0111161")
    assert len(reviews) == 5
    assert "rating" in reviews[0]
    assert "content" in reviews[0]

def test_extract_text_features(temp_extractor, sample_reviews):
    """Test extracting text features"""
    features = temp_extractor.extract_text_features(sample_reviews[0])
    
    assert "word_count" in features
    assert "char_count" in features
    assert "vocabulary_richness" in features
    assert "title_sentiment" in features
    assert "content_sentiment" in features
    assert "sentiment_diff" in features
    
    # Check suspicious patterns in the all-caps review
    suspicious_features = temp_extractor.extract_text_features(sample_reviews[3])
    assert suspicious_features["uppercase_ratio"] > 0.5
    assert suspicious_features["exclamation_count"] > 5

def test_extract_metadata_features(temp_extractor, sample_reviews):
    """Test extracting metadata features"""
    features = temp_extractor.extract_metadata_features(sample_reviews[0], sample_reviews)
    
    assert "rating" in features
    assert "rating_deviation" in features
    assert "helpful_ratio" in features
    assert "normalized_timing" in features

def test_detect_review_bursts(temp_extractor, sample_reviews):
    """Test detecting bursts of reviews"""
    # Create a burst scenario
    burst_reviews = sample_reviews.copy()
    for i in range(5):
        burst_reviews.append({
            "rating": 10.0,
            "date": datetime.now().isoformat(),  # All same timestamp
            "user_id": f"burst_user_{i}",
            "content": "Fantastic movie! Loved it!",
            "title": "Great!"
        })
    
    bursts = temp_extractor.detect_review_bursts(burst_reviews)
    assert len(bursts) > 0

def test_extract_similarity_features(temp_extractor, sample_reviews):
    """Test extracting similarity features"""
    # Add a duplicate review to test similarity
    reviews = sample_reviews.copy()
    duplicate = reviews[0].copy()
    duplicate["user_id"] = "duplicate_user"
    reviews.append(duplicate)
    
    features = temp_extractor.extract_similarity_features(reviews[0], reviews)
    assert "max_similarity" in features
    assert "similar_reviews_count" in features
    assert features["max_similarity"] > 0.9  # Should be very similar to the duplicate

def test_extract_all_features(temp_extractor):
    """Test extracting all features for a movie"""
    features_df = temp_extractor.extract_all_features("tt0111161")
    
    assert not features_df.empty
    assert "review_id" in features_df.columns
    assert "word_count" in features_df.columns
    assert "rating_deviation" in features_df.columns
    assert "max_similarity" in features_df.columns
    
    # There should be one row per review
    assert len(features_df) == 5

def test_save_features(temp_extractor):
    """Test saving features to disk"""
    features_df = temp_extractor.extract_all_features("tt0111161")
    temp_extractor.save_features("tt0111161", features_df)
    
    feature_file = Path(temp_extractor.features_dir) / "tt0111161_features.csv"
    assert feature_file.exists()
    
    # Load and verify
    loaded_df = pd.read_csv(feature_file)
    assert len(loaded_df) == len(features_df)