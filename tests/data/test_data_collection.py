"""
Tests for the data collection module.
"""
import pytest
from pathlib import Path
from src.data.data_collection import DataCollector

def test_data_collector_initialization():
    """Test if DataCollector initializes properly"""
    output_path = Path("data/raw")
    collector = DataCollector(output_path)
    assert collector.output_path == output_path

def test_save_data(tmp_path):
    """Test if save_data creates a file correctly"""
    collector = DataCollector(tmp_path)
    test_data = {"test": "data"}
    filename = "test.json"
    collector.save_data(test_data, filename)
    assert (tmp_path / filename).exists()

def test_collect_movie_reviews(tmp_path):
    """Test movie review collection"""
    collector = DataCollector(tmp_path)
    movie_id = "0111161"  # The Shawshank Redemption
    max_reviews = 5
    
    # Collect reviews
    movie_data = collector.collect_movie_reviews(movie_id, max_reviews)
    
    # Basic validation
    assert movie_data['movie_id'] == movie_id
    assert 'title' in movie_data
    assert 'reviews' in movie_data
    assert len(movie_data['reviews']) <= max_reviews
