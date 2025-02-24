"""
Integration tests for the bot detector model.
"""
import pytest
from pathlib import Path
import tempfile
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.bot_detector_model import BotDetectorModel

@pytest.fixture
def sample_features():
    """Create sample review features for testing"""
    return pd.DataFrame([
        # Suspicious reviews
        {
            'review_id': 'user1',
            'word_count': 50,
            'char_count': 250,
            'avg_word_length': 5.0,
            'exclamation_count': 10,
            'question_count': 0,
            'uppercase_ratio': 0.4,
            'vocabulary_richness': 0.3,
            'title_sentiment': 0.95,
            'content_sentiment': 0.92,
            'content_negativity': 0.02,
            'content_positivity': 0.95,
            'sentiment_diff': 0.03,
            'repeated_chars_max': 4,
            'rating': 10.0,
            'rating_deviation': 3.5,
            'helpful_ratio': 0.05,
            'days_since_earliest': 1,
            'days_until_latest': 10,
            'normalized_timing': 0.1,
            'max_similarity': 0.85,
            'avg_similarity': 0.3,
            'similar_reviews_count': 3,
            'in_burst': 1
        },
        {
            'review_id': 'user2',
            'word_count': 45,
            'char_count': 200,
            'avg_word_length': 4.5,
            'exclamation_count': 8,
            'question_count': 1,
            'uppercase_ratio': 0.3,
            'vocabulary_richness': 0.35,
            'title_sentiment': -0.95,
            'content_sentiment': -0.90,
            'content_negativity': 0.95,
            'content_positivity': 0.02,
            'sentiment_diff': 0.05,
            'repeated_chars_max': 3,
            'rating': 1.0,
            'rating_deviation': 4.0,
            'helpful_ratio': 0.08,
            'days_since_earliest': 2,
            'days_until_latest': 9,
            'normalized_timing': 0.2,
            'max_similarity': 0.82,
            'avg_similarity': 0.25,
            'similar_reviews_count': 2,
            'in_burst': 1
        },
        
        # Legitimate reviews
        {
            'review_id': 'user3',
            'word_count': 150,
            'char_count': 800,
            'avg_word_length': 5.3,
            'exclamation_count': 1,
            'question_count': 2,
            'uppercase_ratio': 0.05,
            'vocabulary_richness': 0.7,
            'title_sentiment': 0.6,
            'content_sentiment': 0.65,
            'content_negativity': 0.1,
            'content_positivity': 0.75,
            'sentiment_diff': 0.05,
            'repeated_chars_max': 1,
            'rating': 8.0,
            'rating_deviation': 0.5,
            'helpful_ratio': 0.85,
            'days_since_earliest': 5,
            'days_until_latest': 5,
            'normalized_timing': 0.5,
            'max_similarity': 0.2,
            'avg_similarity': 0.15,
            'similar_reviews_count': 0,
            'in_burst': 0
        },
        {
            'review_id': 'user4',
            'word_count': 120,
            'char_count': 650,
            'avg_word_length': 5.4,
            'exclamation_count': 0,
            'question_count': 3,
            'uppercase_ratio': 0.03,
            'vocabulary_richness': 0.75,
            'title_sentiment': -0.4,
            'content_sentiment': -0.5,
            'content_negativity': 0.6,
            'content_positivity': 0.1,
            'sentiment_diff': 0.1,
            'repeated_chars_max': 1,
            'rating': 4.0,
            'rating_deviation': 1.0,
            'helpful_ratio': 0.9,
            'days_since_earliest': 7,
            'days_until_latest': 3,
            'normalized_timing': 0.7,
            'max_similarity': 0.25,
            'avg_similarity': 0.2,
            'similar_reviews_count': 0,
            'in_burst': 0
        },
        {
            'review_id': 'user5',
            'word_count': 180,
            'char_count': 900,
            'avg_word_length': 5.0,
            'exclamation_count': 2,
            'question_count': 1,
            'uppercase_ratio': 0.04,
            'vocabulary_richness': 0.8,
            'title_sentiment': 0.3,
            'content_sentiment': 0.35,
            'content_negativity': 0.2,
            'content_positivity': 0.55,
            'sentiment_diff': 0.05,
            'repeated_chars_max': 1,
            'rating': 7.0,
            'rating_deviation': 0.0,
            'helpful_ratio': 0.95,
            'days_since_earliest': 6,
            'days_until_latest': 4,
            'normalized_timing': 0.6,
            'max_similarity': 0.15,
            'avg_similarity': 0.1,
            'similar_reviews_count': 0,
            'in_burst': 0
        }
    ])

@pytest.fixture
def temp_model(sample_features):
    """Create a model detector with temporary directory and sample data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        features_dir = temp_dir_path / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sample features
        movie_id = "tt0111161"  # The Shawshank Redemption
        features_file = features_dir / f"{movie_id}_features.csv"
        sample_features.to_csv(features_file, index=False)
        
        model = BotDetectorModel(data_dir=temp_dir)
        yield model

def test_load_features(temp_model):
    """Test loading features"""
    features = temp_model.load_features("tt0111161")
    assert not features.empty
    assert len(features) == 5
    assert "review_id" in features.columns
    assert "max_similarity" in features.columns

def test_create_labeled_dataset(temp_model, sample_features):
    """Test creating labeled dataset from features"""
    labeled_df = temp_model.create_labeled_dataset(sample_features)
    
    assert "is_suspicious" in labeled_df.columns
    # First two reviews should be labeled as suspicious
    assert labeled_df.loc[0, "is_suspicious"] == 1
    assert labeled_df.loc[1, "is_suspicious"] == 1
    # Last three reviews should be labeled as legitimate
    assert labeled_df.loc[2, "is_suspicious"] == 0
    assert labeled_df.loc[3, "is_suspicious"] == 0
    assert labeled_df.loc[4, "is_suspicious"] == 0

def test_prepare_training_data(temp_model, sample_features):
    """Test preparing training data"""
    labeled_df = temp_model.create_labeled_dataset(sample_features)
    X, y = temp_model.prepare_training_data(labeled_df)
    
    assert len(X) == len(y) == 5
    assert "review_id" not in X.columns
    assert "is_suspicious" not in X.columns
    assert y.sum() == 2  # 2 suspicious reviews

def test_train_model(temp_model, sample_features):
    """Test training a model"""
    labeled_df = temp_model.create_labeled_dataset(sample_features)
    X, y = temp_model.prepare_training_data(labeled_df)
    
    # Train model without hyperparameter tuning (too little data)
    model = temp_model.train_model(X, y, tune_hyperparams=False)
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    
    # Check feature importances for random forest
    assert temp_model.feature_importances is not None
    assert len(temp_model.feature_importances) == X.shape[1]

def test_evaluate_model(temp_model, sample_features):
    """Test evaluating a trained model"""
    labeled_df = temp_model.create_labeled_dataset(sample_features)
    X, y = temp_model.prepare_training_data(labeled_df)
    
    # Train model
    temp_model.train_model(X, y, tune_hyperparams=False)
    
    # Evaluate on same data (normally would use test set)
    metrics = temp_model.evaluate_model(X, y)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics
    
    # Should have perfect metrics on training data
    assert metrics["accuracy"] == 1.0

def test_predict(temp_model, sample_features):
    """Test making predictions with a trained model"""
    labeled_df = temp_model.create_labeled_dataset(sample_features)
    X, y = temp_model.prepare_training_data(labeled_df)
    
    # Train model
    temp_model.train_model(X, y, tune_hyperparams=False)
    
    # Make predictions
    predictions_df = temp_model.predict(sample_features)
    
    assert "prediction" in predictions_df.columns
    assert "probability" in predictions_df.columns
    assert len(predictions_df) == 5
    
    # Check predictions match expected labels
    assert predictions_df.loc[0, "prediction"] == 1
    assert predictions_df.loc[1, "prediction"] == 1
    assert predictions_df.loc[2, "prediction"] == 0
    assert predictions_df.loc[3, "prediction"] == 0
    assert predictions_df.loc[4, "prediction"] == 0

def test_save_and_load_model(temp_model, sample_features):
    """Test saving and loading a model"""
    labeled_df = temp_model.create_labeled_dataset(sample_features)
    X, y = temp_model.prepare_training_data(labeled_df)
    
    # Train model
    temp_model.train_model(X, y, tune_hyperparams=False)
    
    # Save model
    temp_model.save_model("test_model")
    
    # Clear model
    temp_model.model = None
    temp_model.scaler = None
    temp_model.feature_importances = None
    
    # Load model
    temp_model.load_model("test_model")
    
    assert temp_model.model is not None
    assert temp_model.scaler is not None
    assert temp_model.feature_importances is not None
    
    # Check predictions still work
    predictions_df = temp_model.predict(sample_features)
    assert "prediction" in predictions_df.columns

def test_full_training_pipeline(temp_model):
    """Test the full training pipeline"""
    results = temp_model.full_training_pipeline(["tt0111161"], 
                                              model_type="random_forest",
                                              tune_hyperparams=False)
    
    assert "test_metrics" in results
    assert "cross_validation" in results
    assert "data_stats" in results
    
    # Check model files were created
    model_path = temp_model.models_dir / "bot_detector.pkl"
    assert model_path.exists()