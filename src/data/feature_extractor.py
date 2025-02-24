"""
Module for extracting features from movie reviews to detect potential bot activity.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import json
from pathlib import Path
import re
import logging
from datetime import datetime, timedelta
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon.zip')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

class FeatureExtractor:
    """A class to extract features from movie reviews for bot detection."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """Initialize the FeatureExtractor with configuration."""
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.features_dir = self.data_dir / "features"
    
        # Create directories if they don't exist
        self.features_dir.mkdir(parents=True, exist_ok=True)
    
        # Download required NLTK resources
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            logger.warning("Could not download NLTK resources automatically")
    
        # Initialize NLTK tools
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
        logger.info(f"Initialized FeatureExtractor with data directory: {data_dir}")
    
    def load_processed_reviews(self, movie_id: str) -> List[Dict]:
        """
        Load processed reviews for a specific movie.
        
        Args:
            movie_id: IMDB movie ID (with or without 'tt' prefix)
            
        Returns:
            List of processed review dictionaries
        """
        try:
            # Ensure movie_id has 'tt' prefix
            if not movie_id.startswith('tt'):
                movie_id = f"tt{movie_id}"
                
            file_path = self.processed_dir / f"{movie_id}_processed.json"
            
            if not file_path.exists():
                logger.error(f"Processed reviews file not found: {file_path}")
                return []
                
            with open(file_path, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
                
            logger.info(f"Loaded {len(reviews)} processed reviews for movie {movie_id}")
            return reviews
            
        except Exception as e:
            logger.error(f"Error loading processed reviews for movie {movie_id}: {str(e)}")
            return []
    
    def extract_text_features(self, review: Dict) -> Dict:
        """
        Extract features from review text content.
        
        Args:
            review: Dictionary containing processed review data
            
        Returns:
            Dictionary of text-based features
        """
        content = review.get('content', '')
        title = review.get('title', '')
        
        # Skip empty reviews
        if not content:
            return {}
            
        # Basic statistics
        word_count = len(content.split())
        char_count = len(content)
        avg_word_length = char_count / max(word_count, 1)
        
        # Count special characters
        exclamation_count = content.count('!')
        question_count = content.count('?')
        uppercase_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
        
        # Tokenize and analyze
        tokens = word_tokenize(content.lower())
        non_stopwords = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        
        # Vocabulary richness
        unique_words = len(set(non_stopwords))
        richness = unique_words / max(len(non_stopwords), 1)
        
        # Sentiment analysis
        title_sentiment = self.sentiment_analyzer.polarity_scores(title)
        content_sentiment = self.sentiment_analyzer.polarity_scores(content)
        
        # Sentiment consistency
        sentiment_diff = abs(title_sentiment['compound'] - content_sentiment['compound'])
        
        # Check for suspicious patterns
        repeated_chars = max(len(match) for match in re.findall(r'(.)\1+', content)) if re.search(r'(.)\1+', content) else 0
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'uppercase_ratio': uppercase_ratio,
            'vocabulary_richness': richness,
            'title_sentiment': title_sentiment['compound'],
            'content_sentiment': content_sentiment['compound'],
            'content_negativity': content_sentiment['neg'],
            'content_positivity': content_sentiment['pos'],
            'sentiment_diff': sentiment_diff,
            'repeated_chars_max': repeated_chars
        }
    
    def extract_metadata_features(self, review: Dict, all_reviews: List[Dict]) -> Dict:
        """
        Extract features from review metadata and context.
        
        Args:
            review: Dictionary containing processed review data
            all_reviews: List of all reviews for the movie
            
        Returns:
            Dictionary of metadata-based features
        """
        # Review rating deviation from mean
        all_ratings = [r.get('rating', 0) for r in all_reviews if r.get('rating') is not None]
        mean_rating = np.mean(all_ratings) if all_ratings else 0
        
        rating = review.get('rating')
        if rating is None:
            rating_deviation = 0
        else:
            rating_deviation = abs(rating - mean_rating)
        
        # Helpfulness features
        helpful_ratio = review.get('helpful_ratio', 0)
        
        # Date-related features (reviews posted very close to release date might be suspicious)
        review_date = datetime.fromisoformat(review.get('date', datetime.now().isoformat()))
        
        # Find the earliest and latest reviews
        valid_dates = [datetime.fromisoformat(r.get('date')) for r in all_reviews if 'date' in r]
        earliest_date = min(valid_dates) if valid_dates else review_date
        latest_date = max(valid_dates) if valid_dates else review_date
        
        # Calculate time-based features
        days_since_earliest = (review_date - earliest_date).days
        days_until_latest = (latest_date - review_date).days
        review_timespan = (latest_date - earliest_date).days if valid_dates else 0
        normalized_timing = days_since_earliest / max(review_timespan, 1) if review_timespan > 0 else 0
        
        return {
            'rating': rating,
            'rating_deviation': rating_deviation,
            'helpful_ratio': helpful_ratio,
            'days_since_earliest': days_since_earliest,
            'days_until_latest': days_until_latest,
            'normalized_timing': normalized_timing
        }
    
    def detect_review_bursts(self, all_reviews: List[Dict]) -> Dict[str, List[str]]:
        """
        Detect suspicious bursts of reviews within short time periods.
        
        Args:
            all_reviews: List of all reviews for the movie
            
        Returns:
            Dictionary mapping time windows to lists of review IDs in those windows
        """
        # Sort reviews by date
        valid_reviews = [r for r in all_reviews if 'date' in r]
        sorted_reviews = sorted(valid_reviews, key=lambda x: datetime.fromisoformat(x.get('date')))
        
        # Look for bursts using a sliding window approach
        window_size = timedelta(hours=24)
        suspicious_windows = {}
        
        for i, review in enumerate(sorted_reviews):
            review_date = datetime.fromisoformat(review.get('date'))
            window_end = review_date + window_size
            
            # Count reviews in this window
            reviews_in_window = [r for r in sorted_reviews if 
                                review_date <= datetime.fromisoformat(r.get('date')) <= window_end]
            
            # If many reviews in a short window, flag as suspicious
            if len(reviews_in_window) >= 5:  # Threshold can be adjusted
                window_key = f"{review_date.isoformat()}_{window_end.isoformat()}"
                suspicious_windows[window_key] = [r.get('user_id', 'unknown') for r in reviews_in_window]
        
        return suspicious_windows
    
    def extract_similarity_features(self, review: Dict, all_reviews: List[Dict]) -> Dict:
        """
        Detect similarity between reviews which might indicate copying or templates.
        
        Args:
            review: Current review
            all_reviews: All reviews for comparison
            
        Returns:
            Dictionary of similarity-based features
        """
        # Extract all review contents
        review_contents = [r.get('content', '') for r in all_reviews if len(r.get('content', '')) > 0]
        
        # Skip if not enough reviews for comparison
        if len(review_contents) < 2:
            return {
                'max_similarity': 0,
                'avg_similarity': 0,
                'similar_reviews_count': 0
            }
        
        # Use TF-IDF to measure content similarity
        vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        
        try:
            tfidf_matrix = vectorizer.fit_transform(review_contents)
            
            # Get index of current review
            current_index = review_contents.index(review.get('content', ''))
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_row = cosine_similarity(tfidf_matrix[current_index:current_index+1], tfidf_matrix)[0]
            
            # Remove self-similarity (always 1.0)
            similarity_row = np.delete(similarity_row, current_index)
            
            # Extract similarity metrics
            max_similarity = np.max(similarity_row) if len(similarity_row) > 0 else 0
            avg_similarity = np.mean(similarity_row) if len(similarity_row) > 0 else 0
            similar_reviews = sum(1 for sim in similarity_row if sim > 0.5)  # Threshold for similarity
            
            return {
                'max_similarity': max_similarity,
                'avg_similarity': avg_similarity,
                'similar_reviews_count': similar_reviews
            }
        except Exception as e:
            logger.warning(f"Error calculating similarity features: {str(e)}")
            return {
                'max_similarity': 0,
                'avg_similarity': 0,
                'similar_reviews_count': 0
            }
    
    def extract_all_features(self, movie_id: str) -> pd.DataFrame:
        """
        Extract all features for all reviews of a movie.
        
        Args:
            movie_id: IMDB movie ID
            
        Returns:
            DataFrame containing all extracted features
        """
        reviews = self.load_processed_reviews(movie_id)
        if not reviews:
            logger.error(f"No reviews found for movie {movie_id}")
            return pd.DataFrame()
        
        # Detect review bursts across all reviews
        burst_windows = self.detect_review_bursts(reviews)
        
        # Extract features for each review
        all_features = []
        
        for review in reviews:
            try:
                # Basic identification
                review_id = review.get('user_id', 'unknown')
                
                # Extract features
                text_features = self.extract_text_features(review)
                metadata_features = self.extract_metadata_features(review, reviews)
                similarity_features = self.extract_similarity_features(review, reviews)
                
                # Check if review is part of a burst
                in_burst = any(review.get('user_id', 'unknown') in user_ids 
                              for user_ids in burst_windows.values())
                
                # Combine all features
                features = {
                    'review_id': review_id,
                    'in_burst': in_burst,
                    **text_features,
                    **metadata_features,
                    **similarity_features
                }
                
                all_features.append(features)
                
            except Exception as e:
                logger.warning(f"Error extracting features for review {review.get('user_id', 'unknown')}: {str(e)}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        logger.info(f"Extracted features for {len(df)} reviews of movie {movie_id}")
        return df
    
    def save_features(self, movie_id: str, features_df: pd.DataFrame) -> None:
        """
        Save extracted features to disk.
        
        Args:
            movie_id: IMDB movie ID
            features_df: DataFrame containing extracted features
        """
        if features_df.empty:
            logger.warning(f"No features to save for movie {movie_id}")
            return
            
        # Ensure movie_id has 'tt' prefix
        if not movie_id.startswith('tt'):
            movie_id = f"tt{movie_id}"
        
        # Save as CSV
        output_path = self.features_dir / f"{movie_id}_features.csv"
        features_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved features to {output_path}")
    
    def extract_and_save_features(self, movie_id: str) -> pd.DataFrame:
        """
        Extract all features for a movie and save them to disk.
        
        Args:
            movie_id: IMDB movie ID
            
        Returns:
            DataFrame containing all extracted features
        """
        features_df = self.extract_all_features(movie_id)
        self.save_features(movie_id, features_df)
        return features_df