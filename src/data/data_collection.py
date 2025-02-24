"""
Module for collecting movie review data from various sources.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from imdb import IMDb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.ia = IMDb()
        logger.info(f"Initialized DataCollector with data directory: {data_dir}")
    
    def get_movie_metadata(self, movie_id: str) -> dict:
        """
        Collect basic metadata for a movie.
        
        Args:
            movie_id: IMDB movie ID (with or without 'tt' prefix)
            
        Returns:
            Dictionary containing movie metadata
        """
        try:
            # Remove 'tt' prefix if present
            clean_id = movie_id.replace('tt', '')
            movie = self.ia.get_movie(clean_id)
            
            metadata = {
                'movie_id': f"tt{clean_id}",  # Add tt prefix to match test expectation
                'title': movie.get('title'),
                'year': movie.get('year'),
                'rating': movie.get('rating'),
                'genres': movie.get('genres', []),
                'cast': [a.get('name') for a in movie.get('cast', [])[:5]]  # Top 5 cast
            }
            return metadata
            
        except Exception as e:
            logger.error(f"Error collecting metadata for movie {movie_id}: {str(e)}")
            return {}

    def get_movie_reviews(self, movie_id: str, max_reviews: int = 100) -> List[Dict]:
        """
        Collect reviews for a specific movie from IMDB.
        
        Args:
            movie_id: IMDB movie ID
            max_reviews: Maximum number of reviews to collect
            
        Returns:
            List of review dictionaries
        """
        try:
            # Get movie reviews
            movie = self.ia.get_movie(movie_id.replace('tt', ''))
            self.ia.update(movie, 'reviews')
            
            # Extract review data
            reviews = []
            for review in movie.get('reviews', [])[:max_reviews]:
                review_data = {
                    'rating': review.get('rating'),
                    'date': review.get('date'),
                    'helpful': {
                        'up': review.get('helpful', {}).get('up', 0),
                        'total': review.get('helpful', {}).get('total', 0)
                    },
                    'title': review.get('title', ''),
                    'content': review.get('content', ''),
                    'user_id': review.get('user_id', '')
                }
                reviews.append(review_data)
            
            return reviews
            
        except Exception as e:
            logger.error(f"Error collecting reviews for movie {movie_id}: {str(e)}")
            return []

    def collect_movie_data(self, movie_id: str, max_reviews: Optional[int] = None) -> dict:
        """
        Collect all data for a movie.
        
        Args:
            movie_id: IMDB movie ID
            max_reviews: Maximum number of reviews to collect
            
        Returns:
            Dictionary containing movie data and reviews
        """
        metadata = self.get_movie_metadata(movie_id)
        if not metadata:
            return {}
            
        reviews = self.get_movie_reviews(movie_id, max_reviews=max_reviews if max_reviews else 100)
        metadata['reviews'] = reviews
        
        return metadata

    def save_data(self, data: dict, filename: str) -> None:
        """
        Save collected data to disk.
        
        Args:
            data: Dictionary containing collected data
            filename: Name of the file to save data to
        """
        output_file = self.raw_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Data saved to {output_file}")

    def process_reviews(self, reviews: List[Dict]) -> List[Dict]:
        """
        Process and clean review data.
        
        Args:
            reviews: List of raw review dictionaries
            
        Returns:
            List of processed review dictionaries
        """
        processed = []
        for review in reviews:
            try:
                # Convert date string to ISO format
                date = datetime.strptime(review['date'], '%d %B %Y')
                
                processed.append({
                    'rating': float(review['rating']) if review['rating'] else None,
                    'date': date.isoformat(),
                    'helpful_ratio': (
                        review['helpful']['up'] / review['helpful']['total']
                        if review['helpful']['total'] > 0 else 0
                    ),
                    'title': review['title'],
                    'content': review['content'],
                    'content_length': len(review['content']),
                    'user_id': review['user_id']
                })
                
            except Exception as e:
                logger.warning(f"Error processing review: {str(e)}")
                continue
                
        return processed