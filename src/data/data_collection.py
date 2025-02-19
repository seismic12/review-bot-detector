"""
Module for collecting movie review data from various sources.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from imdb import IMDb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.ia = IMDb()
        logger.info(f"Initialized DataCollector with output path: {output_path}")
    
    def collect_movie_reviews(self, movie_id: str, max_reviews: int = 100) -> dict:
        """
        Collect reviews for a specific movie from IMDB.
        
        Args:
            movie_id: IMDB movie ID (without 'tt' prefix)
            max_reviews: Maximum number of reviews to collect
            
        Returns:
            Dictionary containing movie info and reviews
        """
        try:
            logger.info(f"Collecting reviews for movie {movie_id}")
            movie = self.ia.get_movie(movie_id)
            
            # Get movie reviews
            self.ia.update(movie, 'reviews')
            
            # Extract review data
            reviews = []
            for review in movie.get('reviews', [])[:max_reviews]:
                review_data = {
                    'rating': review.get('rating'),
                    'date': review.get('date'),
                    'helpful': review.get('helpful'),
                    'title': review.get('title'),
                    'content': review.get('content'),
                    'user_id': review.get('user_id')
                }
                reviews.append(review_data)
            
            # Compile movie data
            movie_data = {
                'movie_id': movie_id,
                'title': movie.get('title'),
                'year': movie.get('year'),
                'rating': movie.get('rating'),
                'reviews': reviews,
                'collection_date': datetime.now().isoformat()
            }
            
            return movie_data
            
        except Exception as e:
            logger.error(f"Error collecting reviews for movie {movie_id}: {str(e)}")
            raise

    def save_data(self, data: dict, filename: str) -> None:
        """
        Save collected data to disk.
        
        Args:
            data: Dictionary containing collected data
            filename: Name of the file to save data to
        """
        output_file = self.output_path / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Data saved to {output_file}")
