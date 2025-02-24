"""
Module for collecting movie review data from IMDb.
Combines API usage for metadata with web scraping for reviews.
"""
from typing import List, Dict, Optional, Union
import logging
import json
import time
import re
from datetime import datetime
from pathlib import Path
from imdb import IMDb
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    """A class to collect movie reviews and metadata from IMDB."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """Initialize the DataCollector with configuration."""
        self.session = requests.Session()
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.ia = IMDb()
        logger.info(f"Initialized DataCollector with data directory: {self.data_dir}")
        
    def get_movie_metadata(self, movie_id: str) -> Dict:
        """
        Collect basic metadata for a movie using the IMDb package.
        
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
                'movie_id': f"tt{clean_id}",  # Always include tt prefix
                'title': movie.get('title'),
                'year': movie.get('year'),
                'rating': movie.get('rating'),
                'genres': movie.get('genres', []),
                'cast': [a.get('name') for a in movie.get('cast', [])[:5]]
            }
            return metadata
        except Exception as e:
            logger.error(f"Error collecting metadata for movie {movie_id}: {str(e)}")
            return {}

    def get_movie_reviews(self, movie_id: str, max_reviews: Optional[int] = 100) -> List[Dict]:
        """
        Fetch user reviews for a movie via web scraping.
        
        Args:
            movie_id: IMDB movie ID
            max_reviews: Maximum number of reviews to collect
            
        Returns:
            List of review dictionaries
        """
        try:
            reviews = []
            page = 1
            movie_id = movie_id.replace('tt', '')
        
            while len(reviews) < max_reviews:
                time.sleep(1)  # Rate limiting
            
                url = f'https://www.imdb.com/title/tt{movie_id}/reviews'
                if page > 1:
                    url += f'?page={page}'
            
                response = self.session.get(url)
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch page {page} for movie tt{movie_id}: Status {response.status_code}")
                    break
            
                soup = BeautifulSoup(response.text, 'html.parser')
                review_containers = soup.find_all('div', class_='review-container')
            
                if not review_containers:
                    logger.info(f"No more reviews found on page {page} for movie tt{movie_id}")
                    break
            
                for container in review_containers:
                    if len(reviews) >= max_reviews:
                        break
                    
                    try:
                        # Extract rating
                        rating_elem = container.find('span', class_='rating-other-user-rating')
                        rating = rating_elem.find('span').text if rating_elem else None
                    
                        # Extract helpful votes
                        helpful_elem = container.find('div', class_='actions text-muted')
                        helpful_text = helpful_elem.text if helpful_elem else ''
                        helpful_match = re.search(r'(\d+) out of (\d+)', helpful_text)
                        helpful = {'up': 0, 'total': 0}
                        if helpful_match:
                            helpful = {
                                'up': int(helpful_match.group(1)),
                                'total': int(helpful_match.group(2))
                            }
                    
                        review = {
                            'rating': rating,
                            'date': container.find('span', class_='review-date').text,
                            'helpful': helpful,
                            'title': container.find('a', class_='title').text.strip(),
                            'content': container.find('div', class_='text').text.strip(),
                            'user_id': container.find('a', class_='display-name-link')['href'].split('/')[-2]
                        }
                        reviews.append(review)
                    
                    except Exception as e:
                        logger.warning(f"Error parsing review: {str(e)}")
                        continue
            
                page += 1
                logger.info(f"Collected {len(reviews)} reviews for movie tt{movie_id} (page {page-1})")
        
            return reviews[:max_reviews]  # Ensure we don't exceed max_reviews
        
        except Exception as e:
            logger.error(f"Error fetching reviews for movie tt{movie_id}: {str(e)}")
            return []
    
    def collect_movie_data(self, movie_id: str, max_reviews: Optional[int] = None) -> Dict:
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
            logger.error(f"Failed to retrieve metadata for movie {movie_id}")
            return {}
    
        reviews = self.get_movie_reviews(movie_id, max_reviews=max_reviews)
        metadata['reviews'] = reviews
        metadata['review_count'] = len(reviews)

        return metadata

    def save_data(self, data: Dict, filename: str) -> None:
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
        
    def save_processed_data(self, movie_id: str, processed_data: List[Dict]) -> None:
        """
        Save processed review data to disk.
        
        Args:
            movie_id: IMDB movie ID
            processed_data: List of processed review dictionaries
        """
        output_file = self.processed_dir / f"{movie_id}_processed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Processed data saved to {output_file}")
        
    def full_collection_pipeline(self, movie_id: str, max_reviews: Optional[int] = 100) -> Dict:
        """
        Run the full data collection and processing pipeline for a movie.
        
        Args:
            movie_id: IMDB movie ID
            max_reviews: Maximum number of reviews to collect
            
        Returns:
            Dictionary containing collection statistics
        """
        # Collect data
        logger.info(f"Starting data collection for movie {movie_id}")
        movie_data = self.collect_movie_data(movie_id, max_reviews=max_reviews)
        
        if not movie_data:
            logger.error(f"Failed to collect data for movie {movie_id}")
            return {"status": "error", "message": "Failed to collect movie data"}
        
        # Save raw data
        self.save_data(movie_data, f"{movie_id}_raw.json")
        
        # Process reviews
        processed_reviews = self.process_reviews(movie_data['reviews'])
        
        # Save processed data
        self.save_processed_data(movie_id, processed_reviews)
        
        return {
            "status": "success",
            "movie_id": movie_id,
            "title": movie_data.get("title"),
            "reviews_collected": len(movie_data.get("reviews", [])),
            "reviews_processed": len(processed_reviews)
        }