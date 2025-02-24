"""
Module for collecting movie review data from various sources.
Handles metadata and review collection with rate limiting and error handling.
"""
from typing import List, Dict, Optional, Union
import logging
import json
from datetime import datetime
from pathlib import Path
from imdb import IMDb
import time
import re
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
        
    def get_movie_metadata(self, movie_id: str) -> Dict:
        """Collect basic metadata for a movie."""
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
        """Fetch user reviews for a movie."""
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
                    break
            
                soup = BeautifulSoup(response.text, 'html.parser')
                review_containers = soup.find_all('div', class_='review-container')
            
                if not review_containers:
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
        
            return reviews[:max_reviews]  # Ensure we don't exceed max_reviews
        
        except Exception as e:
            logger.error(f"Error fetching reviews for movie {movie_id}: {str(e)}")
            return []
        
    def collect_movie_data(self, movie_id: str, max_reviews: Optional[int] = None) -> Dict:
        """Collect all data for a movie."""
        metadata = self.get_movie_metadata(movie_id)
        if not metadata:
            return {}
    
        reviews = self.get_movie_reviews(movie_id, max_reviews=max_reviews)
        metadata['reviews'] = reviews

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