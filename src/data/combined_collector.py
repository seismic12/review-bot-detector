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
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    """A class to collect movie data from OMDb API."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """Initialize the DataCollector with configuration."""
        self.session = requests.Session()
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Set API key
        self.api_key = "cc8157f9"
        self.base_url = "http://www.omdbapi.com/"
        logger.info(f"Initialized DataCollector with data directory: {self.data_dir}")
    
    def search_movies(self, query: str) -> List[Dict]:
        """
        Search for movies using the OMDb API.
        
        Args:
            query: Search term
            
        Returns:
            List of movie dictionaries
        """
        try:
            # Construct API URL with search parameters
            params = {
                's': query,
                'type': 'movie',
                'apikey': self.api_key
            }
            
            response = self.session.get(self.base_url, params=params)
            
            if response.status_code != 200:
                st.error(f"Failed to search movies: Status {response.status_code}")
                return []
                
            data = response.json()
            
            if data.get('Response') == 'False':
                st.warning(data.get('Error', 'No results found'))
                return []
                
            # Extract relevant movie information
            movies = []
            for movie in data.get('Search', []):
                movies.append({
                    'id': movie.get('imdbID', ''),
                    'title': movie.get('Title', ''),
                    'year': movie.get('Year', ''),
                    'poster': movie.get('Poster', 'N/A')
                })
            
            return movies
            
        except Exception as e:
            st.error(f"Error searching movies: {str(e)}")
            return []

    def get_movie_metadata(self, movie_id: str) -> Dict:
        """
        Get detailed movie data using the OMDb API.
        """
        try:
            # Ensure movie_id has 'tt' prefix
            movie_id = f"tt{movie_id.replace('tt', '')}"
            
            params = {
                'i': movie_id,
                'apikey': self.api_key,
                'plot': 'full',
                'r': 'json'
            }
            
            st.write(f"Fetching data for movie: {movie_id}")
            response = self.session.get(self.base_url, params=params)
            
            if response.status_code != 200:
                st.error(f"API request failed with status {response.status_code}")
                return {}
                
            data = response.json()
            
            if data.get('Response') == 'False':
                st.warning(f"API returned error: {data.get('Error')}")
                return {}
            
            st.success("Successfully retrieved movie data")
            
            # Transform OMDb data to our format
            metadata = {
                'movie_id': movie_id,
                'title': data.get('Title'),
                'year': data.get('Year'),
                'rating': data.get('imdbRating'),
                'votes': data.get('imdbVotes'),
                'genres': data.get('Genre', '').split(', '),
                'director': data.get('Director'),
                'writer': data.get('Writer'),
                'actors': data.get('Actors', '').split(', '),
                'plot': data.get('Plot'),
                'awards': data.get('Awards'),
                'poster': data.get('Poster'),
                'ratings': data.get('Ratings', []),
                'metascore': data.get('Metascore'),
                'box_office': data.get('BoxOffice'),
                'production': data.get('Production'),
                'runtime': data.get('Runtime')
            }
            
            return metadata
            
        except Exception as e:
            st.error(f"Error collecting metadata for movie {movie_id}: {str(e)}")
            return {}

    def get_movie_reviews(self, movie_id: str, max_reviews: Optional[int] = 100) -> List[Dict]:
        """Collect movie reviews"""
        try:
            reviews = []
            clean_id = movie_id.replace('tt', '')
            
            # Initial URL without pagination key
            url = f'https://www.imdb.com/title/tt{clean_id}/reviews/_ajax?paginationKey='
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': f'https://www.imdb.com/title/tt{clean_id}/reviews'
            }
            
            st.write(f"Fetching initial reviews page...")
            
            response = self.session.get(url, headers=headers)
            st.write(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all review containers
                review_containers = soup.find_all('div', class_='lister-item-content')
                
                st.write(f"Found {len(review_containers)} reviews")
                
                for container in review_containers[:max_reviews]:
                    try:
                        # Extract rating
                        rating = None
                        rating_elem = container.find('span', class_='rating-other-user-rating')
                        if rating_elem and rating_elem.find('span'):
                            try:
                                rating = int(rating_elem.find('span').text.strip())
                            except ValueError:
                                pass
                        
                        # Extract content
                        content = ''
                        content_elem = container.find('div', class_='text show-more__control')
                        if content_elem:
                            content = content_elem.text.strip()
                        
                        # Extract title
                        title = ''
                        title_elem = container.find('a', class_='title')
                        if title_elem:
                            title = title_elem.text.strip()
                        
                        # Extract date
                        date = ''
                        date_elem = container.find('span', class_='review-date')
                        if date_elem:
                            date = date_elem.text.strip()
                        
                        # Extract user info
                        user_id = ''
                        user_elem = container.find('div', class_='display-name-date')
                        if user_elem and user_elem.find('a'):
                            href = user_elem.find('a').get('href', '')
                            user_parts = href.split('/')
                            if len(user_parts) > 2:
                                user_id = user_parts[2]
                        
                        # Extract helpful votes
                        helpful = {'up': 0, 'total': 0}
                        helpful_elem = container.find('div', class_='actions text-muted')
                        if helpful_elem:
                            helpful_text = helpful_elem.text
                            helpful_match = re.search(r'(\d+) out of (\d+)', helpful_text)
                            if helpful_match:
                                helpful = {
                                    'up': int(helpful_match.group(1)),
                                    'total': int(helpful_match.group(2))
                                }
                        
                        review = {
                            'rating': rating,
                            'date': date,
                            'helpful': helpful,
                            'title': title,
                            'content': content,
                            'user_id': user_id
                        }
                        
                        # Debug: Print found review (truncated content for readability)
                        debug_review = review.copy()
                        if debug_review['content']:
                            debug_review['content'] = debug_review['content'][:100] + '...'
                        st.write("Found review:", debug_review)
                        
                        reviews.append(review)
                        
                    except Exception as e:
                        st.warning(f"Error parsing individual review: {str(e)}")
                        continue
                
                # Look for pagination key for next page
                next_page = soup.find('div', class_='load-more-data')
                if next_page:
                    pagination_key = next_page.get('data-key')
                    if pagination_key:
                        st.write(f"Found pagination key for next page: {pagination_key}")
                
                if not reviews:
                    st.warning("No reviews were successfully parsed")
                else:
                    st.success(f"Successfully collected {len(reviews)} reviews")
                
                return reviews
                
            else:
                st.error(f"Failed to fetch reviews: {response.status_code}")
                return []
            
        except Exception as e:
            st.error(f"Error collecting reviews: {str(e)}")
            st.exception(e)
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