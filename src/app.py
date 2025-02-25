"""
Streamlit frontend for the IMDb Review Bot Detection System
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import plotly.express as px
from imdb import Cinemagoer  # Updated import for newer versions
from data.combined_collector import DataCollector
from data.feature_extractor import FeatureExtractor
from data.bot_detector_model import BotDetectorModel
import time
from datetime import datetime
import requests

# Configure page
st.set_page_config(
    page_title="IMDb Review Bot Detector",
    page_icon="🤖",
    layout="wide"
)

# Initialize IMDb API
ia = Cinemagoer()

# Initialize data directory
DATA_DIR = Path("data")
for subdir in ["raw", "processed", "features", "models", "results"]:
    (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)

def search_movie(query: str) -> list:
    """Search for movies and return results"""
    try:
        movies = ia.search_movie(query)
        # Filter out non-movies (TV shows, video games, etc.) and format results
        movie_results = []
        for movie in movies:
            if movie.get('kind') == 'movie':
                # Get year if available
                year = f" ({movie.get('year')})" if movie.get('year') else ""
                movie_results.append({
                    'title': f"{movie['title']}{year}",
                    'id': movie.getID(),
                    'year': movie.get('year')
                })
        return movie_results[:10]  # Limit to top 10 results
    except Exception as e:
        st.error(f"Error searching for movie: {str(e)}")
        return []

def initialize_components():
    """Initialize the main components of the system"""
    collector = DataCollector(DATA_DIR)
    feature_extractor = FeatureExtractor(DATA_DIR)
    model = BotDetectorModel(DATA_DIR)
    return collector, feature_extractor, model

def collect_movie_data(collector: DataCollector, movie_id: str, max_reviews: int):
    """Collect movie data with progress bar"""
    try:
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        # Step 1: Check if data already exists
        processed_file = DATA_DIR / "processed" / f"{movie_id}_processed.json"
        if processed_file.exists():
            status_container.info("Found existing processed data. Loading...")
            with open(processed_file, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
            st.success(f"Loaded {len(processed_data)} existing reviews")
            return {
                "status": "success",
                "movie_id": movie_id,
                "reviews_collected": len(processed_data),
                "reviews_processed": len(processed_data)
            }
        
        # Step 2: Collect new data
        status_container.info("Collecting new data...")
        result = collector.get_movie_reviews(movie_id, max_reviews)
        
        if result["status"] == "error":
            status_container.error(result["message"])
            return result
        
        # Debug info
        st.write(f"Reviews collected: {result['reviews_collected']}")
        
        # Step 3: Extract features
        if result["reviews_collected"] > 0:
            status_container.info("Extracting features...")
            feature_extractor = FeatureExtractor(DATA_DIR)
            features_df = feature_extractor.extract_and_save_features(movie_id)
            
            if not features_df.empty:
                st.success(f"Successfully extracted {len(features_df)} feature sets")
            else:
                st.warning("No features were extracted")
        
        return result
        
    except Exception as e:
        st.error(f"Error in collection pipeline: {str(e)}")
        return {"status": "error", "message": str(e)}

def extract_features(extractor: FeatureExtractor, movie_id: str):
    """Extract features with progress bar"""
    with st.spinner("Extracting features..."):
        features_df = extractor.extract_and_save_features(movie_id)
        if not features_df.empty:
            st.success(f"Successfully extracted features for {len(features_df)} reviews")
        else:
            st.error("Failed to extract features")
    return features_df

def train_model(model: BotDetectorModel, movie_ids: list):
    """Train the model with progress bar"""
    with st.spinner("Training model..."):
        results = model.full_training_pipeline(
            movie_ids=movie_ids,
            model_type=st.session_state.get('model_type', 'random_forest'),
            tune_hyperparams=st.session_state.get('tune_hyperparams', True)
        )
        if results:
            st.success("Model training completed!")
        else:
            st.error("Model training failed")
    return results

def display_results(results: dict):
    """Display model results with charts"""
    if not results:
        return

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            'Value': [
                results['test_metrics']['accuracy'],
                results['test_metrics']['precision'],
                results['test_metrics']['recall'],
                results['test_metrics']['f1'],
                results['test_metrics']['roc_auc']
            ]
        })
        st.plotly_chart(px.bar(metrics_df, x='Metric', y='Value', title='Model Metrics'))

    with col2:
        st.subheader("Data Statistics")
        stats = results['data_stats']
        fig = px.pie(
            values=[stats['legitimate_reviews'], stats['suspicious_reviews']],
            names=['Legitimate', 'Suspicious'],
            title='Review Distribution'
        )
        st.plotly_chart(fig)

def analyze_reviews(collector: DataCollector, movie_id: str):
    """Analyze movie reviews for suspicious patterns"""
    try:
        # TMDb API configuration
        TMDB_API_KEY = "d8a2aefcabacecfced07366ffb2d5d2b"
        tmdb_base_url = "https://api.themoviedb.org/3"
        
        st.write("Fetching reviews from TMDb...")
        
        # First get IMDb data for the title
        movie_data = collector.get_movie_metadata(movie_id)
        if not movie_data:
            st.error("Failed to fetch movie data")
            return
            
        # Get TMDb movie ID using IMDb ID
        find_url = f"{tmdb_base_url}/find/{movie_id}?api_key={TMDB_API_KEY}&external_source=imdb_id"
        response = requests.get(find_url)
        tmdb_data = response.json()
        
        if not tmdb_data.get('movie_results'):
            st.error("Could not find movie on TMDb")
            return
            
        tmdb_id = tmdb_data['movie_results'][0]['id']
        st.success(f"Found movie on TMDb (ID: {tmdb_id})")
        
        # Get all pages of reviews
        all_reviews = []
        page = 1
        max_pages = 10  # Adjust this to control how many pages to fetch
        
        with st.spinner("Fetching reviews..."):
            progress_bar = st.progress(0)
            
            while page <= max_pages:
                reviews_url = f"{tmdb_base_url}/movie/{tmdb_id}/reviews?api_key={TMDB_API_KEY}&page={page}"
                response = requests.get(reviews_url)
                reviews_data = response.json()
                
                if not reviews_data.get('results'):
                    break
                    
                all_reviews.extend(reviews_data['results'])
                
                # Update progress
                progress = min(page / max_pages * 100, 100)
                progress_bar.progress(int(progress))
                
                # Check if we've reached the last page
                if page >= reviews_data.get('total_pages', 1):
                    break
                    
                page += 1
        
        if not all_reviews:
            st.warning("No reviews found for analysis")
            return
            
        st.success(f"Found {len(all_reviews)} reviews")
        
        # Process and analyze reviews
        # Sort reviews by date
        all_reviews.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", len(all_reviews))
        with col2:
            avg_length = np.mean([len(r['content'].split()) for r in all_reviews])
            st.metric("Avg Review Length", f"{avg_length:.0f} words")
        with col3:
            ratings = [r.get('author_details', {}).get('rating') for r in all_reviews]
            valid_ratings = [r for r in ratings if r is not None]
            if valid_ratings:
                avg_rating = np.mean(valid_ratings)
                st.metric("Avg Rating", f"{avg_rating:.1f}/10")
        with col4:
            recent_count = sum(1 for r in all_reviews if r.get('created_at', '')[:4] >= '2023')
            st.metric("Recent Reviews", recent_count)
        
        # Show reviews with more details
        st.subheader("Reviews")
        
        # Add sorting options
        sort_option = st.selectbox(
            "Sort reviews by:",
            ["Most Recent", "Highest Rating", "Lowest Rating", "Longest", "Shortest"]
        )
        
        # Sort reviews based on selection
        if sort_option == "Most Recent":
            sorted_reviews = sorted(all_reviews, key=lambda x: x.get('created_at', ''), reverse=True)
        elif sort_option == "Highest Rating":
            sorted_reviews = sorted(all_reviews, 
                                 key=lambda x: x.get('author_details', {}).get('rating', 0) or 0, 
                                 reverse=True)
        elif sort_option == "Lowest Rating":
            sorted_reviews = sorted(all_reviews, 
                                 key=lambda x: x.get('author_details', {}).get('rating', 0) or 0)
        elif sort_option == "Longest":
            sorted_reviews = sorted(all_reviews, 
                                 key=lambda x: len(x.get('content', '').split()), 
                                 reverse=True)
        else:  # Shortest
            sorted_reviews = sorted(all_reviews, 
                                 key=lambda x: len(x.get('content', '').split()))
        
        for review in sorted_reviews:
            with st.expander(f"Review by {review['author']} - {review['created_at'][:10]}"):
                rating = review.get('author_details', {}).get('rating')
                if rating:
                    st.write(f"**Rating:** {'⭐' * int(rating/2)}")
                st.write(f"**Content:** {review['content'][:500]}...")
                word_count = len(review['content'].split())
                st.write(f"**Word count:** {word_count}")
                if review.get('url'):
                    st.write(f"[Read full review]({review['url']})")
        
    except Exception as e:
        st.error(f"Error analyzing reviews: {str(e)}")
        st.exception(e)

def analyze_movie_ratings(collector: DataCollector, movie_id: str):
    """Analyze movie ratings for suspicious patterns"""
    try:
        # Get movie data from OMDb
        st.write("Collecting movie data...")
        movie_data = collector.get_movie_metadata(movie_id)
        
        if not movie_data:
            st.error("Failed to fetch movie data")
            return
            
        st.subheader(f"Analysis for {movie_data['title']} ({movie_data['year']})")
        
        # Display basic movie info
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(movie_data['poster'], use_container_width=True)
        with col2:
            st.write(f"**IMDb Rating:** ⭐ {movie_data['rating']}/10")
            st.write(f"**Total Votes:** {movie_data['votes']}")
            
            # Display all ratings sources
            st.write("**Ratings from different sources:**")
            for rating in movie_data['ratings']:
                st.write(f"- {rating['Source']}: {rating['Value']}")
        
        # Analyze ratings distribution
        if movie_data['ratings']:
            st.subheader("Ratings Analysis")
            
            # Convert ratings to numeric values for analysis
            ratings_data = []
            for rating in movie_data['ratings']:
                try:
                    # Convert different rating formats to percentage
                    if '/' in rating['Value']:
                        num, den = rating['Value'].split('/')
                        value = (float(num) / float(den)) * 100
                    elif '%' in rating['Value']:
                        value = float(rating['Value'].replace('%', ''))
                    else:
                        value = float(rating['Value'])
                    ratings_data.append({
                        'source': rating['Source'],
                        'value': value
                    })
                except:
                    continue
            
            # Check for rating inconsistencies
            if len(ratings_data) > 1:
                values = [r['value'] for r in ratings_data]
                mean_rating = np.mean(values)
                std_rating = np.std(values)
                
                st.write("**Rating Consistency Analysis:**")
                
                if std_rating > 20:  # High variance in ratings
                    st.warning("⚠️ High variance detected between rating sources")
                    st.write(f"Standard deviation: {std_rating:.2f}%")
                    
                    # Identify outliers
                    for rating in ratings_data:
                        if abs(rating['value'] - mean_rating) > std_rating:
                            st.write(f"- {rating['source']} rating differs significantly from average")
                else:
                    st.success("✅ Ratings are consistent across sources")
            
            # IMDb votes analysis
            if movie_data['votes']:
                votes = int(movie_data['votes'].replace(',', ''))
                if votes < 1000:
                    st.warning("⚠️ Low number of votes - ratings may not be representative")
                elif votes > 100000:
                    st.success("✅ Large number of votes - ratings likely reliable")
                
                st.write(f"**Vote Analysis:**")
                st.write(f"- Total votes: {votes:,}")
                st.write(f"- Average rating: {movie_data['rating']}/10")
                
                # Calculate votes per year
                try:
                    year = int(movie_data['year'])
                    current_year = datetime.now().year
                    years_since_release = current_year - year
                    votes_per_year = votes / max(1, years_since_release)
                    
                    st.write(f"- Average votes per year: {votes_per_year:,.0f}")
                    
                    if votes_per_year < 100:
                        st.warning("⚠️ Low engagement rate")
                    elif votes_per_year > 10000:
                        st.success("✅ High engagement rate")
                except:
                    pass
        
    except Exception as e:
        st.error(f"Error analyzing movie: {str(e)}")
        st.exception(e)

def main():
    st.title("🎬 Movie Review & Rating Analyzer")
    
    # Initialize collector
    collector = DataCollector(DATA_DIR)
    
    # Movie search interface
    movie_query = st.text_input("Search for a movie", placeholder="Enter movie title...")
    
    if movie_query:
        with st.spinner("Searching..."):
            movies = collector.search_movies(movie_query)
            
        if movies:
            cols = st.columns(6)
            for i, movie in enumerate(movies[:6]):
                with cols[i % 6]:
                    st.image(
                        movie['poster'] if movie['poster'] != 'N/A' else 'https://via.placeholder.com/150x225?text=No+Poster',
                        caption=f"{movie['title']}\n({movie['year']})",
                        use_container_width=True
                    )
                    if st.button("Analyze", key=movie['id'], use_container_width=True):
                        # Create tabs for different analyses
                        tab1, tab2 = st.tabs(["Ratings Analysis", "Review Analysis"])
                        
                        with tab1:
                            analyze_movie_ratings(collector, movie['id'])
                            
                        with tab2:
                            analyze_reviews(collector, movie['id'])
        else:
            st.warning("No movies found matching your search.")

if __name__ == "__main__":
    main() 