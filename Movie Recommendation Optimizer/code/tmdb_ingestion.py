#!/usr/bin/env python3
"""
TMDB Data Ingestion Script
Fetches movie data from TMDB API and saves raw JSON + normalized data
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tmdb_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TMDBIngestion:
    def __init__(self):
        self.api_key = os.getenv('TMDB_API_KEY')
        if not self.api_key:
            raise ValueError("TMDB_API_KEY not found in environment variables")
        
        self.base_url = "https://api.themoviedb.org/3"
        self.session = requests.Session()
        self.session.params.update({'api_key': self.api_key})
        
        # Create directories
        Path('data/raw/tmdb').mkdir(parents=True, exist_ok=True)
        Path('data/normalized').mkdir(parents=True, exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        
    def get_popular_movies(self, page: int = 1) -> Dict:
        """Fetch popular movies from TMDB"""
        url = f"{self.base_url}/movie/popular"
        params = {'page': page}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching popular movies page {page}: {e}")
            return {}
    
    def get_movie_details(self, movie_id: int) -> Dict:
        """Fetch detailed movie information including credits and streaming providers"""
        # Get basic movie info
        movie_url = f"{self.base_url}/movie/{movie_id}"
        movie_response = self.session.get(movie_url)
        
        if movie_response.status_code != 200:
            return {}
        
        movie_data = movie_response.json()
        
        # Get credits (cast and crew)
        credits_url = f"{self.base_url}/movie/{movie_id}/credits"
        credits_response = self.session.get(credits_url)
        if credits_response.status_code == 200:
            movie_data['credits'] = credits_response.json()
        
        # Get streaming providers (US)
        providers_url = f"{self.base_url}/movie/{movie_id}/watch/providers"
        providers_response = self.session.get(providers_url)
        if providers_response.status_code == 200:
            movie_data['streaming_providers'] = providers_response.json()
        
        return movie_data
    
    def fetch_movies_batch(self, max_pages: int = 10) -> List[Dict]:
        """Fetch movies in batches to avoid rate limiting"""
        all_movies = []
        
        for page in range(1, max_pages + 1):
            logger.info(f"Fetching popular movies page {page}")
            
            popular_data = self.get_popular_movies(page)
            if not popular_data or 'results' not in popular_data:
                break
            
            movies = popular_data['results']
            logger.info(f"Found {len(movies)} movies on page {page}")
            
            # Get detailed info for each movie
            for movie in movies:
                movie_id = movie['id']
                logger.info(f"Fetching details for movie ID: {movie_id}")
                
                detailed_movie = self.get_movie_details(movie_id)
                if detailed_movie:
                    all_movies.append(detailed_movie)
                
                # Rate limiting - TMDB allows 40 requests per 10 seconds
                time.sleep(0.25)
            
            # Save raw data after each page
            self.save_raw_data(all_movies, f"page_{page}")
        
        return all_movies
    
    def save_raw_data(self, data: List[Dict], suffix: str = ""):
        """Save raw JSON data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/raw/tmdb/tmdb_movies_{suffix}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved raw data to {filename}")
    
    def normalize_data(self, movies: List[Dict]) -> pd.DataFrame:
        """Convert raw JSON to normalized DataFrame"""
        normalized_data = []
        
        for movie in movies:
            # Extract basic info
            movie_info = {
                'tmdb_id': movie.get('id'),
                'title': movie.get('title'),
                'original_title': movie.get('original_title'),
                'overview': movie.get('overview'),
                'popularity': movie.get('popularity'),
                'vote_average': movie.get('vote_average'),
                'vote_count': movie.get('vote_count'),
                'release_date': movie.get('release_date'),
                'runtime': movie.get('runtime'),
                'budget': movie.get('budget'),
                'revenue': movie.get('revenue'),
                'status': movie.get('status'),
                'tagline': movie.get('tagline'),
                'adult': movie.get('adult'),
                'video': movie.get('video'),
                'original_language': movie.get('original_language')
            }
            
            # Extract genres
            genres = movie.get('genres', [])
            movie_info['genres'] = ', '.join([g['name'] for g in genres])
            
            # Extract cast (top 10)
            credits = movie.get('credits', {})
            cast = credits.get('cast', [])
            movie_info['cast'] = ', '.join([p['name'] for p in cast[:10]])
            
            # Extract crew (director, producer)
            crew = credits.get('crew', [])
            directors = [p['name'] for p in crew if p['job'] == 'Director']
            producers = [p['name'] for p in crew if p['job'] == 'Producer']
            movie_info['directors'] = ', '.join(directors)
            movie_info['producers'] = ', '.join(producers)
            
            # Extract streaming providers
            providers = movie.get('streaming_providers', {})
            us_providers = providers.get('results', {}).get('US', {})
            flatrate = us_providers.get('flatrate', [])
            movie_info['streaming_providers'] = ', '.join([p['provider_name'] for p in flatrate])
            
            normalized_data.append(movie_info)
        
        df = pd.DataFrame(normalized_data)
        return df
    
    def save_normalized_data(self, df: pd.DataFrame):
        """Save normalized data as Parquet and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as Parquet
        parquet_file = f"data/normalized/tmdb_movies_{timestamp}.parquet"
        df.to_parquet(parquet_file, index=False)
        logger.info(f"Saved normalized Parquet data to {parquet_file}")
        
        # Save as CSV
        csv_file = f"data/normalized/tmdb_movies_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved normalized CSV data to {parquet_file}")
        
        return parquet_file, csv_file

def main():
    """Main execution function"""
    try:
        logger.info("Starting TMDB data ingestion")
        
        # Initialize ingestion
        tmdb = TMDBIngestion()
        
        # Fetch movies
        movies = tmdb.fetch_movies_batch(max_pages=5)  # Start with 5 pages
        
        if movies:
            logger.info(f"Successfully fetched {len(movies)} movies")
            
            # Normalize data
            df = tmdb.normalize_data(movies)
            logger.info(f"Normalized data shape: {df.shape}")
            
            # Save normalized data
            tmdb.save_normalized_data(df)
            
            logger.info("TMDB ingestion completed successfully")
        else:
            logger.error("No movies were fetched")
            
    except Exception as e:
        logger.error(f"TMDB ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()
