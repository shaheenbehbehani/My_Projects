#!/usr/bin/env python3
"""
IMDb and MovieLens Data Ingestion Script
Loads and processes IMDb TSV and MovieLens CSV datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/imdb_movielens_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IMDbMovieLensIngestion:
    def __init__(self):
        # Create directories
        Path('data/raw/imdb').mkdir(parents=True, exist_ok=True)
        Path('data/raw/movielens').mkdir(parents=True, exist_ok=True)
        Path('data/normalized').mkdir(parents=True, exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        
        # Source file paths
        self.imdb_files = {
            'title_basics': 'IMDB datasets/title.basics.tsv',
            'title_crew': 'IMDB datasets/title.crew.tsv',
            'title_ratings': 'IMDB datasets/title.ratings.tsv'
        }
        
        self.movielens_files = {
            'movies': 'movie-lens/movies.csv',
            'links': 'movie-lens/links.csv',
            'ratings': 'movie-lens/ratings.csv',
            'tags': 'movie-lens/tags.csv'
        }
    
    def copy_raw_files(self):
        """Copy raw files to data/raw/ directories"""
        logger.info("Copying raw files to data/raw/ directories")
        
        # Copy IMDb files
        for name, source_path in self.imdb_files.items():
            if Path(source_path).exists():
                dest_path = f"data/raw/imdb/{name}.tsv"
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied {source_path} to {dest_path}")
            else:
                logger.warning(f"Source file not found: {source_path}")
        
        # Copy MovieLens files
        for name, source_path in self.movielens_files.items():
            if Path(source_path).exists():
                dest_path = f"data/raw/movielens/{name}.csv"
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied {source_path} to {dest_path}")
            else:
                logger.warning(f"Source file not found: {source_path}")
    
    def load_imdb_data(self):
        """Load and process IMDb TSV files"""
        logger.info("Loading IMDb data")
        
        # Load title basics
        title_basics = pd.read_csv(
            self.imdb_files['title_basics'], 
            sep='\t', 
            dtype={
                'tconst': str,
                'titleType': str,
                'primaryTitle': str,
                'originalTitle': str,
                'isAdult': str,  # Changed from bool to str to handle '0'/'1' values
                'startYear': str,
                'endYear': str,
                'runtimeMinutes': str,
                'genres': str
            }
        )
        
        # Filter for movies only
        movies_basics = title_basics[title_basics['titleType'] == 'movie'].copy()
        logger.info(f"Loaded {len(movies_basics)} IMDb movies from basics")
        
        # Load title crew
        title_crew = pd.read_csv(
            self.imdb_files['title_crew'], 
            sep='\t',
            dtype={
                'tconst': str,
                'directors': str,
                'writers': str
            }
        )
        
        # Load title ratings
        title_ratings = pd.read_csv(
            self.imdb_files['title_ratings'], 
            sep='\t',
            dtype={
                'tconst': str,
                'averageRating': float,
                'numVotes': int
            }
        )
        
        # Merge IMDb data
        imdb_movies = movies_basics.merge(
            title_crew, on='tconst', how='left'
        ).merge(
            title_ratings, on='tconst', how='left'
        )
        
        # Clean and normalize IMDb data
        imdb_movies['startYear'] = pd.to_numeric(imdb_movies['startYear'], errors='coerce')
        imdb_movies['runtimeMinutes'] = pd.to_numeric(imdb_movies['runtimeMinutes'], errors='coerce')
        imdb_movies['averageRating'] = imdb_movies['averageRating'].fillna(0)
        imdb_movies['numVotes'] = imdb_movies['numVotes'].fillna(0)
        
        # Convert isAdult to boolean (0 -> False, 1 -> True)
        imdb_movies['isAdult'] = imdb_movies['isAdult'].map({'0': False, '1': True}).fillna(False)
        
        # Create normalized columns
        imdb_movies['imdb_id'] = imdb_movies['tconst'].str.replace('tt', '')
        imdb_movies['year'] = imdb_movies['startYear']
        imdb_movies['title'] = imdb_movies['primaryTitle']
        imdb_movies['genres'] = imdb_movies['genres'].fillna('')
        imdb_movies['directors'] = imdb_movies['directors'].fillna('')
        imdb_movies['writers'] = imdb_movies['writers'].fillna('')
        
        # Select final columns
        imdb_normalized = imdb_movies[[
            'tconst', 'imdb_id', 'title', 'year', 'genres', 'directors', 'writers',
            'averageRating', 'numVotes', 'runtimeMinutes', 'isAdult'
        ]].copy()
        
        logger.info(f"Normalized IMDb data shape: {imdb_normalized.shape}")
        return imdb_normalized
    
    def load_movielens_data(self):
        """Load and process MovieLens CSV files"""
        logger.info("Loading MovieLens data")
        
        # Load movies
        movies = pd.read_csv(
            self.movielens_files['movies'],
            dtype={
                'movieId': int,
                'title': str,
                'genres': str
            }
        )
        
        # Load links
        links = pd.read_csv(
            self.movielens_files['links'],
            dtype={
                'movieId': int,
                'imdbId': int,
                'tmdbId': float
            }
        )
        
        # Load ratings
        ratings = pd.read_csv(
            self.movielens_files['ratings'],
            dtype={
                'userId': int,
                'movieId': int,
                'rating': float,
                'timestamp': int
            }
        )
        
        # Load tags
        tags = pd.read_csv(
            self.movielens_files['tags'],
            dtype={
                'userId': int,
                'movieId': int,
                'tag': str,
                'timestamp': int
            }
        )
        
        logger.info(f"Loaded MovieLens data: {len(movies)} movies, {len(ratings)} ratings, {len(tags)} tags")
        
        # Process movies with links
        movies_with_links = movies.merge(links, on='movieId', how='left')
        
        # Extract year from title (format: "Title (Year)")
        movies_with_links['year'] = movies_with_links['title'].str.extract(r'\((\d{4})\)')
        movies_with_links['year'] = pd.to_numeric(movies_with_links['year'], errors='coerce')
        
        # Clean title (remove year)
        movies_with_links['clean_title'] = movies_with_links['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
        
        # Calculate average ratings
        avg_ratings = ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        avg_ratings.columns = ['movieId', 'avg_rating', 'rating_count']
        
        # Get top tags per movie
        top_tags = tags.groupby('movieId')['tag'].apply(
            lambda x: ', '.join(x.value_counts().head(5).index)
        ).reset_index()
        top_tags.columns = ['movieId', 'top_tags']
        
        # Merge all MovieLens data
        movielens_normalized = movies_with_links.merge(
            avg_ratings, on='movieId', how='left'
        ).merge(
            top_tags, on='movieId', how='left'
        )
        
        # Clean and normalize
        movielens_normalized['avg_rating'] = movielens_normalized['avg_rating'].fillna(0)
        movielens_normalized['rating_count'] = movielens_normalized['rating_count'].fillna(0)
        movielens_normalized['top_tags'] = movielens_normalized['top_tags'].fillna('')
        movielens_normalized['imdbId'] = movielens_normalized['imdbId'].fillna(0).astype(int)
        movielens_normalized['tmdbId'] = movielens_normalized['tmdbId'].fillna(0).astype(int)
        
        # Select final columns
        movielens_final = movielens_normalized[[
            'movieId', 'imdbId', 'tmdbId', 'clean_title', 'year', 'genres',
            'avg_rating', 'rating_count', 'top_tags'
        ]].copy()
        
        movielens_final.columns = [
            'movieId', 'imdbId', 'tmdbId', 'title', 'year', 'genres',
            'avg_rating', 'rating_count', 'top_tags'
        ]
        
        logger.info(f"Normalized MovieLens data shape: {movielens_final.shape}")
        return movielens_final
    
    def save_normalized_data(self, imdb_df: pd.DataFrame, movielens_df: pd.DataFrame):
        """Save normalized data as Parquet and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save IMDb data
        imdb_parquet = f"data/normalized/imdb_movies_{timestamp}.parquet"
        imdb_csv = f"data/normalized/imdb_movies_{timestamp}.csv"
        
        imdb_df.to_parquet(imdb_parquet, index=False)
        imdb_df.to_csv(imdb_csv, index=False)
        logger.info(f"Saved IMDb data: {imdb_parquet}, {imdb_csv}")
        
        # Save MovieLens data
        movielens_parquet = f"data/normalized/movielens_movies_{timestamp}.parquet"
        movielens_csv = f"data/normalized/movielens_movies_{timestamp}.csv"
        
        movielens_df.to_parquet(movielens_parquet, index=False)
        movielens_df.to_csv(movielens_csv, index=False)
        logger.info(f"Saved MovieLens data: {movielens_parquet}, {movielens_csv}")
        
        return {
            'imdb': (imdb_parquet, imdb_csv),
            'movielens': (movielens_parquet, movielens_csv)
        }

def main():
    """Main execution function"""
    try:
        logger.info("Starting IMDb and MovieLens data ingestion")
        
        # Initialize ingestion
        ingestion = IMDbMovieLensIngestion()
        
        # Copy raw files
        ingestion.copy_raw_files()
        
        # Load and process IMDb data
        imdb_data = ingestion.load_imdb_data()
        
        # Load and process MovieLens data
        movielens_data = ingestion.load_movielens_data()
        
        # Save normalized data
        saved_files = ingestion.save_normalized_data(imdb_data, movielens_data)
        
        logger.info("IMDb and MovieLens ingestion completed successfully")
        logger.info(f"Saved files: {saved_files}")
        
    except Exception as e:
        logger.error(f"IMDb and MovieLens ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()
