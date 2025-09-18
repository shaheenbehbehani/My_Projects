#!/usr/bin/env python3
"""
Rotten Tomatoes Data Ingestion Script
Loads and processes Rotten Tomatoes CSV datasets
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
        logging.FileHandler('logs/rottentomatoes_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RottenTomatoesIngestion:
    def __init__(self):
        # Create directories
        Path('data/raw/rottentomatoes').mkdir(parents=True, exist_ok=True)
        Path('data/normalized').mkdir(parents=True, exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        
        # Source file paths
        self.rt_files = {
            'movies': 'Rotten Tomatoes/rotten_tomatoes_movies.csv',
            'top_movies': 'Rotten Tomatoes/rotten_tomatoes_top_movies.csv',
            'reviews': 'Rotten Tomatoes/rotten_tomatoes_movie_reviews.csv'
        }
    
    def copy_raw_files(self):
        """Copy raw files to data/raw/ directory"""
        logger.info("Copying Rotten Tomatoes raw files")
        
        for name, source_path in self.rt_files.items():
            if Path(source_path).exists():
                dest_path = f"data/raw/rottentomatoes/{name}.csv"
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied {source_path} to {dest_path}")
            else:
                logger.warning(f"Source file not found: {source_path}")
    
    def load_movies_data(self):
        """Load and process main movies dataset"""
        logger.info("Loading Rotten Tomatoes movies data")
        
        try:
            movies = pd.read_csv(self.rt_files['movies'])
            logger.info(f"Loaded {len(movies)} movies from main dataset")
            
            # Clean and normalize
            movies_clean = movies.copy()
            
            # Handle missing values
            movies_clean['tomatometer_score'] = pd.to_numeric(
                movies_clean['tomatoMeter'], errors='coerce'
            )
            movies_clean['audience_score'] = pd.to_numeric(
                movies_clean['audienceScore'], errors='coerce'
            )
            movies_clean['runtime'] = pd.to_numeric(
                movies_clean['runtimeMinutes'], errors='coerce'
            )
            
            # Extract year from title if available
            if 'title' in movies_clean.columns:
                movies_clean['year'] = movies_clean['title'].str.extract(r'\((\d{4})\)')
                movies_clean['year'] = pd.to_numeric(movies_clean['year'], errors='coerce')
                
                # Clean title (remove year)
                movies_clean['clean_title'] = movies_clean['title'].str.replace(
                    r'\s*\(\d{4}\)', '', regex=True
                )
            else:
                movies_clean['year'] = np.nan
                movies_clean['clean_title'] = movies_clean.get('title', '')
            
            # Fill missing values and handle empty strings
            movies_clean['tomatometer_score'] = movies_clean['tomatometer_score'].fillna(0)
            movies_clean['audience_score'] = movies_clean['audience_score'].fillna(0)
            movies_clean['runtime'] = movies_clean['runtime'].fillna(0)
            movies_clean['year'] = movies_clean['year'].fillna(0)
            
            # Convert empty strings to 0 for numeric fields
            movies_clean['tomatometer_score'] = movies_clean['tomatometer_score'].replace('', 0)
            movies_clean['audience_score'] = movies_clean['audience_score'].replace('', 0)
            movies_clean['runtime'] = movies_clean['runtime'].replace('', 0)
            
            return movies_clean
            
        except Exception as e:
            logger.error(f"Error loading movies data: {e}")
            return pd.DataFrame()
    
    def load_top_movies_data(self):
        """Load and process top movies dataset"""
        logger.info("Loading Rotten Tomatoes top movies data")
        
        try:
            top_movies = pd.read_csv(self.rt_files['top_movies'])
            logger.info(f"Loaded {len(top_movies)} top movies")
            
            # Clean and normalize
            top_movies_clean = top_movies.copy()
            
            # Handle missing values
            top_movies_clean['tomatometer_score'] = pd.to_numeric(
                top_movies_clean['tomatoMeter'], errors='coerce'
            )
            top_movies_clean['audience_score'] = pd.to_numeric(
                top_movies_clean['audienceScore'], errors='coerce'
            )
            
            # Fill missing values and handle empty strings
            top_movies_clean['tomatometer_score'] = top_movies_clean['tomatometer_score'].fillna(0)
            top_movies_clean['audience_score'] = top_movies_clean['audience_score'].fillna(0)
            
            # Convert empty strings to 0 for numeric fields
            top_movies_clean['tomatometer_score'] = top_movies_clean['tomatometer_score'].replace('', 0)
            top_movies_clean['audience_score'] = top_movies_clean['audience_score'].replace('', 0)
            
            return top_movies_clean
            
        except Exception as e:
            logger.error(f"Error loading top movies data: {e}")
            return pd.DataFrame()
    
    def load_reviews_data(self):
        """Load and process movie reviews dataset"""
        logger.info("Loading Rotten Tomatoes movie reviews data")
        
        try:
            reviews = pd.read_csv(self.rt_files['reviews'])
            logger.info(f"Loaded {len(reviews)} reviews")
            
            # Clean and normalize
            reviews_clean = reviews.copy()
            
            # Handle missing values
            reviews_clean['score'] = pd.to_numeric(
                reviews_clean['originalScore'], errors='coerce'
            )
            reviews_clean['score'] = reviews_clean['score'].fillna(0)
            
            return reviews_clean
            
        except Exception as e:
            logger.error(f"Error loading reviews data: {e}")
            return pd.DataFrame()
    
    def create_normalized_dataset(self, movies_df, top_movies_df, reviews_df):
        """Create a normalized dataset combining all Rotten Tomatoes data"""
        logger.info("Creating normalized Rotten Tomatoes dataset")
        
        # Start with movies data
        if not movies_df.empty:
            # Create normalized DataFrame with only the columns we need
            normalized = pd.DataFrame()
            normalized['rt_id'] = range(len(movies_df))
            normalized['title'] = movies_df.get('clean_title', movies_df.get('title', ''))
            normalized['tomatometer'] = movies_df.get('tomatometer_score', 0)
            normalized['audience_score'] = movies_df.get('audience_score', 0)
            normalized['runtime'] = movies_df.get('runtime', 0)
            normalized['year'] = movies_df.get('year', 0)
            normalized['genres'] = movies_df.get('genres', '')
            normalized['directors'] = movies_df.get('directors', '')
            normalized['cast'] = movies_df.get('cast', '')
            normalized['rt_source'] = 'main_movies'
            
        else:
            # Create empty DataFrame with expected columns
            normalized = pd.DataFrame(columns=[
                'rt_id', 'title', 'tomatometer', 'audience_score', 'runtime',
                'year', 'genres', 'directors', 'cast', 'rt_source'
            ])
        
        # Add top movies if available
        if not top_movies_df.empty:
            top_movies_norm = top_movies_df.copy()
            top_movies_norm['rt_source'] = 'top_movies'
            top_movies_norm['rt_id'] = top_movies_norm.index + len(normalized)
            top_movies_norm['title'] = top_movies_norm.get('title', '')
            top_movies_norm['tomatometer'] = top_movies_norm.get('tomatometer_score', 0)
            top_movies_norm['audience_score'] = top_movies_norm.get('audience_score', 0)
            top_movies_norm['runtime'] = 0  # Not available in top movies
            top_movies_norm['year'] = 0     # Not available in top movies
            top_movies_norm['genres'] = ''
            top_movies_norm['directors'] = ''
            top_movies_norm['cast'] = ''
            
            # Select only normalized columns
            top_movies_final = top_movies_norm[[
                'rt_id', 'title', 'tomatometer', 'audience_score', 'runtime',
                'year', 'genres', 'directors', 'cast', 'rt_source'
            ]]
            
            # Append to normalized dataset
            normalized = pd.concat([normalized, top_movies_final], ignore_index=True)
        
        # Add empty review columns (reviews data doesn't have movie titles to link)
        normalized['avg_review_score'] = 0
        normalized['review_count'] = 0
        normalized['review_types'] = ''
        
        # Final cleaning
        normalized = normalized.fillna('')
        normalized['rt_id'] = normalized['rt_id'].astype(int)
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['tomatometer', 'audience_score', 'runtime', 'year', 'avg_review_score', 'review_count']
        for col in numeric_columns:
            if col in normalized.columns:
                normalized[col] = pd.to_numeric(normalized[col], errors='coerce').fillna(0)
        
        logger.info(f"Normalized Rotten Tomatoes data shape: {normalized.shape}")
        return normalized
    
    def save_normalized_data(self, df: pd.DataFrame):
        """Save normalized data as Parquet and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as Parquet
        parquet_file = f"data/normalized/rottentomatoes_movies_{timestamp}.parquet"
        df.to_parquet(parquet_file, index=False)
        logger.info(f"Saved normalized Parquet data to {parquet_file}")
        
        # Save as CSV
        csv_file = f"data/normalized/rottentomatoes_movies_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved normalized CSV data to {csv_file}")
        
        return parquet_file, csv_file

def main():
    """Main execution function"""
    try:
        logger.info("Starting Rotten Tomatoes data ingestion")
        
        # Initialize ingestion
        rt_ingestion = RottenTomatoesIngestion()
        
        # Copy raw files
        rt_ingestion.copy_raw_files()
        
        # Load all datasets
        movies_data = rt_ingestion.load_movies_data()
        top_movies_data = rt_ingestion.load_top_movies_data()
        reviews_data = rt_ingestion.load_reviews_data()
        
        # Create normalized dataset
        normalized_data = rt_ingestion.create_normalized_dataset(
            movies_data, top_movies_data, reviews_data
        )
        
        if not normalized_data.empty:
            # Save normalized data
            saved_files = rt_ingestion.save_normalized_data(normalized_data)
            logger.info(f"Saved normalized data: {saved_files}")
        else:
            logger.warning("No normalized data to save")
        
        logger.info("Rotten Tomatoes ingestion completed successfully")
        
    except Exception as e:
        logger.error(f"Rotten Tomatoes ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()
