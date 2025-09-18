#!/usr/bin/env python3
"""
ID Bridge Creation Script
Builds a bridge table connecting movie IDs across TMDB, IMDb, MovieLens, and Rotten Tomatoes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import glob
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/id_bridge_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IDBridgeCreator:
    def __init__(self):
        # Create directories
        Path('data/normalized').mkdir(parents=True, exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        
        # Column mappings for each source
        self.source_columns = {
            'tmdb': ['tmdb_id', 'title', 'release_date'],
            'imdb': ['tconst', 'imdb_id', 'title', 'year'],
            'movielens': ['movieId', 'imdbId', 'tmdbId', 'title', 'year'],
            'rottentomatoes': ['rt_id', 'title', 'year']
        }
    
    def load_latest_normalized_data(self):
        """Load the most recent normalized data from each source"""
        logger.info("Loading latest normalized data from all sources")
        
        data_sources = {}
        
        # Load TMDB data
        tmdb_files = glob.glob('data/normalized/tmdb_movies_*.parquet')
        if tmdb_files:
            latest_tmdb = max(tmdb_files)
            tmdb_data = pd.read_parquet(latest_tmdb)
            data_sources['tmdb'] = tmdb_data
            logger.info(f"Loaded TMDB data: {latest_tmdb} ({len(tmdb_data)} records)")
        else:
            logger.warning("No TMDB normalized data found")
            data_sources['tmdb'] = pd.DataFrame()
        
        # Load IMDb data
        imdb_files = glob.glob('data/normalized/imdb_movies_*.parquet')
        if imdb_files:
            latest_imdb = max(imdb_files)
            imdb_data = pd.read_parquet(latest_imdb)
            data_sources['imdb'] = imdb_data
            logger.info(f"Loaded IMDb data: {latest_imdb} ({len(imdb_data)} records)")
        else:
            logger.warning("No IMDb normalized data found")
            data_sources['imdb'] = pd.DataFrame()
        
        # Load MovieLens data
        movielens_files = glob.glob('data/normalized/movielens_movies_*.parquet')
        if movielens_files:
            latest_movielens = max(movielens_files)
            movielens_data = pd.read_parquet(latest_movielens)
            data_sources['movielens'] = movielens_data
            logger.info(f"Loaded MovieLens data: {latest_movielens} ({len(movielens_data)} records)")
        else:
            logger.warning("No MovieLens normalized data found")
            data_sources['movielens'] = pd.DataFrame()
        
        # Load Rotten Tomatoes data
        rt_files = glob.glob('data/normalized/rottentomatoes_movies_*.parquet')
        if rt_files:
            latest_rt = max(rt_files)
            rt_data = pd.read_parquet(latest_rt)
            data_sources['rottentomatoes'] = rt_data
            logger.info(f"Loaded Rotten Tomatoes data: {latest_rt} ({len(rt_data)} records)")
        else:
            logger.warning("No Rotten Tomatoes normalized data found")
            data_sources['rottentomatoes'] = pd.DataFrame()
        
        return data_sources
    
    def clean_title_for_matching(self, title: str) -> str:
        """Clean title for better matching across sources"""
        if pd.isna(title) or title == '':
            return ''
        
        # Convert to string and lowercase
        title_str = str(title).lower().strip()
        
        # Remove year patterns like (1999) or [1999]
        title_str = re.sub(r'\s*[\(\[\]\d{4}\)]', '', title_str)
        
        # Remove special characters and extra spaces
        title_str = re.sub(r'[^\w\s]', ' ', title_str)
        title_str = re.sub(r'\s+', ' ', title_str).strip()
        
        return title_str
    
    def extract_year_from_date(self, date_str):
        """Extract year from date string"""
        if pd.isna(date_str) or date_str == '':
            return None
        
        try:
            # Try to parse date and extract year
            if isinstance(date_str, str):
                # Handle various date formats
                if '-' in date_str:
                    return int(date_str.split('-')[0])
                elif '/' in date_str:
                    return int(date_str.split('/')[-1])
                else:
                    # Try to extract 4-digit year
                    year_match = re.search(r'\d{4}', date_str)
                    if year_match:
                        return int(year_match.group())
            return None
        except:
            return None
    
    def create_title_year_matches(self, data_sources):
        """Create matches based on title and year"""
        logger.info("Creating title-year matches across sources")
        
        # Prepare data for matching
        matches = []
        
        # Process each source
        for source_name, source_data in data_sources.items():
            if source_data.empty:
                continue
            
            logger.info(f"Processing {source_name} for title-year matching")
            
            # Extract title and year information
            for idx, row in source_data.iterrows():
                title = row.get('title', '')
                year = None
                
                # Get year from various possible columns
                if 'year' in source_data.columns:
                    year = row.get('year')
                elif 'release_date' in source_data.columns:
                    year = self.extract_year_from_date(row.get('release_date'))
                elif 'startYear' in source_data.columns:
                    year = row.get('startYear')
                
                # Clean title
                clean_title = self.clean_title_for_matching(title)
                
                if clean_title and clean_title != '':
                    # Get source-specific IDs
                    source_ids = {}
                    
                    if source_name == 'tmdb':
                        source_ids['tmdb_id'] = row.get('tmdb_id')
                    elif source_name == 'imdb':
                        source_ids['tconst'] = row.get('tconst')
                        source_ids['imdb_id'] = row.get('imdb_id')
                    elif source_name == 'movielens':
                        source_ids['movieId'] = row.get('movieId')
                        source_ids['imdbId'] = row.get('imdbId')
                        source_ids['tmdbId'] = row.get('tmdbId')
                    elif source_name == 'rottentomatoes':
                        source_ids['rt_id'] = row.get('rt_id')
                    
                    matches.append({
                        'source': source_name,
                        'title': title,
                        'clean_title': clean_title,
                        'year': year,
                        **source_ids
                    })
        
        return pd.DataFrame(matches)
    
    def build_id_bridge(self, matches_df):
        """Build the final ID bridge table"""
        logger.info("Building ID bridge table")
        
        # Group by clean_title and year to find matches
        bridge_records = []
        
        # Group matches by clean title and year
        grouped = matches_df.groupby(['clean_title', 'year'])
        
        for (clean_title, year), group in grouped:
            if len(group) > 1:  # Multiple sources have this movie
                # Initialize record
                record = {
                    'title': clean_title,
                    'year': year,
                    'movieId': None,
                    'imdbId': None,
                    'tconst': None,
                    'tmdbId': None,
                    'rt_id': None
                }
                
                # Collect IDs from all sources
                for _, row in group.iterrows():
                    source = row['source']
                    
                    if source == 'movielens':
                        if pd.notna(row.get('movieId')):
                            record['movieId'] = int(row['movieId'])
                        if pd.notna(row.get('imdbId')) and row.get('imdbId') != 0:
                            record['imdbId'] = int(row['imdbId'])
                        if pd.notna(row.get('tmdbId')) and row.get('tmdbId') != 0:
                            record['tmdbId'] = int(row['tmdbId'])
                    
                    elif source == 'imdb':
                        if pd.notna(row.get('tconst')):
                            record['tconst'] = str(row['tconst'])
                        if pd.notna(row.get('imdb_id')):
                            record['imdbId'] = int(row['imdb_id'])
                    
                    elif source == 'tmdb':
                        if pd.notna(row.get('tmdb_id')):
                            record['tmdbId'] = int(row['tmdb_id'])
                    
                    elif source == 'rottentomatoes':
                        if pd.notna(row.get('rt_id')):
                            record['rt_id'] = int(row['rt_id'])
                
                # Only add if we have at least 2 different IDs
                id_count = sum(1 for v in record.values() if v is not None and v != '')
                if id_count >= 2:
                    bridge_records.append(record)
        
        # Create bridge DataFrame
        bridge_df = pd.DataFrame(bridge_records)
        
        # Add tconst if we have imdbId but no tconst
        if 'imdbId' in bridge_df.columns and 'tconst' in bridge_df.columns:
            mask = (bridge_df['imdbId'].notna()) & (bridge_df['tconst'].isna())
            bridge_df.loc[mask, 'tconst'] = bridge_df.loc[mask, 'imdbId'].apply(
                lambda x: f"tt{str(x).zfill(7)}" if pd.notna(x) else None
            )
        
        # Sort by title and year
        bridge_df = bridge_df.sort_values(['title', 'year']).reset_index(drop=True)
        
        logger.info(f"Created ID bridge with {len(bridge_df)} records")
        return bridge_df
    
    def save_id_bridge(self, bridge_df: pd.DataFrame):
        """Save the ID bridge table"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as Parquet
        parquet_file = f"data/normalized/id_bridge_{timestamp}.parquet"
        bridge_df.to_parquet(parquet_file, index=False)
        logger.info(f"Saved ID bridge to {parquet_file}")
        
        # Save as CSV
        csv_file = f"data/normalized/id_bridge_{timestamp}.csv"
        bridge_df.to_csv(csv_file, index=False)
        logger.info(f"Saved ID bridge to {csv_file}")
        
        # Also save as the standard filename
        standard_parquet = "data/normalized/id_bridge.parquet"
        bridge_df.to_parquet(standard_parquet, index=False)
        logger.info(f"Saved standard ID bridge to {standard_parquet}")
        
        return parquet_file, csv_file, standard_parquet

def main():
    """Main execution function"""
    try:
        logger.info("Starting ID bridge creation")
        
        # Initialize bridge creator
        bridge_creator = IDBridgeCreator()
        
        # Load data from all sources
        data_sources = bridge_creator.load_latest_normalized_data()
        
        # Create title-year matches
        matches_df = bridge_creator.create_title_year_matches(data_sources)
        logger.info(f"Created {len(matches_df)} title-year matches")
        
        # Build ID bridge
        bridge_df = bridge_creator.build_id_bridge(matches_df)
        
        if not bridge_df.empty:
            # Save ID bridge
            saved_files = bridge_creator.save_id_bridge(bridge_df)
            logger.info(f"Saved ID bridge files: {saved_files}")
            
            # Log summary statistics
            logger.info("ID Bridge Summary:")
            logger.info(f"Total records: {len(bridge_df)}")
            logger.info(f"Records with MovieLens ID: {bridge_df['movieId'].notna().sum()}")
            logger.info(f"Records with IMDb ID: {bridge_df['imdbId'].notna().sum()}")
            logger.info(f"Records with TMDB ID: {bridge_df['tmdbId'].notna().sum()}")
            logger.info(f"Records with RT ID: {bridge_df['rt_id'].notna().sum()}")
        else:
            logger.warning("No ID bridge records created")
        
        logger.info("ID bridge creation completed successfully")
        
    except Exception as e:
        logger.error(f"ID bridge creation failed: {e}")
        raise

if __name__ == "__main__":
    main()






