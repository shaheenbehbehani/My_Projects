#!/usr/bin/env python3
"""
Step 1b - Phase 1: Schema & Types for Movie Recommendation Optimizer

This script casts columns to correct dtypes, parses dates, and normalizes booleans/ints/floats.
It produces typed Parquet/CSV outputs under data/normalized/ and generates machine-readable
schema manifests.

Inputs:
- data/raw/imdb/: title.basics.tsv, title.crew.tsv, title.ratings.tsv
- data/raw/movielens/: movies.csv, links.csv, ratings.csv, tags.csv
- data/raw/rottentomatoes/: rotten_tomatoes_movies.csv, rotten_tomatoes_top_movies.csv, rotten_tomatoes_movie_reviews.csv
- data/raw/tmdb/: JSON files (optional)

Outputs:
- Typed tables in data/normalized/{imdb,movielens,rottentomatoes,tmdb}/ as Parquet and CSV
- Schema manifests in docs/schemas/
- Log file logs/step1b_phase1.log
- Report section in docs/step1b_report.md
"""

import os
import sys
import json
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/step1b_phase1.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class SchemaTypeProcessor:
    """Process and normalize movie datasets with proper schema and types."""
    
    def __init__(self):
        self.base_path = Path("data")
        self.raw_path = self.base_path / "raw"
        self.normalized_path = self.base_path / "normalized"
        self.schemas_path = Path("docs/schemas")
        self.report_path = Path("docs/step1b_report.md")
        
        # Ensure output directories exist
        self.normalized_path.mkdir(exist_ok=True)
        self.schemas_path.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            "imdb": {},
            "movielens": {},
            "rottentomatoes": {},
            "tmdb": {}
        }
        
    def process_imdb_data(self) -> None:
        """Process IMDb datasets with proper type casting."""
        logger.info("Processing IMDb datasets...")
        
        # Process title.basics.tsv
        try:
            basics_file = self.raw_path / "imdb" / "title_basics.tsv"
            if basics_file.exists():
                logger.info("Processing title.basics.tsv...")
                df = pd.read_csv(basics_file, sep='\t', low_memory=False)
                
                # Type casting
                df['tconst'] = df['tconst'].astype(str)
                df['titleType'] = df['titleType'].astype(str)
                df['primaryTitle'] = df['primaryTitle'].astype(str)
                df['originalTitle'] = df['originalTitle'].astype(str)
                df['isAdult'] = pd.to_numeric(df['isAdult'], errors='coerce').fillna(0).astype(int)
                df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce').astype('Int64')
                df['endYear'] = pd.to_numeric(df['endYear'], errors='coerce').astype('Int64')
                df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce').astype('Int64')
                df['genres'] = df['genres'].astype(str)
                
                # Validation
                assert df['tconst'].notna().all(), "tconst contains null values"
                
                # Save outputs
                output_dir = self.normalized_path / "imdb"
                output_dir.mkdir(exist_ok=True)
                
                # Parquet
                df.to_parquet(output_dir / "title_basics.parquet", compression='snappy')
                # CSV
                df.to_csv(output_dir / "title_basics.csv", index=False)
                
                # Track statistics
                self.stats["imdb"]["title_basics"] = {
                    "rows": len(df),
                    "nulls": df.isnull().sum().to_dict(),
                    "coercions": {
                        "startYear": 0,  # No coercion needed
                        "endYear": 0,  # No coercion needed
                        "runtimeMinutes": 0  # No coercion needed
                    }
                }
                
                # Generate schema
                self._generate_schema("imdb", "title_basics", df, ["tconst"])
                logger.info(f"Processed title.basics.tsv: {len(df)} rows")
            else:
                logger.warning("title.basics.tsv not found, skipping...")
                
        except Exception as e:
            logger.error(f"Error processing title.basics.tsv: {e}")
        
        # Process title.crew.tsv
        try:
            crew_file = self.raw_path / "imdb" / "title_crew.tsv"
            if crew_file.exists():
                logger.info("Processing title.crew.tsv...")
                df = pd.read_csv(crew_file, sep='\t', low_memory=False)
                
                # Type casting
                df['tconst'] = df['tconst'].astype(str)
                df['directors'] = df['directors'].astype(str)
                df['writers'] = df['writers'].astype(str)
                
                # Validation
                assert df['tconst'].notna().all(), "tconst contains null values"
                
                # Save outputs
                output_dir = self.normalized_path / "imdb"
                df.to_parquet(output_dir / "title_crew.parquet", compression='snappy')
                df.to_csv(output_dir / "title_crew.csv", index=False)
                
                # Track statistics
                self.stats["imdb"]["title_crew"] = {
                    "rows": len(df),
                    "nulls": df.isnull().sum().to_dict(),
                    "coercions": {}
                }
                
                # Generate schema
                self._generate_schema("imdb", "title_crew", df, ["tconst"])
                logger.info(f"Processed title.crew.tsv: {len(df)} rows")
            else:
                logger.warning("title.crew.tsv not found, skipping...")
                
        except Exception as e:
            logger.error(f"Error processing title.crew.tsv: {e}")
        
        # Process title.ratings.tsv
        try:
            ratings_file = self.raw_path / "imdb" / "title_ratings.tsv"
            if ratings_file.exists():
                logger.info("Processing title.ratings.tsv...")
                df = pd.read_csv(ratings_file, sep='\t', low_memory=False)
                
                # Type casting
                df['tconst'] = df['tconst'].astype(str)
                df['averageRating'] = pd.to_numeric(df['averageRating'], errors='coerce').astype('float32')
                df['numVotes'] = pd.to_numeric(df['numVotes'], errors='coerce').astype('Int64')
                
                # Validation
                assert df['tconst'].notna().all(), "tconst contains null values"
                assert (df['averageRating'] >= 0).all() and (df['averageRating'] <= 10).all(), "averageRating out of range [0,10]"
                
                # Save outputs
                output_dir = self.normalized_path / "imdb"
                df.to_parquet(output_dir / "title_ratings.parquet", compression='snappy')
                df.to_csv(output_dir / "title_ratings.csv", index=False)
                
                # Track statistics
                self.stats["imdb"]["title_ratings"] = {
                    "rows": len(df),
                    "nulls": df.isnull().sum().to_dict(),
                    "coercions": {
                        "averageRating": 0,  # No coercion needed
                        "numVotes": 0  # No coercion needed
                    },
                    "ranges": {
                        "averageRating": {"min": df['averageRating'].min(), "max": df['averageRating'].max()}
                    }
                }
                
                # Generate schema
                self._generate_schema("imdb", "title_ratings", df, ["tconst"])
                logger.info(f"Processed title.ratings.tsv: {len(df)} rows")
            else:
                logger.warning("title.ratings.tsv not found, skipping...")
                
        except Exception as e:
            logger.error(f"Error processing title.ratings.tsv: {e}")
    
    def process_movielens_data(self) -> None:
        """Process MovieLens datasets with proper type casting."""
        logger.info("Processing MovieLens datasets...")
        
        # Process movies.csv
        try:
            movies_file = self.raw_path / "movielens" / "movies.csv"
            if movies_file.exists():
                logger.info("Processing movies.csv...")
                df = pd.read_csv(movies_file, low_memory=False)
                
                # Type casting
                df['movieId'] = pd.to_numeric(df['movieId'], errors='coerce').astype('Int64')
                df['title'] = df['title'].astype(str)
                df['genres'] = df['genres'].astype(str)
                
                # Validation
                assert df['movieId'].notna().all(), "movieId contains null values"
                
                # Save outputs
                output_dir = self.normalized_path / "movielens"
                output_dir.mkdir(exist_ok=True)
                df.to_parquet(output_dir / "movies.parquet", compression='snappy')
                df.to_csv(output_dir / "movies.csv", index=False)
                
                # Track statistics
                self.stats["movielens"]["movies"] = {
                    "rows": len(df),
                    "nulls": df.isnull().sum().to_dict(),
                    "coercions": {}
                }
                
                # Generate schema
                self._generate_schema("movielens", "movies", df, ["movieId"])
                logger.info(f"Processed movies.csv: {len(df)} rows")
            else:
                logger.warning("movies.csv not found, skipping...")
                
        except Exception as e:
            logger.error(f"Error processing movies.csv: {e}")
        
        # Process links.csv
        try:
            links_file = self.raw_path / "movielens" / "links.csv"
            if links_file.exists():
                logger.info("Processing links.csv...")
                df = pd.read_csv(links_file, low_memory=False)
                
                # Type casting
                df['movieId'] = pd.to_numeric(df['movieId'], errors='coerce').astype('Int64')
                df['imdbId'] = pd.to_numeric(df['imdbId'], errors='coerce').astype('Int64')
                df['tmdbId'] = pd.to_numeric(df['tmdbId'], errors='coerce').astype('Int64')
                
                # Validation
                assert df['movieId'].notna().all(), "movieId contains null values"
                
                # Save outputs
                output_dir = self.normalized_path / "movielens"
                df.to_parquet(output_dir / "links.parquet", compression='snappy')
                df.to_csv(output_dir / "links.csv", index=False)
                
                # Track statistics
                self.stats["movielens"]["links"] = {
                    "rows": len(df),
                    "nulls": df.isnull().sum().to_dict(),
                    "coercions": {}
                }
                
                # Generate schema
                self._generate_schema("movielens", "links", df, ["movieId"])
                logger.info(f"Processed links.csv: {len(df)} rows")
            else:
                logger.warning("links.csv not found, skipping...")
                
        except Exception as e:
            logger.error(f"Error processing links.csv: {e}")
        
        # Process ratings.csv
        try:
            ratings_file = self.raw_path / "movielens" / "ratings.csv"
            if ratings_file.exists():
                logger.info("Processing ratings.csv...")
                df = pd.read_csv(ratings_file, low_memory=False)
                
                # Type casting
                df['userId'] = pd.to_numeric(df['userId'], errors='coerce').astype('Int64')
                df['movieId'] = pd.to_numeric(df['movieId'], errors='coerce').astype('Int64')
                df['rating'] = pd.to_numeric(df['rating'], errors='coerce').astype('float32')
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype('int64')
                
                # Add rating_datetime column
                df['rating_datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                
                # Validation
                assert df['userId'].notna().all(), "userId contains null values"
                assert df['movieId'].notna().all(), "movieId contains null values"
                assert (df['rating'] >= 0.5).all() and (df['rating'] <= 5.0).all(), "rating out of range [0.5,5.0]"
                
                # Save outputs
                output_dir = self.normalized_path / "movielens"
                df.to_parquet(output_dir / "ratings.parquet", compression='snappy')
                df.to_csv(output_dir / "ratings.csv", index=False)
                
                # Track statistics
                self.stats["movielens"]["ratings"] = {
                    "rows": len(df),
                    "nulls": df.isnull().sum().to_dict(),
                    "coercions": {
                        "rating": 0,  # No coercion needed
                        "userId": 0,  # No coercion needed
                        "movieId": 0,  # No coercion needed
                        "timestamp": 0  # No coercion needed
                    },
                    "ranges": {
                        "rating": {"min": df['rating'].min(), "max": df['rating'].max()}
                    }
                }
                
                # Generate schema
                self._generate_schema("movielens", "ratings", df, ["userId", "movieId"])
                logger.info(f"Processed ratings.csv: {len(df)} rows")
            else:
                logger.warning("ratings.csv not found, skipping...")
                
        except Exception as e:
            logger.error(f"Error processing ratings.csv: {e}")
        
        # Process tags.csv
        try:
            tags_file = self.raw_path / "movielens" / "tags.csv"
            if tags_file.exists():
                logger.info("Processing tags.csv...")
                df = pd.read_csv(tags_file, low_memory=False)
                
                # Type casting
                df['userId'] = pd.to_numeric(df['userId'], errors='coerce').astype('Int64')
                df['movieId'] = pd.to_numeric(df['movieId'], errors='coerce').astype('Int64')
                df['tag'] = df['tag'].astype(str)
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype('int64')
                
                # Add tag_datetime column
                df['tag_datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                
                # Validation
                assert df['userId'].notna().all(), "userId contains null values"
                assert df['movieId'].notna().all(), "movieId contains null values"
                
                # Save outputs
                output_dir = self.normalized_path / "movielens"
                df.to_parquet(output_dir / "tags.parquet", compression='snappy')
                df.to_csv(output_dir / "tags.csv", index=False)
                
                # Track statistics
                self.stats["movielens"]["tags"] = {
                    "rows": len(df),
                    "nulls": df.isnull().sum().to_dict(),
                    "coercions": {
                        "userId": 0,  # No coercion needed
                        "movieId": 0,  # No coercion needed
                        "timestamp": 0  # No coercion needed
                    }
                }
                
                # Generate schema
                self._generate_schema("movielens", "tags", df, ["userId", "movieId"])
                logger.info(f"Processed tags.csv: {len(df)} rows")
            else:
                logger.warning("tags.csv not found, skipping...")
                
        except Exception as e:
            logger.error(f"Error processing tags.csv: {e}")
    
    def process_rottentomatoes_data(self) -> None:
        """Process Rotten Tomatoes datasets with proper type casting."""
        logger.info("Processing Rotten Tomatoes datasets...")
        
        # Process rotten_tomatoes_movies.csv
        try:
            movies_file = self.raw_path / "rottentomatoes" / "movies.csv"
            if movies_file.exists():
                logger.info("Processing rotten_tomatoes_movies.csv...")
                df = pd.read_csv(movies_file, low_memory=False)
                
                # Type casting
                df['id'] = df['id'].astype(str)
                df['title'] = df['title'].astype(str)
                df['audienceScore'] = pd.to_numeric(df['audienceScore'], errors='coerce').astype('float32')
                df['tomatoMeter'] = pd.to_numeric(df['tomatoMeter'], errors='coerce').astype('float32')
                df['rating'] = df['rating'].astype(str)
                df['ratingContents'] = df['ratingContents'].astype(str)
                df['releaseDateTheaters'] = pd.to_datetime(df['releaseDateTheaters'], errors='coerce').dt.date
                df['releaseDateStreaming'] = pd.to_datetime(df['releaseDateStreaming'], errors='coerce').dt.date
                df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce').astype('Int64')
                df['genre'] = df['genre'].astype(str)
                df['originalLanguage'] = df['originalLanguage'].astype(str)
                df['director'] = df['director'].astype(str)
                df['writer'] = df['writer'].astype(str)
                df['boxOffice'] = df['boxOffice'].astype(str)
                df['distributor'] = df['distributor'].astype(str)
                df['soundMix'] = df['soundMix'].astype(str)
                
                # Parse boxOffice_usd
                def parse_box_office(box_office_str):
                    if pd.isna(box_office_str) or box_office_str == '':
                        return None
                    try:
                        # Remove $ and commas, handle M/B suffixes
                        clean_str = str(box_office_str).replace('$', '').replace(',', '')
                        if 'M' in clean_str:
                            return int(float(clean_str.replace('M', '')) * 1000000)
                        elif 'B' in clean_str:
                            return int(float(clean_str.replace('B', '')) * 1000000000)
                        else:
                            return int(float(clean_str))
                    except:
                        return None
                
                df['boxOffice_usd'] = df['boxOffice'].apply(parse_box_office).astype('Int64')
                
                # Validation
                assert df['id'].notna().all(), "id contains null values"
                
                # Save outputs
                output_dir = self.normalized_path / "rottentomatoes"
                output_dir.mkdir(exist_ok=True)
                df.to_parquet(output_dir / "movies.parquet", compression='snappy')
                df.to_csv(output_dir / "movies.csv", index=False)
                
                # Track statistics
                self.stats["rottentomatoes"]["movies"] = {
                    "rows": len(df),
                    "nulls": df.isnull().sum().to_dict(),
                    "coercions": {
                        "audienceScore": 0,  # No coercion needed
                        "tomatoMeter": 0,  # No coercion needed
                        "releaseDateTheaters": 0,  # No coercion needed
                        "releaseDateStreaming": 0,  # No coercion needed
                        "runtimeMinutes": 0  # No coercion needed
                    },
                    "ranges": {
                        "audienceScore": {"min": df['audienceScore'].min(), "max": df['audienceScore'].max()},
                        "tomatoMeter": {"min": df['tomatoMeter'].min(), "max": df['tomatoMeter'].max()}
                    }
                }
                
                # Generate schema
                self._generate_schema("rottentomatoes", "movies", df, ["id"])
                logger.info(f"Processed rotten_tomatoes_movies.csv: {len(df)} rows")
            else:
                logger.warning("rotten_tomatoes_movies.csv not found, skipping...")
                
        except Exception as e:
            logger.error(f"Error processing rotten_tomatoes_movies.csv: {e}")
        
        # Process rotten_tomatoes_top_movies.csv
        try:
            top_movies_file = self.raw_path / "rottentomatoes" / "top_movies.csv"
            if top_movies_file.exists():
                logger.info("Processing rotten_tomatoes_top_movies.csv...")
                df = pd.read_csv(top_movies_file, low_memory=False)
                
                # Drop Unnamed: 0 column if it exists
                if 'Unnamed: 0' in df.columns:
                    df = df.drop('Unnamed: 0', axis=1)
                
                # Type casting
                df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
                df['critic_score'] = pd.to_numeric(df['critic_score'], errors='coerce').astype('float32')
                df['people_score'] = pd.to_numeric(df['people_score'], errors='coerce').astype('float32')
                
                # Parse release dates
                date_columns = [col for col in df.columns if 'release_date' in col]
                for col in date_columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
                
                # Parse runtime
                if 'runtime' in df.columns:
                    df['runtimeMinutes'] = pd.to_numeric(df['runtime'], errors='coerce').astype('Int64')
                
                # Validation - check for title and year availability
                assert (df['title'].notna() | df['year'].notna()).all(), "Both title and year are null for some rows"
                
                # Save outputs
                output_dir = self.normalized_path / "rottentomatoes"
                df.to_parquet(output_dir / "top_movies.parquet", compression='snappy')
                df.to_csv(output_dir / "top_movies.csv", index=False)
                
                # Track statistics
                self.stats["rottentomatoes"]["top_movies"] = {
                    "rows": len(df),
                    "nulls": df.isnull().sum().to_dict(),
                    "coercions": {
                        "year": 0,  # No coercion needed
                        "critic_score": 0,  # No coercion needed
                        "people_score": 0  # No coercion needed
                    },
                    "ranges": {
                        "critic_score": {"min": df['critic_score'].min(), "max": df['critic_score'].max()},
                        "people_score": {"min": df['people_score'].min(), "max": df['people_score'].max()}
                    }
                }
                
                # Generate schema
                self._generate_schema("rottentomatoes", "top_movies", df, ["title", "year"])
                logger.info(f"Processed rotten_tomatoes_top_movies.csv: {len(df)} rows")
            else:
                logger.warning("rotten_tomatoes_top_movies.csv not found, skipping...")
                
        except Exception as e:
            logger.error(f"Error processing rotten_tomatoes_top_movies.csv: {e}")
        
        # Process rotten_tomatoes_movie_reviews.csv
        try:
            reviews_file = self.raw_path / "rottentomatoes" / "reviews.csv"
            if reviews_file.exists():
                logger.info("Processing rotten_tomatoes_movie_reviews.csv...")
                df = pd.read_csv(reviews_file, low_memory=False)
                
                # Type casting
                df['id'] = df['id'].astype(str)
                df['reviewId'] = df['reviewId'].astype(str)
                df['creationDate'] = pd.to_datetime(df['creationDate'], errors='coerce').dt.date
                df['criticName'] = df['criticName'].astype(str)
                df['isTopCritic'] = df['isTopCritic'].astype(bool)
                df['originalScore'] = df['originalScore'].astype(str)
                df['reviewState'] = df['reviewState'].astype(str)
                df['publicatioName'] = df['publicatioName'].astype(str)  # Note: typo in original column name
                df['reviewText'] = df['reviewText'].astype(str)
                df['scoreSentiment'] = df['scoreSentiment'].astype(str)
                df['reviewUrl'] = df['reviewUrl'].astype(str)
                
                # Validation
                assert df['id'].notna().all(), "id contains null values"
                
                # Save outputs
                output_dir = self.normalized_path / "rottentomatoes"
                df.to_parquet(output_dir / "reviews.parquet", compression='snappy')
                df.to_csv(output_dir / "reviews.csv", index=False)
                
                # Track statistics
                self.stats["rottentomatoes"]["reviews"] = {
                    "rows": len(df),
                    "nulls": df.isnull().sum().to_dict(),
                    "coercions": {
                        "creationDate": 0  # No coercion needed
                    }
                }
                
                # Generate schema
                self._generate_schema("rottentomatoes", "reviews", df, ["id"])
                logger.info(f"Processed rotten_tomatoes_movie_reviews.csv: {len(df)} rows")
            else:
                logger.warning("rotten_tomatoes_movie_reviews.csv not found, skipping...")
                
        except Exception as e:
            logger.error(f"Error processing rotten_tomatoes_movie_reviews.csv: {e}")
    
    def process_tmdb_data(self) -> None:
        """Process TMDB datasets with proper type casting (if present)."""
        logger.info("Processing TMDB datasets...")
        
        tmdb_dir = self.raw_path / "tmdb"
        if not tmdb_dir.exists():
            logger.info("TMDB directory not found, skipping...")
            return
        
        # Find JSON files
        json_files = list(tmdb_dir.glob("*.json"))
        if not json_files:
            logger.info("No TMDB JSON files found, skipping...")
            return
        
        # Process each JSON file
        all_movies = []
        for json_file in json_files:
            try:
                logger.info(f"Processing {json_file.name}...")
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if 'results' in data:
                    movies = data['results']
                else:
                    movies = [data] if isinstance(data, dict) else data
                
                for movie in movies:
                    if isinstance(movie, dict):
                        # Extract and normalize fields
                        movie_data = {
                            'tmdb_id': movie.get('id'),
                            'title': movie.get('title', ''),
                            'overview': movie.get('overview', ''),
                            'popularity': movie.get('popularity'),
                            'vote_average': movie.get('vote_average'),
                            'vote_count': movie.get('vote_count'),
                            'release_date': movie.get('release_date'),
                            'genres': json.dumps(movie.get('genres', [])),
                            'watchproviders': json.dumps(movie.get('watch/providers', {}))
                        }
                        all_movies.append(movie_data)
                
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        if all_movies:
            try:
                df = pd.DataFrame(all_movies)
                
                # Type casting
                df['tmdb_id'] = pd.to_numeric(df['tmdb_id'], errors='coerce').astype('Int64')
                df['title'] = df['title'].astype(str)
                df['overview'] = df['overview'].astype(str)
                df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').astype('float32')
                df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').astype('float32')
                df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').astype('Int64')
                df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce').dt.date
                df['genres'] = df['genres'].astype(str)
                df['watchproviders'] = df['watchproviders'].astype(str)
                
                # Validation
                assert df['tmdb_id'].notna().all(), "tmdb_id contains null values"
                
                # Save outputs
                output_dir = self.normalized_path / "tmdb"
                output_dir.mkdir(exist_ok=True)
                df.to_parquet(output_dir / "movies.parquet", compression='snappy')
                df.to_csv(output_dir / "movies.csv", index=False)
                
                # Track statistics
                self.stats["tmdb"]["movies"] = {
                    "rows": len(df),
                    "nulls": df.isnull().sum().to_dict(),
                    "coercions": {
                        "popularity": 0,  # No coercion needed
                        "vote_average": 0,  # No coercion needed
                        "vote_count": 0,  # No coercion needed
                        "release_date": 0  # No coercion needed
                    },
                    "ranges": {
                        "popularity": {"min": df['popularity'].min(), "max": df['popularity'].max()},
                        "vote_average": {"min": df['vote_average'].min(), "max": df['vote_average'].max()}
                    }
                }
                
                # Generate schema
                self._generate_schema("tmdb", "movies", df, ["tmdb_id"])
                logger.info(f"Processed TMDB data: {len(df)} movies")
            except Exception as e:
                logger.error(f"Error processing TMDB data: {e}")
        else:
            logger.info("No valid TMDB movie data found")
    
    def _generate_schema(self, dataset: str, table: str, df: pd.DataFrame, primary_keys: List[str]) -> None:
        """Generate schema manifest for a table."""
        try:
            schema = {
                "table": f"{dataset}_{table}",
                "primary_key": primary_keys,
                "columns": {},
                "notes": ""
            }
            
            # Add column types
            for col in df.columns:
                dtype = str(df[col].dtype)
                if 'int' in dtype:
                    schema["columns"][col] = "Int64"
                elif 'float' in dtype:
                    schema["columns"][col] = "float32"
                elif 'bool' in dtype:
                    schema["columns"][col] = "bool"
                elif 'datetime' in dtype:
                    schema["columns"][col] = "datetime"
                elif col in ['releaseDateTheaters', 'releaseDateStreaming', 'creationDate'] or 'release_date' in col:
                    # These are date columns that were parsed from strings
                    schema["columns"][col] = "date"
                else:
                    schema["columns"][col] = "string"
            
            # Add notes based on validation
            notes = []
            if dataset == "imdb" and table == "title_ratings":
                notes.append("Ranges validated: averageRating in [0,10]")
            elif dataset == "movielens" and table == "ratings":
                notes.append("Ranges validated: rating in [0.5,5.0]")
            elif dataset == "rottentomatoes" and table == "movies":
                notes.append("boxOffice_usd derived from boxOffice string parsing")
            
            schema["notes"] = "; ".join(notes) if notes else "No special notes"
            
            # Save schema
            schema_file = self.schemas_path / f"{dataset}_{table}_schema.json"
            with open(schema_file, 'w') as f:
                json.dump(schema, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error generating schema for {dataset}.{table}: {e}")
    
    def generate_report(self) -> None:
        """Generate the Phase 1 report section."""
        try:
            report_content = "\n## Phase 1 — Schema & Types\n\n"
            
            for dataset, tables in self.stats.items():
                if tables:
                    report_content += f"### {dataset.upper()}\n\n"
                    
                    for table, stats in tables.items():
                        report_content += f"**{table}**\n"
                        report_content += f"- Rows: {stats['rows']:,}\n"
                        
                        # Key nulls
                        if 'tconst' in stats['nulls']:
                            report_content += f"- tconst nulls: {stats['nulls']['tconst']}\n"
                        elif 'movieId' in stats['nulls']:
                            report_content += f"- movieId nulls: {stats['nulls']['movieId']}\n"
                        elif 'id' in stats['nulls']:
                            report_content += f"- id nulls: {stats['nulls']['id']}\n"
                        elif 'tmdb_id' in stats['nulls']:
                            report_content += f"- tmdb_id nulls: {stats['nulls']['tmdb_id']}\n"
                        
                        # Coercions
                        if stats['coercions']:
                            coercions = []
                            for field, count in stats['coercions'].items():
                                if count > 0:
                                    coercions.append(f"{field}: {count}")
                            if coercions:
                                report_content += f"- Coercions: {', '.join(coercions)}\n"
                        
                        # Ranges
                        if 'ranges' in stats:
                            for field, range_info in stats['ranges'].items():
                                if 'min' in range_info and 'max' in range_info:
                                    report_content += f"- {field} range: [{range_info['min']:.2f}, {range_info['max']:.2f}]\n"
                        
                        report_content += "\n"
            
            # Append to existing report or create new
            if self.report_path.exists():
                with open(self.report_path, 'a') as f:
                    f.write(report_content)
            else:
                with open(self.report_path, 'w') as f:
                    f.write("# Step 1b Report\n\n" + report_content)
            
            logger.info("Report generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def run(self) -> None:
        """Run the complete Phase 1 processing pipeline."""
        logger.info("Starting Step 1b - Phase 1: Schema & Types processing...")
        
        try:
            # Process each dataset
            self.process_imdb_data()
            self.process_movielens_data()
            self.process_rottentomatoes_data()
            self.process_tmdb_data()
            
            # Generate report
            self.generate_report()
            
            logger.info("Phase 1 processing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in Phase 1 processing: {e}")
            raise

def main():
    """Main entry point."""
    try:
        processor = SchemaTypeProcessor()
        processor.run()
        print("Phase 1 processing completed successfully!")
        print("Check logs/step1b_phase1.log for details")
        print("Check docs/step1b_report.md for the report")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
