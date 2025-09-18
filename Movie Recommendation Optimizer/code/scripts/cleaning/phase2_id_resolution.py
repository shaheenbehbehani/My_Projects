#!/usr/bin/env python3
"""
Step 1b - Phase 2: ID Resolution & Deduping for Movie Recommendation Optimizer

This script builds a cross-source ID bridge and canonical movie catalog, resolves duplicates
within and across sources, and outputs clean, deduped tables for use in later phases.

Inputs (from Phase 1 outputs):
- data/normalized/imdb/{title.basics.parquet,title.crew.parquet,title.ratings.parquet}
- data/normalized/movielens/{movies.parquet,links.parquet,ratings.parquet,tags.parquet}
- data/normalized/rottentomatoes/{movies.parquet,top_movies.parquet,reviews.parquet}
- data/normalized/tmdb/*.parquet (optional)

Outputs:
- data/normalized/_bridges/id_bridge.parquet - one row per canonical movie with all linked IDs
- data/normalized/_masters/movies_master.parquet - canonical movie record with best metadata
- Per-source deduped tables (overwrite inputs)
- Gap analysis CSVs
- Updated report section
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import warnings

# Try to import RapidFuzz for fuzzy matching, fallback to basic if not available
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logging.warning("RapidFuzz not available, fuzzy matching will be limited")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/step1b_phase2.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def blocked_fuzzy_link(
    left_df: pd.DataFrame, 
    right_df: pd.DataFrame,
    left_key_cols: Tuple[str, str] = ("title_norm", "year_norm"),
    right_key_cols: Tuple[str, str] = ("title_norm", "year_norm"),
    threshold: int = 92, 
    year_tol: int = 1, 
    candidate_cap: int = 500,
    max_rows: Optional[int] = None, 
    progress_every: int = 5000
) -> pd.DataFrame:
    """
    Returns df with: left_index, right_index, score, match_method='fuzzy_block'
    Uses blocking on (prefix of title_norm, year bucket) to limit candidates.
    """
    if not RAPIDFUZZ_AVAILABLE:
        logger.warning("RapidFuzz not available, skipping fuzzy linking")
        return pd.DataFrame()
    
    logger.info(f"Starting blocked fuzzy linking: {len(left_df)} left rows, {len(right_df)} right rows")
    
    # Apply max_rows limit if specified
    if max_rows and len(left_df) > max_rows:
        # Use deterministic sampling
        np.random.seed(42)
        left_df = left_df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        logger.info(f"Limited to {max_rows} rows for fuzzy linking")
    
    # Create blocking keys
    def create_block_key(title: str, year: Any) -> Tuple[str, int]:
        if pd.isna(title) or pd.isna(year):
            return ("", 0)
        
        # Clean title and take first 2 tokens
        import re
        clean_title = re.sub(r'[^a-z0-9 ]', '', str(title).lower())
        tokens = clean_title.split()[:2]
        title_prefix = ' '.join(tokens)
        
        # Year bin
        try:
            year_bin = int(year)
        except (ValueError, TypeError):
            year_bin = 0
            
        return (title_prefix, year_bin)
    
    # Build right_df blocking index
    right_df = right_df.copy()
    right_df['block_key'] = right_df.apply(
        lambda row: create_block_key(row[right_key_cols[0]], row[right_key_cols[1]]), 
        axis=1
    )
    
    # Group right_df by block key
    block_groups = {}
    for idx, row in right_df.iterrows():
        block_key = row['block_key']
        if block_key not in block_groups:
            block_groups[block_key] = []
        block_groups[block_key].append(idx)
    
    logger.info(f"Created {len(block_groups)} blocks for right_df")
    
    # Process left_df rows
    matches = []
    total_processed = 0
    total_accepted = 0
    total_skipped = 0
    
    for left_idx, left_row in left_df.iterrows():
        total_processed += 1
        
        # Create block key for left row
        left_block_key = create_block_key(left_row[left_key_cols[0]], left_row[left_key_cols[1]])
        
        if left_block_key == ("", 0):
            continue
        
        # Gather candidates from this block and adjacent year blocks
        candidates = []
        for year_offset in range(-year_tol, year_tol + 1):
            candidate_key = (left_block_key[0], left_block_key[1] + year_offset)
            if candidate_key in block_groups:
                candidates.extend(block_groups[candidate_key])
        
        # Skip if too many candidates
        if len(candidates) > candidate_cap:
            total_skipped += 1
            continue
        
        if candidates:
            # Get candidate titles
            candidate_titles = [right_df.loc[idx, right_key_cols[0]] for idx in candidates]
            left_title = left_row[left_key_cols[0]]
            
            # Find best match
            try:
                best_match = process.extractOne(
                    left_title, 
                    candidate_titles, 
                    scorer=fuzz.token_set_ratio
                )
                
                if best_match and best_match[1] >= threshold:
                    best_candidate_idx = candidates[best_match[2]]
                    matches.append({
                        'left_index': left_idx,
                        'right_index': best_candidate_idx,
                        'score': best_match[1],
                        'match_method': 'fuzzy_block'
                    })
                    total_accepted += 1
                    
            except Exception as e:
                logger.warning(f"Error in fuzzy matching for row {left_idx}: {e}")
                continue
        
        # Progress logging
        if total_processed % progress_every == 0:
            logger.info(f"Fuzzy progress: {total_processed}/{len(left_df)} processed, "
                       f"{total_accepted} accepted, {total_skipped} skipped")
    
    logger.info(f"Fuzzy linking completed: {total_accepted} matches, {total_skipped} skipped")
    
    return pd.DataFrame(matches)

class IDResolutionProcessor:
    """Process and resolve movie IDs across different sources with deduplication."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_path = Path("data")
        self.normalized_path = self.base_path / "normalized"
        self.bridges_path = self.normalized_path / "_bridges"
        self.masters_path = self.normalized_path / "_masters"
        self.report_path = Path("docs/step1b_report.md")
        
        # Ensure output directories exist
        self.bridges_path.mkdir(exist_ok=True)
        self.masters_path.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            "imdb": {},
            "movielens": {},
            "rottentomatoes": {},
            "tmdb": {},
            "bridges": {},
            "coverage": {},
            "fuzzy": {}
        }
        
        # Data storage
        self.datasets = {}
        self.id_bridge = None
        self.movies_master = None
        
    def normalize_title(self, title: str) -> str:
        """Normalize title: lowercase, strip punctuation, collapse whitespace, remove year."""
        if pd.isna(title) or title == '':
            return ''
        
        # Convert to string and lowercase
        title_str = str(title).lower()
        
        # Remove year in parentheses at the end (e.g., "Movie Title (2020)")
        import re
        title_str = re.sub(r'\s*\(\d{4}\)\s*$', '', title_str)
        
        # Strip punctuation and collapse whitespace
        title_str = re.sub(r'[^\w\s]', ' ', title_str)
        title_str = re.sub(r'\s+', ' ', title_str).strip()
        
        return title_str
    
    def extract_year_from_title(self, title: str) -> Optional[int]:
        """Extract year from title if present in parentheses."""
        if pd.isna(title) or title == '':
            return None
        
        import re
        match = re.search(r'\((\d{4})\)', str(title))
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None
    
    def extract_year_from_date(self, date_col: pd.Series) -> pd.Series:
        """Extract year from date columns."""
        if date_col.dtype == 'object':
            # Try to parse dates and extract years
            parsed_dates = pd.to_datetime(date_col, errors='coerce')
            return parsed_dates.dt.year.astype('Int64')
        elif 'datetime' in str(date_col.dtype):
            return date_col.dt.year.astype('Int64')
        else:
            return date_col.astype('Int64')
    
    def _coerce_rt_dates(self, rt_movies: pd.DataFrame) -> pd.DataFrame:
        """Normalize Rotten Tomatoes dates and titles."""
        logger.info("Normalizing Rotten Tomatoes dates and titles...")
        
        # Ensure these exist even if missing
        for col in ["releaseDateTheaters", "releaseDateStreaming"]:
            if col not in rt_movies.columns:
                rt_movies[col] = pd.Series([pd.NaT] * len(rt_movies))
        
        # Coerce to datetime; keep UTC then drop tz for pure dates
        rt_movies["releaseDateTheaters"] = pd.to_datetime(rt_movies["releaseDateTheaters"], errors="coerce", utc=True)
        rt_movies["releaseDateStreaming"] = pd.to_datetime(rt_movies["releaseDateStreaming"], errors="coerce", utc=True)
        
        # Build best_year from explicit 'year' or the parsed dates
        best_year = rt_movies.get("year")
        if best_year is None or not pd.api.types.is_integer_dtype(best_year):
            best_year = pd.Series([pd.NA] * len(rt_movies), index=rt_movies.index, dtype="Int64")
        
        theaters_year = rt_movies["releaseDateTheaters"].dt.year.astype("Int64")
        streaming_year = rt_movies["releaseDateStreaming"].dt.year.astype("Int64")
        
        rt_movies["year_norm"] = best_year.fillna(theaters_year).fillna(streaming_year).astype("Int64")
        
        # Normalize title
        def _norm_title(s):
            import unicodedata
            import re
            if pd.isna(s):
                return ""
            s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
            s = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9 ]", " ", s)).strip().lower()
            return s
        
        rt_movies["title_norm"] = rt_movies["title"].map(_norm_title)
        
        # Track statistics for reporting
        year_source_counts = {
            'explicit_year': (best_year.notna()).sum(),
            'theaters_date': (theaters_year.notna()).sum(),
            'streaming_date': (streaming_year.notna()).sum(),
            'final_year_norm': rt_movies["year_norm"].notna().sum()
        }
        
        logger.info(f"RT date normalization stats: {year_source_counts}")
        
        return rt_movies, year_source_counts
    
    def heartbeat(self, message: str) -> None:
        """Log progress message with timestamp."""
        logger.info(message)
    
    def link_rt_via_map(self, rt_df: pd.DataFrame, core_mappings: dict) -> pd.DataFrame:
        """Link Rotten Tomatoes records using dictionary mapping instead of merge."""
        import time
        
        n = len(rt_df)
        batch_size = max(1, self.config.get('batch_size', 50000))
        progress_every = self.config.get('progress_every', 5000)
        
        logger.info(f"Starting RT map-based linking for {n:,} records (batch size: {batch_size:,})")
        
        t0 = time.time()
        last = t0
        out = []
        
        for i in range(0, n, batch_size):
            chunk = rt_df.iloc[i:i+batch_size].copy()
            keys = chunk["key_ty"]
            
            # Map to existing IDs
            chunk["tconst_from_ty"] = keys.map(core_mappings['tconst'])
            chunk["movieId_from_ty"] = keys.map(core_mappings['movieId'])
            chunk["tmdbId_from_ty"] = keys.map(core_mappings['tmdbId'])
            
            out.append(chunk[["id", "title_norm", "year_norm", "tconst_from_ty", "movieId_from_ty", "tmdbId_from_ty"]])
            
            # Progress logging
            if ((i // batch_size) % max(1, progress_every // batch_size)) == 0:
                now = time.time()
                logger.info(f"RT linking (map) progress: {min(i+batch_size, n):,}/{n:,} (+{min(batch_size, n-i)}) in {now-last:0.1f}s")
                last = now
        
        linked = pd.concat(out, ignore_index=True)
        total_time = time.time() - t0
        
        # Count matches
        matches = linked[['tconst_from_ty', 'movieId_from_ty', 'tmdbId_from_ty']].notna().any(axis=1).sum()
        logger.info(f"RT map linking completed in {total_time:.1f}s: {matches:,}/{n:,} matched ({matches/n*100:.1f}%)")
        
        return linked
    
    def load_and_prepare_datasets(self) -> None:
        """Load all normalized datasets and prepare them for ID resolution."""
        logger.info("Loading and preparing datasets...")
        
        # Load IMDb datasets
        try:
            imdb_basics = pd.read_parquet(self.normalized_path / "imdb" / "title_basics.parquet")
            imdb_crew = pd.read_parquet(self.normalized_path / "imdb" / "title_crew.parquet")
            imdb_ratings = pd.read_parquet(self.normalized_path / "imdb" / "title_ratings.parquet")
            
            # Prepare IMDb data
            imdb_basics['title_norm'] = imdb_basics['primaryTitle'].apply(self.normalize_title)
            imdb_basics['year_norm'] = imdb_basics['startYear']
            
            # Filter to movies only
            imdb_movies = imdb_basics[imdb_basics['titleType'] == 'movie'].copy()
            
            # Merge with crew and ratings
            imdb_movies = imdb_movies.merge(
                imdb_crew[['tconst', 'directors', 'writers']], 
                on='tconst', 
                how='left'
            )
            imdb_movies = imdb_movies.merge(
                imdb_ratings[['tconst', 'averageRating', 'numVotes']], 
                on='tconst', 
                how='left'
            )
            
            self.datasets['imdb'] = imdb_movies
            logger.info(f"Loaded IMDb: {len(imdb_movies)} movies")
            
        except Exception as e:
            logger.error(f"Error loading IMDb data: {e}")
            self.datasets['imdb'] = pd.DataFrame()
        
        # Load MovieLens datasets
        try:
            ml_movies = pd.read_parquet(self.normalized_path / "movielens" / "movies.parquet")
            ml_links = pd.read_parquet(self.normalized_path / "movielens" / "links.parquet")
            ml_ratings = pd.read_parquet(self.normalized_path / "movielens" / "ratings.parquet")
            ml_tags = pd.read_parquet(self.normalized_path / "movielens" / "tags.parquet")
            
            # Prepare MovieLens data
            ml_movies['title_norm'] = ml_movies['title'].apply(self.normalize_title)
            ml_movies['year_norm'] = ml_movies['title'].apply(self.extract_year_from_title)
            
            # Merge with links
            ml_movies = ml_movies.merge(ml_links, on='movieId', how='left')
            
            # Add derived tconst from imdbId
            ml_movies['tconst'] = ml_movies['imdbId'].apply(
                lambda x: f"tt{str(x).zfill(7)}" if pd.notna(x) else None
            )
            
            # Aggregate ratings and tags
            ratings_agg = ml_ratings.groupby('movieId').agg({
                'rating': ['count', 'mean'],
                'userId': 'nunique'
            }).reset_index()
            ratings_agg.columns = ['movieId', 'rating_count', 'rating_mean', 'unique_users']
            
            tags_agg = ml_tags.groupby('movieId').agg({
                'tag': 'count',
                'userId': 'nunique'
            }).reset_index()
            tags_agg.columns = ['movieId', 'tag_count', 'unique_taggers']
            
            ml_movies = ml_movies.merge(ratings_agg, on='movieId', how='left')
            ml_movies = ml_movies.merge(tags_agg, on='movieId', how='left')
            
            self.datasets['movielens'] = ml_movies
            logger.info(f"Loaded MovieLens: {len(ml_movies)} movies")
            
        except Exception as e:
            logger.error(f"Error loading MovieLens data: {e}")
            self.datasets['movielens'] = pd.DataFrame()
        
        # Load Rotten Tomatoes datasets
        try:
            rt_movies = pd.read_parquet(self.normalized_path / "rottentomatoes" / "movies.parquet")
            rt_top_movies = pd.read_parquet(self.normalized_path / "rottentomatoes" / "top_movies.parquet")
            rt_reviews = pd.read_parquet(self.normalized_path / "rottentomatoes" / "reviews.parquet")
            
            # Normalize RT movies data using the new method
            rt_movies, rt_movies_stats = self._coerce_rt_dates(rt_movies)
            
            # Prepare RT top movies data
            rt_top_movies['title_norm'] = rt_top_movies['title'].apply(self.normalize_title)
            rt_top_movies['year_norm'] = rt_top_movies['year']
            
            # Combine RT datasets
            rt_combined = pd.concat([rt_movies, rt_top_movies], ignore_index=True)
            
            # Aggregate reviews
            reviews_agg = rt_reviews.groupby('id').agg({
                'reviewId': 'count',
                'isTopCritic': 'sum'
            }).reset_index()
            reviews_agg.columns = ['id', 'review_count', 'top_critic_count']
            
            rt_combined = rt_combined.merge(reviews_agg, on='id', how='left')
            
            # Store RT statistics for reporting
            self.stats['rottentomatoes']['date_normalization'] = rt_movies_stats
            
            self.datasets['rottentomatoes'] = rt_combined
            logger.info(f"Loaded Rotten Tomatoes: {len(rt_combined)} movies")
            
        except Exception as e:
            logger.error(f"Error loading Rotten Tomatoes data: {e}")
            self.datasets['rottentomatoes'] = pd.DataFrame()
        
        # Load TMDB datasets
        try:
            tmdb_files = list((self.normalized_path / "tmdb").glob("*.parquet"))
            if tmdb_files:
                tmdb_data = pd.read_parquet(tmdb_files[0])
                tmdb_data['title_norm'] = tmdb_data['title'].apply(self.normalize_title)
                tmdb_data['year_norm'] = self.extract_year_from_date(tmdb_data['release_date'])
                
                self.datasets['tmdb'] = tmdb_data
                logger.info(f"Loaded TMDB: {len(tmdb_data)} movies")
            else:
                logger.info("No TMDB data found")
                self.datasets['tmdb'] = pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading TMDB data: {e}")
            self.datasets['tmdb'] = pd.DataFrame()
    
    def dedupe_intra_source(self) -> None:
        """Remove duplicates within each source."""
        logger.info("Performing intra-source deduplication...")
        
        # IMDb deduplication
        if not self.datasets['imdb'].empty:
            initial_count = len(self.datasets['imdb'])
            self.datasets['imdb'] = self.datasets['imdb'].drop_duplicates(subset=['tconst'])
            
            # If still duplicates, keep row with most non-nulls
            if len(self.datasets['imdb']) < initial_count:
                logger.info(f"IMDb: Removed {initial_count - len(self.datasets['imdb'])} duplicate tconst entries")
            
            self.stats['imdb']['deduped_rows'] = len(self.datasets['imdb'])
            self.stats['imdb']['initial_rows'] = initial_count
        
        # MovieLens deduplication
        if not self.datasets['movielens'].empty:
            initial_count = len(self.datasets['movielens'])
            self.datasets['movielens'] = self.datasets['movielens'].drop_duplicates(subset=['movieId'])
            self.stats['movielens']['deduped_rows'] = len(self.datasets['movielens'])
            self.stats['movielens']['initial_rows'] = initial_count
        
        # Rotten Tomatoes deduplication
        if not self.datasets['rottentomatoes'].empty:
            initial_count = len(self.datasets['rottentomatoes'])
            # For RT, use id if available, otherwise (title_norm, year_norm)
            rt_data = self.datasets['rottentomatoes'].copy()
            rt_data['dedup_key'] = rt_data['id'].fillna(rt_data['title_norm'] + '_' + rt_data['year_norm'].astype(str))
            rt_data = rt_data.drop_duplicates(subset=['dedup_key'])
            rt_data = rt_data.drop('dedup_key', axis=1)
            
            self.datasets['rottentomatoes'] = rt_data
            self.stats['rottentomatoes']['deduped_rows'] = len(rt_data)
            self.stats['rottentomatoes']['initial_rows'] = initial_count
        
        # TMDB deduplication
        if not self.datasets['tmdb'].empty:
            initial_count = len(self.datasets['tmdb'])
            self.datasets['tmdb'] = self.datasets['tmdb'].drop_duplicates(subset=['tmdb_id'])
            self.stats['tmdb']['deduped_rows'] = len(self.datasets['tmdb'])
            self.stats['tmdb']['initial_rows'] = initial_count
    
    def build_id_bridge(self) -> None:
        """Build the cross-source ID bridge."""
        logger.info("Building cross-source ID bridge...")
        
        bridge_records = []
        
        # Start with MovieLens as the base (has links to IMDb and TMDB)
        logger.info("Processing MovieLens records...")
        if not self.datasets['movielens'].empty:
            ml_data = self.datasets['movielens'].copy()
            
            for _, row in ml_data.iterrows():
                record = {
                    'movieId': row['movieId'],
                    'tconst': row['tconst'],
                    'tmdbId': row['tmdbId'],
                    'rt_id': None,  # Will be filled later
                    'title': row['title'],
                    'title_norm': row['title_norm'],
                    'year_norm': row['year_norm'],
                    'has_imdb': pd.notna(row['tconst']),
                    'has_tmdb': pd.notna(row['tmdbId']),
                    'has_rt': False,  # Will be updated
                    'has_ml': True,
                    'link_method': 'via_links' if pd.notna(row['tconst']) or pd.notna(row['tmdbId']) else 'ml_only'
                }
                bridge_records.append(record)
            logger.info(f"Added {len(ml_data)} MovieLens records")
        
        # Add IMDb movies not in MovieLens
        logger.info("Processing IMDb records...")
        if not self.datasets['imdb'].empty:
            imdb_data = self.datasets['imdb'].copy()
            existing_tconsts = set(record['tconst'] for record in bridge_records if pd.notna(record['tconst']))
            
            for _, row in imdb_data.iterrows():
                if pd.notna(row['tconst']) and row['tconst'] not in existing_tconsts:
                    record = {
                        'movieId': None,
                        'tconst': row['tconst'],
                        'tmdbId': None,
                        'rt_id': None,
                        'title': row['primaryTitle'],
                        'title_norm': row['title_norm'],
                        'year_norm': row['year_norm'],
                        'has_imdb': True,
                        'has_tmdb': False,
                        'has_rt': False,
                        'has_ml': False,
                        'link_method': 'imdb_only'
                    }
                    bridge_records.append(record)
            logger.info(f"Added {len(imdb_data)} IMDb records")
        
        # Add TMDB movies not in MovieLens
        logger.info("Processing TMDB records...")
        if not self.datasets['tmdb'].empty:
            tmdb_data = self.datasets['tmdb'].copy()
            existing_tmdb_ids = set(record['tmdbId'] for record in bridge_records if pd.notna(record['tmdbId']))
            
            for _, row in tmdb_data.iterrows():
                if pd.notna(row['tmdb_id']) and row['tmdb_id'] not in existing_tmdb_ids:
                    record = {
                        'movieId': None,
                        'tconst': None,
                        'tmdbId': row['tmdb_id'],
                        'rt_id': None,
                        'title': row['title'],
                        'title_norm': row['title_norm'],
                        'year_norm': row['year_norm'],
                        'has_imdb': False,
                        'has_tmdb': True,
                        'has_rt': False,
                        'has_ml': False,
                        'link_method': 'tmdb_only'
                    }
                    bridge_records.append(record)
            logger.info(f"Added {len(tmdb_data)} TMDB records")
        
        # Add Rotten Tomatoes movies
        if not self.datasets['rottentomatoes'].empty:
            rt_data = self.datasets['rottentomatoes'].copy()
            logger.info(f"Processing Rotten Tomatoes records... total={len(rt_data):,}")
            
            # Filter out rows without year_norm and save them to gap file
            rt_with_year = rt_data[rt_data['year_norm'].notna()].copy()
            rt_no_year = rt_data[rt_data['year_norm'].isna()].copy()
            
            if not rt_no_year.empty:
                gap_file = self.bridges_path / "gaps" / "rt_without_year.csv"
                gap_file.parent.mkdir(exist_ok=True)
                rt_no_year.to_csv(gap_file, index=False)
                logger.info(f"Saved {len(rt_no_year)} RT rows without year to {gap_file}")
            
            # Store gap statistics
            self.stats['rottentomatoes']['gap_stats'] = {
                'total_rows': len(rt_data),
                'with_year': len(rt_with_year),
                'without_year': len(rt_no_year)
            }
            
            # Use map-based linking for better performance
            if self.config.get('rt_link_mode', 'map') == 'map':
                # Build core mappings for fast lookup
                logger.info("Building core mappings for RT linking...")
                
                # Create key_ty for core records
                core_df = pd.DataFrame(bridge_records)
                core_df["year_norm"] = core_df["year_norm"].astype("Int64")
                
                # Build best core mapping for (title_norm, year_norm) → IDs without groupby.apply (fast)
                core_df["_coverage"] = core_df[["tconst","movieId","tmdbId"]].notna().sum(axis=1).astype("int16")
                core_df["_nv"] = core_df.get("numVotes", pd.Series([-1] * len(core_df))).fillna(-1).astype("int32")
                core_df["key_ty"] = core_df["title_norm"].fillna("").astype(str) + "|" + core_df["year_norm"].astype("Int64").astype(str)
                
                core_sorted = core_df.sort_values(["key_ty","_nv","_coverage"], ascending=[True, False, False], kind="stable")
                best_core = core_sorted.drop_duplicates("key_ty", keep="first")[["key_ty","tconst","movieId","tmdbId"]]
                
                logger.info(f"Built core mappings: {len(best_core)} unique keys")
                
                map_tconst = best_core.set_index("key_ty")["tconst"]
                map_movie = best_core.set_index("key_ty")["movieId"]
                map_tmdb = best_core.set_index("key_ty")["tmdbId"]
                
                core_mappings = {
                    'tconst': map_tconst,
                    'movieId': map_movie,
                    'tmdbId': map_tmdb
                }
                
                # Create key_ty for RT records
                rt_with_year["year_norm"] = rt_with_year["year_norm"].astype("Int64")
                rt_with_year["key_ty"] = rt_with_year["title_norm"].fillna("").astype(str) + "|" + rt_with_year["year_norm"].astype("Int64").astype(str)
                
                # Link via map
                linked_rt = self.link_rt_via_map(rt_with_year, core_mappings)
                
                # Ensure linked_rt has the key_ty column for deduplication
                if "key_ty" not in linked_rt.columns:
                    linked_rt["key_ty"] = linked_rt["title_norm"].fillna("").astype(str) + "|" + linked_rt["year_norm"].astype("Int64").astype(str)
                
                # Attach RT ids to core via key, without merge
                rt_id_by_key = rt_with_year.drop_duplicates("key_ty").set_index("key_ty")["id"]
                core_df["rt_id"] = core_df["key_ty"].map(rt_id_by_key)
                
                # Prefer existing IDs; else use RT map outputs
                linked_rt_dedup = linked_rt.drop_duplicates("key_ty")
                core_df["tconst"] = core_df["tconst"].where(core_df["tconst"].notna(), core_df["key_ty"].map(linked_rt_dedup.set_index("key_ty")["tconst_from_ty"]))
                core_df["movieId"] = core_df["movieId"]  # already present for ML
                core_df["tmdbId"] = core_df["tmdbId"].where(core_df["tmdbId"].notna(), core_df["key_ty"].map(linked_rt_dedup.set_index("key_ty")["tmdbId_from_ty"]))
                
                # Build id_bridge with only needed columns and compute canonical_id deterministically (no heavy joins)
                import numpy as np
                
                # Vectorized approach: build a string key then hash once
                canon_key = (
                    core_df["tconst"].fillna("") + "|" +
                    core_df["tmdbId"].fillna(-1).astype("Int64").astype(str) + "|" +
                    core_df["rt_id"].fillna("").astype(str) + "|" +
                    core_df["title_norm"].fillna("") + "|" +
                    core_df["year_norm"].fillna(-1).astype("Int64").astype(str)
                )
                core_df["canonical_id"] = "cn_" + canon_key.map(lambda s: format(pd.util.hash_array(np.array([s], dtype=object))[0] & 0xFFFFFFFFFFFFFFFF, "016x"))
                
                id_bridge = core_df[[
                    "canonical_id","movieId","tconst","tmdbId","rt_id",
                    "title","title_norm","year_norm","link_method"
                ]].copy()
                id_bridge["has_imdb"] = id_bridge["tconst"].notna()
                id_bridge["has_tmdb"] = id_bridge["tmdbId"].notna()
                id_bridge["has_rt"] = id_bridge["rt_id"].notna()
                id_bridge["has_ml"] = id_bridge["movieId"].notna()
                
                # Build movies_master with a simple groupby agg (no apply)
                movies_master = (
                    id_bridge
                    .groupby("canonical_id", as_index=False)
                    .agg({
                        "title_norm":"first",
                        "year_norm":"first"
                    })
                )
                
                # Store the results
                self.id_bridge = id_bridge
                self.movies_master = movies_master
                
                # Save unmatched RT records to gap file
                logger.info(f"Calculating unmatched RT records...")
                logger.info(f"best_core shape: {best_core.shape}, columns: {best_core.columns.tolist()}")
                logger.info(f"rt_with_year shape: {rt_with_year.shape}, columns: {rt_with_year.columns.tolist()}")
                
                unmatched = rt_with_year[~rt_with_year["key_ty"].isin(best_core["key_ty"].values)]
                logger.info(f"Unmatched calculation completed: {len(unmatched)} rows")
                
                if not unmatched.empty:
                    gap_file = self.bridges_path / "gaps" / "rt_no_match_title_year.csv"
                    gap_file.parent.mkdir(exist_ok=True)
                    unmatched.to_csv(gap_file, index=False)
                    logger.info(f"Saved {len(unmatched)} unmatched RT rows to {gap_file}")
                
                # Store map linking statistics
                self.stats['rottentomatoes']['map_linking'] = {
                    'attempted': len(rt_with_year),
                    'matched': len(rt_with_year) - len(unmatched),
                    'unmatched': len(unmatched),
                    'match_rate': (len(rt_with_year) - len(unmatched)) / len(rt_with_year) * 100
                }
                
                logger.info(f"RT map linking completed: {len(rt_with_year) - len(unmatched):,} matched, {len(unmatched):,} unmatched")
                return  # Skip the rest of the bridge building since we've already built it
                
            else:
                # Fallback to original merge-based approach
                logger.info("Using merge-based RT linking...")
                
                # Process only rows with year_norm
                for i, (_, row) in enumerate(rt_with_year.iterrows()):
                    # Progress logging every 50,000 rows
                    if (i % 50000) == 0:
                        logger.info(f"RT linking progress: {i:,}/{len(rt_with_year):,}")
                    
                    # Try to find matches by title/year
                    title_year_match = None
                    for record in bridge_records:
                        if (record['title_norm'] == row['title_norm'] and 
                            pd.notna(record['year_norm']) and 
                            pd.notna(row['year_norm']) and
                            record['year_norm'] == row['year_norm']):
                            title_year_match = record
                            break
                    
                    if title_year_match:
                        # Update existing record
                        title_year_match['rt_id'] = row['id']
                        title_year_match['has_rt'] = True
                        if title_year_match['link_method'] == 'via_links':
                            title_year_match['link_method'] = 'via_title_year'
                        elif title_year_match['link_method'] == 'ml_only':
                            title_year_match['link_method'] = 'via_title_year'
                    else:
                        # Create new record
                        record = {
                            'movieId': None,
                            'tconst': None,
                            'tmdbId': None,
                            'rt_id': row['id'],
                            'title': row['title'],
                            'title_norm': row['title_norm'],
                            'year_norm': row['year_norm'],
                            'has_imdb': False,
                            'has_tmdb': False,
                            'has_rt': True,
                            'has_ml': False,
                            'link_method': 'rt_only'
                        }
                        bridge_records.append(record)
            
            logger.info(f"Completed RT processing: {len(rt_with_year)} rows with year, {len(rt_no_year)} rows without year")
        
        # Create DataFrame and add canonical_id
        logger.info("Creating DataFrame from bridge records...")
        self.id_bridge = pd.DataFrame(bridge_records)
        logger.info(f"DataFrame created with {len(self.id_bridge)} records")
        
        # Convert boolean columns to proper boolean type
        logger.info("Converting boolean columns...")
        for col in ['has_imdb', 'has_tmdb', 'has_rt', 'has_ml']:
            self.id_bridge[col] = self.id_bridge[col].astype(bool)
        logger.info("Boolean columns converted")
        
        # Generate canonical_id based on available IDs
        logger.info("Generating canonical IDs...")
        def generate_canonical_id(row):
            id_parts = []
            if pd.notna(row['tconst']):
                id_parts.append(f"imdb_{row['tconst']}")
            if pd.notna(row['tmdbId']):
                id_parts.append(f"tmdb_{row['tmdbId']}")
            if pd.notna(row['rt_id']):
                id_parts.append(f"rt_{row['rt_id']}")
            if pd.notna(row['movieId']):
                id_parts.append(f"ml_{row['movieId']}")
            
            if not id_parts:
                # Fallback to title and year hash
                fallback = f"{row['title_norm']}_{row['year_norm']}"
                id_parts.append(f"fallback_{hashlib.md5(fallback.encode()).hexdigest()[:8]}")
            
            # Sort for deterministic ordering
            id_parts.sort()
            combined = "_".join(id_parts)
            return f"cn_{hashlib.md5(combined.encode()).hexdigest()[:12]}"
        
        self.id_bridge['canonical_id'] = self.id_bridge.apply(generate_canonical_id, axis=1)
        logger.info("Canonical IDs generated")
        
        # Apply fuzzy linking if enabled
        if not self.config.get('disable_fuzzy', True):
            logger.info("Applying fuzzy linking...")
            fuzzy_matches = self._apply_fuzzy_linking()
            if not fuzzy_matches.empty:
                self._merge_fuzzy_matches(fuzzy_matches)
                logger.info(f"Applied {len(fuzzy_matches)} fuzzy matches")
        else:
            logger.info("Fuzzy linking disabled")
        
        # Ensure canonical_id uniqueness
        if self.id_bridge['canonical_id'].duplicated().any():
            logger.warning("Duplicate canonical_ids found, resolving...")
            # Group by canonical_id and merge records
            grouped = self.id_bridge.groupby('canonical_id').agg({
                'movieId': 'first',
                'tconst': 'first',
                'tmdbId': 'first',
                'rt_id': 'first',
                'title': 'first',
                'title_norm': 'first',
                'year_norm': 'first',
                'has_imdb': lambda x: x.astype(bool).max(),
                'has_tmdb': lambda x: x.astype(bool).max(),
                'has_rt': lambda x: x.astype(bool).max(),
                'has_ml': lambda x: x.astype(bool).max(),
                'link_method': 'first'
            }).reset_index()
            self.id_bridge = grouped
        
        logger.info(f"Built ID bridge with {len(self.id_bridge)} canonical movies")
        
        # Calculate coverage statistics
        total_ml = len(self.datasets['movielens']) if not self.datasets['movielens'].empty else 0
        ml_with_imdb = len(self.id_bridge[(self.id_bridge['has_imdb'] == True) & (self.id_bridge['has_ml'] == True)])
        ml_with_tmdb = len(self.id_bridge[(self.id_bridge['has_tmdb'] == True) & (self.id_bridge['has_ml'] == True)])
        ml_with_rt = len(self.id_bridge[(self.id_bridge['has_rt'] == True) & (self.id_bridge['has_ml'] == True)])
        
        self.stats['coverage'] = {
            'total_canonical': len(self.id_bridge),
            'ml_with_imdb_pct': (ml_with_imdb / total_ml * 100) if total_ml > 0 else 0,
            'ml_with_tmdb_pct': (ml_with_tmdb / total_ml * 100) if total_ml > 0 else 0,
            'ml_with_rt_pct': (ml_with_rt / total_ml * 100) if total_ml > 0 else 0,
            'avg_sources_per_canonical': self.id_bridge[['has_imdb', 'has_tmdb', 'has_rt', 'has_ml']].astype(int).sum(axis=1).mean()
        }
    
    def build_movies_master(self) -> None:
        """Build the canonical movies master table."""
        logger.info("Building movies master table...")
        
        master_records = []
        
        for _, bridge_row in self.id_bridge.iterrows():
            canonical_id = bridge_row['canonical_id']
            
            # Collect best metadata from each source
            best_title = bridge_row['title']
            best_year = bridge_row['year_norm']
            best_runtime = None
            best_overview = None
            best_poster = None
            
            best_title_source = 'bridge'
            best_year_source = 'bridge'
            best_runtime_source = None
            best_overview_source = None
            
            # IMDb metadata
            if pd.notna(bridge_row['tconst']):
                imdb_row = self.datasets['imdb'][self.datasets['imdb']['tconst'] == bridge_row['tconst']]
                if not imdb_row.empty:
                    imdb_data = imdb_row.iloc[0]
                    if pd.notna(imdb_data['primaryTitle']):
                        best_title = imdb_data['primaryTitle']
                        best_title_source = 'imdb'
                    if pd.notna(imdb_data['startYear']):
                        best_year = imdb_data['startYear']
                        best_year_source = 'imdb'
                    if pd.notna(imdb_data['runtimeMinutes']):
                        best_runtime = imdb_data['runtimeMinutes']
                        best_runtime_source = 'imdb'
            
            # TMDB metadata
            if pd.notna(bridge_row['tmdbId']):
                tmdb_row = self.datasets['tmdb'][self.datasets['tmdb']['tmdb_id'] == bridge_row['tmdbId']]
                if not tmdb_row.empty:
                    tmdb_data = tmdb_row.iloc[0]
                    if pd.notna(tmdb_data['overview']):
                        best_overview = tmdb_data['overview']
                        best_overview_source = 'tmdb'
                    if pd.notna(tmdb_data['popularity']):
                        best_poster = tmdb_data['popularity']  # Using popularity as proxy for poster availability
            
            # Rotten Tomatoes metadata
            if pd.notna(bridge_row['rt_id']):
                rt_row = self.datasets['rottentomatoes'][self.datasets['rottentomatoes']['id'] == bridge_row['rt_id']]
                if not rt_row.empty:
                    rt_data = rt_row.iloc[0]
                    if pd.notna(rt_data['runtimeMinutes']):
                        if best_runtime is None or best_runtime_source != 'imdb':  # Prefer IMDb
                            best_runtime = rt_data['runtimeMinutes']
                            best_runtime_source = 'rt'
            
            # Create master record
            master_record = {
                'canonical_id': canonical_id,
                'title': best_title,
                'title_norm': bridge_row['title_norm'],
                'year_norm': best_year,
                'runtime_minutes': best_runtime,
                'overview': best_overview,
                'poster_available': best_poster is not None,
                'best_title_source': best_title_source,
                'best_year_source': best_year_source,
                'best_runtime_source': best_runtime_source,
                'best_overview_source': best_overview_source,
                'has_imdb': bridge_row['has_imdb'],
                'has_tmdb': bridge_row['has_tmdb'],
                'has_rt': bridge_row['has_rt'],
                'has_ml': bridge_row['has_ml'],
                'link_method': bridge_row['link_method']
            }
            
            master_records.append(master_record)
        
        self.movies_master = pd.DataFrame(master_records)
        logger.info(f"Built movies master with {len(self.movies_master)} canonical movies")
    
    def generate_gap_analysis(self) -> None:
        """Generate gap analysis CSVs."""
        logger.info("Generating gap analysis...")
        
        # MovieLens without IDs
        if not self.datasets['movielens'].empty:
            ml_without_ids = self.datasets['movielens'][
                (self.datasets['movielens']['tconst'].isna()) & 
                (self.datasets['movielens']['tmdbId'].isna())
            ].copy()
            
            if not ml_without_ids.empty:
                gap_file = self.bridges_path / "ml_without_ids.csv"
                ml_without_ids.to_csv(gap_file, index=False)
                logger.info(f"Generated gap file: {gap_file} with {len(ml_without_ids)} records")
        
        # Rotten Tomatoes without year
        if not self.datasets['rottentomatoes'].empty:
            rt_without_year = self.datasets['rottentomatoes'][
                self.datasets['rottentomatoes']['year_norm'].isna()
            ].copy()
            
            if not rt_without_year.empty:
                gap_file = self.bridges_path / "rt_without_year.csv"
                rt_without_year.to_csv(gap_file, index=False)
                logger.info(f"Generated gap file: {gap_file} with {len(rt_without_year)} records")
        
        # Fuzzy match candidates for review
        if not self.id_bridge.empty:
            # Find records that might benefit from fuzzy matching
            potential_fuzzy = self.id_bridge[
                (self.id_bridge['link_method'].isin(['ml_only', 'imdb_only', 'tmdb_only', 'rt_only'])) &
                (self.id_bridge['title_norm'].notna()) &
                (self.id_bridge['year_norm'].notna())
            ].copy()
            
            if not potential_fuzzy.empty:
                gap_file = self.bridges_path / "fuzzy_candidates_review.csv"
                potential_fuzzy.to_csv(gap_file, index=False)
                logger.info(f"Generated gap file: {gap_file} with {len(potential_fuzzy)} records")
    
    def save_outputs(self) -> None:
        """Save all outputs."""
        logger.info("Saving outputs...")
        
        # Save ID bridge
        if self.id_bridge is not None:
            bridge_path = self.bridges_path / "id_bridge.parquet"
            self.id_bridge.to_parquet(bridge_path, compression='snappy')
            self.id_bridge.to_csv(self.bridges_path / "id_bridge.csv", index=False)
            self.heartbeat(f"Wrote {bridge_path} ({len(self.id_bridge):,} rows)")
        
        # Save movies master
        if self.movies_master is not None:
            master_path = self.masters_path / "movies_master.parquet"
            self.movies_master.to_parquet(master_path, compression='snappy')
            self.movies_master.to_csv(self.masters_path / "movies_master.csv", index=False)
            self.heartbeat(f"Wrote {master_path} ({len(self.movies_master):,} rows)")
        
        # Save deduped source tables
        if not self.datasets['imdb'].empty:
            self.datasets['imdb'].to_parquet(self.normalized_path / "imdb" / "title_basics.parquet", compression='snappy')
            self.datasets['imdb'].to_csv(self.normalized_path / "imdb" / "title_basics.csv", index=False)
        
        if not self.datasets['movielens'].empty:
            self.datasets['movielens'].to_parquet(self.normalized_path / "movielens" / "movies.parquet", compression='snappy')
            self.datasets['movielens'].to_csv(self.normalized_path / "movielens" / "movies.csv", index=False)
        
        if not self.datasets['rottentomatoes'].empty:
            self.datasets['rottentomatoes'].to_parquet(self.normalized_path / "rottentomatoes" / "movies.parquet", compression='snappy')
            self.datasets['rottentomatoes'].to_csv(self.normalized_path / "rottentomatoes" / "movies.csv", index=False)
        
        if not self.datasets['tmdb'].empty:
            self.datasets['tmdb'].to_parquet(self.normalized_path / "tmdb" / "movies.parquet", compression='snappy')
            self.datasets['tmdb'].to_csv(self.normalized_path / "tmdb" / "movies.csv", index=False)
    
    def _apply_fuzzy_linking(self) -> pd.DataFrame:
        """Apply fuzzy linking to unlinked rows."""
        logger.info("Starting fuzzy linking process...")
        
        # Find unlinked rows (those with only one source)
        unlinked_mask = (
            (self.id_bridge['has_imdb'].astype(int) + 
             self.id_bridge['has_tmdb'].astype(int) + 
             self.id_bridge['has_rt'].astype(int) + 
             self.id_bridge['has_ml'].astype(int)) == 1
        )
        
        unlinked_rows = self.id_bridge[unlinked_mask].copy()
        logger.info(f"Found {len(unlinked_rows)} unlinked rows for fuzzy matching")
        
        if len(unlinked_rows) == 0:
            return pd.DataFrame()
        
        # Early exit guard for very large datasets
        if len(unlinked_rows) > 1000000 and not self.config.get('disable_fuzzy', True):
            logger.warning(f"Too many unlinked rows ({len(unlinked_rows):,}), suggesting --disable-fuzzy")
            logger.info("Proceeding without fuzzy linking")
            return pd.DataFrame()
        
        # Apply fuzzy linking between different source types
        fuzzy_matches = []
        
        # Try to link unlinked IMDb rows with RT/TMDB
        imdb_only = unlinked_rows[unlinked_rows['has_imdb'] == True]
        if not imdb_only.empty:
            rt_tmdb = self.id_bridge[
                (self.id_bridge['has_rt'] == True) | (self.id_bridge['has_tmdb'] == True)
            ]
            if not rt_tmdb.empty:
                matches = blocked_fuzzy_link(
                    imdb_only, rt_tmdb,
                    threshold=self.config.get('fuzzy_threshold', 92),
                    year_tol=self.config.get('year_tolerance', 1),
                    candidate_cap=self.config.get('candidate_cap', 500),
                    max_rows=self.config.get('max_fuzzy', 25000),
                    progress_every=self.config.get('progress_every', 5000)
                )
                if not matches.empty:
                    fuzzy_matches.append(matches)
        
        # Try to link unlinked RT rows with IMDb/TMDB
        rt_only = unlinked_rows[unlinked_rows['has_rt'] == True]
        if not rt_only.empty:
            imdb_tmdb = self.id_bridge[
                (self.id_bridge['has_imdb'] == True) | (self.id_bridge['has_tmdb'] == True)
            ]
            if not imdb_tmdb.empty:
                matches = blocked_fuzzy_link(
                    rt_only, imdb_tmdb,
                    threshold=self.config.get('fuzzy_threshold', 92),
                    year_tol=self.config.get('year_tolerance', 1),
                    candidate_cap=self.config.get('candidate_cap', 500),
                    max_rows=self.config.get('max_fuzzy', 25000),
                    progress_every=self.config.get('progress_every', 5000)
                )
                if not matches.empty:
                    fuzzy_matches.append(matches)
        
        # Combine all matches
        if fuzzy_matches:
            combined_matches = pd.concat(fuzzy_matches, ignore_index=True)
            logger.info(f"Total fuzzy matches found: {len(combined_matches)}")
            
            # Save fuzzy matches
            combined_matches.to_parquet(self.bridges_path / "fuzzy_matches.parquet", compression='snappy')
            combined_matches.to_csv(self.bridges_path / "fuzzy_matches.csv", index=False)
            
            # Update stats
            self.stats['fuzzy'] = {
                'attempted': len(unlinked_rows),
                'accepted': len(combined_matches),
                'threshold': self.config.get('fuzzy_threshold', 92),
                'year_tolerance': self.config.get('year_tolerance', 1),
                'max_fuzzy': self.config.get('max_fuzzy', 25000)
            }
            
            return combined_matches
        else:
            logger.info("No fuzzy matches found")
            return pd.DataFrame()
    
    def _merge_fuzzy_matches(self, fuzzy_matches: pd.DataFrame) -> None:
        """Merge fuzzy matches into the ID bridge."""
        logger.info("Merging fuzzy matches into ID bridge...")
        
        for _, match in fuzzy_matches.iterrows():
            left_idx = match['left_index']
            right_idx = match['right_index']
            score = match['score']
            
            # Get the rows to merge
            left_row = self.id_bridge.iloc[left_idx]
            right_row = self.id_bridge.iloc[right_idx]
            
            # Merge the rows by combining their source flags
            merged_row = left_row.copy()
            merged_row['has_imdb'] = left_row['has_imdb'] or right_row['has_imdb']
            merged_row['has_tmdb'] = left_row['has_tmdb'] or right_row['has_tmdb']
            merged_row['has_rt'] = left_row['has_rt'] or right_row['has_rt']
            merged_row['has_ml'] = left_row['has_ml'] or right_row['has_ml']
            merged_row['link_method'] = 'via_fuzzy'
            
            # Update IDs if they were missing
            if pd.isna(merged_row['tconst']) and pd.notna(right_row['tconst']):
                merged_row['tconst'] = right_row['tconst']
            if pd.isna(merged_row['tmdbId']) and pd.notna(right_row['tmdbId']):
                merged_row['tmdbId'] = right_row['tmdbId']
            if pd.isna(merged_row['rt_id']) and pd.notna(right_row['rt_id']):
                merged_row['rt_id'] = right_row['rt_id']
            if pd.isna(merged_row['movieId']) and pd.notna(right_row['movieId']):
                merged_row['movieId'] = right_row['movieId']
            
            # Update the left row
            self.id_bridge.iloc[left_idx] = merged_row
            
            # Remove the right row (it's now merged)
            self.id_bridge = self.id_bridge.drop(right_idx).reset_index(drop=True)
            
            # Update indices for remaining matches
            fuzzy_matches.loc[fuzzy_matches['left_index'] > right_idx, 'left_index'] -= 1
            fuzzy_matches.loc[fuzzy_matches['right_index'] > right_idx, 'right_index'] -= 1
        
        logger.info("Fuzzy matches merged into ID bridge")
    
    def generate_report(self) -> None:
        """Generate the Phase 2 report section."""
        try:
            report_content = "\n## Phase 2 — ID Resolution & Deduping\n\n"
            
            # Deduplication summary
            report_content += "### Deduplication Summary\n\n"
            for source, stats in self.stats.items():
                if source in ['imdb', 'movielens', 'rottentomatoes', 'tmdb'] and stats:
                    if 'initial_rows' in stats and 'deduped_rows' in stats:
                        removed = stats['initial_rows'] - stats['deduped_rows']
                        report_content += f"**{source.upper()}**: {stats['initial_rows']:,} → {stats['deduped_rows']:,} (removed {removed:,} duplicates)\n"
            
            # Coverage summary
            if self.stats['coverage']:
                coverage = self.stats['coverage']
                report_content += f"\n### Link Coverage Summary\n\n"
                report_content += f"- **Total Canonical Movies**: {coverage['total_canonical']:,}\n"
                report_content += f"- **MovieLens with IMDb**: {coverage['ml_with_imdb_pct']:.1f}%\n"
                report_content += f"- **MovieLens with TMDB**: {coverage['ml_with_tmdb_pct']:.1f}%\n"
                report_content += f"- **MovieLens with RT**: {coverage['ml_with_rt_pct']:.1f}%\n"
                report_content += f"- **Avg Sources per Canonical**: {coverage['avg_sources_per_canonical']:.1f}\n"
            
            # Fuzzy matching statistics
            if self.stats.get('fuzzy'):
                fuzzy_stats = self.stats['fuzzy']
                report_content += f"\n### Fuzzy Matching Statistics\n\n"
                report_content += f"- **Attempted**: {fuzzy_stats['attempted']:,}\n"
                report_content += f"- **Accepted**: {fuzzy_stats['accepted']:,}\n"
                report_content += f"- **Threshold**: {fuzzy_stats['threshold']}\n"
                report_content += f"- **Year Tolerance**: {fuzzy_stats['year_tolerance']}\n"
                report_content += f"- **Max Fuzzy**: {fuzzy_stats['max_fuzzy']:,}\n"
                report_content += f"- **Fuzzy Matches File**: `data/normalized/_bridges/fuzzy_matches.parquet`\n"
            
            # Rotten Tomatoes date normalization statistics
            if self.stats.get('rottentomatoes', {}).get('date_normalization'):
                rt_date_stats = self.stats['rottentomatoes']['date_normalization']
                report_content += f"\n### Rotten Tomatoes Date Normalization\n\n"
                report_content += f"- **Explicit Year**: {rt_date_stats['explicit_year']:,}\n"
                report_content += f"- **Theaters Date**: {rt_date_stats['theaters_date']:,}\n"
                report_content += f"- **Streaming Date**: {rt_date_stats['streaming_date']:,}\n"
                report_content += f"- **Final Year Norm**: {rt_date_stats['final_year_norm']:,}\n"
            
            # RT gap statistics
            if self.stats.get('rottentomatoes', {}).get('gap_stats'):
                rt_gap_stats = self.stats['rottentomatoes']['gap_stats']
                report_content += f"\n### Rotten Tomatoes Gap Analysis\n\n"
                report_content += f"- **Total Rows**: {rt_gap_stats['total_rows']:,}\n"
                report_content += f"- **With Year**: {rt_gap_stats['with_year']:,}\n"
                report_content += f"- **Without Year**: {rt_gap_stats['without_year']:,}\n"
                report_content += f"- **Gap File**: `data/normalized/_bridges/gaps/rt_without_year.csv`\n"
            
            # RT map linking statistics
            if self.stats.get('rottentomatoes', {}).get('map_linking'):
                rt_map_stats = self.stats['rottentomatoes']['map_linking']
                report_content += f"\n### Rotten Tomatoes Map Linking\n\n"
                report_content += f"- **Link Mode**: {self.config.get('rt_link_mode', 'map')}\n"
                report_content += f"- **Attempted**: {rt_map_stats['attempted']:,}\n"
                report_content += f"- **Matched**: {rt_map_stats['matched']:,}\n"
                report_content += f"- **Unmatched**: {rt_map_stats['unmatched']:,}\n"
                report_content += f"- **Match Rate**: {rt_map_stats['match_rate']:.1f}%\n"
                if rt_map_stats['unmatched'] > 0:
                    report_content += f"- **Unmatched File**: `data/normalized/_bridges/gaps/rt_no_match_title_year.csv`\n"
            
            # Gap analysis
            report_content += f"\n### Gap Analysis\n\n"
            report_content += f"- **Gap Files Generated**: Check `data/normalized/_bridges/` for detailed gap analysis\n"
            report_content += f"- **Fuzzy Matching**: {'Available' if RAPIDFUZZ_AVAILABLE else 'Not Available'}\n"
            
            # Append to existing report
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
        """Run the complete Phase 2 processing pipeline."""
        logger.info("Starting Step 1b - Phase 2: ID Resolution & Deduping...")
        
        try:
            # Load and prepare datasets
            self.load_and_prepare_datasets()
            
            # Perform intra-source deduplication
            self.dedupe_intra_source()
            
            # Build ID bridge
            self.build_id_bridge()
            
            # Build movies master
            self.build_movies_master()
            
            # Generate gap analysis
            self.generate_gap_analysis()
            
            # Save all outputs
            self.save_outputs()
            
            # Generate report
            self.generate_report()
            
            logger.info("Phase 2 processing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in Phase 2 processing: {e}")
            raise

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Step 1b - Phase 2: ID Resolution & Deduping')
    parser.add_argument('--disable-fuzzy', action='store_true', 
                       help='Disable fuzzy matching (default: True)')
    parser.add_argument('--fuzzy-threshold', type=int, default=92,
                       help='Fuzzy matching threshold (default: 92)')
    parser.add_argument('--max-fuzzy', type=int, default=25000,
                       help='Maximum rows to attempt fuzzy matching on (default: 25000)')
    parser.add_argument('--year-tolerance', type=int, default=1,
                       help='Year tolerance for fuzzy matching (default: 1)')
    parser.add_argument('--progress-every', type=int, default=5000,
                       help='Log progress every N rows (default: 5000)')
    parser.add_argument('--candidate-cap', type=int, default=500,
                       help='Maximum candidates per block (default: 500)')
    parser.add_argument('--rt-link-mode', choices=['map', 'merge'], default='map',
                       help='RT linking mode: map (fast) or merge (memory-intensive)')
    parser.add_argument('--batch-size', type=int, default=50000,
                       help='Batch size for RT map linking to control memory usage')
    
    args = parser.parse_args()
    
    # Build configuration with environment variable fallbacks
    config = {
        'disable_fuzzy': args.disable_fuzzy or os.getenv('DISABLE_FUZZY', '1') == '1',
        'fuzzy_threshold': int(os.getenv('FUZZY_THRESHOLD', args.fuzzy_threshold)),
        'max_fuzzy': int(os.getenv('MAX_FUZZY', args.max_fuzzy)),
        'year_tolerance': int(os.getenv('YEAR_TOLERANCE', args.year_tolerance)),
        'progress_every': int(os.getenv('PROGRESS_EVERY', args.progress_every)),
        'candidate_cap': int(os.getenv('CANDIDATE_CAP', args.candidate_cap)),
        'rt_link_mode': os.getenv('RT_LINK_MODE', args.rt_link_mode),
        'batch_size': int(os.getenv('BATCH_SIZE', args.batch_size))
    }
    
    # Log configuration
    logger.info(f"Configuration: {config}")
    
    try:
        processor = IDResolutionProcessor(config)
        processor.run()
        print("Phase 2 processing completed successfully!")
        print("Check logs/step1b_phase2.log for details")
        print("Check docs/step1b_report.md for the report")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
