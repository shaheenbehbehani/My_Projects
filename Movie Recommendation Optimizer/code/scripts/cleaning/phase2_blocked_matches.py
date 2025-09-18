#!/usr/bin/env python3
"""
Step 1b Phase 2 - Sub-phase 2.3: Blocked Exact Matches
Links unresolved RT titles to IMDb using blocked exact matches with year ±1 and optional runtime buckets
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/step1b_phase2.log', mode='a'),
        logging.StreamHandler()
    ]
)

def normalize_title(title):
    """Normalize title: lowercase, NFKC, remove non-alnum to single spaces, strip"""
    if pd.isna(title):
        return None
    
    # Convert to string, lowercase, NFKC normalize
    title_str = str(title).lower()
    import unicodedata
    title_str = unicodedata.normalize('NFKC', title_str)
    
    # Remove non-alphanumeric characters, replace with single space
    import re
    title_str = re.sub(r'[^a-z0-9]+', ' ', title_str)
    title_str = title_str.strip()
    
    return title_str if title_str else None

def create_rt_id(title_norm, year):
    """Create stable rt_id: sha1(title_norm + '::' + year)"""
    if pd.isna(title_norm) or pd.isna(year):
        return None
    
    combined = f"{title_norm}::{year}"
    return hashlib.sha1(combined.encode('utf-8')).hexdigest()

def main():
    """Main function for Sub-phase 2.3"""
    logging.info("=== STARTING STEP 1B PHASE 2 SUB-PHASE 2.3: BLOCKED EXACT MATCHES ===")
    
    # Define paths
    unresolved_path = "data/normalized/bridges/checkpoints/linked_exact_unresolved.parquet"
    rt_movies_path = "Rotten Tomatoes/rotten_tomatoes_movies.csv"
    rt_top_movies_path = "Rotten Tomatoes/rotten_tomatoes_top_movies.csv"
    imdb_basics_path = "IMDB datasets/title.basics.tsv"
    output_path = "data/normalized/bridges/checkpoints/linked_blocked.parquet"
    conflicts_path = "data/normalized/bridges/checkpoints/linked_blocked_conflicts.parquet"
    still_unresolved_path = "data/normalized/bridges/checkpoints/linked_blocked_unresolved.parquet"
    
    # Check if output exists and remove for fresh run
    if os.path.exists(output_path):
        os.remove(output_path)
        logging.info(f"Removed existing output: {output_path}")
    
    # Load unresolved RT rows from 2.2
    logging.info(f"Loading unresolved RT rows from: {unresolved_path}")
    unresolved_df = pd.read_parquet(unresolved_path)
    total_unresolved = len(unresolved_df)
    logging.info(f"Total unresolved RT rows input: {total_unresolved:,}")
    
    # Load minimal IMDb basics with same optimization as previous phases
    logging.info("Loading minimal IMDb basics table...")
    imdb_cols = ["tconst", "primaryTitle", "originalTitle", "titleType", "startYear", "runtimeMinutes"]
    imdb_dtypes = {
        "tconst": "string",
        "primaryTitle": "string",
        "originalTitle": "string",
        "titleType": "string",
        "startYear": "string",  # will coerce to Int32 later
        "runtimeMinutes": "string",  # will coerce to Int32 later
    }
    
    basics = pd.read_csv(
        imdb_basics_path,
        sep="\t",
        usecols=imdb_cols,
        dtype=imdb_dtypes,
        na_values="\\N",
        low_memory=False,
    )
    
    # Coerce startYear and runtimeMinutes to numeric
    basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce").astype("Int32")
    basics["runtimeMinutes"] = pd.to_numeric(basics["runtimeMinutes"], errors="coerce").astype("Int32")
    
    # Filter to movie-like types and valid years
    movie_types = ["movie", "tvMovie"]
    basics = basics[basics["titleType"].isin(movie_types)]
    basics = basics.dropna(subset=["startYear"])
    
    # Create title_norm for IMDb side
    basics["title_norm"] = basics["primaryTitle"].map(normalize_title)
    basics = basics.dropna(subset=["title_norm"])
    
    # Build lookup index for fast blocked matches
    # Group by title_norm and startYear for efficient blocking
    basics_grouped = basics.groupby(["title_norm", "startYear"]).agg({
        "tconst": "first",
        "titleType": "first",
        "runtimeMinutes": "first"
    }).reset_index()
    
    logging.info(f"IMDb basics indexed: {len(basics_grouped):,} movie titles with valid year")
    
    # Load RT datasets to get runtime information
    logging.info("Loading RT datasets for runtime information...")
    
    # RT Movies
    rt_movies = pd.read_csv(rt_movies_path)
    logging.info(f"RT Movies loaded: {len(rt_movies):,} rows")
    
    # RT Top Movies
    rt_top = pd.read_csv(rt_top_movies_path)
    logging.info(f"RT Top Movies loaded: {len(rt_top):,} rows")
    
    # Create RT runtime lookup
    rt_runtime_lookup = {}
    
    # Process RT Movies for runtime
    for _, row in rt_movies.iterrows():
        title = None
        if 'movie_title' in rt_movies.columns and pd.notna(row['movie_title']):
            title = row['movie_title']
        elif 'title' in rt_movies.columns and pd.notna(row['title']):
            title = row['title']
        
        year = None
        if 'year' in rt_movies.columns and pd.notna(row['year']):
            try:
                year = int(row['year'])
            except (ValueError, TypeError):
                pass
        elif 'release_year' in rt_movies.columns and pd.notna(row['release_year']):
            try:
                year = int(row['release_year'])
            except (ValueError, TypeError):
                pass
        
        runtime = None
        if 'runtime' in rt_movies.columns and pd.notna(row['runtime']):
            try:
                runtime = int(row['runtime'])
            except (ValueError, TypeError):
                pass
        elif 'duration' in rt_movies.columns and pd.notna(row['duration']):
            try:
                runtime = int(row['duration'])
            except (ValueError, TypeError):
                pass
        
        if title and year:
            title_norm = normalize_title(title)
            if title_norm:
                rt_runtime_lookup[(title_norm, year)] = runtime
    
    # Process RT Top Movies for runtime
    for _, row in rt_top.iterrows():
        title = None
        if 'movie_title' in rt_top.columns and pd.notna(row['movie_title']):
            title = row['movie_title']
        elif 'title' in rt_top.columns and pd.notna(row['title']):
            title = row['title']
        
        year = None
        if 'year' in rt_top.columns and pd.notna(row['year']):
            try:
                year = int(row['year'])
            except (ValueError, TypeError):
                pass
        elif 'release_year' in rt_top.columns and pd.notna(row['release_year']):
            try:
                year = int(row['release_year'])
            except (ValueError, TypeError):
                pass
        
        runtime = None
        if 'runtime' in rt_top.columns and pd.notna(row['runtime']):
            try:
                runtime = int(row['runtime'])
            except (ValueError, TypeError):
                pass
        elif 'duration' in rt_top.columns and pd.notna(row['duration']):
            try:
                runtime = int(row['duration'])
            except (ValueError, TypeError):
                pass
        
        if title and year:
            title_norm = normalize_title(title)
            if title_norm:
                rt_runtime_lookup[(title_norm, year)] = runtime
    
    logging.info(f"RT runtime lookup created with {len(rt_runtime_lookup):,} entries")
    
    # Process unresolved RT rows in batches
    batch_size = 20
    all_results = []
    conflicts = []
    still_unresolved = []
    
    logging.info(f"Processing unresolved RT rows in batches of {batch_size:,}")
    
    for i in range(0, len(unresolved_df), batch_size):
        batch_end = min(i + batch_size, len(unresolved_df))
        batch = unresolved_df.iloc[i:batch_end]
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(unresolved_df)-1)//batch_size + 1} (rows {i:,}–{batch_end:,})")
        
        for j, (_, rt_row) in enumerate(batch.iterrows()):
            if j % 10 == 0:
                logging.info(f"  Processing row {i+j:,}/{len(unresolved_df):,}")
            
            rt_id = rt_row['rt_id']
            title_norm = rt_row['title_norm']
            year = rt_row['year']
            
            # Get RT runtime if available
            rt_runtime = rt_runtime_lookup.get((title_norm, year))
            
            # Find IMDb candidates within year block (±1)
            year_candidates = basics_grouped[
                (basics_grouped["startYear"] >= year - 1) & 
                (basics_grouped["startYear"] <= year + 1)
            ].copy()
            
            # Filter by exact title_norm match
            title_candidates = year_candidates[year_candidates["title_norm"] == title_norm]
            
            # Apply runtime bucket filter if both RT and IMDb runtime available
            if rt_runtime is not None and len(title_candidates) > 0:
                title_candidates = title_candidates[
                    (title_candidates["runtimeMinutes"] >= rt_runtime - 5) & 
                    (title_candidates["runtimeMinutes"] <= rt_runtime + 5)
                ]
            
            # Handle matches
            if len(title_candidates) == 0:
                # No matches found
                still_unresolved.append(rt_row)
            elif len(title_candidates) == 1:
                # Single match found
                imdb_row = title_candidates.iloc[0]
                result_row = {
                    'rt_id': rt_id,
                    'tconst': imdb_row['tconst'],
                    'tmdbId': None,  # Set null in 2.3
                    'movieId': None,  # Set null in 2.3
                    'title_norm': title_norm,
                    'year': year,
                    'title_source': 'rt_title',
                    'link_method': 'blocked_exact',
                    'match_score': None,  # null in 2.3
                    'source_ml': False,
                    'source_imdb': True,
                    'source_rt': True,
                    'canonical_id': imdb_row['tconst'],
                    'titleType': imdb_row['titleType']
                }
                all_results.append(result_row)
            else:
                # Multiple matches found - handle conflicts
                # Prefer movie types
                movie_candidates = title_candidates[title_candidates["titleType"] == "movie"]
                if len(movie_candidates) == 1:
                    # Single movie candidate - use it
                    imdb_row = movie_candidates.iloc[0]
                    result_row = {
                        'rt_id': rt_id,
                        'tconst': imdb_row['tconst'],
                        'tmdbId': None,
                        'movieId': None,
                        'title_norm': title_norm,
                        'year': year,
                        'title_source': 'rt_title',
                        'link_method': 'blocked_exact',
                        'match_score': None,
                        'source_ml': False,
                        'source_imdb': True,
                        'source_rt': True,
                        'canonical_id': imdb_row['tconst'],
                        'titleType': imdb_row['titleType']
                    }
                    all_results.append(result_row)
                elif len(movie_candidates) > 1:
                    # Multiple movie candidates - mark as conflict
                    for _, imdb_row in movie_candidates.iterrows():
                        conflict_row = {
                            'rt_id': rt_id,
                            'tconst': imdb_row['tconst'],
                            'tmdbId': None,
                            'movieId': None,
                            'title_norm': title_norm,
                            'year': year,
                            'title_source': 'rt_title',
                            'link_method': 'blocked_exact',
                            'match_score': None,
                            'source_ml': False,
                            'source_imdb': True,
                            'source_rt': True,
                            'canonical_id': imdb_row['tconst'],
                            'titleType': imdb_row['titleType']
                        }
                        conflicts.append(conflict_row)
                else:
                    # No movie candidates, use first available
                    imdb_row = title_candidates.iloc[0]
                    result_row = {
                        'rt_id': rt_id,
                        'tconst': imdb_row['tconst'],
                        'tmdbId': None,
                        'movieId': None,
                        'title_norm': title_norm,
                        'year': year,
                        'title_source': 'rt_title',
                        'link_method': 'blocked_exact',
                        'match_score': None,
                        'source_ml': False,
                        'source_imdb': True,
                        'source_rt': True,
                        'canonical_id': imdb_row['tconst'],
                        'titleType': imdb_row['titleType']
                    }
                    all_results.append(result_row)
    
    # Combine all results
    logging.info("Combining batch results...")
    if all_results:
        matched_df = pd.DataFrame(all_results)
        
        # Convert to proper dtypes
        matched_df['rt_id'] = matched_df['rt_id'].astype('string')
        matched_df['tconst'] = matched_df['tconst'].astype('string')
        matched_df['tmdbId'] = matched_df['tmdbId'].astype('Int64')
        matched_df['movieId'] = matched_df['movieId'].astype('Int64')
        matched_df['title_norm'] = matched_df['title_norm'].astype('string')
        matched_df['year'] = matched_df['year'].astype('Int32')
        matched_df['title_source'] = matched_df['title_source'].astype('string')
        matched_df['link_method'] = matched_df['link_method'].astype('string')
        matched_df['match_score'] = matched_df['match_score'].astype('Float32')
        matched_df['source_ml'] = matched_df['source_ml'].astype('boolean')
        matched_df['source_imdb'] = matched_df['source_imdb'].astype('boolean')
        matched_df['source_rt'] = matched_df['source_rt'].astype('boolean')
        matched_df['canonical_id'] = matched_df['canonical_id'].astype('string')
        matched_df['titleType'] = matched_df['titleType'].astype('string')
        
        # Ensure schema order
        schema_order = [
            'rt_id', 'tconst', 'tmdbId', 'movieId', 'title_norm', 'year',
            'title_source', 'link_method', 'match_score', 'source_ml', 
            'source_imdb', 'source_rt', 'canonical_id', 'titleType'
        ]
        
        matched_df = matched_df[schema_order]
        
        # Save main output
        matched_df.to_parquet(output_path, index=False)
        logging.info(f"Main output saved to: {output_path} ({len(matched_df):,} rows)")
    else:
        logging.info("No matches found - creating empty output")
        # Create empty DataFrame with correct schema
        empty_df = pd.DataFrame(columns=[
            'rt_id', 'tconst', 'tmdbId', 'movieId', 'title_norm', 'year',
            'title_source', 'link_method', 'match_score', 'source_ml', 
            'source_imdb', 'source_rt', 'canonical_id', 'titleType'
        ])
        empty_df.to_parquet(output_path, index=False)
        logging.info(f"Empty output saved to: {output_path}")
    
    # Handle conflicts
    if conflicts:
        conflicts_df = pd.DataFrame(conflicts)
        conflicts_df.to_parquet(conflicts_path, index=False)
        logging.info(f"Conflicts written to: {conflicts_path} ({len(conflicts):,} rows)")
    else:
        logging.info("No conflicts found")
    
    # Handle still unresolved
    if still_unresolved:
        still_unresolved_df = pd.DataFrame(still_unresolved)
        still_unresolved_df.to_parquet(still_unresolved_path, index=False)
        logging.info(f"Still unresolved rows written to: {still_unresolved_path} ({len(still_unresolved):,} rows)")
    else:
        logging.info("No still unresolved rows")
    
    # QA counts and analysis
    logging.info("=== QA ANALYSIS ===")
    
    matched_count = len(all_results) if all_results else 0
    conflicts_count = len(conflicts)
    still_unresolved_count = len(still_unresolved)
    
    logging.info(f"Total unresolved RT rows input: {total_unresolved:,}")
    logging.info(f"Number matched in blocked exact pass: {matched_count:,}")
    logging.info(f"Number remaining unresolved after 2.3: {still_unresolved_count:,}")
    logging.info(f"Number written to _conflicts.parquet: {conflicts_count:,}")
    
    # Log sample matched rows
    logging.info("=== 5 SAMPLE MATCHED ROWS ===")
    if matched_count > 0:
        sample_rows = matched_df.sample(n=min(5, matched_count), random_state=42)
        for i, (_, row) in enumerate(sample_rows.iterrows()):
            logging.info(f"Sample {i+1}:")
            logging.info(f"  rt_id: {row['rt_id']}")
            logging.info(f"  title_norm: {row['title_norm']}")
            logging.info(f"  year: {row['year']}")
            logging.info(f"  tconst: {row['tconst']}")
            logging.info("")
    else:
        logging.info("No matched rows to sample")
    
    # Final summary
    logging.info("=== SUB-PHASE 2.3 COMPLETE ===")
    logging.info(f"Matched: {matched_count:,} rows")
    logging.info(f"Conflicts: {conflicts_count:,} rows")
    logging.info(f"Still unresolved: {still_unresolved_count:,} rows")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


























