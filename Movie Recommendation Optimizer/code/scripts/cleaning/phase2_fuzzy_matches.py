#!/usr/bin/env python3
"""
Step 1b Phase 2 - Sub-phase 2.4: Fuzzy Title Matches
Attempts fuzzy title matching for unresolved RT titles using constrained blocks (year ±1, optional runtime)
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
from fuzzywuzzy import fuzz

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
    """Main function for Sub-phase 2.4"""
    logging.info("=== STARTING STEP 1B PHASE 2 SUB-PHASE 2.4: FUZZY TITLE MATCHES ===")
    
    # Define paths
    unresolved_path = "data/normalized/bridges/checkpoints/linked_blocked_unresolved.parquet"
    rt_movies_path = "Rotten Tomatoes/rotten_tomatoes_movies.csv"
    rt_top_movies_path = "Rotten Tomatoes/rotten_tomatoes_top_movies.csv"
    imdb_basics_path = "IMDB datasets/title.basics.tsv"
    output_path = "data/normalized/bridges/checkpoints/linked_fuzzy.parquet"
    conflicts_path = "data/normalized/bridges/checkpoints/linked_fuzzy_conflicts.parquet"
    still_unresolved_path = "data/normalized/bridges/checkpoints/linked_fuzzy_unresolved.parquet"
    
    # Check if output exists and remove for fresh run
    if os.path.exists(output_path):
        os.remove(output_path)
        logging.info(f"Removed existing output: {output_path}")
    
    # Load unresolved RT rows from 2.3
    logging.info(f"Loading unresolved RT rows from: {unresolved_path}")
    unresolved_df = pd.read_parquet(unresolved_path)
    total_unresolved = len(unresolved_df)
    logging.info(f"Input unresolved count: {total_unresolved:,}")
    
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
    
    # Process unresolved RT rows for fuzzy matching
    all_results = []
    conflicts = []
    still_unresolved = []
    
    logging.info(f"Processing {total_unresolved:,} unresolved RT rows for fuzzy matching...")
    
    for i, (_, rt_row) in enumerate(unresolved_df.iterrows()):
        if i % 10 == 0:
            logging.info(f"Processing RT row {i:,}/{total_unresolved:,}")
        
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
        
        # Apply runtime bucket filter if both RT and IMDb runtime available
        if rt_runtime is not None and len(year_candidates) > 0:
            year_candidates = year_candidates[
                (year_candidates["runtimeMinutes"] >= rt_runtime - 5) & 
                (year_candidates["runtimeMinutes"] <= rt_runtime + 5)
            ]
        
        if len(year_candidates) == 0:
            # No candidates in block
            still_unresolved.append(rt_row)
            continue
        
        # Compute fuzzy similarity scores
        scores = []
        for _, imdb_row in year_candidates.iterrows():
            imdb_title_norm = imdb_row['title_norm']
            # Use token sort ratio for better handling of word order differences
            score = fuzz.token_sort_ratio(title_norm, imdb_title_norm)
            scores.append({
                'score': score,
                'imdb_row': imdb_row
            })
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply thresholds
        best_score = scores[0]['score']
        best_imdb_row = scores[0]['imdb_row']
        
        if best_score >= 90:
            # High confidence match
            # Check if there are multiple ≥90 matches
            high_score_matches = [s for s in scores if s['score'] >= 90]
            
            if len(high_score_matches) == 1:
                # Unique best match
                result_row = {
                    'rt_id': rt_id,
                    'tconst': best_imdb_row['tconst'],
                    'tmdbId': None,  # Set null in 2.4
                    'movieId': None,  # Set null in 2.4
                    'title_norm': title_norm,
                    'year': year,
                    'title_source': 'rt_title',
                    'link_method': 'fuzzy_title_year',
                    'match_score': float(best_score),
                    'source_ml': False,
                    'source_imdb': True,
                    'source_rt': True,
                    'canonical_id': best_imdb_row['tconst'],
                    'titleType': best_imdb_row['titleType']
                }
                all_results.append(result_row)
            else:
                # Multiple high-score matches - resolve by titleType preference
                movie_matches = [s for s in high_score_matches if s['imdb_row']['titleType'] == 'movie']
                if len(movie_matches) == 1:
                    # Single movie match - use it
                    best_match = movie_matches[0]
                    result_row = {
                        'rt_id': rt_id,
                        'tconst': best_match['imdb_row']['tconst'],
                        'tmdbId': None,
                        'movieId': None,
                        'title_norm': title_norm,
                        'year': year,
                        'title_source': 'rt_title',
                        'link_method': 'fuzzy_title_year',
                        'match_score': float(best_match['score']),
                        'source_ml': False,
                        'source_imdb': True,
                        'source_rt': True,
                        'canonical_id': best_match['imdb_row']['tconst'],
                        'titleType': best_match['imdb_row']['titleType']
                    }
                    all_results.append(result_row)
                else:
                    # Multiple movie matches or no movie matches - send to conflicts
                    for match in high_score_matches:
                        conflict_row = {
                            'rt_id': rt_id,
                            'tconst': match['imdb_row']['tconst'],
                            'tmdbId': None,
                            'movieId': None,
                            'title_norm': title_norm,
                            'year': year,
                            'title_source': 'rt_title',
                            'link_method': 'fuzzy_title_year',
                            'match_score': float(match['score']),
                            'source_ml': False,
                            'source_imdb': True,
                            'source_rt': True,
                            'canonical_id': match['imdb_row']['tconst'],
                            'titleType': match['imdb_row']['titleType']
                        }
                        conflicts.append(conflict_row)
        
        elif best_score >= 80:
            # Borderline match - send to conflicts for manual review
            for match in scores[:3]:  # Top 3 borderline matches
                if match['score'] >= 80:
                    conflict_row = {
                        'rt_id': rt_id,
                        'tconst': match['imdb_row']['tconst'],
                        'tmdbId': None,
                        'movieId': None,
                        'title_norm': title_norm,
                        'year': year,
                        'title_source': 'rt_title',
                        'link_method': 'fuzzy_title_year',
                        'match_score': float(match['score']),
                        'source_ml': False,
                        'source_imdb': True,
                        'source_rt': True,
                        'canonical_id': match['imdb_row']['tconst'],
                        'titleType': match['imdb_row']['titleType']
                    }
                    conflicts.append(conflict_row)
        
        else:
            # No match above threshold
            still_unresolved.append(rt_row)
    
    # Combine all results
    logging.info("Combining fuzzy matching results...")
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
        logging.info("No fuzzy matches found - creating empty output")
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
    
    logging.info(f"Input unresolved count: {total_unresolved:,}")
    logging.info(f"Count matched ≥90 (main output): {matched_count:,}")
    logging.info(f"Count borderline 80–89 (conflicts): {conflicts_count:,}")
    logging.info(f"Count <80 unresolved (still unmatched): {still_unresolved_count:,}")
    
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
            logging.info(f"  match_score: {row['match_score']:.1f}")
            logging.info("")
    else:
        logging.info("No matched rows to sample")
    
    # Final summary
    logging.info("=== SUB-PHASE 2.4 COMPLETE ===")
    logging.info(f"Fuzzy matches ≥90: {matched_count:,} rows")
    logging.info(f"Borderline 80-89: {conflicts_count:,} rows")
    logging.info(f"Still unresolved: {still_unresolved_count:,} rows")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


























