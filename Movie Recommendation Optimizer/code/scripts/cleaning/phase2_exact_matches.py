#!/usr/bin/env python3
"""
Step 1b Phase 2 - Sub-phase 2.2: Exact Title+Year Matches
Links Rotten Tomatoes titles to IMDb using exact (title_norm, year) matches
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
    """Main function for Sub-phase 2.2"""
    logging.info("=== STARTING STEP 1B PHASE 2 SUB-PHASE 2.2: EXACT TITLE+YEAR MATCHES ===")
    
    # Define paths
    rt_movies_path = "Rotten Tomatoes/rotten_tomatoes_movies.csv"
    rt_top_movies_path = "Rotten Tomatoes/rotten_tomatoes_top_movies.csv"
    imdb_basics_path = "IMDB datasets/title.basics.tsv"
    output_path = "data/normalized/bridges/checkpoints/linked_exact.parquet"
    conflicts_path = "data/normalized/bridges/checkpoints/linked_exact_conflicts.parquet"
    unresolved_path = "data/normalized/bridges/checkpoints/linked_exact_unresolved.parquet"
    
    # Check if output exists and remove for fresh run
    if os.path.exists(output_path):
        os.remove(output_path)
        logging.info(f"Removed existing output: {output_path}")
    
    # Load minimal IMDb basics with same optimization as 2.1
    logging.info("Loading minimal IMDb basics table...")
    imdb_cols = ["tconst", "primaryTitle", "originalTitle", "titleType", "startYear"]
    imdb_dtypes = {
        "tconst": "string",
        "primaryTitle": "string",
        "originalTitle": "string",
        "titleType": "string",
        "startYear": "string",  # will coerce to Int32 later
    }
    
    basics = pd.read_csv(
        imdb_basics_path,
        sep="\t",
        usecols=imdb_cols,
        dtype=imdb_dtypes,
        na_values="\\N",
        low_memory=False,
    )
    
    # Coerce startYear to numeric and filter to movie-like types
    basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce").astype("Int32")
    basics = basics.dropna(subset=["startYear"])
    
    # Filter to movie-like types
    movie_types = ["movie", "tvMovie"]
    basics = basics[basics["titleType"].isin(movie_types)]
    
    # Create title_norm for IMDb side
    basics["title_norm"] = basics["primaryTitle"].map(normalize_title)
    basics = basics.dropna(subset=["title_norm"])
    
    # Create lookup key for exact matches
    basics["lookup_key"] = basics["title_norm"] + "::" + basics["startYear"].astype(str)
    
    # Build lookup index for fast exact matches
    # Handle potential duplicates by keeping first occurrence
    basics = basics.drop_duplicates(subset=["lookup_key"], keep="first")
    lookup_index = basics.set_index("lookup_key")[["tconst", "titleType"]]
    logging.info(f"IMDb basics indexed: {len(basics):,} movie titles with valid year (deduplicated)")
    
    # Load RT inputs
    logging.info("Loading Rotten Tomatoes datasets...")
    
    # RT Movies
    rt_movies = pd.read_csv(rt_movies_path)
    logging.info(f"RT Movies loaded: {len(rt_movies):,} rows")
    
    # RT Top Movies
    rt_top = pd.read_csv(rt_top_movies_path)
    logging.info(f"RT Top Movies loaded: {len(rt_top):,} rows")
    
    # Prepare RT inputs
    rt_inputs = []
    
    # Process RT Movies
    for _, row in rt_movies.iterrows():
        # Choose title column
        title = None
        if 'movie_title' in rt_movies.columns and pd.notna(row['movie_title']):
            title = row['movie_title']
        elif 'title' in rt_movies.columns and pd.notna(row['title']):
            title = row['title']
        
        # Choose year column
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
        
        if title and year:
            rt_inputs.append({
                'title': title,
                'year': year,
                'source': 'rt_movies'
            })
    
    # Process RT Top Movies
    for _, row in rt_top.iterrows():
        # Choose title column
        title = None
        if 'movie_title' in rt_top.columns and pd.notna(row['movie_title']):
            title = row['movie_title']
        elif 'title' in rt_top.columns and pd.notna(row['title']):
            title = row['title']
        
        # Choose year column
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
        
        if title and year:
            rt_inputs.append({
                'title': title,
                'year': year,
                'source': 'rt_top'
            })
    
    # Convert to DataFrame
    rt_df = pd.DataFrame(rt_inputs)
    logging.info(f"RT inputs prepared: {len(rt_df):,} rows with valid title and year")
    
    # Build title_norm and rt_id
    rt_df["title_norm"] = rt_df["title"].map(normalize_title)
    rt_df["rt_id"] = rt_df.apply(lambda row: create_rt_id(row["title_norm"], row["year"]), axis=1)
    
    # Drop rows without valid title_norm or rt_id
    rt_df = rt_df.dropna(subset=["title_norm", "rt_id"])
    
    # De-duplicate RT rows on rt_id (keep first)
    before_dedup = len(rt_df)
    rt_df = rt_df.drop_duplicates(subset=["rt_id"], keep="first")
    after_dedup = len(rt_df)
    logging.info(f"RT deduplication: {before_dedup:,} → {after_dedup:,} rows")
    
    # Process RT rows in chunks for exact matching
    chunk_size = 50000
    all_results = []
    
    logging.info(f"Processing RT rows in chunks of {chunk_size:,}")
    
    for start in range(0, len(rt_df), chunk_size):
        end = min(start + chunk_size, len(rt_df))
        chunk = rt_df.iloc[start:end].copy()
        logging.info(f"Processing RT rows {start:,}–{end:,} / {len(rt_df):,}")
        
        # Add heartbeat logging every 5k rows within the chunk
        for i in range(0, len(chunk), 5000):
            logging.info(f"… chunk offset {i:,}/{len(chunk):,}")
        
        # Create lookup key for exact matching
        chunk["lookup_key"] = chunk["title_norm"] + "::" + chunk["year"].astype(str)
        
        # Find exact matches
        chunk["tconst"] = chunk["lookup_key"].map(lookup_index["tconst"])
        chunk["titleType"] = chunk["lookup_key"].map(lookup_index["titleType"])
        
        # Create result rows
        for _, row in chunk.iterrows():
            result_row = {
                'rt_id': row['rt_id'],
                'tconst': row['tconst'],
                'tmdbId': None,  # Set null in 2.2
                'movieId': None,  # Set null in 2.2
                'title_norm': row['title_norm'],
                'year': row['year'],
                'title_source': 'rt_title',
                'link_method': 'exact_title_year',
                'match_score': None,  # null in 2.2
                'source_ml': False,
                'source_imdb': pd.notna(row['tconst']),
                'source_rt': True,
                'canonical_id': row['tconst'] if pd.notna(row['tconst']) else f"rt:{row['rt_id']}",
                'titleType': row['titleType']  # Include titleType for analysis
            }
            all_results.append(result_row)
    
    # Combine all results
    logging.info("Combining chunk results...")
    combined_df = pd.DataFrame(all_results)
    
    # Convert to proper dtypes
    combined_df['rt_id'] = combined_df['rt_id'].astype('string')
    combined_df['tconst'] = combined_df['tconst'].astype('string')
    combined_df['tmdbId'] = combined_df['tmdbId'].astype('Int64')
    combined_df['movieId'] = combined_df['movieId'].astype('Int64')
    combined_df['title_norm'] = combined_df['title_norm'].astype('string')
    combined_df['year'] = combined_df['year'].astype('Int32')
    combined_df['title_source'] = combined_df['title_source'].astype('string')
    combined_df['link_method'] = combined_df['link_method'].astype('string')
    combined_df['match_score'] = combined_df['match_score'].astype('Float32')
    combined_df['source_ml'] = combined_df['source_ml'].astype('boolean')
    combined_df['source_imdb'] = combined_df['source_imdb'].astype('boolean')
    combined_df['source_rt'] = combined_df['source_rt'].astype('boolean')
    combined_df['canonical_id'] = combined_df['canonical_id'].astype('string')
    combined_df['titleType'] = combined_df['titleType'].astype('string') # Ensure titleType is string
    
    # Drop exact duplicates (keep first)
    before = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    dropped = before - len(combined_df)
    logging.info(f"Duplicate rows dropped: {dropped:,}")
    
    # Handle conflicts (multiple tconst for same rt_id)
    conflicts = []
    resolved_conflicts = []
    
    for rt_id in combined_df['rt_id'].unique():
        rt_rows = combined_df[combined_df['rt_id'] == rt_id]
        if len(rt_rows) > 1:
            # Check if there are multiple non-null tconst
            non_null_tconst = rt_rows[rt_rows['tconst'].notna()]
            if len(non_null_tconst) > 1:
                # Keep first, move others to conflicts
                first_row = non_null_tconst.iloc[0]
                resolved_conflicts.append(first_row)
                
                conflict_rows = non_null_tconst.iloc[1:]
                for _, conflict_row in conflict_rows.iterrows():
                    conflicts.append(conflict_row)
                
                # Remove all rows for this rt_id from main df
                combined_df = combined_df[combined_df['rt_id'] != rt_id]
                
                # Add back the first resolved row
                combined_df = pd.concat([combined_df, pd.DataFrame([first_row])], ignore_index=True)
    
    # Create conflicts dataframe if any exist
    if conflicts:
        conflicts_df = pd.DataFrame(conflicts)
        conflicts_df.to_parquet(conflicts_path, index=False)
        logging.info(f"Conflicts written to: {conflicts_path} ({len(conflicts):,} rows)")
    else:
        logging.info("No conflicts found")
    
    # Identify unresolved rows (no tconst)
    unresolved = combined_df[combined_df['tconst'].isna()]
    
    if len(unresolved) > 0:
        unresolved.to_parquet(unresolved_path, index=False)
        logging.info(f"Unresolved rows written to: {unresolved_path} ({len(unresolved):,} rows)")
    else:
        logging.info("No unresolved rows found")
    
    # Final output (exclude unresolved rows)
    final_df = combined_df[combined_df['tconst'].notna()].copy()
    
    # Ensure schema order
    schema_order = [
        'rt_id', 'tconst', 'tmdbId', 'movieId', 'title_norm', 'year',
        'title_source', 'link_method', 'match_score', 'source_ml', 
        'source_imdb', 'source_rt', 'canonical_id', 'titleType'
    ]
    
    final_df = final_df[schema_order]
    
    # Save main output
    final_df.to_parquet(output_path, index=False)
    logging.info(f"Main output saved to: {output_path} ({len(final_df):,} rows)")
    
    # QA counts and analysis
    logging.info("=== QA ANALYSIS ===")
    
    # RT ingestion stats
    logging.info(f"RT total rows ingested (after de-dupe): {after_dedup:,}")
    
    # Exact match hit rate
    total_rt = len(combined_df)
    matched_rt = len(final_df)
    hit_rate = (matched_rt / total_rt) * 100 if total_rt > 0 else 0
    logging.info(f"Exact-match hit rate: {matched_rt:,}/{total_rt:,} ({hit_rate:.1f}%)")
    
    # Counts by titleType among matches
    if len(final_df) > 0:
        title_type_counts = final_df['titleType'].value_counts()
        logging.info("Title type distribution among matches:")
        for title_type, count in title_type_counts.items():
            logging.info(f"  {title_type}: {count:,}")
    
    # Log sample rows
    logging.info("=== 5 SAMPLE MATCHED ROWS ===")
    if len(final_df) > 0:
        sample_rows = final_df.sample(n=min(5, len(final_df)), random_state=42)
        for i, (_, row) in enumerate(sample_rows.iterrows()):
            logging.info(f"Sample {i+1}:")
            logging.info(f"  rt_id: {row['rt_id']}")
            logging.info(f"  title_norm: {row['title_norm']}")
            logging.info(f"  year: {row['year']}")
            logging.info(f"  tconst: {row['tconst']}")
            logging.info("")
    
    # Final summary
    logging.info("=== SUB-PHASE 2.2 COMPLETE ===")
    logging.info(f"Final output: {len(final_df):,} rows")
    logging.info(f"Conflicts: {len(conflicts):,} rows")
    logging.info(f"Unresolved: {len(unresolved):,} rows")
    logging.info(f"Duplicates dropped: {dropped:,} rows")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
