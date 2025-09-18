#!/usr/bin/env python3
"""
Step 1b Phase 2 - Sub-phase 2.6: Master Table Build
Builds canonical movies master table from resolved_links with enriched ratings/metadata
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
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

def normalize_genres(genres_str):
    """Normalize IMDb genres into canonical list"""
    if pd.isna(genres_str) or genres_str == '\\N':
        return None
    
    # Split on comma, strip, lowercase
    genres = [g.strip().lower() for g in str(genres_str).split(',')]
    genres = [g for g in genres if g and g != '\\n']
    
    # Map common variants/synonyms
    genre_mapping = {
        'sci-fi': 'science fiction',
        'rom-com': 'romance',
        'romcom': 'romance',
        'sci fi': 'science fiction',
        'sci-fantasy': 'science fiction',
        'action-adventure': 'action',
        'drama-romance': 'drama',
        'comedy-drama': 'comedy',
        'thriller-drama': 'thriller',
        'horror-thriller': 'horror'
    }
    
    normalized = []
    for genre in genres:
        normalized.append(genre_mapping.get(genre, genre))
    
    # Deduplicate and return
    return list(set(normalized))

def extract_rt_score(row, score_columns, default=None):
    """Extract RT score from various possible column names"""
    for col in score_columns:
        if col in row.index and pd.notna(row[col]):
            try:
                score = float(row[col])
                # Clamp to 0-100 range
                return max(0, min(100, int(score)))
            except (ValueError, TypeError):
                continue
    return default

def main():
    """Main function for Sub-phase 2.6"""
    logging.info("=== STARTING STEP 1B PHASE 2 SUB-PHASE 2.6: MASTER TABLE BUILD ===")
    
    # Define paths
    resolved_links_path = "data/normalized/bridges/checkpoints/resolved_links.parquet"
    imdb_basics_path = "IMDB datasets/title.basics.tsv"
    imdb_ratings_path = "IMDB datasets/title.ratings.tsv"
    rt_movies_path = "Rotten Tomatoes/rotten_tomatoes_movies.csv"
    rt_top_movies_path = "Rotten Tomatoes/rotten_tomatoes_top_movies.csv"
    
    output_path = "data/normalized/movies_master.parquet"
    preview_path = "data/normalized/movies_master_preview.csv"
    
    # Check if output exists and remove for fresh run
    if os.path.exists(output_path):
        os.remove(output_path)
        logging.info(f"Removed existing output: {output_path}")
    
    # Load resolved links
    logging.info("Loading resolved links from 2.5...")
    resolved_df = pd.read_parquet(resolved_links_path)
    total_input = len(resolved_df)
    logging.info(f"Loaded {total_input:,} rows from resolved_links")
    
    # Load minimal IMDb basics
    logging.info("Loading IMDb basics...")
    imdb_cols = ["tconst", "primaryTitle", "originalTitle", "titleType", "startYear", "runtimeMinutes", "genres"]
    imdb_dtypes = {
        "tconst": "string", "primaryTitle": "string", "originalTitle": "string",
        "titleType": "string", "startYear": "string", "runtimeMinutes": "string", "genres": "string"
    }
    
    basics = pd.read_csv(
        imdb_basics_path, sep="\t", usecols=imdb_cols, dtype=imdb_dtypes,
        na_values="\\N", low_memory=False
    )
    
    # Coerce numeric fields
    basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce").astype("Int32")
    basics["runtimeMinutes"] = pd.to_numeric(basics["runtimeMinutes"], errors="coerce").astype("Int32")
    
    # Create lookup Series for fast joins
    basics_lookup = basics.set_index("tconst")
    logging.info(f"IMDb basics indexed: {len(basics_lookup):,} titles")
    
    # Load IMDb ratings
    logging.info("Loading IMDb ratings...")
    ratings_cols = ["tconst", "averageRating", "numVotes"]
    ratings_dtypes = {
        "tconst": "string", "averageRating": "string", "numVotes": "string"
    }
    
    ratings = pd.read_csv(
        imdb_ratings_path, sep="\t", usecols=ratings_cols, dtype=ratings_dtypes,
        na_values="\\N", low_memory=False
    )
    
    # Coerce numeric fields
    ratings["averageRating"] = pd.to_numeric(ratings["averageRating"], errors="coerce").astype("Float32")
    ratings["numVotes"] = pd.to_numeric(ratings["numVotes"], errors="coerce").astype("Int32")
    
    # Create lookup Series
    ratings_lookup = ratings.set_index("tconst")
    logging.info(f"IMDb ratings indexed: {len(ratings_lookup):,} titles")
    
    # Load RT datasets
    logging.info("Loading Rotten Tomatoes datasets...")
    rt_movies = pd.read_csv(rt_movies_path)
    rt_top = pd.read_csv(rt_top_movies_path)
    
    # Create RT lookup for scores
    rt_lookup = {}
    
    # Process RT Movies
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
        
        if title and year:
            # Extract scores
            tomatometer = extract_rt_score(row, 
                ['tomatometer_rating', 'tomatometer', 'critic_score', 'tomato_meter'], None)
            audience = extract_rt_score(row, 
                ['audience_rating', 'audience_score', 'audience'], None)
            
            if tomatometer is not None or audience is not None:
                rt_lookup[(title.lower().strip(), year)] = {
                    'tomatometer': tomatometer,
                    'audience': audience
                }
    
    # Process RT Top Movies
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
        
        if title and year:
            # Extract scores
            tomatometer = extract_rt_score(row, 
                ['tomatometer_rating', 'tomatometer', 'critic_score', 'tomato_meter'], None)
            audience = extract_rt_score(row, 
                ['audience_rating', 'audience_score', 'audience'], None)
            
            if tomatometer is not None or audience is not None:
                rt_lookup[(title.lower().strip(), year)] = {
                    'tomatometer': tomatometer,
                    'audience': audience
                }
    
    logging.info(f"RT lookup created with {len(rt_lookup):,} entries")
    
    # Process resolved links in chunks
    chunk_size = 50_000
    all_results = []
    
    logging.info(f"Processing {total_input:,} rows in chunks of {chunk_size:,}...")
    
    for start in range(0, total_input, chunk_size):
        end = min(start + chunk_size, total_input)
        chunk = resolved_df.iloc[start:end].copy()
        logging.info(f"Processing chunk {start//chunk_size + 1}/{(total_input-1)//chunk_size + 1} (rows {start:,}–{end:,})")
        
        chunk_results = []
        
        for _, row in chunk.iterrows():
            # Determine canonical_id (should already be set from 2.5)
            canonical_id = row['canonical_id']
            
            # Get IMDb data
            tconst = row.get('tconst')
            imdb_data = {}
            if tconst and tconst in basics_lookup.index:
                imdb_row = basics_lookup.loc[tconst]
                imdb_data = {
                    'primaryTitle': imdb_row.get('primaryTitle'),
                    'originalTitle': imdb_row.get('originalTitle'),
                    'titleType': imdb_row.get('titleType'),
                    'startYear': imdb_row.get('startYear'),
                    'runtimeMinutes': imdb_row.get('runtimeMinutes'),
                    'genres': imdb_row.get('genres')
                }
            
            # Get IMDb ratings
            imdb_rating = None
            imdb_votes = None
            if tconst and tconst in ratings_lookup.index:
                ratings_row = ratings_lookup.loc[tconst]
                imdb_rating = ratings_row.get('averageRating')
                imdb_votes = ratings_row.get('numVotes')
            
            # Get RT scores
            rt_tomatometer = None
            rt_audience = None
            if pd.notna(row.get('rt_id')) and pd.notna(row.get('title_norm')) and pd.notna(row.get('year')):
                rt_key = (row['title_norm'], row['year'])
                if rt_key in rt_lookup:
                    rt_scores = rt_lookup[rt_key]
                    rt_tomatometer = rt_scores['tomatometer']
                    rt_audience = rt_scores['audience']
            
            # Create display title (IMDb-first priority)
            display_title = None
            if pd.notna(imdb_data.get('primaryTitle')):
                display_title = imdb_data['primaryTitle']
            elif pd.notna(imdb_data.get('originalTitle')):
                display_title = imdb_data['originalTitle']
            elif pd.notna(row.get('title_norm')):
                display_title = row['title_norm'].title()
            else:
                display_title = "Unknown Title"
            
            # Normalize genres
            genres_norm = None
            genres_str = None
            if pd.notna(imdb_data.get('genres')):
                genres_norm = normalize_genres(imdb_data['genres'])
                if genres_norm:
                    genres_str = '|'.join(sorted(genres_norm))
            
            # Build result row
            result_row = {
                'canonical_id': canonical_id,
                'tconst': tconst,
                'tmdbId': row.get('tmdbId'),
                'movieId': row.get('movieId'),
                'rt_id': row.get('rt_id'),
                'title': display_title,
                'title_norm': row.get('title_norm', ''),
                'year': imdb_data.get('startYear') if pd.notna(imdb_data.get('startYear')) else row.get('year'),
                'titleType': imdb_data.get('titleType'),
                'runtimeMinutes': imdb_data.get('runtimeMinutes'),
                'genres_norm': genres_norm,
                'genres_str': genres_str,
                'imdb_rating': imdb_rating,
                'imdb_votes': imdb_votes,
                'rt_tomatometer': rt_tomatometer,
                'rt_audience': rt_audience,
                'link_method': row['link_method'],
                'match_score': row.get('match_score'),
                'source_ml': row['source_ml'],
                'source_imdb': row['source_imdb'],
                'source_rt': row['source_rt']
            }
            
            chunk_results.append(result_row)
        
        all_results.extend(chunk_results)
    
    # Convert to DataFrame
    logging.info("Combining all results...")
    master_df = pd.DataFrame(all_results)
    
    # Ensure unique canonical_id (should already be unique from 2.5)
    before_dedup = len(master_df)
    master_df = master_df.drop_duplicates(subset=['canonical_id'], keep='first')
    after_dedup = len(master_df)
    
    if before_dedup != after_dedup:
        logging.warning(f"Found {before_dedup - after_dedup} duplicate canonical_ids - kept first occurrence")
    
    # Enforce final schema and dtypes
    final_schema = [
        'canonical_id', 'tconst', 'tmdbId', 'movieId', 'rt_id', 'title', 'title_norm', 'year',
        'titleType', 'runtimeMinutes', 'genres_norm', 'genres_str', 'imdb_rating', 'imdb_votes',
        'rt_tomatometer', 'rt_audience', 'link_method', 'match_score', 'source_ml', 'source_imdb', 'source_rt'
    ]
    
    master_df = master_df[final_schema].copy()
    
    # Convert to proper dtypes
    master_df['canonical_id'] = master_df['canonical_id'].astype('string')
    master_df['tconst'] = master_df['tconst'].astype('string')
    master_df['tmdbId'] = master_df['tmdbId'].astype('Int64')
    master_df['movieId'] = master_df['movieId'].astype('Int64')
    master_df['rt_id'] = master_df['rt_id'].astype('string')
    master_df['title'] = master_df['title'].astype('string')
    master_df['title_norm'] = master_df['title_norm'].astype('string')
    master_df['year'] = master_df['year'].astype('Int32')
    master_df['titleType'] = master_df['titleType'].astype('string')
    master_df['runtimeMinutes'] = master_df['runtimeMinutes'].astype('Int32')
    master_df['genres_norm'] = master_df['genres_norm'].astype('object')  # list type
    master_df['genres_str'] = master_df['genres_str'].astype('string')
    master_df['imdb_rating'] = master_df['imdb_rating'].astype('Float32')
    master_df['imdb_votes'] = master_df['imdb_votes'].astype('Int32')
    master_df['rt_tomatometer'] = master_df['rt_tomatometer'].astype('Int16')
    master_df['rt_audience'] = master_df['rt_audience'].astype('Int16')
    master_df['link_method'] = master_df['link_method'].astype('string')
    master_df['match_score'] = master_df['match_score'].astype('Float32')
    master_df['source_ml'] = master_df['source_ml'].astype('boolean')
    master_df['source_imdb'] = master_df['source_imdb'].astype('boolean')
    master_df['source_rt'] = master_df['source_rt'].astype('boolean')
    
    # Save main output
    master_df.to_parquet(output_path, index=False)
    logging.info(f"Main output saved to: {output_path} ({len(master_df):,} rows)")
    
    # Save preview CSV
    preview_df = master_df.head(1000)
    preview_df.to_csv(preview_path, index=False)
    logging.info(f"Preview CSV saved to: {preview_path} ({len(preview_df):,} rows)")
    
    # QA analysis
    logging.info("=== QA ANALYSIS ===")
    
    logging.info(f"Total input rows from resolved_links: {total_input:,}")
    logging.info(f"Final unique count in master: {len(master_df):,}")
    
    # Coverage stats
    imdb_rating_coverage = master_df['imdb_rating'].notna().sum() / len(master_df) * 100
    imdb_votes_coverage = (master_df['imdb_votes'] > 0).sum() / len(master_df) * 100
    rt_tomatometer_coverage = master_df['rt_tomatometer'].notna().sum() / len(master_df) * 100
    rt_audience_coverage = master_df['rt_audience'].notna().sum() / len(master_df) * 100
    
    logging.info(f"Coverage stats:")
    logging.info(f"  % with imdb_rating: {imdb_rating_coverage:.1f}%")
    logging.info(f"  % with imdb_votes>0: {imdb_votes_coverage:.1f}%")
    logging.info(f"  % with rt_tomatometer: {rt_tomatometer_coverage:.1f}%")
    logging.info(f"  % with rt_audience: {rt_audience_coverage:.1f}%")
    
    # Top genres
    all_genres = []
    for genres_list in master_df['genres_norm'].dropna():
        all_genres.extend(genres_list)
    
    if all_genres:
        genre_counts = pd.Series(all_genres).value_counts()
        logging.info(f"Top 10 genres by frequency:")
        for genre, count in genre_counts.head(10).items():
            logging.info(f"  {genre}: {count:,}")
    
    # Basic sanity checks
    year_summary = master_df['year'].describe()
    logging.info(f"Year distribution (5-number summary):")
    logging.info(f"  min: {year_summary['min']:.0f}")
    logging.info(f"  25%: {year_summary['25%']:.0f}")
    logging.info(f"  50%: {year_summary['50%']:.0f}")
    logging.info(f"  75%: {year_summary['75%']:.0f}")
    logging.info(f"  max: {year_summary['max']:.0f}")
    
    rating_summary = master_df['imdb_rating'].describe()
    logging.info(f"IMDb rating range: {rating_summary['min']:.1f} - {rating_summary['max']:.1f}")
    
    rt_summary = master_df['rt_tomatometer'].describe()
    logging.info(f"RT Tomatometer range: {rt_summary['min']:.0f} - {rt_summary['max']:.0f}")
    
    # Sample rows
    logging.info("=== 10 RANDOM SAMPLE ROWS ===")
    sample_rows = master_df.sample(n=10, random_state=42)
    for i, (_, row) in enumerate(sample_rows.iterrows()):
        logging.info(f"Sample {i+1}:")
        logging.info(f"  canonical_id: {row['canonical_id']}")
        logging.info(f"  title: {row['title']}")
        logging.info(f"  year: {row['year']}")
        logging.info(f"  imdb_rating: {row['imdb_rating']:.1f}" if pd.notna(row['imdb_rating']) else "  imdb_rating: None")
        logging.info(f"  rt_tomatometer: {row['rt_tomatometer']}" if pd.notna(row['rt_tomatometer']) else "  rt_tomatometer: None")
        logging.info(f"  genres_str: {row['genres_str']}" if pd.notna(row['genres_str']) else "  genres_str: None")
        logging.info("")
    
    # Final summary
    logging.info("=== SUB-PHASE 2.6 COMPLETE ===")
    logging.info(f"Movies master table: {len(master_df):,} unique movies")
    logging.info(f"Schema enforced: {len(final_schema)} columns in exact order")
    logging.info(f"All dtypes properly cast and validated")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
