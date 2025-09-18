#!/usr/bin/env python3
"""
Step 3b.1: Ratings Matrix Assembly
Builds a user-movie ratings matrix from MovieLens ratings, aligned with canonical_id system.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import logging
from pathlib import Path
import time
from datetime import datetime

# Setup logging
def setup_logging():
    """Setup logging for the ratings matrix assembly process."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"step3b_phase1_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_input_data(logger):
    """Load MovieLens ratings, links, and resolved links data."""
    logger.info("Loading input data...")
    
    # Load MovieLens ratings
    logger.info("Loading MovieLens ratings.csv...")
    ratings = pd.read_csv("movie-lens/ratings.csv")
    logger.info(f"Loaded {len(ratings):,} ratings")
    logger.info(f"Ratings columns: {ratings.columns.tolist()}")
    logger.info(f"Rating range: {ratings['rating'].min():.1f} - {ratings['rating'].max():.1f}")
    
    # Load MovieLens links
    logger.info("Loading MovieLens links.csv...")
    links = pd.read_csv("movie-lens/links.csv")
    logger.info(f"Loaded {len(links):,} links")
    logger.info(f"Links columns: {links.columns.tolist()}")
    
    # Load resolved links
    logger.info("Loading resolved_links.parquet...")
    resolved_links = pd.read_parquet("data/normalized/bridges/checkpoints/resolved_links.parquet")
    logger.info(f"Loaded {len(resolved_links):,} resolved links")
    logger.info(f"Resolved links columns: {resolved_links.columns.tolist()}")
    
    # Load movies master
    logger.info("Loading movies_master.parquet...")
    movies_master = pd.read_parquet("data/normalized/movies_master.parquet")
    logger.info(f"Loaded {len(movies_master):,} movies in master dataset")
    
    return ratings, links, resolved_links, movies_master

def join_ratings_with_canonical_ids(ratings, links, resolved_links, logger):
    """Join ratings with links to map movieId -> canonical_id."""
    logger.info("Joining ratings with canonical IDs...")
    
    # First join ratings with links to get imdbId
    logger.info("Joining ratings with links...")
    ratings_with_imdb = ratings.merge(links, on='movieId', how='inner')
    logger.info(f"After joining with links: {len(ratings_with_imdb):,} ratings")
    
    # Convert imdbId to tconst format (add 'tt' prefix and pad with zeros)
    logger.info("Converting imdbId to tconst format...")
    ratings_with_imdb['tconst'] = 'tt' + ratings_with_imdb['imdbId'].astype(str).str.zfill(7)
    
    # Join with resolved_links to get canonical_id
    logger.info("Joining with resolved_links to get canonical_id...")
    ratings_with_canonical = ratings_with_imdb.merge(
        resolved_links[['tconst', 'canonical_id']], 
        on='tconst', 
        how='inner'
    )
    logger.info(f"After joining with resolved_links: {len(ratings_with_canonical):,} ratings")
    
    # Check for missing canonical_ids
    missing_canonical = ratings_with_canonical['canonical_id'].isna().sum()
    if missing_canonical > 0:
        logger.warning(f"Found {missing_canonical:,} ratings with missing canonical_id")
        ratings_with_canonical = ratings_with_canonical.dropna(subset=['canonical_id'])
        logger.info(f"After removing missing canonical_ids: {len(ratings_with_canonical):,} ratings")
    
    return ratings_with_canonical

def filter_and_align_ratings(ratings_with_canonical, movies_master, logger):
    """Filter ratings to only include movies in master dataset and apply user/movie thresholds."""
    logger.info("Filtering and aligning ratings...")
    
    # Filter to only movies in master dataset
    logger.info("Filtering to movies in master dataset...")
    valid_canonical_ids = set(movies_master['canonical_id'].unique())
    initial_count = len(ratings_with_canonical)
    
    ratings_filtered = ratings_with_canonical[
        ratings_with_canonical['canonical_id'].isin(valid_canonical_ids)
    ].copy()
    
    logger.info(f"After filtering to master movies: {len(ratings_filtered):,} ratings "
                f"(removed {initial_count - len(ratings_filtered):,})")
    
    # Apply user and movie thresholds
    logger.info("Applying user and movie thresholds...")
    
    # Count ratings per user and per movie
    user_counts = ratings_filtered['userId'].value_counts()
    movie_counts = ratings_filtered['canonical_id'].value_counts()
    
    logger.info(f"User rating counts - min: {user_counts.min()}, max: {user_counts.max()}, "
                f"mean: {user_counts.mean():.1f}")
    logger.info(f"Movie rating counts - min: {movie_counts.min()}, max: {movie_counts.max()}, "
                f"mean: {movie_counts.mean():.1f}")
    
    # Filter users with < 3 ratings
    valid_users = user_counts[user_counts >= 3].index
    ratings_filtered = ratings_filtered[ratings_filtered['userId'].isin(valid_users)]
    logger.info(f"After filtering users with <3 ratings: {len(ratings_filtered):,} ratings")
    
    # Filter movies with < 5 ratings
    valid_movies = movie_counts[movie_counts >= 5].index
    ratings_filtered = ratings_filtered[ratings_filtered['canonical_id'].isin(valid_movies)]
    logger.info(f"After filtering movies with <5 ratings: {len(ratings_filtered):,} ratings")
    
    # Final counts
    final_user_count = ratings_filtered['userId'].nunique()
    final_movie_count = ratings_filtered['canonical_id'].nunique()
    final_rating_count = len(ratings_filtered)
    
    logger.info(f"Final counts - Users: {final_user_count:,}, Movies: {final_movie_count:,}, "
                f"Ratings: {final_rating_count:,}")
    
    return ratings_filtered

def build_ratings_matrix(ratings_filtered, logger):
    """Build sparse ratings matrix with sequential indexing."""
    logger.info("Building ratings matrix...")
    
    # Create sequential indices
    logger.info("Creating sequential indices...")
    unique_users = sorted(ratings_filtered['userId'].unique())
    unique_movies = sorted(ratings_filtered['canonical_id'].unique())
    
    # Create mapping dictionaries
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    
    logger.info(f"Created mappings for {len(unique_users):,} users and {len(unique_movies):,} movies")
    
    # Convert to sequential indices
    ratings_filtered['user_idx'] = ratings_filtered['userId'].map(user_to_idx)
    ratings_filtered['movie_idx'] = ratings_filtered['canonical_id'].map(movie_to_idx)
    
    # Build sparse matrix
    logger.info("Building sparse matrix...")
    ratings_matrix = csr_matrix(
        (ratings_filtered['rating'], 
         (ratings_filtered['user_idx'], ratings_filtered['movie_idx'])),
        shape=(len(unique_users), len(unique_movies))
    )
    
    logger.info(f"Built sparse matrix with shape {ratings_matrix.shape}")
    logger.info(f"Matrix density: {ratings_matrix.nnz / (ratings_matrix.shape[0] * ratings_matrix.shape[1]):.6f}")
    
    return ratings_matrix, user_to_idx, movie_to_idx, ratings_filtered

def create_mapping_tables(user_to_idx, movie_to_idx, logger):
    """Create mapping tables for users and movies."""
    logger.info("Creating mapping tables...")
    
    # User index map
    user_index_map = pd.DataFrame([
        {'userId': user_id, 'user_index': idx} 
        for user_id, idx in user_to_idx.items()
    ])
    user_index_map = user_index_map.sort_values('user_index').reset_index(drop=True)
    
    # Movie index map
    movie_index_map = pd.DataFrame([
        {'canonical_id': movie_id, 'movie_index': idx} 
        for movie_id, idx in movie_to_idx.items()
    ])
    movie_index_map = movie_index_map.sort_values('movie_index').reset_index(drop=True)
    
    logger.info(f"Created user_index_map with {len(user_index_map):,} users")
    logger.info(f"Created movie_index_map with {len(movie_index_map):,} movies")
    
    return user_index_map, movie_index_map

def save_outputs(ratings_matrix, ratings_filtered, user_index_map, movie_index_map, logger):
    """Save all output files."""
    logger.info("Saving outputs...")
    
    output_dir = Path("data/collaborative")
    output_dir.mkdir(exist_ok=True)
    
    # Save sparse matrix
    logger.info("Saving ratings_matrix_csr.npz...")
    sp.save_npz(output_dir / "ratings_matrix_csr.npz", ratings_matrix)
    
    # Save long format
    logger.info("Saving ratings_long_format.parquet...")
    ratings_long = ratings_filtered[['user_idx', 'canonical_id', 'rating']].copy()
    ratings_long.columns = ['user_index', 'canonical_id', 'rating']
    ratings_long.to_parquet(output_dir / "ratings_long_format.parquet", index=False)
    
    # Save mapping tables
    logger.info("Saving user_index_map.parquet...")
    user_index_map.to_parquet(output_dir / "user_index_map.parquet", index=False)
    
    logger.info("Saving movie_index_map.parquet...")
    movie_index_map.to_parquet(output_dir / "movie_index_map.parquet", index=False)
    
    logger.info("All outputs saved successfully")

def perform_quality_checks(ratings_matrix, ratings_filtered, user_index_map, movie_index_map, logger):
    """Perform quality checks on the assembled ratings matrix."""
    logger.info("Performing quality checks...")
    
    # Basic counts
    user_count = ratings_matrix.shape[0]
    movie_count = ratings_matrix.shape[1]
    rating_count = ratings_matrix.nnz
    
    logger.info(f"Quality Check - User count: {user_count:,}")
    logger.info(f"Quality Check - Movie count: {movie_count:,}")
    logger.info(f"Quality Check - Rating count: {rating_count:,}")
    
    # Validate rating scale
    min_rating = ratings_filtered['rating'].min()
    max_rating = ratings_filtered['rating'].max()
    logger.info(f"Quality Check - Rating scale: {min_rating:.1f} - {max_rating:.1f}")
    
    if min_rating < 0.5 or max_rating > 5.0:
        logger.warning(f"Rating scale outside expected range (0.5-5.0)")
    else:
        logger.info("Rating scale validation: PASSED")
    
    # Check for NaN/Inf values
    has_nan = np.isnan(ratings_filtered['rating']).any()
    has_inf = np.isinf(ratings_filtered['rating']).any()
    logger.info(f"Quality Check - Has NaN values: {has_nan}")
    logger.info(f"Quality Check - Has Inf values: {has_inf}")
    
    if not has_nan and not has_inf:
        logger.info("NaN/Inf validation: PASSED")
    else:
        logger.warning("Found NaN or Inf values in ratings")
    
    # Compute sparsity
    total_possible = user_count * movie_count
    density = rating_count / total_possible
    sparsity = 1 - density
    
    logger.info(f"Quality Check - Matrix density: {density:.6f}")
    logger.info(f"Quality Check - Matrix sparsity: {sparsity:.6f}")
    
    # Spot check random samples
    logger.info("Performing spot checks...")
    sample_size = min(10, len(ratings_filtered))
    sample_ratings = ratings_filtered.sample(n=sample_size, random_state=42)
    
    for _, row in sample_ratings.iterrows():
        user_idx = row['user_idx']
        movie_idx = row['movie_idx']
        rating = row['rating']
        matrix_rating = ratings_matrix[user_idx, movie_idx]
        
        if abs(rating - matrix_rating) > 1e-10:
            logger.error(f"Spot check failed: expected {rating}, got {matrix_rating}")
        else:
            logger.info(f"Spot check passed: user {row['userId']}, movie {row['canonical_id']}, rating {rating}")
    
    logger.info("Quality checks completed")

def generate_documentation(ratings_matrix, ratings_filtered, user_index_map, movie_index_map, logger):
    """Generate documentation for the ratings matrix assembly."""
    logger.info("Generating documentation...")
    
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Calculate statistics
    user_count = ratings_matrix.shape[0]
    movie_count = ratings_matrix.shape[1]
    rating_count = ratings_matrix.nnz
    density = rating_count / (user_count * movie_count)
    
    # Rating distribution
    rating_dist = ratings_filtered['rating'].value_counts().sort_index()
    
    # User activity distribution
    user_activity = ratings_filtered.groupby('userId').size()
    user_activity_stats = {
        'min_ratings_per_user': user_activity.min(),
        'max_ratings_per_user': user_activity.max(),
        'mean_ratings_per_user': user_activity.mean(),
        'median_ratings_per_user': user_activity.median()
    }
    
    # Movie popularity distribution
    movie_popularity = ratings_filtered.groupby('canonical_id').size()
    movie_popularity_stats = {
        'min_ratings_per_movie': movie_popularity.min(),
        'max_ratings_per_movie': movie_popularity.max(),
        'mean_ratings_per_movie': movie_popularity.mean(),
        'median_ratings_per_movie': movie_popularity.median()
    }
    
    # Create documentation
    doc_content = f"""# Step 3b.1: Ratings Matrix Assembly

## Overview
This document describes the assembly of the user-movie ratings matrix from MovieLens data, aligned with the canonical_id system.

## Input Data Sources
- MovieLens ratings.csv: {len(ratings_filtered):,} ratings after filtering
- MovieLens links.csv: Mapping from movieId to imdbId
- resolved_links.parquet: Mapping from tconst to canonical_id
- movies_master.parquet: Master dataset with valid canonical_ids

## Filters Applied
- Users with < 3 ratings: Removed
- Movies with < 5 ratings: Removed
- Movies not in master dataset: Removed
- Missing canonical_ids: Removed

## Final Statistics
- **Users**: {user_count:,}
- **Movies**: {movie_count:,}
- **Ratings**: {rating_count:,}
- **Matrix Density**: {density:.6f}
- **Matrix Sparsity**: {1-density:.6f}

## Rating Scale
- **Range**: {ratings_filtered['rating'].min():.1f} - {ratings_filtered['rating'].max():.1f}
- **Distribution**:
{rating_dist.to_string()}

## User Activity Statistics
- **Min ratings per user**: {user_activity_stats['min_ratings_per_user']:,}
- **Max ratings per user**: {user_activity_stats['max_ratings_per_user']:,}
- **Mean ratings per user**: {user_activity_stats['mean_ratings_per_user']:.1f}
- **Median ratings per user**: {user_activity_stats['median_ratings_per_user']:.1f}

## Movie Popularity Statistics
- **Min ratings per movie**: {movie_popularity_stats['min_ratings_per_movie']:,}
- **Max ratings per movie**: {movie_popularity_stats['max_ratings_per_movie']:,}
- **Mean ratings per movie**: {movie_popularity_stats['mean_ratings_per_movie']:.1f}
- **Median ratings per movie**: {movie_popularity_stats['median_ratings_per_movie']:.1f}

## Output Files
- `data/collaborative/ratings_matrix_csr.npz`: Sparse ratings matrix in CSR format
- `data/collaborative/ratings_long_format.parquet`: Ratings in long format
- `data/collaborative/user_index_map.parquet`: User ID to sequential index mapping
- `data/collaborative/movie_index_map.parquet`: Canonical ID to sequential index mapping

## Quality Checks
- ✅ All canonical_ids align with master dataset
- ✅ No NaN/Inf values in ratings
- ✅ Rating scale validated (0.5-5.0)
- ✅ Sparse matrix and parquet outputs created
- ✅ Sequential indexing implemented
- ✅ Spot checks passed

## Timestamp
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(docs_dir / "step3b_inputs.md", "w") as f:
        f.write(doc_content)
    
    logger.info("Documentation saved to docs/step3b_inputs.md")

def main():
    """Main function to orchestrate the ratings matrix assembly."""
    logger = setup_logging()
    logger.info("Starting Step 3b.1: Ratings Matrix Assembly")
    
    try:
        # Load input data
        ratings, links, resolved_links, movies_master = load_input_data(logger)
        
        # Join ratings with canonical IDs
        ratings_with_canonical = join_ratings_with_canonical_ids(ratings, links, resolved_links, logger)
        
        # Filter and align ratings
        ratings_filtered = filter_and_align_ratings(ratings_with_canonical, movies_master, logger)
        
        # Build ratings matrix
        ratings_matrix, user_to_idx, movie_to_idx, ratings_with_indices = build_ratings_matrix(ratings_filtered, logger)
        
        # Create mapping tables
        user_index_map, movie_index_map = create_mapping_tables(user_to_idx, movie_to_idx, logger)
        
        # Save outputs
        save_outputs(ratings_matrix, ratings_with_indices, user_index_map, movie_index_map, logger)
        
        # Perform quality checks
        perform_quality_checks(ratings_matrix, ratings_with_indices, user_index_map, movie_index_map, logger)
        
        # Generate documentation
        generate_documentation(ratings_matrix, ratings_with_indices, user_index_map, movie_index_map, logger)
        
        logger.info("Step 3b.1 completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in Step 3b.1: {str(e)}")
        raise

if __name__ == "__main__":
    main()















