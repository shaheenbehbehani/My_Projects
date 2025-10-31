#!/usr/bin/env python3
"""
Build compact runtime artifacts from MovieLens data for Streamlit app.

Creates minimal artifacts (<30MB total):
- id_map.parquet: movieId, title, year, genres, normalized title, external IDs
- tfidf_title.pkl: TF-IDF vectorizer for title+genres search
- tfidf_vocab.json: Vocabulary mapping
- Optional: small SVD factors for collaborative filtering
"""

import sys
import os
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib

# Add app directory to path
SCRIPT_DIR = Path(__file__).parent
APP_DIR = SCRIPT_DIR.parent
DATA_DIR = APP_DIR / "data"

# Find repo root (go up from app/ to repo root)
# Structure: repo_root/movie-lens/
#           repo_root/Movie Recommendation Optimizer/app/
current = Path(__file__).resolve()
while current.parent != current:
    if (current / "movie-lens").exists():
        REPO_ROOT = current
        break
    current = current.parent
else:
    # Fallback: assume we're in Netflix/ directory
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MOVIELENS_DIR = REPO_ROOT / "movie-lens"
print(f"Using MovieLens directory: {MOVIELENS_DIR}")

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

def normalize_title(title):
    """Normalize movie title: lowercase, remove year, strip"""
    # Remove year pattern like (1995) or (2001)
    title = re.sub(r'\s*\(\d{4}\)\s*', '', title)
    # Normalize whitespace and lowercase
    title = re.sub(r'\s+', ' ', title.strip().lower())
    return title

def extract_year(title):
    """Extract year from title like 'Movie Name (1995)'"""
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    return None

def build_id_map():
    """Build id_map.parquet from MovieLens data."""
    print("Loading MovieLens movies and links...")
    
    # Load movies
    movies_path = MOVIELENS_DIR / "movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError(f"Movies file not found: {movies_path}")
    
    movies = pd.read_csv(movies_path)
    print(f"  Loaded {len(movies)} movies")
    
    # Load links if available
    links_path = MOVIELENS_DIR / "links.csv"
    links = None
    if links_path.exists():
        links = pd.read_csv(links_path)
        print(f"  Loaded {len(links)} links")
    
    # Process movies
    id_map = pd.DataFrame({
        'movieId': movies['movieId'],
        'title': movies['title'],
        'title_norm': movies['title'].apply(normalize_title),
        'year': movies['title'].apply(extract_year),
        'genres': movies['genres']
    })
    
    # Add external IDs if links exist
    if links is not None:
        id_map = id_map.merge(
            links[['movieId', 'imdbId', 'tmdbId']],
            on='movieId',
            how='left'
        )
    else:
        id_map['imdbId'] = None
        id_map['tmdbId'] = None
    
    # Drop rows with missing titles
    id_map = id_map.dropna(subset=['title'])
    
    # Save
    output_path = DATA_DIR / "id_map.parquet"
    id_map.to_parquet(output_path, index=False, compression='snappy')
    print(f"✓ Saved {len(id_map)} movies to {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return id_map

def build_tfidf(id_map):
    """Build TF-IDF vectorizer for title+genres search."""
    print("\nBuilding TF-IDF vectorizer...")
    
    # Combine title and genres for search
    # Replace | with space in genres, combine with title
    search_text = id_map.apply(
        lambda row: f"{row['title_norm']} {row['genres'].replace('|', ' ').lower() if pd.notna(row['genres']) else ''}",
        axis=1
    ).tolist()
    
    # Build TF-IDF with compact settings
    vectorizer = TfidfVectorizer(
        min_df=3,           # Ignore terms in <3 documents
        max_features=25000, # Limit vocabulary size
        ngram_range=(1, 2), # Unigrams and bigrams
        lowercase=True,
        strip_accents='unicode',
        token_pattern=r'\b\w+\b'  # Word tokens
    )
    
    print("  Fitting TF-IDF on titles+genres...")
    X = vectorizer.fit_transform(search_text)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Matrix shape: {X.shape}")
    print(f"  Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.4f}")
    
    # Save vectorizer
    vectorizer_path = DATA_DIR / "tfidf_title.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"✓ Saved vectorizer to {vectorizer_path}")
    print(f"  Size: {vectorizer_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save vocabulary as JSON (more readable, smaller than full vectorizer)
    vocab_path = DATA_DIR / "tfidf_vocab.json"
    # Convert numpy int64 to Python int for JSON serialization
    vocab_dict = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    with open(vocab_path, 'w') as f:
        json.dump(vocab_dict, f, indent=2)
    print(f"✓ Saved vocabulary to {vocab_path}")
    print(f"  Size: {vocab_path.stat().st_size / 1024:.2f} KB")
    
    return vectorizer, X

def build_svd_factors(id_map, max_users=30000, max_movies=10000, k=64):
    """Build lightweight SVD factors from sampled ratings."""
    print("\nBuilding lightweight SVD factors...")
    
    ratings_path = MOVIELENS_DIR / "ratings.csv"
    if not ratings_path.exists():
        print("  ⚠ Ratings file not found, skipping SVD")
        return None, None
    
    print(f"  Loading ratings from {ratings_path}...")
    print("  (This may take a moment for large files...)")
    
    # Sample ratings efficiently
    ratings = pd.read_csv(ratings_path, nrows=1000000)  # Limit to first 1M rows for speed
    print(f"  Loaded {len(ratings):,} ratings")
    
    # Filter to movies in our id_map
    movie_ids_set = set(id_map['movieId'].unique())
    ratings = ratings[ratings['movieId'].isin(movie_ids_set)]
    print(f"  {len(ratings):,} ratings match our movies")
    
    # Sample users (top users by rating count)
    user_counts = ratings['userId'].value_counts()
    top_users = user_counts.head(max_users).index
    ratings = ratings[ratings['userId'].isin(top_users)]
    
    # Sample movies (top movies by rating count)
    movie_counts = ratings['movieId'].value_counts()
    top_movies = movie_counts.head(max_movies).index
    ratings = ratings[ratings['movieId'].isin(top_movies)]
    
    print(f"  After sampling: {len(ratings):,} ratings from {ratings['userId'].nunique():,} users × {ratings['movieId'].nunique():,} movies")
    
    if len(ratings) < 1000:
        print("  ⚠ Too few ratings after filtering, skipping SVD")
        return None, None
    
    # Create pivot matrix
    print("  Creating user-item matrix...")
    pivot = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )
    print(f"  Matrix shape: {pivot.shape}")
    
    # Compute truncated SVD
    print(f"  Computing TruncatedSVD with k={k}...")
    svd = TruncatedSVD(n_components=k, random_state=42)
    svd_components = svd.fit_transform(pivot.T)  # Transpose: items × users -> items × k
    
    # Save movie factors
    factors_path = DATA_DIR / "movie_factors.npy"
    np.save(factors_path, svd_components)
    print(f"✓ Saved movie factors to {factors_path}")
    print(f"  Shape: {svd_components.shape}")
    print(f"  Size: {factors_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save movie index mapping
    movie_idx_map = pd.DataFrame({
        'movieId': pivot.columns.values,
        'factor_index': range(len(pivot.columns))
    })
    idx_map_path = DATA_DIR / "movie_factor_index.parquet"
    movie_idx_map.to_parquet(idx_map_path, index=False)
    print(f"✓ Saved movie index map to {idx_map_path}")
    print(f"  Size: {idx_map_path.stat().st_size / 1024:.2f} KB")
    
    return svd_components, movie_idx_map

def main():
    """Build all runtime artifacts."""
    print("=" * 60)
    print("Building Runtime Artifacts for Streamlit App")
    print("=" * 60)
    
    total_size_before = sum(f.stat().st_size for f in DATA_DIR.glob("*") if f.is_file())
    
    # Build artifacts
    id_map = build_id_map()
    vectorizer, tfidf_matrix = build_tfidf(id_map)
    
    # Try to build SVD (optional)
    try:
        svd_factors, svd_idx_map = build_svd_factors(id_map)
    except Exception as e:
        print(f"\n⚠ SVD build failed: {e}")
        print("  Continuing without SVD factors...")
        svd_factors = None
    
    # Calculate total size
    total_size_after = sum(f.stat().st_size for f in DATA_DIR.glob("*") if f.is_file())
    artifact_size = total_size_after - total_size_before
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print("\nCreated files:")
    for f in sorted(DATA_DIR.glob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            size_kb = f.stat().st_size / 1024
            if size_mb >= 1:
                print(f"  {f.name}: {size_mb:.2f} MB")
            else:
                print(f"  {f.name}: {size_kb:.2f} KB")
    
    print(f"\nTotal artifact size: {artifact_size / 1024 / 1024:.2f} MB")
    
    if artifact_size > 30 * 1024 * 1024:
        print("\n⚠ WARNING: Total size exceeds 30MB target!")
        print("  Consider reducing max_features or k for SVD")
    else:
        print("✓ Total size under 30MB target")
    
    print("\n✓ Build complete!")

if __name__ == "__main__":
    main()

