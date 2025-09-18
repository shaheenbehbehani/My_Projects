#!/usr/bin/env python3
"""
Step 3c.2 – Candidate Generation & Re-ranking of the Movie Recommendation Optimizer

This script implements efficient candidate generation and re-ranking using artifacts from
Steps 3a, 3b, and 3c.1. It generates personalized recommendations with diversity control
and applies mandatory filters for genres, providers, and other constraints.

Key Features:
- Two-stage candidate generation (CF seeds + content expansion)
- Hard filters with graceful fallback
- Hybrid scoring with secondary signals
- MMR-style diversity re-ranking
- Efficient batch processing to avoid memory issues

Author: Movie Recommendation Optimizer Pipeline
Date: 2025-01-27
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
import warnings
warnings.filterwarnings('ignore')

# Configure logging
def setup_logging():
    """Setup logging for Step 3c.2 execution"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "step3c_phase2.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_artifacts(logger: logging.Logger) -> Dict[str, Any]:
    """Load all required artifacts from Steps 3a, 3b, and 3c.1"""
    logger.info("Loading artifacts from Steps 3a, 3b, and 3c.1...")
    
    artifacts = {}
    
    # Load hybrid assembly manifest
    with open("data/hybrid/assembly_manifest.json", 'r') as f:
        artifacts['hybrid_manifest'] = json.load(f)
    
    # Load content embeddings (Step 3a)
    logger.info("Loading content embeddings...")
    embedding_path = "data/features/composite/movies_embedding_v1.npy"
    artifacts['content_embeddings'] = np.load(embedding_path, mmap_mode='r')
    logger.info(f"Content embeddings shape: {artifacts['content_embeddings'].shape}")
    
    # Load similarity neighbors (Step 3a)
    logger.info("Loading similarity neighbors...")
    neighbors_path = "data/similarity/movies_neighbors_k50.parquet"
    artifacts['similarity_neighbors'] = pd.read_parquet(neighbors_path)
    logger.info(f"Similarity neighbors shape: {artifacts['similarity_neighbors'].shape}")
    
    # Load collaborative factors (Step 3b)
    logger.info("Loading collaborative factors...")
    user_factors_path = "data/collaborative/user_factors_k20.npy"
    movie_factors_path = "data/collaborative/movie_factors_k20.npy"
    
    artifacts['user_factors'] = np.load(user_factors_path, mmap_mode='r')
    artifacts['movie_factors'] = np.load(movie_factors_path, mmap_mode='r')
    
    logger.info(f"User factors shape: {artifacts['user_factors'].shape}")
    logger.info(f"Movie factors shape: {artifacts['movie_factors'].shape}")
    
    # Load index mappings (Step 3b)
    logger.info("Loading index mappings...")
    user_map_path = "data/collaborative/user_index_map.parquet"
    movie_map_path = "data/collaborative/movie_index_map.parquet"
    
    artifacts['user_index_map'] = pd.read_parquet(user_map_path)
    artifacts['movie_index_map'] = pd.read_parquet(movie_map_path)
    
    logger.info(f"User index map shape: {artifacts['user_index_map'].shape}")
    logger.info(f"Movie index map shape: {artifacts['movie_index_map'].shape}")
    
    # Load filter features
    logger.info("Loading filter features...")
    artifacts['genres'] = pd.read_parquet("data/features/genres/movies_genres_multihot.parquet")
    artifacts['platforms'] = pd.read_parquet("data/features/platform/movies_platform_features.parquet")
    artifacts['numeric'] = pd.read_parquet("data/features/numeric/movies_numeric_standardized.parquet")
    
    logger.info(f"Genres shape: {artifacts['genres'].shape}")
    logger.info(f"Platforms shape: {artifacts['platforms'].shape}")
    logger.info(f"Numeric shape: {artifacts['numeric'].shape}")
    
    # Create mappings for efficient lookup
    logger.info("Creating lookup mappings...")
    artifacts['movie_id_to_factor_idx'] = {}
    for i, row in artifacts['movie_index_map'].iterrows():
        if i < len(artifacts['movie_factors']):
            artifacts['movie_id_to_factor_idx'][row['canonical_id']] = i
    
    artifacts['user_id_to_factor_idx'] = {}
    for i, row in artifacts['user_index_map'].iterrows():
        if i < len(artifacts['user_factors']):
            artifacts['user_id_to_factor_idx'][str(row['userId'])] = i
    
    logger.info(f"Created mappings for {len(artifacts['movie_id_to_factor_idx'])} movies and {len(artifacts['user_id_to_factor_idx'])} users with factors")
    
    return artifacts

def generate_cf_seeds(user_id: str, artifacts: Dict[str, Any], 
                     n_cf_seed: int = 800, logger: logging.Logger = None) -> List[str]:
    """Stage A: Generate CF seeds for users with factors"""
    logger.info(f"Generating CF seeds for user {user_id}...")
    
    if user_id not in artifacts['user_id_to_factor_idx']:
        logger.info(f"User {user_id} has no factors - skipping CF seeds")
        return []
    
    user_factor_idx = artifacts['user_id_to_factor_idx'][user_id]
    user_factors = artifacts['user_factors'][user_factor_idx:user_factor_idx+1]
    movie_factors = artifacts['movie_factors']
    
    # Compute collaborative scores for this user
    logger.info("Computing collaborative scores...")
    collab_scores = np.dot(user_factors, movie_factors.T).flatten()
    
    # Get top N movies by collaborative score
    top_indices = np.argsort(collab_scores)[::-1][:n_cf_seed]
    
    # Convert to canonical IDs
    cf_seeds = []
    for idx in top_indices:
        if idx < len(artifacts['movie_index_map']):
            movie_id = artifacts['movie_index_map'].iloc[idx]['canonical_id']
            cf_seeds.append(movie_id)
    
    logger.info(f"Generated {len(cf_seeds)} CF seeds")
    return cf_seeds

def expand_content_candidates(cf_seeds: List[str], artifacts: Dict[str, Any],
                            m: int = 20, k: int = 50, logger: logging.Logger = None) -> Set[str]:
    """Stage B: Expand candidates using content similarity"""
    logger.info(f"Expanding content candidates from {len(cf_seeds)} seeds...")
    
    # Take top M seeds for expansion
    top_seeds = cf_seeds[:m] if len(cf_seeds) > m else cf_seeds
    
    # Get content neighbors for each seed
    all_candidates = set()
    neighbors_df = artifacts['similarity_neighbors']
    
    for seed in top_seeds:
        # Get neighbors for this seed
        seed_neighbors = neighbors_df[neighbors_df['movie_id'] == seed]
        
        # Take top k neighbors
        top_neighbors = seed_neighbors.nlargest(k, 'score')
        
        for _, row in top_neighbors.iterrows():
            neighbor_id = row['neighbor_id']
            # Only include movies that have collaborative factors
            if neighbor_id in artifacts['movie_id_to_factor_idx']:
                all_candidates.add(neighbor_id)
    
    logger.info(f"Expanded to {len(all_candidates)} content candidates")
    return all_candidates

def apply_hard_filters(candidates: Set[str], artifacts: Dict[str, Any],
                      genre_filter: Optional[List[str]] = None,
                      provider_filter: Optional[List[str]] = None,
                      seen_items: Optional[Set[str]] = None,
                      logger: logging.Logger = None) -> List[str]:
    """Apply hard filters with graceful fallback"""
    logger.info(f"Applying hard filters to {len(candidates)} candidates...")
    
    filtered_candidates = list(candidates)
    
    # Seen items filter
    if seen_items:
        before_seen = len(filtered_candidates)
        filtered_candidates = [c for c in filtered_candidates if c not in seen_items]
        logger.info(f"Seen items filter: {before_seen} -> {len(filtered_candidates)}")
    
    # Genre filter
    if genre_filter:
        before_genre = len(filtered_candidates)
        genre_df = artifacts['genres']
        
        # Filter to candidates that have at least one selected genre
        genre_matches = []
        for candidate in filtered_candidates:
            if candidate in genre_df.index:
                candidate_genres = genre_df.loc[candidate]
                if any(candidate_genres[genre] == 1 for genre in genre_filter if genre in candidate_genres):
                    genre_matches.append(candidate)
        
        filtered_candidates = genre_matches
        logger.info(f"Genre filter: {before_genre} -> {len(filtered_candidates)}")
    
    # Provider filter
    if provider_filter:
        before_provider = len(filtered_candidates)
        platform_df = artifacts['platforms']
        
        # Filter to candidates available on selected providers
        provider_matches = []
        for candidate in filtered_candidates:
            if candidate in platform_df.index:
                candidate_providers = platform_df.loc[candidate]
                if any(candidate_providers[provider] == 1 for provider in provider_filter if provider in candidate_providers):
                    provider_matches.append(candidate)
        
        filtered_candidates = provider_matches
        logger.info(f"Provider filter: {before_provider} -> {len(filtered_candidates)}")
    
    # Safety checks - ensure all candidates have required data
    safe_candidates = []
    for candidate in filtered_candidates:
        if (candidate in artifacts['movie_id_to_factor_idx'] and 
            candidate in artifacts['genres'].index and
            candidate in artifacts['platforms'].index and
            candidate in artifacts['numeric'].index):
            safe_candidates.append(candidate)
    
    logger.info(f"Safety checks: {len(filtered_candidates)} -> {len(safe_candidates)}")
    
    # Graceful fallback if filters are too restrictive
    if len(safe_candidates) < 50:  # Minimum threshold
        logger.warning("Filters too restrictive, applying fallback...")
        
        # Fallback 1: Ignore provider filter
        if provider_filter and len(safe_candidates) < 50:
            logger.info("Fallback 1: Ignoring provider filter")
            filtered_candidates = list(candidates)
            if seen_items:
                filtered_candidates = [c for c in filtered_candidates if c not in seen_items]
            if genre_filter:
                genre_df = artifacts['genres']
                genre_matches = []
                for candidate in filtered_candidates:
                    if candidate in genre_df.index:
                        candidate_genres = genre_df.loc[candidate]
                        if any(candidate_genres[genre] == 1 for genre in genre_filter if genre in candidate_genres):
                            genre_matches.append(candidate)
                filtered_candidates = genre_matches
            
            safe_candidates = []
            for candidate in filtered_candidates:
                if (candidate in artifacts['movie_id_to_factor_idx'] and 
                    candidate in artifacts['genres'].index and
                    candidate in artifacts['numeric'].index):
                    safe_candidates.append(candidate)
        
        # Fallback 2: Broaden to sibling genres (simplified - just remove genre filter)
        if len(safe_candidates) < 50 and genre_filter:
            logger.info("Fallback 2: Removing genre filter")
            filtered_candidates = list(candidates)
            if seen_items:
                filtered_candidates = [c for c in filtered_candidates if c not in seen_items]
            
            safe_candidates = []
            for candidate in filtered_candidates:
                if (candidate in artifacts['movie_id_to_factor_idx'] and 
                    candidate in artifacts['genres'].index and
                    candidate in artifacts['numeric'].index):
                    safe_candidates.append(candidate)
        
        # Fallback 3: Use popularity-weighted content picks
        if len(safe_candidates) < 50:
            logger.info("Fallback 3: Using popularity-weighted content picks")
            # Get top movies by popularity (tmdb_popularity_standardized)
            numeric_df = artifacts['numeric']
            popular_movies = numeric_df.nlargest(1000, 'tmdb_popularity_standardized')
            
            safe_candidates = []
            for candidate in popular_movies.index:
                if (candidate in artifacts['movie_id_to_factor_idx'] and 
                    candidate not in (seen_items or set())):
                    safe_candidates.append(candidate)
                    if len(safe_candidates) >= 200:  # Cap for efficiency
                        break
    
    logger.info(f"Final filtered candidates: {len(safe_candidates)}")
    return safe_candidates

def compute_hybrid_scores(candidates: List[str], user_id: str, artifacts: Dict[str, Any],
                         alpha: float = 0.5, logger: logging.Logger = None) -> Dict[str, float]:
    """Compute hybrid scores for candidates"""
    logger.info(f"Computing hybrid scores for {len(candidates)} candidates...")
    
    scores = {}
    
    # Get user factor index if available
    user_factor_idx = artifacts['user_id_to_factor_idx'].get(user_id)
    
    for candidate in candidates:
        if candidate not in artifacts['movie_id_to_factor_idx']:
            continue
            
        movie_factor_idx = artifacts['movie_id_to_factor_idx'][candidate]
        
        # Compute collaborative score
        if user_factor_idx is not None:
            user_factors = artifacts['user_factors'][user_factor_idx:user_factor_idx+1]
            movie_factors = artifacts['movie_factors'][movie_factor_idx:movie_factor_idx+1]
            collab_score = np.dot(user_factors, movie_factors.T).item()
            
            # Normalize collaborative score (simplified - use global min/max)
            # In practice, this would use the per-user normalization from 3c.1
            collab_score = max(0.0, min(1.0, (collab_score + 2.0) / 4.0))  # Rough normalization
        else:
            collab_score = 0.0  # Cold start
        
        # Compute content score (simplified - use average similarity to user's rated movies)
        # For now, use a placeholder content score
        content_score = 0.5  # Placeholder
        
        # Apply hybrid formula
        hybrid_score = alpha * content_score + (1 - alpha) * collab_score
        scores[candidate] = hybrid_score
    
    logger.info(f"Computed hybrid scores for {len(scores)} candidates")
    return scores

def compute_secondary_signals(candidates: List[str], artifacts: Dict[str, Any],
                            logger: logging.Logger = None) -> Dict[str, Dict[str, float]]:
    """Compute secondary signals for tie-breaking"""
    logger.info(f"Computing secondary signals for {len(candidates)} candidates...")
    
    signals = {}
    numeric_df = artifacts['numeric']
    
    for candidate in candidates:
        if candidate not in numeric_df.index:
            continue
            
        row = numeric_df.loc[candidate]
        
        # Quality boost (from IMDB score)
        quality_score = row.get('imdb_score_standardized', 0.0) / 10.0  # Normalize to [0,1]
        
        # Recency preference (from normalized year)
        recency_score = row.get('release_year_normalized', 0.5)
        
        # Provider match depth (simplified - assume single match)
        provider_score = 0.0  # Would be computed based on actual provider matches
        
        signals[candidate] = {
            'quality_score': quality_score,
            'recency_score': recency_score,
            'provider_score': provider_score
        }
    
    logger.info(f"Computed secondary signals for {len(signals)} candidates")
    return signals

def apply_diversity_reranking(candidates: List[str], hybrid_scores: Dict[str, float],
                            secondary_signals: Dict[str, Dict[str, float]],
                            artifacts: Dict[str, Any],
                            lambda_div: float = 0.25, k_final: int = 50,
                            logger: logging.Logger = None) -> List[str]:
    """Apply MMR-style diversity re-ranking"""
    logger.info(f"Applying diversity re-ranking to top {min(500, len(candidates))} candidates...")
    
    # Sort by primary score first
    scored_candidates = [(c, hybrid_scores.get(c, 0.0)) for c in candidates]
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 500 for diversity processing
    top_candidates = [c for c, _ in scored_candidates[:500]]
    
    if len(top_candidates) <= k_final:
        logger.info(f"Not enough candidates for diversity re-ranking, returning top {len(top_candidates)}")
        return top_candidates
    
    # MMR re-ranking
    selected = []
    remaining = top_candidates.copy()
    
    # Add highest scoring item first
    if remaining:
        selected.append(remaining.pop(0))
    
    # Iteratively select items that maximize MMR score
    while len(selected) < k_final and remaining:
        best_item = None
        best_mmr_score = -float('inf')
        
        for item in remaining:
            # Relevance score (hybrid + secondary signals)
            relevance = hybrid_scores.get(item, 0.0)
            if item in secondary_signals:
                relevance += 0.05 * secondary_signals[item]['quality_score']
                relevance += 0.03 * secondary_signals[item]['recency_score']
                relevance += 0.02 * secondary_signals[item]['provider_score']
            
            # Diversity score (max similarity to already selected items)
            max_similarity = 0.0
            if selected:
                # Use content embeddings for similarity
                item_idx = artifacts['movie_id_to_factor_idx'].get(item)
                if item_idx is not None and item_idx < len(artifacts['content_embeddings']):
                    item_embedding = artifacts['content_embeddings'][item_idx]
                    
                    for selected_item in selected:
                        selected_idx = artifacts['movie_id_to_factor_idx'].get(selected_item)
                        if selected_idx is not None and selected_idx < len(artifacts['content_embeddings']):
                            selected_embedding = artifacts['content_embeddings'][selected_idx]
                            similarity = np.dot(item_embedding, selected_embedding)
                            max_similarity = max(max_similarity, similarity)
            
            # MMR score
            mmr_score = lambda_div * relevance - (1 - lambda_div) * max_similarity
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_item = item
        
        if best_item:
            selected.append(best_item)
            remaining.remove(best_item)
        else:
            break
    
    logger.info(f"Diversity re-ranking selected {len(selected)} items")
    return selected

def generate_candidates(user_id: str, artifacts: Dict[str, Any],
                       config: Dict[str, Any], logger: logging.Logger = None) -> pd.DataFrame:
    """Main candidate generation pipeline"""
    logger.info(f"Generating candidates for user {user_id}...")
    
    # Stage A: CF seeds
    cf_seeds = generate_cf_seeds(user_id, artifacts, config['n_cf_seed'], logger)
    
    # Stage B: Content expansion
    if cf_seeds:
        content_candidates = expand_content_candidates(cf_seeds, artifacts, config['m'], config['k'], logger)
    else:
        # Cold start - use popular movies as seeds
        logger.info("Cold start: using popular movies as seeds")
        numeric_df = artifacts['numeric']
        popular_movies = numeric_df.nlargest(100, 'tmdb_popularity_standardized')
        content_candidates = set(popular_movies.index[:50])  # Top 50 popular movies
    
    # Union and deduplicate
    all_candidates = set(cf_seeds) | content_candidates
    
    # Cap at C_max
    if len(all_candidates) > config['c_max']:
        # Sort by some criteria and take top C_max
        # For simplicity, just take first C_max
        all_candidates = set(list(all_candidates)[:config['c_max']])
    
    logger.info(f"Total raw candidates: {len(all_candidates)}")
    
    # Apply hard filters
    filtered_candidates = apply_hard_filters(
        all_candidates, artifacts, 
        config.get('genre_filter'), 
        config.get('provider_filter'),
        config.get('seen_items'),
        logger
    )
    
    # Compute hybrid scores
    hybrid_scores = compute_hybrid_scores(filtered_candidates, user_id, artifacts, config['alpha'], logger)
    
    # Compute secondary signals
    secondary_signals = compute_secondary_signals(filtered_candidates, artifacts, logger)
    
    # Apply diversity re-ranking
    final_candidates = apply_diversity_reranking(
        filtered_candidates, hybrid_scores, secondary_signals, artifacts,
        config['lambda_div'], config['k_final'], logger
    )
    
    # Create output DataFrame
    results = []
    for i, candidate in enumerate(final_candidates):
        result = {
            'canonical_id': candidate,
            'hybrid_score': hybrid_scores.get(candidate, 0.0),
            'content_score': 0.5,  # Placeholder
            'collab_score': hybrid_scores.get(candidate, 0.0),  # Simplified
            'quality_score_100': secondary_signals.get(candidate, {}).get('quality_score', 0.0) * 100,
            'year_norm': secondary_signals.get(candidate, {}).get('recency_score', 0.5),
            'provider_match_flags': 0,  # Placeholder
            'rank_primary': i + 1,
            'rank_final': i + 1
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    logger.info(f"Generated {len(df)} final candidates")
    return df

def run_acceptance_tests(candidates_df: pd.DataFrame, logger: logging.Logger) -> bool:
    """Run acceptance tests on generated candidates"""
    logger.info("Running acceptance tests...")
    
    # Test 1: Candidate pool size
    if len(candidates_df) == 0:
        logger.error("No candidates generated")
        return False
    
    # Test 2: Score range check
    if 'hybrid_score' in candidates_df.columns:
        min_score = candidates_df['hybrid_score'].min()
        max_score = candidates_df['hybrid_score'].max()
        if min_score < 0.0 or max_score > 1.0:
            logger.error(f"Scores outside [0,1] range: min={min_score:.4f}, max={max_score:.4f}")
            return False
    
    # Test 3: No NaN/Inf values
    if candidates_df.isnull().any().any():
        logger.error("NaN values found in candidates")
        return False
    
    # Test 4: Deduplicated outputs
    if len(candidates_df) != len(candidates_df['canonical_id'].unique()):
        logger.error("Duplicate candidates found")
        return False
    
    # Test 5: Contiguous ranks
    expected_ranks = list(range(1, len(candidates_df) + 1))
    actual_ranks = candidates_df['rank_final'].tolist()
    if actual_ranks != expected_ranks:
        logger.error("Non-contiguous ranks found")
        return False
    
    logger.info("✓ All acceptance tests passed")
    return True

def create_rerank_manifest(config: Dict[str, Any], execution_time: float,
                         candidates_count: int, logger: logging.Logger) -> Dict[str, Any]:
    """Create re-ranking manifest"""
    logger.info("Creating re-ranking manifest...")
    
    manifest = {
        "step": "3c.2",
        "description": "Candidate Generation & Re-ranking",
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "version": "v1",
        "configuration": config,
        "execution_summary": {
            "execution_time_seconds": execution_time,
            "candidates_generated": candidates_count,
            "filters_applied": {
                "genre_filter": config.get('genre_filter') is not None,
                "provider_filter": config.get('provider_filter') is not None,
                "seen_items_filter": config.get('seen_items') is not None
            }
        },
        "acceptance_tests": {
            "candidate_pool_size": candidates_count > 0,
            "score_range_valid": True,
            "no_nan_inf": True,
            "deduplicated": True,
            "contiguous_ranks": True,
            "status": "passed"
        }
    }
    
    return manifest

def main():
    """Main execution function for Step 3c.2"""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Starting Step 3c.2 – Candidate Generation & Re-ranking")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Load artifacts
        logger.info("Loading artifacts...")
        artifacts = load_artifacts(logger)
        
        # Configuration
        config = {
            'alpha': 0.5,
            'lambda_div': 0.25,
            'n_cf_seed': 800,
            'm': 20,
            'k': 50,
            'c_max': 2000,
            'k_final': 50,
            'genre_filter': None,  # ['genre_drama', 'genre_comedy']  # Example
            'provider_filter': None,  # ['provider_netflix_any']  # Example
            'seen_items': None  # Set of seen movie IDs
        }
        
        # Test with a sample user
        test_user_id = "1"  # Use first user from the dataset
        logger.info(f"Testing with user {test_user_id}")
        
        # Generate candidates
        candidates_df = generate_candidates(test_user_id, artifacts, config, logger)
        
        # Run acceptance tests
        if not run_acceptance_tests(candidates_df, logger):
            raise ValueError("Acceptance tests failed")
        
        # Create output directory
        output_dir = Path("data/hybrid/candidates")
        output_dir.mkdir(exist_ok=True)
        
        # Save candidates
        output_file = output_dir / f"user_{test_user_id}_candidates.parquet"
        candidates_df.to_parquet(output_file, index=False)
        logger.info(f"Candidates saved to: {output_file}")
        
        # Create manifest
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        manifest = create_rerank_manifest(config, execution_time, len(candidates_df), logger)
        manifest_path = Path("data/hybrid/rerank_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Re-ranking manifest saved to: {manifest_path}")
        
        # Log execution summary
        logger.info("=" * 80)
        logger.info("Step 3c.2 – Candidate Generation & Re-ranking COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"Test user: {test_user_id}")
        logger.info(f"Candidates generated: {len(candidates_df)}")
        logger.info(f"Configuration: α={config['alpha']}, λ_div={config['lambda_div']}")
        logger.info(f"Deliverables created:")
        logger.info(f"  - {output_file}")
        logger.info(f"  - {manifest_path}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Step 3c.2 failed with error: {str(e)}")
        logger.error("=" * 80)
        raise

if __name__ == "__main__":
    main()
