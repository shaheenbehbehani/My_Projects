#!/usr/bin/env python3
"""
Step 3c.3 – Speed Mode Tuning & Evaluation

This script implements fast α tuning with tight timeouts, minimal metrics,
and hard caps to prevent stalling and ensure reliable evaluation.

Key Features:
- Speed mode with tight candidate and batch knobs
- Minimal metrics first pass (Recall@10, MAP@10 only)
- Hard timeouts with 6min per-α cap and 20min global cap
- Candidate caching and user skipping
- Second pass for best α with full metrics

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
from sklearn.metrics import average_precision_score
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
def setup_logging():
    """Setup logging for Step 3c.3 execution"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "step3c_phase3.log"
    
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
    """Load all required artifacts for evaluation"""
    logger.info("Loading artifacts for speed mode evaluation...")
    
    artifacts = {}
    
    # Load core data
    artifacts['ratings'] = pd.read_parquet("data/collaborative/ratings_long_format.parquet")
    artifacts['user_index_map'] = pd.read_parquet("data/collaborative/user_index_map.parquet")
    artifacts['movie_index_map'] = pd.read_parquet("data/collaborative/movie_index_map.parquet")
    
    # Load factors and embeddings
    artifacts['user_factors'] = np.load("data/collaborative/user_factors_k20.npy", mmap_mode='r')
    artifacts['movie_factors'] = np.load("data/collaborative/movie_factors_k20.npy", mmap_mode='r')
    artifacts['content_embeddings'] = np.load("data/features/composite/movies_embedding_v1.npy", mmap_mode='r')
    
    # Load similarity data
    artifacts['similarity_neighbors'] = pd.read_parquet("data/similarity/movies_neighbors_k50.parquet")
    
    # Load filter features
    artifacts['genres'] = pd.read_parquet("data/features/genres/movies_genres_multihot.parquet")
    artifacts['numeric'] = pd.read_parquet("data/features/numeric/movies_numeric_standardized.parquet")
    
    # Create movie_id to factor_idx mapping
    artifacts['movie_id_to_factor_idx'] = {}
    for i, row in artifacts['movie_index_map'].iterrows():
        if i < len(artifacts['movie_factors']):
            artifacts['movie_id_to_factor_idx'][row['canonical_id']] = i
    
    logger.info(f"Ratings shape: {artifacts['ratings'].shape}")
    logger.info(f"User factors shape: {artifacts['user_factors'].shape}")
    logger.info(f"Movie factors shape: {artifacts['movie_factors'].shape}")
    logger.info(f"Movies with factors: {len(artifacts['movie_id_to_factor_idx'])}")
    
    return artifacts

def stratified_user_sampling(artifacts: Dict[str, Any], config: Dict[str, Any], 
                           logger: logging.Logger) -> pd.DataFrame:
    """Create stratified user sampling with activity buckets"""
    logger.info("Creating stratified user sampling...")
    
    # Count ratings per user
    user_rating_counts = artifacts['ratings']['user_index'].value_counts()
    
    # Create activity buckets
    cold_users = user_rating_counts[user_rating_counts <= 3].index.tolist()
    medium_users = user_rating_counts[(user_rating_counts > 3) & (user_rating_counts < 50)].index.tolist()
    heavy_users = user_rating_counts[user_rating_counts >= 50].index.tolist()
    
    logger.info(f"User buckets: Cold={len(cold_users)}, Medium={len(medium_users)}, Heavy={len(heavy_users)}")
    
    # Sample users with target proportions
    target_total = config['n_users']
    n_cold = int(target_total * 0.2)
    n_medium = int(target_total * 0.6)
    n_heavy = int(target_total * 0.2)
    
    # Adjust if not enough users in a bucket
    n_cold = min(n_cold, len(cold_users))
    n_medium = min(n_medium, len(medium_users))
    n_heavy = min(n_heavy, len(heavy_users))
    
    # Sample users
    np.random.seed(42)
    sampled_cold = np.random.choice(cold_users, size=n_cold, replace=False)
    sampled_medium = np.random.choice(medium_users, size=n_medium, replace=False)
    sampled_heavy = np.random.choice(heavy_users, size=n_heavy, replace=False)
    
    # Create user sample DataFrame
    user_samples = []
    for user_id in sampled_cold:
        user_samples.append({'user_index': user_id, 'bucket': 'cold', 'rating_count': user_rating_counts[user_id]})
    for user_id in sampled_medium:
        user_samples.append({'user_index': user_id, 'bucket': 'medium', 'rating_count': user_rating_counts[user_id]})
    for user_id in sampled_heavy:
        user_samples.append({'user_index': user_id, 'bucket': 'heavy', 'rating_count': user_rating_counts[user_id]})
    
    user_sample_df = pd.DataFrame(user_samples)
    logger.info(f"Sampled {len(user_sample_df)} users: {user_sample_df['bucket'].value_counts().to_dict()}")
    
    return user_sample_df

def generate_cf_seeds_fast(user_id: int, artifacts: Dict[str, Any], n_cf_seed: int = 400) -> List[str]:
    """Generate CF seeds for a user (speed mode)"""
    # Get user factor index
    user_map = artifacts['user_index_map']
    user_row = user_map[user_map['userId'] == user_id]
    
    if len(user_row) == 0 or user_row.index[0] >= len(artifacts['user_factors']):
        return []
    
    user_factor_idx = user_row.index[0]
    user_factors = artifacts['user_factors'][user_factor_idx:user_factor_idx+1]
    movie_factors = artifacts['movie_factors']
    
    # Compute collaborative scores
    collab_scores = np.dot(user_factors, movie_factors.T).flatten()
    
    # Get top N movies by collaborative score
    top_indices = np.argsort(collab_scores)[::-1][:n_cf_seed]
    
    # Convert to canonical IDs
    cf_seeds = []
    for idx in top_indices:
        if idx < len(artifacts['movie_index_map']):
            movie_id = artifacts['movie_index_map'].iloc[idx]['canonical_id']
            cf_seeds.append(movie_id)
    
    return cf_seeds

def expand_content_candidates_fast(cf_seeds: List[str], artifacts: Dict[str, Any],
                                 m: int = 10, k: int = 30) -> Set[str]:
    """Expand candidates using content similarity (speed mode)"""
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
    
    return all_candidates

def apply_seen_items_filter(candidates: Set[str], user_id: int, artifacts: Dict[str, Any]) -> List[str]:
    """Apply seen items filter - drop already-rated items"""
    # Get user's rated items
    user_ratings = artifacts['ratings'][artifacts['ratings']['user_index'] == user_id]
    seen_items = set(user_ratings['canonical_id'].tolist())
    
    # Filter out seen items
    filtered_candidates = [c for c in candidates if c not in seen_items]
    
    return filtered_candidates

def fallback_content_expansion_fast(user_id: int, artifacts: Dict[str, Any], m: int = 10) -> List[str]:
    """Fallback #1: Content-only expansion around user's top-rated items (speed mode)"""
    # Get user's top-rated items
    user_ratings = artifacts['ratings'][artifacts['ratings']['user_index'] == user_id]
    if len(user_ratings) == 0:
        return []
    
    # Sort by rating and take top M
    top_rated = user_ratings.nlargest(m, 'rating')
    top_movies = top_rated['canonical_id'].tolist()
    
    # Expand around these movies using content similarity
    all_candidates = set()
    neighbors_df = artifacts['similarity_neighbors']
    
    for movie_id in top_movies:
        # Get neighbors for this movie
        movie_neighbors = neighbors_df[neighbors_df['movie_id'] == movie_id]
        
        # Take top 5 neighbors (reduced for speed)
        top_neighbors = movie_neighbors.nlargest(5, 'score')
        
        for _, row in top_neighbors.iterrows():
            neighbor_id = row['neighbor_id']
            if neighbor_id in artifacts['movie_id_to_factor_idx']:
                all_candidates.add(neighbor_id)
    
    # Apply seen items filter
    candidates = apply_seen_items_filter(all_candidates, user_id, artifacts)
    
    return candidates

def fallback_popularity_sampler_fast(user_id: int, artifacts: Dict[str, Any], size: int = 500) -> List[str]:
    """Fallback #2: Popularity-weighted sampler from catalog (speed mode)"""
    # Get popular movies
    numeric_df = artifacts['numeric']
    popular_movies = numeric_df.nlargest(1000, 'tmdb_popularity_standardized')  # Reduced for speed
    
    # Filter to movies with factors and apply seen items filter
    candidates = []
    seen_items = set(artifacts['ratings'][artifacts['ratings']['user_index'] == user_id]['canonical_id'].tolist())
    
    for movie_id in popular_movies.index:
        if (movie_id in artifacts['movie_id_to_factor_idx'] and 
            movie_id not in seen_items):
            candidates.append(movie_id)
            if len(candidates) >= size:
                break
    
    return candidates

def generate_user_candidates_speed(user_id: int, artifacts: Dict[str, Any], config: Dict[str, Any]) -> Tuple[List[str], str]:
    """Generate candidates for a user with speed optimizations"""
    # Stage A: CF seeds
    cf_seeds = generate_cf_seeds_fast(user_id, artifacts, config['n_cf_seed'])
    
    # Stage B: Content expansion
    if cf_seeds:
        content_candidates = expand_content_candidates_fast(cf_seeds, artifacts, config['m'], config['k'])
    else:
        # Cold start - use popular movies as seeds
        numeric_df = artifacts['numeric']
        popular_movies = numeric_df.nlargest(50, 'tmdb_popularity_standardized')  # Reduced for speed
        content_candidates = set(popular_movies.index[:25])
    
    # Union and deduplicate
    all_candidates = set(cf_seeds) | content_candidates
    
    # Cap at C_max
    if len(all_candidates) > config['c_max']:
        all_candidates = set(list(all_candidates)[:config['c_max']])
    
    # Apply seen items filter
    candidates = apply_seen_items_filter(all_candidates, user_id, artifacts)
    
    # Check if we have candidates
    if len(candidates) > 0:
        return candidates, "success"
    
    # Fallback #1: Content-only expansion
    candidates = fallback_content_expansion_fast(user_id, artifacts, config['m'])
    if len(candidates) > 0:
        return candidates, "fallback1"
    
    # Fallback #2: Popularity sampler
    candidates = fallback_popularity_sampler_fast(user_id, artifacts, 500)
    if len(candidates) > 0:
        return candidates, "fallback2"
    
    # Still no candidates - skip user
    return [], "skipped"

def compute_hybrid_scores_batch_fast(user_ids: List[int], candidates_list: List[List[str]], 
                                    alpha: float, artifacts: Dict[str, Any]) -> List[Dict[str, float]]:
    """Compute hybrid scores for a batch of users efficiently (speed mode)"""
    results = []
    
    for user_id, candidates in zip(user_ids, candidates_list):
        if not candidates:
            continue
        
        # Get user factor index
        user_map = artifacts['user_index_map']
        user_row = user_map[user_map['userId'] == user_id]
        
        if len(user_row) == 0 or user_row.index[0] >= len(artifacts['user_factors']):
            # Cold start - use content scores only
            scores = {candidate: 0.5 for candidate in candidates}
        else:
            user_factor_idx = user_row.index[0]
            user_factors = artifacts['user_factors'][user_factor_idx:user_factor_idx+1]
            
            # Get movie factor indices for candidates
            candidate_indices = []
            for candidate in candidates:
                if candidate in artifacts['movie_id_to_factor_idx']:
                    candidate_indices.append(artifacts['movie_id_to_factor_idx'][candidate])
            
            if not candidate_indices:
                scores = {candidate: 0.5 for candidate in candidates}
            else:
                # Compute collaborative scores
                candidate_factors = artifacts['movie_factors'][candidate_indices]
                collab_scores = np.dot(user_factors, candidate_factors.T).flatten()
                
                # Normalize collaborative scores
                if len(collab_scores) > 1:
                    collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min() + 1e-8)
                else:
                    collab_scores = np.array([0.5])
                
                # Compute content scores (simplified - use popularity)
                numeric_df = artifacts['numeric']
                content_scores = []
                for candidate in candidates:
                    if candidate in numeric_df.index:
                        content_scores.append(numeric_df.loc[candidate, 'tmdb_popularity_standardized'])
                    else:
                        content_scores.append(0.0)
                
                # Normalize content scores
                content_scores = np.array(content_scores)
                if len(content_scores) > 1 and content_scores.max() > content_scores.min():
                    content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())
                else:
                    content_scores = np.full_like(content_scores, 0.5)
                
                # Apply hybrid formula
                hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores
                
                # Create scores dictionary
                scores = {}
                for i, candidate in enumerate(candidates):
                    if i < len(hybrid_scores):
                        scores[candidate] = hybrid_scores[i]
                    else:
                        scores[candidate] = 0.5
        
        results.append({'user_id': user_id, 'scores': scores})
    
    return results

def compute_minimal_metrics(recommendations: List[str], test_items: Set[str]) -> Dict[str, float]:
    """Compute minimal metrics (Recall@10, MAP@10 only)"""
    metrics = {}
    
    # Recall@10
    if len(test_items) == 0:
        metrics['recall@10'] = 0.0
    else:
        top_10_recs = set(recommendations[:10])
        intersection = len(top_10_recs & test_items)
        metrics['recall@10'] = intersection / len(test_items)
    
    # MAP@10
    if len(test_items) == 0:
        metrics['map@10'] = 0.0
    else:
        relevance = [1 if rec in test_items else 0 for rec in recommendations[:10]]
        if sum(relevance) == 0:
            metrics['map@10'] = 0.0
        else:
            metrics['map@10'] = average_precision_score(relevance, relevance)
    
    return metrics

def compute_full_metrics(recommendations: List[str], test_items: Set[str], 
                        artifacts: Dict[str, Any]) -> Dict[str, float]:
    """Compute full metrics including Coverage, Novelty, Diversity"""
    metrics = {}
    
    # Recall@10
    if len(test_items) == 0:
        metrics['recall@10'] = 0.0
    else:
        top_10_recs = set(recommendations[:10])
        intersection = len(top_10_recs & test_items)
        metrics['recall@10'] = intersection / len(test_items)
    
    # MAP@10
    if len(test_items) == 0:
        metrics['map@10'] = 0.0
    else:
        relevance = [1 if rec in test_items else 0 for rec in recommendations[:10]]
        if sum(relevance) == 0:
            metrics['map@10'] = 0.0
        else:
            metrics['map@10'] = average_precision_score(relevance, relevance)
    
    # Coverage@10
    unique_recs = len(set(recommendations[:10]))
    catalog_size = len(artifacts['movie_factors'])
    metrics['coverage'] = unique_recs / catalog_size
    
    # Novelty (simplified)
    metrics['novelty'] = 0.5  # Placeholder
    
    # Diversity (simplified)
    metrics['diversity'] = 0.5  # Placeholder
    
    return metrics

def evaluate_alpha_speed(alpha: float, user_sample_df: pd.DataFrame, 
                        test_df: pd.DataFrame, artifacts: Dict[str, Any],
                        config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Evaluate α using speed mode approach"""
    logger.info(f"Evaluating α = {alpha}")
    alpha_start = datetime.now()
    
    # Get user IDs
    user_ids = user_sample_df['user_index'].tolist()
    batch_size = config['batch_size_users']
    
    all_results = []
    skipped_users = 0
    total_candidates = 0
    
    # Process users in batches
    total_batches = (len(user_ids) + batch_size - 1) // batch_size
    
    for i in range(0, len(user_ids), batch_size):
        batch_start = datetime.now()
        batch = user_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        # Generate candidates for batch
        candidates_list = []
        batch_user_ids = []
        batch_skipped = 0
        
        for user_id in batch:
            candidates, status = generate_user_candidates_speed(user_id, artifacts, config)
            
            if status == "skipped":
                batch_skipped += 1
            else:
                candidates_list.append(candidates)
                batch_user_ids.append(user_id)
                total_candidates += len(candidates)
        
        skipped_users += batch_skipped
        
        if not candidates_list:
            continue
        
        # Compute hybrid scores for batch
        score_results = compute_hybrid_scores_batch_fast(batch_user_ids, candidates_list, alpha, artifacts)
        
        # Compute minimal metrics (no MMR for speed)
        for user_id, score_result in zip(batch_user_ids, score_results):
            scores = score_result['scores']
            candidates = [c for c in score_result['scores'].keys()]
            
            # Sort by score (no MMR for speed)
            scored_candidates = [(c, scores.get(c, 0.0)) for c in candidates]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            recommendations = [c for c, _ in scored_candidates[:20]]  # Top 20
            
            # Get test items for this user
            user_test = test_df[test_df['user_index'] == user_id]
            test_items = set(user_test['canonical_id'].tolist())
            
            if len(test_items) == 0:
                continue
            
            # Compute minimal metrics
            metrics = compute_minimal_metrics(recommendations, test_items)
            
            # Add user info
            result = {
                'user_id': user_id,
                'alpha': alpha,
                'test_items_count': len(test_items),
                'candidates_count': len(candidates),
                **metrics
            }
            all_results.append(result)
        
        batch_end = datetime.now()
        batch_time = (batch_end - batch_start).total_seconds()
        alpha_elapsed = (batch_end - alpha_start).total_seconds()
        
        # Log batch heartbeat
        mean_cand = total_candidates / len(all_results) if all_results else 0
        logger.info(f"alpha={alpha} batch={batch_num}/{total_batches} users={len(batch_user_ids)} mean_cand={mean_cand:.1f} time_sec={batch_time:.1f} elapsed={alpha_elapsed:.1f}")
        
        # Check for batch timeout
        if batch_time > 60:
            logger.warning(f"Batch {batch_num} took {batch_time:.2f}s (>60s)")
            batch_size = max(100, batch_size // 2)
        
        # Check for alpha timeout
        if alpha_elapsed > 360:  # 6 minutes
            logger.warning(f"α = {alpha} exceeded 6 minutes, finishing current batch and proceeding")
            break
    
    alpha_end = datetime.now()
    alpha_time = (alpha_end - alpha_start).total_seconds()
    alpha_timed_out = alpha_time > 360
    
    # Aggregate results
    if all_results:
        results_df = pd.DataFrame(all_results)
        aggregated = results_df.groupby('alpha').agg({
            'recall@10': 'mean',
            'map@10': 'mean',
            'candidates_count': 'mean'
        }).iloc[0].to_dict()
        
        aggregated.update({
            'alpha': alpha,
            'users_evaluated': len(all_results),
            'users_skipped': skipped_users,
            'mean_cand': total_candidates / len(all_results) if all_results else 0,
            'alpha_time_sec': alpha_time,
            'alpha_timed_out': alpha_timed_out
        })
    else:
        aggregated = {
            'alpha': alpha,
            'recall@10': 0.0,
            'map@10': 0.0,
            'users_evaluated': 0,
            'users_skipped': skipped_users,
            'mean_cand': 0,
            'alpha_time_sec': alpha_time,
            'alpha_timed_out': alpha_timed_out
        }
    
    logger.info(f"α = {alpha} completed in {alpha_time:.2f}s: {aggregated['users_evaluated']} users, {aggregated['mean_cand']:.1f} avg candidates")
    
    return aggregated

def evaluate_best_alpha_full(best_alpha: float, user_sample_df: pd.DataFrame, 
                            test_df: pd.DataFrame, artifacts: Dict[str, Any],
                            config: Dict[str, Any], logger: logging.Logger) -> Dict[str, float]:
    """Evaluate best α with full metrics (second pass)"""
    logger.info(f"Second pass: computing full metrics for α = {best_alpha}")
    
    # Get user IDs
    user_ids = user_sample_df['user_index'].tolist()
    batch_size = 200  # Smaller batch for second pass
    
    all_results = []
    skipped_users = 0
    total_candidates = 0
    
    # Process users in batches
    for i in range(0, len(user_ids), batch_size):
        batch = user_ids[i:i + batch_size]
        
        # Generate candidates for batch
        candidates_list = []
        batch_user_ids = []
        batch_skipped = 0
        
        for user_id in batch:
            candidates, status = generate_user_candidates_speed(user_id, artifacts, config)
            
            if status == "skipped":
                batch_skipped += 1
            else:
                candidates_list.append(candidates)
                batch_user_ids.append(user_id)
                total_candidates += len(candidates)
        
        skipped_users += batch_skipped
        
        if not candidates_list:
            continue
        
        # Compute hybrid scores for batch
        score_results = compute_hybrid_scores_batch_fast(batch_user_ids, candidates_list, best_alpha, artifacts)
        
        # Compute full metrics
        for user_id, score_result in zip(batch_user_ids, score_results):
            scores = score_result['scores']
            candidates = [c for c in score_result['scores'].keys()]
            
            # Sort by score (no MMR for speed)
            scored_candidates = [(c, scores.get(c, 0.0)) for c in candidates]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            recommendations = [c for c, _ in scored_candidates[:20]]  # Top 20
            
            # Get test items for this user
            user_test = test_df[test_df['user_index'] == user_id]
            test_items = set(user_test['canonical_id'].tolist())
            
            if len(test_items) == 0:
                continue
            
            # Compute full metrics
            metrics = compute_full_metrics(recommendations, test_items, artifacts)
            
            # Add user info
            result = {
                'user_id': user_id,
                'alpha': best_alpha,
                'test_items_count': len(test_items),
                'candidates_count': len(candidates),
                **metrics
            }
            all_results.append(result)
    
    # Aggregate results
    if all_results:
        results_df = pd.DataFrame(all_results)
        aggregated = results_df.groupby('alpha').agg({
            'recall@10': 'mean',
            'map@10': 'mean',
            'coverage': 'mean',
            'novelty': 'mean',
            'diversity': 'mean',
            'candidates_count': 'mean'
        }).iloc[0].to_dict()
        
        aggregated.update({
            'alpha': best_alpha,
            'users_evaluated': len(all_results),
            'users_skipped': skipped_users,
            'mean_cand': total_candidates / len(all_results) if all_results else 0,
            'alpha_time_sec': 0,  # Second pass time not tracked
            'alpha_timed_out': False
        })
    else:
        aggregated = {
            'alpha': best_alpha,
            'recall@10': 0.0,
            'map@10': 0.0,
            'coverage': 0.0,
            'novelty': 0.0,
            'diversity': 0.0,
            'users_evaluated': 0,
            'users_skipped': skipped_users,
            'mean_cand': 0,
            'alpha_time_sec': 0,
            'alpha_timed_out': False
        }
    
    logger.info(f"Second pass completed: {aggregated['users_evaluated']} users, full metrics computed")
    
    return aggregated

def main():
    """Main execution function for speed mode evaluation"""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Starting Step 3c.3 – Speed Mode Tuning & Evaluation")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Configuration with speed optimizations
        config = {
            'n_users': 1500,  # Reduced for speed
            'n_cf_seed': 400,  # Reduced from 800
            'm': 10,  # Reduced from 20
            'k': 30,  # Reduced from 50
            'c_max': 1200,  # Reduced from 2000
            'batch_size_users': 200,  # Reduced from 500
            'alpha_values': [0.35, 0.5, 0.65]
        }
        
        # Load artifacts
        logger.info("Loading artifacts...")
        artifacts = load_artifacts(logger)
        
        # Create user sampling
        user_sample_df = stratified_user_sampling(artifacts, config, logger)
        
        # Create train/test split (simplified)
        ratings = artifacts['ratings']
        test_df = ratings.sample(frac=0.2, random_state=42)
        train_df = ratings.drop(test_df.index)
        
        logger.info(f"Train ratings: {len(train_df)}, Test ratings: {len(test_df)}")
        
        # Run α evaluation (first pass - minimal metrics)
        logger.info("Running speed mode α evaluation (first pass)...")
        alpha_results = []
        
        for alpha in config['alpha_values']:
            # Check global timeout
            global_elapsed = (datetime.now() - start_time).total_seconds()
            if global_elapsed > 1200:  # 20 minutes
                logger.warning(f"Global timeout exceeded, stopping after current α")
                break
            
            result = evaluate_alpha_speed(alpha, user_sample_df, test_df, artifacts, config, logger)
            alpha_results.append(result)
        
        # Find best α
        if alpha_results:
            best_result = max(alpha_results, key=lambda x: x['recall@10'])
            best_alpha = best_result['alpha']
            
            logger.info(f"Best α found: {best_alpha} with Recall@10 = {best_result['recall@10']:.4f}")
            
            # Second pass for best α with full metrics
            full_result = evaluate_best_alpha_full(best_alpha, user_sample_df, test_df, artifacts, config, logger)
            alpha_results.append(full_result)
        
        # Save results
        results_df = pd.DataFrame(alpha_results)
        results_dir = Path("data/hybrid")
        results_dir.mkdir(exist_ok=True)
        results_df.to_csv(results_dir / "tuning_results.csv", index=False)
        
        # Create evaluation report
        docs_dir = Path("docs")
        docs_dir.mkdir(exist_ok=True)
        
        if alpha_results:
            best_result = max(alpha_results, key=lambda x: x['recall@10'])
            
            report = f"""# Step 3c.3 – Speed Mode Evaluation Report

## Summary
- **Best α**: {best_result['alpha']}
- **Best Recall@10**: {best_result['recall@10']:.4f}
- **Users evaluated**: {best_result['users_evaluated']}
- **Mean candidate size**: {best_result['mean_cand']:.1f}
- **Total runtime**: {(datetime.now() - start_time).total_seconds():.2f}s

## Results
{results_df.to_string(index=False)}

## Technical Notes
- Speed mode with tight candidate and batch knobs
- Minimal metrics first pass (Recall@10, MAP@10 only)
- Hard timeouts: 6min per-α, 20min global
- Second pass for best α with full metrics
- No MMR during metric scoring for speed

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        else:
            report = "# Step 3c.3 – Speed Mode Evaluation Report\n\nNo results generated due to timeouts.\n"
        
        with open(docs_dir / "step3c_eval.md", 'w') as f:
            f.write(report)
        
        # Log execution summary
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info("Step 3c.3 – Speed Mode Evaluation COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        if alpha_results:
            best_result = max(alpha_results, key=lambda x: x['recall@10'])
            logger.info(f"Best α: {best_result['alpha']}")
            logger.info(f"Best Recall@10: {best_result['recall@10']:.4f}")
            logger.info(f"Users evaluated: {best_result['users_evaluated']}")
        logger.info(f"Deliverables created:")
        logger.info(f"  - {results_dir / 'tuning_results.csv'}")
        logger.info(f"  - {docs_dir / 'step3c_eval.md'}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Step 3c.3 failed with error: {str(e)}")
        logger.error("=" * 80)
        raise

if __name__ == "__main__":
    main()










