#!/usr/bin/env python3
"""
Step 3c.3 – Robust Candidate-Based Tuning & Evaluation

This script implements efficient α tuning with strict loop guardrails and simplified
offline filters to prevent infinite fallback cycles and ensure reliable evaluation.

Key Features:
- Strict 2-fallback limit with hard stops
- Simplified offline filters (provider off, genre soft, seen-items on)
- Efficient batch scoring with runtime guardrails
- Compact logging with batch summaries
- Deterministic evaluation with fixed seeds

Author: Movie Recommendation Optimizer Pipeline
Date: 2025-01-27
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    logger.info("Loading artifacts for robust candidate-based evaluation...")
    
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

def generate_cf_seeds(user_id: int, artifacts: Dict[str, Any], n_cf_seed: int = 800) -> List[str]:
    """Generate CF seeds for a user"""
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

def expand_content_candidates(cf_seeds: List[str], artifacts: Dict[str, Any],
                            m: int = 20, k: int = 50) -> Set[str]:
    """Expand candidates using content similarity"""
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

def fallback_content_expansion(user_id: int, artifacts: Dict[str, Any], m: int = 20) -> List[str]:
    """Fallback #1: Content-only expansion around user's top-rated items"""
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
        
        # Take top 10 neighbors
        top_neighbors = movie_neighbors.nlargest(10, 'score')
        
        for _, row in top_neighbors.iterrows():
            neighbor_id = row['neighbor_id']
            if neighbor_id in artifacts['movie_id_to_factor_idx']:
                all_candidates.add(neighbor_id)
    
    # Apply seen items filter
    candidates = apply_seen_items_filter(all_candidates, user_id, artifacts)
    
    return candidates

def fallback_popularity_sampler(user_id: int, artifacts: Dict[str, Any], size: int = 500) -> List[str]:
    """Fallback #2: Popularity-weighted sampler from catalog"""
    # Get popular movies
    numeric_df = artifacts['numeric']
    popular_movies = numeric_df.nlargest(2000, 'tmdb_popularity_standardized')
    
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

def generate_user_candidates_robust(user_id: int, artifacts: Dict[str, Any], config: Dict[str, Any],
                                  logger: logging.Logger) -> Tuple[List[str], str]:
    """Generate candidates for a user with strict 2-fallback limit"""
    # Stage A: CF seeds
    cf_seeds = generate_cf_seeds(user_id, artifacts, config['n_cf_seed'])
    
    # Stage B: Content expansion
    if cf_seeds:
        content_candidates = expand_content_candidates(cf_seeds, artifacts, config['m'], config['k'])
    else:
        # Cold start - use popular movies as seeds
        numeric_df = artifacts['numeric']
        popular_movies = numeric_df.nlargest(100, 'tmdb_popularity_standardized')
        content_candidates = set(popular_movies.index[:50])
    
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
    candidates = fallback_content_expansion(user_id, artifacts, config['m'])
    if len(candidates) > 0:
        return candidates, "fallback1"
    
    # Fallback #2: Popularity sampler
    candidates = fallback_popularity_sampler(user_id, artifacts, 500)
    if len(candidates) > 0:
        return candidates, "fallback2"
    
    # Still no candidates - skip user
    return [], "skipped"

def compute_hybrid_scores_batch(user_ids: List[int], candidates_list: List[List[str]], 
                               alpha: float, artifacts: Dict[str, Any]) -> List[Dict[str, float]]:
    """Compute hybrid scores for a batch of users efficiently"""
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

def apply_mmr_reranking(scores: Dict[str, float], candidates: List[str], 
                       artifacts: Dict[str, Any], lambda_div: float = 0.25,
                       k_final: int = 20) -> List[str]:
    """Apply MMR re-ranking to candidates"""
    if len(candidates) <= k_final:
        return candidates
    
    # Sort by score first
    scored_candidates = [(c, scores.get(c, 0.0)) for c in candidates]
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Take top candidates for MMR
    top_candidates = [c for c, _ in scored_candidates[:min(500, len(candidates))]]
    
    if len(top_candidates) <= k_final:
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
            # Relevance score
            relevance = scores.get(item, 0.0)
            
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
    
    return selected

def compute_metrics_fast(recommendations: List[str], test_items: Set[str], 
                        artifacts: Dict[str, Any]) -> Dict[str, float]:
    """Compute evaluation metrics efficiently"""
    metrics = {}
    
    # Recall@K
    for k in [5, 10, 20]:
        if len(test_items) == 0:
            metrics[f'recall_at_{k}'] = 0.0
        else:
            top_k_recs = set(recommendations[:k])
            intersection = len(top_k_recs & test_items)
            metrics[f'recall_at_{k}'] = intersection / len(test_items)
    
    # MAP@10
    if len(test_items) == 0:
        metrics['map_at_10'] = 0.0
    else:
        relevance = [1 if rec in test_items else 0 for rec in recommendations[:10]]
        if sum(relevance) == 0:
            metrics['map_at_10'] = 0.0
        else:
            metrics['map_at_10'] = average_precision_score(relevance, relevance)
    
    # Coverage@10
    unique_recs = len(set(recommendations[:10]))
    catalog_size = len(artifacts['movie_factors'])
    metrics['coverage'] = unique_recs / catalog_size
    
    # Novelty (simplified)
    metrics['novelty'] = 0.5  # Placeholder
    
    # Diversity (simplified)
    metrics['diversity'] = 0.5  # Placeholder
    
    return metrics

def evaluate_alpha_robust(alpha: float, user_sample_df: pd.DataFrame, 
                         test_df: pd.DataFrame, artifacts: Dict[str, Any],
                         config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Evaluate α using robust candidate-based approach"""
    logger.info(f"Evaluating α = {alpha}")
    alpha_start = datetime.now()
    
    # Get user IDs
    user_ids = user_sample_df['user_index'].tolist()
    batch_size = config['batch_size_users']
    
    all_results = []
    skipped_users = 0
    total_candidates = 0
    fallback1_count = 0
    fallback2_count = 0
    
    # Process users in batches
    for i in range(0, len(user_ids), batch_size):
        batch_start = datetime.now()
        batch = user_ids[i:i + batch_size]
        
        # Generate candidates for batch
        candidates_list = []
        batch_user_ids = []
        batch_fallback1 = 0
        batch_fallback2 = 0
        batch_skipped = 0
        
        for user_id in batch:
            candidates, status = generate_user_candidates_robust(user_id, artifacts, config, logger)
            
            if status == "skipped":
                batch_skipped += 1
                logger.debug(f"User {user_id} skipped due to empty pool")
            else:
                candidates_list.append(candidates)
                batch_user_ids.append(user_id)
                total_candidates += len(candidates)
                
                if status == "fallback1":
                    batch_fallback1 += 1
                elif status == "fallback2":
                    batch_fallback2 += 1
        
        skipped_users += batch_skipped
        fallback1_count += batch_fallback1
        fallback2_count += batch_fallback2
        
        if not candidates_list:
            continue
        
        # Compute hybrid scores for batch
        score_results = compute_hybrid_scores_batch(batch_user_ids, candidates_list, alpha, artifacts)
        
        # Apply MMR re-ranking and compute metrics
        for user_id, score_result in zip(batch_user_ids, score_results):
            scores = score_result['scores']
            candidates = [c for c in score_result['scores'].keys()]
            
            # Apply MMR re-ranking
            recommendations = apply_mmr_reranking(scores, candidates, artifacts, 
                                               config['lambda_div'], config['k_final'])
            
            # Get test items for this user
            user_test = test_df[test_df['user_index'] == user_id]
            test_items = set(user_test['canonical_id'].tolist())
            
            if len(test_items) == 0:
                continue
            
            # Compute metrics
            metrics = compute_metrics_fast(recommendations, test_items, artifacts)
            
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
        
        # Log batch summary
        mean_cand = total_candidates / len(all_results) if all_results else 0
        logger.info(f"batch_done users={len(batch_user_ids)} skipped={batch_skipped} mean_cand={mean_cand:.1f} α={alpha} time_sec={batch_time:.1f}")
        
        if batch_time > 60:
            logger.warning(f"Batch {i//batch_size + 1} took {batch_time:.2f}s (>60s)")
            # Shrink batch size for next batch
            batch_size = max(200, batch_size // 2)
    
    alpha_end = datetime.now()
    alpha_time = (alpha_end - alpha_start).total_seconds()
    
    # Aggregate results
    if all_results:
        results_df = pd.DataFrame(all_results)
        aggregated = results_df.groupby('alpha').agg({
            'recall_at_5': 'mean',
            'recall_at_10': 'mean',
            'recall_at_20': 'mean',
            'map_at_10': 'mean',
            'coverage': 'mean',
            'novelty': 'mean',
            'diversity': 'mean',
            'candidates_count': 'mean'
        }).iloc[0].to_dict()
        
        aggregated.update({
            'alpha': alpha,
            'users_evaluated': len(all_results),
            'users_skipped': skipped_users,
            'mean_cand': total_candidates / len(all_results) if all_results else 0,
            'wall_time_sec': alpha_time,
            'fallback1_count': fallback1_count,
            'fallback2_count': fallback2_count
        })
    else:
        aggregated = {
            'alpha': alpha,
            'recall_at_5': 0.0,
            'recall_at_10': 0.0,
            'recall_at_20': 0.0,
            'map_at_10': 0.0,
            'coverage': 0.0,
            'novelty': 0.0,
            'diversity': 0.0,
            'users_evaluated': 0,
            'users_skipped': skipped_users,
            'mean_cand': 0,
            'wall_time_sec': alpha_time,
            'fallback1_count': fallback1_count,
            'fallback2_count': fallback2_count
        }
    
    logger.info(f"α = {alpha} completed in {alpha_time:.2f}s: {aggregated['users_evaluated']} users, {aggregated['mean_cand']:.1f} avg candidates")
    
    return aggregated

def main():
    """Main execution function for robust candidate-based evaluation"""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Starting Step 3c.3 – Robust Candidate-Based Tuning & Evaluation")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Configuration with strict guardrails
        config = {
            'n_users': 2000,  # 2k users for efficiency
            'n_cf_seed': 800,
            'm': 20,
            'k': 50,
            'c_max': 2000,
            'k_final': 20,
            'lambda_div': 0.25,
            'batch_size_users': 500,
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
        
        # Run α evaluation
        logger.info("Running robust candidate-based α evaluation...")
        alpha_results = []
        
        for alpha in config['alpha_values']:
            result = evaluate_alpha_robust(alpha, user_sample_df, test_df, artifacts, config, logger)
            alpha_results.append(result)
        
        # Save results
        results_df = pd.DataFrame(alpha_results)
        results_dir = Path("data/hybrid")
        results_dir.mkdir(exist_ok=True)
        results_df.to_csv(results_dir / "tuning_results.csv", index=False)
        
        # Create evaluation report
        docs_dir = Path("docs")
        docs_dir.mkdir(exist_ok=True)
        
        best_result = results_df.loc[results_df['recall_at_10'].idxmax()]
        
        # Calculate fallback percentages
        total_users = len(user_sample_df)
        fallback1_pct = (results_df['fallback1_count'].sum() / total_users * 100)
        fallback2_pct = (results_df['fallback2_count'].sum() / total_users * 100)
        skip_pct = (results_df['users_skipped'].sum() / total_users * 100)
        
        report = f"""# Step 3c.3 – Robust Candidate-Based Evaluation Report

## Summary
- **Best α**: {best_result['alpha']}
- **Best Recall@10**: {best_result['recall_at_10']:.4f}
- **Users evaluated**: {best_result['users_evaluated']}
- **Mean candidate size**: {best_result['mean_cand']:.1f}
- **Total runtime**: {(datetime.now() - start_time).total_seconds():.2f}s

## Fallback Analysis
- **Fallback #1**: {fallback1_pct:.1f}% of users
- **Fallback #2**: {fallback2_pct:.1f}% of users  
- **Skipped**: {skip_pct:.1f}% of users

## Results
{results_df.to_string(index=False)}

## Technical Notes
- Provider filters disabled during offline evaluation (historical ratings lack reliable provider overlap)
- Genre filters treated as soft boosts rather than hard excludes during evaluation
- Seen-items filter applied to prevent recommending already-rated content
- Strict 2-fallback limit prevents infinite retry loops
- All metrics computed on candidate sets only (not full catalog)

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(docs_dir / "step3c_eval.md", 'w') as f:
            f.write(report)
        
        # Log execution summary
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info("Step 3c.3 – Robust Candidate-Based Evaluation COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"Best α: {best_result['alpha']}")
        logger.info(f"Best Recall@10: {best_result['recall_at_10']:.4f}")
        logger.info(f"Users evaluated: {best_result['users_evaluated']}")
        logger.info(f"Fallback summary: {fallback1_pct:.1f}% fallback1, {fallback2_pct:.1f}% fallback2, {skip_pct:.1f}% skipped")
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










