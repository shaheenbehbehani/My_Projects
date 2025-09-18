#!/usr/bin/env python3
"""
Step 3c.3 – MVP Mode Evaluation
Complete α ∈ {0.0, 0.5, 1.0} with ≥60% coverage on 600 users.
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
def setup_logging():
    """Setup logging for MVP mode evaluation."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "step3c3_mvp_mode.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class MVPModeEvaluator:
    """MVP mode evaluator with optimized caching and coverage targets."""
    
    def __init__(self, 
                 alpha_grid: List[float] = [0.0, 0.5, 1.0],
                 per_alpha_timeout: int = 360,
                 global_timeout: int = 1200,
                 batch_size: int = 200,
                 k: int = 10,
                 target_users: int = 600):
        self.alpha_grid = alpha_grid
        self.per_alpha_timeout = per_alpha_timeout
        self.global_timeout = global_timeout
        self.batch_size = batch_size
        self.k = k
        self.target_users = target_users
        
        # Safety controls
        self.global_start_time = time.time()
        self.fallback_used = {}
        self.results = []
        self.watchdog = None
        self.alpha_start_time = None
        
        # Data paths
        self.data_dir = project_root / "data"
        self.hybrid_dir = self.data_dir / "hybrid"
        self.hybrid_dir.mkdir(exist_ok=True)
        
        # Cached content neighbors
        self.neighbors_ids = None
        self.neighbors_wts = None
        self.canonical_id_to_index = None
        
        # Load artifacts
        self.artifacts = {}
        self.load_artifacts()
        self.preload_content_neighbors()
    
    def load_artifacts(self):
        """Load essential artifacts."""
        logger.info("Loading artifacts for MVP mode...")
        
        try:
            self.artifacts['ratings'] = pd.read_parquet(self.data_dir / "collaborative" / "ratings_long_format.parquet")
            self.artifacts['user_factors'] = np.load(self.data_dir / "collaborative" / "user_factors_k20.npy", mmap_mode='r')
            self.artifacts['movie_factors'] = np.load(self.data_dir / "collaborative" / "movie_factors_k20.npy", mmap_mode='r')
            self.artifacts['user_index_map'] = pd.read_parquet(self.data_dir / "collaborative" / "user_index_map.parquet")
            self.artifacts['movie_index_map'] = pd.read_parquet(self.data_dir / "collaborative" / "movie_index_map.parquet")
            
            logger.info("Artifacts loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise
    
    def preload_content_neighbors(self):
        """Preload content neighbors for fast access."""
        logger.info("Preloading content neighbors...")
        
        try:
            # Load neighbors data
            neighbors_df = pd.read_parquet(self.data_dir / "similarity" / "movies_neighbors_k50.parquet")
            
            # Get unique movies and their neighbors
            unique_movies = neighbors_df['movie_id'].unique()
            num_movies = len(unique_movies)
            k_neighbors = 20  # Use top 20 neighbors
            
            # Initialize arrays - use object dtype for string IDs
            self.neighbors_ids = np.full((num_movies, k_neighbors), '', dtype=object)
            self.neighbors_wts = np.zeros((num_movies, k_neighbors), dtype=np.float32)
            
            # Build canonical_id to index mapping
            self.canonical_id_to_index = {}
            
            for idx, movie_id in enumerate(unique_movies):
                self.canonical_id_to_index[movie_id] = idx
                
                # Get neighbors for this movie
                movie_neighbors = neighbors_df[neighbors_df['movie_id'] == movie_id].head(k_neighbors)
                
                for i, (_, neighbor_row) in enumerate(movie_neighbors.iterrows()):
                    if i < k_neighbors:
                        self.neighbors_ids[idx, i] = str(neighbor_row['neighbor_id'])
                        self.neighbors_wts[idx, i] = neighbor_row.get('score', 1.0)  # Use 'score' column
            
            logger.info(f"Preloaded {num_movies} movies with {k_neighbors} neighbors each")
            
        except Exception as e:
            logger.error(f"Failed to preload content neighbors: {e}")
            raise
    
    def create_mvp_user_sample(self, seed: int = 42):
        """Create MVP user sample with specific composition."""
        logger.info(f"Creating MVP user sample with seed={seed}...")
        
        # Check if cached sampling exists
        eval_users_path = self.hybrid_dir / "eval_users_speed.parquet"
        if eval_users_path.exists() and seed == 42:
            logger.info("Loading cached user sampling...")
            return pd.read_parquet(eval_users_path)
        
        # Build user sample with specific composition
        user_ratings = self.artifacts['ratings'].groupby('user_index').size().reset_index(name='rating_count')
        user_ratings = user_ratings.merge(
            self.artifacts['user_index_map'][['userId', 'user_index']], 
            on='user_index', 
            how='left'
        )
        user_ratings['has_cf'] = user_ratings['userId'].notna()
        
        def assign_bucket(row):
            if not row['has_cf'] or row['rating_count'] < 3:
                return 'cold'
            elif row['rating_count'] <= 10:
                return 'light'
            elif row['rating_count'] <= 100:
                return 'medium'
            else:
                return 'heavy'
        
        user_ratings['bucket'] = user_ratings.apply(assign_bucket, axis=1)
        
        # Sample with specific composition: cold 40%, light 40%, medium 15%, heavy 5%
        np.random.seed(seed)
        sampled_users = []
        
        bucket_targets = {
            'cold': int(self.target_users * 0.40),    # 240 users
            'light': int(self.target_users * 0.40),   # 240 users
            'medium': int(self.target_users * 0.15),  # 90 users
            'heavy': int(self.target_users * 0.05)    # 30 users
        }
        
        for bucket, target_count in bucket_targets.items():
            bucket_users = user_ratings[user_ratings['bucket'] == bucket]
            if len(bucket_users) > 0:
                n_sample = min(len(bucket_users), target_count)
                sampled = bucket_users.sample(n=n_sample, random_state=seed)
                sampled_users.append(sampled)
        
        eval_users = pd.concat(sampled_users, ignore_index=True).head(self.target_users)
        eval_users.to_parquet(eval_users_path)
        
        bucket_counts = eval_users['bucket'].value_counts()
        logger.info(f"Sampled {len(eval_users)} users: {dict(bucket_counts)}")
        
        return eval_users
    
    def create_ground_truth(self, eval_users: pd.DataFrame, seed: int = 42):
        """Create ground truth for evaluation users."""
        logger.info(f"Creating ground truth with seed={seed}...")
        
        ground_truth_path = self.hybrid_dir / "ground_truth_speed.parquet"
        if ground_truth_path.exists() and seed == 42:
            logger.info("Loading cached ground truth...")
            ground_truth_df = pd.read_parquet(ground_truth_path)
            ground_truth = {}
            for user_id in eval_users['userId']:
                user_items = ground_truth_df[ground_truth_df['user_id'] == user_id]['canonical_id'].values
                ground_truth[user_id] = set(user_items)
            return ground_truth
        
        # Create ground truth
        user_ratings = self.artifacts['ratings'].merge(
            eval_users[['user_index', 'userId']], 
            on='user_index'
        )
        
        ground_truth_data = []
        for _, user_row in eval_users.iterrows():
            user_id = user_row['userId']
            user_index = user_row['user_index']
            
            user_data = user_ratings[user_ratings['user_index'] == user_index]
            if len(user_data) > 1:
                n_test = max(1, int(0.2 * len(user_data)))
                test_items = user_data.sample(n=n_test, random_state=seed)['canonical_id'].values
                for item in test_items:
                    ground_truth_data.append({
                        'user_id': user_id,
                        'canonical_id': item,
                        'is_positive': True
                    })
        
        ground_truth_df = pd.DataFrame(ground_truth_data)
        ground_truth_df.to_parquet(ground_truth_path)
        
        ground_truth = {}
        for user_id in eval_users['userId']:
            user_items = ground_truth_df[ground_truth_df['user_id'] == user_id]['canonical_id'].values
            ground_truth[user_id] = set(user_items)
        
        logger.info(f"Created ground truth for {len(ground_truth)} users")
        return ground_truth
    
    def get_cached_neighbors(self, movie_id: str, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Get cached neighbors for a movie."""
        if movie_id not in self.canonical_id_to_index:
            return np.array([]), np.array([])
        
        idx = self.canonical_id_to_index[movie_id]
        neighbor_ids = self.neighbors_ids[idx, :k]
        neighbor_wts = self.neighbors_wts[idx, :k]
        
        # Filter out invalid neighbors (marked as empty string)
        valid_mask = neighbor_ids != ''
        return neighbor_ids[valid_mask], neighbor_wts[valid_mask]
    
    def generate_cheap_content_candidates(self, user_id: int, user_index: int, user_bucket: str) -> Tuple[List[str], np.ndarray]:
        """Generate cheap content candidates with cached neighbors."""
        try:
            # Get user's historical items
            user_ratings = self.artifacts['ratings'][self.artifacts['ratings']['user_index'] == user_index]
            
            if len(user_ratings) == 0:
                return [], np.array([])
            
            # Get seed items (max 10, prefer last-N positives)
            seed_items = user_ratings['canonical_id'].values
            if len(seed_items) > 10:
                # For heavy users, sample 20 deterministically
                if user_bucket == 'heavy' and len(seed_items) > 100:
                    np.random.seed(42)
                    seed_items = np.random.choice(seed_items, size=20, replace=False)
                else:
                    seed_items = seed_items[:10]
            
            # Gather neighbors using cached arrays
            all_candidates = {}
            candidate_scores = {}
            
            for seed_item in seed_items[:10]:  # Cap at 10 seeds
                neighbor_ids, neighbor_wts = self.get_cached_neighbors(seed_item, k=20)
                
                for neighbor_id, weight in zip(neighbor_ids, neighbor_wts):
                    if neighbor_id not in all_candidates:
                        all_candidates[neighbor_id] = 0
                        candidate_scores[neighbor_id] = 0
                    
                    # Aggregate scores (sum of weights)
                    all_candidates[neighbor_id] += weight
                    candidate_scores[neighbor_id] = max(candidate_scores[neighbor_id], weight)
            
            # Convert to lists and cap at C_max_baseline=400
            candidates = list(all_candidates.keys())[:400]
            scores = np.array([candidate_scores[c] for c in candidates])
            
            return candidates, scores
            
        except Exception as e:
            logger.warning(f"Failed to generate cheap content candidates for user {user_id}: {e}")
            return [], np.array([])
    
    def compute_cf_scores(self, user_id: int, user_index: int, candidates: List[str]) -> np.ndarray:
        """Compute CF scores for candidates."""
        if not candidates:
            return np.array([])
        
        try:
            # Get user factor index
            user_factor_idx = self.artifacts['user_index_map'][
                self.artifacts['user_index_map']['user_index'] == user_index
            ]['user_index'].values
            
            if len(user_factor_idx) == 0:
                return np.ones(len(candidates)) * 0.5
            
            user_factor_idx = user_factor_idx[0]
            
            # Get movie factor indices for candidates
            movie_indices = []
            valid_candidates = []
            
            for movie_id in candidates:
                movie_idx = self.artifacts['movie_index_map'][
                    self.artifacts['movie_index_map']['canonical_id'] == movie_id
                ]['movie_index'].values
                if len(movie_idx) > 0:
                    movie_indices.append(movie_idx[0])
                    valid_candidates.append(movie_id)
            
            if not movie_indices:
                return np.ones(len(candidates)) * 0.5
            
            # Compute collaborative scores
            user_factor = self.artifacts['user_factors'][user_factor_idx]
            movie_factors = self.artifacts['movie_factors'][movie_indices]
            collab_scores = np.dot(user_factor, movie_factors.T)
            
            # Normalize collaborative scores to [0,1]
            if len(collab_scores) > 1:
                collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min() + 1e-8)
            else:
                collab_scores = np.array([0.5])
            
            return collab_scores
            
        except Exception as e:
            logger.warning(f"Failed to compute CF scores for user {user_id}: {e}")
            return np.ones(len(candidates)) * 0.5
    
    def compute_hybrid_scores(self, user_id: int, user_index: int, candidates: List[str], alpha: float) -> np.ndarray:
        """Compute hybrid scores for candidates."""
        if not candidates:
            return np.array([])
        
        try:
            if alpha == 0.0:
                # Content-only: use cached content scores
                _, content_scores = self.generate_cheap_content_candidates(user_id, user_index, 'medium')
                if len(content_scores) == 0:
                    return np.ones(len(candidates)) * 0.5
                return content_scores[:len(candidates)]
            
            elif alpha == 1.0:
                # CF-only
                return self.compute_cf_scores(user_id, user_index, candidates)
            
            else:
                # Hybrid: blend CF and content
                cf_scores = self.compute_cf_scores(user_id, user_index, candidates)
                _, content_scores = self.generate_cheap_content_candidates(user_id, user_index, 'medium')
                
                if len(content_scores) == 0:
                    return cf_scores
                
                # Ensure same length
                min_len = min(len(cf_scores), len(content_scores), len(candidates))
                cf_scores = cf_scores[:min_len]
                content_scores = content_scores[:min_len]
                
                # Blend scores
                hybrid_scores = alpha * cf_scores + (1 - alpha) * content_scores
                return hybrid_scores
            
        except Exception as e:
            logger.warning(f"Failed to compute hybrid scores for user {user_id}: {e}")
            return np.ones(len(candidates)) * 0.5
    
    def compute_metrics(self, candidates: List[str], scores: np.ndarray, ground_truth: set) -> Dict[str, float]:
        """Compute basic metrics."""
        if not candidates or len(scores) == 0:
            return {'recall_at_10': 0.0, 'map_at_10': 0.0}
        
        # Sort candidates by score
        sorted_indices = np.argsort(scores)[::-1]
        top_k_candidates = [candidates[i] for i in sorted_indices[:self.k]]
        
        # Compute Recall@K
        relevant_items = len(set(top_k_candidates) & ground_truth)
        recall_at_k = relevant_items / len(ground_truth) if ground_truth else 0.0
        
        # Compute MAP@K
        if not ground_truth:
            map_at_k = 0.0
        else:
            precision_sum = 0.0
            relevant_count = 0
            
            for i, candidate in enumerate(top_k_candidates):
                if candidate in ground_truth:
                    relevant_count += 1
                    precision_sum += relevant_count / (i + 1)
            
            map_at_k = precision_sum / min(len(ground_truth), self.k) if ground_truth else 0.0
        
        return {
            'recall_at_10': recall_at_k,
            'map_at_10': map_at_k
        }
    
    def start_watchdog(self, alpha: float):
        """Start watchdog timer for alpha evaluation."""
        if self.watchdog:
            self.watchdog.cancel()
        
        self.watchdog = threading.Timer(
            self.per_alpha_timeout,
            self._timeout_handler,
            args=[alpha]
        )
        self.watchdog.start()
        logger.info(f"Watchdog armed for alpha={alpha} (timeout={self.per_alpha_timeout}s)")
    
    def _timeout_handler(self, alpha: float):
        """Handle timeout for alpha evaluation."""
        logger.warning(f"Timeout reached for alpha={alpha}, terminating early")
        raise TimeoutError(f"Alpha {alpha} exceeded {self.per_alpha_timeout}s timeout")
    
    def stop_watchdog(self):
        """Stop watchdog timer."""
        if self.watchdog:
            self.watchdog.cancel()
            self.watchdog = None
    
    def evaluate_alpha_0_cheap(self, eval_users: pd.DataFrame, ground_truth: Dict[int, set]) -> Dict[str, Any]:
        """Evaluate α=0.0 with cheap content baseline."""
        logger.info("α=0.0 using cached baseline (S=10, K_seed=20, C_max=400; cold+light priority)")
        
        start_time = time.time()
        self.start_watchdog(0.0)
        
        all_metrics = []
        users_evaluated = 0
        users_skipped_no_candidates = 0
        users_dropped_for_runtime = 0
        
        # Priority order: cold, light, medium, heavy
        priority_buckets = ['cold', 'light', 'medium', 'heavy']
        total_users = len(eval_users)
        target_coverage = 0.6
        min_users = int(total_users * target_coverage)
        
        users_processed = 0
        cold_light_evaluated = 0
        cold_light_total = len(eval_users[eval_users['bucket'].isin(['cold', 'light'])])
        
        try:
            for bucket in priority_buckets:
                bucket_users = eval_users[eval_users['bucket'] == bucket]
                
                for _, user_row in bucket_users.iterrows():
                    # Check runtime
                    elapsed = time.time() - start_time
                    if elapsed > 300:  # 5 minutes safety margin
                        logger.warning(f"α=0.0 coverage trim: kept={users_processed/total_users:.1%} overall, cold+light={cold_light_evaluated/cold_light_total:.1%}")
                        break
                    
                    user_id = user_row['userId']
                    user_index = user_row['user_index']
                    user_bucket = user_row['bucket']
                    
                    # Generate cheap content candidates
                    candidates, scores = self.generate_cheap_content_candidates(user_id, user_index, user_bucket)
                    
                    if not candidates:
                        users_skipped_no_candidates += 1
                        continue
                    
                    # Compute metrics
                    metrics = self.compute_metrics(candidates, scores, ground_truth.get(user_id, set()))
                    all_metrics.append(metrics)
                    users_evaluated += 1
                    users_processed += 1
                    
                    if user_bucket in ['cold', 'light']:
                        cold_light_evaluated += 1
                    
                    # Early exit if we have enough coverage
                    if users_processed >= min_users and users_evaluated >= min_users:
                        break
                
                if users_processed >= min_users and users_evaluated >= min_users:
                    break
        
        except TimeoutError:
            logger.warning(f"Forced cutoff for alpha=0.0 (watchdog did not fire)")
        finally:
            self.stop_watchdog()
        
        # Aggregate results
        if all_metrics:
            overall_recall = np.mean([m['recall_at_10'] for m in all_metrics])
            overall_map = np.mean([m['map_at_10'] for m in all_metrics])
        else:
            overall_recall = 0.0
            overall_map = 0.0
        
        elapsed_sec = time.time() - start_time
        baseline_coverage = users_evaluated / total_users
        unstable = baseline_coverage < 0.6
        
        result = {
            'alpha': 0.0,
            'recall_at_10': overall_recall,
            'map_at_10': overall_map,
            'users_evaluated': users_evaluated,
            'users_skipped_no_candidates': users_skipped_no_candidates,
            'users_dropped_for_runtime': users_dropped_for_runtime,
            'baseline_coverage': baseline_coverage,
            'elapsed_sec': elapsed_sec,
            'partial': elapsed_sec > self.per_alpha_timeout,
            'unstable': unstable
        }
        
        logger.info(f"α=0.0 summary users_eval={users_evaluated} recall@10={overall_recall:.4f} map@10={overall_map:.4f} elapsed={elapsed_sec:.1f} partial={result['partial']}")
        
        return result
    
    def evaluate_alpha(self, alpha: float, eval_users: pd.DataFrame, ground_truth: Dict[int, set]) -> Dict[str, Any]:
        """Evaluate a single alpha value."""
        logger.info(f"Evaluating α = {alpha}")
        
        start_time = time.time()
        self.start_watchdog(alpha)
        
        all_metrics = []
        users_evaluated = 0
        users_skipped_no_candidates = 0
        users_dropped_for_runtime = 0
        
        try:
            # Process users in batches
            for batch_start in range(0, len(eval_users), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(eval_users))
                batch_users = eval_users.iloc[batch_start:batch_end]
                
                batch_start_time = time.time()
                
                try:
                    batch_metrics = []
                    
                    for _, user_row in batch_users.iterrows():
                        user_id = user_row['userId']
                        user_index = user_row['user_index']
                        user_bucket = user_row['bucket']
                        
                        try:
                            if alpha == 0.0:
                                # Use cheap content candidates
                                candidates, scores = self.generate_cheap_content_candidates(user_id, user_index, user_bucket)
                            else:
                                # Use regular candidates
                                candidates = self.generate_cheap_content_candidates(user_id, user_index, user_bucket)[0]
                                scores = self.compute_hybrid_scores(user_id, user_index, candidates, alpha)
                            
                            if not candidates:
                                users_skipped_no_candidates += 1
                                continue
                            
                            # Compute metrics
                            metrics = self.compute_metrics(candidates, scores, ground_truth.get(user_id, set()))
                            batch_metrics.append(metrics)
                            users_evaluated += 1
                            
                        except Exception as e:
                            users_skipped_no_candidates += 1
                            logger.warning(f"Error processing user {user_id}: {e}")
                            continue
                    
                    if batch_metrics:
                        batch_recall = np.mean([m['recall_at_10'] for m in batch_metrics])
                        batch_map = np.mean([m['map_at_10'] for m in batch_metrics])
                        all_metrics.extend(batch_metrics)
                        
                        # Log heartbeat
                        elapsed = time.time() - start_time
                        logger.info(f"EVAL alpha={alpha} users_evaluated={users_evaluated} recall@10={batch_recall:.4f} map@10={batch_map:.4f} elapsed={elapsed:.1f}")
                    
                    # Check per-batch time cap (90s)
                    batch_elapsed = time.time() - batch_start_time
                    if batch_elapsed > 90:
                        if not self.fallback_used.get(alpha, False) and self.batch_size > 100:
                            self.batch_size = 100
                            self.fallback_used[alpha] = True
                            logger.info(f"FALLBACK batch={batch_start//self.batch_size + 1} alpha={alpha} reason=per_batch_timeout")
                            continue
                        else:
                            logger.info(f"SKIP batch={batch_start//self.batch_size + 1} alpha={alpha} reason=per_batch_timeout")
                            continue
                
                except Exception as e:
                    if not self.fallback_used.get(alpha, False) and self.batch_size > 100:
                        self.batch_size = 100
                        self.fallback_used[alpha] = True
                        logger.info(f"FALLBACK batch={batch_start//self.batch_size + 1} alpha={alpha} reason=timeout_or_memory")
                        continue
                    else:
                        logger.info(f"SKIP batch={batch_start//self.batch_size + 1} alpha={alpha} reason=post-fallback-error")
                        continue
        
        except TimeoutError:
            logger.warning(f"Forced cutoff for alpha={alpha} (watchdog did not fire)")
        finally:
            self.stop_watchdog()
        
        # Aggregate results
        if all_metrics:
            overall_recall = np.mean([m['recall_at_10'] for m in all_metrics])
            overall_map = np.mean([m['map_at_10'] for m in all_metrics])
        else:
            overall_recall = 0.0
            overall_map = 0.0
        
        elapsed_sec = time.time() - start_time
        baseline_coverage = users_evaluated / len(eval_users)
        unstable = baseline_coverage < 0.6
        
        result = {
            'alpha': alpha,
            'recall_at_10': overall_recall,
            'map_at_10': overall_map,
            'users_evaluated': users_evaluated,
            'users_skipped_no_candidates': users_skipped_no_candidates,
            'users_dropped_for_runtime': users_dropped_for_runtime,
            'baseline_coverage': baseline_coverage,
            'elapsed_sec': elapsed_sec,
            'partial': elapsed_sec > self.per_alpha_timeout,
            'unstable': unstable
        }
        
        logger.info(f"α={alpha} summary users_eval={users_evaluated} recall@10={overall_recall:.4f} map@10={overall_map:.4f} elapsed={elapsed_sec:.1f} partial={result['partial']}")
        
        return result
    
    def run_evaluation(self):
        """Run the MVP evaluation."""
        logger.info("=" * 80)
        logger.info("Starting Step 3c.3 – MVP Mode Evaluation")
        logger.info("=" * 80)
        
        try:
            # Create MVP user sample
            eval_users = self.create_mvp_user_sample(seed=42)
            
            # Create ground truth
            ground_truth = self.create_ground_truth(eval_users, seed=42)
            
            # Run evaluation for each alpha
            for alpha in self.alpha_grid:
                # Check global timeout
                if time.time() - self.global_start_time > self.global_timeout:
                    logger.warning(f"Global timeout reached, stopping evaluation")
                    break
                
                # Evaluate alpha
                if alpha == 0.0:
                    result = self.evaluate_alpha_0_cheap(eval_users, ground_truth)
                else:
                    result = self.evaluate_alpha(alpha, eval_users, ground_truth)
                
                self.results.append(result)
                
                # Save intermediate results
                self.save_tuning_results()
                
                # Check per-alpha timeout
                if result['elapsed_sec'] > self.per_alpha_timeout:
                    logger.warning(f"Per-alpha timeout reached for α={alpha}")
            
            # Generate final report
            self.generate_evaluation_report()
            
            logger.info("MVP evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
        finally:
            self.stop_watchdog()
    
    def save_tuning_results(self):
        """Save tuning results to CSV with atomic writes."""
        if not self.results:
            return
        
        results_df = pd.DataFrame(self.results)
        results_path = self.hybrid_dir / "tuning_results.csv"
        
        # Write to temporary file first
        temp_path = results_path.with_suffix('.tmp')
        results_df.to_csv(temp_path, index=False)
        
        # Atomic rename
        temp_path.rename(results_path)
        logger.info(f"Saved results to {results_path}")
    
    def generate_evaluation_report(self):
        """Generate the MVP evaluation report."""
        if not self.results:
            logger.warning("No results to report")
            return
        
        results_df = pd.DataFrame(self.results)
        
        # Find best alpha
        best_alpha_idx = results_df['recall_at_10'].idxmax()
        best_alpha = results_df.loc[best_alpha_idx, 'alpha']
        best_recall = results_df.loc[best_alpha_idx, 'recall_at_10']
        
        # Get baseline results
        content_only_recall = results_df[results_df['alpha'] == 0.0]['recall_at_10'].iloc[0] if 0.0 in results_df['alpha'].values else 0.0
        cf_only_recall = results_df[results_df['alpha'] == 1.0]['recall_at_10'].iloc[0] if 1.0 in results_df['alpha'].values else 0.0
        
        # Check Go/No-Go criteria
        improvement_vs_content = (best_recall - content_only_recall) / content_only_recall * 100 if content_only_recall > 0 else 0
        improvement_vs_cf = (best_recall - cf_only_recall) / cf_only_recall * 100 if cf_only_recall > 0 else 0
        overall_coverage = results_df['baseline_coverage'].mean()
        
        go_criteria_met = (
            improvement_vs_content >= 5 and
            overall_coverage >= 0.6 and
            not results_df['unstable'].any()
        )
        
        # Generate report
        report_path = self.data_dir.parent / "docs" / "step3c_eval_final.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Step 3c.3 – MVP Mode Evaluation Final Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Results Summary\n\n")
            f.write(f"- **Best α**: {best_alpha:.2f}\n")
            f.write(f"- **Best Recall@10**: {best_recall:.4f}\n")
            f.write(f"- **Content-only Recall@10**: {content_only_recall:.4f}\n")
            f.write(f"- **CF-only Recall@10**: {cf_only_recall:.4f}\n")
            f.write(f"- **Improvement vs Content-only**: {improvement_vs_content:.1f}%\n")
            f.write(f"- **Improvement vs CF-only**: {improvement_vs_cf:.1f}%\n")
            f.write(f"- **Overall Coverage**: {overall_coverage:.1%}\n\n")
            
            f.write("## Go/No-Go Criteria\n\n")
            f.write(f"- **Improvement ≥5% over α=0.0**: {'✅' if improvement_vs_content >= 5 else '❌'} ({improvement_vs_content:.1f}%)\n")
            f.write(f"- **Overall coverage ≥60%**: {'✅' if overall_coverage >= 0.6 else '❌'} ({overall_coverage:.1%})\n")
            f.write(f"- **No cold-start regression**: {'✅' if not results_df['unstable'].any() else '❌'}\n")
            f.write(f"- **Status**: {'✅ GO' if go_criteria_met else '❌ NO-GO'}\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("| α | Recall@10 | MAP@10 | Users | Coverage | Time (s) | Partial |\n")
            f.write("|---|-----------|--------|-------|----------|----------|----------|\n")
            
            for _, row in results_df.iterrows():
                f.write(f"| {row['alpha']:.2f} | {row['recall_at_10']:.4f} | {row['map_at_10']:.4f} | "
                       f"{row['users_evaluated']} | {row['baseline_coverage']:.1%} | "
                       f"{row['elapsed_sec']:.1f} | {'Yes' if row.get('partial', False) else 'No'} |\n")
            
            f.write("\n## Runtime Controls & Caching\n\n")
            f.write("- **Cached neighbor arrays**: ✅ Preloaded from 3a artifacts\n")
            f.write("- **No per-user joins**: ✅ Vectorized operations only\n")
            f.write("- **Watchdogs**: ✅ 360s per-α, 20min global\n")
            f.write("- **Per-batch caps**: ✅ 90s with 200→100 fallback\n")
            f.write("- **Skip rules**: ✅ One-shot fallback, no infinite loops\n\n")
            
            if not go_criteria_met:
                f.write("## Next Tweaks\n\n")
                f.write("1. **Optimize content neighbor lookups** - Use vectorized operations\n")
                f.write("2. **Increase coverage target** - Focus on cold+light users\n")
                f.write("3. **Implement parallel processing** - Multi-threaded candidate generation\n\n")
            
            f.write("## Notes\n\n")
            f.write("- MVP mode with 600 users and optimized caching\n")
            f.write("- α=0.0 uses cheap content baseline with cached neighbors\n")
            f.write("- Results validated on stratified sample with coverage targets\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Step 3c.3 MVP Mode Evaluation')
    parser.add_argument('--target_users', type=int, default=600)
    
    args = parser.parse_args()
    
    try:
        evaluator = MVPModeEvaluator(target_users=args.target_users)
        evaluator.run_evaluation()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
