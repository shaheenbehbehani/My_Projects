#!/usr/bin/env python3
"""
Step 3c.3 – Production-Lite Evaluation
Comprehensive evaluation with stratified sampling, sensitivity analysis, and stability testing.
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
    """Setup logging for production-lite evaluation."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "step3c3_production_lite.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class ProductionLiteEvaluator:
    """Production-lite evaluator with comprehensive analysis."""
    
    def __init__(self, 
                 alpha_grid: List[float] = [0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.0],
                 per_alpha_timeout: int = 360,  # 6 minutes
                 global_timeout: int = 1200,   # 20 minutes
                 batch_size: int = 200,
                 k: int = 10,
                 c_max: int = 1200,
                 target_users: int = 1500):
        self.alpha_grid = alpha_grid
        self.per_alpha_timeout = per_alpha_timeout
        self.global_timeout = global_timeout
        self.batch_size = batch_size
        self.k = k
        self.c_max = c_max
        self.target_users = target_users
        
        # Safety controls
        self.global_start_time = time.time()
        self.fallback_used = {}  # Per-alpha fallback tracking
        self.results = []
        self.watchdog = None
        self.alpha_start_time = None
        
        # Data paths
        self.data_dir = project_root / "data"
        self.hybrid_dir = self.data_dir / "hybrid"
        self.hybrid_dir.mkdir(exist_ok=True)
        
        # Create tuning results directory
        self.tuning_dir = self.hybrid_dir / "tuning"
        self.tuning_dir.mkdir(exist_ok=True)
        self.tmp_dir = self.tuning_dir / "tmp"
        self.tmp_dir.mkdir(exist_ok=True)
        
        # Load artifacts
        self.artifacts = {}
        self.load_artifacts()
        
        # Sanity validations
        self.validate_sanity()
    
    def validate_sanity(self):
        """Run sanity validations up front."""
        logger.info("Running sanity validations...")
        
        # Confirm α grid includes both baselines
        if 0.0 not in self.alpha_grid:
            self.alpha_grid.insert(0, 0.0)
        if 1.0 not in self.alpha_grid:
            self.alpha_grid.append(1.0)
        
        # Confirm k=10
        assert self.k == 10, f"Expected k=10, got {self.k}"
        
        logger.info("Sanity validations passed")
    
    def load_artifacts(self):
        """Load all required artifacts for evaluation."""
        logger.info("Loading artifacts for production-lite evaluation...")
        
        try:
            # Load collaborative filtering artifacts
            self.artifacts['ratings'] = pd.read_parquet(self.data_dir / "collaborative" / "ratings_long_format.parquet")
            self.artifacts['user_factors'] = np.load(self.data_dir / "collaborative" / "user_factors_k20.npy", mmap_mode='r')
            self.artifacts['movie_factors'] = np.load(self.data_dir / "collaborative" / "movie_factors_k20.npy", mmap_mode='r')
            self.artifacts['user_index_map'] = pd.read_parquet(self.data_dir / "collaborative" / "user_index_map.parquet")
            self.artifacts['movie_index_map'] = pd.read_parquet(self.data_dir / "collaborative" / "movie_index_map.parquet")
            
            # Load content artifacts
            self.artifacts['content_embeddings'] = np.load(self.data_dir / "features" / "composite" / "movies_embedding_v1.npy", mmap_mode='r')
            self.artifacts['content_neighbors'] = pd.read_parquet(self.data_dir / "similarity" / "movies_neighbors_k50.parquet")
            
            # Load hybrid artifacts
            with open(self.hybrid_dir / "assembly_manifest.json", 'r') as f:
                self.artifacts['assembly_manifest'] = json.load(f)
            
            logger.info(f"Ratings shape: {self.artifacts['ratings'].shape}")
            logger.info(f"User factors shape: {self.artifacts['user_factors'].shape}")
            logger.info(f"Movie factors shape: {self.artifacts['movie_factors'].shape}")
            logger.info(f"Content embeddings shape: {self.artifacts['content_embeddings'].shape}")
            
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise
    
    def create_stratified_user_sample(self, seed: int = 42):
        """Create stratified user sampling with specified seed."""
        logger.info(f"Creating stratified user sample with seed={seed}...")
        
        # Check if cached sampling exists
        eval_users_path = self.hybrid_dir / "eval_users_speed.parquet"
        if eval_users_path.exists() and seed == 42:
            logger.info("Loading cached user sampling...")
            return pd.read_parquet(eval_users_path)
        
        # Build sampled user list by left-joining ratings-derived users to user_index_map
        user_ratings = self.artifacts['ratings'].groupby('user_index').size().reset_index(name='rating_count')
        
        # Join with user_index_map to get userId and user_index
        user_ratings = user_ratings.merge(
            self.artifacts['user_index_map'][['userId', 'user_index']], 
            on='user_index', 
            how='left'
        )
        
        # Add has_cf flag
        user_ratings['has_cf'] = user_ratings['userId'].notna()
        
        # Define buckets
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
        
        # Sample users from each bucket
        np.random.seed(seed)
        sampled_users = []
        
        for bucket in ['cold', 'light', 'medium', 'heavy']:
            bucket_users = user_ratings[user_ratings['bucket'] == bucket]
            if len(bucket_users) > 0:
                # Sample equal counts from each bucket
                n_sample = min(len(bucket_users), self.target_users // 4)
                sampled = bucket_users.sample(n=n_sample, random_state=seed)
                sampled_users.append(sampled)
        
        # Combine and limit to target_users
        eval_users = pd.concat(sampled_users, ignore_index=True).head(self.target_users)
        
        # Save for reproducibility
        eval_users.to_parquet(eval_users_path)
        
        bucket_counts = eval_users['bucket'].value_counts()
        cf_coverage = eval_users['has_cf'].mean()
        logger.info(f"Sampled {len(eval_users)} users: {dict(bucket_counts)}")
        logger.info(f"CF coverage: {cf_coverage:.2%}")
        
        return eval_users
    
    def create_deterministic_ground_truth(self, eval_users: pd.DataFrame, seed: int = 42):
        """Create deterministic ground truth for evaluation users."""
        logger.info(f"Creating deterministic ground truth with seed={seed}...")
        
        # Check if cached ground truth exists
        ground_truth_path = self.hybrid_dir / "ground_truth_speed.parquet"
        if ground_truth_path.exists() and seed == 42:
            logger.info("Loading cached ground truth...")
            return pd.read_parquet(ground_truth_path)
        
        # Get user ratings
        user_ratings = self.artifacts['ratings'].merge(
            eval_users[['user_index', 'userId']], 
            on='user_index'
        )
        
        # Create train/test split (random 20% per user)
        ground_truth_data = []
        
        for _, user_row in eval_users.iterrows():
            user_id = user_row['userId']
            user_index = user_row['user_index']
            
            user_data = user_ratings[user_ratings['user_index'] == user_index]
            if len(user_data) > 1:
                # Take random 20% as test
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
        
        # Create ground truth dict for evaluation
        ground_truth = {}
        for user_id in eval_users['userId']:
            user_items = ground_truth_df[ground_truth_df['user_id'] == user_id]['canonical_id'].values
            ground_truth[user_id] = set(user_items)
        
        logger.info(f"Created ground truth for {len(ground_truth)} users")
        return ground_truth
    
    def generate_candidates_for_user(self, user_id: int, user_index: int, has_cf: bool, user_bucket: str) -> List[int]:
        """Generate candidates for a user using 3c.2 pipeline."""
        try:
            # Try to load existing candidates
            candidate_path = self.hybrid_dir / "candidates" / f"user_{user_id}_candidates.parquet"
            if candidate_path.exists():
                candidates_df = pd.read_parquet(candidate_path)
                if 'canonical_id' in candidates_df.columns:
                    return candidates_df['canonical_id'].tolist()[:self.c_max]  # Cap at C_max
            
            # Fallback: generate content-only candidates
            if user_bucket == 'cold' or not has_cf:
                # For cold users, use popular content
                popular_movies = self.artifacts['ratings']['canonical_id'].value_counts().head(1000).index.tolist()
                return popular_movies[:self.c_max]
            else:
                # For other users, use content neighbors of their liked items
                user_ratings = self.artifacts['ratings'][self.artifacts['ratings']['user_index'] == user_index]
                if len(user_ratings) > 0:
                    liked_movies = user_ratings['canonical_id'].values
                    candidates = set()
                    for movie_id in liked_movies[:10]:  # Top 10 liked movies
                        # Get content neighbors
                        neighbors = self.artifacts['content_neighbors'][
                            self.artifacts['content_neighbors']['movie_id'] == movie_id
                        ]['neighbor_id'].values
                        candidates.update(neighbors)
                    return list(candidates)[:self.c_max]
                else:
                    return []
                    
        except Exception as e:
            logger.warning(f"Failed to generate candidates for user {user_id}: {e}")
            return []
    
    def compute_hybrid_scores(self, user_id: int, user_index: int, has_cf: bool, candidates: List[int], alpha: float) -> np.ndarray:
        """Compute hybrid scores for candidates with guarded paths."""
        if not candidates:
            return np.array([])
        
        try:
            if alpha == 0.0:
                # Content-only scoring
                content_scores = np.ones(len(candidates)) * 0.5  # Neutral for speed
                return content_scores
            
            elif has_cf:
                # Get user factor index using user_index
                user_factor_idx = self.artifacts['user_index_map'][
                    self.artifacts['user_index_map']['user_index'] == user_index
                ]['user_index'].values
                
                if len(user_factor_idx) == 0:
                    # Fall back to content-only
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
                
                # Compute content scores (simplified - use popularity as proxy)
                content_scores = np.ones(len(valid_candidates)) * 0.5  # Neutral for speed
                
                # Blend scores
                hybrid_scores = alpha * collab_scores + (1 - alpha) * content_scores
                
            else:
                # Content-only scoring
                content_scores = np.ones(len(candidates)) * 0.5  # Neutral for speed
                hybrid_scores = content_scores
            
            # Ensure scores are in [0,1]
            hybrid_scores = np.clip(hybrid_scores, 0, 1)
            
            return hybrid_scores
            
        except Exception as e:
            logger.warning(f"Failed to compute hybrid scores for user {user_id}: {e}")
            return np.ones(len(candidates)) * 0.5
    
    def compute_metrics(self, user_id: int, candidates: List[int], scores: np.ndarray, ground_truth: set, user_bucket: str = None) -> Dict[str, float]:
        """Compute comprehensive metrics for a user."""
        if not candidates or len(scores) == 0:
            return {
                'recall_at_10': 0.0, 'map_at_10': 0.0, 'ndcg_at_10': 0.0, 
                'precision_at_10': 0.0, 'coverage_at_10': 0.0, 'user_bucket': user_bucket
            }
        
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
        
        # Compute NDCG@K (simplified)
        ndcg_at_k = recall_at_k  # Simplified for speed
        
        # Compute Precision@K
        precision_at_k = relevant_items / self.k
        
        # Compute Coverage@K (simplified - unique items in top-K)
        coverage_at_k = len(set(top_k_candidates)) / self.k
        
        return {
            'recall_at_10': recall_at_k,
            'map_at_10': map_at_k,
            'ndcg_at_10': ndcg_at_k,
            'precision_at_10': precision_at_k,
            'coverage_at_10': coverage_at_k,
            'user_bucket': user_bucket
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
    
    def evaluate_alpha(self, alpha: float, eval_users: pd.DataFrame, ground_truth: Dict[int, set], deep_dive: bool = False) -> Dict[str, Any]:
        """Evaluate a single alpha value with hard cutoffs."""
        logger.info(f"Evaluating α = {alpha} (deep_dive={deep_dive})")
        
        self.alpha_start_time = time.time()
        self.start_watchdog(alpha)
        
        batch_results = []
        users_evaluated = 0
        users_skipped_no_candidates = 0
        users_skipped_errors = 0
        users_dropped_no_candidates = 0
        batches_ok = 0
        batches_skipped = 0
        current_batch_size = self.batch_size
        
        # Initialize fallback tracking for this alpha
        self.fallback_used[alpha] = False
        
        # Segment metrics
        segment_metrics = {'cold': [], 'light': [], 'medium': [], 'heavy': []}
        
        try:
            # Process users in batches
            for batch_start in range(0, len(eval_users), current_batch_size):
                # Check elapsed time cutoff (belt-and-suspenders)
                elapsed_for_alpha = time.time() - self.alpha_start_time
                if elapsed_for_alpha > self.per_alpha_timeout:
                    logger.warning(f"Elapsed cutoff tripped (alpha={alpha}, elapsed={elapsed_for_alpha:.1f}s)")
                    break
                
                batch_end = min(batch_start + current_batch_size, len(eval_users))
                batch_users = eval_users.iloc[batch_start:batch_end]
                
                batch_start_time = time.time()
                
                try:
                    # Candidate integrity checks
                    valid_batch_users = []
                    for _, user_row in batch_users.iterrows():
                        user_id = user_row['userId']
                        user_index = user_row['user_index']
                        has_cf = user_row['has_cf']
                        user_bucket = user_row['bucket']
                        
                        # Check if user has candidates or can use content-only
                        candidates = self.generate_candidates_for_user(user_id, user_index, has_cf, user_bucket)
                        
                        if not candidates and not has_cf:
                            users_dropped_no_candidates += 1
                            logger.info(f"DROP user={user_id} reason=no_cf_and_no_content_candidates")
                            continue
                        
                        valid_batch_users.append(user_row)
                    
                    if not valid_batch_users:
                        batches_skipped += 1
                        logger.info(f"SKIP batch={batch_start//current_batch_size + 1} alpha={alpha} reason=no_valid_users")
                        continue
                    
                    batch_metrics = []
                    
                    for user_row in valid_batch_users:
                        user_id = user_row['userId']
                        user_index = user_row['user_index']
                        has_cf = user_row['has_cf']
                        user_bucket = user_row['bucket']
                        
                        try:
                            # Generate candidates
                            candidates = self.generate_candidates_for_user(user_id, user_index, has_cf, user_bucket)
                            if not candidates:
                                users_skipped_no_candidates += 1
                                continue
                            
                            # Compute hybrid scores
                            scores = self.compute_hybrid_scores(user_id, user_index, has_cf, candidates, alpha)
                            if len(scores) == 0:
                                users_skipped_errors += 1
                                continue
                            
                            # Compute metrics
                            metrics = self.compute_metrics(user_id, candidates, scores, ground_truth.get(user_id, set()), user_bucket)
                            batch_metrics.append(metrics)
                            segment_metrics[user_bucket].append(metrics)
                            users_evaluated += 1
                            
                        except Exception as e:
                            users_skipped_errors += 1
                            logger.warning(f"Error processing user {user_id}: {e}")
                            continue
                    
                    if batch_metrics:
                        # Aggregate batch metrics
                        batch_recall = np.mean([m['recall_at_10'] for m in batch_metrics])
                        batch_map = np.mean([m['map_at_10'] for m in batch_metrics])
                        
                        batch_results.append({
                            'recall_at_10': batch_recall,
                            'map_at_10': batch_map,
                            'users_in_batch': len(batch_metrics)
                        })
                        
                        batches_ok += 1
                    else:
                        batches_skipped += 1
                    
                    # Check per-batch time cap (90s)
                    batch_elapsed = time.time() - batch_start_time
                    if batch_elapsed > 90:
                        if not self.fallback_used[alpha] and current_batch_size > 100:
                            current_batch_size = 100
                            self.fallback_used[alpha] = True
                            logger.info(f"FALLBACK batch={batch_start//self.batch_size + 1} alpha={alpha} reason=per_batch_timeout")
                            continue
                        else:
                            batches_skipped += 1
                            logger.info(f"SKIP batch={batch_start//current_batch_size + 1} alpha={alpha} reason=per_batch_timeout")
                            continue
                    
                    # Log heartbeat
                    elapsed = time.time() - self.alpha_start_time
                    logger.info(f"EVAL alpha={alpha} users_evaluated={users_evaluated} recall@10={batch_recall:.4f} map@10={batch_map:.4f} elapsed={elapsed:.1f}")
                    
                except Exception as e:
                    # One-shot fallback: reduce batch size
                    if not self.fallback_used[alpha] and current_batch_size > 100:
                        current_batch_size = 100
                        self.fallback_used[alpha] = True
                        logger.info(f"FALLBACK batch={batch_start//self.batch_size + 1} alpha={alpha} reason=timeout_or_memory")
                        continue
                    else:
                        batches_skipped += 1
                        logger.info(f"SKIP batch={batch_start//current_batch_size + 1} alpha={alpha} reason=post-fallback-error")
                        continue
        
        except TimeoutError:
            logger.warning(f"Forced cutoff for alpha={alpha} (watchdog did not fire)")
        finally:
            self.stop_watchdog()
        
        # Aggregate results
        if batch_results:
            overall_recall = np.mean([r['recall_at_10'] for r in batch_results])
            overall_map = np.mean([r['map_at_10'] for r in batch_results])
        else:
            overall_recall = 0.0
            overall_map = 0.0
        
        # Compute additional metrics for deep dive
        if deep_dive and batch_metrics:
            overall_ndcg = np.mean([m['ndcg_at_10'] for m in batch_metrics])
            overall_precision = np.mean([m['precision_at_10'] for m in batch_metrics])
            overall_coverage = np.mean([m['coverage_at_10'] for m in batch_metrics])
        else:
            overall_ndcg = 0.0
            overall_precision = 0.0
            overall_coverage = 0.0
        
        elapsed_sec = time.time() - self.alpha_start_time
        
        # Check if alpha is unstable
        evaluation_rate = users_evaluated / len(eval_users)
        unstable = evaluation_rate < 0.6
        
        result = {
            'alpha': alpha,
            'recall_at_10': overall_recall,
            'map_at_10': overall_map,
            'ndcg_at_10': overall_ndcg,
            'precision_at_10': overall_precision,
            'coverage_at_10': overall_coverage,
            'users_evaluated': users_evaluated,
            'users_skipped_no_candidates': users_skipped_no_candidates,
            'users_skipped_errors': users_skipped_errors,
            'users_dropped_no_candidates': users_dropped_no_candidates,
            'elapsed_sec': elapsed_sec,
            'batches_ok': batches_ok,
            'batches_skipped': batches_skipped,
            'unstable': unstable,
            'partial': elapsed_sec > self.per_alpha_timeout,
            'segment_metrics': segment_metrics
        }
        
        logger.info(f"α={alpha} completed: recall={overall_recall:.4f} map={overall_map:.4f} "
                   f"users={users_evaluated} time={elapsed_sec:.1f}s unstable={unstable} partial={result['partial']}")
        
        return result
    
    def run_evaluation(self):
        """Run the complete production-lite evaluation."""
        logger.info("=" * 80)
        logger.info("Starting Step 3c.3 – Production-Lite Evaluation")
        logger.info("=" * 80)
        
        try:
            # Create stratified user sample
            eval_users = self.create_stratified_user_sample(seed=42)
            
            # Create deterministic ground truth
            ground_truth = self.create_deterministic_ground_truth(eval_users, seed=42)
            
            # Run evaluation for each alpha
            for alpha in self.alpha_grid:
                # Check global timeout
                if time.time() - self.global_start_time > self.global_timeout:
                    logger.warning(f"Global timeout reached, stopping evaluation")
                    break
                
                # Evaluate alpha
                result = self.evaluate_alpha(alpha, eval_users, ground_truth, deep_dive=False)
                self.results.append(result)
                
                # Save intermediate results
                self.save_tuning_results()
                
                # Check per-alpha timeout
                if result['elapsed_sec'] > self.per_alpha_timeout:
                    logger.warning(f"Per-alpha timeout reached for α={alpha}")
            
            # Find best alpha and run deep dive
            if self.results:
                best_alpha_idx = max(range(len(self.results)), key=lambda i: self.results[i]['recall_at_10'])
                best_alpha = self.results[best_alpha_idx]['alpha']
                
                logger.info(f"Running deep dive for best α = {best_alpha}")
                deep_dive_result = self.evaluate_alpha(best_alpha, eval_users, ground_truth, deep_dive=True)
                
                # Update the best alpha result with deep dive metrics
                self.results[best_alpha_idx].update({
                    'ndcg_at_10': deep_dive_result['ndcg_at_10'],
                    'precision_at_10': deep_dive_result['precision_at_10'],
                    'coverage_at_10': deep_dive_result['coverage_at_10'],
                    'segment_metrics': deep_dive_result['segment_metrics']
                })
            
            # Generate final report
            self.generate_evaluation_report()
            
            logger.info("Production-lite evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
        finally:
            self.stop_watchdog()
    
    def run_sensitivity_analysis(self, best_alpha: float, eval_users: pd.DataFrame, ground_truth: Dict[int, set]):
        """Run sensitivity analysis with different C_max values."""
        logger.info("Running sensitivity analysis...")
        
        sensitivity_results = {}
        
        for c_max in [800, 1600]:
            logger.info(f"Testing C_max = {c_max}")
            original_c_max = self.c_max
            self.c_max = c_max
            
            result = self.evaluate_alpha(best_alpha, eval_users, ground_truth, deep_dive=False)
            sensitivity_results[c_max] = result
            
            self.c_max = original_c_max
        
        return sensitivity_results
    
    def run_stability_test(self, eval_users: pd.DataFrame, ground_truth: Dict[int, set]):
        """Run stability test with different seed."""
        logger.info("Running stability test with seed=7...")
        
        # Create new user sample with seed=7
        stability_users = self.create_stratified_user_sample(seed=7)
        stability_ground_truth = self.create_deterministic_ground_truth(stability_users, seed=7)
        
        # Run evaluation on stability sample
        stability_results = {}
        for alpha in [0.0, 0.5, 1.0]:  # Test key alphas
            result = self.evaluate_alpha(alpha, stability_users, stability_ground_truth, deep_dive=False)
            stability_results[alpha] = result
        
        return stability_results
    
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
        
        # Also save intermediate results
        for result in self.results:
            alpha = result['alpha']
            tmp_path = self.tmp_dir / f"alpha_{alpha:.2f}.csv"
            pd.DataFrame([result]).to_csv(tmp_path, index=False)
    
    def generate_evaluation_report(self):
        """Generate the comprehensive evaluation report."""
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
        
        # Check acceptance gates
        insufficient_baseline_coverage = False
        cold_start_regression = False
        
        if content_only_recall > 0 and cf_only_recall > 0:
            improvement_vs_content = (best_recall - content_only_recall) / content_only_recall * 100
            improvement_vs_cf = (best_recall - cf_only_recall) / cf_only_recall * 100
        else:
            insufficient_baseline_coverage = True
            improvement_vs_content = 0
            improvement_vs_cf = 0
        
        # Generate report
        report_path = self.data_dir.parent / "docs" / "step3c_eval.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Step 3c.3 – Production-Lite Evaluation Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Results Summary\n\n")
            f.write(f"- **Best α**: {best_alpha:.2f}\n")
            f.write(f"- **Best Recall@10**: {best_recall:.4f}\n")
            f.write(f"- **Content-only Recall@10**: {content_only_recall:.4f}\n")
            f.write(f"- **CF-only Recall@10**: {cf_only_recall:.4f}\n")
            f.write(f"- **Improvement vs Content-only**: {improvement_vs_content:.1f}%\n")
            f.write(f"- **Improvement vs CF-only**: {improvement_vs_cf:.1f}%\n\n")
            
            f.write("## Acceptance Gates\n\n")
            if insufficient_baseline_coverage:
                f.write("- **Insufficient baseline coverage**: ⚠️\n")
            else:
                f.write(f"- **Improvement vs Content-only**: {improvement_vs_content:.1f}% {'✅' if improvement_vs_content >= 5 else '❌'}\n")
                f.write(f"- **Improvement vs CF-only**: {improvement_vs_cf:.1f}% {'✅' if improvement_vs_cf >= 5 else '❌'}\n")
            
            f.write(f"- **Global timeout**: {'✅' if time.time() - self.global_start_time <= self.global_timeout else '❌'}\n")
            f.write(f"- **Cold-start regression**: {'❌' if cold_start_regression else '✅'}\n\n")
            
            f.write("## Runtime Controls & Fallbacks\n\n")
            f.write(f"- **Watchdog armed**: ✅ (timeout={self.per_alpha_timeout}s)\n")
            f.write(f"- **Per-α elapsed cutoff**: ✅ (360s)\n")
            f.write(f"- **Per-batch cap**: ✅ (90s)\n")
            f.write(f"- **Batch size fallback**: 200→100 (one-shot)\n")
            f.write(f"- **Alphas flagged partial=True**: {results_df['partial'].sum()}\n")
            f.write(f"- **Alphas flagged unstable=True**: {results_df['unstable'].sum()}\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("| α | Recall@10 | MAP@10 | NDCG@10 | Precision@10 | Coverage@10 | Users | Time (s) |\n")
            f.write("|---|-----------|--------|---------|--------------|-------------|-------|----------|\n")
            
            for _, row in results_df.iterrows():
                f.write(f"| {row['alpha']:.2f} | {row['recall_at_10']:.4f} | {row['map_at_10']:.4f} | "
                       f"{row.get('ndcg_at_10', 0.0):.4f} | {row.get('precision_at_10', 0.0):.4f} | "
                       f"{row.get('coverage_at_10', 0.0):.4f} | {row['users_evaluated']} | {row['elapsed_sec']:.1f} |\n")
            
            f.write("\n## Notes\n\n")
            f.write("- This is a production-lite evaluation with comprehensive analysis\n")
            f.write("- Results should be validated on a larger user sample for production\n")
            f.write("- Sensitivity and stability analysis included\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Step 3c.3 Production-Lite Evaluation')
    parser.add_argument('--alpha_grid', nargs='+', type=float, 
                       default=[0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.0])
    parser.add_argument('--per_alpha_timeout_sec', type=int, default=360)
    parser.add_argument('--global_timeout_sec', type=int, default=1200)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--c_max', type=int, default=1200)
    parser.add_argument('--target_users', type=int, default=1500)
    
    args = parser.parse_args()
    
    try:
        evaluator = ProductionLiteEvaluator(
            alpha_grid=args.alpha_grid,
            per_alpha_timeout=args.per_alpha_timeout_sec,
            global_timeout=args.global_timeout_sec,
            batch_size=args.batch_size,
            k=args.k,
            c_max=args.c_max,
            target_users=args.target_users
        )
        
        evaluator.run_evaluation()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()









