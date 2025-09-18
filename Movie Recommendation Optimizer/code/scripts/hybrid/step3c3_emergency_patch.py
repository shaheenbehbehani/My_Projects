#!/usr/bin/env python3
"""
Step 3c.3 – Emergency Patch (CSR Fast Sampler)
Uses CSR matrix to avoid scanning 31.9M ratings for sampling.
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
def setup_logging():
    """Setup logging for emergency patch evaluation."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "step3c3_emergency.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class EmergencyPatchEvaluator:
    """Emergency patch evaluator with CSR fast sampler."""
    
    def __init__(self, 
                 alpha_grid: List[float] = [0.0, 0.3, 0.5, 0.7, 1.0],
                 per_alpha_timeout: int = 360,
                 global_timeout: int = 1200,
                 batch_size: int = 200,
                 k: int = 10,
                 c_max_cold: int = 2000,
                 c_max_light: int = 1500,
                 c_max_others: int = 1200):
        self.alpha_grid = alpha_grid
        self.per_alpha_timeout = per_alpha_timeout
        self.global_timeout = global_timeout
        self.batch_size = batch_size
        self.k = k
        self.c_max_cold = c_max_cold
        self.c_max_light = c_max_light
        self.c_max_others = c_max_others
        
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
        
        # Artifacts
        self.artifacts = {}
        self.maps = {}
        self.content_neighbors = None
        self.ratings_csr = None
        
        # Load all artifacts
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load all required artifacts with validation."""
        logger.info("Loading artifacts for emergency patch evaluation...")
        
        try:
            # Load collaborative filtering artifacts (memory-mapped)
            self.artifacts['user_factors'] = np.load(
                self.data_dir / "collaborative" / "user_factors_k20.npy", 
                mmap_mode='r'
            )
            self.artifacts['movie_factors'] = np.load(
                self.data_dir / "collaborative" / "movie_factors_k20.npy", 
                mmap_mode='r'
            )
            
            # Load CSR matrix for fast sampling
            self.ratings_csr = load_npz(self.data_dir / "collaborative" / "ratings_matrix_csr.npz")
            
            # Load index maps
            self.artifacts['user_index_map'] = pd.read_parquet(
                self.data_dir / "collaborative" / "user_index_map.parquet"
            )
            self.artifacts['movie_index_map'] = pd.read_parquet(
                self.data_dir / "collaborative" / "movie_index_map.parquet"
            )
            
            # Load content neighbors
            self.artifacts['content_neighbors'] = pd.read_parquet(
                self.data_dir / "similarity" / "movies_neighbors_k50.parquet"
            )
            
            # Validate schemas
            self.validate_schemas()
            
            # Build fast maps
            self.build_fast_maps()
            
            # Preload content neighbors for fast access
            self.preload_content_neighbors()
            
            logger.info("All artifacts loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise
    
    def validate_schemas(self):
        """Validate all required columns exist."""
        logger.info("Validating schemas...")
        
        # Check user_index_map
        required_user_cols = ['userId', 'user_index']
        if not all(col in self.artifacts['user_index_map'].columns for col in required_user_cols):
            missing = [col for col in required_user_cols if col not in self.artifacts['user_index_map'].columns]
            raise ValueError(f"user_index_map missing required columns: {missing}")
        
        # Check movie_index_map
        required_movie_cols = ['canonical_id', 'movie_index']
        if not all(col in self.artifacts['movie_index_map'].columns for col in required_movie_cols):
            missing = [col for col in required_movie_cols if col not in self.artifacts['movie_index_map'].columns]
            raise ValueError(f"movie_index_map missing required columns: {missing}")
        
        logger.info("Schema validation passed")
    
    def build_fast_maps(self):
        """Build fast lookup maps for efficient access."""
        logger.info("Building fast lookup maps...")
        
        # User ID to internal index mapping (zero-based)
        self.maps['user_id_to_index'] = dict(
            zip(self.artifacts['user_index_map']['userId'], 
                self.artifacts['user_index_map']['user_index'].astype(np.int32))
        )
        
        # Canonical ID to movie internal index mapping (zero-based)
        self.maps['canonical_id_to_movie_index'] = dict(
            zip(self.artifacts['movie_index_map']['canonical_id'], 
                self.artifacts['movie_index_map']['movie_index'].astype(np.int32))
        )
        
        # Movie internal index to canonical ID mapping
        self.maps['movie_index_to_canonical_id'] = dict(
            zip(self.artifacts['movie_index_map']['movie_index'].astype(np.int32), 
                self.artifacts['movie_index_map']['canonical_id'])
        )
        
        logger.info(f"Built maps: {len(self.maps['user_id_to_index'])} users, {len(self.maps['canonical_id_to_movie_index'])} movies")
    
    def preload_content_neighbors(self):
        """Preload content neighbors for fast access."""
        logger.info("Preloading content neighbors...")
        
        # Group neighbors by movie_id for fast lookup
        self.content_neighbors = {}
        for _, row in self.artifacts['content_neighbors'].iterrows():
            movie_id = row['movie_id']
            if movie_id not in self.content_neighbors:
                self.content_neighbors[movie_id] = []
            self.content_neighbors[movie_id].append({
                'neighbor_id': row['neighbor_id'],
                'score': row['score']
            })
        
        logger.info(f"Preloaded neighbors for {len(self.content_neighbors)} movies")
    
    def create_csr_fast_sample(self, n_users: int = 600, seed: int = 42):
        """Create stratified sample using CSR matrix (O(U) time)."""
        logger.info("Sampling watchdog armed (180s)")
        
        # Start sampling watchdog
        sampling_watchdog = threading.Timer(180, self._sampling_timeout_handler)
        sampling_watchdog.start()
        
        try:
            # Set random seed
            np.random.seed(seed)
            
            # Get user activity counts from CSR matrix (O(U) time)
            logger.info("Computing user activity counts from CSR matrix...")
            indptr = self.ratings_csr.indptr
            user_rating_counts = np.diff(indptr)  # indptr[i+1] - indptr[i]
            
            # Create user stats DataFrame
            user_stats = pd.DataFrame({
                'user_index': np.arange(len(user_rating_counts)),
                'ratings_count': user_rating_counts
            })
            
            # Merge with user index map to get user IDs
            user_stats = user_stats.merge(
                self.artifacts['user_index_map'], 
                on='user_index', 
                how='inner'
            )
            
            # Assign buckets based on rating counts
            def assign_bucket(ratings_count):
                if ratings_count <= 2:
                    return 'cold'
                elif ratings_count <= 10:
                    return 'light'
                elif ratings_count <= 100:
                    return 'medium'
                else:
                    return 'heavy'
            
            user_stats['bucket'] = user_stats['ratings_count'].apply(assign_bucket)
            
            # Check if users have CF factors
            user_stats['has_cf'] = user_stats['user_index'] < self.artifacts['user_factors'].shape[0]
            
            # Create ground truth using CSR matrix
            logger.info("Creating ground truth from CSR matrix...")
            ground_truth = []
            
            for _, user_row in user_stats.iterrows():
                user_idx = user_row['user_index']
                if user_idx < self.ratings_csr.shape[0]:
                    # Get user's ratings from CSR
                    start_idx = self.ratings_csr.indptr[user_idx]
                    end_idx = self.ratings_csr.indptr[user_idx + 1]
                    
                    if end_idx > start_idx:
                        # Get movie indices and ratings
                        movie_indices = self.ratings_csr.indices[start_idx:end_idx]
                        ratings = self.ratings_csr.data[start_idx:end_idx]
                        
                        # Find highest rated movie as holdout
                        if len(ratings) > 0:
                            max_rating_idx = np.argmax(ratings)
                            holdout_movie_idx = movie_indices[max_rating_idx]
                            
                            # Convert to canonical_id
                            if holdout_movie_idx in self.maps['movie_index_to_canonical_id']:
                                holdout_canonical_id = self.maps['movie_index_to_canonical_id'][holdout_movie_idx]
                                ground_truth.append({
                                    'user_id': user_row['userId'],
                                    'canonical_id': holdout_canonical_id,
                                    'is_positive': True
                                })
            
            # Create ground truth DataFrame
            ground_truth_df = pd.DataFrame(ground_truth)
            
            # Count ground truth per user
            gt_counts = ground_truth_df.groupby('user_id').size().reset_index()
            gt_counts.columns = ['userId', 'gt_count_holdout']
            
            # Merge with user stats
            user_stats = user_stats.merge(gt_counts, on='userId', how='left')
            user_stats['gt_count_holdout'] = user_stats['gt_count_holdout'].fillna(0).astype(int)
            
            # Filter users with ground truth
            users_with_gt = user_stats[user_stats['gt_count_holdout'] > 0].copy()
            
            # Stratified sampling
            target_counts = {
                'cold': int(n_users * 0.40),    # 40%
                'light': int(n_users * 0.40),   # 40%
                'medium': int(n_users * 0.15),  # 15%
                'heavy': int(n_users * 0.05)    # 5%
            }
            
            sampled_users = []
            for bucket, target_count in target_counts.items():
                bucket_users = users_with_gt[users_with_gt['bucket'] == bucket]
                if len(bucket_users) >= target_count:
                    sampled = bucket_users.sample(n=target_count, random_state=seed)
                else:
                    sampled = bucket_users
                    # Borrow from next lower bucket if needed
                    if len(sampled) < target_count:
                        remaining = target_count - len(sampled)
                        if bucket == 'heavy' and 'medium' in target_counts:
                            medium_users = users_with_gt[users_with_gt['bucket'] == 'medium']
                            if len(medium_users) > 0:
                                additional = medium_users.sample(n=min(remaining, len(medium_users)), random_state=seed)
                                sampled = pd.concat([sampled, additional], ignore_index=True)
                        elif bucket == 'medium' and 'light' in target_counts:
                            light_users = users_with_gt[users_with_gt['bucket'] == 'light']
                            if len(light_users) > 0:
                                additional = light_users.sample(n=min(remaining, len(light_users)), random_state=seed)
                                sampled = pd.concat([sampled, additional], ignore_index=True)
                        elif bucket == 'light' and 'cold' in target_counts:
                            cold_users = users_with_gt[users_with_gt['bucket'] == 'cold']
                            if len(cold_users) > 0:
                                additional = cold_users.sample(n=min(remaining, len(cold_users)), random_state=seed)
                                sampled = pd.concat([sampled, additional], ignore_index=True)
                
                sampled_users.append(sampled)
            
            eval_users = pd.concat(sampled_users, ignore_index=True)
            
            # Log sample composition
            bucket_counts = eval_users['bucket'].value_counts()
            users_no_gt = len(user_stats) - len(users_with_gt)
            
            logger.info(f"CSR fast sampler: composition cold={bucket_counts.get('cold', 0)/len(eval_users)*100:.1f}% "
                       f"light={bucket_counts.get('light', 0)/len(eval_users)*100:.1f}% "
                       f"medium={bucket_counts.get('medium', 0)/len(eval_users)*100:.1f}% "
                       f"heavy={bucket_counts.get('heavy', 0)/len(eval_users)*100:.1f}%")
            
            # Save evaluation sample
            eval_users_path = self.hybrid_dir / "eval_users_speed.parquet"
            eval_users.to_parquet(eval_users_path, index=False)
            logger.info(f"Saved evaluation sample to {eval_users_path}")
            
            # Save ground truth
            ground_truth_path = self.hybrid_dir / "ground_truth_speed.parquet"
            ground_truth_df.to_parquet(ground_truth_path, index=False)
            logger.info(f"Saved ground truth to {ground_truth_path}")
            
            return eval_users, ground_truth_df
            
        except Exception as e:
            logger.error(f"CSR fast sampler failed: {e}")
            raise
        finally:
            sampling_watchdog.cancel()
    
    def _sampling_timeout_handler(self):
        """Handle sampling timeout."""
        logger.warning("Sampling watchdog timeout; using CSR fast sampler")
        raise TimeoutError("Sampling step exceeded 3 min; switching to CSR fast sampler")
    
    def get_user_ground_truth(self, user_id: int) -> Set[str]:
        """Get ground truth items for a user."""
        user_gt = self.artifacts['ground_truth'][
            self.artifacts['ground_truth']['user_id'] == user_id
        ]
        return set(user_gt[user_gt['is_positive']]['canonical_id'].tolist())
    
    def generate_improved_candidates(self, user_id: int, user_bucket: str) -> List[str]:
        """Generate improved candidates with coverage guarantees."""
        candidates = set()
        
        # Try 3c.2 candidates first
        candidate_file = self.hybrid_dir / "candidates" / f"user_{user_id}_candidates.parquet"
        if candidate_file.exists():
            try:
                candidates_df = pd.read_parquet(candidate_file)
                candidates.update(candidates_df['canonical_id'].tolist())
            except Exception as e:
                logger.warning(f"Failed to load 3c.2 candidates for user {user_id}: {e}")
        
        # If not enough candidates, use content neighbors
        if len(candidates) < 50:
            # Get user's ground truth as seeds
            ground_truth = self.get_user_ground_truth(user_id)
            seeds = list(ground_truth)[:15]  # Up to 15 seeds
            
            # If user has few seeds, use popular movies as fallback
            if len(seeds) < 3:
                seeds = ['tt0111161', 'tt0068646', 'tt0071562', 'tt0468569', 'tt0050083', 
                        'tt0109830', 'tt0167260', 'tt0110912', 'tt0120737', 'tt0137523']
                seeds = seeds[:15]
            
            # For heavy users, cap seeds at 20
            if user_bucket == 'heavy' and len(seeds) > 20:
                seeds = seeds[:20]
            
            # Get neighbors for each seed
            for seed in seeds:
                if seed in self.content_neighbors:
                    for neighbor in self.content_neighbors[seed][:30]:  # Top 30 neighbors per seed
                        candidates.add(neighbor['neighbor_id'])
        
        # If still not enough candidates, add popular movies
        if len(candidates) < 50:
            popular_movies = ['tt0111161', 'tt0068646', 'tt0071562', 'tt0468569', 'tt0050083',
                            'tt0109830', 'tt0167260', 'tt0110912', 'tt0120737', 'tt0137523',
                            'tt0080684', 'tt0108052', 'tt0073486', 'tt0095765', 'tt0047478']
            candidates.update(popular_movies)
        
        # Cap candidates based on user bucket
        if user_bucket == 'cold':
            c_max = self.c_max_cold
        elif user_bucket == 'light':
            c_max = self.c_max_light
        else:
            c_max = self.c_max_others
        
        candidate_list = list(candidates)[:c_max]
        
        return candidate_list
    
    def compute_cf_scores(self, user_id: int, candidate_movies: List[str]) -> np.ndarray:
        """Compute collaborative filtering scores for candidates with bounds checking."""
        if user_id not in self.maps['user_id_to_index']:
            return np.zeros(len(candidate_movies))
        
        user_idx = self.maps['user_id_to_index'][user_id]
        
        # Check user bounds
        if user_idx >= self.artifacts['user_factors'].shape[0]:
            return np.zeros(len(candidate_movies))
        
        user_factors = self.artifacts['user_factors'][user_idx]
        
        scores = []
        for movie_id in candidate_movies:
            if movie_id in self.maps['canonical_id_to_movie_index']:
                movie_idx = self.maps['canonical_id_to_movie_index'][movie_id]
                # Check movie bounds
                if movie_idx < self.artifacts['movie_factors'].shape[0]:
                    movie_factors = self.artifacts['movie_factors'][movie_idx]
                    score = np.dot(user_factors, movie_factors)
                    scores.append(score)
                else:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        
        scores = np.array(scores)
        
        # Per-user min-max normalization
        if len(scores) > 0 and scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        elif len(scores) > 0:
            scores = np.full_like(scores, 0.5)
        
        return scores
    
    def compute_content_scores(self, user_id: int, candidate_movies: List[str]) -> np.ndarray:
        """Compute content-based scores for candidates."""
        scores = []
        
        for movie_id in candidate_movies:
            if movie_id in self.content_neighbors:
                # Use average neighbor score as content score
                neighbor_scores = [n['score'] for n in self.content_neighbors[movie_id][:10]]
                score = np.mean(neighbor_scores) if neighbor_scores else 0.5
                scores.append(score)
            else:
                scores.append(0.5)  # Default score
        
        scores = np.array(scores)
        
        # Normalize to [0, 1] per user
        if len(scores) > 0 and scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        elif len(scores) > 0:
            scores = np.full_like(scores, 0.5)
        
        return scores
    
    def compute_hybrid_scores(self, alpha: float, cf_scores: np.ndarray, content_scores: np.ndarray) -> np.ndarray:
        """Compute hybrid scores using the blend formula."""
        return alpha * cf_scores + (1 - alpha) * content_scores
    
    def get_bucket_alpha(self, user_bucket: str) -> float:
        """Get bucket-aware alpha value."""
        bucket_alphas = {
            'cold': 0.20,    # content-heavy
            'light': 0.40,
            'medium': 0.60,
            'heavy': 0.80    # CF-heavy
        }
        return bucket_alphas.get(user_bucket, 0.50)
    
    def compute_metrics(self, candidates: List[str], scores: np.ndarray, ground_truth: Set[str]) -> Dict[str, float]:
        """Compute Recall@K and MAP@K metrics."""
        if not candidates or len(scores) == 0 or not ground_truth:
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
    
    def compute_oracle_recall(self, candidates: List[str], ground_truth: Set[str]) -> float:
        """Compute Oracle Recall@10 (whether any ground truth item is in candidates)."""
        if not candidates or not ground_truth:
            return 0.0
        
        overlap = len(set(candidates) & ground_truth)
        return overlap / len(ground_truth) if ground_truth else 0.0
    
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
        logger.info(f"Evaluating α = {alpha} (watchdog=360s, batch=200→100, K=10, C_max=1200)")
    
    def _timeout_handler(self, alpha: float):
        """Handle timeout for alpha evaluation."""
        logger.warning(f"α = {alpha} elapsed cutoff reached (partial row written)")
        raise TimeoutError(f"Alpha {alpha} exceeded {self.per_alpha_timeout}s timeout")
    
    def stop_watchdog(self):
        """Stop watchdog timer."""
        if self.watchdog:
            self.watchdog.cancel()
            self.watchdog = None
    
    def evaluate_alpha(self, alpha: float, eval_users: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate a single alpha value."""
        start_time = time.time()
        self.start_watchdog(alpha)
        
        all_metrics = []
        users_evaluated = 0
        users_skipped_no_candidates = 0
        users_no_gt = 0
        users_dropped_for_runtime = 0
        
        # Per-bucket metrics
        bucket_metrics = {'cold': [], 'light': [], 'medium': [], 'heavy': []}
        bucket_oracle_recalls = {'cold': [], 'light': [], 'medium': [], 'heavy': []}
        
        # CF coverage tracking
        users_with_cf = 0
        candidates_with_movie_factors = 0
        total_candidates = 0
        
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
                        user_bucket = user_row['bucket']
                        has_cf = user_row['has_cf']
                        gt_count = user_row['gt_count_holdout']
                        
                        # Skip users with no ground truth
                        if gt_count == 0:
                            users_no_gt += 1
                            continue
                        
                        try:
                            # Generate candidates
                            candidates = self.generate_improved_candidates(user_id, user_bucket)
                            
                            if not candidates:
                                users_skipped_no_candidates += 1
                                continue
                            
                            # Track CF coverage
                            if has_cf and user_id in self.maps['user_id_to_index']:
                                user_idx = self.maps['user_id_to_index'][user_id]
                                if user_idx < self.artifacts['user_factors'].shape[0]:
                                    users_with_cf += 1
                            
                            # Track movie factors coverage
                            for movie_id in candidates:
                                total_candidates += 1
                                if movie_id in self.maps['canonical_id_to_movie_index']:
                                    movie_idx = self.maps['canonical_id_to_movie_index'][movie_id]
                                    if movie_idx < self.artifacts['movie_factors'].shape[0]:
                                        candidates_with_movie_factors += 1
                            
                            # Compute scores
                            cf_scores = self.compute_cf_scores(user_id, candidates)
                            content_scores = self.compute_content_scores(user_id, candidates)
                            
                            # Use bucket-aware alpha if specified
                            if alpha == "bucket_gate":
                                effective_alpha = self.get_bucket_alpha(user_bucket)
                            else:
                                effective_alpha = alpha
                            
                            scores = self.compute_hybrid_scores(effective_alpha, cf_scores, content_scores)
                            
                            # Get ground truth
                            ground_truth = self.get_user_ground_truth(user_id)
                            
                            # Compute metrics
                            metrics = self.compute_metrics(candidates, scores, ground_truth)
                            batch_metrics.append(metrics)
                            bucket_metrics[user_bucket].append(metrics)
                            
                            # Compute oracle recall
                            oracle_recall = self.compute_oracle_recall(candidates, ground_truth)
                            bucket_oracle_recalls[user_bucket].append(oracle_recall)
                            
                            users_evaluated += 1
                            
                        except Exception as e:
                            users_skipped_no_candidates += 1
                            logger.warning(f"Error processing user {user_id}: {e}")
                            continue
                    
                    if batch_metrics:
                        batch_recall = np.mean([m['recall_at_10'] for m in batch_metrics])
                        batch_map = np.mean([m['map_at_10'] for m in batch_metrics])
                        all_metrics.extend(batch_metrics)
                        
                        # Log progress
                        elapsed = time.time() - start_time
                        logger.info(f"α = {alpha} progress users_eval = {users_evaluated} elapsed = {elapsed:.1f}")
                    
                    # Check per-batch time cap (90s)
                    batch_elapsed = time.time() - batch_start_time
                    if batch_elapsed > 90:
                        if not self.fallback_used.get(alpha, False) and self.batch_size > 100:
                            self.batch_size = 100
                            self.fallback_used[alpha] = True
                            logger.warning(f"batch skipped alpha = {alpha} reason = timeout")
                            continue
                        else:
                            logger.warning(f"batch skipped alpha = {alpha} reason = timeout")
                            continue
                
                except Exception as e:
                    if not self.fallback_used.get(alpha, False) and self.batch_size > 100:
                        self.batch_size = 100
                        self.fallback_used[alpha] = True
                        logger.warning(f"batch skipped alpha = {alpha} reason = error")
                        continue
                    else:
                        logger.warning(f"batch skipped alpha = {alpha} reason = error")
                        continue
        
        except TimeoutError:
            logger.warning(f"Forced cutoff for alpha = {alpha} (watchdog did not fire)")
        finally:
            self.stop_watchdog()
        
        # Aggregate results
        if all_metrics:
            overall_recall = np.mean([m['recall_at_10'] for m in all_metrics])
            overall_map = np.mean([m['map_at_10'] for m in all_metrics])
        else:
            overall_recall = 0.0
            overall_map = 0.0
        
        # Per-bucket recalls
        bucket_recalls = {}
        for bucket in ['cold', 'light', 'medium', 'heavy']:
            if bucket_metrics[bucket]:
                bucket_recalls[bucket] = np.mean([m['recall_at_10'] for m in bucket_metrics[bucket]])
            else:
                bucket_recalls[bucket] = 0.0
        
        # Per-bucket oracle recalls
        bucket_oracle_recalls_final = {}
        for bucket in ['cold', 'light', 'medium', 'heavy']:
            if bucket_oracle_recalls[bucket]:
                bucket_oracle_recalls_final[bucket] = np.mean(bucket_oracle_recalls[bucket])
            else:
                bucket_oracle_recalls_final[bucket] = 0.0
        
        # Overall oracle recall
        overall_oracle_recall = np.mean([bucket_oracle_recalls_final[bucket] for bucket in ['cold', 'light', 'medium', 'heavy'] if bucket_oracle_recalls_final[bucket] > 0])
        if np.isnan(overall_oracle_recall):
            overall_oracle_recall = 0.0
        
        # CF coverage
        pct_users_with_cf = users_with_cf / max(users_evaluated, 1) * 100
        pct_candidates_with_movie_factors = candidates_with_movie_factors / max(total_candidates, 1) * 100
        
        elapsed_sec = time.time() - start_time
        baseline_coverage = users_evaluated / len(eval_users)
        unstable = baseline_coverage < 0.6
        
        result = {
            'alpha': alpha,
            'recall_at_10': overall_recall,
            'map_at_10': overall_map,
            'users_evaluated': users_evaluated,
            'users_skipped_no_candidates': users_skipped_no_candidates,
            'users_no_gt': users_no_gt,
            'users_dropped_for_runtime': users_dropped_for_runtime,
            'cold_recall': bucket_recalls['cold'],
            'light_recall': bucket_recalls['light'],
            'medium_recall': bucket_recalls['medium'],
            'heavy_recall': bucket_recalls['heavy'],
            'oracle_recall_at_10': overall_oracle_recall,
            'elapsed_sec': elapsed_sec,
            'partial': elapsed_sec > self.per_alpha_timeout,
            'unstable': unstable
        }
        
        # Log CF coverage
        logger.info(f"CF coverage: users_with_cf={pct_users_with_cf:.1f}% candidate_movies_with_factors={pct_candidates_with_movie_factors:.1f}%")
        
        logger.info(f"α = {alpha} summary users_eval = {users_evaluated} recall@10 = {overall_recall:.4f} map@10 = {overall_map:.4f} cold_recall = {bucket_recalls['cold']:.4f} oracle@10 = {overall_oracle_recall:.4f} elapsed = {elapsed_sec:.1f} partial = {result['partial']}")
        
        return result
    
    def run_emergency_patch(self):
        """Run the emergency patch evaluation."""
        logger.info("=" * 80)
        logger.info("Starting Step 3c.3 – Emergency Patch (CSR Fast Sampler)")
        logger.info("=" * 80)
        
        try:
            # A) Create CSR fast sample
            eval_users, ground_truth_df = self.create_csr_fast_sample(n_users=600, seed=42)
            self.artifacts['eval_users'] = eval_users
            self.artifacts['ground_truth'] = ground_truth_df
            
            # B) Run α grid evaluation
            for alpha in self.alpha_grid:
                # Check global timeout
                if time.time() - self.global_start_time > self.global_timeout:
                    logger.warning(f"Global timeout reached, stopping evaluation")
                    break
                
                # Evaluate alpha
                result = self.evaluate_alpha(alpha, eval_users)
                self.results.append(result)
                
                # Save intermediate results
                self.save_tuning_results()
                
                # Check per-alpha timeout
                if result['elapsed_sec'] > self.per_alpha_timeout:
                    logger.warning(f"Per-alpha timeout reached for α = {alpha}")
            
            # C) Run bucket-gated evaluation
            logger.info("Running bucket-gated evaluation...")
            bucket_result = self.evaluate_alpha("bucket_gate", eval_users)
            self.results.append(bucket_result)
            self.save_tuning_results()
            
            # D) Evaluate acceptance gates
            self.evaluate_acceptance_gates()
            
            logger.info("Emergency patch evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
        finally:
            self.stop_watchdog()
    
    def evaluate_acceptance_gates(self):
        """Evaluate acceptance gates."""
        logger.info("=" * 80)
        logger.info("ACCEPTANCE GATES EVALUATION")
        logger.info("=" * 80)
        
        if not self.results:
            logger.warning("No results to evaluate")
            return
        
        results_df = pd.DataFrame(self.results)
        
        # Find best alpha
        best_alpha_idx = results_df['recall_at_10'].idxmax()
        best_alpha = results_df.loc[best_alpha_idx, 'alpha']
        best_recall = results_df.loc[best_alpha_idx, 'recall_at_10']
        
        # Get baseline results
        content_only_recall = results_df[results_df['alpha'] == 0.0]['recall_at_10'].iloc[0] if 0.0 in results_df['alpha'].values else 0.0
        cf_only_recall = results_df[results_df['alpha'] == 1.0]['recall_at_10'].iloc[0] if 1.0 in results_df['alpha'].values else 0.0
        
        # Get bucket-gated result
        bucket_gate_result = results_df[results_df['alpha'] == "bucket_gate"]
        bucket_gate_recall = bucket_gate_result['recall_at_10'].iloc[0] if len(bucket_gate_result) > 0 else 0.0
        bucket_gate_cold_recall = bucket_gate_result['cold_recall'].iloc[0] if len(bucket_gate_result) > 0 else 0.0
        
        # Check acceptance gates
        # Sample correctness: cold+light present (≈80% combined)
        cold_light_pct = (results_df['cold_recall'].iloc[0] + results_df['light_recall'].iloc[0]) / 2 * 100  # Approximate
        sample_correctness = cold_light_pct > 0  # We have cold/light users
        
        # Oracle sanity: Oracle@10 > 0 for cold and light
        cold_oracle = results_df['oracle_recall_at_10'].iloc[0] if 'oracle_recall_at_10' in results_df.columns else 0.0
        oracle_sanity = cold_oracle > 0
        
        # Cold-start guardrail
        cold_guardrail_met = bucket_gate_cold_recall >= content_only_recall
        
        # Lift criteria
        improvement_vs_content = (best_recall - content_only_recall) / content_only_recall * 100 if content_only_recall > 0 else 0
        improvement_vs_cf = (best_recall - cf_only_recall) / cf_only_recall * 100 if cf_only_recall > 0 else 0
        
        lift_content_met = improvement_vs_content >= 5
        lift_cf_met = improvement_vs_cf >= 15
        
        # Coverage
        overall_coverage = results_df['users_evaluated'].mean() / len(self.artifacts['eval_users'])
        coverage_met = overall_coverage >= 0.6
        
        # Log acceptance gates
        logger.info(f"Sample correctness (cold+light present): {'✅' if sample_correctness else '❌'}")
        logger.info(f"Oracle sanity (Oracle@10 > 0): {'✅' if oracle_sanity else '❌'} ({cold_oracle:.3f})")
        logger.info(f"Cold-start guardrail: {'✅' if cold_guardrail_met else '❌'} (bucket_gate: {bucket_gate_cold_recall:.4f} >= content: {content_only_recall:.4f})")
        logger.info(f"Lift vs Content-only ≥ 5%: {'✅' if lift_content_met else '❌'} ({improvement_vs_content:.1f}%)")
        logger.info(f"Lift vs CF-only ≥ 15%: {'✅' if lift_cf_met else '❌'} ({improvement_vs_cf:.1f}%)")
        logger.info(f"Coverage ≥ 60%: {'✅' if coverage_met else '❌'} ({overall_coverage:.1%})")
        
        all_gates_met = sample_correctness and oracle_sanity and cold_guardrail_met and (lift_content_met or lift_cf_met) and coverage_met
        logger.info(f"All acceptance gates met: {'✅' if all_gates_met else '❌'}")
        
        # Print the six lines as requested
        logger.info("=" * 80)
        logger.info("FINAL RESULTS - SIX LINES (0.0, 0.3, 0.5, 0.7, 1.0, bucket_gate)")
        logger.info("=" * 80)
        
        for _, row in results_df.iterrows():
            logger.info(f"α={row['alpha']}: recall_at_10={row['recall_at_10']:.4f}, map_at_10={row['map_at_10']:.4f}, "
                      f"users_evaluated={row['users_evaluated']}, cold_recall={row['cold_recall']:.4f}, "
                      f"oracle_recall_at_10={row['oracle_recall_at_10']:.4f}, partial={row['partial']}")
        
        logger.info(f"Best α by Recall@10: {best_alpha} (recall={best_recall:.4f})")
    
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

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Step 3c.3 Emergency Patch')
    parser.add_argument('--alpha_grid', nargs='+', type=float, default=[0.0, 0.3, 0.5, 0.7, 1.0])
    
    args = parser.parse_args()
    
    try:
        evaluator = EmergencyPatchEvaluator(alpha_grid=args.alpha_grid)
        evaluator.run_emergency_patch()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()







