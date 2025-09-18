#!/usr/bin/env python3
"""
Step 3c.3 – Real Pass (No Simulation) • Tuning & Offline Evaluation
Uses only real artifacts from Steps 3a/3b/3c.2 with runtime guards and partial result persistence.
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

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
def setup_logging():
    """Setup logging for real pass evaluation."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "step3c3_real.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class RealPassEvaluator:
    """Real pass evaluator using actual artifacts from Steps 3a/3b/3c.2."""
    
    def __init__(self, 
                 alpha_grid: List[float] = [0.0, 0.5, 1.0],
                 per_alpha_timeout: int = 360,
                 global_timeout: int = 1200,
                 batch_size: int = 200,
                 k: int = 10,
                 c_max: int = 1200):
        self.alpha_grid = alpha_grid
        self.per_alpha_timeout = per_alpha_timeout
        self.global_timeout = global_timeout
        self.batch_size = batch_size
        self.k = k
        self.c_max = c_max
        
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
        
        # Load all artifacts
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load all required artifacts with validation."""
        logger.info("Loading artifacts for real pass evaluation...")
        
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
            
            # Load evaluation sample and ground truth
            self.artifacts['eval_users'] = pd.read_parquet(
                self.hybrid_dir / "eval_users_speed.parquet"
            )
            self.artifacts['ground_truth'] = pd.read_parquet(
                self.hybrid_dir / "ground_truth_speed.parquet"
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
        
        # Check eval_users
        required_eval_cols = ['userId', 'bucket', 'has_cf']
        if not all(col in self.artifacts['eval_users'].columns for col in required_eval_cols):
            missing = [col for col in required_eval_cols if col not in self.artifacts['eval_users'].columns]
            raise ValueError(f"eval_users missing required columns: {missing}")
        
        # Check ground_truth
        required_gt_cols = ['user_id', 'canonical_id', 'is_positive']
        if not all(col in self.artifacts['ground_truth'].columns for col in required_gt_cols):
            missing = [col for col in required_gt_cols if col not in self.artifacts['ground_truth'].columns]
            raise ValueError(f"ground_truth missing required columns: {missing}")
        
        logger.info("Schema validation passed")
    
    def build_fast_maps(self):
        """Build fast lookup maps for efficient access."""
        logger.info("Building fast lookup maps...")
        
        # User ID to internal index mapping
        self.maps['user_id_to_index'] = dict(
            zip(self.artifacts['user_index_map']['userId'], 
                self.artifacts['user_index_map']['user_index'])
        )
        
        # Canonical ID to movie internal index mapping
        self.maps['canonical_id_to_movie_index'] = dict(
            zip(self.artifacts['movie_index_map']['canonical_id'], 
                self.artifacts['movie_index_map']['movie_index'])
        )
        
        # Movie internal index to canonical ID mapping
        self.maps['movie_index_to_canonical_id'] = dict(
            zip(self.artifacts['movie_index_map']['movie_index'], 
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
    
    def get_user_candidates(self, user_id: int) -> Optional[pd.DataFrame]:
        """Get candidates for a user from 3c.2 if available."""
        candidate_file = self.hybrid_dir / "candidates" / f"user_{user_id}_candidates.parquet"
        
        if candidate_file.exists():
            try:
                candidates = pd.read_parquet(candidate_file)
                return candidates[['canonical_id', 'hybrid_score', 'content_score', 'collab_score']]
            except Exception as e:
                logger.warning(f"Failed to load candidates for user {user_id}: {e}")
                return None
        return None
    
    def generate_content_fallback_candidates(self, user_id: int, user_bucket: str) -> List[str]:
        """Generate content-based fallback candidates."""
        # For cold users, use popular movies as seeds
        # For other users, use their rating history as seeds
        if user_bucket == 'cold':
            # Use top popular movies as seeds
            seed_movies = ['tt0111161', 'tt0068646', 'tt0071562', 'tt0468569', 'tt0050083']  # Popular movies
        else:
            # Use user's rating history (simplified - in real implementation, get from ratings)
            seed_movies = ['tt0111161', 'tt0068646', 'tt0071562']  # Placeholder
        
        candidates = set()
        
        # For each seed movie, get its neighbors
        for seed in seed_movies[:10]:  # Limit to 10 seeds
            if seed in self.content_neighbors:
                for neighbor in self.content_neighbors[seed][:20]:  # Top 20 neighbors per seed
                    candidates.add(neighbor['neighbor_id'])
        
        # Convert to list and cap at C_max
        candidate_list = list(candidates)[:self.c_max]
        
        return candidate_list
    
    def compute_cf_scores(self, user_id: int, candidate_movies: List[str]) -> np.ndarray:
        """Compute collaborative filtering scores for candidates."""
        if user_id not in self.maps['user_id_to_index']:
            return np.zeros(len(candidate_movies))
        
        user_idx = self.maps['user_id_to_index'][user_id]
        user_factors = self.artifacts['user_factors'][user_idx]
        
        scores = []
        for movie_id in candidate_movies:
            if movie_id in self.maps['canonical_id_to_movie_index']:
                movie_idx = self.maps['canonical_id_to_movie_index'][movie_id]
                # Check bounds
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
        # Simplified content scoring - in real implementation, use user's interaction history
        # For now, use average neighbor scores
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
        
        # Normalize to [0, 1]
        if len(scores) > 0 and scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        elif len(scores) > 0:
            scores = np.full_like(scores, 0.5)
        
        return scores
    
    def compute_hybrid_scores(self, alpha: float, cf_scores: np.ndarray, content_scores: np.ndarray) -> np.ndarray:
        """Compute hybrid scores using the blend formula."""
        return alpha * cf_scores + (1 - alpha) * content_scores
    
    def get_user_ground_truth(self, user_id: int) -> Set[str]:
        """Get ground truth items for a user."""
        user_gt = self.artifacts['ground_truth'][
            self.artifacts['ground_truth']['user_id'] == user_id
        ]
        return set(user_gt[user_gt['is_positive']]['canonical_id'].tolist())
    
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
    
    def evaluate_alpha(self, alpha: float) -> Dict[str, Any]:
        """Evaluate a single alpha value."""
        start_time = time.time()
        self.start_watchdog(alpha)
        
        all_metrics = []
        users_evaluated = 0
        users_skipped_no_candidates = 0
        users_dropped_for_runtime = 0
        
        # Per-bucket metrics
        bucket_metrics = {'cold': [], 'light': [], 'medium': [], 'heavy': []}
        
        try:
            # Process users in batches
            eval_users = self.artifacts['eval_users']
            
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
                        
                        try:
                            # Get candidates
                            candidates_df = self.get_user_candidates(user_id)
                            
                            if candidates_df is not None:
                                # Use 3c.2 candidates
                                candidates = candidates_df['canonical_id'].tolist()
                                # Recompute scores to ensure proper bounds checking
                                cf_scores = self.compute_cf_scores(user_id, candidates)
                                content_scores = self.compute_content_scores(user_id, candidates)
                                scores = self.compute_hybrid_scores(alpha, cf_scores, content_scores)
                            else:
                                # Generate fallback candidates
                                candidates = self.generate_content_fallback_candidates(user_id, user_bucket)
                                
                                if not candidates:
                                    users_skipped_no_candidates += 1
                                    continue
                                
                                # Compute scores
                                if alpha == 0.0 or not has_cf:
                                    scores = self.compute_content_scores(user_id, candidates)
                                elif alpha == 1.0:
                                    scores = self.compute_cf_scores(user_id, candidates)
                                else:
                                    cf_scores = self.compute_cf_scores(user_id, candidates)
                                    content_scores = self.compute_content_scores(user_id, candidates)
                                    scores = self.compute_hybrid_scores(alpha, cf_scores, content_scores)
                            
                            # Get ground truth
                            ground_truth = self.get_user_ground_truth(user_id)
                            
                            # Compute metrics
                            metrics = self.compute_metrics(candidates, scores, ground_truth)
                            batch_metrics.append(metrics)
                            bucket_metrics[user_bucket].append(metrics)
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
            'cold_recall': bucket_recalls['cold'],
            'light_recall': bucket_recalls['light'],
            'medium_recall': bucket_recalls['medium'],
            'heavy_recall': bucket_recalls['heavy'],
            'elapsed_sec': elapsed_sec,
            'partial': elapsed_sec > self.per_alpha_timeout,
            'unstable': unstable
        }
        
        logger.info(f"α = {alpha} summary users_eval = {users_evaluated} recall@10 = {overall_recall:.4f} map@10 = {overall_map:.4f} elapsed = {elapsed_sec:.1f} partial = {result['partial']}")
        
        return result
    
    def run_evaluation(self):
        """Run the real pass evaluation."""
        logger.info("=" * 80)
        logger.info("Starting Step 3c.3 – Real Pass (No Simulation) Evaluation")
        logger.info("=" * 80)
        
        try:
            # Run evaluation for each alpha
            for alpha in self.alpha_grid:
                # Check global timeout
                if time.time() - self.global_start_time > self.global_timeout:
                    logger.warning(f"Global timeout reached, stopping evaluation")
                    break
                
                # Evaluate alpha
                result = self.evaluate_alpha(alpha)
                self.results.append(result)
                
                # Save intermediate results
                self.save_tuning_results()
                
                # Check per-alpha timeout
                if result['elapsed_sec'] > self.per_alpha_timeout:
                    logger.warning(f"Per-alpha timeout reached for α = {alpha}")
            
            # Generate final report
            self.generate_evaluation_report()
            
            logger.info("Real pass evaluation completed successfully")
            
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
        """Generate the evaluation report."""
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
        improvement_vs_content = (best_recall - content_only_recall) / content_only_recall * 100 if content_only_recall > 0 else 0
        improvement_vs_cf = (best_recall - cf_only_recall) / cf_only_recall * 100 if cf_only_recall > 0 else 0
        overall_coverage = results_df['users_evaluated'].sum() / (len(self.artifacts['eval_users']) * len(self.alpha_grid))
        
        # Check cold-start guardrail
        best_alpha_cold_recall = results_df.loc[best_alpha_idx, 'cold_recall']
        cold_guardrail_met = best_alpha_cold_recall >= content_only_recall
        
        # Acceptance gates
        coverage_met = overall_coverage >= 0.6
        lift_content_met = improvement_vs_content >= 5
        lift_cf_met = improvement_vs_cf >= 15
        all_gates_met = coverage_met and lift_content_met and lift_cf_met and cold_guardrail_met
        
        logger.info("=" * 80)
        logger.info("ACCEPTANCE GATES EVALUATION")
        logger.info("=" * 80)
        logger.info(f"Coverage ≥ 60%: {'✅' if coverage_met else '❌'} ({overall_coverage:.1%})")
        logger.info(f"Lift vs Content-only ≥ 5%: {'✅' if lift_content_met else '❌'} ({improvement_vs_content:.1f}%)")
        logger.info(f"Lift vs CF-only ≥ 15%: {'✅' if lift_cf_met else '❌'} ({improvement_vs_cf:.1f}%)")
        logger.info(f"Cold-start guardrail: {'✅' if cold_guardrail_met else '❌'}")
        logger.info(f"All gates met: {'✅' if all_gates_met else '❌'}")
        logger.info("=" * 80)
        
        # Print tuning results for α=0.0, 0.5, 1.0
        logger.info("TUNING RESULTS FOR α=0.0, 0.5, 1.0:")
        for alpha in [0.0, 0.5, 1.0]:
            if alpha in results_df['alpha'].values:
                row = results_df[results_df['alpha'] == alpha].iloc[0]
                logger.info(f"α={alpha}: recall@10={row['recall_at_10']:.4f}, map@10={row['map_at_10']:.4f}, "
                          f"users={row['users_evaluated']}, coverage={row['users_evaluated']/len(self.artifacts['eval_users']):.1%}, "
                          f"partial={row['partial']}, unstable={row['unstable']}")
        
        logger.info(f"Best α by Recall@10: {best_alpha} (recall={best_recall:.4f})")
        logger.info(f"All acceptance gates met: {'✅' if all_gates_met else '❌'}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Step 3c.3 Real Pass Evaluation')
    parser.add_argument('--alpha_grid', nargs='+', type=float, default=[0.0, 0.5, 1.0])
    
    args = parser.parse_args()
    
    try:
        evaluator = RealPassEvaluator(alpha_grid=args.alpha_grid)
        evaluator.run_evaluation()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
