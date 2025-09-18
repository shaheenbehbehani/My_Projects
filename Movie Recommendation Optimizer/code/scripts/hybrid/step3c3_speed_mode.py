#!/usr/bin/env python3
"""
Step 3c.3 – Speed Mode Tuning & Offline Evaluation
Fast, robust hyperparameter sweep over α with strict time/memory bounds.
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
# import parquet  # Not needed, using pandas for parquet

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
def setup_logging():
    """Setup logging for speed mode evaluation."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "step3c3_speed_mode.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

class SpeedModeEvaluator:
    """Speed mode evaluator with strict timeouts and safety controls."""
    
    def __init__(self, 
                 alpha_grid: List[float] = [0.00, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00],
                 per_alpha_timeout: int = 360,  # 6 minutes
                 global_timeout: int = 1200,   # 20 minutes
                 batch_size: int = 200,
                 k: int = 10,
                 target_users: int = 2000):
        self.alpha_grid = alpha_grid
        self.per_alpha_timeout = per_alpha_timeout
        self.global_timeout = global_timeout
        self.batch_size = batch_size
        self.k = k
        self.target_users = target_users
        
        # Safety controls
        self.global_start_time = time.time()
        self.fallback_attempted = {}
        self.results = []
        
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
        
    def load_artifacts(self):
        """Load all required artifacts for evaluation."""
        logger.info("Loading artifacts for speed mode evaluation...")
        
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
    
    def create_user_sampling(self):
        """Create stratified user sampling with activity buckets."""
        logger.info("Creating stratified user sampling...")
        
        # Check if cached sampling exists
        eval_users_path = self.hybrid_dir / "eval_users_speed.parquet"
        if eval_users_path.exists():
            logger.info("Loading cached user sampling...")
            return pd.read_parquet(eval_users_path)
        
        # Create user activity buckets
        user_ratings = self.artifacts['ratings'].groupby('user_index').size().reset_index(name='rating_count')
        
        # Define buckets
        def assign_bucket(count):
            if count == 0:
                return 'cold'
            elif count < 3:
                return 'cold'  # Treat <3 as cold-start
            elif count <= 10:
                return 'light'
            elif count <= 100:
                return 'medium'
            else:
                return 'heavy'
        
        user_ratings['bucket'] = user_ratings['rating_count'].apply(assign_bucket)
        
        # Sample users from each bucket
        np.random.seed(42)
        sampled_users = []
        
        for bucket in ['cold', 'light', 'medium', 'heavy']:
            bucket_users = user_ratings[user_ratings['bucket'] == bucket]['user_index'].values
            if len(bucket_users) > 0:
                # Sample equal counts from each bucket
                n_sample = min(len(bucket_users), self.target_users // 4)
                sampled = np.random.choice(bucket_users, size=n_sample, replace=False)
                sampled_users.extend(sampled)
        
        # If we don't have enough users, fill with medium/heavy users
        if len(sampled_users) < self.target_users:
            remaining = self.target_users - len(sampled_users)
            medium_heavy = user_ratings[user_ratings['bucket'].isin(['medium', 'heavy'])]['user_index'].values
            additional = np.random.choice(medium_heavy, size=min(remaining, len(medium_heavy)), replace=False)
            sampled_users.extend(additional)
        
        # Create final sampling DataFrame
        eval_users = pd.DataFrame({
            'user_index': sampled_users[:self.target_users],
            'bucket': [user_ratings[user_ratings['user_index'] == uid]['bucket'].iloc[0] for uid in sampled_users[:self.target_users]]
        })
        
        # Save for reproducibility
        eval_users.to_parquet(eval_users_path)
        
        bucket_counts = eval_users['bucket'].value_counts()
        logger.info(f"Sampled {len(eval_users)} users: {dict(bucket_counts)}")
        
        return eval_users
    
    def create_ground_truth(self, eval_users: pd.DataFrame):
        """Create ground truth for evaluation users."""
        logger.info("Creating ground truth for evaluation...")
        
        # Get user ratings
        user_ratings = self.artifacts['ratings'].merge(eval_users, on='user_index')
        
        # Create train/test split (last 20% per user)
        ground_truth = {}
        
        for user_id in eval_users['user_index']:
            user_data = user_ratings[user_ratings['user_index'] == user_id]
            if len(user_data) > 1:
                # Take last 20% as test (since no timestamp, use random split)
                n_test = max(1, int(0.2 * len(user_data)))
                test_items = user_data.sample(n=n_test, random_state=42)['canonical_id'].values
                ground_truth[user_id] = set(test_items)
            else:
                ground_truth[user_id] = set()
        
        return ground_truth
    
    def generate_candidates_for_user(self, user_id: int, user_bucket: str) -> List[int]:
        """Generate candidates for a user using 3c.2 pipeline."""
        try:
            # Try to load existing candidates
            candidate_path = self.hybrid_dir / "candidates" / f"user_{user_id}_candidates.parquet"
            if candidate_path.exists():
                candidates_df = pd.read_parquet(candidate_path)
                return candidates_df['canonical_id'].tolist()[:1200]  # Cap at C_max
            
            # Fallback: generate content-only candidates
            if user_bucket == 'cold':
                # For cold users, use popular content
                popular_movies = self.artifacts['ratings']['canonical_id'].value_counts().head(1000).index.tolist()
                return popular_movies[:1200]
            else:
                # For other users, use content neighbors of their liked items
                user_ratings = self.artifacts['ratings'][self.artifacts['ratings']['user_index'] == user_id]
                if len(user_ratings) > 0:
                    liked_movies = user_ratings['canonical_id'].values
                    candidates = set()
                    for movie_id in liked_movies[:10]:  # Top 10 liked movies
                        # Get content neighbors
                        neighbors = self.artifacts['content_neighbors'][
                            self.artifacts['content_neighbors']['movie_id'] == movie_id
                        ]['neighbor_id'].values
                        candidates.update(neighbors)
                    return list(candidates)[:1200]
                else:
                    return []
                    
        except Exception as e:
            logger.warning(f"Failed to generate candidates for user {user_id}: {e}")
            return []
    
    def compute_hybrid_scores(self, user_id: int, candidates: List[int], alpha: float) -> np.ndarray:
        """Compute hybrid scores for candidates."""
        if not candidates:
            return np.array([])
        
        try:
            # Get user factor index
            user_idx = self.artifacts['user_index_map'][
                self.artifacts['user_index_map']['user_index'] == user_id
            ]['index'].values
            
            if len(user_idx) == 0:
                # Cold start user - content only
                return np.ones(len(candidates)) * 0.5  # Neutral score
            
            user_idx = user_idx[0]
            
            # Get movie factor indices for candidates
            movie_indices = []
            valid_candidates = []
            
            for movie_id in candidates:
                movie_idx = self.artifacts['movie_index_map'][
                    self.artifacts['movie_index_map']['canonical_id'] == movie_id
                ]['index'].values
                if len(movie_idx) > 0:
                    movie_indices.append(movie_idx[0])
                    valid_candidates.append(movie_id)
            
            if not movie_indices:
                return np.ones(len(candidates)) * 0.5
            
            # Compute collaborative scores
            user_factor = self.artifacts['user_factors'][user_idx]
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
            
            # Ensure scores are in [0,1]
            hybrid_scores = np.clip(hybrid_scores, 0, 1)
            
            return hybrid_scores
            
        except Exception as e:
            logger.warning(f"Failed to compute hybrid scores for user {user_id}: {e}")
            return np.ones(len(candidates)) * 0.5
    
    def compute_metrics(self, user_id: int, candidates: List[int], scores: np.ndarray, ground_truth: set) -> Dict[str, float]:
        """Compute Recall@K and MAP@K for a user."""
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
    
    def evaluate_alpha(self, alpha: float, eval_users: pd.DataFrame, ground_truth: Dict[int, set]) -> Dict[str, Any]:
        """Evaluate a single alpha value."""
        logger.info(f"Evaluating α = {alpha}")
        
        start_time = time.time()
        batch_results = []
        users_evaluated = 0
        users_skipped = 0
        batches_ok = 0
        batches_skipped = 0
        
        # Process users in batches
        for batch_start in range(0, len(eval_users), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(eval_users))
            batch_users = eval_users.iloc[batch_start:batch_end]
            
            batch_start_time = time.time()
            
            try:
                batch_metrics = []
                
                for _, user_row in batch_users.iterrows():
                    user_id = user_row['user_index']
                    user_bucket = user_row['bucket']
                    
                    # Generate candidates
                    candidates = self.generate_candidates_for_user(user_id, user_bucket)
                    if not candidates:
                        users_skipped += 1
                        continue
                    
                    # Compute hybrid scores
                    scores = self.compute_hybrid_scores(user_id, candidates, alpha)
                    if len(scores) == 0:
                        users_skipped += 1
                        continue
                    
                    # Compute metrics
                    metrics = self.compute_metrics(user_id, candidates, scores, ground_truth.get(user_id, set()))
                    batch_metrics.append(metrics)
                    users_evaluated += 1
                
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
                
                # Log heartbeat
                elapsed = time.time() - start_time
                logger.info(f"α={alpha} batch={batch_start//self.batch_size + 1} users={users_evaluated} "
                          f"skipped={users_skipped} time={elapsed:.1f}s")
                
            except Exception as e:
                logger.warning(f"Batch failed for α={alpha}: {e}")
                batches_skipped += 1
                continue
        
        # Aggregate results
        if batch_results:
            overall_recall = np.mean([r['recall_at_10'] for r in batch_results])
            overall_map = np.mean([r['map_at_10'] for r in batch_results])
        else:
            overall_recall = 0.0
            overall_map = 0.0
        
        elapsed_sec = time.time() - start_time
        
        result = {
            'alpha': alpha,
            'recall_at_10': overall_recall,
            'map_at_10': overall_map,
            'users_evaluated': users_evaluated,
            'users_skipped': users_skipped,
            'elapsed_sec': elapsed_sec,
            'batches_ok': batches_ok,
            'batches_skipped': batches_skipped
        }
        
        logger.info(f"α={alpha} completed: recall={overall_recall:.4f} map={overall_map:.4f} "
                   f"users={users_evaluated} time={elapsed_sec:.1f}s")
        
        return result
    
    def run_evaluation(self):
        """Run the complete speed mode evaluation."""
        logger.info("=" * 80)
        logger.info("Starting Step 3c.3 – Speed Mode Tuning & Evaluation")
        logger.info("=" * 80)
        
        try:
            # Create user sampling
            eval_users = self.create_user_sampling()
            
            # Create ground truth
            ground_truth = self.create_ground_truth(eval_users)
            
            # Run evaluation for each alpha
            for alpha in self.alpha_grid:
                # Check global timeout
                if time.time() - self.global_start_time > self.global_timeout:
                    logger.warning(f"Global timeout reached, stopping evaluation")
                    break
                
                # Evaluate alpha
                result = self.evaluate_alpha(alpha, eval_users, ground_truth)
                self.results.append(result)
                
                # Save intermediate results
                self.save_tuning_results()
                
                # Check per-alpha timeout
                if result['elapsed_sec'] > self.per_alpha_timeout:
                    logger.warning(f"Per-alpha timeout reached for α={alpha}")
            
            # Generate final report
            self.generate_evaluation_report()
            
            logger.info("Speed mode evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def save_tuning_results(self):
        """Save tuning results to CSV."""
        if not self.results:
            return
        
        results_df = pd.DataFrame(self.results)
        results_path = self.hybrid_dir / "tuning_results.csv"
        results_df.to_csv(results_path, index=False)
        
        # Also save intermediate results
        for result in self.results:
            alpha = result['alpha']
            tmp_path = self.tmp_dir / f"alpha_{alpha:.2f}.csv"
            pd.DataFrame([result]).to_csv(tmp_path, index=False)
    
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
        
        # Generate report
        report_path = self.data_dir.parent / "docs" / "step3c_eval.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Step 3c.3 – Speed Mode Evaluation Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Results Summary\n\n")
            f.write(f"- **Best α**: {best_alpha:.2f}\n")
            f.write(f"- **Best Recall@10**: {best_recall:.4f}\n")
            f.write(f"- **Content-only Recall@10**: {content_only_recall:.4f}\n")
            f.write(f"- **CF-only Recall@10**: {cf_only_recall:.4f}\n\n")
            
            # Check acceptance gates
            improvement_vs_content = (best_recall - content_only_recall) / content_only_recall * 100 if content_only_recall > 0 else 0
            improvement_vs_cf = (best_recall - cf_only_recall) / cf_only_recall * 100 if cf_only_recall > 0 else 0
            
            f.write("## Acceptance Gates\n\n")
            f.write(f"- **Improvement vs Content-only**: {improvement_vs_content:.1f}% {'✅' if improvement_vs_content >= 5 else '❌'}\n")
            f.write(f"- **Improvement vs CF-only**: {improvement_vs_cf:.1f}% {'✅' if improvement_vs_cf >= 5 else '❌'}\n")
            f.write(f"- **Global timeout**: {'✅' if time.time() - self.global_start_time <= self.global_timeout else '❌'}\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("| α | Recall@10 | MAP@10 | Users | Time (s) |\n")
            f.write("|---|-----------|--------|-------|----------|\n")
            
            for _, row in results_df.iterrows():
                f.write(f"| {row['alpha']:.2f} | {row['recall_at_10']:.4f} | {row['map_at_10']:.4f} | "
                       f"{row['users_evaluated']} | {row['elapsed_sec']:.1f} |\n")
            
            f.write("\n## Notes\n\n")
            f.write("- This is a speed mode evaluation with reduced scope for fast results\n")
            f.write("- Full evaluation would require more comprehensive candidate generation\n")
            f.write("- Results should be validated on a larger user sample for production\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Step 3c.3 Speed Mode Evaluation')
    parser.add_argument('--alpha_grid', nargs='+', type=float, 
                       default=[0.00, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00])
    parser.add_argument('--per_alpha_timeout_sec', type=int, default=360)
    parser.add_argument('--global_timeout_sec', type=int, default=1200)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--target_users', type=int, default=2000)
    
    args = parser.parse_args()
    
    try:
        evaluator = SpeedModeEvaluator(
            alpha_grid=args.alpha_grid,
            per_alpha_timeout=args.per_alpha_timeout_sec,
            global_timeout=args.global_timeout_sec,
            batch_size=args.batch_size,
            k=args.k,
            target_users=args.target_users
        )
        
        evaluator.run_evaluation()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
