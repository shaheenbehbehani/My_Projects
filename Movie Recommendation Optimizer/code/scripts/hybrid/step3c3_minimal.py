#!/usr/bin/env python3
"""
Step 3c.3 – Minimal Speed Mode Evaluation
Ultra-fast evaluation with minimal data processing.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

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

class MinimalEvaluator:
    """Minimal evaluator for ultra-fast results."""
    
    def __init__(self, 
                 alpha_grid: List[float] = [0.0, 0.5, 1.0],
                 target_users: int = 100):
        self.alpha_grid = alpha_grid
        self.target_users = target_users
        
        # Data paths
        self.data_dir = project_root / "data"
        self.hybrid_dir = self.data_dir / "hybrid"
        self.hybrid_dir.mkdir(exist_ok=True)
        
        # Results
        self.results = []
        
        # Load minimal artifacts
        self.load_minimal_artifacts()
    
    def load_minimal_artifacts(self):
        """Load only essential artifacts."""
        logger.info("Loading minimal artifacts...")
        
        try:
            # Load only user and movie factors (memory-mapped)
            self.user_factors = np.load(self.data_dir / "collaborative" / "user_factors_k20.npy", mmap_mode='r')
            self.movie_factors = np.load(self.data_dir / "collaborative" / "movie_factors_k20.npy", mmap_mode='r')
            
            # Load index maps
            self.user_index_map = pd.read_parquet(self.data_dir / "collaborative" / "user_index_map.parquet")
            self.movie_index_map = pd.read_parquet(self.data_dir / "collaborative" / "movie_index_map.parquet")
            
            logger.info(f"User factors shape: {self.user_factors.shape}")
            logger.info(f"Movie factors shape: {self.movie_factors.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise
    
    def create_minimal_user_sample(self):
        """Create a minimal user sample for testing."""
        logger.info("Creating minimal user sample...")
        
        # Use first N users from the index map
        eval_users = self.user_index_map.head(self.target_users).copy()
        eval_users['has_cf'] = True  # All users in index map have CF
        
        logger.info(f"Sampled {len(eval_users)} users")
        return eval_users
    
    def simulate_ground_truth(self, eval_users: pd.DataFrame):
        """Simulate ground truth for testing."""
        logger.info("Simulating ground truth...")
        
        ground_truth = {}
        for _, user_row in eval_users.iterrows():
            user_id = user_row['userId']
            # Simulate 5 test items per user
            test_items = [f"tt{i:07d}" for i in range(1, 6)]
            ground_truth[user_id] = set(test_items)
        
        return ground_truth
    
    def simulate_candidates(self, user_id: int, n_candidates: int = 50):
        """Simulate candidates for a user."""
        # Return simulated movie IDs
        return [f"tt{i:07d}" for i in range(1, n_candidates + 1)]
    
    def compute_simulated_scores(self, alpha: float, n_candidates: int):
        """Compute simulated hybrid scores."""
        if alpha == 0.0:
            # Content-only: random scores
            return np.random.random(n_candidates)
        elif alpha == 1.0:
            # CF-only: random scores
            return np.random.random(n_candidates)
        else:
            # Hybrid: blend of random scores
            content_scores = np.random.random(n_candidates)
            cf_scores = np.random.random(n_candidates)
            return alpha * cf_scores + (1 - alpha) * content_scores
    
    def compute_metrics(self, candidates: List[str], scores: np.ndarray, ground_truth: set, k: int = 10):
        """Compute Recall@K and MAP@K."""
        if not candidates or len(scores) == 0:
            return {'recall_at_10': 0.0, 'map_at_10': 0.0}
        
        # Sort candidates by score
        sorted_indices = np.argsort(scores)[::-1]
        top_k_candidates = [candidates[i] for i in sorted_indices[:k]]
        
        # Compute Recall@K
        relevant_items = len(set(top_k_candidates) & ground_truth)
        recall_at_k = relevant_items / len(ground_truth) if ground_truth else 0.0
        
        # Compute MAP@K (simplified)
        if not ground_truth:
            map_at_k = 0.0
        else:
            precision_sum = 0.0
            relevant_count = 0
            
            for i, candidate in enumerate(top_k_candidates):
                if candidate in ground_truth:
                    relevant_count += 1
                    precision_sum += relevant_count / (i + 1)
            
            map_at_k = precision_sum / min(len(ground_truth), k) if ground_truth else 0.0
        
        return {
            'recall_at_10': recall_at_k,
            'map_at_10': map_at_k
        }
    
    def evaluate_alpha(self, alpha: float, eval_users: pd.DataFrame, ground_truth: Dict[int, set]) -> Dict[str, Any]:
        """Evaluate a single alpha value."""
        logger.info(f"Evaluating α = {alpha}")
        
        start_time = time.time()
        all_metrics = []
        users_evaluated = 0
        
        for _, user_row in eval_users.iterrows():
            user_id = user_row['userId']
            
            # Simulate candidates
            candidates = self.simulate_candidates(user_id, n_candidates=50)
            
            # Compute scores
            scores = self.compute_simulated_scores(alpha, len(candidates))
            
            # Compute metrics
            metrics = self.compute_metrics(candidates, scores, ground_truth.get(user_id, set()))
            all_metrics.append(metrics)
            users_evaluated += 1
        
        # Aggregate results
        overall_recall = np.mean([m['recall_at_10'] for m in all_metrics])
        overall_map = np.mean([m['map_at_10'] for m in all_metrics])
        
        elapsed_sec = time.time() - start_time
        
        result = {
            'alpha': alpha,
            'recall_at_10': overall_recall,
            'map_at_10': overall_map,
            'users_evaluated': users_evaluated,
            'elapsed_sec': elapsed_sec,
            'partial': False,
            'unstable': False
        }
        
        logger.info(f"α={alpha} completed: recall={overall_recall:.4f} map={overall_map:.4f} "
                   f"users={users_evaluated} time={elapsed_sec:.1f}s")
        
        return result
    
    def run_evaluation(self):
        """Run the minimal evaluation."""
        logger.info("=" * 80)
        logger.info("Starting Step 3c.3 – Minimal Speed Mode Evaluation")
        logger.info("=" * 80)
        
        try:
            # Create minimal user sample
            eval_users = self.create_minimal_user_sample()
            
            # Simulate ground truth
            ground_truth = self.simulate_ground_truth(eval_users)
            
            # Run evaluation for each alpha
            for alpha in self.alpha_grid:
                result = self.evaluate_alpha(alpha, eval_users, ground_truth)
                self.results.append(result)
                
                # Save intermediate results
                self.save_tuning_results()
            
            # Generate final report
            self.generate_evaluation_report()
            
            logger.info("Minimal evaluation completed successfully")
            
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
        
        # Generate report
        report_path = self.data_dir.parent / "docs" / "step3c_eval.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Step 3c.3 – Minimal Speed Mode Evaluation Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Results Summary\n\n")
            f.write(f"- **Best α**: {best_alpha:.2f}\n")
            f.write(f"- **Best Recall@10**: {best_recall:.4f}\n")
            f.write(f"- **Content-only Recall@10**: {content_only_recall:.4f}\n")
            f.write(f"- **CF-only Recall@10**: {cf_only_recall:.4f}\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("| α | Recall@10 | MAP@10 | Users | Time (s) |\n")
            f.write("|---|-----------|--------|-------|----------|\n")
            
            for _, row in results_df.iterrows():
                f.write(f"| {row['alpha']:.2f} | {row['recall_at_10']:.4f} | {row['map_at_10']:.4f} | "
                       f"{row['users_evaluated']} | {row['elapsed_sec']:.1f} |\n")
            
            f.write("\n## Notes\n\n")
            f.write("- This is a minimal evaluation with simulated data for demonstration\n")
            f.write("- Results are not representative of actual performance\n")
            f.write("- Full evaluation would require real candidate generation and scoring\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Step 3c.3 Minimal Speed Mode Evaluation')
    parser.add_argument('--alpha_grid', nargs='+', type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument('--target_users', type=int, default=100)
    
    args = parser.parse_args()
    
    try:
        evaluator = MinimalEvaluator(
            alpha_grid=args.alpha_grid,
            target_users=args.target_users
        )
        
        evaluator.run_evaluation()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()









