"""
Step 4.1.2: Content-Based Evaluation - Optimized Pipeline
=========================================================

Optimized version with speed modes, instrumentation, and performance monitoring.
Supports smoke, speed, and full evaluation modes with batching and heartbeats.
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
import time
import psutil
import cProfile
import pstats
import io
from typing import Dict, List, Tuple, Any, Optional
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from scripts.eval.metrics import MetricsFramework, MetricConfig

class OptimizedContentEvaluator:
    """Optimized content-based recommendation system evaluator with instrumentation."""
    
    def __init__(self, data_dir: str = "data", mode: str = "smoke"):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.results = {}
        
        # Set mode-specific parameters
        self._set_mode_parameters()
        
        # Initialize metrics framework with mode-specific K values
        self.metrics = MetricsFramework(MetricConfig(k_values=self.k_values))
        
        # Performance tracking
        self.start_time = time.time()
        self.last_heartbeat = time.time()
        self.batch_count = 0
        self.users_processed = 0
        self.profile_data = None
        
        # Memory monitoring
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Setup logging
        self._setup_logging()
        
        # Load data
        self._load_data()
        
    def _set_mode_parameters(self):
        """Set parameters based on evaluation mode."""
        if self.mode == "smoke":
            self.max_users = 500
            self.batch_size = 100
            self.candidates_cap_per_user = 600
            self.heartbeat_sec = 30
            self.k_values = [10, 20]
            self.batch_timeout = 180
        elif self.mode == "speed":
            self.max_users = 1500
            self.batch_size = 200
            self.candidates_cap_per_user = 1000
            self.heartbeat_sec = 30
            self.k_values = [5, 10, 20, 50]
            self.batch_timeout = 360
        else:  # full
            self.max_users = None  # No limit
            self.batch_size = 200
            self.candidates_cap_per_user = 1000
            self.heartbeat_sec = 30
            self.k_values = [5, 10, 20, 50]
            self.batch_timeout = 360
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Main logger
        self.logger = logging.getLogger(f"content_eval_{self.mode}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        fh = logging.FileHandler(log_dir / f"content_eval_{self.mode}.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Progress CSV setup
        self.progress_file = log_dir / "content_eval_progress.csv"
        if not self.progress_file.exists():
            with open(self.progress_file, 'w') as f:
                f.write("timestamp,users_done,batches_done,elapsed_sec,rss_mb,mode\n")
        
        # Partial results setup
        self.partial_dir = Path("data/eval/tmp")
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        self.partial_file = self.partial_dir / "content_eval_partial.jsonl"
    
    def _load_data(self):
        """Load all required data with memory optimization."""
        self.logger.info(f"Loading data for {self.mode} mode...")
        
        # Load movie metadata first
        self.logger.info("Loading movie metadata...")
        self.movie_metadata = pd.read_parquet(self.data_dir / "features" / "composite" / "movies_features_v1.parquet")
        self.logger.info(f"Loaded movie metadata: {self.movie_metadata.shape}")
        
        # Create movie ID mappings
        self.movie_id_to_idx = dict(zip(self.movie_metadata['canonical_id'], 
                                      self.movie_metadata['canonical_idx']))
        self.idx_to_movie_id = dict(zip(self.movie_metadata['canonical_idx'], 
                                      self.movie_metadata['canonical_id']))
        
        # Load kNN arrays directly (much faster than parquet)
        self.logger.info("Loading kNN arrays...")
        self.knn_indices = np.load(self.data_dir / "similarity" / "knn_indices_k50.npz")['indices']
        self.knn_scores = np.load(self.data_dir / "similarity" / "knn_scores_k50.npz")['scores']
        self.logger.info(f"Loaded kNN arrays: indices {self.knn_indices.shape}, scores {self.knn_scores.shape}")
        
        # Load ground truth data
        self.logger.info("Loading ground truth data...")
        self.train_data = pd.read_parquet(self.data_dir / "eval" / "checkpoints" / "ratings_split_train.parquet")
        self.test_data = pd.read_parquet(self.data_dir / "eval" / "checkpoints" / "ratings_split_test.parquet")
        self.logger.info(f"Loaded train data: {self.train_data.shape}")
        self.logger.info(f"Loaded test data: {self.test_data.shape}")
        
        self.logger.info(f"Created movie ID mappings for {len(self.movie_id_to_idx)} movies")
        self._log_memory_usage("After data loading")
    
    
    def _log_memory_usage(self, stage: str):
        """Log current memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.logger.info(f"Memory usage {stage}: {current_memory:.1f} MB (Δ: {current_memory - self.initial_memory:+.1f} MB)")
        return current_memory
    
    def _heartbeat(self, users_done: int, batch_id: str = None):
        """Log heartbeat with progress information."""
        current_time = time.time()
        if current_time - self.last_heartbeat >= self.heartbeat_sec:
            elapsed = current_time - self.start_time
            memory = self._log_memory_usage("heartbeat")
            
            if users_done > 0:
                rate = users_done / elapsed
                eta = (self.max_users - users_done) / rate if self.max_users else "N/A"
                self.logger.info(f"Progress: {users_done}/{self.max_users or '∞'} users, "
                               f"{self.batch_count} batches, {elapsed:.1f}s elapsed, "
                               f"rate: {rate:.1f} users/s, ETA: {eta}")
            else:
                self.logger.info(f"Progress: {users_done} users, {self.batch_count} batches, "
                               f"{elapsed:.1f}s elapsed")
            
            # Update progress CSV
            with open(self.progress_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()},{users_done},{self.batch_count},"
                       f"{elapsed:.1f},{memory:.1f},{self.mode}\n")
            
            self.last_heartbeat = current_time
    
    def _generate_candidates_optimized(self, user_movies: List[str]) -> List[str]:
        """Optimized candidate generation using precomputed kNN arrays."""
        if not user_movies:
            return []
        
        # Convert movie IDs to indices
        user_movie_indices = []
        for movie_id in user_movies:
            if movie_id in self.movie_id_to_idx:
                user_movie_indices.append(self.movie_id_to_idx[movie_id])
        
        if not user_movie_indices:
            return []
        
        # Get all neighbors using vectorized operations
        all_neighbor_indices = []
        all_neighbor_scores = []
        
        for movie_idx in user_movie_indices:
            # Get neighbors for this movie (kNN arrays are already sorted by similarity)
            neighbors = self.knn_indices[movie_idx]
            scores = self.knn_scores[movie_idx]
            
            all_neighbor_indices.extend(neighbors)
            all_neighbor_scores.extend(scores)
        
        # Convert to numpy arrays for efficient processing
        all_neighbor_indices = np.array(all_neighbor_indices)
        all_neighbor_scores = np.array(all_neighbor_scores)
        
        # Exclude user's training movies
        user_movie_indices_set = set(user_movie_indices)
        mask = ~np.isin(all_neighbor_indices, list(user_movie_indices_set))
        
        filtered_indices = all_neighbor_indices[mask]
        filtered_scores = all_neighbor_scores[mask]
        
        if len(filtered_indices) == 0:
            return []
        
        # Find unique neighbors and aggregate scores
        unique_indices, inverse_indices = np.unique(filtered_indices, return_inverse=True)
        unique_scores = np.zeros(len(unique_indices))
        
        # Aggregate scores (use max score for each neighbor)
        for i, score in enumerate(filtered_scores):
            unique_scores[inverse_indices[i]] = max(unique_scores[inverse_indices[i]], score)
        
        # Count frequency of each neighbor
        unique_counts = np.bincount(inverse_indices)
        
        # Sort by frequency first, then by score
        sort_indices = np.lexsort((-unique_scores, -unique_counts))
        sorted_indices = unique_indices[sort_indices]
        
        # Convert back to movie IDs
        candidates = []
        for idx in sorted_indices[:self.candidates_cap_per_user]:
            if idx in self.idx_to_movie_id:
                candidates.append(self.idx_to_movie_id[idx])
        
        return candidates
    
    def _process_batch(self, user_batch: List[int], batch_id: str) -> Dict[str, Any]:
        """Process a batch of users with timeout protection."""
        batch_start = time.time()
        batch_results = {
            'batch_id': batch_id,
            'users': [],
            'recommendations': {},
            'metrics': {}
        }
        
        try:
            for user_idx in user_batch:
                # Get user's training movies
                user_train_movies = self.train_data[
                    self.train_data['user_index'] == user_idx
                ]['canonical_id'].tolist()
                
                # Generate recommendations
                user_recs = self._generate_candidates_optimized(user_train_movies)
                batch_results['recommendations'][str(user_idx)] = user_recs
                batch_results['users'].append(user_idx)
                
                # Check timeout
                if time.time() - batch_start > self.batch_timeout:
                    self.logger.warning(f"Batch {batch_id} timeout after {self.batch_timeout}s, "
                                      f"processed {len(batch_results['users'])}/{len(user_batch)} users")
                    # Reduce candidates cap for next batch
                    self.candidates_cap_per_user = max(100, int(self.candidates_cap_per_user * 0.5))
                    self.logger.info(f"Reduced candidates_cap_per_user to {self.candidates_cap_per_user}")
                    break
            
            # Compute metrics for this batch
            if batch_results['recommendations']:
                batch_metrics = self._compute_batch_metrics(batch_results['recommendations'])
                batch_results['metrics'] = batch_metrics
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Error in batch {batch_id}: {str(e)}")
            return batch_results
    
    def _compute_batch_metrics(self, recommendations: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compute metrics for a batch of recommendations."""
        # Get ground truth for these users
        user_ids = list(recommendations.keys())
        ground_truth = {}
        
        for user_id in user_ids:
            user_test_movies = self.test_data[
                self.test_data['user_index'] == int(user_id)
            ]['canonical_id'].tolist()
            if user_test_movies:
                ground_truth[user_id] = user_test_movies
        
        if not ground_truth:
            return {}
        
        # Compute ranking metrics
        ranking_results = self.metrics.evaluate_ranking_metrics(ground_truth, recommendations)
        
        # Compute coverage metrics
        all_users = list(set(self.train_data['user_index'].astype(str).tolist() + 
                           self.test_data['user_index'].astype(str).tolist()))
        all_items = list(set(self.train_data['canonical_id'].astype(str).tolist() + 
                           self.test_data['canonical_id'].astype(str).tolist()))
        
        coverage_results = self.metrics.evaluate_coverage_metrics(recommendations, all_users, all_items)
        
        return {
            'ranking_metrics': ranking_results,
            'coverage_metrics': coverage_results,
            'num_users': len(recommendations),
            'num_ground_truth_users': len(ground_truth)
        }
    
    def evaluate_holdout_split_optimized(self) -> Dict[str, Any]:
        """Optimized holdout split evaluation with batching and instrumentation."""
        self.logger.info(f"Running optimized holdout split evaluation ({self.mode} mode)...")
        
        # Get test users
        test_users = self.test_data['user_index'].unique()
        if self.max_users:
            test_users = test_users[:self.max_users]
        
        self.logger.info(f"Processing {len(test_users)} test users in batches of {self.batch_size}")
        
        # Process in batches
        all_recommendations = {}
        all_metrics = []
        
        for i in range(0, len(test_users), self.batch_size):
            batch_users = test_users[i:i + self.batch_size]
            batch_id = f"batch_{i//self.batch_size + 1}"
            
            self.logger.info(f"Processing {batch_id}: users {i+1}-{min(i+self.batch_size, len(test_users))}")
            
            # Process batch with timeout protection
            batch_results = self._process_batch(batch_users, batch_id)
            
            # Update global results
            all_recommendations.update(batch_results['recommendations'])
            if batch_results['metrics']:
                all_metrics.append(batch_results['metrics'])
            
            # Update counters
            self.batch_count += 1
            self.users_processed += len(batch_results['users'])
            
            # Log partial results
            self._log_partial_results(batch_results)
            
            # Heartbeat
            self._heartbeat(self.users_processed, batch_id)
            
            # Memory check
            current_memory = self._log_memory_usage(f"after {batch_id}")
            if current_memory > self.initial_memory * 1.8:  # 80% increase
                self.candidates_cap_per_user = max(100, int(self.candidates_cap_per_user * 0.75))
                self.logger.warning(f"High memory usage, reduced candidates_cap_per_user to {self.candidates_cap_per_user}")
        
        # Compute final metrics
        self.logger.info("Computing final metrics...")
        final_results = self._compute_final_metrics(all_recommendations, all_metrics)
        
        return final_results
    
    def _log_partial_results(self, batch_results: Dict[str, Any]):
        """Log partial results to JSONL file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        batch_results_serializable = convert_numpy_types(batch_results)
        
        with open(self.partial_file, 'a') as f:
            json.dump(batch_results_serializable, f)
            f.write('\n')
    
    def _compute_final_metrics(self, recommendations: Dict[str, List[str]], 
                              batch_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute final aggregated metrics."""
        # Get ground truth for all users
        ground_truth = {}
        for user_id in recommendations.keys():
            user_test_movies = self.test_data[
                self.test_data['user_index'] == int(user_id)
            ]['canonical_id'].tolist()
            if user_test_movies:
                ground_truth[user_id] = user_test_movies
        
        # Compute final ranking metrics
        ranking_results = self.metrics.evaluate_ranking_metrics(ground_truth, recommendations)
        
        # Compute final coverage metrics
        all_users = list(set(self.train_data['user_index'].astype(str).tolist() + 
                           self.test_data['user_index'].astype(str).tolist()))
        all_items = list(set(self.train_data['canonical_id'].astype(str).tolist() + 
                           self.test_data['canonical_id'].astype(str).tolist()))
        
        coverage_results = self.metrics.evaluate_coverage_metrics(recommendations, all_users, all_items)
        
        # Aggregate batch metrics
        aggregated_batch_metrics = self._aggregate_batch_metrics(batch_metrics)
        
        return {
            'strategy': 'holdout_split_optimized',
            'mode': self.mode,
            'ranking_metrics': ranking_results,
            'coverage_metrics': coverage_results,
            'batch_metrics': aggregated_batch_metrics,
            'num_test_users': len(recommendations),
            'num_ground_truth_users': len(ground_truth),
            'num_test_interactions': len(self.test_data),
            'total_batches': self.batch_count,
            'total_users_processed': self.users_processed,
            'evaluation_time_seconds': time.time() - self.start_time,
            'candidates_cap_per_user_final': self.candidates_cap_per_user
        }
    
    def _aggregate_batch_metrics(self, batch_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across batches."""
        if not batch_metrics:
            return {}
        
        # Simple aggregation for now
        total_users = sum(bm.get('num_users', 0) for bm in batch_metrics)
        total_gt_users = sum(bm.get('num_ground_truth_users', 0) for bm in batch_metrics)
        
        return {
            'total_users_across_batches': total_users,
            'total_ground_truth_users_across_batches': total_gt_users,
            'num_batches': len(batch_metrics)
        }
    
    def generate_visualizations(self, results: Dict[str, Any], output_dir: str = "data/eval"):
        """Generate visualization charts."""
        self.logger.info("Generating visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Ranking metrics vs K
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Content-Based Recommendation Metrics vs K ({self.mode.upper()} Mode)', 
                    fontsize=16, fontweight='bold')
        
        k_values = self.k_values
        ranking_metrics = results['ranking_metrics']
        
        # Recall@K
        axes[0, 0].plot(k_values, [ranking_metrics['recall'][k] for k in k_values], 
                        marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_title('Recall@K', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('K')
        axes[0, 0].set_ylabel('Recall')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # Precision@K
        axes[0, 1].plot(k_values, [ranking_metrics['precision'][k] for k in k_values], 
                        marker='s', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_title('Precision@K', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('K')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # MAP@K
        axes[1, 0].plot(k_values, [ranking_metrics['map'][k] for k in k_values], 
                        marker='^', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_title('MAP@K', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('K')
        axes[1, 0].set_ylabel('MAP')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # NDCG@K
        axes[1, 1].plot(k_values, [ranking_metrics['ndcg'][k] for k in k_values], 
                        marker='d', linewidth=2, markersize=8, color='red')
        axes[1, 1].set_title('NDCG@K', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('K')
        axes[1, 1].set_ylabel('NDCG')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'content_eval_ranking_metrics_{self.mode}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Coverage metrics bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        coverage_metrics = results['coverage_metrics']
        metrics = list(coverage_metrics.keys())
        values = list(coverage_metrics.values())
        
        bars = ax.bar(metrics, values, color=['skyblue', 'lightcoral'], alpha=0.8)
        ax.set_title(f'Content-Based Recommendation Coverage Metrics ({self.mode.upper()} Mode)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Coverage Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / f'content_eval_coverage_metrics_{self.mode}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to {output_dir}")
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """Save evaluation results to JSON file."""
        if output_path is None:
            output_path = f"data/eval/content_eval_results_{self.mode}.json"
        
        self.logger.info(f"Saving results to {output_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy_types(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete content-based evaluation."""
        self.logger.info(f"Starting {self.mode} mode content-based evaluation...")
        
        # Start profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            # Run evaluation
            results = self.evaluate_holdout_split_optimized()
            
            # Generate visualizations
            self.generate_visualizations(results)
            
            # Save results
            self.save_results(results)
            
            # Final memory check
            final_memory = self._log_memory_usage("final")
            
            # Add performance summary
            results['performance_summary'] = {
                'total_time_seconds': time.time() - self.start_time,
                'users_per_second': self.users_processed / (time.time() - self.start_time),
                'batches_processed': self.batch_count,
                'initial_memory_mb': self.initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': final_memory - self.initial_memory,
                'candidates_cap_per_user_final': self.candidates_cap_per_user
            }
            
            self.logger.info(f"Evaluation completed successfully in {time.time() - self.start_time:.1f} seconds")
            return results
            
        finally:
            # Stop profiling and save
            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(25)
            
            profile_file = Path("logs") / "content_eval_profile.txt"
            with open(profile_file, 'w') as f:
                f.write(s.getvalue())
            
            self.logger.info(f"Profile saved to {profile_file}")

def main():
    """Main execution function with CLI arguments."""
    parser = argparse.ArgumentParser(description='Content-Based Evaluation - Optimized Pipeline')
    parser.add_argument('--mode', choices=['smoke', 'speed', 'full'], default='smoke',
                       help='Evaluation mode (default: smoke)')
    parser.add_argument('--max_users', type=int, help='Override max users for mode')
    parser.add_argument('--batch_size', type=int, help='Override batch size for mode')
    parser.add_argument('--candidates_cap_per_user', type=int, help='Override candidates cap for mode')
    parser.add_argument('--heartbeat_sec', type=int, help='Override heartbeat interval for mode')
    
    args = parser.parse_args()
    
    print(f"Starting Step 4.1.2: Content-Based Evaluation ({args.mode.upper()} mode)")
    print("="*60)
    
    # Initialize evaluator
    evaluator = OptimizedContentEvaluator(mode=args.mode)
    
    # Override parameters if provided
    if args.max_users:
        evaluator.max_users = args.max_users
    if args.batch_size:
        evaluator.batch_size = args.batch_size
    if args.candidates_cap_per_user:
        evaluator.candidates_cap_per_user = args.candidates_cap_per_user
    if args.heartbeat_sec:
        evaluator.heartbeat_sec = args.heartbeat_sec
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Print summary
    print("\n" + "="*60)
    print(f"CONTENT-BASED EVALUATION SUMMARY ({args.mode.upper()} MODE)")
    print("="*60)
    
    print(f"\nTest Users: {results['num_test_users']}")
    print(f"Ground Truth Users: {results['num_ground_truth_users']}")
    print(f"Test Interactions: {results['num_test_interactions']}")
    print(f"Batches Processed: {results['total_batches']}")
    print(f"Evaluation Time: {results['evaluation_time_seconds']:.1f} seconds")
    print(f"Users per Second: {results['performance_summary']['users_per_second']:.1f}")
    
    print(f"\nRanking Metrics:")
    for metric, scores in results['ranking_metrics'].items():
        print(f"  {metric.upper()}:")
        for k, score in scores.items():
            print(f"    @{k}: {score:.3f}")
    
    print(f"\nCoverage Metrics:")
    for metric, score in results['coverage_metrics'].items():
        print(f"  {metric}: {score:.3f}")
    
    print(f"\nResults saved to: data/eval/content_eval_results_{args.mode}.json")
    print(f"Visualizations saved to: data/eval/")
    print(f"Progress log: logs/content_eval_progress.csv")
    print("="*60)

if __name__ == "__main__":
    main()
