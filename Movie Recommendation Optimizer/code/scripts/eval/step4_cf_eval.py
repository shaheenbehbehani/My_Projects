"""
Step 4.1.3: Collaborative Filtering Evaluation
==============================================

This script evaluates the collaborative filtering model from Step 3b using the
metrics framework from Step 4.1.1. Supports both rating predictions and top-K
ranking evaluation with optimized pipeline and instrumentation.
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

class CollaborativeFilteringEvaluator:
    """Collaborative filtering recommendation system evaluator."""
    
    def __init__(self, data_dir: str = "data", mode: str = "smoke"):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.results = {}
        
        # Set mode-specific parameters
        self._set_mode_parameters()
        
        # Initialize metrics framework
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
            self.heartbeat_sec = 30
            self.k_values = [10, 20]
            self.batch_timeout = 180
        elif self.mode == "speed":
            self.max_users = 1500
            self.batch_size = 200
            self.heartbeat_sec = 30
            self.k_values = [5, 10, 20, 50]
            self.batch_timeout = 360
        else:  # full
            self.max_users = None  # No limit
            self.batch_size = 200
            self.heartbeat_sec = 30
            self.k_values = [5, 10, 20, 50]
            self.batch_timeout = 360
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Main logger
        self.logger = logging.getLogger(f"cf_eval_{self.mode}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        fh = logging.FileHandler(log_dir / f"cf_eval_{self.mode}.log")
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
        self.progress_file = log_dir / "cf_eval_progress.csv"
        if not self.progress_file.exists():
            with open(self.progress_file, 'w') as f:
                f.write("timestamp,users_done,batches_done,elapsed_sec,rss_mb,mode\n")
        
        # Partial results setup
        self.partial_dir = Path("data/eval/tmp")
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        self.partial_file = self.partial_dir / "cf_eval_partial.jsonl"
    
    def _load_data(self):
        """Load all required data from Step 3b and ground truth."""
        self.logger.info(f"Loading data for {self.mode} mode...")
        
        # Load collaborative filtering data
        self.logger.info("Loading collaborative filtering data...")
        self.user_factors = np.load(self.data_dir / "collaborative" / "user_factors_k20.npy")
        self.movie_factors = np.load(self.data_dir / "collaborative" / "movie_factors_k20.npy")
        self.logger.info(f"Loaded user factors: {self.user_factors.shape}")
        self.logger.info(f"Loaded movie factors: {self.movie_factors.shape}")
        
        # Load index mappings
        self.user_index_map = pd.read_parquet(self.data_dir / "collaborative" / "user_index_map.parquet")
        self.movie_index_map = pd.read_parquet(self.data_dir / "collaborative" / "movie_index_map.parquet")
        self.logger.info(f"Loaded user index map: {self.user_index_map.shape}")
        self.logger.info(f"Loaded movie index map: {self.movie_index_map.shape}")
        
        # Load configuration
        with open(self.data_dir / "collaborative" / "factorization_config.json", 'r') as f:
            self.cf_config = json.load(f)
        self.logger.info(f"Loaded CF config: {self.cf_config['algorithm']} k={self.cf_config['n_components']}")
        
        # Note: Both user and movie factors only cover first N items due to filtering during training
        self.num_users_in_factors = self.user_factors.shape[0]
        self.num_movies_in_factors = self.movie_factors.shape[0]
        self.logger.info(f"User factors cover first {self.num_users_in_factors} users out of {len(self.user_index_map)} total")
        self.logger.info(f"Movie factors cover first {self.num_movies_in_factors} movies out of {len(self.movie_index_map)} total")
        
        # Create mappings only for users and movies that have factors
        user_subset = self.user_index_map.iloc[:self.num_users_in_factors]
        movie_subset = self.movie_index_map.iloc[:self.num_movies_in_factors]
        
        self.user_id_to_idx = dict(zip(user_subset['userId'], user_subset['user_index']))
        self.idx_to_user_id = dict(zip(user_subset['user_index'], user_subset['userId']))
        self.canonical_id_to_idx = dict(zip(movie_subset['canonical_id'], movie_subset['movie_index']))
        self.idx_to_canonical_id = dict(zip(movie_subset['movie_index'], movie_subset['canonical_id']))
        
        # Load ground truth data
        self.logger.info("Loading ground truth data...")
        self.train_data = pd.read_parquet(self.data_dir / "eval" / "checkpoints" / "ratings_split_train.parquet")
        self.test_data = pd.read_parquet(self.data_dir / "eval" / "checkpoints" / "ratings_split_test.parquet")
        self.logger.info(f"Loaded train data: {self.train_data.shape}")
        self.logger.info(f"Loaded test data: {self.test_data.shape}")
        
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
    
    def predict_rating(self, user_id: int, canonical_id: str) -> Optional[float]:
        """Predict rating for a user-movie pair."""
        try:
            user_idx = self.user_id_to_idx[user_id]
            movie_idx = self.canonical_id_to_idx[canonical_id]
            return np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx])
        except KeyError:
            return None  # User or movie not in training set
    
    def generate_cf_recommendations(self, user_id: int, k: int = 50) -> List[str]:
        """Generate collaborative filtering recommendations for a user."""
        try:
            user_idx = self.user_id_to_idx[user_id]
            user_vector = self.user_factors[user_idx]
            
            # Compute scores for all movies
            scores = np.dot(self.movie_factors, user_vector)
            
            # Get user's training movies to exclude
            user_train_movies = set(self.train_data[
                self.train_data['user_index'] == user_id
            ]['canonical_id'].tolist())
            
            # Filter out training movies and get top-K
            movie_indices = np.arange(len(self.movie_factors))
            movie_scores = list(zip(movie_indices, scores))
            
            # Sort by score (descending) and filter out training movies
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for movie_idx, score in movie_scores:
                canonical_id = self.idx_to_canonical_id[movie_idx]
                if canonical_id not in user_train_movies:
                    recommendations.append(canonical_id)
                    if len(recommendations) >= k:
                        break
            
            return recommendations
            
        except KeyError:
            return []  # User not in training set
    
    def _process_batch(self, user_batch: List[int], batch_id: str) -> Dict[str, Any]:
        """Process a batch of users with timeout protection."""
        batch_start = time.time()
        batch_results = {
            'batch_id': batch_id,
            'users': [],
            'recommendations': {},
            'predictions': {},
            'metrics': {}
        }
        
        try:
            for user_id in user_batch:
                # Generate recommendations
                user_recs = self.generate_cf_recommendations(user_id, k=50)
                batch_results['recommendations'][str(user_id)] = user_recs
                batch_results['users'].append(user_id)
                
                # Generate rating predictions for test interactions
                user_test_movies = self.test_data[
                    self.test_data['user_index'] == user_id
                ]['canonical_id'].tolist()
                
                for movie_id in user_test_movies:
                    pred_rating = self.predict_rating(user_id, movie_id)
                    if pred_rating is not None:
                        batch_results['predictions'][f"{user_id}_{movie_id}"] = pred_rating
                
                # Check timeout
                if time.time() - batch_start > self.batch_timeout:
                    self.logger.warning(f"Batch {batch_id} timeout after {self.batch_timeout}s, "
                                      f"processed {len(batch_results['users'])}/{len(user_batch)} users")
                    break
            
            # Compute metrics for this batch
            if batch_results['recommendations']:
                batch_metrics = self._compute_batch_metrics(batch_results)
                batch_results['metrics'] = batch_metrics
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Error in batch {batch_id}: {str(e)}")
            return batch_results
    
    def _compute_batch_metrics(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute metrics for a batch of recommendations and predictions."""
        # Get ground truth for these users
        user_ids = list(batch_results['recommendations'].keys())
        ground_truth = {}
        
        for user_id in user_ids:
            user_test_movies = self.test_data[
                self.test_data['user_index'] == int(user_id)
            ]['canonical_id'].tolist()
            if user_test_movies:
                ground_truth[user_id] = user_test_movies
        
        metrics = {}
        
        # Compute ranking metrics
        if ground_truth:
            ranking_results = self.metrics.evaluate_ranking_metrics(
                ground_truth, batch_results['recommendations']
            )
            metrics['ranking_metrics'] = ranking_results
        
        # Compute prediction metrics
        if batch_results['predictions']:
            true_ratings = []
            pred_ratings = []
            
            for pred_key, pred_rating in batch_results['predictions'].items():
                user_id, movie_id = pred_key.split('_', 1)
                user_id = int(user_id)
                
                # Get true rating
                true_rating = self.test_data[
                    (self.test_data['user_index'] == user_id) & 
                    (self.test_data['canonical_id'] == movie_id)
                ]['rating'].iloc[0]
                
                true_ratings.append(true_rating)
                pred_ratings.append(pred_rating)
            
            if true_ratings:
                prediction_results = self.metrics.evaluate_prediction_metrics(
                    true_ratings, pred_ratings
                )
                metrics['prediction_metrics'] = prediction_results
        
        # Compute coverage metrics
        all_users = list(set(self.train_data['user_index'].astype(str).tolist() + 
                           self.test_data['user_index'].astype(str).tolist()))
        all_items = list(set(self.train_data['canonical_id'].astype(str).tolist() + 
                           self.test_data['canonical_id'].astype(str).tolist()))
        
        coverage_results = self.metrics.evaluate_coverage_metrics(
            batch_results['recommendations'], all_users, all_items
        )
        metrics['coverage_metrics'] = coverage_results
        
        return metrics
    
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
        all_predictions = {}
        all_metrics = []
        
        for i in range(0, len(test_users), self.batch_size):
            batch_users = test_users[i:i + self.batch_size]
            batch_id = f"batch_{i//self.batch_size + 1}"
            
            self.logger.info(f"Processing {batch_id}: users {i+1}-{min(i+self.batch_size, len(test_users))}")
            
            # Process batch with timeout protection
            batch_results = self._process_batch(batch_users, batch_id)
            
            # Update global results
            all_recommendations.update(batch_results['recommendations'])
            all_predictions.update(batch_results['predictions'])
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
        
        # Compute final metrics
        self.logger.info("Computing final metrics...")
        final_results = self._compute_final_metrics(all_recommendations, all_predictions, all_metrics)
        
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
                              predictions: Dict[str, float], 
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
        
        # Compute final prediction metrics
        prediction_results = {}
        if predictions:
            true_ratings = []
            pred_ratings = []
            
            for pred_key, pred_rating in predictions.items():
                user_id, movie_id = pred_key.split('_', 1)
                user_id = int(user_id)
                
                # Get true rating
                true_rating = self.test_data[
                    (self.test_data['user_index'] == user_id) & 
                    (self.test_data['canonical_id'] == movie_id)
                ]['rating'].iloc[0]
                
                true_ratings.append(true_rating)
                pred_ratings.append(pred_rating)
            
            if true_ratings:
                prediction_results = self.metrics.evaluate_prediction_metrics(
                    true_ratings, pred_ratings
                )
        
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
            'prediction_metrics': prediction_results,
            'coverage_metrics': coverage_results,
            'batch_metrics': aggregated_batch_metrics,
            'num_test_users': len(recommendations),
            'num_ground_truth_users': len(ground_truth),
            'num_test_interactions': len(self.test_data),
            'num_predictions': len(predictions),
            'total_batches': self.batch_count,
            'total_users_processed': self.users_processed,
            'evaluation_time_seconds': time.time() - self.start_time,
            'cf_config': self.cf_config
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
        fig.suptitle(f'Collaborative Filtering Metrics vs K ({self.mode.upper()} Mode)', 
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
        plt.savefig(output_dir / f'cf_eval_ranking_metrics_{self.mode}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Coverage metrics bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        coverage_metrics = results['coverage_metrics']
        metrics = list(coverage_metrics.keys())
        values = list(coverage_metrics.values())
        
        bars = ax.bar(metrics, values, color=['skyblue', 'lightcoral'], alpha=0.8)
        ax.set_title(f'Collaborative Filtering Coverage Metrics ({self.mode.upper()} Mode)', 
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
        plt.savefig(output_dir / f'cf_eval_coverage_metrics_{self.mode}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Prediction metrics (if available)
        if results['prediction_metrics']:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            pred_metrics = results['prediction_metrics']
            metrics = list(pred_metrics.keys())
            values = list(pred_metrics.values())
            
            bars = ax.bar(metrics, values, color=['lightgreen', 'lightblue'], alpha=0.8)
            ax.set_title(f'Collaborative Filtering Prediction Metrics ({self.mode.upper()} Mode)', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Error Score (Lower is Better)')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(output_dir / f'cf_eval_prediction_metrics_{self.mode}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Visualizations saved to {output_dir}")
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """Save evaluation results to JSON file."""
        if output_path is None:
            output_path = f"data/eval/cf_eval_results_{self.mode}.json"
        
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
        """Run complete collaborative filtering evaluation."""
        self.logger.info(f"Starting {self.mode} mode collaborative filtering evaluation...")
        
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
                'memory_increase_mb': final_memory - self.initial_memory
            }
            
            self.logger.info(f"Evaluation completed successfully in {time.time() - self.start_time:.1f} seconds")
            return results
            
        finally:
            # Stop profiling and save
            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(25)
            
            profile_file = Path("logs") / "cf_eval_profile.txt"
            with open(profile_file, 'w') as f:
                f.write(s.getvalue())
            
            self.logger.info(f"Profile saved to {profile_file}")

def main():
    """Main execution function with CLI arguments."""
    parser = argparse.ArgumentParser(description='Collaborative Filtering Evaluation - Optimized Pipeline')
    parser.add_argument('--mode', choices=['smoke', 'speed', 'full'], default='smoke',
                       help='Evaluation mode (default: smoke)')
    parser.add_argument('--max_users', type=int, help='Override max users for mode')
    parser.add_argument('--batch_size', type=int, help='Override batch size for mode')
    parser.add_argument('--heartbeat_sec', type=int, help='Override heartbeat interval for mode')
    
    args = parser.parse_args()
    
    print(f"Starting Step 4.1.3: Collaborative Filtering Evaluation ({args.mode.upper()} mode)")
    print("="*70)
    
    # Initialize evaluator
    evaluator = CollaborativeFilteringEvaluator(mode=args.mode)
    
    # Override parameters if provided
    if args.max_users:
        evaluator.max_users = args.max_users
    if args.batch_size:
        evaluator.batch_size = args.batch_size
    if args.heartbeat_sec:
        evaluator.heartbeat_sec = args.heartbeat_sec
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Print summary
    print("\n" + "="*70)
    print(f"COLLABORATIVE FILTERING EVALUATION SUMMARY ({args.mode.upper()} MODE)")
    print("="*70)
    
    print(f"\nTest Users: {results['num_test_users']}")
    print(f"Ground Truth Users: {results['num_ground_truth_users']}")
    print(f"Test Interactions: {results['num_test_interactions']}")
    print(f"Predictions Made: {results['num_predictions']}")
    print(f"Batches Processed: {results['total_batches']}")
    print(f"Evaluation Time: {results['evaluation_time_seconds']:.1f} seconds")
    print(f"Users per Second: {results['performance_summary']['users_per_second']:.1f}")
    
    print(f"\nRanking Metrics:")
    for metric, scores in results['ranking_metrics'].items():
        print(f"  {metric.upper()}:")
        for k, score in scores.items():
            print(f"    @{k}: {score:.3f}")
    
    if results['prediction_metrics']:
        print(f"\nPrediction Metrics:")
        for metric, score in results['prediction_metrics'].items():
            print(f"  {metric.upper()}: {score:.3f}")
    
    print(f"\nCoverage Metrics:")
    for metric, score in results['coverage_metrics'].items():
        print(f"  {metric}: {score:.3f}")
    
    print(f"\nResults saved to: data/eval/cf_eval_results_{args.mode}.json")
    print(f"Visualizations saved to: data/eval/")
    print(f"Progress log: logs/cf_eval_progress.csv")
    print("="*70)

if __name__ == "__main__":
    main()
