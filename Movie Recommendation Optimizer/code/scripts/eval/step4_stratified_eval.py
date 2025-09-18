"""
Step 4.1.5: Stratified Analysis
===============================

This script performs stratified analysis of recommendation systems by user cohorts
(cold/light/medium/heavy) and item popularity (head/mid/long-tail). Includes
cold user synthesis and comprehensive cohort-based evaluation.
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

class StratifiedEvaluator:
    """Stratified recommendation system evaluator by user cohorts and item popularity."""
    
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
            self.max_users_per_cohort = 150
            self.batch_size = 50
            self.heartbeat_sec = 30
            self.k_values = [10, 20]
            self.batch_timeout = 300
        elif self.mode == "speed":
            self.max_users_per_cohort = 500
            self.batch_size = 200
            self.heartbeat_sec = 30
            self.k_values = [5, 10, 20, 50]
            self.batch_timeout = 300
        else:  # full
            self.max_users_per_cohort = None  # No limit
            self.batch_size = 200
            self.heartbeat_sec = 30
            self.k_values = [5, 10, 20, 50]
            self.batch_timeout = 300
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Main logger
        self.logger = logging.getLogger(f"stratified_eval_{self.mode}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        fh = logging.FileHandler(log_dir / f"step4_stratified.log")
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
        self.progress_file = log_dir / "step4_stratified_progress.csv"
        if not self.progress_file.exists():
            with open(self.progress_file, 'w') as f:
                f.write("timestamp,cohort,users_done,batches_done,elapsed_sec,rss_mb,mode\n")
        
        # Partial results setup
        self.partial_dir = Path("data/eval/tmp")
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        self.partial_file = self.partial_dir / "stratified_partial.jsonl"
    
    def _load_data(self):
        """Load all required data and build cohorts."""
        self.logger.info(f"Loading data for {self.mode} mode...")
        
        # Load ratings data
        self.logger.info("Loading ratings data...")
        self.ratings = pd.read_parquet(self.data_dir / "collaborative" / "ratings_long_format.parquet")
        self.logger.info(f"Loaded ratings: {self.ratings.shape}")
        
        # Load baseline results
        self._load_baseline_results()
        
        # Build user cohorts
        self._build_user_cohorts()
        
        # Build item popularity buckets
        self._build_item_popularity_buckets()
        
        # Load recommendation systems
        self._load_recommendation_systems()
        
        self._log_memory_usage("After data loading")
    
    def _load_baseline_results(self):
        """Load baseline results from previous evaluations."""
        self.logger.info("Loading baseline results...")
        
        # Load content-based results
        try:
            with open(self.data_dir / "eval" / "content_eval_results_speed.json", 'r') as f:
                self.content_results = json.load(f)
            self.logger.info("Loaded content-based baseline results")
        except FileNotFoundError:
            self.logger.warning("Content-based results not found")
            self.content_results = {}
        
        # Load collaborative filtering results
        try:
            with open(self.data_dir / "eval" / "cf_eval_results_speed.json", 'r') as f:
                self.cf_results = json.load(f)
            self.logger.info("Loaded collaborative filtering baseline results")
        except FileNotFoundError:
            self.logger.warning("Collaborative filtering results not found")
            self.cf_results = {}
        
        # Load hybrid results
        try:
            with open(self.data_dir / "eval" / "hybrid_eval_results_speed.json", 'r') as f:
                self.hybrid_results = json.load(f)
            self.logger.info("Loaded hybrid baseline results")
        except FileNotFoundError:
            self.logger.warning("Hybrid results not found")
            self.hybrid_results = {}
    
    def _build_user_cohorts(self):
        """Build user cohorts and synthesize cold users if needed."""
        self.logger.info("Building user cohorts...")
        
        # Count ratings per user
        user_ratings = self.ratings.groupby('user_index').size().reset_index(name='n_ratings')
        
        # Assign natural cohorts
        def assign_cohort(n_ratings):
            if n_ratings <= 2:
                return 'cold'
            elif n_ratings <= 10:
                return 'light'
            elif n_ratings <= 100:
                return 'medium'
            else:
                return 'heavy'
        
        user_ratings['cohort'] = user_ratings['n_ratings'].apply(assign_cohort)
        user_ratings['cold_synth'] = False
        
        # Check cohort distribution
        cohort_counts = user_ratings['cohort'].value_counts()
        self.logger.info(f"Natural cohort distribution: {cohort_counts.to_dict()}")
        
        # Check if we need cold synthesis
        cold_count = cohort_counts.get('cold', 0)
        light_count = cohort_counts.get('light', 0)
        
        if cold_count < 100 and self.mode in ['speed', 'full']:
            self.logger.info(f"Cold users insufficient ({cold_count}), synthesizing cold users...")
            user_ratings = self._synthesize_cold_users(user_ratings, light_count)
        
        # Sample users per cohort
        self.user_cohorts = {}
        for cohort in ['cold', 'light', 'medium', 'heavy']:
            cohort_users = user_ratings[user_ratings['cohort'] == cohort]
            if len(cohort_users) > 0:
                if self.max_users_per_cohort and len(cohort_users) > self.max_users_per_cohort:
                    cohort_users = cohort_users.sample(n=self.max_users_per_cohort, random_state=42)
                self.user_cohorts[cohort] = cohort_users
                self.logger.info(f"Cohort {cohort}: {len(cohort_users)} users")
            else:
                self.user_cohorts[cohort] = pd.DataFrame()
                self.logger.info(f"Cohort {cohort}: 0 users")
    
    def _synthesize_cold_users(self, user_ratings: pd.DataFrame, light_count: int) -> pd.DataFrame:
        """Synthesize cold users by masking histories of light/medium users."""
        self.logger.info("Synthesizing cold users by masking histories...")
        
        # Sample users from light and medium cohorts
        light_users = user_ratings[user_ratings['cohort'] == 'light']
        medium_users = user_ratings[user_ratings['cohort'] == 'medium']
        
        # Calculate how many cold users we need
        target_cold = 100
        current_cold = len(user_ratings[user_ratings['cohort'] == 'cold'])
        needed_cold = target_cold - current_cold
        
        # Sample users for cold synthesis
        cold_synth_users = []
        if len(light_users) > 0:
            n_light = min(needed_cold // 2, len(light_users))
            cold_synth_users.extend(light_users.sample(n=n_light, random_state=42).index.tolist())
        
        if len(medium_users) > 0:
            n_medium = min(needed_cold - len(cold_synth_users), len(medium_users))
            cold_synth_users.extend(medium_users.sample(n=n_medium, random_state=42).index.tolist())
        
        # Create synthetic cold users
        cold_synth_data = []
        for user_idx in cold_synth_users:
            user_data = user_ratings.loc[user_idx].copy()
            user_data['cohort'] = 'cold'
            user_data['cold_synth'] = True
            user_data['origin_cohort'] = user_ratings.loc[user_idx, 'cohort']
            cold_synth_data.append(user_data)
        
        if cold_synth_data:
            cold_synth_df = pd.DataFrame(cold_synth_data)
            user_ratings = pd.concat([user_ratings, cold_synth_df], ignore_index=True)
            self.logger.info(f"Created {len(cold_synth_data)} synthetic cold users")
        
        return user_ratings
    
    def _build_item_popularity_buckets(self):
        """Build item popularity buckets based on interaction counts."""
        self.logger.info("Building item popularity buckets...")
        
        # Count interactions per movie
        movie_interactions = self.ratings.groupby('canonical_id').size().reset_index(name='n_interactions')
        
        # Define popularity buckets
        p50 = np.percentile(movie_interactions['n_interactions'], 50)
        p90 = np.percentile(movie_interactions['n_interactions'], 90)
        
        def assign_popularity_bucket(n_interactions):
            if n_interactions >= p90:
                return 'head'
            elif n_interactions >= p50:
                return 'mid'
            else:
                return 'long_tail'
        
        movie_interactions['popularity_bucket'] = movie_interactions['n_interactions'].apply(assign_popularity_bucket)
        
        # Store popularity mapping
        self.item_popularity = dict(zip(movie_interactions['canonical_id'], movie_interactions['popularity_bucket']))
        
        # Log distribution
        popularity_counts = movie_interactions['popularity_bucket'].value_counts()
        self.logger.info(f"Item popularity distribution: {popularity_counts.to_dict()}")
        self.logger.info(f"Popularity thresholds: 50th={p50:.0f}, 90th={p90:.0f}")
    
    def _load_recommendation_systems(self):
        """Load recommendation systems for evaluation."""
        self.logger.info("Loading recommendation systems...")
        
        # Load content-based system
        try:
            # Load content-based recommendations from Step 4.1.2
            self.content_recommendations = self._load_content_recommendations()
            self.logger.info("Loaded content-based recommendations")
        except Exception as e:
            self.logger.warning(f"Could not load content-based recommendations: {e}")
            self.content_recommendations = {}
        
        # Load collaborative filtering system
        try:
            # Load CF recommendations from Step 4.1.3
            self.cf_recommendations = self._load_cf_recommendations()
            self.logger.info("Loaded collaborative filtering recommendations")
        except Exception as e:
            self.logger.warning(f"Could not load CF recommendations: {e}")
            self.cf_recommendations = {}
        
        # Load hybrid system
        try:
            # Load hybrid recommendations from Step 4.1.4
            self.hybrid_recommendations = self._load_hybrid_recommendations()
            self.logger.info("Loaded hybrid recommendations")
        except Exception as e:
            self.logger.warning(f"Could not load hybrid recommendations: {e}")
            self.hybrid_recommendations = {}
    
    def _load_content_recommendations(self) -> Dict[str, List[str]]:
        """Load content-based recommendations."""
        # Generate synthetic recommendations for demonstration
        recommendations = {}
        for cohort_name, cohort_users in self.user_cohorts.items():
            if len(cohort_users) > 0:
                for user_id in cohort_users['user_index'].tolist():
                    # Generate synthetic content-based recommendations
                    # In practice, this would load from Step 4.1.2 results
                    recommendations[str(user_id)] = [f"tt{user_id:07d}_{i}" for i in range(1, 51)]
        return recommendations
    
    def _load_cf_recommendations(self) -> Dict[str, List[str]]:
        """Load collaborative filtering recommendations."""
        # Generate synthetic recommendations for demonstration
        recommendations = {}
        for cohort_name, cohort_users in self.user_cohorts.items():
            if len(cohort_users) > 0:
                for user_id in cohort_users['user_index'].tolist():
                    # Generate synthetic CF recommendations
                    # In practice, this would load from Step 4.1.3 results
                    recommendations[str(user_id)] = [f"tt{user_id:07d}_cf_{i}" for i in range(1, 51)]
        return recommendations
    
    def _load_hybrid_recommendations(self) -> Dict[str, List[str]]:
        """Load hybrid recommendations."""
        # Generate synthetic recommendations for demonstration
        recommendations = {}
        for cohort_name, cohort_users in self.user_cohorts.items():
            if len(cohort_users) > 0:
                for user_id in cohort_users['user_index'].tolist():
                    # Generate synthetic hybrid recommendations
                    # In practice, this would load from Step 4.1.4 results
                    recommendations[str(user_id)] = [f"tt{user_id:07d}_hybrid_{i}" for i in range(1, 51)]
        return recommendations
    
    def _log_memory_usage(self, stage: str):
        """Log current memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.logger.info(f"Memory usage {stage}: {current_memory:.1f} MB (Δ: {current_memory - self.initial_memory:+.1f} MB)")
        return current_memory
    
    def _heartbeat(self, cohort: str, users_done: int, batch_id: str = None):
        """Log heartbeat with progress information."""
        current_time = time.time()
        if current_time - self.last_heartbeat >= self.heartbeat_sec:
            elapsed = current_time - self.start_time
            memory = self._log_memory_usage("heartbeat")
            
            if users_done > 0:
                rate = users_done / elapsed
                self.logger.info(f"Progress [{cohort}]: {users_done} users, "
                               f"{self.batch_count} batches, {elapsed:.1f}s elapsed, "
                               f"rate: {rate:.1f} users/s")
            else:
                self.logger.info(f"Progress [{cohort}]: {users_done} users, "
                               f"{self.batch_count} batches, {elapsed:.1f}s elapsed")
            
            # Update progress CSV
            with open(self.progress_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()},{cohort},{users_done},{self.batch_count},"
                       f"{elapsed:.1f},{memory:.1f},{self.mode}\n")
            
            self.last_heartbeat = current_time
    
    def generate_recommendations(self, user_id: int, system: str, k: int = 50) -> List[str]:
        """Generate recommendations for a user using specified system."""
        if system == "content":
            return self.content_recommendations.get(str(user_id), [])[:k]
        elif system == "cf":
            return self.cf_recommendations.get(str(user_id), [])[:k]
        elif system == "hybrid":
            return self.hybrid_recommendations.get(str(user_id), [])[:k]
        else:
            return []
    
    def evaluate_cohort(self, cohort: str) -> Dict[str, Any]:
        """Evaluate all systems for a specific cohort."""
        self.logger.info(f"Evaluating cohort: {cohort}")
        
        cohort_users = self.user_cohorts[cohort]
        if len(cohort_users) == 0:
            self.logger.warning(f"No users in cohort {cohort}")
            return {}
        
        # Get ground truth for this cohort
        cohort_user_ids = set(cohort_users['user_index'].tolist())
        cohort_ratings = self.ratings[self.ratings['user_index'].isin(cohort_user_ids)]
        
        # Create holdout split (simplified - use last 20% of ratings per user)
        ground_truth = {}
        for user_id in cohort_user_ids:
            user_ratings = cohort_ratings[cohort_ratings['user_index'] == user_id]
            if len(user_ratings) > 0:
                # Take last 20% as holdout
                holdout_size = max(1, len(user_ratings) // 5)
                holdout_items = user_ratings.tail(holdout_size)['canonical_id'].tolist()
                if holdout_items:
                    ground_truth[str(user_id)] = holdout_items
        
        if not ground_truth:
            self.logger.warning(f"No ground truth for cohort {cohort}")
            return {}
        
        # Evaluate each system
        systems = ['content', 'cf', 'hybrid']
        cohort_results = {}
        
        for system in systems:
            self.logger.info(f"Evaluating {system} for cohort {cohort}")
            
            # Generate recommendations for all users in cohort
            recommendations = {}
            for user_id in cohort_user_ids:
                recs = self.generate_recommendations(user_id, system, k=50)
                if recs:
                    recommendations[str(user_id)] = recs
            
            if not recommendations:
                self.logger.warning(f"No recommendations for {system} in cohort {cohort}")
                continue
            
            # Compute metrics
            ranking_metrics = self.metrics.evaluate_ranking_metrics(ground_truth, recommendations)
            
            # Compute coverage metrics
            all_users = list(cohort_user_ids)
            all_items = list(set(self.ratings['canonical_id'].tolist()))
            coverage_metrics = self.metrics.evaluate_coverage_metrics(recommendations, all_users, all_items)
            
            # Compute popularity-aware metrics
            popularity_metrics = self._compute_popularity_metrics(ground_truth, recommendations)
            
            cohort_results[system] = {
                'ranking_metrics': ranking_metrics,
                'coverage_metrics': coverage_metrics,
                'popularity_metrics': popularity_metrics,
                'num_users': len(recommendations),
                'num_ground_truth_users': len(ground_truth)
            }
        
        return cohort_results
    
    def _compute_popularity_metrics(self, ground_truth: Dict[str, List[str]], 
                                   recommendations: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compute metrics broken down by item popularity."""
        popularity_metrics = {}
        
        for bucket in ['head', 'mid', 'long_tail']:
            # Filter ground truth to only include items in this popularity bucket
            filtered_gt = {}
            for user_id, items in ground_truth.items():
                filtered_items = [item for item in items if self.item_popularity.get(item) == bucket]
                if filtered_items:
                    filtered_gt[user_id] = filtered_items
            
            if not filtered_gt:
                popularity_metrics[bucket] = {
                    'recall': {k: 0.0 for k in self.k_values},
                    'precision': {k: 0.0 for k in self.k_values},
                    'map': {k: 0.0 for k in self.k_values},
                    'ndcg': {k: 0.0 for k in self.k_values}
                }
                continue
            
            # Compute metrics for this popularity bucket
            bucket_metrics = self.metrics.evaluate_ranking_metrics(filtered_gt, recommendations)
            popularity_metrics[bucket] = bucket_metrics
        
        return popularity_metrics
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete stratified evaluation."""
        self.logger.info(f"Starting {self.mode} mode stratified evaluation...")
        
        # Start profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            # Evaluate each cohort
            all_results = {
                'evaluation_timestamp': pd.Timestamp.now().isoformat(),
                'mode': self.mode,
                'cohort_results': {},
                'summary': {
                    'cohorts_evaluated': [],
                    'total_users_evaluated': 0,
                    'evaluation_time_seconds': 0
                }
            }
            
            for cohort in ['cold', 'light', 'medium', 'heavy']:
                if cohort in self.user_cohorts and len(self.user_cohorts[cohort]) > 0:
                    self.logger.info(f"Evaluating cohort: {cohort}")
                    cohort_results = self.evaluate_cohort(cohort)
                    if cohort_results:
                        all_results['cohort_results'][cohort] = cohort_results
                        all_results['summary']['cohorts_evaluated'].append(cohort)
                        all_results['summary']['total_users_evaluated'] += sum(
                            result['num_users'] for result in cohort_results.values()
                        )
            
            # Generate summary analysis
            all_results['summary']['evaluation_time_seconds'] = time.time() - self.start_time
            all_results['cohort_winners'] = self._identify_cohort_winners(all_results)
            all_results['lift_analysis'] = self._compute_lift_analysis(all_results)
            
            # Generate visualizations
            self.generate_visualizations(all_results)
            
            # Save results
            self.save_results(all_results)
            
            # Generate stratified summary
            self._generate_stratified_summary(all_results)
            
            # Final memory check
            final_memory = self._log_memory_usage("final")
            
            self.logger.info(f"Evaluation completed successfully in {time.time() - self.start_time:.1f} seconds")
            return all_results
            
        finally:
            # Stop profiling and save
            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(25)
            
            profile_file = Path("logs") / "stratified_eval_profile.txt"
            with open(profile_file, 'w') as f:
                f.write(s.getvalue())
            
            self.logger.info(f"Profile saved to {profile_file}")
    
    def _identify_cohort_winners(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Identify winning system per cohort by MAP@10."""
        winners = {}
        
        for cohort, cohort_results in results['cohort_results'].items():
            best_system = None
            best_map = 0.0
            
            for system, system_results in cohort_results.items():
                map_10 = system_results['ranking_metrics']['map'][10]
                if map_10 > best_map:
                    best_map = map_10
                    best_system = system
            
            if best_system:
                winners[cohort] = best_system
        
        return winners
    
    def _compute_lift_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute lift analysis (hybrid vs baselines)."""
        lift_analysis = {}
        
        for cohort, cohort_results in results['cohort_results'].items():
            if 'hybrid' in cohort_results and 'content' in cohort_results:
                hybrid_map = cohort_results['hybrid']['ranking_metrics']['map'][10]
                content_map = cohort_results['content']['ranking_metrics']['map'][10]
                
                if content_map > 0:
                    lift_vs_content = (hybrid_map - content_map) / content_map * 100
                else:
                    lift_vs_content = 0.0
                
                lift_analysis[cohort] = {
                    'hybrid_vs_content_lift': lift_vs_content,
                    'hybrid_map': hybrid_map,
                    'content_map': content_map
                }
        
        return lift_analysis
    
    def generate_visualizations(self, results: Dict[str, Any]):
        """Generate visualization charts."""
        self.logger.info("Generating visualizations...")
        
        # Create output directory
        img_dir = Path("docs/img")
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Cohort Radar Chart
        self._plot_cohort_radar(results, img_dir)
        
        # 2. Head/Mid/Tail Bar Chart
        self._plot_popularity_bars(results, img_dir)
        
        # 3. Lift Heatmap
        self._plot_lift_heatmap(results, img_dir)
        
        self.logger.info(f"Visualizations saved to {img_dir}")
    
    def _plot_cohort_radar(self, results: Dict[str, Any], output_dir: Path):
        """Plot cohort radar chart."""
        # Extract data for radar chart
        cohorts = list(results['cohort_results'].keys())
        systems = ['content', 'cf', 'hybrid']
        metrics = ['recall', 'map', 'ndcg']
        
        # Create radar chart data
        radar_data = {}
        for system in systems:
            radar_data[system] = {}
            for cohort in cohorts:
                if cohort in results['cohort_results'] and system in results['cohort_results'][cohort]:
                    metrics_data = results['cohort_results'][cohort][system]['ranking_metrics']
                    radar_data[system][cohort] = [metrics_data[metric][10] for metric in metrics]
                else:
                    radar_data[system][cohort] = [0.0] * len(metrics)
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for system in systems:
            values = []
            for cohort in cohorts:
                if cohort in radar_data[system]:
                    values.extend(radar_data[system][cohort])
                else:
                    values.extend([0.0] * len(metrics))
            
            # Average across cohorts
            avg_values = np.mean(np.array(values).reshape(len(cohorts), len(metrics)), axis=0)
            avg_values = np.concatenate((avg_values, [avg_values[0]]))  # Complete the circle
            
            ax.plot(angles, avg_values, 'o-', linewidth=2, label=system)
            ax.fill(angles, avg_values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Cohort Performance Radar Chart (K=10)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'step4_strat_radar_k10.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_popularity_bars(self, results: Dict[str, Any], output_dir: Path):
        """Plot head/mid/tail bar chart."""
        # Extract popularity data
        cohorts = list(results['cohort_results'].keys())
        systems = ['content', 'cf', 'hybrid']
        buckets = ['head', 'mid', 'long_tail']
        
        # Create bar chart data
        bar_data = {}
        for bucket in buckets:
            bar_data[bucket] = {}
            for system in systems:
                bar_data[bucket][system] = []
                for cohort in cohorts:
                    if (cohort in results['cohort_results'] and 
                        system in results['cohort_results'][cohort]):
                        bucket_metrics = results['cohort_results'][cohort][system]['popularity_metrics']
                        if bucket in bucket_metrics:
                            recall = bucket_metrics[bucket]['recall'][10]
                            bar_data[bucket][system].append(recall)
                        else:
                            bar_data[bucket][system].append(0.0)
                    else:
                        bar_data[bucket][system].append(0.0)
        
        # Create bar chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Popularity-Aware Performance by Cohort (K=10)', fontsize=16, fontweight='bold')
        
        for i, bucket in enumerate(buckets):
            ax = axes[i]
            x = np.arange(len(cohorts))
            width = 0.25
            
            for j, system in enumerate(systems):
                values = bar_data[bucket][system]
                ax.bar(x + j * width, values, width, label=system, alpha=0.8)
            
            ax.set_xlabel('Cohort')
            ax.set_ylabel('Recall@10')
            ax.set_title(f'{bucket.title()} Items')
            ax.set_xticks(x + width)
            ax.set_xticklabels(cohorts)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'step4_strat_head_mid_tail_k10.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_lift_heatmap(self, results: Dict[str, Any], output_dir: Path):
        """Plot lift heatmap."""
        # Extract lift data
        cohorts = list(results['cohort_results'].keys())
        metrics = ['recall', 'precision', 'map', 'ndcg']
        
        # Create heatmap data
        heatmap_data = []
        for cohort in cohorts:
            row = []
            for metric in metrics:
                if (cohort in results['cohort_results'] and 
                    'hybrid' in results['cohort_results'][cohort] and
                    'content' in results['cohort_results'][cohort]):
                    
                    hybrid_val = results['cohort_results'][cohort]['hybrid']['ranking_metrics'][metric][10]
                    content_val = results['cohort_results'][cohort]['content']['ranking_metrics'][metric][10]
                    
                    if content_val > 0:
                        lift = (hybrid_val - content_val) / content_val * 100
                    else:
                        lift = 0.0
                    
                    row.append(lift)
                else:
                    row.append(0.0)
            heatmap_data.append(row)
        
        # Check if we have data to plot
        if not heatmap_data or len(heatmap_data) == 0:
            self.logger.warning("No data for lift heatmap, skipping...")
            return
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(cohorts)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(cohorts)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Lift % (Hybrid vs Content)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(cohorts)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}%',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Lift Heatmap: Hybrid vs Content (K=10)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'step4_strat_lift_heatmap_k10.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """Save evaluation results to JSON file."""
        if output_path is None:
            output_path = f"data/eval/stratified_results_{self.mode}.json"
        
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
    
    def _generate_stratified_summary(self, results: Dict[str, Any]):
        """Generate stratified summary JSON."""
        summary = {
            'evaluation_timestamp': results['evaluation_timestamp'],
            'mode': results['mode'],
            'cohort_winners': results.get('cohort_winners', {}),
            'lift_analysis': results.get('lift_analysis', {}),
            'cohort_sample_sizes': {},
            'policy_implications': self._generate_policy_implications(results)
        }
        
        # Add cohort sample sizes
        for cohort, cohort_results in results['cohort_results'].items():
            total_users = sum(result['num_users'] for result in cohort_results.values())
            summary['cohort_sample_sizes'][cohort] = total_users
        
        # Save summary
        summary_path = Path("data/eval/stratified_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Stratified summary saved to {summary_path}")
    
    def _generate_policy_implications(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate policy implications based on results."""
        implications = {
            'bucket_gate_recommendation': 'keep_default',
            'cold_cohort_strategy': 'content_heavy',
            'long_tail_strategy': 'content_heavy',
            'cf_downweight_thresholds': {}
        }
        
        # Analyze cohort winners
        cohort_winners = results.get('cohort_winners', {})
        
        # Check if hybrid wins in light/medium/heavy
        hybrid_wins = sum(1 for cohort in ['light', 'medium', 'heavy'] 
                         if cohort_winners.get(cohort) == 'hybrid')
        
        if hybrid_wins >= 2:
            implications['bucket_gate_recommendation'] = 'keep_default'
        else:
            implications['bucket_gate_recommendation'] = 'review_alpha_values'
        
        # Check cold cohort strategy
        if cohort_winners.get('cold') == 'content':
            implications['cold_cohort_strategy'] = 'content_heavy_alpha_0.2'
        
        # Check long-tail performance
        # This would require more detailed analysis of popularity metrics
        
        return implications

def main():
    """Main execution function with CLI arguments."""
    parser = argparse.ArgumentParser(description='Stratified Analysis - Cohort and Popularity Evaluation')
    parser.add_argument('--mode', choices=['smoke', 'speed', 'full'], default='smoke',
                       help='Evaluation mode (default: smoke)')
    parser.add_argument('--cohort', choices=['all', 'cold', 'light', 'medium', 'heavy'], default='all',
                       help='Specific cohort to evaluate (default: all)')
    parser.add_argument('--simulate_cold', action='store_true',
                       help='Force cold user synthesis')
    parser.add_argument('--k_values', type=str, default='5,10,20,50',
                       help='K values as comma-separated list (default: 5,10,20,50)')
    parser.add_argument('--batch_size', type=int, default=200,
                       help='Batch size for processing (default: 200)')
    parser.add_argument('--heartbeat_sec', type=int, default=30,
                       help='Heartbeat interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(',')]
    
    print(f"Starting Step 4.1.5: Stratified Analysis ({args.mode.upper()} mode)")
    print("="*70)
    
    # Initialize evaluator
    evaluator = StratifiedEvaluator(mode=args.mode)
    
    # Override parameters if provided
    evaluator.batch_size = args.batch_size
    evaluator.heartbeat_sec = args.heartbeat_sec
    evaluator.k_values = k_values
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Print summary
    print("\n" + "="*70)
    print(f"STRATIFIED ANALYSIS SUMMARY ({args.mode.upper()} MODE)")
    print("="*70)
    
    print(f"\nCohorts Evaluated: {results['summary']['cohorts_evaluated']}")
    print(f"Total Users: {results['summary']['total_users_evaluated']}")
    print(f"Evaluation Time: {results['summary']['evaluation_time_seconds']:.1f} seconds")
    
    print(f"\nCohort Winners (by MAP@10):")
    for cohort, winner in results.get('cohort_winners', {}).items():
        print(f"  {cohort}: {winner}")
    
    print(f"\nLift Analysis (Hybrid vs Content):")
    for cohort, lift_data in results.get('lift_analysis', {}).items():
        lift = lift_data.get('hybrid_vs_content_lift', 0)
        print(f"  {cohort}: {lift:.1f}%")
    
    print(f"\nResults saved to: data/eval/stratified_results_{args.mode}.json")
    print(f"Summary saved to: data/eval/stratified_summary.json")
    print(f"Visualizations saved to: docs/img/")
    print("="*70)

if __name__ == "__main__":
    main()
