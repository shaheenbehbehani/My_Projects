"""
Step 4.1.2: Content-Based Evaluation
====================================

This script evaluates the content-based recommendation system from Step 3a
using the metrics framework from Step 4.1.1.

Features:
- Uses composite embeddings and kNN neighbors from Step 3a
- Applies both holdout split and user-sampled split evaluation strategies
- Computes ranking metrics (Recall@K, Precision@K, MAP@K, NDCG@K) for K=[5,10,20,50]
- Computes coverage metrics (user and item coverage)
- Generates visualizations and detailed reports
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from scripts.eval.metrics import MetricsFramework, MetricConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentBasedEvaluator:
    """Content-based recommendation system evaluator."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.results = {}
        
        # Initialize metrics framework
        self.metrics = MetricsFramework()
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load all required data from Step 3a and ground truth."""
        logger.info("Loading data from Step 3a and ground truth...")
        
        # Load kNN neighbors data
        self.neighbors_df = pd.read_parquet(self.data_dir / "similarity" / "movies_neighbors_k50.parquet")
        logger.info(f"Loaded neighbors data: {self.neighbors_df.shape}")
        
        # Load movie metadata
        self.movie_metadata = pd.read_parquet(self.data_dir / "features" / "composite" / "movies_features_v1.parquet")
        logger.info(f"Loaded movie metadata: {self.movie_metadata.shape}")
        
        # Load ground truth data
        self.train_data = pd.read_parquet(self.data_dir / "eval" / "checkpoints" / "ratings_split_train.parquet")
        self.test_data = pd.read_parquet(self.data_dir / "eval" / "checkpoints" / "ratings_split_test.parquet")
        logger.info(f"Loaded train data: {self.train_data.shape}")
        logger.info(f"Loaded test data: {self.test_data.shape}")
        
        # Create movie ID to index mapping
        self.movie_id_to_idx = dict(zip(self.movie_metadata['canonical_id'], 
                                      self.movie_metadata['canonical_idx']))
        self.idx_to_movie_id = dict(zip(self.movie_metadata['canonical_idx'], 
                                      self.movie_metadata['canonical_id']))
        
        logger.info(f"Created movie ID mappings for {len(self.movie_id_to_idx)} movies")
    
    def generate_content_recommendations(self, user_movies: List[str], k: int = 50) -> List[str]:
        """
        Generate content-based recommendations for a user based on their movie history.
        
        Args:
            user_movies: List of canonical movie IDs the user has rated
            k: Number of recommendations to generate
            
        Returns:
            List of recommended movie IDs
        """
        if not user_movies:
            return []
        
        # Get all neighbors of user's movies
        all_neighbors = []
        neighbor_scores = {}
        
        for movie_id in user_movies:
            if movie_id in self.movie_id_to_idx:
                # Get neighbors for this movie
                movie_neighbors = self.neighbors_df[
                    self.neighbors_df['movie_id'] == movie_id
                ].sort_values('rank')
                
                for _, neighbor in movie_neighbors.iterrows():
                    neighbor_id = neighbor['neighbor_id']
                    score = neighbor['score']
                    
                    # Skip movies the user has already rated
                    if neighbor_id not in user_movies:
                        if neighbor_id not in neighbor_scores:
                            neighbor_scores[neighbor_id] = []
                        neighbor_scores[neighbor_id].append(score)
        
        if not neighbor_scores:
            return []
        
        # Aggregate scores (use max score for each movie)
        aggregated_scores = {movie_id: max(scores) for movie_id, scores in neighbor_scores.items()}
        
        # Sort by score and return top-k
        sorted_movies = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [movie_id for movie_id, _ in sorted_movies[:k]]
        
        return recommendations
    
    def evaluate_holdout_split(self) -> Dict[str, Any]:
        """Evaluate content-based system using holdout split strategy."""
        logger.info("Running holdout split evaluation...")
        
        # Generate recommendations for test users
        test_users = self.test_data['user_index'].unique()
        recommendations = {}
        
        logger.info(f"Generating recommendations for {len(test_users)} test users...")
        
        for user_idx in test_users:
            # Get user's training movies
            user_train_movies = self.train_data[
                self.train_data['user_index'] == user_idx
            ]['canonical_id'].tolist()
            
            # Generate recommendations
            user_recs = self.generate_content_recommendations(user_train_movies, k=50)
            recommendations[str(user_idx)] = user_recs
        
        # Evaluate using metrics framework
        results = self.metrics.evaluate_holdout_split(
            self.train_data, self.test_data, recommendations
        )
        
        # Add content-specific analysis
        results['content_analysis'] = self._analyze_content_performance(recommendations)
        
        return results
    
    def evaluate_user_sampled_split(self, sample_ratio: float = 0.2) -> Dict[str, Any]:
        """Evaluate content-based system using user-sampled split strategy."""
        logger.info(f"Running user-sampled split evaluation (sample_ratio={sample_ratio})...")
        
        # Prepare user data for sampled split
        user_data = {}
        for user_idx in self.train_data['user_index'].unique():
            user_movies = self.train_data[
                self.train_data['user_index'] == user_idx
            ][['canonical_id', 'rating']].values.tolist()
            user_data[str(user_idx)] = [(movie_id, rating) for movie_id, rating in user_movies]
        
        # Generate recommendations for all users
        recommendations = {}
        for user_idx, movies in user_data.items():
            movie_ids = [movie_id for movie_id, _ in movies]
            user_recs = self.generate_content_recommendations(movie_ids, k=50)
            recommendations[user_idx] = user_recs
        
        # Evaluate using metrics framework
        results = self.metrics.evaluate_user_sampled_split(
            user_data, recommendations, sample_ratio
        )
        
        # Add content-specific analysis
        results['content_analysis'] = self._analyze_content_performance(recommendations)
        
        return results
    
    def _analyze_content_performance(self, recommendations: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze content-based performance characteristics."""
        logger.info("Analyzing content-based performance...")
        
        # Analyze recommendation diversity
        all_recommended_movies = set()
        for user_recs in recommendations.values():
            all_recommended_movies.update(user_recs)
        
        # Analyze movie popularity in recommendations
        movie_recommendation_counts = {}
        for user_recs in recommendations.values():
            for movie_id in user_recs:
                movie_recommendation_counts[movie_id] = movie_recommendation_counts.get(movie_id, 0) + 1
        
        # Get movie metadata for analysis
        recommended_movies_metadata = self.movie_metadata[
            self.movie_metadata['canonical_id'].isin(all_recommended_movies)
        ].copy()
        
        # Add recommendation frequency
        recommended_movies_metadata['rec_frequency'] = recommended_movies_metadata['canonical_id'].map(
            movie_recommendation_counts
        )
        
        # Analyze by feature family contributions
        feature_analysis = {}
        for family in ['bert', 'tfidf', 'genres', 'crew', 'numeric', 'platform']:
            norm_col = f'{family}_norm'
            if norm_col in recommended_movies_metadata.columns:
                feature_analysis[f'{family}_avg_norm'] = recommended_movies_metadata[norm_col].mean()
                feature_analysis[f'{family}_std_norm'] = recommended_movies_metadata[norm_col].std()
        
        analysis = {
            'total_recommended_movies': len(all_recommended_movies),
            'total_movies_in_catalog': len(self.movie_metadata),
            'recommendation_diversity': len(all_recommended_movies) / len(self.movie_metadata),
            'avg_recommendations_per_user': np.mean([len(recs) for recs in recommendations.values()]),
            'max_recommendation_frequency': max(movie_recommendation_counts.values()) if movie_recommendation_counts else 0,
            'feature_analysis': feature_analysis,
            'top_recommended_movies': sorted(movie_recommendation_counts.items(), 
                                           key=lambda x: x[1], reverse=True)[:10]
        }
        
        return analysis
    
    def generate_visualizations(self, results: Dict[str, Any], output_dir: str = "data/eval"):
        """Generate visualization charts for content-based evaluation."""
        logger.info("Generating visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Ranking metrics vs K
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Content-Based Recommendation Metrics vs K', fontsize=16, fontweight='bold')
        
        k_values = self.metrics.config.k_values
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
        plt.savefig(output_dir / 'content_eval_ranking_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Coverage metrics bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        coverage_metrics = results['coverage_metrics']
        metrics = list(coverage_metrics.keys())
        values = list(coverage_metrics.values())
        
        bars = ax.bar(metrics, values, color=['skyblue', 'lightcoral'], alpha=0.8)
        ax.set_title('Content-Based Recommendation Coverage Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Coverage Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'content_eval_coverage_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature contribution analysis (if available)
        if 'content_analysis' in results and 'feature_analysis' in results['content_analysis']:
            feature_analysis = results['content_analysis']['feature_analysis']
            
            if feature_analysis:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                families = [k.replace('_avg_norm', '') for k in feature_analysis.keys() if k.endswith('_avg_norm')]
                avg_norms = [feature_analysis[f'{f}_avg_norm'] for f in families]
                
                bars = ax.bar(families, avg_norms, color=plt.cm.Set3(np.linspace(0, 1, len(families))))
                ax.set_title('Average Feature Family Norms in Recommendations', fontsize=14, fontweight='bold')
                ax.set_ylabel('Average L2 Norm')
                ax.set_xlabel('Feature Family')
                
                # Add value labels
                for bar, value in zip(bars, avg_norms):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / 'content_eval_feature_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def save_results(self, results: Dict[str, Any], output_path: str = "data/eval/content_eval_results.json"):
        """Save evaluation results to JSON file."""
        logger.info(f"Saving results to {output_path}")
        
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
        
        logger.info(f"Results saved to {output_path}")
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete content-based evaluation."""
        logger.info("Starting full content-based evaluation...")
        
        # Run both evaluation strategies
        holdout_results = self.evaluate_holdout_split()
        user_sampled_results = self.evaluate_user_sampled_split()
        
        # Combine results
        full_results = {
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'evaluation_strategies': {
                'holdout_split': holdout_results,
                'user_sampled_split': user_sampled_results
            },
            'summary': {
                'total_movies_in_catalog': len(self.movie_metadata),
                'total_train_interactions': len(self.train_data),
                'total_test_interactions': len(self.test_data),
                'unique_test_users': len(self.test_data['user_index'].unique()),
                'k_values_evaluated': self.metrics.config.k_values
            }
        }
        
        # Generate visualizations
        self.generate_visualizations(holdout_results)
        
        # Save results
        self.save_results(full_results)
        
        logger.info("Content-based evaluation completed successfully!")
        return full_results

def main():
    """Main execution function."""
    logger.info("Starting Step 4.1.2: Content-Based Evaluation")
    
    # Initialize evaluator
    evaluator = ContentBasedEvaluator()
    
    # Run full evaluation
    results = evaluator.run_full_evaluation()
    
    # Print summary
    print("\n" + "="*60)
    print("CONTENT-BASED EVALUATION SUMMARY")
    print("="*60)
    
    holdout_results = results['evaluation_strategies']['holdout_split']
    print(f"\nHoldout Split Results:")
    print(f"  Test Users: {holdout_results['num_test_users']}")
    print(f"  Test Interactions: {holdout_results['num_test_interactions']}")
    
    print(f"\nRanking Metrics (Holdout Split):")
    for metric, scores in holdout_results['ranking_metrics'].items():
        print(f"  {metric.upper()}:")
        for k, score in scores.items():
            print(f"    @{k}: {score:.3f}")
    
    print(f"\nCoverage Metrics (Holdout Split):")
    for metric, score in holdout_results['coverage_metrics'].items():
        print(f"  {metric}: {score:.3f}")
    
    if 'content_analysis' in holdout_results:
        analysis = holdout_results['content_analysis']
        print(f"\nContent Analysis:")
        print(f"  Recommended Movies: {analysis['total_recommended_movies']}")
        print(f"  Recommendation Diversity: {analysis['recommendation_diversity']:.3f}")
        print(f"  Avg Recommendations per User: {analysis['avg_recommendations_per_user']:.1f}")
    
    print(f"\nResults saved to: data/eval/content_eval_results.json")
    print(f"Visualizations saved to: data/eval/")
    print("="*60)

if __name__ == "__main__":
    main()



