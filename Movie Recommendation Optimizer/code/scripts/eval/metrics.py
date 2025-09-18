"""
Movie Recommendation Optimizer - Metrics Framework
Step 4.1.1: Metric Framework Setup

This module provides comprehensive evaluation metrics for movie recommendation systems.
Supports ranking metrics, prediction metrics, and coverage metrics for content-based,
collaborative filtering, and hybrid recommendation systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import defaultdict
import logging
from dataclasses import dataclass
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetricConfig:
    """Configuration for evaluation metrics."""
    k_values: List[int] = None
    ranking_metrics: List[str] = None
    prediction_metrics: List[str] = None
    coverage_metrics: List[str] = None
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [5, 10, 20, 50]
        if self.ranking_metrics is None:
            self.ranking_metrics = ['recall', 'precision', 'map', 'ndcg']
        if self.prediction_metrics is None:
            self.prediction_metrics = ['rmse', 'mae']
        if self.coverage_metrics is None:
            self.coverage_metrics = ['user_coverage', 'item_coverage']

class MetricsFramework:
    """
    Comprehensive metrics framework for movie recommendation evaluation.
    
    Supports:
    - Ranking metrics: Recall@K, Precision@K, MAP@K, NDCG@K
    - Prediction metrics: RMSE, MAE
    - Coverage metrics: User coverage, Item coverage
    - Multiple evaluation strategies: Holdout split, User-sampled split
    """
    
    def __init__(self, config: Optional[MetricConfig] = None):
        self.config = config or MetricConfig()
        self.results = {}
        
    def _dcg(self, relevance_scores: List[float], k: int = None) -> float:
        """Calculate Discounted Cumulative Gain."""
        if k is None:
            k = len(relevance_scores)
        
        relevance_scores = np.array(relevance_scores[:k])
        if len(relevance_scores) == 0:
            return 0.0
            
        # DCG@k = sum(relevance_i / log2(i + 1)) for i in 1 to k
        positions = np.arange(1, len(relevance_scores) + 1)
        dcg = np.sum(relevance_scores / np.log2(positions + 1))
        return dcg
    
    def _idcg(self, relevance_scores: List[float], k: int = None) -> float:
        """Calculate Ideal Discounted Cumulative Gain."""
        if k is None:
            k = len(relevance_scores)
        
        # Sort in descending order for ideal ranking
        sorted_scores = sorted(relevance_scores, reverse=True)[:k]
        return self._dcg(sorted_scores, k)
    
    def recall_at_k(self, ground_truth: List[str], recommendations: List[str], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            ground_truth: List of relevant items (ground truth)
            recommendations: List of recommended items
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if len(ground_truth) == 0:
            return 0.0
            
        ground_truth_set = set(ground_truth)
        top_k_recommendations = recommendations[:k]
        
        relevant_recommended = len(ground_truth_set.intersection(set(top_k_recommendations)))
        recall = relevant_recommended / len(ground_truth_set)
        
        return recall
    
    def precision_at_k(self, ground_truth: List[str], recommendations: List[str], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            ground_truth: List of relevant items (ground truth)
            recommendations: List of recommended items
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
            
        ground_truth_set = set(ground_truth)
        top_k_recommendations = recommendations[:k]
        
        relevant_recommended = len(ground_truth_set.intersection(set(top_k_recommendations)))
        precision = relevant_recommended / k
        
        return precision
    
    def map_at_k(self, ground_truth: List[str], recommendations: List[str], k: int) -> float:
        """
        Calculate Mean Average Precision@K.
        
        Args:
            ground_truth: List of relevant items (ground truth)
            recommendations: List of recommended items
            k: Number of top recommendations to consider
            
        Returns:
            MAP@K score (0.0 to 1.0)
        """
        if len(ground_truth) == 0:
            return 0.0
            
        ground_truth_set = set(ground_truth)
        top_k_recommendations = recommendations[:k]
        
        if len(ground_truth_set) == 0:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recommendations):
            if item in ground_truth_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        map_score = precision_sum / len(ground_truth_set)
        return map_score
    
    def ndcg_at_k(self, ground_truth: List[str], recommendations: List[str], k: int, 
                  relevance_scores: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            ground_truth: List of relevant items (ground truth)
            recommendations: List of recommended items
            k: Number of top recommendations to consider
            relevance_scores: Optional dict mapping items to relevance scores
            
        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        if len(ground_truth) == 0:
            return 0.0
            
        ground_truth_set = set(ground_truth)
        top_k_recommendations = recommendations[:k]
        
        # Default relevance scores: 1.0 for relevant items, 0.0 for others
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in ground_truth}
        
        # Get relevance scores for recommended items
        rec_scores = [relevance_scores.get(item, 0.0) for item in top_k_recommendations]
        gt_scores = [relevance_scores.get(item, 0.0) for item in ground_truth]
        
        dcg = self._dcg(rec_scores, k)
        idcg = self._idcg(gt_scores, k)
        
        if idcg == 0:
            return 0.0
            
        ndcg = dcg / idcg
        return ndcg
    
    def rmse(self, true_ratings: List[float], predicted_ratings: List[float]) -> float:
        """
        Calculate Root Mean Square Error.
        
        Args:
            true_ratings: List of true rating values
            predicted_ratings: List of predicted rating values
            
        Returns:
            RMSE score (lower is better)
        """
        if len(true_ratings) != len(predicted_ratings):
            raise ValueError("True and predicted ratings must have the same length")
        
        if len(true_ratings) == 0:
            return 0.0
            
        true_ratings = np.array(true_ratings)
        predicted_ratings = np.array(predicted_ratings)
        
        mse = np.mean((true_ratings - predicted_ratings) ** 2)
        rmse = np.sqrt(mse)
        
        return rmse
    
    def mae(self, true_ratings: List[float], predicted_ratings: List[float]) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            true_ratings: List of true rating values
            predicted_ratings: List of predicted rating values
            
        Returns:
            MAE score (lower is better)
        """
        if len(true_ratings) != len(predicted_ratings):
            raise ValueError("True and predicted ratings must have the same length")
        
        if len(true_ratings) == 0:
            return 0.0
            
        true_ratings = np.array(true_ratings)
        predicted_ratings = np.array(predicted_ratings)
        
        mae = np.mean(np.abs(true_ratings - predicted_ratings))
        
        return mae
    
    def user_coverage(self, recommendations: Dict[str, List[str]], 
                     all_users: List[str]) -> float:
        """
        Calculate user coverage - percentage of users with at least one recommendation.
        
        Args:
            recommendations: Dict mapping user_id to list of recommended items
            all_users: List of all users in the system
            
        Returns:
            User coverage score (0.0 to 1.0)
        """
        if len(all_users) == 0:
            return 0.0
            
        users_with_recommendations = len([u for u in all_users if u in recommendations and len(recommendations[u]) > 0])
        coverage = users_with_recommendations / len(all_users)
        
        return coverage
    
    def item_coverage(self, recommendations: Dict[str, List[str]], 
                     all_items: List[str]) -> float:
        """
        Calculate item coverage - percentage of items that appear in at least one recommendation.
        
        Args:
            recommendations: Dict mapping user_id to list of recommended items
            all_items: List of all items in the system
            
        Returns:
            Item coverage score (0.0 to 1.0)
        """
        if len(all_items) == 0:
            return 0.0
            
        recommended_items = set()
        for user_recs in recommendations.values():
            recommended_items.update(user_recs)
        
        items_recommended = len(recommended_items.intersection(set(all_items)))
        coverage = items_recommended / len(all_items)
        
        return coverage
    
    def evaluate_ranking_metrics(self, ground_truth: Dict[str, List[str]], 
                                recommendations: Dict[str, List[str]]) -> Dict[str, Dict[int, float]]:
        """
        Evaluate all ranking metrics for multiple users.
        
        Args:
            ground_truth: Dict mapping user_id to list of relevant items
            recommendations: Dict mapping user_id to list of recommended items
            
        Returns:
            Dict with metric names as keys and Dict[k, average_score] as values
        """
        results = {metric: {k: [] for k in self.config.k_values} 
                  for metric in self.config.ranking_metrics}
        
        for user_id in ground_truth:
            if user_id not in recommendations:
                continue
                
            user_gt = ground_truth[user_id]
            user_recs = recommendations[user_id]
            
            for k in self.config.k_values:
                if 'recall' in self.config.ranking_metrics:
                    recall = self.recall_at_k(user_gt, user_recs, k)
                    results['recall'][k].append(recall)
                
                if 'precision' in self.config.ranking_metrics:
                    precision = self.precision_at_k(user_gt, user_recs, k)
                    results['precision'][k].append(precision)
                
                if 'map' in self.config.ranking_metrics:
                    map_score = self.map_at_k(user_gt, user_recs, k)
                    results['map'][k].append(map_score)
                
                if 'ndcg' in self.config.ranking_metrics:
                    ndcg = self.ndcg_at_k(user_gt, user_recs, k)
                    results['ndcg'][k].append(ndcg)
        
        # Calculate averages
        for metric in results:
            for k in results[metric]:
                if results[metric][k]:
                    results[metric][k] = np.mean(results[metric][k])
                else:
                    results[metric][k] = 0.0
        
        return results
    
    def evaluate_prediction_metrics(self, true_ratings: List[float], 
                                   predicted_ratings: List[float]) -> Dict[str, float]:
        """
        Evaluate prediction metrics for rating predictions.
        
        Args:
            true_ratings: List of true rating values
            predicted_ratings: List of predicted rating values
            
        Returns:
            Dict with metric names as keys and scores as values
        """
        results = {}
        
        if 'rmse' in self.config.prediction_metrics:
            results['rmse'] = self.rmse(true_ratings, predicted_ratings)
        
        if 'mae' in self.config.prediction_metrics:
            results['mae'] = self.mae(true_ratings, predicted_ratings)
        
        return results
    
    def evaluate_coverage_metrics(self, recommendations: Dict[str, List[str]], 
                                 all_users: List[str], all_items: List[str]) -> Dict[str, float]:
        """
        Evaluate coverage metrics.
        
        Args:
            recommendations: Dict mapping user_id to list of recommended items
            all_users: List of all users in the system
            all_items: List of all items in the system
            
        Returns:
            Dict with metric names as keys and scores as values
        """
        results = {}
        
        if 'user_coverage' in self.config.coverage_metrics:
            results['user_coverage'] = self.user_coverage(recommendations, all_users)
        
        if 'item_coverage' in self.config.coverage_metrics:
            results['item_coverage'] = self.item_coverage(recommendations, all_items)
        
        return results
    
    def evaluate_holdout_split(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                              recommendations: Dict[str, List[str]], 
                              predictions: Optional[Dict[Tuple[str, str], float]] = None) -> Dict[str, Any]:
        """
        Evaluate using holdout split strategy.
        
        Args:
            train_data: Training data with columns ['user_index', 'canonical_id', 'rating']
            test_data: Test data with columns ['user_index', 'canonical_id', 'rating']
            recommendations: Dict mapping user_id to list of recommended items
            predictions: Optional dict mapping (user_id, item_id) to predicted rating
            
        Returns:
            Dict containing all evaluation results
        """
        logger.info("Evaluating with holdout split strategy")
        
        # Prepare ground truth for ranking metrics
        ground_truth = {}
        for _, row in test_data.iterrows():
            user_id = str(row['user_index'])
            item_id = str(row['canonical_id'])
            if user_id not in ground_truth:
                ground_truth[user_id] = []
            ground_truth[user_id].append(item_id)
        
        # Evaluate ranking metrics
        ranking_results = self.evaluate_ranking_metrics(ground_truth, recommendations)
        
        # Evaluate prediction metrics if predictions provided
        prediction_results = {}
        if predictions:
            true_ratings = []
            pred_ratings = []
            for _, row in test_data.iterrows():
                user_id = str(row['user_index'])
                item_id = str(row['canonical_id'])
                rating = row['rating']
                pred_key = (user_id, item_id)
                if pred_key in predictions:
                    true_ratings.append(rating)
                    pred_ratings.append(predictions[pred_key])
            
            if true_ratings:
                prediction_results = self.evaluate_prediction_metrics(true_ratings, pred_ratings)
        
        # Evaluate coverage metrics
        all_users = list(set(train_data['user_index'].astype(str).tolist() + 
                           test_data['user_index'].astype(str).tolist()))
        all_items = list(set(train_data['canonical_id'].astype(str).tolist() + 
                           test_data['canonical_id'].astype(str).tolist()))
        
        coverage_results = self.evaluate_coverage_metrics(recommendations, all_users, all_items)
        
        results = {
            'strategy': 'holdout_split',
            'ranking_metrics': ranking_results,
            'prediction_metrics': prediction_results,
            'coverage_metrics': coverage_results,
            'num_test_users': len(ground_truth),
            'num_test_interactions': len(test_data)
        }
        
        return results
    
    def evaluate_user_sampled_split(self, user_data: Dict[str, List[Tuple[str, float]]],
                                   recommendations: Dict[str, List[str]], 
                                   sample_ratio: float = 0.2) -> Dict[str, Any]:
        """
        Evaluate using user-sampled split strategy.
        
        Args:
            user_data: Dict mapping user_id to list of (item_id, rating) tuples
            recommendations: Dict mapping user_id to list of recommended items
            sample_ratio: Fraction of user's history to hold out for testing
            
        Returns:
            Dict containing all evaluation results
        """
        logger.info(f"Evaluating with user-sampled split strategy (sample_ratio={sample_ratio})")
        
        ground_truth = {}
        all_users = list(user_data.keys())
        all_items = set()
        
        for user_id, interactions in user_data.items():
            if len(interactions) < 2:  # Need at least 2 interactions for train/test split
                continue
                
            # Randomly sample items for test set
            np.random.seed(42)  # For reproducibility
            n_test = max(1, int(len(interactions) * sample_ratio))
            test_indices = np.random.choice(len(interactions), n_test, replace=False)
            
            test_items = [interactions[i][0] for i in test_indices]
            ground_truth[user_id] = test_items
            all_items.update([item[0] for item in interactions])
        
        # Evaluate ranking metrics
        ranking_results = self.evaluate_ranking_metrics(ground_truth, recommendations)
        
        # Evaluate coverage metrics
        coverage_results = self.evaluate_coverage_metrics(recommendations, all_users, list(all_items))
        
        results = {
            'strategy': 'user_sampled_split',
            'ranking_metrics': ranking_results,
            'coverage_metrics': coverage_results,
            'num_test_users': len(ground_truth),
            'sample_ratio': sample_ratio
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")

def create_metrics_config(output_path: str = "data/eval/metrics_config.json"):
    """Create metrics configuration manifest."""
    config = {
        "framework_version": "1.0.0",
        "step": "4.1.1",
        "description": "Metrics framework for movie recommendation evaluation",
        "k_values": [5, 10, 20, 50],
        "ranking_metrics": {
            "recall": {
                "description": "Recall@K - Fraction of relevant items that are recommended",
                "formula": "|relevant ∩ recommended| / |relevant|",
                "range": "[0, 1]",
                "higher_is_better": True
            },
            "precision": {
                "description": "Precision@K - Fraction of recommended items that are relevant",
                "formula": "|relevant ∩ recommended| / K",
                "range": "[0, 1]",
                "higher_is_better": True
            },
            "map": {
                "description": "Mean Average Precision@K - Average precision across all relevant items",
                "formula": "Σ(precision@i × rel_i) / |relevant|",
                "range": "[0, 1]",
                "higher_is_better": True
            },
            "ndcg": {
                "description": "Normalized Discounted Cumulative Gain@K - Ranking quality with position discount",
                "formula": "DCG@K / IDCG@K",
                "range": "[0, 1]",
                "higher_is_better": True
            }
        },
        "prediction_metrics": {
            "rmse": {
                "description": "Root Mean Square Error - Square root of mean squared differences",
                "formula": "√(Σ(rating_true - rating_pred)² / n)",
                "range": "[0, ∞)",
                "higher_is_better": False
            },
            "mae": {
                "description": "Mean Absolute Error - Mean of absolute differences",
                "formula": "Σ|rating_true - rating_pred| / n",
                "range": "[0, ∞)",
                "higher_is_better": False
            }
        },
        "coverage_metrics": {
            "user_coverage": {
                "description": "Percentage of users with at least one recommendation",
                "formula": "|users_with_recommendations| / |all_users|",
                "range": "[0, 1]",
                "higher_is_better": True
            },
            "item_coverage": {
                "description": "Percentage of items that appear in at least one recommendation",
                "formula": "|recommended_items| / |all_items|",
                "range": "[0, 1]",
                "higher_is_better": True
            }
        },
        "evaluation_strategies": {
            "holdout_split": {
                "description": "Split ratings into train/test sets globally",
                "use_case": "General model evaluation"
            },
            "user_sampled_split": {
                "description": "For each user, hold out a fraction of their history for testing",
                "use_case": "Personalized recommendation evaluation"
            }
        },
        "implementation_paths": {
            "metrics_module": "scripts/eval/metrics.py",
            "test_module": "scripts/eval/test_metrics.py",
            "documentation": "docs/step4_metrics_framework.md"
        },
        "created_at": "2024-01-XX",
        "last_updated": "2024-01-XX"
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Metrics configuration saved to {output_path}")
    return config

if __name__ == "__main__":
    # Example usage and testing
    logger.info("Initializing metrics framework...")
    
    # Create configuration
    config = create_metrics_config()
    
    # Initialize framework
    framework = MetricsFramework()
    
    logger.info("Metrics framework setup complete!")
    logger.info(f"Supported K values: {framework.config.k_values}")
    logger.info(f"Ranking metrics: {framework.config.ranking_metrics}")
    logger.info(f"Prediction metrics: {framework.config.prediction_metrics}")
    logger.info(f"Coverage metrics: {framework.config.coverage_metrics}")



