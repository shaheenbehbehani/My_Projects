"""
Unit tests for the metrics framework.
Tests all metrics with synthetic examples where results can be verified manually.
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List
import sys
import os

# Add the project root to the path to import metrics
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from scripts.eval.metrics import MetricsFramework, MetricConfig

class TestMetricsFramework(unittest.TestCase):
    """Test cases for the metrics framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MetricConfig(k_values=[3, 5])
        self.framework = MetricsFramework(self.config)
        
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        # Test case 1: Perfect recall
        ground_truth = ['A', 'B', 'C']
        recommendations = ['A', 'B', 'C', 'D', 'E']
        recall = self.framework.recall_at_k(ground_truth, recommendations, k=3)
        self.assertEqual(recall, 1.0)
        
        # Test case 2: Partial recall
        ground_truth = ['A', 'B', 'C']
        recommendations = ['A', 'D', 'B', 'E', 'F']
        recall = self.framework.recall_at_k(ground_truth, recommendations, k=3)
        self.assertEqual(recall, 2/3)
        
        # Test case 3: No relevant items
        ground_truth = ['A', 'B', 'C']
        recommendations = ['D', 'E', 'F']
        recall = self.framework.recall_at_k(ground_truth, recommendations, k=3)
        self.assertEqual(recall, 0.0)
        
        # Test case 4: Empty ground truth
        ground_truth = []
        recommendations = ['A', 'B', 'C']
        recall = self.framework.recall_at_k(ground_truth, recommendations, k=3)
        self.assertEqual(recall, 0.0)
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        # Test case 1: Perfect precision
        ground_truth = ['A', 'B', 'C']
        recommendations = ['A', 'B', 'C']
        precision = self.framework.precision_at_k(ground_truth, recommendations, k=3)
        self.assertEqual(precision, 1.0)
        
        # Test case 2: Partial precision
        ground_truth = ['A', 'B', 'C']
        recommendations = ['A', 'D', 'B', 'E', 'F']
        precision = self.framework.precision_at_k(ground_truth, recommendations, k=3)
        self.assertEqual(precision, 2/3)
        
        # Test case 3: No relevant items
        ground_truth = ['A', 'B', 'C']
        recommendations = ['D', 'E', 'F']
        precision = self.framework.precision_at_k(ground_truth, recommendations, k=3)
        self.assertEqual(precision, 0.0)
        
        # Test case 4: k=0
        ground_truth = ['A', 'B', 'C']
        recommendations = ['A', 'B', 'C']
        precision = self.framework.precision_at_k(ground_truth, recommendations, k=0)
        self.assertEqual(precision, 0.0)
    
    def test_map_at_k(self):
        """Test MAP@K calculation."""
        # Test case 1: Perfect MAP
        ground_truth = ['A', 'B', 'C']
        recommendations = ['A', 'B', 'C', 'D', 'E']
        map_score = self.framework.map_at_k(ground_truth, recommendations, k=5)
        self.assertEqual(map_score, 1.0)
        
        # Test case 2: Partial MAP
        # Ground truth: A, B, C
        # Recommendations: A, D, B, E, C
        # At position 1: A is relevant, precision = 1/1 = 1.0
        # At position 3: B is relevant, precision = 2/3 = 0.67
        # At position 5: C is relevant, precision = 3/5 = 0.6
        # MAP = (1.0 + 0.67 + 0.6) / 3 ≈ 0.7556
        ground_truth = ['A', 'B', 'C']
        recommendations = ['A', 'D', 'B', 'E', 'C']
        map_score = self.framework.map_at_k(ground_truth, recommendations, k=5)
        self.assertAlmostEqual(map_score, 0.7556, places=3)
        
        # Test case 3: No relevant items
        ground_truth = ['A', 'B', 'C']
        recommendations = ['D', 'E', 'F']
        map_score = self.framework.map_at_k(ground_truth, recommendations, k=3)
        self.assertEqual(map_score, 0.0)
    
    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        # Test case 1: Perfect NDCG
        ground_truth = ['A', 'B', 'C']
        recommendations = ['A', 'B', 'C', 'D', 'E']
        ndcg = self.framework.ndcg_at_k(ground_truth, recommendations, k=3)
        self.assertEqual(ndcg, 1.0)
        
        # Test case 2: Partial NDCG
        ground_truth = ['A', 'B', 'C']
        recommendations = ['A', 'D', 'B', 'E', 'F']
        ndcg = self.framework.ndcg_at_k(ground_truth, recommendations, k=3)
        # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1 + 0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.63 + 0.5 = 2.13
        # NDCG = 1.5 / 2.13 ≈ 0.704
        self.assertAlmostEqual(ndcg, 0.704, places=2)
        
        # Test case 3: With relevance scores
        ground_truth = ['A', 'B', 'C']
        recommendations = ['A', 'D', 'B', 'E', 'F']
        relevance_scores = {'A': 3.0, 'B': 2.0, 'C': 1.0}
        ndcg = self.framework.ndcg_at_k(ground_truth, recommendations, k=3, 
                                       relevance_scores=relevance_scores)
        self.assertGreater(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)
    
    def test_rmse(self):
        """Test RMSE calculation."""
        # Test case 1: Perfect predictions
        true_ratings = [3.0, 4.0, 5.0]
        predicted_ratings = [3.0, 4.0, 5.0]
        rmse = self.framework.rmse(true_ratings, predicted_ratings)
        self.assertEqual(rmse, 0.0)
        
        # Test case 2: Some errors
        true_ratings = [3.0, 4.0, 5.0]
        predicted_ratings = [2.0, 5.0, 4.0]
        rmse = self.framework.rmse(true_ratings, predicted_ratings)
        # MSE = ((3-2)² + (4-5)² + (5-4)²) / 3 = (1 + 1 + 1) / 3 = 1
        # RMSE = √1 = 1
        self.assertEqual(rmse, 1.0)
        
        # Test case 3: Empty lists
        true_ratings = []
        predicted_ratings = []
        rmse = self.framework.rmse(true_ratings, predicted_ratings)
        self.assertEqual(rmse, 0.0)
        
        # Test case 4: Mismatched lengths
        true_ratings = [3.0, 4.0]
        predicted_ratings = [3.0, 4.0, 5.0]
        with self.assertRaises(ValueError):
            self.framework.rmse(true_ratings, predicted_ratings)
    
    def test_mae(self):
        """Test MAE calculation."""
        # Test case 1: Perfect predictions
        true_ratings = [3.0, 4.0, 5.0]
        predicted_ratings = [3.0, 4.0, 5.0]
        mae = self.framework.mae(true_ratings, predicted_ratings)
        self.assertEqual(mae, 0.0)
        
        # Test case 2: Some errors
        true_ratings = [3.0, 4.0, 5.0]
        predicted_ratings = [2.0, 5.0, 4.0]
        mae = self.framework.mae(true_ratings, predicted_ratings)
        # MAE = (|3-2| + |4-5| + |5-4|) / 3 = (1 + 1 + 1) / 3 = 1
        self.assertEqual(mae, 1.0)
        
        # Test case 3: Empty lists
        true_ratings = []
        predicted_ratings = []
        mae = self.framework.mae(true_ratings, predicted_ratings)
        self.assertEqual(mae, 0.0)
    
    def test_user_coverage(self):
        """Test user coverage calculation."""
        # Test case 1: Perfect coverage
        recommendations = {'user1': ['A', 'B'], 'user2': ['C', 'D'], 'user3': ['E', 'F']}
        all_users = ['user1', 'user2', 'user3']
        coverage = self.framework.user_coverage(recommendations, all_users)
        self.assertEqual(coverage, 1.0)
        
        # Test case 2: Partial coverage
        recommendations = {'user1': ['A', 'B'], 'user3': ['E', 'F']}
        all_users = ['user1', 'user2', 'user3']
        coverage = self.framework.user_coverage(recommendations, all_users)
        self.assertEqual(coverage, 2/3)
        
        # Test case 3: No coverage
        recommendations = {}
        all_users = ['user1', 'user2', 'user3']
        coverage = self.framework.user_coverage(recommendations, all_users)
        self.assertEqual(coverage, 0.0)
        
        # Test case 4: Empty user list
        recommendations = {'user1': ['A', 'B']}
        all_users = []
        coverage = self.framework.user_coverage(recommendations, all_users)
        self.assertEqual(coverage, 0.0)
    
    def test_item_coverage(self):
        """Test item coverage calculation."""
        # Test case 1: Perfect coverage
        recommendations = {'user1': ['A', 'B'], 'user2': ['C', 'D']}
        all_items = ['A', 'B', 'C', 'D']
        coverage = self.framework.item_coverage(recommendations, all_items)
        self.assertEqual(coverage, 1.0)
        
        # Test case 2: Partial coverage
        recommendations = {'user1': ['A', 'B'], 'user2': ['C', 'E']}
        all_items = ['A', 'B', 'C', 'D']
        coverage = self.framework.item_coverage(recommendations, all_items)
        self.assertEqual(coverage, 3/4)  # A, B, C are covered, D is not
        
        # Test case 3: No coverage
        recommendations = {'user1': ['E', 'F'], 'user2': ['G', 'H']}
        all_items = ['A', 'B', 'C', 'D']
        coverage = self.framework.item_coverage(recommendations, all_items)
        self.assertEqual(coverage, 0.0)
    
    def test_evaluate_ranking_metrics(self):
        """Test evaluation of ranking metrics for multiple users."""
        ground_truth = {
            'user1': ['A', 'B', 'C'],
            'user2': ['D', 'E'],
            'user3': ['F', 'G', 'H', 'I']
        }
        recommendations = {
            'user1': ['A', 'D', 'B', 'E', 'F'],
            'user2': ['D', 'G', 'E', 'H', 'I'],
            'user3': ['F', 'J', 'G', 'K', 'H']
        }
        
        results = self.framework.evaluate_ranking_metrics(ground_truth, recommendations)
        
        # Check that all metrics are present
        self.assertIn('recall', results)
        self.assertIn('precision', results)
        self.assertIn('map', results)
        self.assertIn('ndcg', results)
        
        # Check that all k values are present
        for metric in results:
            for k in self.config.k_values:
                self.assertIn(k, results[metric])
                self.assertGreaterEqual(results[metric][k], 0.0)
                self.assertLessEqual(results[metric][k], 1.0)
    
    def test_evaluate_prediction_metrics(self):
        """Test evaluation of prediction metrics."""
        true_ratings = [3.0, 4.0, 5.0, 2.0, 4.5]
        predicted_ratings = [2.5, 4.2, 4.8, 2.1, 4.0]
        
        results = self.framework.evaluate_prediction_metrics(true_ratings, predicted_ratings)
        
        self.assertIn('rmse', results)
        self.assertIn('mae', results)
        self.assertGreaterEqual(results['rmse'], 0.0)
        self.assertGreaterEqual(results['mae'], 0.0)
    
    def test_evaluate_coverage_metrics(self):
        """Test evaluation of coverage metrics."""
        recommendations = {
            'user1': ['A', 'B'],
            'user2': ['C', 'D'],
            'user3': ['A', 'E']
        }
        all_users = ['user1', 'user2', 'user3', 'user4']
        all_items = ['A', 'B', 'C', 'D', 'E', 'F']
        
        results = self.framework.evaluate_coverage_metrics(recommendations, all_users, all_items)
        
        self.assertIn('user_coverage', results)
        self.assertIn('item_coverage', results)
        self.assertGreaterEqual(results['user_coverage'], 0.0)
        self.assertLessEqual(results['user_coverage'], 1.0)
        self.assertGreaterEqual(results['item_coverage'], 0.0)
        self.assertLessEqual(results['item_coverage'], 1.0)
    
    def test_evaluate_holdout_split(self):
        """Test holdout split evaluation."""
        # Create sample data
        train_data = pd.DataFrame({
            'user_index': [1, 1, 2, 2, 3, 3],
            'canonical_id': ['A', 'B', 'C', 'D', 'E', 'F'],
            'rating': [4.0, 5.0, 3.0, 4.0, 5.0, 2.0]
        })
        
        test_data = pd.DataFrame({
            'user_index': [1, 2, 3],
            'canonical_id': ['C', 'A', 'B'],
            'rating': [3.0, 4.0, 5.0]
        })
        
        recommendations = {
            '1': ['A', 'B', 'C', 'D'],
            '2': ['C', 'A', 'E', 'F'],
            '3': ['B', 'A', 'C', 'D']
        }
        
        results = self.framework.evaluate_holdout_split(train_data, test_data, recommendations)
        
        self.assertEqual(results['strategy'], 'holdout_split')
        self.assertIn('ranking_metrics', results)
        self.assertIn('coverage_metrics', results)
        self.assertEqual(results['num_test_users'], 3)
        self.assertEqual(results['num_test_interactions'], 3)
    
    def test_evaluate_user_sampled_split(self):
        """Test user-sampled split evaluation."""
        user_data = {
            'user1': [('A', 4.0), ('B', 5.0), ('C', 3.0), ('D', 4.0)],
            'user2': [('E', 3.0), ('F', 4.0), ('G', 5.0)],
            'user3': [('H', 2.0), ('I', 4.0)]
        }
        
        recommendations = {
            'user1': ['A', 'B', 'C', 'D'],
            'user2': ['E', 'F', 'G', 'H'],
            'user3': ['H', 'I', 'J', 'K']
        }
        
        results = self.framework.evaluate_user_sampled_split(user_data, recommendations, sample_ratio=0.5)
        
        self.assertEqual(results['strategy'], 'user_sampled_split')
        self.assertIn('ranking_metrics', results)
        self.assertIn('coverage_metrics', results)
        self.assertEqual(results['sample_ratio'], 0.5)
        self.assertGreater(results['num_test_users'], 0)

def run_synthetic_examples():
    """Run synthetic examples to demonstrate metrics."""
    print("Running synthetic examples for metrics verification...")
    
    framework = MetricsFramework()
    
    # Example 1: Perfect recommendations
    print("\n=== Example 1: Perfect Recommendations ===")
    ground_truth = ['movie_A', 'movie_B', 'movie_C']
    recommendations = ['movie_A', 'movie_B', 'movie_C', 'movie_D', 'movie_E']
    
    for k in [3, 5]:
        recall = framework.recall_at_k(ground_truth, recommendations, k)
        precision = framework.precision_at_k(ground_truth, recommendations, k)
        map_score = framework.map_at_k(ground_truth, recommendations, k)
        ndcg = framework.ndcg_at_k(ground_truth, recommendations, k)
        
        print(f"K={k}: Recall={recall:.3f}, Precision={precision:.3f}, MAP={map_score:.3f}, NDCG={ndcg:.3f}")
    
    # Example 2: Partial recommendations
    print("\n=== Example 2: Partial Recommendations ===")
    ground_truth = ['movie_A', 'movie_B', 'movie_C']
    recommendations = ['movie_A', 'movie_X', 'movie_B', 'movie_Y', 'movie_Z']
    
    for k in [3, 5]:
        recall = framework.recall_at_k(ground_truth, recommendations, k)
        precision = framework.precision_at_k(ground_truth, recommendations, k)
        map_score = framework.map_at_k(ground_truth, recommendations, k)
        ndcg = framework.ndcg_at_k(ground_truth, recommendations, k)
        
        print(f"K={k}: Recall={recall:.3f}, Precision={precision:.3f}, MAP={map_score:.3f}, NDCG={ndcg:.3f}")
    
    # Example 3: Rating predictions
    print("\n=== Example 3: Rating Predictions ===")
    true_ratings = [4.0, 3.0, 5.0, 2.0, 4.5]
    predicted_ratings = [3.8, 3.2, 4.9, 2.1, 4.2]
    
    rmse = framework.rmse(true_ratings, predicted_ratings)
    mae = framework.mae(true_ratings, predicted_ratings)
    
    print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    
    # Example 4: Coverage
    print("\n=== Example 4: Coverage Metrics ===")
    recommendations = {
        'user1': ['movie_A', 'movie_B'],
        'user2': ['movie_C', 'movie_D'],
        'user3': ['movie_A', 'movie_E']
    }
    all_users = ['user1', 'user2', 'user3', 'user4']
    all_items = ['movie_A', 'movie_B', 'movie_C', 'movie_D', 'movie_E', 'movie_F']
    
    user_coverage = framework.user_coverage(recommendations, all_users)
    item_coverage = framework.item_coverage(recommendations, all_items)
    
    print(f"User Coverage: {user_coverage:.3f}, Item Coverage: {item_coverage:.3f}")

if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run synthetic examples
    run_synthetic_examples()
