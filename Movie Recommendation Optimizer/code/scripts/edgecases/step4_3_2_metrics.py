#!/usr/bin/env python3
"""
Step 4.3.2 - Edge Case Testing Metrics
Movie Recommendation Optimizer - Calculate Metrics for Edge Case Testing

This module provides metrics calculation functions for edge case testing,
including Recall@K, MAP@K, coverage, and other evaluation metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

class EdgeCaseMetrics:
    """
    Calculate metrics for edge case testing results.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.k_values = [5, 10, 20, 50]
    
    def calculate_recall_at_k(self, recommendations: List[str], ground_truth: List[str], k: int) -> float:
        """Calculate Recall@K for recommendations."""
        if not ground_truth or k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_items = set(ground_truth)
        recommended_items = set(top_k_recs)
        
        intersection = relevant_items.intersection(recommended_items)
        return len(intersection) / len(relevant_items)
    
    def calculate_precision_at_k(self, recommendations: List[str], ground_truth: List[str], k: int) -> float:
        """Calculate Precision@K for recommendations."""
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_items = set(ground_truth)
        recommended_items = set(top_k_recs)
        
        intersection = relevant_items.intersection(recommended_items)
        return len(intersection) / k
    
    def calculate_map_at_k(self, recommendations: List[str], ground_truth: List[str], k: int) -> float:
        """Calculate MAP@K for recommendations."""
        if not ground_truth or k == 0:
            return 0.0
        
        relevant_items = set(ground_truth)
        top_k_recs = recommendations[:k]
        
        if not relevant_items.intersection(set(top_k_recs)):
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_items)
    
    def calculate_ndcg_at_k(self, recommendations: List[str], ground_truth: List[str], k: int) -> float:
        """Calculate NDCG@K for recommendations."""
        if not ground_truth or k == 0:
            return 0.0
        
        # For simplicity, assume binary relevance
        relevance_scores = [1 if item in ground_truth else 0 for item in recommendations[:k]]
        
        # Calculate DCG
        dcg = 0.0
        for i, score in enumerate(relevance_scores):
            dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = [1] * min(len(ground_truth), k)
        idcg = 0.0
        for i, score in enumerate(ideal_relevance):
            idcg += score / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_coverage(self, all_recommendations: List[List[str]], catalog: List[str]) -> float:
        """Calculate item coverage for recommendations."""
        if not catalog:
            return 0.0
        
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        
        return len(recommended_items) / len(catalog)
    
    def calculate_diversity(self, recommendations: List[str], item_features: Dict[str, Dict[str, Any]]) -> float:
        """Calculate diversity of recommendations based on item features."""
        if len(recommendations) < 2:
            return 0.0
        
        # Calculate pairwise diversity based on genres
        diversity_scores = []
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                item1 = recommendations[i]
                item2 = recommendations[j]
                
                if item1 in item_features and item2 in item_features:
                    genres1 = set(item_features[item1].get('genres', []))
                    genres2 = set(item_features[item2].get('genres', []))
                    
                    # Jaccard diversity
                    intersection = len(genres1.intersection(genres2))
                    union = len(genres1.union(genres2))
                    diversity = 1 - (intersection / union) if union > 0 else 0
                    diversity_scores.append(diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def calculate_novelty(self, recommendations: List[str], popularity_scores: Dict[str, float]) -> float:
        """Calculate novelty of recommendations based on popularity."""
        if not recommendations or not popularity_scores:
            return 0.0
        
        novelty_scores = []
        for item in recommendations:
            if item in popularity_scores:
                # Novelty is inverse of popularity (higher novelty = less popular)
                novelty = 1 - popularity_scores[item]
                novelty_scores.append(novelty)
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    def calculate_serendipity(self, recommendations: List[str], user_history: List[str], 
                            item_similarity: Dict[Tuple[str, str], float]) -> float:
        """Calculate serendipity of recommendations."""
        if not recommendations or not user_history:
            return 0.0
        
        serendipity_scores = []
        for item in recommendations:
            if item not in user_history:
                # Calculate average similarity to user history
                similarities = []
                for hist_item in user_history:
                    key = tuple(sorted([item, hist_item]))
                    if key in item_similarity:
                        similarities.append(item_similarity[key])
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    # Serendipity is inverse of similarity (higher serendipity = less similar)
                    serendipity = 1 - avg_similarity
                    serendipity_scores.append(serendipity)
        
        return np.mean(serendipity_scores) if serendipity_scores else 0.0
    
    def calculate_system_metrics(self, test_results: List[Dict[str, Any]], 
                               ground_truth: Optional[Dict[str, List[str]]] = None,
                               item_features: Optional[Dict[str, Dict[str, Any]]] = None,
                               popularity_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a system."""
        metrics = {}
        
        # Collect all recommendations by system
        system_recommendations = defaultdict(list)
        all_recommendations = []
        
        for test_result in test_results:
            if test_result.get("status") == "success":
                for system, system_result in test_result.get("systems", {}).items():
                    if system_result.get("status") == "success":
                        recs = [r.get("movie_id", "") for r in system_result.get("recommendations", [])]
                        system_recommendations[system].extend(recs)
                        all_recommendations.extend(recs)
        
        # Calculate metrics for each K value
        for k in self.k_values:
            k_metrics = {}
            
            for system, recommendations in system_recommendations.items():
                # Basic metrics
                k_metrics[f"{system}_num_recommendations"] = len(recommendations)
                k_metrics[f"{system}_avg_score"] = 0.0  # Would need actual scores
                
                # Ranking metrics (if ground truth available)
                if ground_truth:
                    for user_id, user_ground_truth in ground_truth.items():
                        recall = self.calculate_recall_at_k(recommendations, user_ground_truth, k)
                        precision = self.calculate_precision_at_k(recommendations, user_ground_truth, k)
                        map_score = self.calculate_map_at_k(recommendations, user_ground_truth, k)
                        ndcg = self.calculate_ndcg_at_k(recommendations, user_ground_truth, k)
                        
                        k_metrics[f"{system}_recall_at_{k}"] = recall
                        k_metrics[f"{system}_precision_at_{k}"] = precision
                        k_metrics[f"{system}_map_at_{k}"] = map_score
                        k_metrics[f"{system}_ndcg_at_{k}"] = ndcg
            
            # Coverage metrics
            if item_features:
                catalog = list(item_features.keys())
                k_metrics[f"coverage_at_{k}"] = self.calculate_coverage([all_recommendations], catalog)
            
            # Diversity metrics
            if item_features:
                k_metrics[f"diversity_at_{k}"] = self.calculate_diversity(all_recommendations, item_features)
            
            # Novelty metrics
            if popularity_scores:
                k_metrics[f"novelty_at_{k}"] = self.calculate_novelty(all_recommendations, popularity_scores)
            
            metrics[f"k_{k}"] = k_metrics
        
        return metrics
    
    def calculate_scenario_metrics(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for an entire scenario."""
        metrics = {}
        
        # Aggregate metrics across all K values
        for k_key, k_results in scenario_results.get("test_results", {}).items():
            if k_key.startswith("k_"):
                k_value = int(k_key.split("_")[1])
                
                # Calculate success rate
                total_tests = len(k_results)
                successful_tests = sum(1 for r in k_results if r.get("status") == "success")
                success_rate = successful_tests / total_tests if total_tests > 0 else 0
                
                metrics[f"k_{k_value}_total_tests"] = total_tests
                metrics[f"k_{k_value}_successful_tests"] = successful_tests
                metrics[f"k_{k_value}_success_rate"] = success_rate
                
                # Calculate system-specific metrics
                system_metrics = defaultdict(list)
                for test_result in k_results:
                    if test_result.get("status") == "success":
                        for system, system_result in test_result.get("systems", {}).items():
                            if system_result.get("status") == "success":
                                num_recs = len(system_result.get("recommendations", []))
                                system_metrics[f"{system}_num_recommendations"].append(num_recs)
                                
                                # Calculate average score
                                scores = [r.get("score", 0) for r in system_result.get("recommendations", [])]
                                if scores:
                                    system_metrics[f"{system}_avg_score"].append(np.mean(scores))
                
                # Aggregate system metrics
                for metric_name, values in system_metrics.items():
                    if values:
                        metrics[f"k_{k_value}_{metric_name}_mean"] = np.mean(values)
                        metrics[f"k_{k_value}_{metric_name}_std"] = np.std(values)
        
        return metrics
    
    def generate_metrics_summary(self, all_scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary metrics across all scenarios."""
        summary = {
            "total_scenarios": len(all_scenario_results),
            "scenario_metrics": {},
            "overall_metrics": {}
        }
        
        # Calculate metrics for each scenario
        for scenario_id, scenario_result in all_scenario_results.items():
            if scenario_result.get("status") == "completed":
                scenario_metrics = self.calculate_scenario_metrics(scenario_result)
                summary["scenario_metrics"][scenario_id] = scenario_metrics
        
        # Calculate overall metrics
        all_success_rates = []
        for scenario_metrics in summary["scenario_metrics"].values():
            for metric_name, value in scenario_metrics.items():
                if metric_name.endswith("_success_rate"):
                    all_success_rates.append(value)
        
        if all_success_rates:
            summary["overall_metrics"]["avg_success_rate"] = np.mean(all_success_rates)
            summary["overall_metrics"]["min_success_rate"] = np.min(all_success_rates)
            summary["overall_metrics"]["max_success_rate"] = np.max(all_success_rates)
        
        return summary


def main():
    """CLI entrypoint for metrics calculation."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Edge Case Testing Metrics Calculator')
    parser.add_argument('--results', required=True, help='Path to results JSON file')
    parser.add_argument('--output', help='Output file for metrics (JSON format)')
    
    args = parser.parse_args()
    
    try:
        # Load results
        with open(args.results, 'r') as f:
            results = json.load(f)
        
        # Calculate metrics
        metrics_calc = EdgeCaseMetrics()
        summary = metrics_calc.generate_metrics_summary(results.get("scenario_results", {}))
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Metrics saved to {args.output}")
        else:
            print(json.dumps(summary, indent=2))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

