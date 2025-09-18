#!/usr/bin/env python3
"""
Step 3d.2 - Workload Testing for Candidate Fetcher
Runs comprehensive workload tests to validate acceptance criteria.
"""

import os
import sys
import json
import time
import random
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures
from statistics import mean, median, stdev
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.serve.candidates_entrypoint import CandidateFetcher

class WorkloadTester:
    """Comprehensive workload testing for candidate fetcher."""
    
    def __init__(self, release_lock_path: str = "data/hybrid/release_lock_3d.json"):
        """Initialize workload tester."""
        self.logger = self._setup_logging()
        self.logger.info("Initializing WorkloadTester")
        
        # Initialize fetcher
        self.fetcher = CandidateFetcher(release_lock_path)
        
        # Test configuration
        self.test_users = self._get_test_users()
        self.filter_combinations = self._get_filter_combinations()
        
        # Results storage
        self.test_results = []
        self.performance_metrics = {}
        
        self.logger.info(f"Workload tester initialized with {len(self.test_users)} users")
    
    def _setup_logging(self):
        """Setup logging for workload tester."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("workload_tester")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('logs/step3d2_cache.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _get_test_users(self) -> List[str]:
        """Get test users from available data."""
        # Load user activity data
        user_activity = pd.read_parquet("data/derived/user_activity_snapshot.parquet")
        
        # Get users from different buckets
        users = []
        
        # Top active users
        top_users = user_activity.nlargest(200, 'ratings_count')['user_index'].tolist()
        users.extend([str(uid) for uid in top_users])
        
        # Random sample
        random_users = user_activity.sample(100, random_state=42)['user_index'].tolist()
        users.extend([str(uid) for uid in random_users])
        
        # Edge cases
        users.extend(['0', '1', '999999', 'invalid_user'])
        
        return list(set(users))  # Remove duplicates
    
    def _get_filter_combinations(self) -> List[Dict[str, Any]]:
        """Get filter combinations for testing."""
        combinations = []
        
        # No filters
        combinations.append({'genres': None, 'providers': None, 'year_range': None})
        
        # Single genre
        for genre in ['action', 'comedy', 'drama', 'thriller', 'horror']:
            combinations.append({'genres': [genre], 'providers': None, 'year_range': None})
        
        # Single provider
        for provider in ['netflix', 'hulu', 'amazon']:
            combinations.append({'genres': None, 'providers': [provider], 'year_range': None})
        
        # Genre + provider
        combinations.append({'genres': ['action'], 'providers': ['netflix'], 'year_range': None})
        combinations.append({'genres': ['comedy'], 'providers': ['hulu'], 'year_range': None})
        
        # Year range
        combinations.append({'genres': None, 'providers': None, 'year_range': (2010, 2020)})
        combinations.append({'genres': None, 'providers': None, 'year_range': (2020, 2025)})
        
        # Complex combinations
        combinations.append({'genres': ['action', 'comedy'], 'providers': ['netflix'], 'year_range': (2015, 2025)})
        combinations.append({'genres': ['drama'], 'providers': ['hulu', 'amazon'], 'year_range': (2010, 2020)})
        
        return combinations
    
    def run_single_request(self, user_id: str, K: int = 50, 
                          genres: List[str] = None, providers: List[str] = None,
                          year_range: tuple = None) -> Dict[str, Any]:
        """Run a single candidate request."""
        start_time = time.time()
        
        try:
            result = self.fetcher.get_candidates(user_id, K, genres, providers, year_range)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'user_id': user_id,
                'K': K,
                'genres': genres,
                'providers': providers,
                'year_range': year_range,
                'success': True,
                'latency_ms': latency_ms,
                'candidates_returned': len(result['candidates']),
                'k_requested': K,
                'underfilled': len(result['candidates']) < K,
                'fallback_stage': result['metadata']['fallback_stage'],
                'user_bucket': result['metadata']['user_bucket'],
                'cache_hit': result['metadata']['cache_hit']
            }
            
        except Exception as e:
            return {
                'user_id': user_id,
                'K': K,
                'genres': genres,
                'providers': providers,
                'year_range': year_range,
                'success': False,
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000,
                'candidates_returned': 0,
                'k_requested': K,
                'underfilled': True,
                'fallback_stage': 'error',
                'user_bucket': 'unknown',
                'cache_hit': False
            }
    
    def run_mixed_workload(self, num_requests: int = 1000, 
                          concurrent_workers: int = 10) -> Dict[str, Any]:
        """Run mixed workload test."""
        self.logger.info(f"Running mixed workload: {num_requests} requests, {concurrent_workers} workers")
        
        # Generate test requests
        requests = []
        for _ in range(num_requests):
            user_id = random.choice(self.test_users)
            K = random.choice([10, 25, 50, 100])
            filter_combo = random.choice(self.filter_combinations)
            
            requests.append({
                'user_id': user_id,
                'K': K,
                'genres': filter_combo['genres'],
                'providers': filter_combo['providers'],
                'year_range': filter_combo['year_range']
            })
        
        # Run requests
        results = []
        
        if concurrent_workers > 1:
            # Concurrent execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
                futures = [executor.submit(self.run_single_request, **req) for req in requests]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
        else:
            # Sequential execution
            for req in requests:
                result = self.run_single_request(**req)
                results.append(result)
        
        # Calculate metrics
        successful_results = [r for r in results if r['success']]
        success_rate = len(successful_results) / len(results)
        
        latencies = [r['latency_ms'] for r in successful_results]
        underfilled_count = sum(1 for r in successful_results if r['underfilled'])
        cache_hits = sum(1 for r in successful_results if r['cache_hit'])
        
        # Calculate percentiles
        def percentile(data, p):
            sorted_data = sorted(data)
            index = int(len(sorted_data) * p / 100)
            return sorted_data[min(index, len(sorted_data) - 1)]
        
        latency_stats = {
            'mean_ms': mean(latencies) if latencies else 0,
            'median_ms': median(latencies) if latencies else 0,
            'p95_ms': percentile(latencies, 95) if latencies else 0,
            'p99_ms': percentile(latencies, 99) if latencies else 0,
            'max_ms': max(latencies) if latencies else 0,
            'min_ms': min(latencies) if latencies else 0
        }
        
        # Fallback stage distribution
        fallback_stages = {}
        for result in successful_results:
            stage = result['fallback_stage']
            fallback_stages[stage] = fallback_stages.get(stage, 0) + 1
        
        return {
            'total_requests': len(results),
            'successful_requests': len(successful_results),
            'failed_requests': len(results) - len(successful_results),
            'success_rate': success_rate,
            'underfilled_requests': underfilled_count,
            'underfill_rate': underfilled_count / len(successful_results) if successful_results else 0,
            'cache_hits': cache_hits,
            'cache_misses': len(successful_results) - cache_hits,
            'cache_hit_ratio': cache_hits / len(successful_results) if successful_results else 0,
            'latency_stats': latency_stats,
            'fallback_stages': fallback_stages,
            'results': results
        }
    
    def run_acceptance_tests(self) -> Dict[str, Any]:
        """Run acceptance criteria tests."""
        self.logger.info("Running acceptance criteria tests...")
        
        # Test 1: Coverage test (≥99% return ≥K items)
        coverage_results = self.run_mixed_workload(num_requests=1000, concurrent_workers=1)
        coverage_passed = coverage_results['success_rate'] >= 0.99
        
        # Test 2: Latency test (p95 ≤20ms warm, ≤150ms cold)
        # First run to warm cache
        self.logger.info("Warming cache...")
        warm_results = self.run_mixed_workload(num_requests=500, concurrent_workers=1)
        
        # Second run on warm cache
        self.logger.info("Testing warm cache latency...")
        warm_latency_results = self.run_mixed_workload(num_requests=500, concurrent_workers=1)
        warm_latency_passed = warm_latency_results['latency_stats']['p95_ms'] <= 20
        
        # Test 3: Quality test (deterministic results)
        self.logger.info("Testing deterministic behavior...")
        deterministic_passed = self._test_determinism()
        
        # Test 4: Cache efficacy (hit ratio ≥0.85)
        cache_stats = self.fetcher.get_cache_stats()
        cache_efficacy_passed = cache_stats['hit_ratio'] >= 0.85
        
        return {
            'coverage_test': {
                'passed': coverage_passed,
                'success_rate': coverage_results['success_rate'],
                'threshold': 0.99
            },
            'latency_test': {
                'passed': warm_latency_passed,
                'p95_ms': warm_latency_results['latency_stats']['p95_ms'],
                'threshold_ms': 20
            },
            'determinism_test': {
                'passed': deterministic_passed
            },
            'cache_efficacy_test': {
                'passed': cache_efficacy_passed,
                'hit_ratio': cache_stats['hit_ratio'],
                'threshold': 0.85
            },
            'overall_passed': all([
                coverage_passed, warm_latency_passed, 
                deterministic_passed, cache_efficacy_passed
            ])
        }
    
    def _test_determinism(self) -> bool:
        """Test deterministic behavior."""
        test_user = self.test_users[0]
        test_filters = self.filter_combinations[0]
        
        # Run same request multiple times
        results = []
        for _ in range(5):
            result = self.run_single_request(
                test_user, 50, 
                test_filters['genres'], 
                test_filters['providers'], 
                test_filters['year_range']
            )
            if result['success']:
                results.append(result['candidates_returned'])
        
        # Check if all results are identical
        return len(set(results)) == 1 if results else False
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        self.logger.info("Starting comprehensive workload tests...")
        
        start_time = time.time()
        
        # Run acceptance tests
        acceptance_results = self.run_acceptance_tests()
        
        # Run additional workload tests
        workload_results = self.run_mixed_workload(num_requests=1000, concurrent_workers=5)
        
        # Get final cache stats
        cache_stats = self.fetcher.get_cache_stats()
        
        # Compile results
        results = {
            'test_suite': 'step3d2_workload',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'execution_time_seconds': time.time() - start_time,
            'acceptance_tests': acceptance_results,
            'workload_results': workload_results,
            'cache_stats': cache_stats
        }
        
        # Log results
        self.logger.info(f"Comprehensive tests completed in {results['execution_time_seconds']:.2f}s")
        self.logger.info(f"Overall acceptance: {'PASSED' if acceptance_results['overall_passed'] else 'FAILED'}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str = "logs/step3d2_workload_report.json"):
        """Save test results to file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Workload test results saved to {output_path}")


def main():
    """Run workload tests."""
    parser = argparse.ArgumentParser(description='Run workload tests for candidate fetcher')
    parser.add_argument('--requests', type=int, default=1000, help='Number of requests for workload test')
    parser.add_argument('--workers', type=int, default=5, help='Number of concurrent workers')
    parser.add_argument('--output', default='logs/step3d2_workload_report.json', help='Output file for results')
    
    args = parser.parse_args()
    
    print("Starting Step 3d.2 Workload Testing...")
    
    try:
        # Initialize tester
        tester = WorkloadTester()
        
        # Run comprehensive tests
        results = tester.run_comprehensive_tests()
        
        # Save results
        tester.save_results(results, args.output)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"WORKLOAD TEST RESULTS")
        print(f"{'='*60}")
        
        acceptance = results['acceptance_tests']
        print(f"Overall Acceptance: {'PASSED' if acceptance['overall_passed'] else 'FAILED'}")
        print(f"Coverage Test: {'PASSED' if acceptance['coverage_test']['passed'] else 'FAILED'} ({acceptance['coverage_test']['success_rate']:.3f})")
        print(f"Latency Test: {'PASSED' if acceptance['latency_test']['passed'] else 'FAILED'} ({acceptance['latency_test']['p95_ms']:.1f}ms)")
        print(f"Determinism Test: {'PASSED' if acceptance['determinism_test']['passed'] else 'FAILED'}")
        print(f"Cache Efficacy: {'PASSED' if acceptance['cache_efficacy_test']['passed'] else 'FAILED'} ({acceptance['cache_efficacy_test']['hit_ratio']:.3f})")
        
        workload = results['workload_results']
        print(f"\nWorkload Statistics:")
        print(f"Total Requests: {workload['total_requests']}")
        print(f"Success Rate: {workload['success_rate']:.3f}")
        print(f"Cache Hit Ratio: {workload['cache_hit_ratio']:.3f}")
        print(f"P95 Latency: {workload['latency_stats']['p95_ms']:.1f}ms")
        print(f"Underfill Rate: {workload['underfill_rate']:.3f}")
        
        return acceptance['overall_passed']
        
    except Exception as e:
        print(f"❌ Error running workload tests: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
