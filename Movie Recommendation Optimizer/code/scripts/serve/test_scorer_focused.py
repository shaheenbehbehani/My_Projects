#!/usr/bin/env python3
"""
Step 3d.1 - Focused Scoring Service Test Suite
Focused testing for the MovieRecommendationScorer with available data.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures
from statistics import mean, median, stdev

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.serve.scorer_entrypoint import MovieRecommendationScorer

class FocusedScorerTestSuite:
    """Focused test suite for the scoring service with available data."""
    
    def __init__(self, release_lock_path: str = "data/hybrid/release_lock_3d.json"):
        """Initialize focused test suite."""
        self.logger = self._setup_logging()
        self.logger.info("Initializing FocusedScorerTestSuite")
        
        # Initialize scorer
        self.scorer = MovieRecommendationScorer(release_lock_path)
        
        # Get users with actual candidate files
        self.test_users = self._get_users_with_candidates()
        
        self.logger.info(f"Focused test suite initialized with {len(self.test_users)} users with candidates")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for test suite."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("focused_test_suite")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('logs/step3d1_scorer.log')
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
    
    def _get_users_with_candidates(self) -> List[str]:
        """Get users that have candidate files."""
        candidates_dir = Path("data/hybrid/candidates")
        users = []
        
        if candidates_dir.exists():
            for file in candidates_dir.glob("user_*_candidates.parquet"):
                user_id = file.stem.replace("user_", "").replace("_candidates", "")
                users.append(user_id)
        
        # Sort for deterministic testing
        users.sort(key=int)
        return users
    
    def test_deterministic_behavior(self) -> Dict[str, Any]:
        """Test that the scorer produces deterministic results."""
        self.logger.info("Testing deterministic behavior...")
        
        if not self.test_users:
            return {
                'test_name': 'deterministic_behavior',
                'passed': False,
                'error': 'No users with candidates available'
            }
        
        test_user = self.test_users[0]
        K = 10
        
        # Run multiple times with same parameters
        results = []
        for i in range(5):
            result = self.scorer.recommend(test_user, K)
            results.append(result)
        
        # Check that all results are identical
        first_result = results[0]
        is_deterministic = all(
            result['recommendations'] == first_result['recommendations']
            for result in results[1:]
        )
        
        # Check that scores are identical (within floating point precision)
        scores_identical = all(
            np.allclose(result['scores']['hybrid_scores'], first_result['scores']['hybrid_scores'])
            for result in results[1:]
        )
        
        return {
            'test_name': 'deterministic_behavior',
            'passed': is_deterministic and scores_identical,
            'num_runs': len(results),
            'recommendations_identical': is_deterministic,
            'scores_identical': scores_identical,
            'sample_result': first_result
        }
    
    def test_smoke_tests(self) -> Dict[str, Any]:
        """Run smoke tests on users with candidates."""
        self.logger.info(f"Running smoke tests on {len(self.test_users)} users with candidates...")
        
        results = []
        errors = []
        
        for i, user_id in enumerate(self.test_users):
            try:
                result = self.scorer.recommend(user_id, K=10)
                results.append({
                    'user_id': user_id,
                    'success': True,
                    'num_recommendations': len(result['recommendations']),
                    'alpha_used': result['metadata']['alpha_used'],
                    'processing_time_ms': result['metadata']['processing_time_ms'],
                    'user_bucket': result['metadata']['user_bucket'],
                    'scoring_method': result['metadata']['scoring_method']
                })
                
                if i % 5 == 0:
                    self.logger.info(f"Processed {i+1}/{len(self.test_users)} users")
                    
            except Exception as e:
                error_info = {
                    'user_id': user_id,
                    'success': False,
                    'error': str(e)
                }
                errors.append(error_info)
                self.logger.error(f"Error processing user {user_id}: {e}")
        
        # Calculate statistics
        successful_results = [r for r in results if r['success']]
        success_rate = len(successful_results) / len(self.test_users) if self.test_users else 0
        
        processing_times = [r['processing_time_ms'] for r in successful_results]
        
        return {
            'test_name': 'smoke_tests',
            'total_users': len(self.test_users),
            'successful_users': len(successful_results),
            'failed_users': len(errors),
            'success_rate': success_rate,
            'passed': success_rate >= 0.95,  # 95% success rate required
            'processing_times': {
                'mean_ms': mean(processing_times) if processing_times else 0,
                'median_ms': median(processing_times) if processing_times else 0,
                'std_ms': stdev(processing_times) if len(processing_times) > 1 else 0,
                'min_ms': min(processing_times) if processing_times else 0,
                'max_ms': max(processing_times) if processing_times else 0
            },
            'errors': errors[:5]  # First 5 errors for debugging
        }
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance under various load conditions."""
        self.logger.info("Running performance benchmarks...")
        
        if not self.test_users:
            return {
                'test_name': 'performance_benchmarks',
                'passed': False,
                'error': 'No users with candidates available'
            }
        
        # Single user performance
        single_user_times = []
        test_user = self.test_users[0]
        
        for _ in range(10):
            start_time = time.time()
            self.scorer.recommend(test_user, K=50)
            single_user_times.append((time.time() - start_time) * 1000)
        
        # Concurrent performance (simulate multiple users)
        concurrent_times = []
        
        def process_user(user_id):
            start_time = time.time()
            result = self.scorer.recommend(user_id, K=50)
            return (time.time() - start_time) * 1000
        
        # Test with 5 concurrent users
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_user, user_id) for user_id in self.test_users[:5]]
            concurrent_times = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Calculate percentiles
        def percentile(data, p):
            sorted_data = sorted(data)
            index = int(len(sorted_data) * p / 100)
            return sorted_data[min(index, len(sorted_data) - 1)]
        
        single_user_stats = {
            'mean_ms': mean(single_user_times),
            'median_ms': median(single_user_times),
            'p95_ms': percentile(single_user_times, 95),
            'p99_ms': percentile(single_user_times, 99),
            'max_ms': max(single_user_times)
        }
        
        concurrent_stats = {
            'mean_ms': mean(concurrent_times),
            'median_ms': median(concurrent_times),
            'p95_ms': percentile(concurrent_times, 95),
            'p99_ms': percentile(concurrent_times, 99),
            'max_ms': max(concurrent_times)
        }
        
        # Performance targets (adjust based on requirements)
        p95_target_ms = 1000  # 1 second p95 latency target
        single_user_passed = single_user_stats['p95_ms'] <= p95_target_ms
        concurrent_passed = concurrent_stats['p95_ms'] <= p95_target_ms * 2  # Allow 2x for concurrency
        
        return {
            'test_name': 'performance_benchmarks',
            'single_user': single_user_stats,
            'concurrent_5_users': concurrent_stats,
            'targets': {
                'p95_target_ms': p95_target_ms,
                'concurrent_p95_target_ms': p95_target_ms * 2
            },
            'passed': single_user_passed and concurrent_passed,
            'single_user_passed': single_user_passed,
            'concurrent_passed': concurrent_passed
        }
    
    def test_offline_validation(self) -> Dict[str, Any]:
        """Validate against offline hybrid scores from Step 3c."""
        self.logger.info("Validating against offline scores...")
        
        if not self.test_users:
            return {
                'test_name': 'offline_validation',
                'passed': False,
                'error': 'No users with candidates available'
            }
        
        # Load offline results for comparison
        offline_results = self._load_offline_results()
        
        validation_results = []
        
        # Test on users that have both online and offline results
        test_users = [user for user in self.test_users if user in offline_results]
        
        for user_id in test_users[:10]:  # Test first 10 users
            try:
                # Get online results
                online_result = self.scorer.recommend(user_id, K=10)
                online_recs = online_result['recommendations']
                
                # Get offline results
                offline_recs = offline_results[user_id]['recommendations']
                
                # Calculate overlap metrics
                online_set = set(online_recs)
                offline_set = set(offline_recs)
                
                intersection = online_set.intersection(offline_set)
                union = online_set.union(offline_set)
                
                jaccard_similarity = len(intersection) / len(union) if union else 0
                recall_at_10 = len(intersection) / len(offline_set) if offline_set else 0
                
                validation_results.append({
                    'user_id': user_id,
                    'jaccard_similarity': jaccard_similarity,
                    'recall_at_10': recall_at_10,
                    'online_recs': online_recs,
                    'offline_recs': offline_recs
                })
                
            except Exception as e:
                self.logger.warning(f"Validation failed for user {user_id}: {e}")
        
        # Calculate aggregate metrics
        jaccard_scores = [r['jaccard_similarity'] for r in validation_results]
        recall_scores = [r['recall_at_10'] for r in validation_results]
        
        return {
            'test_name': 'offline_validation',
            'users_tested': len(validation_results),
            'jaccard_similarity': {
                'mean': mean(jaccard_scores) if jaccard_scores else 0,
                'median': median(jaccard_scores) if jaccard_scores else 0,
                'min': min(jaccard_scores) if jaccard_scores else 0,
                'max': max(jaccard_scores) if jaccard_scores else 0
            },
            'recall_at_10': {
                'mean': mean(recall_scores) if recall_scores else 0,
                'median': median(recall_scores) if recall_scores else 0,
                'min': min(recall_scores) if recall_scores else 0,
                'max': max(recall_scores) if recall_scores else 0
            },
            'passed': mean(jaccard_scores) >= 0.5 if jaccard_scores else False,  # 50% similarity threshold
            'detailed_results': validation_results[:3]  # First 3 for debugging
        }
    
    def _load_offline_results(self) -> Dict[str, Any]:
        """Load offline results for validation."""
        offline_results = {}
        
        # Load from candidates directory (these are pre-computed)
        candidates_dir = Path("data/hybrid/candidates")
        for file in candidates_dir.glob("user_*_candidates.parquet"):
            user_id = file.stem.replace("user_", "").replace("_candidates", "")
            df = pd.read_parquet(file)
            offline_results[user_id] = {
                'recommendations': df['canonical_id'].tolist(),
                'scores': df['hybrid_score'].tolist()
            }
        
        return offline_results
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with invalid inputs."""
        self.logger.info("Testing error handling...")
        
        error_tests = [
            {'user_id': 'invalid_user', 'k': 10, 'expected_behavior': 'fallback'},
            {'user_id': '999999', 'k': 10, 'expected_behavior': 'fallback'},
            {'user_id': '1', 'k': 0, 'expected_behavior': 'empty_result'},
            {'user_id': '1', 'k': -1, 'expected_behavior': 'error_or_fallback'}
        ]
        
        results = []
        
        for test in error_tests:
            try:
                result = self.scorer.recommend(test['user_id'], test['k'])
                results.append({
                    'test_case': test,
                    'success': True,
                    'num_recommendations': len(result['recommendations']),
                    'has_error': 'error' in result,
                    'scoring_method': result['metadata']['scoring_method']
                })
            except Exception as e:
                results.append({
                    'test_case': test,
                    'success': False,
                    'error': str(e)
                })
        
        # Check that all tests handled gracefully (no crashes)
        all_handled = all(r['success'] for r in results)
        
        return {
            'test_name': 'error_handling',
            'passed': all_handled,
            'test_cases': len(error_tests),
            'successful_handling': sum(1 for r in results if r['success']),
            'results': results
        }
    
    def test_cli_functionality(self) -> Dict[str, Any]:
        """Test CLI functionality."""
        self.logger.info("Testing CLI functionality...")
        
        import subprocess
        
        cli_tests = [
            {'args': ['--user-id', '1', '--k', '5'], 'expected_success': True},
            {'args': ['--user-id', '999', '--k', '10'], 'expected_success': True},  # Should fallback
            {'args': ['--user-id', '1', '--k', '10', '--output', '/tmp/test_output.json'], 'expected_success': True}
        ]
        
        results = []
        
        for test in cli_tests:
            try:
                cmd = ['python', 'scripts/serve/scorer_entrypoint.py'] + test['args']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                success = result.returncode == 0
                results.append({
                    'test_case': test,
                    'success': success,
                    'returncode': result.returncode,
                    'stdout': result.stdout[:200] if result.stdout else '',
                    'stderr': result.stderr[:200] if result.stderr else ''
                })
                
            except Exception as e:
                results.append({
                    'test_case': test,
                    'success': False,
                    'error': str(e)
                })
        
        # Check that all tests handled gracefully
        all_handled = all(r['success'] for r in results)
        
        return {
            'test_name': 'cli_functionality',
            'passed': all_handled,
            'test_cases': len(cli_tests),
            'successful_handling': sum(1 for r in results if r['success']),
            'results': results
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all focused tests and return comprehensive results."""
        self.logger.info("Starting focused test suite...")
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_deterministic_behavior(),
            self.test_smoke_tests(),
            self.test_performance_benchmarks(),
            self.test_offline_validation(),
            self.test_error_handling(),
            self.test_cli_functionality()
        ]
        
        # Calculate overall results
        total_tests = len(tests)
        passed_tests = sum(1 for test in tests if test['passed'])
        
        overall_passed = passed_tests == total_tests
        
        # Compile results
        results = {
            'test_suite': 'step3d1_scorer_focused',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_passed': overall_passed,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'execution_time_seconds': time.time() - start_time,
            'tests': {test['test_name']: test for test in tests}
        }
        
        # Log results
        self.logger.info(f"Focused test suite completed: {passed_tests}/{total_tests} tests passed")
        self.logger.info(f"Overall result: {'PASSED' if overall_passed else 'FAILED'}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str = "logs/step3d1_focused_test_results.json"):
        """Save test results to file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Focused test results saved to {output_path}")


def main():
    """Run the focused test suite."""
    print("Starting Step 3d.1 Focused Scoring Service Test Suite...")
    
    # Initialize test suite
    test_suite = FocusedScorerTestSuite()
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Save results
    test_suite.save_results(results)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"FOCUSED TEST SUITE RESULTS")
    print(f"{'='*60}")
    print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    print(f"Tests: {results['passed_tests']}/{results['total_tests']} passed")
    print(f"Execution time: {results['execution_time_seconds']:.2f} seconds")
    
    print(f"\nDetailed Results:")
    for test_name, test_result in results['tests'].items():
        status = "PASSED" if test_result['passed'] else "FAILED"
        print(f"  {test_name}: {status}")
    
    return results['overall_passed']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)






