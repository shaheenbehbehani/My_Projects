#!/usr/bin/env python3
"""
Test Script for Experiment Assignment Function

This script validates the experiment assignment function by testing:
- Assignment consistency across multiple calls
- Proper bucket distribution
- Hash function determinism
- Configuration loading

Author: Movie Recommendation Optimizer Team
Version: 1.0
Created: 2025-09-07
"""

import sys
import os
import json
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from scripts.serve.experiment_assignment import ExperimentAssignment


def test_assignment_consistency():
    """Test that user assignments are consistent across multiple calls."""
    print("🧪 Testing assignment consistency...")
    
    assigner = ExperimentAssignment("hybrid_v1")
    test_users = [f"user_{i}" for i in range(1000)]
    
    # Get assignments twice
    assignments1 = [assigner.assign_user(user_id) for user_id in test_users]
    assignments2 = [assigner.assign_user(user_id) for user_id in test_users]
    
    # Compare assignments
    inconsistent_count = 0
    for i, (a1, a2) in enumerate(zip(assignments1, assignments2)):
        if a1 != a2:
            print(f"❌ Inconsistent assignment for user {test_users[i]}: {a1} != {a2}")
            inconsistent_count += 1
    
    if inconsistent_count == 0:
        print("✅ All assignments are consistent")
        return True
    else:
        print(f"❌ Found {inconsistent_count} inconsistent assignments")
        return False


def test_bucket_distribution():
    """Test that bucket distribution matches expected percentages."""
    print("\n🧪 Testing bucket distribution...")
    
    assigner = ExperimentAssignment("hybrid_v1")
    test_users = [f"user_{i}" for i in range(10000)]  # Larger sample for better accuracy
    
    # Get assignments
    assignments = [assigner.assign_user(user_id) for user_id in test_users]
    
    # Count bucket distribution
    bucket_counts = {}
    for assignment in assignments:
        bucket = assignment["bucket"]
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    
    # Calculate percentages
    total_users = len(test_users)
    print(f"📊 Bucket Distribution (n={total_users}):")
    
    expected_distributions = {
        "control": 20.0,
        "treatment_a": 40.0,
        "treatment_b": 40.0
    }
    
    all_within_tolerance = True
    for bucket, count in bucket_counts.items():
        percentage = count / total_users * 100
        expected = expected_distributions.get(bucket, 0)
        tolerance = 2.0  # 2% tolerance
        
        within_tolerance = abs(percentage - expected) <= tolerance
        status = "✅" if within_tolerance else "❌"
        
        print(f"  {status} {bucket}: {count} users ({percentage:.1f}%) [expected: {expected:.1f}%]")
        
        if not within_tolerance:
            all_within_tolerance = False
    
    return all_within_tolerance


def test_hash_determinism():
    """Test that hash function is deterministic."""
    print("\n🧪 Testing hash determinism...")
    
    assigner = ExperimentAssignment("hybrid_v1")
    test_users = [f"user_{i}" for i in range(100)]
    
    # Get hash values twice
    hashes1 = [assigner._compute_user_hash(user_id) for user_id in test_users]
    hashes2 = [assigner._compute_user_hash(user_id) for user_id in test_users]
    
    # Compare hashes
    inconsistent_count = 0
    for i, (h1, h2) in enumerate(zip(hashes1, hashes2)):
        if h1 != h2:
            print(f"❌ Inconsistent hash for user {test_users[i]}: {h1} != {h2}")
            inconsistent_count += 1
    
    if inconsistent_count == 0:
        print("✅ All hashes are deterministic")
        return True
    else:
        print(f"❌ Found {inconsistent_count} inconsistent hashes")
        return False


def test_bucket_gate_alpha():
    """Test bucket-gate alpha selection."""
    print("\n🧪 Testing bucket-gate alpha selection...")
    
    assigner = ExperimentAssignment("hybrid_v1")
    
    # Test different activity levels
    activity_levels = ["cold", "light", "medium", "heavy"]
    expected_alphas = [0.20, 0.40, 0.60, 0.80]
    
    all_correct = True
    for activity_level, expected_alpha in zip(activity_levels, expected_alphas):
        actual_alpha = assigner._get_bucket_gate_alpha(activity_level)
        if actual_alpha == expected_alpha:
            print(f"✅ {activity_level}: α = {actual_alpha}")
        else:
            print(f"❌ {activity_level}: expected α = {expected_alpha}, got α = {actual_alpha}")
            all_correct = False
    
    return all_correct


def test_configuration_loading():
    """Test configuration loading from file."""
    print("\n🧪 Testing configuration loading...")
    
    # Test with default config
    assigner1 = ExperimentAssignment("hybrid_v1")
    summary1 = assigner1.get_experiment_summary()
    
    print(f"📋 Experiment Summary:")
    print(f"  Experiment ID: {summary1['experiment_id']}")
    print(f"  Total Traffic: {summary1['total_traffic']}%")
    
    for bucket_name, bucket_info in summary1['buckets'].items():
        print(f"  {bucket_name}: {bucket_info['traffic_percentage']:.1f}% traffic")
        print(f"    Algorithm: {bucket_info['algorithm']}")
        print(f"    Alpha: {bucket_info['alpha']}")
    
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing edge cases...")
    
    assigner = ExperimentAssignment("hybrid_v1")
    
    # Test with empty user ID
    try:
        assignment = assigner.assign_user("")
        print(f"✅ Empty user ID handled: {assignment['bucket']}")
    except Exception as e:
        print(f"❌ Empty user ID failed: {e}")
        return False
    
    # Test with special characters
    try:
        assignment = assigner.assign_user("user@domain.com")
        print(f"✅ Special characters handled: {assignment['bucket']}")
    except Exception as e:
        print(f"❌ Special characters failed: {e}")
        return False
    
    # Test with very long user ID
    try:
        long_user_id = "user_" + "x" * 1000
        assignment = assigner.assign_user(long_user_id)
        print(f"✅ Long user ID handled: {assignment['bucket']}")
    except Exception as e:
        print(f"❌ Long user ID failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("🚀 Starting Experiment Assignment Tests\n")
    
    tests = [
        ("Assignment Consistency", test_assignment_consistency),
        ("Bucket Distribution", test_bucket_distribution),
        ("Hash Determinism", test_hash_determinism),
        ("Bucket-Gate Alpha", test_bucket_gate_alpha),
        ("Configuration Loading", test_configuration_loading),
        ("Edge Cases", test_edge_cases)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED\n")
            else:
                print(f"❌ {test_name} FAILED\n")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}\n")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Experiment assignment is ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix issues before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())





