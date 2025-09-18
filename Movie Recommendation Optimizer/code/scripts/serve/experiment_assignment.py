#!/usr/bin/env python3
"""
Experiment Assignment Function - Movie Recommendation Optimizer

This module provides deterministic user-to-bucket assignment for A/B experiments.
It ensures stable, reproducible assignment of users to experiment buckets while
maintaining proper traffic splits and configuration management.

Author: Movie Recommendation Optimizer Team
Version: 1.0
Created: 2025-09-07
"""

import hashlib
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BucketConfig:
    """Configuration for an experiment bucket."""
    name: str
    min_hash: int
    max_hash: int
    alpha: Any  # Can be float or "bucket_gate"
    algorithm: str
    description: str


class ExperimentAssignment:
    """
    Deterministic user-to-bucket assignment for A/B experiments.
    
    This class provides stable, reproducible assignment of users to experiment
    buckets using consistent hashing. It supports multiple experiment versions
    and maintains proper traffic splits.
    """
    
    def __init__(self, experiment_id: str = "hybrid_v1", config_path: Optional[str] = None):
        """
        Initialize experiment assignment.
        
        Args:
            experiment_id: Unique identifier for the experiment
            config_path: Optional path to bucket configuration file
        """
        self.experiment_id = experiment_id
        self.config_path = config_path or "data/experiments/bucket_config.json"
        
        # Load bucket configuration
        self.bucket_configs = self._load_bucket_configs()
        
        logger.info(f"Initialized experiment assignment for '{experiment_id}'")
        logger.info(f"Loaded {len(self.bucket_configs)} bucket configurations")
    
    def _load_bucket_configs(self) -> Dict[str, BucketConfig]:
        """Load bucket configurations from file or use defaults."""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                return self._parse_config_data(config_data)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")
                logger.info("Using default bucket configuration")
        
        return self._get_default_configs()
    
    def _parse_config_data(self, config_data: Dict[str, Any]) -> Dict[str, BucketConfig]:
        """Parse configuration data into BucketConfig objects."""
        configs = {}
        for bucket_name, config in config_data.get("buckets", {}).items():
            configs[bucket_name] = BucketConfig(
                name=bucket_name,
                min_hash=config["min_hash"],
                max_hash=config["max_hash"],
                alpha=config["alpha"],
                algorithm=config["algorithm"],
                description=config.get("description", "")
            )
        return configs
    
    def _get_default_configs(self) -> Dict[str, BucketConfig]:
        """Get default bucket configurations."""
        return {
            "control": BucketConfig(
                name="control",
                min_hash=0,
                max_hash=19,
                alpha=0.0,
                algorithm="content_only",
                description="Content-only baseline (20% traffic)"
            ),
            "treatment_a": BucketConfig(
                name="treatment_a",
                min_hash=20,
                max_hash=59,
                alpha="bucket_gate",
                algorithm="hybrid",
                description="Hybrid with bucket-gate logic (40% traffic)"
            ),
            "treatment_b": BucketConfig(
                name="treatment_b",
                min_hash=60,
                max_hash=99,
                alpha=1.0,
                algorithm="collaborative_only",
                description="Collaborative-only baseline (40% traffic)"
            )
        }
    
    def _compute_user_hash(self, user_id: str) -> int:
        """
        Compute deterministic hash for user assignment.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Integer hash value (0-99)
        """
        # Create deterministic seed from user_id + experiment_id
        seed_string = f"{user_id}:{self.experiment_id}"
        seed_hash = hashlib.sha256(seed_string.encode()).hexdigest()
        
        # Use first 8 hex characters as integer, then mod 100
        seed_int = int(seed_hash[:8], 16)
        return seed_int % 100
    
    def _get_user_activity_level(self, user_id: str) -> str:
        """
        Determine user activity level for bucket-gate alpha selection.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Activity level: 'cold', 'light', 'medium', 'heavy'
        """
        # TODO: Implement actual user activity lookup
        # For now, return a deterministic assignment based on user_id
        hash_val = self._compute_user_hash(user_id)
        
        if hash_val < 10:
            return "cold"
        elif hash_val < 30:
            return "light"
        elif hash_val < 80:
            return "medium"
        else:
            return "heavy"
    
    def _get_bucket_gate_alpha(self, activity_level: str) -> float:
        """
        Get alpha value based on user activity level.
        
        Args:
            activity_level: User activity level
            
        Returns:
            Alpha value for hybrid scoring
        """
        alpha_map = {
            "cold": 0.20,
            "light": 0.40,
            "medium": 0.60,
            "heavy": 0.80
        }
        return alpha_map.get(activity_level, 0.50)  # Default to 0.50
    
    def assign_user(self, user_id: str) -> Dict[str, Any]:
        """
        Assign user to experiment bucket with configuration.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Dictionary containing bucket assignment and configuration
        """
        # Compute deterministic hash
        user_hash = self._compute_user_hash(user_id)
        
        # Find matching bucket
        for bucket_name, config in self.bucket_configs.items():
            if config.min_hash <= user_hash <= config.max_hash:
                assignment = {
                    "user_id": user_id,
                    "bucket": bucket_name,
                    "alpha": config.alpha,
                    "algorithm": config.algorithm,
                    "user_hash": user_hash,
                    "assignment_hash": hashlib.sha256(f"{user_id}:{self.experiment_id}".encode()).hexdigest()[:16],
                    "experiment_id": self.experiment_id
                }
                
                # Resolve bucket-gate alpha if needed
                if config.alpha == "bucket_gate":
                    activity_level = self._get_user_activity_level(user_id)
                    assignment["alpha"] = self._get_bucket_gate_alpha(activity_level)
                    assignment["activity_level"] = activity_level
                
                return assignment
        
        # Fallback to control bucket
        logger.warning(f"User {user_id} hash {user_hash} not in any bucket, assigning to control")
        return self._get_fallback_assignment(user_id, user_hash)
    
    def _get_fallback_assignment(self, user_id: str, user_hash: int) -> Dict[str, Any]:
        """Get fallback assignment for users not in any bucket."""
        control_config = self.bucket_configs["control"]
        return {
            "user_id": user_id,
            "bucket": "control",
            "alpha": control_config.alpha,
            "algorithm": control_config.algorithm,
            "user_hash": user_hash,
            "assignment_hash": hashlib.sha256(f"{user_id}:{self.experiment_id}".encode()).hexdigest()[:16],
            "experiment_id": self.experiment_id,
            "fallback": True
        }
    
    def get_bucket_distribution(self, user_ids: list) -> Dict[str, int]:
        """
        Get bucket distribution for a list of users.
        
        Args:
            user_ids: List of user identifiers
            
        Returns:
            Dictionary mapping bucket names to user counts
        """
        assignments = [self.assign_user(user_id) for user_id in user_ids]
        distribution = {}
        
        for assignment in assignments:
            bucket = assignment["bucket"]
            distribution[bucket] = distribution.get(bucket, 0) + 1
        
        return distribution
    
    def validate_assignment_consistency(self, user_ids: list) -> bool:
        """
        Validate that user assignments are consistent across multiple calls.
        
        Args:
            user_ids: List of user identifiers to test
            
        Returns:
            True if all assignments are consistent, False otherwise
        """
        logger.info(f"Validating assignment consistency for {len(user_ids)} users")
        
        # Get assignments twice
        assignments1 = [self.assign_user(user_id) for user_id in user_ids]
        assignments2 = [self.assign_user(user_id) for user_id in user_ids]
        
        # Compare assignments
        for i, (a1, a2) in enumerate(zip(assignments1, assignments2)):
            if a1 != a2:
                logger.error(f"Inconsistent assignment for user {user_ids[i]}: {a1} != {a2}")
                return False
        
        logger.info("All assignments are consistent")
        return True
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get summary of experiment configuration.
        
        Returns:
            Dictionary containing experiment summary
        """
        total_range = 100
        summary = {
            "experiment_id": self.experiment_id,
            "total_traffic": total_range,
            "buckets": {}
        }
        
        for bucket_name, config in self.bucket_configs.items():
            traffic_pct = (config.max_hash - config.min_hash + 1) / total_range * 100
            summary["buckets"][bucket_name] = {
                "traffic_percentage": traffic_pct,
                "hash_range": f"{config.min_hash}-{config.max_hash}",
                "alpha": config.alpha,
                "algorithm": config.algorithm,
                "description": config.description
            }
        
        return summary


def main():
    """Test the experiment assignment function."""
    # Initialize assignment
    assigner = ExperimentAssignment("hybrid_v1")
    
    # Test with sample users
    test_users = [f"user_{i}" for i in range(1000)]
    
    # Validate consistency
    if assigner.validate_assignment_consistency(test_users[:100]):
        print("✅ Assignment consistency validated")
    else:
        print("❌ Assignment consistency failed")
        return
    
    # Get bucket distribution
    distribution = assigner.get_bucket_distribution(test_users)
    print(f"\n📊 Bucket Distribution:")
    for bucket, count in distribution.items():
        percentage = count / len(test_users) * 100
        print(f"  {bucket}: {count} users ({percentage:.1f}%)")
    
    # Show experiment summary
    summary = assigner.get_experiment_summary()
    print(f"\n🔧 Experiment Summary:")
    print(f"  Experiment ID: {summary['experiment_id']}")
    print(f"  Total Traffic: {summary['total_traffic']}%")
    
    for bucket_name, bucket_info in summary['buckets'].items():
        print(f"  {bucket_name}: {bucket_info['traffic_percentage']:.1f}% traffic")
        print(f"    Algorithm: {bucket_info['algorithm']}")
        print(f"    Alpha: {bucket_info['alpha']}")
    
    # Test individual user assignment
    print(f"\n👤 Sample User Assignments:")
    for user_id in test_users[:5]:
        assignment = assigner.assign_user(user_id)
        print(f"  {user_id}: {assignment['bucket']} (α={assignment['alpha']}, {assignment['algorithm']})")


if __name__ == "__main__":
    main()





