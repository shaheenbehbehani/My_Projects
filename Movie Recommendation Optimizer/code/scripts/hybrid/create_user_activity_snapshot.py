#!/usr/bin/env python3
"""
Create user activity snapshot from raw ratings for fast stratification.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
def setup_logging():
    """Setup logging for user activity snapshot."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "user_activity_snapshot.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def create_user_activity_snapshot(force_rebuild: bool = False):
    """Create user activity snapshot from raw ratings."""
    logger.info("Creating user activity snapshot...")
    
    # Paths
    data_dir = project_root / "data"
    derived_dir = data_dir / "derived"
    derived_dir.mkdir(exist_ok=True)
    
    snapshot_path = derived_dir / "user_activity_snapshot.parquet"
    
    # Check if snapshot exists and force_rebuild is False
    if snapshot_path.exists() and not force_rebuild:
        logger.info(f"User activity snapshot already exists at {snapshot_path}")
        return
    
    # Load raw ratings
    ratings_path = data_dir / "collaborative" / "ratings_long_format.parquet"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
    
    logger.info(f"Loading ratings from {ratings_path}")
    ratings = pd.read_parquet(ratings_path)
    
    # Group by user_index and compute statistics
    logger.info("Computing user activity statistics...")
    user_stats = ratings.groupby('user_index').agg({
        'rating': 'count'
    }).reset_index()
    
    # Flatten column names
    user_stats.columns = ['user_index', 'ratings_count']
    
    # Add dummy timestamps since they're not available
    user_stats['first_ts'] = 0
    user_stats['last_ts'] = 1
    
    # Sort by user_index for consistency
    user_stats = user_stats.sort_values('user_index').reset_index(drop=True)
    
    # Save snapshot
    user_stats.to_parquet(snapshot_path, index=False)
    logger.info(f"Saved user activity snapshot to {snapshot_path}")
    logger.info(f"Snapshot contains {len(user_stats)} users")
    logger.info(f"Rating counts: min={user_stats['ratings_count'].min()}, max={user_stats['ratings_count'].max()}, mean={user_stats['ratings_count'].mean():.1f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create user activity snapshot')
    parser.add_argument('--force-snapshot', action='store_true', help='Force rebuild snapshot')
    args = parser.parse_args()
    
    create_user_activity_snapshot(force_rebuild=args.force_snapshot)
