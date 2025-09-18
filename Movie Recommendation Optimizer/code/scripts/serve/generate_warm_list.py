#!/usr/bin/env python3
"""
Generate cache warm list for Step 3d.2 Candidate Fetcher.
Creates a comprehensive list of user-filter combinations for cache warming.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def generate_warm_list(output_path: str = "data/runtime/cache_warm_list_3d2.csv"):
    """Generate comprehensive cache warm list."""
    
    print("Generating cache warm list...")
    
    # Load user activity data
    user_activity = pd.read_parquet("data/derived/user_activity_snapshot.parquet")
    
    # Get top 1000 most active users
    top_users = user_activity.nlargest(1000, 'ratings_count')['user_index'].tolist()
    print(f"Selected {len(top_users)} most active users")
    
    # Define common filter combinations
    genre_combinations = [
        [],  # No genre filter
        ['action'],
        ['comedy'],
        ['drama'],
        ['action', 'comedy'],
        ['drama', 'romance'],
        ['action', 'drama'],
        ['comedy', 'romance'],
        ['thriller'],
        ['horror'],
        ['sci-fi'],
        ['action', 'sci-fi'],
        ['comedy', 'drama'],
        ['action', 'thriller'],
        ['drama', 'thriller']
    ]
    
    provider_combinations = [
        [],  # No provider filter
        ['netflix'],
        ['hulu'],
        ['amazon'],
        ['netflix', 'hulu'],
        ['netflix', 'amazon'],
        ['hulu', 'amazon'],
        ['netflix', 'hulu', 'amazon']
    ]
    
    year_ranges = [
        None,  # No year filter
        (2010, 2020),
        (2015, 2025),
        (2020, 2025),
        (2000, 2010),
        (1990, 2000)
    ]
    
    k_values = [10, 25, 50, 100]
    
    # Generate warm list entries
    warm_entries = []
    
    # Add top users with various filter combinations
    for user_id in top_users[:500]:  # Top 500 users
        for genres in genre_combinations[:10]:  # Top 10 genre combinations
            for providers in provider_combinations[:5]:  # Top 5 provider combinations
                for year_range in year_ranges[:3]:  # Top 3 year ranges
                    for K in k_values[:2]:  # Top 2 K values
                        warm_entries.append({
                            'user_id': user_id,
                            'genres': genres,
                            'providers': providers,
                            'year_range': year_range,
                            'K': K,
                            'priority': 'high'
                        })
    
    # Add synthetic users for comprehensive coverage
    synthetic_users = list(range(100000, 100100))  # 100 synthetic users
    
    for user_id in synthetic_users:
        for genres in genre_combinations[:5]:  # Fewer combinations for synthetic users
            for providers in provider_combinations[:3]:
                for year_range in year_ranges[:2]:
                    for K in [50]:  # Standard K value
                        warm_entries.append({
                            'user_id': user_id,
                            'genres': genres,
                            'providers': providers,
                            'year_range': year_range,
                            'K': K,
                            'priority': 'medium'
                        })
    
    # Add edge cases
    edge_cases = [
        {'user_id': '999999', 'genres': [], 'providers': [], 'year_range': None, 'K': 50, 'priority': 'edge'},
        {'user_id': '0', 'genres': ['action'], 'providers': ['netflix'], 'year_range': (2020, 2025), 'K': 10, 'priority': 'edge'},
        {'user_id': '1', 'genres': ['comedy', 'drama'], 'providers': ['hulu'], 'year_range': None, 'K': 25, 'priority': 'edge'},
    ]
    
    warm_entries.extend(edge_cases)
    
    # Create DataFrame
    warm_df = pd.DataFrame(warm_entries)
    
    # Shuffle for realistic load pattern
    warm_df = warm_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    warm_df.to_csv(output_path, index=False)
    
    print(f"Generated {len(warm_entries)} warm list entries")
    print(f"Saved to: {output_path}")
    
    # Print summary
    print("\nWarm List Summary:")
    print(f"Total entries: {len(warm_entries)}")
    print(f"Unique users: {warm_df['user_id'].nunique()}")
    print(f"Genre combinations: {len(genre_combinations)}")
    print(f"Provider combinations: {len(provider_combinations)}")
    print(f"Year ranges: {len(year_ranges)}")
    print(f"K values: {k_values}")
    
    return warm_df

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate cache warm list')
    parser.add_argument('--output', default='data/runtime/cache_warm_list_3d2.csv',
                       help='Output path for warm list CSV')
    
    args = parser.parse_args()
    
    try:
        warm_df = generate_warm_list(args.output)
        print("✅ Warm list generation completed successfully")
        
    except Exception as e:
        print(f"❌ Error generating warm list: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()






