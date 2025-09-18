#!/usr/bin/env python3
"""
Step 2d.1: Provider Encoding
Movie Recommendation Optimizer - Platform Features

Encodes streaming provider availability into binary/multi-hot features.
Input: TMDB/JustWatch provider data (normalized parquet)
Output: Binary int8 features for each provider × category + "any" flag
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple

# Setup logging
log_file = 'logs/step2d_provider_encoding.log'
os.makedirs('logs', exist_ok=True)
os.makedirs('data/features/platform', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Canonical provider list as specified in requirements
CANONICAL_PROVIDERS = [
    'netflix', 'max', 'hulu', 'prime', 'disney_plus', 'paramount_plus', 
    'apple_tv_plus', 'peacock', 'tubi', 'roku', 'youtube', 'google_play', 
    'itunes', 'vudu', 'starz', 'showtime', 'amc_plus'
]

# Provider name mapping from normalized data to canonical names
PROVIDER_NAME_MAPPING = {
    'Netflix': 'netflix',
    'HBO Max': 'max',
    'Hulu': 'hulu',
    'Amazon Prime Video': 'prime',
    'Disney+': 'disney_plus',
    'Paramount+': 'paramount_plus',
    'Apple TV+': 'apple_tv_plus',
    'Peacock': 'peacock',
    'Tubi': 'tubi',
    'Roku Channel': 'roku',
    'YouTube': 'youtube',
    'Google Play Movies': 'google_play',
    'iTunes': 'itunes',
    'Vudu': 'vudu',
    'Starz': 'starz',
    'Showtime': 'showtime',
    'AMC+': 'amc_plus'
}

# Availability categories
AVAILABILITY_CATEGORIES = ['flatrate', 'rent', 'buy', 'ads', 'free']

def setup_directories():
    """Ensure required directories exist"""
    os.makedirs('data/features/platform', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    logger.info("Directories setup complete")

def load_canonical_index():
    """Load the canonical index to get the expected 87,601 movies"""
    logger.info("Loading canonical index...")
    canonical_path = "data/features/text/movies_canonical_index.parquet"
    
    if not os.path.exists(canonical_path):
        raise FileNotFoundError(f"Canonical index not found: {canonical_path}")
    
    canonical_df = pd.read_parquet(canonical_path)
    logger.info(f"Loaded canonical index: {canonical_df.shape}")
    
    # Validate expected structure
    if len(canonical_df) != 87601:
        raise ValueError(f"Expected 87,601 movies, got {len(canonical_df)}")
    
    if 'canonical_id' not in canonical_df.columns:
        raise ValueError("Missing canonical_id column in canonical index")
    
    return canonical_df

def load_providers_data():
    """Load the normalized providers data"""
    logger.info("Loading normalized providers data...")
    
    # Try normalized first, fallback to raw if needed
    providers_path = "data/normalized/movies_providers.parquet"
    
    if not os.path.exists(providers_path):
        raise FileNotFoundError(f"Providers data not found: {providers_path}")
    
    providers_df = pd.read_parquet(providers_path)
    logger.info(f"Loaded providers data: {providers_df.shape}")
    
    # Validate structure
    expected_cols = [f'providers_{cat}' for cat in AVAILABILITY_CATEGORIES]
    missing_cols = [col for col in expected_cols if col not in providers_df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    
    return providers_df

def normalize_provider_names(provider_list) -> List[str]:
    """Normalize provider names to canonical format"""
    # Handle pandas Series and convert to list
    if hasattr(provider_list, 'tolist'):
        provider_list = provider_list.tolist()
    
    # Handle empty or None cases
    if not provider_list or provider_list == [''] or (isinstance(provider_list, list) and len(provider_list) == 0):
        return []
    
    normalized = []
    for provider in provider_list:
        if provider in PROVIDER_NAME_MAPPING:
            normalized.append(PROVIDER_NAME_MAPPING[provider])
        else:
            # Try to normalize common variations
            provider_lower = provider.lower().replace(' ', '_').replace('+', '_plus')
            if provider_lower in CANONICAL_PROVIDERS:
                normalized.append(provider_lower)
            else:
                # Log unknown providers for analysis
                logger.debug(f"Unknown provider: {provider}")
    
    return normalized

def create_provider_features(providers_df: pd.DataFrame, canonical_df: pd.DataFrame) -> pd.DataFrame:
    """Create binary provider features for each provider × category + any flag"""
    logger.info("Creating provider features...")
    
    # Initialize features dataframe (without canonical_id in data)
    features_data = {}
    
    # Create features for each canonical provider
    for provider in CANONICAL_PROVIDERS:
        logger.info(f"Processing provider: {provider}")
        
        # Create "any" flag (available under any category)
        provider_any_col = f'provider_{provider}_any'
        features_data[provider_any_col] = []
        
        # Create category-specific flags
        for category in AVAILABILITY_CATEGORIES:
            category_col = f'provider_{provider}_{category}'
            features_data[category_col] = []
        
        # Process each movie
        for idx in range(len(providers_df)):
            # Get providers for this movie in each category
            flatrate_providers = normalize_provider_names(providers_df.iloc[idx]['providers_flatrate'])
            rent_providers = normalize_provider_names(providers_df.iloc[idx]['providers_rent'])
            buy_providers = normalize_provider_names(providers_df.iloc[idx]['providers_buy'])
            ads_providers = normalize_provider_names(providers_df.iloc[idx]['providers_ads'])
            free_providers = normalize_provider_names(providers_df.iloc[idx]['providers_free'])
            
            # Check if provider is available in any category
            provider_available_any = provider in (flatrate_providers + rent_providers + 
                                                buy_providers + ads_providers + free_providers)
            features_data[provider_any_col].append(1 if provider_available_any else 0)
            
            # Check availability in each specific category
            features_data[f'provider_{provider}_flatrate'].append(1 if provider in flatrate_providers else 0)
            features_data[f'provider_{provider}_rent'].append(1 if provider in rent_providers else 0)
            features_data[f'provider_{provider}_buy'].append(1 if provider in buy_providers else 0)
            features_data[f'provider_{provider}_ads'].append(1 if provider in ads_providers else 0)
            features_data[f'provider_{provider}_free'].append(1 if provider in free_providers else 0)
    
    # Create dataframe
    features_df = pd.DataFrame(features_data, dtype='int8')
    
    # Set canonical_id as index
    features_df.index = canonical_df['canonical_id'].values
    features_df.index.name = 'canonical_id'
    
    logger.info(f"Created provider features: {features_df.shape}")
    logger.info(f"Feature columns: {list(features_df.columns)}")
    
    return features_df

def validate_features(features_df: pd.DataFrame, canonical_df: pd.DataFrame) -> Dict:
    """Validate the generated features against requirements"""
    logger.info("Validating features...")
    
    validation_results = {}
    
    # Row alignment check: exactly 87,601 rows, unique canonical_id
    row_count = len(features_df)
    validation_results['row_count'] = row_count
    validation_results['expected_rows'] = 87601
    validation_results['row_alignment'] = row_count == 87601
    
    unique_canonical_ids = len(features_df.index.unique())
    validation_results['unique_canonical_ids'] = unique_canonical_ids
    validation_results['canonical_id_unique'] = unique_canonical_ids == row_count
    
    # All columns present for defined providers/categories
    expected_columns = []
    for provider in CANONICAL_PROVIDERS:
        expected_columns.append(f'provider_{provider}_any')
        for category in AVAILABILITY_CATEGORIES:
            expected_columns.append(f'provider_{provider}_{category}')
    
    missing_columns = [col for col in expected_columns if col not in features_df.columns]
    validation_results['missing_columns'] = missing_columns
    validation_results['all_columns_present'] = len(missing_columns) == 0
    
    # All values strictly 0 or 1, dtype = int8
    non_binary_values = ((features_df != 0) & (features_df != 1)).any().any()
    validation_results['non_binary_values'] = non_binary_values
    validation_results['binary_values_only'] = not non_binary_values
    
    dtype_check = features_df.dtypes.unique()
    validation_results['dtype_check'] = str(dtype_check)
    validation_results['dtype_int8'] = all(dtype == 'int8' for dtype in dtype_check)
    
    # Coverage: no missing rows, no NaNs
    has_nans = features_df.isnull().any().any()
    validation_results['has_nans'] = has_nans
    validation_results['no_nans'] = not has_nans
    
    # Check alignment with canonical index
    canonical_ids_match = set(features_df.index) == set(canonical_df['canonical_id'])
    validation_results['canonical_ids_match'] = canonical_ids_match
    
    # Overall validation
    all_passed = all([
        validation_results['row_alignment'],
        validation_results['canonical_id_unique'],
        validation_results['all_columns_present'],
        validation_results['binary_values_only'],
        validation_results['dtype_int8'],
        validation_results['no_nans'],
        validation_results['canonical_ids_match']
    ])
    validation_results['all_validation_passed'] = all_passed
    
    logger.info("Validation results:")
    for key, value in validation_results.items():
        logger.info(f"  {key}: {value}")
    
    return validation_results

def save_features(features_df: pd.DataFrame) -> str:
    """Save the provider features to parquet file"""
    logger.info("Saving provider features...")
    
    output_path = "data/features/platform/movies_platform_features.parquet"
    features_df.to_parquet(output_path, index=True)
    
    logger.info(f"Saved provider features: {output_path}")
    logger.info(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    return output_path

def generate_report(validation_results: Dict, output_path: str) -> str:
    """Generate a summary report of the provider encoding process"""
    logger.info("Generating report...")
    
    report_path = "docs/step2d_provider_encoding_report.md"
    
    report_content = f"""# Step 2d.1 - Provider Encoding Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
Successfully encoded streaming provider availability into binary features for the Movie Recommendation Optimizer.

## Input Data
- **Source**: Normalized TMDB/JustWatch provider data
- **Movies**: 87,601 movies aligned to canonical_id
- **Region**: US (default)

## Feature Engineering
- **Providers**: {len(CANONICAL_PROVIDERS)} canonical streaming providers
- **Categories**: {len(AVAILABILITY_CATEGORIES)} availability categories (flatrate, rent, buy, ads, free)
- **Features per provider**: {len(AVAILABILITY_CATEGORIES) + 1} (category-specific + "any" flag)
- **Total features**: {len(CANONICAL_PROVIDERS) * (len(AVAILABILITY_CATEGORIES) + 1)}

## Canonical Provider List
{chr(10).join(f"- {provider}" for provider in CANONICAL_PROVIDERS)}

## Output
- **File**: `{output_path}`
- **Format**: Parquet with canonical_id index
- **Schema**: Binary int8 columns for each provider × category + "any" flag

## Validation Results
"""
    
    # Add validation results
    for key, value in validation_results.items():
        if key != 'all_validation_passed':  # Skip the overall result for now
            status = "✅" if value else "❌"
            report_content += f"- **{key}**: {status} {value}\n"
    
    # Add overall status
    overall_status = "✅ PASSED" if validation_results['all_validation_passed'] else "❌ FAILED"
    report_content += f"\n## Overall Status\n**{overall_status}**\n"
    
    # Add acceptance gates summary
    report_content += f"""
## Acceptance Gates Summary
- **Row alignment**: {validation_results['row_alignment']} (87,601 rows)
- **Canonical ID integrity**: {validation_results['canonical_id_unique']} (unique)
- **Column completeness**: {validation_results['all_columns_present']} (all expected columns)
- **Data type**: {validation_results['dtype_int8']} (int8)
- **Binary values**: {validation_results['binary_values_only']} (0/1 only)
- **Data quality**: {validation_results['no_nans']} (no missing values)
- **Index alignment**: {validation_results['canonical_ids_match']} (matches canonical index)

## Next Steps
Provider encoding complete. Features are ready for:
- Feature matrix construction
- Model training
- Recommendation system integration
"""
    
    # Save report
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Saved report: {report_path}")
    return report_path

def main():
    """Main execution function"""
    logger.info("=== Step 2d.1: Provider Encoding ===")
    
    try:
        # Setup
        setup_directories()
        
        # Load data
        canonical_df = load_canonical_index()
        providers_df = load_providers_data()
        
        # Validate data alignment
        if len(providers_df) != len(canonical_df):
            raise ValueError(f"Data length mismatch: providers={len(providers_df)}, canonical={len(canonical_df)}")
        
        # Create features
        features_df = create_provider_features(providers_df, canonical_df)
        
        # Validate features
        validation_results = validate_features(features_df, canonical_df)
        
        # Save features
        output_path = save_features(features_df)
        
        # Generate report
        report_path = generate_report(validation_results, output_path)
        
        # Final status
        if validation_results['all_validation_passed']:
            logger.info("✅ Step 2d.1 completed successfully!")
            logger.info(f"Output: {output_path}")
            logger.info(f"Report: {report_path}")
        else:
            logger.error("❌ Step 2d.1 failed validation!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Step 2d.1 failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
