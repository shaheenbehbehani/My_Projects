#!/usr/bin/env python3
"""
Step 2d.2: Index & QA for Platform Features
Movie Recommendation Optimizer - Platform Features Validation

Validates and documents the platform features generated in Step 2d.1.
Performs structural QA, coverage analysis, and generates visualizations.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
log_file = 'logs/step2d_phase2.log'
os.makedirs('logs', exist_ok=True)
os.makedirs('docs/img', exist_ok=True)
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

# Expected configuration
EXPECTED_ROWS = 87601
EXPECTED_COLUMNS = 102
CANONICAL_PROVIDERS = [
    'netflix', 'max', 'hulu', 'prime', 'disney_plus', 'paramount_plus', 
    'apple_tv_plus', 'peacock', 'tubi', 'roku', 'youtube', 'google_play', 
    'itunes', 'vudu', 'starz', 'showtime', 'amc_plus'
]
AVAILABILITY_CATEGORIES = ['any', 'flatrate', 'rent', 'buy', 'ads', 'free']

# Matplotlib settings for high-quality output
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

def setup_directories():
    """Ensure required directories exist"""
    os.makedirs('docs/img', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/features/platform', exist_ok=True)
    logger.info("Directories setup complete")

def load_platform_features():
    """Load the platform features for validation"""
    logger.info("Loading platform features...")
    
    input_path = "data/features/platform/movies_platform_features.parquet"
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Platform features not found: {input_path}")
    
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded platform features: {df.shape}")
    
    return df

def structural_qa(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform structural QA checks"""
    logger.info("Performing structural QA...")
    
    qa_results = {}
    
    # File existence and readability
    qa_results['file_exists'] = True
    qa_results['file_readable'] = True
    
    # Basic structure
    qa_results['row_count'] = len(df)
    qa_results['column_count'] = len(df.columns)
    qa_results['index_name'] = df.index.name
    
    # Row alignment check
    qa_results['row_alignment'] = len(df) == EXPECTED_ROWS
    qa_results['expected_rows'] = EXPECTED_ROWS
    qa_results['actual_rows'] = len(df)
    
    # Index integrity
    unique_canonical_ids = len(df.index.unique())
    qa_results['unique_canonical_ids'] = unique_canonical_ids
    qa_results['canonical_id_unique'] = unique_canonical_ids == len(df)
    qa_results['canonical_id_alignment'] = unique_canonical_ids == EXPECTED_ROWS
    
    # Column completeness
    expected_columns = []
    for provider in CANONICAL_PROVIDERS:
        for category in AVAILABILITY_CATEGORIES:
            expected_columns.append(f'provider_{provider}_{category}')
    
    actual_columns = set(df.columns)
    missing_columns = [col for col in expected_columns if col not in actual_columns]
    extra_columns = [col for col in actual_columns if col not in expected_columns]
    
    qa_results['missing_columns'] = missing_columns
    qa_results['extra_columns'] = extra_columns
    qa_results['all_columns_present'] = len(missing_columns) == 0
    qa_results['no_extra_columns'] = len(extra_columns) == 0
    qa_results['column_completeness'] = len(missing_columns) == 0 and len(extra_columns) == 0
    
    # Data type checks
    non_int8_cols = [col for col in df.columns if df[col].dtype != 'int8']
    qa_results['non_int8_columns'] = non_int8_cols
    qa_results['all_int8'] = len(non_int8_cols) == 0
    
    # Binary value checks
    non_binary_mask = ((df != 0) & (df != 1)).any()
    non_binary_cols = non_binary_mask[non_binary_mask].index.tolist()
    qa_results['non_binary_columns'] = non_binary_cols
    qa_results['binary_values_only'] = len(non_binary_cols) == 0
    
    # NaN and Inf checks
    has_nans = df.isnull().any().any()
    has_infs = np.isinf(df.select_dtypes(include=[np.number])).any().any()
    qa_results['has_nans'] = has_nans
    qa_results['has_infs'] = has_infs
    qa_results['no_nans_or_infs'] = not has_nans and not has_infs
    
    # Overall structural QA
    qa_results['structural_qa_passed'] = all([
        qa_results['row_alignment'],
        qa_results['canonical_id_alignment'],
        qa_results['column_completeness'],
        qa_results['all_int8'],
        qa_results['binary_values_only'],
        qa_results['no_nans_or_infs']
    ])
    
    logger.info("Structural QA completed")
    return qa_results

def coverage_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze provider coverage and distribution"""
    logger.info("Performing coverage analysis...")
    
    # Get _any columns for coverage analysis
    any_columns = [col for col in df.columns if col.endswith('_any')]
    
    # Coverage by provider
    coverage_data = []
    for col in any_columns:
        provider_name = col.replace('provider_', '').replace('_any', '')
        count = df[col].sum()
        percentage = (count / len(df)) * 100
        coverage_data.append({
            'provider': provider_name,
            'count': count,
            'percentage': percentage
        })
    
    coverage_df = pd.DataFrame(coverage_data)
    coverage_df = coverage_df.sort_values('percentage', ascending=False).reset_index(drop=True)
    
    # Providers per movie distribution
    any_data = df[any_columns]
    providers_per_movie = any_data.sum(axis=1)
    
    # Create distribution buckets
    distribution_data = []
    for i in range(9):  # 0, 1, 2, 3, 4, 5, 6, 7, 8+
        if i == 8:
            count = (providers_per_movie >= 8).sum()
            label = '8+'
        else:
            count = (providers_per_movie == i).sum()
            label = str(i)
        
        percentage = (count / len(df)) * 100
        distribution_data.append({
            'providers': label,
            'count': count,
            'percentage': percentage
        })
    
    distribution_df = pd.DataFrame(distribution_data)
    
    logger.info("Coverage analysis completed")
    return coverage_df, distribution_df

def create_visualizations(df: pd.DataFrame, coverage_df: pd.DataFrame, distribution_df: pd.DataFrame):
    """Create all required visualizations"""
    logger.info("Creating visualizations...")
    
    # Set style
    sns.set_palette("husl")
    
    # Chart 1: Availability by provider (_any only)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(coverage_df)), coverage_df['percentage'], 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    plt.title('Provider Availability by Percentage', fontsize=14, fontweight='bold')
    plt.xlabel('Providers', fontsize=12)
    plt.ylabel('Availability (%)', fontsize=12)
    plt.xticks(range(len(coverage_df)), coverage_df['provider'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars (only for bars with height > 0.5%)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0.5:  # Only label bars with significant height
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('docs/img/step2d_bar_provider_any.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved provider availability chart")
    
    # Chart 2: Stacked bar across categories for top 10 providers
    top_10_providers = coverage_df.head(10)['provider'].tolist()
    
    # Get category data for top providers
    category_data = []
    for provider in top_10_providers:
        provider_data = {'provider': provider}
        for category in ['flatrate', 'ads', 'free', 'rent', 'buy']:
            col = f'provider_{provider}_{category}'
            if col in df.columns:
                count = df[col].sum()
                percentage = (count / len(df)) * 100
                provider_data[category] = percentage
            else:
                provider_data[category] = 0
        category_data.append(provider_data)
    
    category_df = pd.DataFrame(category_data)
    
    plt.figure(figsize=(14, 7))
    
    # Create stacked bars
    categories = ['flatrate', 'ads', 'free', 'rent', 'buy']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    bottom = np.zeros(len(top_10_providers))
    for i, (category, color) in enumerate(zip(categories, colors)):
        values = category_df[category].values
        plt.bar(range(len(top_10_providers)), values, bottom=bottom, 
               label=category, color=color, alpha=0.8)
        bottom += values
    
    plt.title('Provider Availability by Category (Top 10 Providers)', fontsize=14, fontweight='bold')
    plt.xlabel('Providers', fontsize=12)
    plt.ylabel('Availability (%)', fontsize=12)
    plt.xticks(range(len(top_10_providers)), top_10_providers, rotation=45, ha='right')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/img/step2d_bar_provider_categories.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved provider categories chart")
    
    # Chart 3: Distribution of providers per movie
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(distribution_df)), distribution_df['count'], 
                   color='lightcoral', edgecolor='darkred', alpha=0.7)
    
    plt.title('Distribution of Providers per Movie', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Providers', fontsize=12)
    plt.ylabel('Number of Movies', fontsize=12)
    plt.xticks(range(len(distribution_df)), distribution_df['providers'])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{height:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('docs/img/step2d_bar_providers_per_movie.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved providers per movie distribution chart")

def save_csv_outputs(coverage_df: pd.DataFrame, distribution_df: pd.DataFrame):
    """Save CSV outputs for coverage and distribution"""
    logger.info("Saving CSV outputs...")
    
    # Save coverage data
    coverage_path = "docs/categorical_platform_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    logger.info(f"Saved coverage data: {coverage_path}")
    
    # Save distribution data
    distribution_path = "docs/categorical_platform_providers_per_movie.csv"
    distribution_df.to_csv(distribution_path, index=False)
    logger.info(f"Saved distribution data: {distribution_path}")

def create_manifest(df: pd.DataFrame) -> Dict[str, Any]:
    """Create manifest file for platform features"""
    logger.info("Creating manifest...")
    
    manifest = {
        "path": "data/features/platform/movies_platform_features.parquet",
        "row_count": len(df),
        "column_count": len(df.columns),
        "providers": CANONICAL_PROVIDERS,
        "categories": AVAILABILITY_CATEGORIES,
        "dtype": "int8",
        "created_utc": datetime.utcnow().isoformat(),
        "schema_version": "2d.1",
        "description": "Binary platform features for movie streaming availability"
    }
    
    manifest_path = "data/features/platform/manifest_platform_features.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Saved manifest: {manifest_path}")
    return manifest

def generate_report(qa_results: Dict[str, Any], coverage_df: pd.DataFrame, 
                   distribution_df: pd.DataFrame, manifest: Dict[str, Any]) -> str:
    """Generate the comprehensive QA report"""
    logger.info("Generating report...")
    
    report_path = "docs/step2d_report.md"
    
    # Calculate file sizes
    input_size = os.path.getsize("data/features/platform/movies_platform_features.parquet") / 1024
    coverage_size = os.path.getsize("docs/categorical_platform_coverage.csv") / 1024
    distribution_size = os.path.getsize("docs/categorical_platform_providers_per_movie.csv") / 1024
    
    report_content = f"""# Step 2d.2 - Platform Features Index & QA Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
Comprehensive validation and documentation of platform features generated in Step 2d.1. This QA process ensures data integrity, completeness, and readiness for downstream modeling by validating structural requirements, analyzing coverage patterns, and generating visual insights.

## Structural QA Results

### ✅ Row & Index Validation
- **Row count**: {qa_results['actual_rows']:,} (expected: {qa_results['expected_rows']:,})
- **Index name**: `{qa_results['index_name']}`
- **Unique canonical_ids**: {qa_results['unique_canonical_ids']:,}
- **Row alignment**: {'✅ PASS' if qa_results['row_alignment'] else '❌ FAIL'}

### ✅ Column Validation
- **Total columns**: {qa_results['column_count']} (expected: {EXPECTED_COLUMNS})
- **Missing columns**: {len(qa_results['missing_columns'])}
- **Extra columns**: {len(qa_results['extra_columns'])}
- **Column completeness**: {'✅ PASS' if qa_results['column_completeness'] else '❌ FAIL'}

### ✅ Data Quality Validation
- **Data type**: {'✅ All int8' if qa_results['all_int8'] else '❌ Mixed types'}
- **Binary values**: {'✅ 0/1 only' if qa_results['binary_values_only'] else '❌ Non-binary values'}
- **Missing values**: {'✅ No NaNs' if not qa_results['has_nans'] else '❌ NaNs found'}
- **Infinite values**: {'✅ No Inf' if not qa_results['has_infs'] else '❌ Inf found'}

### 🎯 Overall Structural QA Status
**{'✅ PASSED' if qa_results['structural_qa_passed'] else '❌ FAILED'}**

## Coverage Analysis

### Top 10 Providers by Availability
| Rank | Provider | Movies Available | Percentage |
|------|----------|------------------|------------|
"""
    
    # Add top 10 providers table
    for i, row in coverage_df.head(10).iterrows():
        report_content += f"| {i+1} | {row['provider']} | {row['count']:,} | {row['percentage']:.2f}% |\n"
    
    report_content += f"""

### Coverage Summary
- **Total providers analyzed**: {len(coverage_df)}
- **Highest availability**: {coverage_df.iloc[0]['provider']} ({coverage_df.iloc[0]['percentage']:.2f}%)
- **Lowest availability**: {coverage_df.iloc[-1]['provider']} ({coverage_df.iloc[-1]['percentage']:.2f}%)

## Distribution Analysis

### Providers per Movie
| Providers | Movie Count | Percentage |
|-----------|-------------|------------|
"""
    
    # Add distribution table
    for _, row in distribution_df.iterrows():
        report_content += f"| {row['providers']} | {row['count']:,} | {row['percentage']:.2f}% |\n"
    
    report_content += f"""

### Distribution Insights
- **Most common**: {distribution_df.loc[distribution_df['count'].idxmax(), 'providers']} providers per movie
- **Movies with 0 providers**: {distribution_df.loc[distribution_df['providers'] == '0', 'count'].iloc[0]:,} ({distribution_df.loc[distribution_df['providers'] == '0', 'percentage'].iloc[0]:.2f}%)
- **Movies with 1+ providers**: {distribution_df.loc[distribution_df['providers'] != '0', 'count'].sum():,} ({(distribution_df.loc[distribution_df['providers'] != '0', 'percentage'].sum()):.2f}%)

## Visualizations

### Generated Charts
1. **Provider Availability**: `docs/img/step2d_bar_provider_any.png` - Bar chart showing availability percentages for all providers
2. **Category Breakdown**: `docs/img/step2d_bar_provider_categories.png` - Stacked bar chart showing availability across categories for top 10 providers
3. **Distribution**: `docs/img/step2d_bar_providers_per_movie.png` - Distribution of how many providers each movie has

## Output Files

### Data Files
- **Platform Features**: `data/features/platform/movies_platform_features.parquet` ({input_size:.1f} KB)
- **Manifest**: `data/features/platform/manifest_platform_features.json`

### Analysis Files
- **Coverage Data**: `docs/categorical_platform_coverage.csv` ({coverage_size:.1f} KB)
- **Distribution Data**: `docs/categorical_platform_providers_per_movie.csv` ({distribution_size:.1f} KB)

### Documentation
- **QA Report**: `docs/step2d_report.md` (this file)
- **Detailed Log**: `logs/step2d_phase2.log`

## Acceptance Gates Summary

| Gate | Status | Details |
|------|--------|---------|
| Row Alignment | {'✅ PASS' if qa_results['row_alignment'] else '❌ FAIL'} | {qa_results['actual_rows']:,} rows, {qa_results['unique_canonical_ids']:,} unique IDs |
| Column Completeness | {'✅ PASS' if qa_results['column_completeness'] else '❌ FAIL'} | {qa_results['column_count']} columns, {len(qa_results['missing_columns'])} missing |
| Data Types | {'✅ PASS' if qa_results['all_int8'] else '❌ FAIL'} | All feature columns int8 |
| Binary Values | {'✅ PASS' if qa_results['binary_values_only'] else '❌ FAIL'} | Values only 0 or 1 |
| Data Quality | {'✅ PASS' if qa_results['no_nans_or_infs'] else '❌ FAIL'} | No NaN or Inf values |
| Documentation | ✅ PASS | Report, logs, and visualizations generated |

## Conclusions

{'✅ **READY FOR DOWNSTREAM MODELING**' if qa_results['structural_qa_passed'] else '❌ **NOT READY - VALIDATION FAILED**'}

The platform features have been thoroughly validated and documented. All acceptance gates have passed, ensuring data integrity and completeness for the Movie Recommendation Optimizer.

### Key Strengths
- Perfect row alignment with canonical index (87,601 movies)
- Complete feature coverage (102 columns across 17 providers × 6 categories)
- High-quality binary encoding (int8, no missing values)
- Comprehensive documentation and visualizations

### Next Steps
Platform features are ready for:
- Feature matrix construction
- Model training and evaluation
- Recommendation system integration
- Performance analysis and optimization

---

**Generated by**: Step 2d.2 Index & QA Script  
**Input**: `{manifest['path']}`  
**Outputs**: {len([f for f in os.listdir('docs') if f.startswith('step2d')])} documentation files, {len([f for f in os.listdir('docs/img') if f.startswith('step2d')])} visualizations
"""
    
    # Save report
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Saved report: {report_path}")
    return report_path

def main():
    """Main execution function"""
    logger.info("=== Step 2d.2: Platform Features Index & QA ===")
    
    try:
        # Setup
        setup_directories()
        
        # Load data
        df = load_platform_features()
        
        # Structural QA
        qa_results = structural_qa(df)
        
        # Check if QA passed before proceeding
        if not qa_results['structural_qa_passed']:
            logger.error("❌ Structural QA failed! Cannot proceed with analysis.")
            logger.error("Failed checks:")
            for key, value in qa_results.items():
                if key.endswith('_passed') and not value:
                    logger.error(f"  - {key}: {value}")
            sys.exit(1)
        
        # Coverage analysis
        coverage_df, distribution_df = coverage_analysis(df)
        
        # Create visualizations
        create_visualizations(df, coverage_df, distribution_df)
        
        # Save CSV outputs
        save_csv_outputs(coverage_df, distribution_df)
        
        # Create manifest
        manifest = create_manifest(df)
        
        # Generate report
        report_path = generate_report(qa_results, coverage_df, distribution_df, manifest)
        
        # Final status
        logger.info("✅ Step 2d.2 completed successfully!")
        logger.info(f"Report: {report_path}")
        logger.info("All acceptance gates passed!")
        
    except Exception as e:
        logger.error(f"Step 2d.2 failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
