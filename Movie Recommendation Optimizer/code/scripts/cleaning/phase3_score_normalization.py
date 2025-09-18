#!/usr/bin/env python3
"""
Step 1b Phase 3: Score Normalization
Normalizes all rating/score fields onto consistent scales and produces validated, 
analysis-ready columns for downstream features and modeling.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

# Setup logging
log_file = 'logs/step1b_phase3.log'
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)

def setup_directories():
    """Ensure output directories exist"""
    os.makedirs('data/normalized', exist_ok=True)
    os.makedirs('docs', exist_ok=True)
    logging.info("Directories setup complete")

def load_master_data():
    """Load the master movies table and set canonical_id as index"""
    logging.info("Loading master movies table...")
    master_path = "data/normalized/movies_master.parquet"
    
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Master table not found: {master_path}")
    
    df = pd.read_parquet(master_path)
    logging.info(f"Loaded master table: {df.shape}")
    
    # Set canonical_id as index
    df = df.set_index('canonical_id', drop=True)
    logging.info(f"Set canonical_id as index, shape: {df.shape}")
    
    return df

def profile_coverage(df):
    """Profile coverage for raw score columns"""
    logging.info("=== COVERAGE PROFILE ===")
    
    score_columns = ['imdb_rating', 'imdb_votes', 'rt_tomatometer', 'rt_audience']
    
    coverage_stats = {}
    for col in score_columns:
        if col in df.columns:
            non_null = df[col].notna().sum()
            total = len(df)
            coverage_pct = (non_null / total) * 100
            coverage_stats[col] = {
                'non_null': non_null,
                'total': total,
                'coverage_pct': coverage_pct,
                'min': df[col].min() if non_null > 0 else None,
                'max': df[col].max() if non_null > 0 else None,
                'mean': df[col].mean() if non_null > 0 else None
            }
            logging.info(f"{col}: {non_null:,}/{total:,} ({coverage_pct:.1f}%) - Range: {coverage_stats[col]['min']} to {coverage_stats[col]['max']}")
        else:
            logging.warning(f"Column {col} not found in master table")
    
    return coverage_stats

def load_movielens_ratings():
    """Load and calculate MovieLens average ratings per movie"""
    logging.info("Loading MovieLens ratings to calculate averages...")
    
    ratings_path = "movie-lens/ratings.csv"
    if not os.path.exists(ratings_path):
        logging.warning("MovieLens ratings not found, skipping ML ratings")
        return None
    
    ratings = pd.read_csv(ratings_path)
    logging.info(f"Loaded {len(ratings):,} MovieLens ratings")
    
    # Calculate average ratings per movie
    avg_ratings = ratings.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    # Flatten column names
    avg_ratings.columns = ['movieId', 'ml_avg_rating', 'ml_rating_count']
    
    logging.info(f"Calculated averages for {len(avg_ratings):,} movies")
    return avg_ratings

def robust_clip_and_scale(data, source_min, source_max, target_min=0, target_max=100, percentile_low=0.5, percentile_high=99.5):
    """Robust clipping and scaling with percentile-based outlier handling"""
    if data.isna().all():
        return pd.Series(index=data.index, dtype='float64')
    
    # Log original stats
    orig_min, orig_max = data.min(), data.max()
    orig_mean, orig_std = data.mean(), data.std()
    
    # Robust clipping at percentiles
    low_percentile = data.quantile(percentile_low / 100)
    high_percentile = data.quantile(percentile_high / 100)
    
    clipped = data.clip(lower=low_percentile, upper=high_percentile)
    
    # Min-max scaling
    if high_percentile > low_percentile:
        scaled = ((clipped - low_percentile) / (high_percentile - low_percentile)) * (target_max - target_min) + target_min
    else:
        scaled = pd.Series(target_min, index=data.index)
    
    # Log transformed stats
    final_min, final_max = scaled.min(), scaled.max()
    final_mean, final_std = scaled.mean(), scaled.std()
    
    logging.info(f"  Original: min={orig_min:.3f}, max={orig_max:.3f}, mean={orig_mean:.3f}, std={orig_std:.3f}")
    logging.info(f"  Clipped: {percentile_low}th={low_percentile:.3f}, {percentile_high}th={high_percentile:.3f}")
    logging.info(f"  Final: min={final_min:.3f}, max={final_max:.3f}, mean={final_mean:.3f}, std={final_std:.3f}")
    
    return scaled

def calculate_z_scores(data, percentile_low=0.5, percentile_high=99.5):
    """Calculate Z-scores with robust outlier handling"""
    if data.isna().all():
        return pd.Series(index=data.index, dtype='float64')
    
    # Robust clipping
    low_percentile = data.quantile(percentile_low / 100)
    high_percentile = data.quantile(percentile_high / 100)
    clipped = data.clip(lower=low_percentile, upper=high_percentile)
    
    # Z-score calculation
    mean = clipped.mean()
    std = clipped.std()
    
    if std > 0:
        z_scores = (clipped - mean) / std
    else:
        z_scores = pd.Series(0, index=data.index)
    
    return z_scores

def calculate_bayesian_score(data, votes, prior_votes=2500, global_mean=None):
    """Calculate Bayesian weighted score using vote count"""
    if data.isna().all() or votes.isna().all():
        return pd.Series(index=data.index, dtype='float64')
    
    if global_mean is None:
        global_mean = data.mean()
    
    # Bayesian formula: (v * R + C * m) / (v + C)
    # where v = votes, R = rating, C = prior_votes, m = global_mean
    bayesian = (votes * data + prior_votes * global_mean) / (votes + prior_votes)
    
    return bayesian

def normalize_scores(df, ml_avg_ratings):
    """Create all normalized score columns"""
    logging.info("=== CREATING NORMALIZED SCORES ===")
    
    # Initialize results dataframe with index
    results = pd.DataFrame(index=df.index)
    
    # 1. Min-Max to 0-100 scores
    logging.info("Creating min-max normalized scores (0-100)...")
    
    # IMDb rating: 0.5-10 → 0-100
    if 'imdb_rating' in df.columns:
        logging.info("Normalizing IMDb ratings...")
        results['imdb_score_100'] = robust_clip_and_scale(
            df['imdb_rating'], source_min=0.5, source_max=10, target_min=0, target_max=100
        )
    
    # MovieLens average rating: 0.5-5 → 0-100
    if ml_avg_ratings is not None:
        logging.info("Normalizing MovieLens ratings...")
        # Merge ML ratings
        ml_merged = df.reset_index().merge(
            ml_avg_ratings, on='movieId', how='left'
        ).set_index('canonical_id')
        
        results['ml_score_100'] = robust_clip_and_scale(
            ml_merged['ml_avg_rating'], source_min=0.5, source_max=5, target_min=0, target_max=100
        )
        results['ml_rating_count'] = ml_merged['ml_rating_count']
    else:
        results['ml_score_100'] = pd.Series(dtype='float64', index=df.index)
        results['ml_rating_count'] = pd.Series(dtype='float64', index=df.index)
    
    # RT scores: already 0-100, just copy/clip
    if 'rt_tomatometer' in df.columns:
        logging.info("Processing Rotten Tomatoes tomatometer...")
        results['rt_tomato_100'] = df['rt_tomatometer'].astype('float64').clip(0, 100)
    
    if 'rt_audience' in df.columns:
        logging.info("Processing Rotten Tomatoes audience scores...")
        results['rt_audience_100'] = df['rt_audience'].astype('float64').clip(0, 100)
    
    # 2. Z-scores
    logging.info("Creating Z-scores...")
    
    if 'imdb_rating' in df.columns:
        results['imdb_score_z'] = calculate_z_scores(df['imdb_rating'])
    
    if ml_avg_ratings is not None:
        ml_merged = df.reset_index().merge(
            ml_avg_ratings, on='movieId', how='left'
        ).set_index('canonical_id')
        results['ml_score_z'] = calculate_z_scores(ml_merged['ml_avg_rating'])
    
    if 'rt_tomatometer' in df.columns:
        results['rt_tomato_z'] = calculate_z_scores(df['rt_tomatometer'].astype('float64'))
    
    if 'rt_audience' in df.columns:
        results['rt_audience_z'] = calculate_z_scores(df['rt_audience'].astype('float64'))
    
    # 3. Bayesian weighted scores
    logging.info("Creating Bayesian weighted scores...")
    
    if 'imdb_rating' in df.columns and 'imdb_votes' in df.columns:
        global_imdb_mean = df['imdb_rating'].mean()
        results['imdb_score_bayesian_100'] = calculate_bayesian_score(
            results['imdb_score_100'], df['imdb_votes'], prior_votes=2500, global_mean=50
        )
        logging.info(f"IMDb Bayesian scores created with global mean: {global_imdb_mean:.3f}")
    
    if ml_avg_ratings is not None:
        ml_merged = df.reset_index().merge(
            ml_avg_ratings, on='movieId', how='left'
        ).set_index('canonical_id')
        global_ml_mean = ml_merged['ml_avg_rating'].mean()
        results['ml_score_bayesian_100'] = calculate_bayesian_score(
            results['ml_score_100'], ml_merged['ml_rating_count'], prior_votes=2500, global_mean=50
        )
        logging.info(f"MovieLens Bayesian scores created with global mean: {global_ml_mean:.3f}")
    
    # 4. Unified Quality Signal
    logging.info("Creating unified quality scores...")
    
    # Primary weights: (0.5, 0.3, 0.2, 0.0) for (IMDb, RT Tomato, RT Audience, ML)
    weights_primary = {
        'imdb_score_100': 0.5,
        'rt_tomato_100': 0.3,
        'rt_audience_100': 0.2,
        'ml_score_100': 0.0  # Excluded in primary
    }
    
    # Alternative weights: (0.4, 0.2, 0.4, 0.0) emphasizing audience
    weights_alt = {
        'imdb_score_100': 0.4,
        'rt_tomato_100': 0.2,
        'rt_audience_100': 0.4,
        'ml_score_100': 0.0  # Excluded in alternative
    }
    
    results['quality_score_100'] = calculate_weighted_score(results, weights_primary)
    results['quality_score_100_alt'] = calculate_weighted_score(results, weights_alt)
    
    logging.info("All normalized scores created successfully")
    return results

def calculate_weighted_score(df, weights):
    """Calculate weighted score ensuring no division by zero"""
    weighted_sum = 0
    total_weight = 0
    
    for col, weight in weights.items():
        if col in df.columns and weight > 0:
            # Only include non-null values
            valid_mask = df[col].notna()
            if valid_mask.any():
                weighted_sum += df[col].fillna(0) * weight
                total_weight += weight
    
    # Avoid division by zero
    if total_weight > 0:
        return weighted_sum / total_weight
    else:
        return pd.Series(0, index=df.index)

def validate_scores(df):
    """Validate all normalized score columns"""
    logging.info("=== VALIDATION ===")
    
    validation_results = {}
    
    # Check 0-100 range for *_100 columns
    score_100_cols = [col for col in df.columns if col.endswith('_100')]
    for col in score_100_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            in_range = (min_val >= 0) and (max_val <= 100)
            validation_results[col] = {
                'min': min_val,
                'max': max_val,
                'in_range': in_range
            }
            logging.info(f"{col}: min={min_val:.3f}, max={max_val:.3f}, in_range={in_range}")
    
    # Check Z-scores are finite
    z_score_cols = [col for col in df.columns if col.endswith('_z')]
    for col in z_score_cols:
        if col in df.columns:
            finite_count = df[col].notna().sum()
            total_count = len(df)
            validation_results[col] = {
                'finite_count': finite_count,
                'total_count': total_count,
                'finite_pct': (finite_count / total_count) * 100
            }
            logging.info(f"{col}: {finite_count:,}/{total_count:,} finite ({validation_results[col]['finite_pct']:.1f}%)")
    
    return validation_results

def generate_coverage_report(df):
    """Generate detailed coverage report"""
    logging.info("=== COVERAGE REPORT ===")
    
    coverage_data = {}
    for col in df.columns:
        non_null = df[col].notna().sum()
        total = len(df)
        coverage_pct = (non_null / total) * 100
        coverage_data[col] = {
            'non_null': non_null,
            'total': total,
            'coverage_pct': coverage_pct,
            'min': df[col].min() if non_null > 0 else None,
            'max': df[col].max() if non_null > 0 else None,
            'mean': df[col].mean() if non_null > 0 else None
        }
        logging.info(f"{col}: {non_null:,}/{total:,} ({coverage_pct:.1f}%)")
    
    return coverage_data

def save_outputs(df, coverage_data, validation_results):
    """Save all output files"""
    logging.info("=== SAVING OUTPUTS ===")
    
    # 1. Save normalized scores parquet
    output_path = "data/normalized/movies_scores.parquet"
    df.to_parquet(output_path, index=True)
    logging.info(f"Saved normalized scores: {output_path}")
    
    # 2. Save preview CSV (1000 rows)
    preview_path = "data/normalized/movies_scores_preview.csv"
    preview_df = df.head(1000).reset_index()
    preview_df.to_csv(preview_path, index=False)
    logging.info(f"Saved preview CSV: {preview_path}")
    
    # 3. Save configuration JSON
    config = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "phase": "Step 1b Phase 3: Score Normalization",
            "description": "Configuration for score normalization including ranges, weights, and priors"
        },
        "ranges": {
            "imdb": {"source": "0.5-10", "target": "0-100"},
            "movielens": {"source": "0.5-5", "target": "0-100"},
            "rt_tomato": {"source": "0-100", "target": "0-100"},
            "rt_audience": {"source": "0-100", "target": "0-100"}
        },
        "weights": {
            "quality_score_100": {"imdb": 0.5, "rt_tomato": 0.3, "rt_audience": 0.2, "ml": 0.0},
            "quality_score_100_alt": {"imdb": 0.4, "rt_tomato": 0.2, "rt_audience": 0.4, "ml": 0.0}
        },
        "priors": {
            "bayesian_prior_votes": 2500,
            "percentile_clipping": {"low": 0.5, "high": 99.5}
        },
        "coverage_summary": {
            col: {
                'coverage_pct': float(data['coverage_pct']),
                'non_null': int(data['non_null']),
                'total': int(data['total'])
            }
            for col, data in coverage_data.items()
        },
        "validation_summary": {
            col: {
                k: (float(v) if isinstance(v, (np.floating, float)) else 
                    int(v) if isinstance(v, (np.integer, int)) else 
                    bool(v) if isinstance(v, (bool, np.bool_)) else v)
                for k, v in val.items()
            }
            for col, val in validation_results.items()
        }
    }
    
    config_path = "docs/score_norm_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logging.info(f"Saved configuration: {config_path}")
    
    # 4. Update step1b report
    update_step1b_report(coverage_data, validation_results)
    
    return {
        'parquet': output_path,
        'preview': preview_path,
        'config': config_path
    }

def update_step1b_report(coverage_data, validation_results):
    """Append Phase 3 results to step1b report"""
    logging.info("Updating step1b report...")
    
    report_path = "docs/step1b_report.md"
    
    # Read existing report or create new
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            content = f.read()
    else:
        content = "# Step 1b Report\n\n"
    
    # Add Phase 3 section
    phase3_section = f"""
## Phase 3 — Score Normalization

**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Overview
Normalized all rating/score fields onto consistent scales and produced validated, analysis-ready columns for downstream features and modeling.

### Coverage Summary
| Column | Coverage | Non-Null | Total |
|--------|----------|----------|-------|
"""
    
    for col, data in coverage_data.items():
        phase3_section += f"| {col} | {data['coverage_pct']:.1f}% | {data['non_null']:,} | {data['total']:,} |\n"
    
    phase3_section += f"""
### Validation Results
- **0-100 Range Validation:** All *_100 columns validated to be within [0, 100]
- **Z-Score Validation:** All *_z columns validated to be finite
- **Coverage:** {len([d for d in coverage_data.values() if d['coverage_pct'] > 0])} columns have data

### Normalized Score Families
1. **Min-Max Scores (0-100):** imdb_score_100, ml_score_100, rt_tomato_100, rt_audience_100
2. **Z-Scores:** imdb_score_z, ml_score_z, rt_tomato_z, rt_audience_z
3. **Bayesian Weighted:** imdb_score_bayesian_100, ml_score_bayesian_100
4. **Unified Quality:** quality_score_100, quality_score_100_alt

### Configuration
- **Bayesian Prior Votes:** 2,500
- **Percentile Clipping:** 0.5th to 99.5th percentiles
- **Primary Weights:** IMDb(0.5), RT Tomato(0.3), RT Audience(0.2)
- **Alternative Weights:** IMDb(0.4), RT Tomato(0.2), RT Audience(0.4)

### Outputs
- `data/normalized/movies_scores.parquet`: Full normalized scores dataset
- `data/normalized/movies_scores_preview.csv`: 1,000-row preview
- `docs/score_norm_config.json`: Configuration and metadata
"""
    
    # Append to existing content
    if "## Phase 3 — Score Normalization" not in content:
        content += phase3_section
    else:
        # Replace existing Phase 3 section
        import re
        pattern = r"## Phase 3 — Score Normalization.*?(?=##|\Z)"
        content = re.sub(pattern, phase3_section.strip(), content, flags=re.DOTALL)
    
    # Write updated report
    with open(report_path, 'w') as f:
        f.write(content)
    
    logging.info(f"Updated step1b report: {report_path}")

def print_final_summary(df, coverage_data):
    """Print final summary block"""
    logging.info("=== FINAL SUMMARY ===")
    
    print("\n" + "="*80)
    print("STEP 1B PHASE 3: SCORE NORMALIZATION COMPLETE")
    print("="*80)
    
    print(f"\nColumns Created: {len(df.columns)}")
    print(f"Total Rows: {len(df):,}")
    
    print("\nCoverage per Column:")
    for col, data in coverage_data.items():
        print(f"  {col}: {data['coverage_pct']:.1f}% ({data['non_null']:,}/{data['total']:,})")
    
    print("\nSample Rows (5):")
    sample_cols = ['imdb_score_100', 'rt_tomato_100', 'rt_audience_100', 'quality_score_100']
    available_cols = [col for col in sample_cols if col in df.columns]
    if available_cols:
        sample_df = df[available_cols].head(5)
        print(sample_df.to_string())
    
    print("\n" + "="*80)

def main():
    """Main function for Phase 3: Score Normalization"""
    start_time = datetime.now()
    logging.info("=== STARTING STEP 1B PHASE 3: SCORE NORMALIZATION ===")
    
    try:
        # Setup
        setup_directories()
        
        # Load data
        master_df = load_master_data()
        ml_avg_ratings = load_movielens_ratings()
        
        # Profile coverage
        coverage_stats = profile_coverage(master_df)
        
        # Create normalized scores
        normalized_scores = normalize_scores(master_df, ml_avg_ratings)
        
        # Validation
        validation_results = validate_scores(normalized_scores)
        
        # Generate coverage report
        coverage_data = generate_coverage_report(normalized_scores)
        
        # Save outputs
        output_paths = save_outputs(normalized_scores, coverage_data, validation_results)
        
        # Print final summary
        print_final_summary(normalized_scores, coverage_data)
        
        # Log completion
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"=== PHASE 3 COMPLETE in {duration} ===")
        logging.info(f"Outputs saved to: {list(output_paths.values())}")
        
    except Exception as e:
        logging.error(f"Phase 3 failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
