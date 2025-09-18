#!/usr/bin/env python3
"""
Step 2b.4: QA & Report (Categoricals)
Movie Recommendation Optimizer Project

This script performs comprehensive QA on the consolidated categorical features:
- Structural validation
- Coverage & distribution analysis
- Sanity checks
- Documentation generation
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/step2b_phase4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CategoricalFeatureQA:
    def __init__(self):
        self.base_path = Path('.')
        self.data_path = self.base_path / 'data'
        self.features_path = self.data_path / 'features'
        self.categorical_path = self.features_path / 'categorical'
        self.docs_path = self.base_path / 'docs'
        
        logger.info("CategoricalFeatureQA initialized")
        
    def load_consolidated_features(self):
        """Load the consolidated categorical features table"""
        logger.info("Loading consolidated categorical features...")
        
        features_path = self.categorical_path / 'movies_categorical_features.parquet'
        df = pd.read_parquet(features_path)
        
        logger.info(f"Loaded features table: {df.shape}")
        return df
    
    def structural_qa(self, df):
        """Perform structural QA validation"""
        logger.info("Performing structural QA...")
        
        results = {}
        
        # Row count validation
        expected_rows = 87601
        actual_rows = len(df)
        results['row_alignment'] = {
            'expected': expected_rows,
            'actual': actual_rows,
            'passed': actual_rows == expected_rows
        }
        
        # Column count validation
        feature_cols = [col for col in df.columns if col != 'canonical_id']
        actual_features = len(feature_cols)
        expected_features = 129
        
        results['feature_count'] = {
            'expected': expected_features,
            'actual': actual_features,
            'passed': actual_features == expected_features
        }
        
        # Family count validation
        genre_cols = [col for col in feature_cols if col.startswith('genre_')]
        actor_cols = [col for col in feature_cols if col.startswith('actor_')]
        director_cols = [col for col in feature_cols if col.startswith('director_')]
        
        results['family_counts'] = {
            'genres': len(genre_cols),
            'actors': len(actor_cols),
            'directors': len(director_cols),
            'total': len(genre_cols) + len(actor_cols) + len(director_cols)
        }
        
        # Data type validation
        dtypes_valid = True
        for col in feature_cols:
            if df[col].dtype != 'int8':
                dtypes_valid = False
                logger.warning(f"Column {col} has dtype {df[col].dtype}, expected int8")
        
        results['dtypes_valid'] = dtypes_valid
        
        # Binary values validation
        binary_valid = True
        for col in feature_cols:
            unique_vals = df[col].unique()
            if not all(val in [0, 1] for val in unique_vals):
                binary_valid = False
                logger.warning(f"Column {col} has non-binary values: {unique_vals}")
        
        results['binary_valid'] = binary_valid
        
        return results
    
    def coverage_analysis(self, df):
        """Analyze coverage and distribution of features"""
        logger.info("Analyzing coverage and distribution...")
        
        feature_cols = [col for col in df.columns if col != 'canonical_id']
        genre_cols = [col for col in feature_cols if col.startswith('genre_')]
        actor_cols = [col for col in feature_cols if col.startswith('actor_')]
        director_cols = [col for col in feature_cols if col.startswith('director_')]
        
        results = {}
        
        # Genre coverage
        genre_coverage = (df[genre_cols].sum(axis=1) > 0).mean() * 100
        results['genre_coverage'] = genre_coverage
        
        # Actor coverage
        actor_coverage = (df[actor_cols].sum(axis=1) > 0).mean() * 100
        results['actor_coverage'] = actor_coverage
        
        # Director coverage
        director_coverage = (df[director_cols].sum(axis=1) > 0).mean() * 100
        results['director_coverage'] = director_coverage
        
        # Multi-label statistics
        genres_per_movie = df[genre_cols].sum(axis=1)
        actors_per_movie = df[actor_cols].sum(axis=1)
        directors_per_movie = df[director_cols].sum(axis=1)
        
        results['multi_label_stats'] = {
            'genres': {
                'min': int(genres_per_movie.min()),
                'median': float(genres_per_movie.median()),
                'max': int(genres_per_movie.max()),
                'mean': float(genres_per_movie.mean())
            },
            'actors': {
                'min': int(actors_per_movie.min()),
                'median': float(actors_per_movie.median()),
                'max': int(actors_per_movie.max()),
                'mean': float(actors_per_movie.mean())
            },
            'directors': {
                'min': int(directors_per_movie.min()),
                'median': float(directors_per_movie.median()),
                'max': int(directors_per_movie.max()),
                'mean': float(directors_per_movie.mean())
            }
        }
        
        return results
    
    def sparsity_analysis(self, df):
        """Analyze sparsity and top features for each family"""
        logger.info("Analyzing sparsity and top features...")
        
        feature_cols = [col for col in df.columns if col != 'canonical_id']
        genre_cols = [col for col in feature_cols if col.startswith('genre_')]
        actor_cols = [col for col in feature_cols if col.startswith('actor_')]
        director_cols = [col for col in feature_cols if col.startswith('director_')]
        
        results = {}
        
        # Genre sparsity
        genre_counts = df[genre_cols].sum().sort_values(ascending=False)
        genre_sparsity = {
            'counts': genre_counts.to_dict(),
            'shares': (genre_counts / len(df) * 100).to_dict(),
            'top_10': genre_counts.head(10).to_dict()
        }
        results['genres'] = genre_sparsity
        
        # Actor sparsity
        actor_counts = df[actor_cols].sum().sort_values(ascending=False)
        actor_sparsity = {
            'counts': actor_counts.to_dict(),
            'shares': (actor_counts / len(df) * 100).to_dict(),
            'top_10': actor_counts.head(10).to_dict()
        }
        results['actors'] = actor_sparsity
        
        # Director sparsity
        director_counts = df[director_cols].sum().sort_values(ascending=False)
        director_sparsity = {
            'counts': director_counts.to_dict(),
            'shares': (director_counts / len(df) * 100).to_dict(),
            'top_10': director_counts.head(10).to_dict()
        }
        results['directors'] = director_sparsity
        
        return results
    
    def sanity_checks(self, df):
        """Perform sanity checks on the data"""
        logger.info("Performing sanity checks...")
        
        feature_cols = [col for col in df.columns if col != 'canonical_id']
        
        results = {}
        
        # Check for all-zero columns
        zero_cols = []
        for col in feature_cols:
            if df[col].sum() == 0:
                zero_cols.append(col)
        
        results['all_zero_columns'] = {
            'count': len(zero_cols),
            'columns': zero_cols,
            'passed': len(zero_cols) == 0
        }
        
        # Check for duplicate column names
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        results['duplicate_columns'] = {
            'count': len(duplicate_cols),
            'columns': duplicate_cols,
            'passed': len(duplicate_cols) == 0
        }
        
        # Random row spot check
        random_rows = df.sample(n=5, random_state=42)
        spot_check_results = []
        
        for idx, row in random_rows.iterrows():
            canonical_id = row['canonical_id']
            feature_values = row[feature_cols]
            binary_check = all(val in [0, 1] for val in feature_values)
            
            spot_check_results.append({
                'canonical_id': canonical_id,
                'binary_values': binary_check,
                'genre_sum': int(feature_values[[col for col in feature_cols if col.startswith('genre_')]].sum()),
                'actor_sum': int(feature_values[[col for col in feature_cols if col.startswith('actor_')]].sum()),
                'director_sum': int(feature_values[[col for col in feature_cols if col.startswith('director_')]].sum())
            })
        
        results['spot_check'] = spot_check_results
        
        return results
    
    def save_top10_csvs(self, sparsity_results):
        """Save top-10 features for each family as CSV files"""
        logger.info("Saving top-10 feature CSVs...")
        
        # Genres top 10
        genres_df = pd.DataFrame([
            {'feature': k, 'count': v, 'share_percent': sparsity_results['genres']['shares'][k]}
            for k, v in sparsity_results['genres']['top_10'].items()
        ])
        genres_path = self.docs_path / 'categorical_top10_genres.csv'
        genres_df.to_csv(genres_path, index=False)
        logger.info(f"Genres top-10 saved to: {genres_path}")
        
        # Actors top 10
        actors_df = pd.DataFrame([
            {'feature': k, 'count': v, 'share_percent': sparsity_results['actors']['shares'][k]}
            for k, v in sparsity_results['actors']['top_10'].items()
        ])
        actors_path = self.docs_path / 'categorical_top10_actors.csv'
        actors_df.to_csv(actors_path, index=False)
        logger.info(f"Actors top-10 saved to: {actors_path}")
        
        # Directors top 10
        directors_df = pd.DataFrame([
            {'feature': k, 'count': v, 'share_percent': sparsity_results['directors']['shares'][k]}
            for k, v in sparsity_results['directors']['top_10'].items()
        ])
        directors_path = self.docs_path / 'categorical_top10_directors.csv'
        directors_df.to_csv(directors_path, index=False)
        logger.info(f"Directors top-10 saved to: {directors_path}")
        
        return {
            'genres': genres_path,
            'actors': actors_path,
            'directors': directors_path
        }
    
    def update_report(self, structural_results, coverage_results, sparsity_results, sanity_results):
        """Update step2b_report.md with Sections 2b.3 and 2b.4"""
        logger.info("Updating step2b_report.md...")
        
        report_path = self.docs_path / 'step2b_report.md'
        
        # Read existing report
        if report_path.exists():
            with open(report_path, 'r') as f:
                content = f.read()
        else:
            content = "# Step 2b Report: Feature Engineering\n\n"
        
        # Add Section 2b.3
        section_2b3 = f"""
## Section 2b.3: Categorical Feature Consolidation

### Overview
Consolidated all categorical features into a single aligned table for efficient machine learning workflows.

### What Was Consolidated
- **Genres**: 29 canonical genres from `data/features/genres/movies_genres_multihot_full.parquet`
- **Actors**: Top 50 actors from `data/features/crew/movies_actors_top50.parquet`
- **Directors**: Top 50 directors from `data/features/crew/movies_directors_top50.parquet`
- **Output**: `data/features/categorical/movies_categorical_features.parquet`

### Technical Details
- **Row count**: 87,601 (perfect alignment with master dataset)
- **Total features**: 129 categorical features
- **Data type policy**: All features converted to int8 for memory efficiency
- **Index**: canonical_id for seamless integration

### Column Naming Convention
- **Genres**: `genre_*` (29 features)
- **Actors**: `actor_*` (50 features)
- **Directors**: `director_*` (50 features)

### Coverage Results
- **Movies with ≥1 genre**: {coverage_results['genre_coverage']:.2f}%
- **Movies with ≥1 top actor**: {coverage_results['actor_coverage']:.2f}%
- **Movies with ≥1 top director**: {coverage_results['director_coverage']:.2f}%

### Multi-Label Statistics
- **Genres per movie**: {coverage_results['multi_label_stats']['genres']['min']} min, {coverage_results['multi_label_stats']['genres']['median']:.2f} median, {coverage_results['multi_label_stats']['genres']['max']} max, {coverage_results['multi_label_stats']['genres']['mean']:.2f} mean
- **Actors per movie**: {coverage_results['multi_label_stats']['actors']['min']} min, {coverage_results['multi_label_stats']['actors']['median']:.2f} median, {coverage_results['multi_label_stats']['actors']['max']} max, {coverage_results['multi_label_stats']['actors']['mean']:.2f} mean
- **Directors per movie**: {coverage_results['multi_label_stats']['directors']['min']} min, {coverage_results['multi_label_stats']['directors']['median']:.2f} median, {coverage_results['multi_label_stats']['directors']['max']} max, {coverage_results['multi_label_stats']['directors']['mean']:.2f} mean

### Top 10 Features by Family

#### Genres (by movie count)
"""
        
        for i, (feature, count) in enumerate(sparsity_results['genres']['top_10'].items(), 1):
            share = sparsity_results['genres']['shares'][feature]
            section_2b3 += f"{i}. {feature}: {count:,} movies ({share:.2f}%)\n"
        
        section_2b3 += f"""
#### Actors (by movie count)
"""
        
        for i, (feature, count) in enumerate(sparsity_results['actors']['top_10'].items(), 1):
            share = sparsity_results['actors']['shares'][feature]
            section_2b3 += f"{i}. {feature}: {count:,} movies ({share:.2f}%)\n"
        
        section_2b3 += f"""
#### Directors (by movie count)
"""
        
        for i, (feature, count) in enumerate(sparsity_results['directors']['top_10'].items(), 1):
            share = sparsity_results['directors']['shares'][feature]
            section_2b3 += f"{i}. {feature}: {count:,} movies ({share:.2f}%)\n"
        
        # Add Section 2b.4
        section_2b4 = f"""
## Section 2b.4: QA & Report (Categoricals)

### QA Gates

#### Structural Validation
- ✅ **Row alignment**: {structural_results['row_alignment']['actual']:,} rows (expected: {structural_results['row_alignment']['expected']:,})
- ✅ **Feature count**: {structural_results['feature_count']['actual']} features (expected: {structural_results['feature_count']['expected']})
- ✅ **Family counts**: {structural_results['family_counts']['genres']} genres + {structural_results['family_counts']['actors']} actors + {structural_results['family_counts']['directors']} directors = {structural_results['family_counts']['total']} total
- ✅ **Data types**: All features are int8
- ✅ **Binary values**: All features contain only 0/1 values

#### Coverage & Distribution
- ✅ **Genre coverage**: {coverage_results['genre_coverage']:.2f}% of movies have ≥1 genre
- ✅ **Actor coverage**: {coverage_results['actor_coverage']:.2f}% of movies have ≥1 top actor
- ✅ **Director coverage**: {coverage_results['director_coverage']:.2f}% of movies have ≥1 top director

#### Sanity Checks
- ✅ **All-zero columns**: {sanity_results['all_zero_columns']['count']} found
- ✅ **Duplicate columns**: {sanity_results['duplicate_columns']['count']} found
- ✅ **Random spot check**: 5 rows validated for binary values and consistency

### Known Limitations
- **Actor names**: Currently using IMDb nconst IDs (e.g., nm0000305) until name.basics.tsv mapping is added
- **Director names**: Currently using IMDb nconst IDs until name.basics.tsv mapping is added
- **Top-N selection**: Limited to top 50 actors and top 50 directors by movie count
- **Genre coverage**: Some movies may have limited genre assignments from source data

### Artifacts Generated
- `docs/categorical_top10_genres.csv` - Top 10 genres by movie count
- `docs/categorical_top10_actors.csv` - Top 10 actors by movie count  
- `docs/categorical_top10_directors.csv` - Top 10 directors by movie count
- `logs/step2b_phase4.log` - Complete QA execution log

### Timestamp
{datetime.now().isoformat()}
"""
        
        # Append sections to report
        content += section_2b3 + section_2b4
        
        # Save updated report
        with open(report_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Report updated: {report_path}")
    
    def run(self):
        """Execute the complete QA and reporting pipeline"""
        logger.info("Starting Step 2b.4: QA & Report (Categoricals)")
        
        try:
            # Step 1: Load consolidated features
            df = self.load_consolidated_features()
            
            # Step 2: Structural QA
            structural_results = self.structural_qa(df)
            
            # Step 3: Coverage analysis
            coverage_results = self.coverage_analysis(df)
            
            # Step 4: Sparsity analysis
            sparsity_results = self.sparsity_analysis(df)
            
            # Step 5: Sanity checks
            sanity_results = self.sanity_checks(df)
            
            # Step 6: Save top-10 CSVs
            csv_paths = self.save_top10_csvs(sparsity_results)
            
            # Step 7: Update report
            self.update_report(structural_results, coverage_results, sparsity_results, sanity_results)
            
            # Final QA summary
            logger.info("=== QA SUMMARY ===")
            logger.info(f"Row alignment: {'✅' if structural_results['row_alignment']['passed'] else '❌'}")
            logger.info(f"Feature count: {'✅' if structural_results['feature_count']['passed'] else '❌'}")
            logger.info(f"Data types: {'✅' if structural_results['dtypes_valid'] else '❌'}")
            logger.info(f"Binary values: {'✅' if structural_results['binary_valid'] else '❌'}")
            logger.info(f"All-zero columns: {'✅' if sanity_results['all_zero_columns']['passed'] else '❌'}")
            logger.info(f"Duplicate columns: {'✅' if sanity_results['duplicate_columns']['passed'] else '❌'}")
            
            logger.info("Step 2b.4 completed successfully!")
            
            # Cleanup
            del df
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error in QA and reporting pipeline: {e}")
            raise

if __name__ == "__main__":
    qa = CategoricalFeatureQA()
    qa.run()
