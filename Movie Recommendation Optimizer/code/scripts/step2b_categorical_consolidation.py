#!/usr/bin/env python3
"""
Step 2b.3: Categorical Feature Consolidation
Movie Recommendation Optimizer Project

This script consolidates all categorical features into a single aligned table:
- Genres: 29 canonical genres (from step 2b.1)
- Actors: Top 50 actors (from step 2b.2)
- Directors: Top 50 directors (from step 2b.2)

Output: Single parquet file with 129 feature columns + canonical_id
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
        logging.FileHandler('logs/step2b_phase3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CategoricalFeatureConsolidator:
    def __init__(self):
        self.base_path = Path('.')
        self.data_path = self.base_path / 'data'
        self.features_path = self.data_path / 'features'
        self.categorical_path = self.features_path / 'categorical'
        
        # Ensure directories exist
        self.categorical_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("CategoricalFeatureConsolidator initialized")
        
    def load_feature_tables(self):
        """Load all categorical feature tables"""
        logger.info("Loading categorical feature tables...")
        
        # Load master dataset to get canonical_id mapping
        master_path = self.data_path / 'normalized' / 'movies_master.parquet'
        logger.info(f"Loading master dataset from: {master_path}")
        master_df = pd.read_parquet(master_path)
        logger.info(f"Master shape: {master_df.shape}")
        
        # Load genres (29 features) - add canonical_id from master
        genres_path = self.features_path / 'genres' / 'movies_genres_multihot_full.parquet'
        logger.info(f"Loading genres from: {genres_path}")
        genres_df = pd.read_parquet(genres_path)
        logger.info(f"Genres shape: {genres_df.shape}")
        
        # Add canonical_id to genres (assuming same row order as master)
        genres_df['canonical_id'] = master_df['canonical_id'].values
        logger.info("Added canonical_id to genres table")
        
        # Load actors (50 features)
        actors_path = self.features_path / 'crew' / 'movies_actors_top50.parquet'
        logger.info(f"Loading actors from: {actors_path}")
        actors_df = pd.read_parquet(actors_path)
        logger.info(f"Actors shape: {actors_df.shape}")
        
        # Load directors (50 features)
        directors_path = self.features_path / 'crew' / 'movies_directors_top50.parquet'
        logger.info(f"Loading directors from: {directors_path}")
        directors_df = pd.read_parquet(directors_path)
        logger.info(f"Directors shape: {directors_df.shape}")
        
        return genres_df, actors_df, directors_df
    
    def consolidate_features(self, genres_df, actors_df, directors_df):
        """Consolidate all features into a single table"""
        logger.info("Consolidating categorical features...")
        
        # Ensure all tables have canonical_id
        required_cols = ['canonical_id']
        for df, name in [(genres_df, 'genres'), (actors_df, 'actors'), (directors_df, 'directors')]:
            if 'canonical_id' not in df.columns:
                raise ValueError(f"Missing canonical_id in {name} table")
        
        # Set canonical_id as index for merging
        genres_indexed = genres_df.set_index('canonical_id')
        actors_indexed = actors_df.set_index('canonical_id')
        directors_indexed = directors_df.set_index('canonical_id')
        
        # Merge all tables on canonical_id
        logger.info("Merging feature tables...")
        consolidated = genres_indexed.join(actors_indexed, how='inner')
        consolidated = consolidated.join(directors_indexed, how='inner')
        
        # Reset index to make canonical_id a column again
        consolidated = consolidated.reset_index()
        
        logger.info(f"Consolidated shape: {consolidated.shape}")
        
        return consolidated
    
    def validate_consolidated_table(self, consolidated_df):
        """Validate the consolidated table meets requirements"""
        logger.info("Validating consolidated table...")
        
        expected_rows = 87601
        expected_features = 29 + 50 + 50  # genres + actors + directors
        
        # Check row count
        actual_rows = len(consolidated_df)
        row_check = actual_rows == expected_rows
        logger.info(f"Row count check: {actual_rows} == {expected_rows} -> {row_check}")
        
        # Check canonical_id presence
        canonical_id_check = 'canonical_id' in consolidated_df.columns
        logger.info(f"Canonical ID presence: {canonical_id_check}")
        
        # Check no missing canonical_id values
        no_missing_canonical = not consolidated_df['canonical_id'].isnull().any()
        logger.info(f"No missing canonical_id: {no_missing_canonical}")
        
        # Get feature columns (exclude canonical_id)
        feature_cols = [col for col in consolidated_df.columns if col != 'canonical_id']
        actual_features = len(feature_cols)
        feature_count_check = actual_features == expected_features
        logger.info(f"Feature count check: {actual_features} == {expected_features} -> {feature_count_check}")
        
        # Check dtypes for feature columns
        dtypes_check = True
        for col in feature_cols:
            if consolidated_df[col].dtype not in ['int8', 'int64']:
                dtypes_check = False
                logger.warning(f"Column {col} has dtype {consolidated_df[col].dtype}, expected int8/int64")
        
        logger.info(f"All feature columns are int8/int64: {dtypes_check}")
        
        # Check binary values (0/1 only)
        binary_check = True
        for col in feature_cols:
            unique_vals = consolidated_df[col].unique()
            if not all(val in [0, 1] for val in unique_vals):
                binary_check = False
                logger.warning(f"Column {col} has non-binary values: {unique_vals}")
        
        logger.info(f"All feature columns are binary (0/1): {binary_check}")
        
        validation_results = {
            'row_count_match': row_check,
            'canonical_id_present': canonical_id_check,
            'no_missing_canonical_id': no_missing_canonical,
            'feature_count_match': feature_count_check,
            'dtypes_correct': dtypes_check,
            'binary_values': binary_check,
            'expected_rows': expected_rows,
            'actual_rows': actual_rows,
            'expected_features': expected_features,
            'actual_features': actual_features
        }
        
        return validation_results
    
    def optimize_dtypes(self, consolidated_df):
        """Convert feature columns to int8 for memory efficiency"""
        logger.info("Optimizing dtypes to int8...")
        
        # Convert feature columns to int8
        feature_cols = [col for col in consolidated_df.columns if col != 'canonical_id']
        
        for col in feature_cols:
            consolidated_df[col] = consolidated_df[col].astype('int8')
        
        logger.info("Dtype optimization complete")
        return consolidated_df
    
    def save_consolidated_table(self, consolidated_df):
        """Save the consolidated table and preview"""
        logger.info("Saving consolidated categorical features...")
        
        # Save main table
        main_path = self.categorical_path / 'movies_categorical_features.parquet'
        consolidated_df.to_parquet(main_path, index=False)
        logger.info(f"Main table saved to: {main_path}")
        
        # Save preview
        preview_path = self.categorical_path / 'movies_categorical_features_preview.csv'
        consolidated_df.head(1000).to_csv(preview_path, index=False)
        logger.info(f"Preview saved to: {preview_path}")
        
        return main_path, preview_path
    
    def create_column_list(self, consolidated_df):
        """Create ordered list of all feature columns"""
        logger.info("Creating column list...")
        
        # Get feature columns (exclude canonical_id)
        feature_cols = [col for col in consolidated_df.columns if col != 'canonical_id']
        
        # Sort columns by family and then alphabetically
        genre_cols = sorted([col for col in feature_cols if col.startswith('genre_')])
        actor_cols = sorted([col for col in feature_cols if col.startswith('actor_')])
        director_cols = sorted([col for col in feature_cols if col.startswith('director_')])
        
        ordered_cols = genre_cols + actor_cols + director_cols
        
        # Save column list
        columns_path = self.categorical_path / 'columns_categorical.json'
        columns_data = {
            'total_features': len(ordered_cols),
            'genre_features': len(genre_cols),
            'actor_features': len(actor_cols),
            'director_features': len(director_cols),
            'columns': ordered_cols,
            'created_utc': datetime.now().isoformat()
        }
        
        with open(columns_path, 'w') as f:
            json.dump(columns_data, f, indent=2)
        
        logger.info(f"Column list saved to: {columns_path}")
        return columns_path, columns_data
    
    def create_manifest(self, validation_results, columns_data, file_paths):
        """Create comprehensive manifest file"""
        logger.info("Creating categorical manifest...")
        
        # Get column metadata
        feature_cols = [col for col in columns_data['columns'] if col != 'canonical_id']
        
        # Create column mapping with metadata
        columns_metadata = {}
        for col in feature_cols:
            if col.startswith('genre_'):
                family = 'genre'
            elif col.startswith('actor_'):
                family = 'actor'
            elif col.startswith('director_'):
                family = 'director'
            else:
                family = 'unknown'
            
            columns_metadata[col] = {
                "dtype": "int8",
                "family": family
            }
        
        manifest = {
            'feature_group': 'categorical_v1',
            'row_count': validation_results['actual_rows'],
            'columns': columns_metadata,
            'files': [
                str(file_paths['main']),
                str(file_paths['preview']),
                str(file_paths['columns'])
            ],
            'source_artifacts': [
                'data/features/genres/movies_genres_multihot_full.parquet',
                'data/features/crew/movies_actors_top50.parquet',
                'data/features/crew/movies_directors_top50.parquet'
            ],
            'created_utc': datetime.now().isoformat(),
            'notes': 'Consolidated categorical features: 29 canonical genres, top 50 actors by movie count, top 50 directors by movie count. All features are binary (0/1) encoded and aligned with canonical_id.',
            'validation_results': validation_results
        }
        
        # Save manifest
        manifest_path = self.categorical_path / 'categorical_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Manifest saved to: {manifest_path}")
        return manifest_path
    
    def run(self):
        """Execute the complete categorical consolidation pipeline"""
        logger.info("Starting Step 2b.3: Categorical Feature Consolidation")
        
        try:
            # Step 1: Load feature tables
            genres_df, actors_df, directors_df = self.load_feature_tables()
            
            # Step 2: Consolidate features
            consolidated_df = self.consolidate_features(genres_df, actors_df, directors_df)
            
            # Step 3: Validate consolidated table
            validation_results = self.validate_consolidated_table(consolidated_df)
            
            # Step 4: Optimize dtypes
            consolidated_df = self.optimize_dtypes(consolidated_df)
            
            # Step 5: Save consolidated table
            main_path, preview_path = self.save_consolidated_table(consolidated_df)
            
            # Step 6: Create column list
            columns_path, columns_data = self.create_column_list(consolidated_df)
            
            # Step 7: Create manifest
            file_paths = {
                'main': main_path,
                'preview': preview_path,
                'columns': columns_path
            }
            manifest_path = self.create_manifest(validation_results, columns_data, file_paths)
            file_paths['manifest'] = manifest_path
            
            # Final validation summary
            logger.info("=== CONSOLIDATION SUMMARY ===")
            logger.info(f"Rows: {validation_results['actual_rows']:,}")
            logger.info(f"Features: {validation_results['actual_features']:,}")
            logger.info(f"Genres: {columns_data['genre_features']}")
            logger.info(f"Actors: {columns_data['actor_features']}")
            logger.info(f"Directors: {columns_data['director_features']}")
            logger.info(f"All validations passed: {all([
                validation_results['row_count_match'],
                validation_results['canonical_id_present'],
                validation_results['no_missing_canonical_id'],
                validation_results['feature_count_match'],
                validation_results['dtypes_correct'],
                validation_results['binary_values']
            ])}")
            
            logger.info("Step 2b.3 completed successfully!")
            
            # Cleanup
            del consolidated_df, genres_df, actors_df, directors_df
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error in categorical consolidation pipeline: {e}")
            raise

if __name__ == "__main__":
    consolidator = CategoricalFeatureConsolidator()
    consolidator.run()
