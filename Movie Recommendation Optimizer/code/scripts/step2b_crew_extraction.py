#!/usr/bin/env python3
"""
Step 2b.2: Crew Extraction & Encoding
Movie Recommendation Optimizer Project

This script implements crew feature engineering by:
1. Auditing title.principals.tsv and title.crew.tsv
2. Extracting top 50 actors and top 50 directors
3. Creating multi-hot encoded features
4. Aligning with movies_master.parquet
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
        logging.FileHandler('logs/step2b_phase2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CrewFeatureExtractor:
    def __init__(self):
        self.base_path = Path('.')
        self.imdb_path = self.base_path / 'IMDB datasets'
        self.data_path = self.base_path / 'data'
        self.features_path = self.data_path / 'features' / 'crew'
        self.docs_path = self.base_path / 'docs'
        
        # Ensure directories exist
        self.features_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("CrewFeatureExtractor initialized")
        
    def audit_title_principals(self):
        """Audit title.principals.tsv and create summary"""
        logger.info("Starting audit of title.principals.tsv")
        
        file_path = self.imdb_path / 'title.principals .tsv'  # Note the space in filename
        
        # Read sample to understand structure
        sample_df = pd.read_csv(file_path, sep='\t', nrows=1000)
        logger.info(f"Sample columns: {list(sample_df.columns)}")
        
        # Count total rows efficiently
        logger.info("Counting total rows...")
        with open(file_path, 'r') as f:
            total_rows = sum(1 for line in f) - 1  # Subtract header
        
        logger.info(f"Total rows: {total_rows:,}")
        
        # Analyze categories
        logger.info("Analyzing categories...")
        categories = {}
        null_counts = {}
        
        # Process in chunks to handle large file
        chunk_size = 100000
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, sep='\t', chunksize=chunk_size)):
            if chunk_num == 0:  # First chunk
                null_counts = chunk.isnull().sum().to_dict()
            
            # Count categories
            if 'category' in chunk.columns:
                cat_counts = chunk['category'].value_counts()
                for cat, count in cat_counts.items():
                    categories[cat] = categories.get(cat, 0) + count
            
            if chunk_num % 10 == 0:
                logger.info(f"Processed chunk {chunk_num}")
        
        # Create summary
        summary = {
            'file_path': str(file_path),
            'total_rows': total_rows,
            'columns': list(sample_df.columns),
            'null_counts': null_counts,
            'category_frequencies': categories,
            'audit_timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = self.docs_path / 'title_principals_summary.md'
        with open(summary_path, 'w') as f:
            f.write("# title.principals.tsv Audit Summary\n\n")
            f.write(f"**Audit Date:** {summary['audit_timestamp']}\n\n")
            f.write(f"**File Path:** {summary['file_path']}\n\n")
            f.write(f"**Total Rows:** {summary['total_rows']:,}\n\n")
            
            f.write("## Columns\n")
            for col in summary['columns']:
                f.write(f"- {col}\n")
            f.write("\n")
            
            f.write("## Null Counts\n")
            for col, null_count in summary['null_counts'].items():
                f.write(f"- {col}: {null_count:,}\n")
            f.write("\n")
            
            f.write("## Category Frequencies\n")
            for cat, count in sorted(summary['category_frequencies'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {cat}: {count:,}\n")
        
        logger.info(f"Summary saved to {summary_path}")
        return summary
    
    def extract_top_actors(self, summary):
        """Extract top 50 actors by movie appearances"""
        logger.info("Extracting top 50 actors...")
        
        file_path = self.imdb_path / 'title.principals .tsv'
        movies_master_path = self.data_path / 'normalized' / 'movies_master.parquet'
        
        # Load movies master to get valid tconsts
        logger.info("Loading movies master...")
        movies_master = pd.read_parquet(movies_master_path)
        valid_tconsts = set(movies_master['tconst'].dropna())
        logger.info(f"Valid tconsts in master: {len(valid_tconsts):,}")
        
        # Process actors in chunks
        actor_counts = {}
        chunk_size = 100000
        
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, sep='\t', chunksize=chunk_size)):
            # Filter for actors/actresses and valid tconsts
            actor_chunk = chunk[
                (chunk['category'].isin(['actor', 'actress'])) & 
                (chunk['tconst'].isin(valid_tconsts))
            ]
            
            # Count appearances per actor
            for _, row in actor_chunk.iterrows():
                nconst = row['nconst']
                if pd.notna(nconst):
                    actor_counts[nconst] = actor_counts.get(nconst, 0) + 1
            
            if chunk_num % 20 == 0:
                logger.info(f"Processed actor chunk {chunk_num}")
        
        # Get top 50 actors
        top_actors = sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)[:50]
        
        # Save top actors list
        actors_data = {
            'extraction_timestamp': datetime.now().isoformat(),
            'total_actors_found': len(actor_counts),
            'top_50_actors': [
                {'nconst': nconst, 'movie_count': count} 
                for nconst, count in top_actors
            ]
        }
        
        actors_path = self.docs_path / 'crew_top50_actors.json'
        with open(actors_path, 'w') as f:
            json.dump(actors_data, f, indent=2)
        
        logger.info(f"Top 50 actors saved to {actors_path}")
        logger.info(f"Top 10 actors: {top_actors[:10]}")
        
        return top_actors
    
    def create_actors_features(self, top_actors):
        """Create multi-hot encoded features for top 50 actors"""
        logger.info("Creating actors features...")
        
        file_path = self.imdb_path / 'title.principals .tsv'
        movies_master_path = self.data_path / 'normalized' / 'movies_master.parquet'
        
        # Load movies master
        movies_master = pd.read_parquet(movies_master_path)
        logger.info(f"Movies master shape: {movies_master.shape}")
        
        # Create actor feature columns
        actor_nconsts = [actor[0] for actor in top_actors]
        actor_features = pd.DataFrame(index=movies_master.index)
        
        # Initialize all features to 0
        for nconst in actor_nconsts:
            actor_features[f'actor_{nconst}'] = 0
        
        # Process in chunks to find actor appearances
        chunk_size = 100000
        processed_movies = set()
        
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, sep='\t', chunksize=chunk_size)):
            # Filter for actors/actresses and valid tconsts
            actor_chunk = chunk[
                (chunk['category'].isin(['actor', 'actress'])) & 
                (chunk['tconst'].isin(movies_master['tconst']))
            ]
            
            # Process each actor appearance
            for _, row in actor_chunk.iterrows():
                tconst = row['tconst']
                nconst = row['nconst']
                
                if pd.notna(tconst) and pd.notna(nconst) and nconst in actor_nconsts:
                    # Find corresponding row in movies_master
                    master_idx = movies_master[movies_master['tconst'] == tconst].index
                    if len(master_idx) > 0:
                        actor_features.loc[master_idx, f'actor_{nconst}'] = 1
                        processed_movies.add(tconst)
            
            if chunk_num % 20 == 0:
                logger.info(f"Processed actor features chunk {chunk_num}")
        
        # Add canonical_id for alignment
        actor_features['canonical_id'] = movies_master['canonical_id']
        
        # Save features
        features_path = self.features_path / 'movies_actors_top50.parquet'
        actor_features.to_parquet(features_path, index=False)
        
        # Save preview
        preview_path = self.features_path / 'movies_actors_top50_preview.csv'
        actor_features.head(1000).to_csv(preview_path, index=False)
        
        logger.info(f"Actor features saved to {features_path}")
        logger.info(f"Preview saved to {preview_path}")
        logger.info(f"Shape: {actor_features.shape}")
        
        return actor_features
    
    def extract_top_directors(self):
        """Extract top 50 directors by movie appearances"""
        logger.info("Extracting top 50 directors...")
        
        file_path = self.imdb_path / 'title.crew.tsv'
        movies_master_path = self.data_path / 'normalized' / 'movies_master.parquet'
        
        # Load movies master to get valid tconsts
        movies_master = pd.read_parquet(movies_master_path)
        valid_tconsts = set(movies_master['tconst'].dropna())
        
        # Process directors
        director_counts = {}
        chunk_size = 100000
        
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, sep='\t', chunksize=chunk_size)):
            # Filter for valid tconsts
            valid_chunk = chunk[chunk['tconst'].isin(valid_tconsts)]
            
            # Parse directors field (comma-separated nconsts)
            for _, row in valid_chunk.iterrows():
                directors = row['directors']
                if pd.notna(directors) and directors != '\\N':
                    director_list = directors.split(',')
                    for director in director_list:
                        director = director.strip()
                        if director:
                            director_counts[director] = director_counts.get(director, 0) + 1
            
            if chunk_num % 10 == 0:
                logger.info(f"Processed director chunk {chunk_num}")
        
        # Get top 50 directors
        top_directors = sorted(director_counts.items(), key=lambda x: x[1], reverse=True)[:50]
        
        # Save top directors list
        directors_data = {
            'extraction_timestamp': datetime.now().isoformat(),
            'total_directors_found': len(director_counts),
            'top_50_directors': [
                {'nconst': nconst, 'movie_count': count} 
                for nconst, count in top_directors
            ]
        }
        
        directors_path = self.docs_path / 'crew_top50_directors.json'
        with open(directors_path, 'w') as f:
            json.dump(directors_data, f, indent=2)
        
        logger.info(f"Top 50 directors saved to {directors_path}")
        logger.info(f"Top 10 directors: {top_directors[:10]}")
        
        return top_directors
    
    def create_directors_features(self, top_directors):
        """Create multi-hot encoded features for top 50 directors"""
        logger.info("Creating directors features...")
        
        file_path = self.imdb_path / 'title.crew.tsv'
        movies_master_path = self.data_path / 'normalized' / 'movies_master.parquet'
        
        # Load movies master
        movies_master = pd.read_parquet(movies_master_path)
        
        # Create director feature columns
        director_nconsts = [director[0] for director in top_directors]
        director_features = pd.DataFrame(index=movies_master.index)
        
        # Initialize all features to 0
        for nconst in director_nconsts:
            director_features[f'director_{nconst}'] = 0
        
        # Process in chunks to find director appearances
        chunk_size = 100000
        
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, sep='\t', chunksize=chunk_size)):
            # Filter for valid tconsts
            valid_chunk = chunk[chunk['tconst'].isin(movies_master['tconst'])]
            
            # Process each director appearance
            for _, row in valid_chunk.iterrows():
                tconst = row['tconst']
                directors = row['directors']
                
                if pd.notna(tconst) and pd.notna(directors) and directors != '\\N':
                    director_list = directors.split(',')
                    for director in director_list:
                        director = director.strip()
                        if director in director_nconsts:
                            # Find corresponding row in movies_master
                            master_idx = movies_master[movies_master['tconst'] == tconst].index
                            if len(master_idx) > 0:
                                director_features.loc[master_idx, f'director_{director}'] = 1
            
            if chunk_num % 10 == 0:
                logger.info(f"Processed director features chunk {chunk_num}")
        
        # Add canonical_id for alignment
        director_features['canonical_id'] = movies_master['canonical_id']
        
        # Save features
        features_path = self.features_path / 'movies_directors_top50.parquet'
        director_features.to_parquet(features_path, index=False)
        
        # Save preview
        preview_path = self.features_path / 'movies_directors_top50_preview.csv'
        director_features.head(1000).to_csv(preview_path, index=False)
        
        logger.info(f"Director features saved to {features_path}")
        logger.info(f"Preview saved to {preview_path}")
        logger.info(f"Shape: {director_features.shape}")
        
        return director_features
    
    def validate_features(self, actor_features, director_features):
        """Validate the created features"""
        logger.info("Validating features...")
        
        movies_master_path = self.data_path / 'normalized' / 'movies_master.parquet'
        movies_master = pd.read_parquet(movies_master_path)
        
        # Check row counts
        expected_rows = len(movies_master)
        actor_rows = len(actor_features)
        director_rows = len(director_features)
        
        logger.info(f"Expected rows: {expected_rows:,}")
        logger.info(f"Actor features rows: {actor_rows:,}")
        logger.info(f"Director features rows: {director_rows:,}")
        
        # Check feature types
        actor_cols = [col for col in actor_features.columns if col.startswith('actor_')]
        director_cols = [col for col in director_features.columns if col.startswith('director_')]
        
        actor_dtypes = actor_features[actor_cols].dtypes
        director_dtypes = director_features[director_cols].dtypes
        
        logger.info(f"Actor feature dtypes: {actor_dtypes.unique()}")
        logger.info(f"Director feature dtypes: {director_dtypes.unique()}")
        
        # Calculate coverage statistics
        actor_coverage = (actor_features[actor_cols].sum(axis=1) > 0).mean() * 100
        director_coverage = (director_features[director_cols].sum(axis=1) > 0).mean() * 100
        
        logger.info(f"Movies with at least one top actor: {actor_coverage:.2f}%")
        logger.info(f"Movies with at least one top director: {director_coverage:.2f}%")
        
        # Validate canonical_id integrity
        actor_canonical_ids = set(actor_features['canonical_id'].dropna())
        director_canonical_ids = set(director_features['canonical_id'].dropna())
        master_canonical_ids = set(movies_master['canonical_id'].dropna())
        
        actor_integrity = len(actor_canonical_ids - master_canonical_ids) == 0
        director_integrity = len(director_canonical_ids - master_canonical_ids) == 0
        
        logger.info(f"Actor canonical_id integrity: {actor_integrity}")
        logger.info(f"Director canonical_id integrity: {director_integrity}")
        
        validation_results = {
            'row_counts_match': actor_rows == expected_rows and director_rows == expected_rows,
            'feature_types_binary': all(dtype in ['int8', 'int64'] for dtype in actor_dtypes) and all(dtype in ['int8', 'int64'] for dtype in director_dtypes),
            'actor_coverage_percent': actor_coverage,
            'director_coverage_percent': director_coverage,
            'canonical_id_integrity': actor_integrity and director_integrity
        }
        
        return validation_results
    
    def update_documentation(self, summary, top_actors, top_directors, validation_results):
        """Update step2b_report.md with Section 2b.2"""
        logger.info("Updating documentation...")
        
        report_path = self.docs_path / 'step2b_report.md'
        
        # Read existing report
        if report_path.exists():
            with open(report_path, 'r') as f:
                content = f.read()
        else:
            content = "# Step 2b Report: Feature Engineering\n\n"
        
        # Add Section 2b.2
        section_2b2 = f"""
## Section 2b.2: Crew Features (Actors + Directors)

### Methodology
- **Actors Extraction**: Filtered `title.principals.tsv` for category ∈ {{actor, actress}}
- **Directors Extraction**: Parsed `title.crew.tsv` directors field (comma-separated nconsts)
- **Top 50 Selection**: Ranked by movie appearance count
- **Multi-Hot Encoding**: Binary features (actor_<id>, director_<id>)
- **Alignment**: Matched with `movies_master.parquet` canonical_id (87,601 movies)

### Top 10 Actors (by movie count)
"""
        
        for i, (nconst, count) in enumerate(top_actors[:10], 1):
            section_2b2 += f"{i}. {nconst}: {count:,} movies\n"
        
        section_2b2 += f"\n### Top 10 Directors (by movie count)\n"
        
        for i, (nconst, count) in enumerate(top_directors[:10], 1):
            section_2b2 += f"{i}. {nconst}: {count:,} movies\n"
        
        section_2b2 += f"""
### Coverage Statistics
- **Movies with at least one top actor**: {validation_results['actor_coverage_percent']:.2f}%
- **Movies with at least one top director**: {validation_results['director_coverage_percent']:.2f}%
- **Total feature columns**: {len(top_actors) + len(top_directors)} (50 actors + 50 directors)

### Validation Results
- **Row counts match**: {validation_results['row_counts_match']}
- **Feature types binary**: {validation_results['feature_types_binary']}
- **Canonical ID integrity**: {validation_results['canonical_id_integrity']}

### Output Files
- `data/features/crew/movies_actors_top50.parquet`
- `data/features/crew/movies_directors_top50.parquet`
- `docs/crew_top50_actors.json`
- `docs/crew_top50_directors.json`

### Timestamp
{datetime.now().isoformat()}
"""
        
        # Append section to report
        content += section_2b2
        
        # Save updated report
        with open(report_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Documentation updated: {report_path}")
    
    def run(self):
        """Execute the complete crew extraction pipeline"""
        logger.info("Starting Step 2b.2: Crew Extraction & Encoding")
        
        try:
            # Step 1: Audit & Snapshot
            summary = self.audit_title_principals()
            
            # Step 2: Actors Extraction
            top_actors = self.extract_top_actors(summary)
            
            # Step 3: Actors Multi-Hot Encoding
            actor_features = self.create_actors_features(top_actors)
            
            # Step 4: Directors Extraction
            top_directors = self.extract_top_directors()
            
            # Step 5: Directors Multi-Hot Encoding
            director_features = self.create_directors_features(top_directors)
            
            # Step 6: QA & Validation
            validation_results = self.validate_features(actor_features, director_features)
            
            # Step 7: Documentation
            self.update_documentation(summary, top_actors, top_directors, validation_results)
            
            logger.info("Step 2b.2 completed successfully!")
            
            # Cleanup
            del actor_features, director_features
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error in crew extraction pipeline: {e}")
            raise

if __name__ == "__main__":
    extractor = CrewFeatureExtractor()
    extractor.run()





















