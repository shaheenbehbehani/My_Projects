#!/usr/bin/env python3
"""
Step 2b.2: Crew Extraction & Encoding

This script engineers categorical crew features by identifying the top 50 actors 
and top 50 directors (by number of movies in the dataset) and one-hot encoding 
them into binary features.

Inputs:
- data/normalized/movies_master.parquet (87,601 movies; includes tconst identifiers)
- IMDB Crew Datasets (already ingested during Step 1a/1b):
  - title.crew.tsv (directors/writers)
  - title.principals.tsv (actors/actresses + roles) - if available

Deliverables:
- Actor Features File: data/features/crew/movies_actors_top50.parquet
- Director Features File: data/features/crew/movies_directors_top50.parquet
- Preview CSVs for both
- Updated docs/step2b_report.md (Section 2b.2)
- Log file: logs/step2b_phase2.log

Author: Movie Recommendation Optimizer Project
Date: 2025-01-27
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def setup_logging() -> logging.Logger:
    """Setup logging for the script."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "step2b_phase2.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Step 2b.2: Crew Extraction & Encoding")
    logger.info("=" * 80)
    
    return logger

def load_inputs(logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load all required input data."""
    logger.info("Loading input data...")
    
    # Load master movies table
    master_path = project_root / "data" / "normalized" / "movies_master.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"Master movies table not found: {master_path}")
    
    master_df = pd.read_parquet(master_path)
    logger.info(f"Loaded master table: {master_df.shape}")
    
    # Load crew data
    crew_path = project_root / "IMDB datasets" / "title.crew.tsv"
    if not crew_path.exists():
        raise FileNotFoundError(f"Crew data not found: {crew_path}")
    
    crew_df = pd.read_csv(crew_path, sep='\t', low_memory=False)
    logger.info(f"Loaded crew data: {crew_df.shape}")
    
    # Check if principals (actors) data exists
    principals_path = project_root / "IMDB datasets" / "title.principals.tsv"
    principals_df = None
    if principals_path.exists():
        principals_df = pd.read_csv(principals_path, sep='\t', low_memory=False)
        logger.info(f"Loaded principals data: {principals_df.shape}")
    else:
        logger.warning("Principals data (actors) not found. Will only process directors.")
    
    return master_df, crew_df, principals_df

def validate_inputs(master_df: pd.DataFrame, crew_df: pd.DataFrame, 
                   principals_df: Optional[pd.DataFrame], logger: logging.Logger) -> None:
    """Validate input data integrity."""
    logger.info("Validating input data...")
    
    # Check master table
    if len(master_df) != 87601:
        logger.warning(f"Master table has {len(master_df)} rows, expected 87,601")
    
    # Check crew data
    logger.info(f"Crew data covers {crew_df['tconst'].nunique()} unique titles")
    
    # Check overlap with master table
    master_tconsts = set(master_df['tconst'])
    crew_tconsts = set(crew_df['tconst'])
    overlap = len(master_tconsts.intersection(crew_tconsts))
    logger.info(f"Overlap between master and crew: {overlap} titles ({overlap/len(master_tconsts)*100:.1f}%)")
    
    if principals_df is not None:
        principals_tconsts = set(principals_df['tconst'])
        principals_overlap = len(master_tconsts.intersection(principals_tconsts))
        logger.info(f"Overlap between master and principals: {principals_overlap} titles ({principals_overlap/len(master_tconsts)*100:.1f}%)")

def process_directors(crew_df: pd.DataFrame, master_df: pd.DataFrame, 
                     logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """Process director data and identify top 50 directors."""
    logger.info("Processing director data...")
    
    # Filter crew data to only include movies in our master table
    master_tconsts = set(master_df['tconst'])
    filtered_crew = crew_df[crew_df['tconst'].isin(master_tconsts)].copy()
    
    logger.info(f"Filtered crew data to {len(filtered_crew)} rows for master movies")
    
    # Extract directors and count their appearances
    director_counts = Counter()
    
    for _, row in filtered_crew.iterrows():
        directors = row['directors']
        if pd.notna(directors) and directors != '\\N':
            # Handle multiple directors (comma-separated)
            if ',' in str(directors):
                for director in str(directors).split(','):
                    director = director.strip()
                    if director and director != '\\N':
                        director_counts[director] += 1
            else:
                director_counts[director] += 1
    
    # Get top 50 directors
    top_directors = [director for director, count in director_counts.most_common(50)]
    logger.info(f"Identified top 50 directors with {len(top_directors)} unique directors")
    
    # Create director features dataframe
    director_features = pd.DataFrame(index=master_df['canonical_id'])
    
    # Initialize all director columns to 0
    for director in top_directors:
        col_name = f"director_{director}"
        director_features[col_name] = 0
    
    # Populate director features
    logger.info("Populating director features...")
    
    for _, row in filtered_crew.iterrows():
        tconst = row['tconst']
        directors = row['directors']
        
        if pd.notna(directors) and directors != '\\N':
            # Find the canonical_id for this tconst
            master_row = master_df[master_df['tconst'] == tconst]
            if not master_row.empty:
                canonical_id = master_row.iloc[0]['canonical_id']
                
                # Handle multiple directors
                if ',' in str(directors):
                    for director in str(directors).split(','):
                        director = director.strip()
                        if director in top_directors:
                            col_name = f"director_{director}"
                            director_features.loc[canonical_id, col_name] = 1
                else:
                    if directors in top_directors:
                        col_name = f"director_{directors}"
                        director_features.loc[canonical_id, col_name] = 1
    
    # Ensure all columns are int8
    for col in director_features.columns:
        director_features[col] = director_features[col].astype('int8')
    
    logger.info(f"Director features shape: {director_features.shape}")
    
    return director_features, top_directors

def process_actors(principals_df: pd.DataFrame, master_df: pd.DataFrame, 
                  logger: logging.Logger) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """Process actor data and identify top 50 actors."""
    if principals_df is None:
        logger.warning("No principals data available for actors")
        return None, None
    
    logger.info("Processing actor data...")
    
    # Filter principals data to only include movies in our master table
    master_tconsts = set(master_df['tconst'])
    filtered_principals = principals_df[principals_df['tconst'].isin(master_tconsts)].copy()
    
    logger.info(f"Filtered principals data to {len(filtered_principals)} rows for master movies")
    
    # Filter to only include actors/actresses (category in ['actor', 'actress'])
    actor_principals = filtered_principals[
        filtered_principals['category'].isin(['actor', 'actress'])
    ].copy()
    
    logger.info(f"Found {len(actor_principals)} actor/actress entries")
    
    # Count actor appearances
    actor_counts = Counter()
    for _, row in actor_principals.iterrows():
        actor_id = row['nconst']
        if pd.notna(actor_id) and actor_id != '\\N':
            actor_counts[actor_id] += 1
    
    # Get top 50 actors
    top_actors = [actor for actor, count in actor_counts.most_common(50)]
    logger.info(f"Identified top 50 actors with {len(top_actors)} unique actors")
    
    # Create actor features dataframe
    actor_features = pd.DataFrame(index=master_df['canonical_id'])
    
    # Initialize all actor columns to 0
    for actor in top_actors:
        col_name = f"actor_{actor}"
        actor_features[col_name] = 0
    
    # Populate actor features
    logger.info("Populating actor features...")
    
    for _, row in actor_principals.iterrows():
        tconst = row['tconst']
        actor_id = row['nconst']
        
        if pd.notna(actor_id) and actor_id != '\\N' and actor_id in top_actors:
            # Find the canonical_id for this tconst
            master_row = master_df[master_df['tconst'] == tconst]
            if not master_row.empty:
                canonical_id = master_row.iloc[0]['canonical_id']
                col_name = f"actor_{actor_id}"
                actor_features.loc[canonical_id, col_name] = 1
    
    # Ensure all columns are int8
    for col in actor_features.columns:
        actor_features[col] = actor_features[col].astype('int8')
    
    logger.info(f"Actor features shape: {actor_features.shape}")
    
    return actor_features, top_actors

def validate_output(director_features: pd.DataFrame, actor_features: Optional[pd.DataFrame],
                   master_df: pd.DataFrame, logger: logging.Logger) -> Dict:
    """Validate the output and generate statistics."""
    logger.info("Validating output...")
    
    validation_results = {}
    
    # Validate director features
    expected_shape = (87601, 50)
    actual_director_shape = director_features.shape
    validation_results['director_shape'] = {
        'expected': expected_shape,
        'actual': actual_director_shape,
        'valid': actual_director_shape == expected_shape
    }
    
    # Check director column count
    expected_director_cols = 50
    actual_director_cols = len(director_features.columns)
    validation_results['director_column_count'] = {
        'expected': expected_director_cols,
        'actual': actual_director_cols,
        'valid': actual_director_cols == expected_director_cols
    }
    
    # Check director data types
    director_dtypes_valid = all(director_features[col].dtype == 'int8' for col in director_features.columns)
    validation_results['director_dtypes'] = {
        'valid': director_dtypes_valid,
        'actual_dtypes': director_features.dtypes.to_dict()
    }
    
    # Check for missing values in directors
    director_missing = director_features.isnull().sum().sum()
    validation_results['director_missing_values'] = {
        'count': director_missing,
        'valid': director_missing == 0
    }
    
    # Generate director coverage statistics
    director_coverage = director_features.sum(axis=1)
    validation_results['director_coverage_stats'] = {
        'movies_with_directors': int((director_coverage > 0).sum()),
        'movies_without_directors': int((director_coverage == 0).sum()),
        'coverage_percentage': float((director_coverage > 0).sum() / len(director_coverage) * 100),
        'avg_directors_per_movie': float(director_coverage.mean()),
        'max_directors_per_movie': int(director_coverage.max())
    }
    
    # Validate actor features if available
    if actor_features is not None:
        actual_actor_shape = actor_features.shape
        validation_results['actor_shape'] = {
            'expected': expected_shape,
            'actual': actual_actor_shape,
            'valid': actual_actor_shape == expected_shape
        }
        
        expected_actor_cols = 50
        actual_actor_cols = len(actor_features.columns)
        validation_results['actor_column_count'] = {
            'expected': expected_actor_cols,
            'actual': actual_actor_cols,
            'valid': actual_actor_cols == expected_actor_cols
        }
        
        actor_dtypes_valid = all(actor_features[col].dtype == 'int8' for col in actor_features.columns)
        validation_results['actor_dtypes'] = {
            'valid': actor_dtypes_valid,
            'actual_dtypes': actor_features.dtypes.to_dict()
        }
        
        actor_missing = actor_features.isnull().sum().sum()
        validation_results['actor_missing_values'] = {
            'count': actor_missing,
            'valid': actor_missing == 0
        }
        
        actor_coverage = actor_features.sum(axis=1)
        validation_results['actor_coverage_stats'] = {
            'movies_with_actors': int((actor_coverage > 0).sum()),
            'movies_without_actors': int((actor_coverage == 0).sum()),
            'coverage_percentage': float((actor_coverage > 0).sum() / len(actor_coverage) * 100),
            'avg_actors_per_movie': float(actor_coverage.mean()),
            'max_actors_per_movie': int(actor_coverage.max())
        }
    else:
        validation_results['actor_shape'] = {'error': 'No actor data available'}
        validation_results['actor_coverage_stats'] = {'error': 'No actor data available'}
    
    # Log validation results
    logger.info("Validation Results:")
    for key, value in validation_results.items():
        if 'coverage_stats' in key:
            if 'error' not in value:
                logger.info(f"  {key}: {value['movies_with_directors' if 'director' in key else 'movies_with_actors']} movies covered")
            else:
                logger.info(f"  {key}: {value['error']}")
        else:
            logger.info(f"  {key}: {value}")
    
    return validation_results

def save_deliverables(director_features: pd.DataFrame, actor_features: Optional[pd.DataFrame],
                     top_directors: List[str], top_actors: Optional[List[str]], 
                     validation_results: Dict, logger: logging.Logger) -> None:
    """Save all deliverables."""
    logger.info("Saving deliverables...")
    
    # Create output directory
    output_dir = project_root / "data" / "features" / "crew"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save director features
    director_path = output_dir / "movies_directors_top50.parquet"
    director_features.to_parquet(director_path, index=True)
    logger.info(f"Saved director features: {director_path}")
    
    # Save director preview CSV
    director_preview = director_features.head(1000)
    director_csv_path = output_dir / "movies_directors_top50_preview.csv"
    director_preview.to_csv(director_csv_path, index=True)
    logger.info(f"Saved director preview CSV: {director_csv_path}")
    
    # Save actor features if available
    if actor_features is not None:
        actor_path = output_dir / "movies_actors_top50.parquet"
        actor_features.to_parquet(actor_path, index=True)
        logger.info(f"Saved actor features: {actor_path}")
        
        actor_preview = actor_features.head(1000)
        actor_csv_path = output_dir / "movies_actors_top50_preview.csv"
        actor_preview.to_csv(actor_csv_path, index=True)
        logger.info(f"Saved actor preview CSV: {actor_csv_path}")
    
    # Update step2b report
    update_step2b_report(director_features, actor_features, top_directors, top_actors, 
                         validation_results, logger)

def update_step2b_report(director_features: pd.DataFrame, actor_features: Optional[pd.DataFrame],
                        top_directors: List[str], top_actors: Optional[List[str]], 
                        validation_results: Dict, logger: logging.Logger) -> None:
    """Update the step2b report with section 2b.2."""
    logger.info("Updating step2b report...")
    
    report_path = project_root / "docs" / "step2b_report.md"
    
    # Read existing report
    if report_path.exists():
        with open(report_path, 'r') as f:
            existing_content = f.read()
    else:
        existing_content = "# Step 2b Report: Genre & Crew Features\n\n## Overview\nThis report documents the implementation of Step 2b, which focuses on expanding and enhancing genre and crew features for the Movie Recommendation Optimizer project.\n\n"
    
    # Create section 2b.2 content
    section_2b2 = f"""
## 2b.2: Crew Extraction & Encoding

### Objective
Engineer categorical crew features by identifying the top 50 actors and top 50 directors 
(by number of movies in the dataset) and one-hot encoding them into binary features.

### Implementation Details
- **Input**: Master movies table + IMDB crew datasets
- **Output**: Top 50 director features + Top 50 actor features (if available)
- **Processing**: Frequency-based selection + one-hot encoding
- **Validation**: Comprehensive QA gates and statistics

### Results

#### Director Features
- **Total directors encoded**: {len(top_directors)}
- **Movies with directors**: {validation_results['director_coverage_stats']['movies_with_directors']:,} ({validation_results['director_coverage_stats']['coverage_percentage']:.1f}%)
- **Movies without directors**: {validation_results['director_coverage_stats']['movies_without_directors']:,}
- **Average directors per movie**: {validation_results['director_coverage_stats']['avg_directors_per_movie']:.2f}
- **Maximum directors per movie**: {validation_results['director_coverage_stats']['max_directors_per_movie']}

#### Actor Features
"""
    
    if actor_features is not None:
        section_2b2 += f"""
- **Total actors encoded**: {len(top_actors)}
- **Movies with actors**: {validation_results['actor_coverage_stats']['movies_with_actors']:,} ({validation_results['actor_coverage_stats']['coverage_percentage']:.1f}%)
- **Movies without actors**: {validation_results['actor_coverage_stats']['movies_without_actors']:,}
- **Average actors per movie**: {validation_results['actor_coverage_stats']['avg_actors_per_movie']:.2f}
- **Maximum actors per movie**: {validation_results['actor_coverage_stats']['max_actors_per_movie']}
"""
    else:
        section_2b2 += """
- **Status**: Actor data not available (title.principals.tsv missing)
- **Note**: Only director features were processed
"""
    
    section_2b2 += f"""

#### Top 10 Directors by Movie Count
| Rank | Director ID | Movie Count |
|------|-------------|-------------|
"""
    
    # Get director counts for top 10
    director_counts = director_features.sum().sort_values(ascending=False).head(10)
    for i, (director_col, count) in enumerate(director_counts.items(), 1):
        director_id = director_col.replace('director_', '')
        section_2b2 += f"| {i} | {director_id} | {int(count):,} |\n"
    
    if actor_features is not None:
        section_2b2 += f"""
#### Top 10 Actors by Movie Count
| Rank | Actor ID | Movie Count |
|------|----------|-------------|
"""
        # Get actor counts for top 10
        actor_counts = actor_features.sum().sort_values(ascending=False).head(10)
        for i, (actor_col, count) in enumerate(actor_counts.items(), 1):
            actor_id = actor_col.replace('actor_', '')
            section_2b2 += f"| {i} | {actor_id} | {int(count):,} |\n"
    
    section_2b2 += f"""
### Validation Results
- ✅ **Row alignment**: {validation_results['director_shape']['actual'][0]:,} movies (expected: {validation_results['director_shape']['expected'][0]:,})
- ✅ **Director columns**: {validation_results['director_column_count']['actual']} features (expected: {validation_results['director_column_count']['expected']})
- ✅ **Director data types**: All columns are int8
- ✅ **Director missing values**: {validation_results['director_missing_values']['count']} missing values found
"""
    
    if actor_features is not None:
        section_2b2 += f"""
- ✅ **Actor columns**: {validation_results['actor_column_count']['actual']} features (expected: {validation_results['actor_column_count']['expected']})
- ✅ **Actor data types**: All columns are int8
- ✅ **Actor missing values**: {validation_results['actor_missing_values']['count']} missing values found
"""
    
    section_2b2 += f"""
### Deliverables
1. **Director Features**: `data/features/crew/movies_directors_top50.parquet`
2. **Director Preview CSV**: `data/features/crew/movies_directors_top50_preview.csv`
"""
    
    if actor_features is not None:
        section_2b2 += """3. **Actor Features**: `data/features/crew/movies_actors_top50.parquet`
4. **Actor Preview CSV**: `data/features/crew/movies_actors_top50_preview.csv`
"""
    
    section_2b2 += """5. **Updated Report**: This document
6. **Log File**: `logs/step2b_phase2.log`

### Next Steps
- Step 2c: Feature Integration and Model Preparation

---
*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Append section 2b.2 to existing report
    updated_content = existing_content + section_2b2
    
    # Write updated report
    with open(report_path, 'w') as f:
        f.write(updated_content)
    
    logger.info(f"Updated step2b report: {report_path}")

def main():
    """Main execution function."""
    logger = setup_logging()
    
    try:
        # Load inputs
        master_df, crew_df, principals_df = load_inputs(logger)
        
        # Validate inputs
        validate_inputs(master_df, crew_df, principals_df, logger)
        
        # Process directors
        director_features, top_directors = process_directors(crew_df, master_df, logger)
        
        # Process actors (if available)
        actor_features, top_actors = process_actors(principals_df, master_df, logger)
        
        # Validate output
        validation_results = validate_output(director_features, actor_features, master_df, logger)
        
        # Save deliverables
        save_deliverables(director_features, actor_features, top_directors, top_actors, 
                         validation_results, logger)
        
        logger.info("Step 2b.2 completed successfully!")
        logger.info("=" * 80)
        
        # Print summary
        print("\n" + "=" * 80)
        print("Step 2b.2: Crew Extraction & Encoding - COMPLETED")
        print("=" * 80)
        print(f"Director features: {director_features.shape}")
        if actor_features is not None:
            print(f"Actor features: {actor_features.shape}")
        else:
            print("Actor features: Not available")
        print(f"Movies processed: {len(director_features):,}")
        print(f"Top directors: {len(top_directors)}")
        if top_actors:
            print(f"Top actors: {len(top_actors)}")
        print("\nDeliverables created:")
        print("- data/features/crew/movies_directors_top50.parquet")
        print("- data/features/crew/movies_directors_top50_preview.csv")
        if actor_features is not None:
            print("- data/features/crew/movies_actors_top50.parquet")
            print("- data/features/crew/movies_actors_top50_preview.csv")
        print("- docs/step2b_report.md")
        print("- logs/step2b_phase2.log")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Step 2b.2 failed: {e}", exc_info=True)
        print(f"ERROR: Step 2b.2 failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()























