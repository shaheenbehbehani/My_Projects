#!/usr/bin/env python3
"""
Step 1b Phase 2 - Sub-phase 2.0: Setup & Snapshots
Takes baseline snapshots of all input datasets for ID Resolution & Deduping
"""

import os
import json
import hashlib
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/step1b_phase2.log', mode='a'),
        logging.StreamHandler()
    ]
)

def calculate_file_hash(file_path):
    """Calculate SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def take_dataset_snapshot(file_path, file_type='csv'):
    """Take a snapshot of a dataset file"""
    try:
        logging.info(f"Processing: {file_path}")
        
        # Get file stats
        file_size = os.path.getsize(file_path)
        file_hash = calculate_file_hash(file_path)
        
        # Read the file based on type
        if file_type == 'tsv':
            df = pd.read_csv(file_path, sep='\t', nrows=5)
            full_df = pd.read_csv(file_path, sep='\t')
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path)
            full_df = df.copy()
        else:  # csv
            df = pd.read_csv(file_path, nrows=5)
            full_df = pd.read_csv(file_path)
        
        # Get row count
        row_count = len(full_df)
        
        # Get column names
        columns = list(full_df.columns)
        
        # Get sample rows (first 5)
        sample_rows = df.head().to_dict('records')
        
        # Create snapshot
        snapshot = {
            "file_path": str(file_path),
            "file_size_bytes": file_size,
            "file_hash_sha256": file_hash,
            "row_count": row_count,
            "column_names": columns,
            "sample_rows": sample_rows,
            "timestamp": datetime.now().isoformat()
        }
        
        logging.info(f"Successfully processed {file_path}: {row_count} rows, {len(columns)} columns")
        return snapshot
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    """Main function to take snapshots of all input datasets"""
    logging.info("=== STARTING STEP 1B PHASE 2 SUB-PHASE 2.0: SETUP & SNAPSHOTS ===")
    
    # Define input datasets
    input_datasets = [
        {
            "path": "data/normalized/id_bridge.parquet",
            "type": "parquet",
            "description": "ID Bridge dataset"
        },
        {
            "path": "IMDB datasets/title.basics.tsv",
            "type": "tsv",
            "description": "IMDB Title Basics"
        },
        {
            "path": "IMDB datasets/title.ratings.tsv",
            "type": "tsv",
            "description": "IMDB Title Ratings"
        },
        {
            "path": "IMDB datasets/title.crew.tsv",
            "type": "tsv",
            "description": "IMDB Title Crew"
        },
        {
            "path": "movie-lens/links.csv",
            "type": "csv",
            "description": "MovieLens Links"
        },
        {
            "path": "Rotten Tomatoes/rotten_tomatoes_movies.csv",
            "type": "csv",
            "description": "Rotten Tomatoes Movies"
        },
        {
            "path": "Rotten Tomatoes/rotten_tomatoes_top_movies.csv",
            "type": "csv",
            "description": "Rotten Tomatoes Top Movies"
        },
        {
            "path": "Rotten Tomatoes/rotten_tomatoes_movie_reviews.csv",
            "type": "csv",
            "description": "Rotten Tomatoes Movie Reviews"
        }
    ]
    
    # Create output directory
    output_dir = Path("docs/step1b_phase2_inputs")
    output_dir.mkdir(exist_ok=True)
    
    # Take snapshots
    snapshots = []
    successful_count = 0
    
    for dataset in input_datasets:
        file_path = Path(dataset["path"])
        
        if not file_path.exists():
            logging.error(f"File not found: {file_path}")
            continue
            
        snapshot = take_dataset_snapshot(file_path, dataset["type"])
        
        if snapshot:
            snapshots.append(snapshot)
            successful_count += 1
            
            # Save individual snapshot
            filename = f"snapshot_{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            snapshot_file = output_dir / filename
            
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            
            logging.info(f"Saved snapshot to: {snapshot_file}")
    
    # Save combined snapshot
    combined_snapshot = {
        "phase": "2.0",
        "description": "Step 1b Phase 2 Input Snapshots",
        "timestamp": datetime.now().isoformat(),
        "total_datasets": len(input_datasets),
        "successful_snapshots": successful_count,
        "snapshots": snapshots
    }
    
    combined_file = output_dir / f"combined_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_file, 'w') as f:
        json.dump(combined_snapshot, f, indent=2, default=str)
    
    # Log summary
    logging.info("=== SNAPSHOT SUMMARY ===")
    for snapshot in snapshots:
        logging.info(f"{snapshot['file_path']}: {snapshot['row_count']} rows, hash: {snapshot['file_hash_sha256'][:16]}...")
    
    logging.info(f"Total snapshots taken: {successful_count}/{len(input_datasets)}")
    logging.info(f"Combined snapshot saved to: {combined_file}")
    logging.info("=== SUB-PHASE 2.0 COMPLETE ===")
    
    return successful_count == len(input_datasets)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


























