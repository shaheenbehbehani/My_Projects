#!/usr/bin/env python3
"""
Create a proper 88,194 row index that works with existing TF-IDF and BERT matrices.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_88194_index():
    """Create a proper 88,194 row index that works with existing matrices."""
    
    # Load cleaned text data
    print("Loading cleaned text data...")
    cleaned_df = pd.read_parquet('data/features/text/checkpoints/cleaned_text.parquet')
    print(f"Cleaned text data: {len(cleaned_df)} rows")
    print(f"Canonical ID unique: {cleaned_df['canonical_id'].is_unique}")
    print(f"Duplicates: {cleaned_df['canonical_id'].duplicated().sum()}")
    
    # Create a mapping that preserves the 88,194 rows but maps to unique canonical_ids
    # For duplicates, we'll keep the first occurrence and map subsequent ones to the same canonical_id
    print("Creating 88,194 row index...")
    
    # Get unique canonical_ids in order of first appearance
    unique_canonical_ids = []
    seen = set()
    for canonical_id in cleaned_df['canonical_id']:
        if canonical_id not in seen:
            unique_canonical_ids.append(canonical_id)
            seen.add(canonical_id)
    
    print(f"Unique canonical IDs: {len(unique_canonical_ids)}")
    
    # Create the index with 88,194 rows
    # For each row, use the canonical_id from the cleaned data
    # This preserves the row order that was used to create the TF-IDF and BERT matrices
    canonical_index = pd.DataFrame({
        'canonical_id': cleaned_df['canonical_id'].values,
        'row_index': np.arange(len(cleaned_df))
    })
    
    print(f"Canonical index: {len(canonical_index)} rows")
    print(f"Canonical ID unique: {canonical_index['canonical_id'].is_unique}")
    
    # Note: This index will have duplicate canonical_ids, but that's what the matrices expect
    # The important thing is that row_index maps correctly to the matrix rows
    
    # Save canonical index
    output_path = 'data/features/text/movies_text_tfidf_index.parquet'
    canonical_index.to_parquet(output_path, index=False)
    print(f"✓ TF-IDF index saved: {output_path}")
    
    # Also save a summary of the duplicates for reference
    duplicate_summary = cleaned_df.groupby('canonical_id').size().reset_index(name='count')
    duplicate_summary = duplicate_summary[duplicate_summary['count'] > 1].sort_values('count', ascending=False)
    
    summary_path = 'data/features/text/duplicate_summary.csv'
    duplicate_summary.to_csv(summary_path, index=False)
    print(f"✓ Duplicate summary saved: {summary_path}")
    print(f"  Movies with duplicates: {len(duplicate_summary)}")
    print(f"  Total duplicate rows: {duplicate_summary['count'].sum() - len(duplicate_summary)}")
    
    return canonical_index

if __name__ == "__main__":
    create_88194_index()























