#!/usr/bin/env python3
"""
Fix canonical index by deduplicating the cleaned text data and creating a proper 88,194 row index.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def fix_88194_index():
    """Create a proper canonical index with 88,194 rows by deduplicating cleaned text data."""
    
    # Load cleaned text data
    print("Loading cleaned text data...")
    cleaned_df = pd.read_parquet('data/features/text/checkpoints/cleaned_text.parquet')
    print(f"Cleaned text data: {len(cleaned_df)} rows")
    print(f"Canonical ID unique: {cleaned_df['canonical_id'].is_unique}")
    print(f"Duplicates: {cleaned_df['canonical_id'].duplicated().sum()}")
    
    # Deduplicate by canonical_id, keeping the first occurrence
    print("Deduplicating by canonical_id...")
    dedup_df = cleaned_df.drop_duplicates(subset=['canonical_id'], keep='first')
    print(f"After deduplication: {len(dedup_df)} rows")
    print(f"Canonical ID unique: {dedup_df['canonical_id'].is_unique}")
    
    # Create canonical index
    canonical_index = pd.DataFrame({
        'canonical_id': dedup_df['canonical_id'].values,
        'row_index': np.arange(len(dedup_df))
    })
    
    print(f"Canonical index: {len(canonical_index)} rows")
    print(f"Canonical ID unique: {canonical_index['canonical_id'].is_unique}")
    
    # Save canonical index
    output_path = 'data/features/text/movies_canonical_index_88194.parquet'
    canonical_index.to_parquet(output_path, index=False)
    print(f"✓ Canonical index saved: {output_path}")
    
    # Also save as TF-IDF index for compatibility
    tfidf_index_path = 'data/features/text/movies_text_tfidf_index.parquet'
    canonical_index.to_parquet(tfidf_index_path, index=False)
    print(f"✓ TF-IDF index saved: {tfidf_index_path}")
    
    # Save deduplicated cleaned text
    cleaned_dedup_path = 'data/features/text/checkpoints/cleaned_text_dedup.parquet'
    dedup_df.to_parquet(cleaned_dedup_path, index=False)
    print(f"✓ Deduplicated cleaned text saved: {cleaned_dedup_path}")
    
    return canonical_index, dedup_df

if __name__ == "__main__":
    fix_88194_index()























