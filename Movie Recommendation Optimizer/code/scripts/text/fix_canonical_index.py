#!/usr/bin/env python3
"""
Fix canonical index by creating a proper unique index from the master dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def fix_canonical_index():
    """Create a proper canonical index from the master dataset."""
    
    # Load master dataset
    print("Loading master dataset...")
    master_df = pd.read_parquet('data/normalized/movies_master.parquet')
    print(f"Master dataset: {len(master_df)} rows")
    
    # Verify canonical_id is unique in master
    if not master_df['canonical_id'].is_unique:
        print("ERROR: Master dataset has duplicate canonical_ids")
        return
    
    # Create canonical index
    canonical_index = pd.DataFrame({
        'canonical_id': master_df['canonical_id'].values,
        'row_index': np.arange(len(master_df))
    })
    
    print(f"Canonical index: {len(canonical_index)} rows")
    print(f"Canonical ID unique: {canonical_index['canonical_id'].is_unique}")
    
    # Save canonical index
    output_path = 'data/features/text/movies_canonical_index.parquet'
    canonical_index.to_parquet(output_path, index=False)
    print(f"✓ Canonical index saved: {output_path}")
    
    # Also save as TF-IDF index for compatibility
    tfidf_index_path = 'data/features/text/movies_text_tfidf_index.parquet'
    canonical_index.to_parquet(tfidf_index_path, index=False)
    print(f"✓ TF-IDF index saved: {tfidf_index_path}")
    
    return canonical_index

if __name__ == "__main__":
    fix_canonical_index()























