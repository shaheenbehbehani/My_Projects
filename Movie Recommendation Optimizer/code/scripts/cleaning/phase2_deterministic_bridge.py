#!/usr/bin/env python3
"""
Step 1b Phase 2 - Sub-phase 2.1: Deterministic Cross-Source Bridge
Builds a deterministic bridge using MovieLens links.csv as the only mapping signal
HOTFIX 2.1-A: Optimized with pre-indexed Series and map lookups
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/step1b_phase2.log', mode='a'),
        logging.StreamHandler()
    ]
)

def imdb_to_tconst(x):
    """Convert imdbId to tconst format: 'tt' + zero-padded to 7+ digits"""
    if pd.isna(x):
        return pd.NA
    try:
        i = int(x)
        return "tt" + str(i).zfill(7)
    except Exception:
        return pd.NA

def main():
    """Main function for Sub-phase 2.1"""
    logging.info("=== STARTING STEP 1B PHASE 2 SUB-PHASE 2.1: DETERMINISTIC BRIDGE (HOTFIX 2.1-A) ===")
    
    # Define paths
    links_path = "movie-lens/links.csv"
    imdb_basics_path = "IMDB datasets/title.basics.tsv"
    output_path = "data/normalized/bridges/checkpoints/linked_deterministic.parquet"
    conflicts_path = "data/normalized/bridges/checkpoints/linked_deterministic_conflicts.parquet"
    unresolved_path = "data/normalized/bridges/checkpoints/linked_deterministic_unresolved.parquet"
    
    # Check if output exists and remove for fresh run
    if os.path.exists(output_path):
        os.remove(output_path)
        logging.info(f"Removed existing output: {output_path}")
    
    # Preload minimal IMDb basics table with only needed columns
    logging.info("Loading minimal IMDb basics table...")
    imdb_cols = ["tconst", "primaryTitle", "originalTitle", "titleType", "startYear"]
    imdb_dtypes = {
        "tconst": "string",
        "primaryTitle": "string",
        "originalTitle": "string",
        "titleType": "string",
        "startYear": "string",  # will coerce to Int32 later
    }
    
    basics = pd.read_csv(
        imdb_basics_path,
        sep="\t",
        usecols=imdb_cols,
        dtype=imdb_dtypes,
        na_values="\\N",
        low_memory=False,
    )
    
    # Coerce startYear to numeric and drop duplicates
    basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce").astype("Int32")
    basics = basics.drop_duplicates(subset=["tconst"])
    basics = basics.set_index("tconst", drop=True)
    
    # Create Series for fast map lookups
    s_primary = basics["primaryTitle"]
    s_original = basics["originalTitle"]
    s_type = basics["titleType"]
    s_year = basics["startYear"]
    
    del basics  # free memory; we keep the Series objects
    logging.info(f"IMDb basics indexed: {len(s_primary):,} titles")
    
    # Read MovieLens links with proper dtypes
    logging.info(f"Reading MovieLens links from: {links_path}")
    links = pd.read_csv(links_path, dtype={"movieId": "Int64", "imdbId": "Int64", "tmdbId": "Int64"})
    total = len(links)
    logging.info(f"Total rows in links.csv: {total:,}")
    
    # Count non-null imdbId
    non_null_imdb = links['imdbId'].notna().sum()
    logging.info(f"Rows with non-null imdbId: {non_null_imdb:,}")
    
    # Process links in 20k chunks
    chunk_size = 20_000
    outputs = []
    
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = links.iloc[start:end].copy()
        logging.info(f"Processing rows {start:,}–{end:,} / {total:,}")
        
        # Add heartbeat logging every 2,000 rows within the chunk
        for i in range(0, len(chunk), 2_000):
            logging.info(f"… chunk offset {i:,}/{len(chunk):,}")
        
        # Convert imdbId to tconst
        chunk["imdbId_raw"] = chunk["imdbId"]
        chunk["tconst"] = chunk["imdbId_raw"].map(imdb_to_tconst).astype("string")
        
        # Map lookups from pre-indexed Series (vectorized, fast)
        chunk["primaryTitle"] = chunk["tconst"].map(s_primary)
        chunk["originalTitle"] = chunk["tconst"].map(s_original)
        chunk["titleType"] = chunk["tconst"].map(s_type)
        chunk["year"] = chunk["tconst"].map(s_year)
        
        # Choose title_source and normalize title
        use_primary = chunk["primaryTitle"].notna()
        chunk["title_source"] = np.where(
            use_primary, "imdb_primaryTitle",
            np.where(chunk["originalTitle"].notna(), "imdb_originalTitle", pd.NA)
        )
        
        title_for_norm = chunk["primaryTitle"].fillna(chunk["originalTitle"])
        chunk["title_norm"] = (
            title_for_norm
            .astype("string")
            .str.lower()
            .str.normalize("NFKC")
            .str.replace(r"[^0-9a-z]+", " ", regex=True)
            .str.strip()
            .replace({"": pd.NA})
        )
        
        # Fixed fields
        chunk["link_method"] = "deterministic_links"
        chunk["match_score"] = pd.Series([pd.NA] * len(chunk), dtype="Float32")
        chunk["source_ml"] = True
        chunk["source_imdb"] = chunk["tconst"].notna()
        chunk["source_rt"] = False
        
        # canonical_id preference
        chunk["canonical_id"] = np.where(
            chunk["tconst"].notna(), chunk["tconst"],
            np.where(
                chunk["tmdbId"].notna(), 
                "tmdb:" + chunk["tmdbId"].astype("Int64").astype("string"),
                "ml:" + chunk["movieId"].astype("Int64").astype("string")
            )
        )
        
        # Project to required schema & dtypes
        out = chunk[[
            "movieId", "imdbId_raw", "tconst", "tmdbId", "title_norm", "year",
            "title_source", "link_method", "match_score", "source_ml", "source_imdb", "source_rt", "canonical_id"
        ]].copy()
        
        # Enforce output schema dtypes
        out = out.astype({
            "movieId": "Int64",
            "imdbId_raw": "Int64",
            "tconst": "string",
            "tmdbId": "Int64",
            "title_norm": "string",
            "year": "Int32",
            "title_source": "string",
            "link_method": "string",
            "match_score": "Float32",
            "source_ml": "boolean",
            "source_imdb": "boolean",
            "source_rt": "boolean",
            "canonical_id": "string",
        }, errors="ignore")
        
        outputs.append(out)
    
    # Concatenate all outputs
    logging.info("Concatenating chunk results...")
    final = pd.concat(outputs, ignore_index=True)
    
    # Drop exact duplicates
    before = len(final)
    final = final.drop_duplicates()
    dropped = before - len(final)
    logging.info(f"Duplicate rows dropped: {dropped:,}")
    
    # Handle conflicts (same movieId mapping to multiple non-null tconst values)
    dup_keys = final.dropna(subset=["tconst"]).duplicated(subset=["movieId", "tconst"], keep="first")
    conflicts = final[dup_keys]
    
    if len(conflicts):
        conflicts.to_parquet(conflicts_path, index=False)
        logging.info(f"Conflicts written to: {conflicts_path} ({len(conflicts):,} rows)")
    else:
        logging.info("No conflicts found")
    
    # Identify unresolved rows (no tconst and no tmdbId)
    unresolved = final[final["tconst"].isna() & final["tmdbId"].isna()]
    
    if len(unresolved):
        unresolved.to_parquet(unresolved_path, index=False)
        logging.info(f"Unresolved rows written to: {unresolved_path} ({len(unresolved):,} rows)")
    else:
        logging.info("No unresolved rows found")
    
    # Save main output
    final.to_parquet(output_path, index=False)
    logging.info(f"Main output saved to: {output_path} ({len(final):,} rows)")
    
    # QA counts and analysis
    logging.info("=== QA ANALYSIS ===")
    
    # tconst conversion stats
    tconst_converted = final['tconst'].notna().sum()
    logging.info(f"Rows successfully converted to tconst: {tconst_converted:,}")
    
    # Join hit rate
    join_hits = final['source_imdb'].sum()
    join_rate = (join_hits / len(final)) * 100 if len(final) > 0 else 0
    logging.info(f"Join hit-rate to title.basics.tsv: {join_hits:,}/{len(final):,} ({join_rate:.1f}%)")
    
    # Canonical ID type counts
    tconst_count = sum(1 for x in final['canonical_id'] if x and not x.startswith('tmdb:') and not x.startswith('ml:'))
    tmdb_count = sum(1 for x in final['canonical_id'] if x and x.startswith('tmdb:'))
    ml_count = sum(1 for x in final['canonical_id'] if x and x.startswith('ml:'))
    
    logging.info(f"Canonical ID types: tconst={tconst_count:,}, tmdb:{tmdb_count:,}, ml:{ml_count:,}")
    
    # Log sample rows
    logging.info("=== 5 RANDOM SAMPLE ROWS ===")
    sample_rows = final.sample(n=min(5, len(final)), random_state=42)
    for i, (_, row) in enumerate(sample_rows.iterrows()):
        logging.info(f"Sample {i+1}:")
        for col in final.columns:
            value = row[col]
            if pd.isna(value):
                value = "null"
            logging.info(f"  {col}: {value}")
        logging.info("")
    
    # Final summary
    logging.info("=== SUB-PHASE 2.1 COMPLETE (HOTFIX 2.1-A) ===")
    logging.info(f"Final output: {len(final):,} rows")
    logging.info(f"Conflicts: {len(conflicts):,} rows")
    logging.info(f"Unresolved: {len(unresolved):,} rows")
    logging.info(f"Duplicates dropped: {dropped:,} rows")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
