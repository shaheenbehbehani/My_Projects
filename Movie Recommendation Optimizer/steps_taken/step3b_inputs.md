# Step 3b.1: Ratings Matrix Assembly

## Overview
This document describes the assembly of the user-movie ratings matrix from MovieLens data, aligned with the canonical_id system.

## Input Data Sources
- MovieLens ratings.csv: 31,921,467 ratings after filtering
- MovieLens links.csv: Mapping from movieId to imdbId
- resolved_links.parquet: Mapping from tconst to canonical_id
- movies_master.parquet: Master dataset with valid canonical_ids

## Filters Applied
- Users with < 3 ratings: Removed
- Movies with < 5 ratings: Removed
- Movies not in master dataset: Removed
- Missing canonical_ids: Removed

## Final Statistics
- **Users**: 200,948
- **Movies**: 43,884
- **Ratings**: 31,921,467
- **Matrix Density**: 0.003620
- **Matrix Sparsity**: 0.996380

## Rating Scale
- **Range**: 0.5 - 5.0
- **Distribution**:
rating
0.5     520073
1.0     942495
1.5     527262
2.0    2021228
2.5    1675301
3.0    6038416
3.5    4276865
4.0    8357920
4.5    2970665
5.0    4591242

## User Activity Statistics
- **Min ratings per user**: 15
- **Max ratings per user**: 25,642
- **Mean ratings per user**: 158.9
- **Median ratings per user**: 73.0

## Movie Popularity Statistics
- **Min ratings per movie**: 5
- **Max ratings per movie**: 102,929
- **Mean ratings per movie**: 727.4
- **Median ratings per movie**: 23.0

## Output Files
- `data/collaborative/ratings_matrix_csr.npz`: Sparse ratings matrix in CSR format
- `data/collaborative/ratings_long_format.parquet`: Ratings in long format
- `data/collaborative/user_index_map.parquet`: User ID to sequential index mapping
- `data/collaborative/movie_index_map.parquet`: Canonical ID to sequential index mapping

## Quality Checks
- ✅ All canonical_ids align with master dataset
- ✅ No NaN/Inf values in ratings
- ✅ Rating scale validated (0.5-5.0)
- ✅ Sparse matrix and parquet outputs created
- ✅ Sequential indexing implemented
- ✅ Spot checks passed

## Timestamp
Generated on: 2025-09-02 09:07:38
