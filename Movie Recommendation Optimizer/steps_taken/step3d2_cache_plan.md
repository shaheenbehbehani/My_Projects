# Step 3d.2 - Candidate Fetcher & Cache Design

## Overview
This document outlines the design and implementation of the candidate fetcher and cache system for the Movie Recommendation Optimizer. The system provides fast, deterministic candidate assembly with intelligent caching to support the scoring service.

## Cache Architecture

### Key Spaces
The cache is organized into three main key spaces:

#### U: User-Centric Pools
- **Format**: `U:{user_id}|F:{filter_sig}|v:{manifest_hash_short}`
- **Purpose**: Store pre-computed candidate pools for specific users with specific filter combinations
- **TTL**: 24 hours
- **Example**: `U:12345|F:action,comedy|v:a1b2c3`

#### M: Item Neighbors
- **Format**: `M:{movie_id}|N:{n}|v:{manifest_hash_short}`
- **Purpose**: Store content-based neighbor lists for movies
- **TTL**: 7 days (more stable than user preferences)
- **Example**: `M:tt0114709|N:10|v:a1b2c3`

#### G: Group/Popularity Shards
- **Format**: `G:{group_type}|F:{filter_sig}|v:{manifest_hash_short}`
- **Purpose**: Store popularity-based candidate pools for fallback scenarios
- **TTL**: 12 hours
- **Examples**: 
  - `G:genre_provider|F:action,netflix|v:a1b2c3`
  - `G:provider_only|F:netflix|v:a1b2c3`
  - `G:global|F:|v:a1b2c3`

### Cache Configuration
- **Storage**: Compressed Parquet files in `data/runtime/cache/`
- **Eviction**: LRU with TTL-based expiration
- **Version Pinning**: All keys include manifest hash for automatic invalidation
- **Serialization**: Stable format with compression
- **Memory Budget**: 2GB maximum (configurable)

## Fallback Chain

### Stage A - CF Seeds
- **Target**: ≥800 unique movie IDs from collaborative filtering
- **Source**: User factors × movie factors dot products
- **Fallback**: If no CF factors available, skip to Stage B

### Stage B - Content Expansion
- **Process**: For each CF seed, pull top-k content neighbors (k=5-10)
- **Union**: Combine all neighbor lists
- **Cap**: Limit total pool to 1,500-2,000 items
- **Fallback**: If insufficient content neighbors, proceed to Stage C

### Stage C - Filter-Aware Pruning
- **Filters Applied**: Providers, genres, year range
- **Relaxation Order**:
  1. Year range (if pool < 1.2×K)
  2. Genre filters (minimal relaxation)
  3. Never relax provider filters
- **Target**: Maintain pool size ≥ 1.2×K

### Stage D - Popularity Backfill
- **Order** (strict):
  1. Genre×provider popularity
  2. Provider-only popularity  
  3. Global popularity
- **Purpose**: Ensure minimum K candidates

## Filter Signature Canonicalization

### Format
```
F:{sorted_genres}|{sorted_providers}|{year_min}-{year_max}
```

### Examples
- `F:action,comedy|netflix,hulu|2010-2020`
- `F:|netflix|`
- `F:action|hulu|2015-2025`

### Rules
- Genres: Alphabetically sorted, comma-separated
- Providers: Alphabetically sorted, comma-separated
- Year range: `{min}-{max}` or empty for no filter
- Empty filters: Use empty string, not "all"

## Cache Warm Plan

### Target Coverage
1. **Top 1,000 Most Active Users**: Based on rating count from user activity snapshot
2. **Top 50 Genre×Provider Combinations**: From Step 3c evaluation scenarios
3. **500 Most Popular Movies**: For neighbor cache pre-warming

### Warm Process
1. Load user activity data and identify top users
2. Generate filter combinations from evaluation scenarios
3. Pre-compute and cache all target combinations
4. Record warm duration, object counts, and memory usage

## Metrics & Telemetry

### Per-Request Metrics
- Cache hit/miss status
- Assembly latency (ms)
- Pool size before/after filtering
- Fallback stage reached
- Final list length
- Memory footprint snapshot

### Aggregate Metrics
- Cache hit ratio
- P50/P95/P99 latencies
- % requests needing fallback
- % underfilled lists (< K candidates)
- Memory usage trends

### Logging
- **File**: `logs/step3d2_cache.log`
- **Format**: Structured JSON with timestamps
- **Rotation**: Daily with compression
- **Retention**: 30 days

## Acceptance Criteria

### Coverage
- ≥99% of 1,000-request mixed workload return ≥K items
- Test across user buckets and filter signatures

### Latency
- P95 candidate fetch ≤20ms on warm cache
- P95 candidate fetch ≤150ms on cold cache
- Single-threaded, local execution

### Quality
- Identical top-K outputs compared to direct assembly
- Within tie-breaking rules
- Deterministic ordering

### Determinism
- Same inputs + manifest → byte-identical candidate IDs
- Order preserved across runs
- Fixed random seed for any randomness

### Cache Efficacy
- Hit ratio ≥0.85 after warm
- Memory budget documented and respected
- Version invalidation working correctly

## Implementation Details

### Dependencies
- **Release Lock**: `data/hybrid/release_lock_3d.json`
- **Policy**: `data/hybrid/policy_provisional.json`
- **CF Factors**: `data/collaborative/user_factors_k20.npy`, `data/collaborative/movie_factors_k20.npy`
- **Content Neighbors**: `data/similarity/movies_neighbors_k50.parquet`
- **User Activity**: `data/derived/user_activity_snapshot.parquet`

### Error Handling
- Graceful degradation for missing artifacts
- Fallback to popularity-based candidates
- Comprehensive error logging
- No service interruption on cache failures

### Security Considerations
- Input validation for all parameters
- Rate limiting for cache operations
- Secure file permissions for cache storage
- No sensitive data in cache keys

## Future Optimizations

### Potential Improvements
1. **Distributed Caching**: Redis/Memcached for multi-instance deployments
2. **Predictive Warming**: ML-based cache pre-warming
3. **Compression**: Advanced compression algorithms
4. **Partitioning**: Shard cache by user segments
5. **Metrics**: Real-time monitoring dashboard

### Scalability Considerations
- Horizontal scaling with consistent hashing
- Cache warming strategies for new deployments
- Memory usage monitoring and alerting
- Performance regression detection






