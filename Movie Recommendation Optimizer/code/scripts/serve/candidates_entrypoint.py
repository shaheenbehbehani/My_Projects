#!/usr/bin/env python3
"""
Step 3d.2 - Candidate Fetcher & Cache
Movie Recommendation Optimizer - Fast Candidate Assembly with Caching

This module provides a fast, deterministic candidate assembly layer backed by a cache
to support the scoring service with high-quality candidate pools and low latency.
"""

import os
import sys
import json
import time
import logging
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import OrderedDict
import pickle
import gzip
from datetime import datetime, timedelta
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class CandidateCache:
    """
    LRU cache with TTL and version pinning for candidate data.
    """
    
    def __init__(self, cache_dir: str = "data/runtime/cache", max_size: int = 1000, 
                 default_ttl_hours: int = 24):
        """Initialize the cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size = max_size
        self.default_ttl_hours = default_ttl_hours
        
        # In-memory LRU cache
        self.cache = OrderedDict()
        self.access_times = {}
        self.ttl_times = {}
        
        # Cache manifest
        self.manifest_path = self.cache_dir / "cache_manifest_3d2.json"
        self.manifest = self._load_manifest()
        
        self.logger = logging.getLogger("candidate_cache")
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load cache manifest."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache manifest: {e}")
        
        return {
            "version": "3d.2",
            "created_at": datetime.now().isoformat(),
            "namespaces": {},
            "total_objects": 0,
            "total_bytes": 0,
            "manifest_hash": ""
        }
    
    def _save_manifest(self):
        """Save cache manifest."""
        try:
            with open(self.manifest_path, 'w') as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache manifest: {e}")
    
    def _get_cache_key(self, namespace: str, key: str, version_hash: str) -> str:
        """Generate cache key with version."""
        return f"{namespace}:{key}|v:{version_hash[:8]}"
    
    def _is_expired(self, cache_key: str) -> bool:
        """Check if cache entry is expired."""
        if cache_key not in self.ttl_times:
            return True
        
        return datetime.now() > self.ttl_times[cache_key]
    
    def _evict_expired(self):
        """Remove expired entries."""
        expired_keys = [k for k in self.cache.keys() if self._is_expired(k)]
        for key in expired_keys:
            self._remove_key(key)
    
    def _evict_lru(self):
        """Remove least recently used entry."""
        if not self.cache:
            return
        
        # Remove oldest entry
        oldest_key = next(iter(self.cache))
        self._remove_key(oldest_key)
    
    def _remove_key(self, key: str):
        """Remove key from cache and update manifest."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.ttl_times[key]
            
            # Remove disk file
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")
            
            # Update manifest
            namespace = key.split(':')[0]
            if namespace in self.manifest["namespaces"]:
                self.manifest["namespaces"][namespace]["count"] -= 1
                self.manifest["total_objects"] -= 1
    
    def get(self, namespace: str, key: str, version_hash: str) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._get_cache_key(namespace, key, version_hash)
        
        # Check if expired
        if self._is_expired(cache_key):
            self._remove_key(cache_key)
            return None
        
        # Move to end (most recently used)
        if cache_key in self.cache:
            value = self.cache.pop(cache_key)
            self.cache[cache_key] = value
            self.access_times[cache_key] = time.time()
            return value
        
        # Try to load from disk
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                # Add to memory cache
                self.cache[cache_key] = value
                self.access_times[cache_key] = time.time()
                self.ttl_times[cache_key] = datetime.now() + timedelta(hours=self.default_ttl_hours)
                return value
            except Exception as e:
                self.logger.warning(f"Failed to load cache file {cache_file}: {e}")
                cache_file.unlink()  # Remove corrupted file
        
        return None
    
    def put(self, namespace: str, key: str, value: Any, version_hash: str, 
            ttl_hours: Optional[int] = None) -> str:
        """Put value in cache."""
        cache_key = self._get_cache_key(namespace, key, version_hash)
        
        # Remove if exists
        if cache_key in self.cache:
            self._remove_key(cache_key)
        
        # Evict if at capacity
        while len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Add to cache
        self.cache[cache_key] = value
        self.access_times[cache_key] = time.time()
        
        ttl = ttl_hours or self.default_ttl_hours
        self.ttl_times[cache_key] = datetime.now() + timedelta(hours=ttl)
        
        # Save to disk
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache file {cache_file}: {e}")
        
        # Update manifest
        if namespace not in self.manifest["namespaces"]:
            self.manifest["namespaces"][namespace] = {"count": 0, "bytes": 0}
        
        self.manifest["namespaces"][namespace]["count"] += 1
        self.manifest["total_objects"] += 1
        
        # Estimate bytes (rough approximation)
        try:
            size_bytes = len(pickle.dumps(value))
            self.manifest["namespaces"][namespace]["bytes"] += size_bytes
            self.manifest["total_bytes"] += size_bytes
        except:
            pass
        
        return cache_key
    
    def clear(self):
        """Clear all cache entries."""
        # Remove all disk files
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        self.cache.clear()
        self.access_times.clear()
        self.ttl_times.clear()
        self.manifest = {
            "version": "3d.2",
            "created_at": datetime.now().isoformat(),
            "namespaces": {},
            "total_objects": 0,
            "total_bytes": 0,
            "manifest_hash": ""
        }
        self._save_manifest()


class CandidateFetcher:
    """
    Fast, deterministic candidate assembly with caching.
    """
    
    def __init__(self, release_lock_path: str = "data/hybrid/release_lock_3d.json"):
        """Initialize the candidate fetcher."""
        self.logger = self._setup_logging()
        self.logger.info("Initializing CandidateFetcher")
        
        # Load release lock and policy
        self.release_lock = self._load_release_lock(release_lock_path)
        self.policy = self._load_policy()
        
        # Compute version hash
        self.version_hash = self._compute_version_hash()
        
        # Initialize cache
        self.cache = CandidateCache()
        
        # Load artifacts
        self._load_artifacts()
        
        # Initialize metrics
        self.metrics = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_latency_ms": 0,
            "fallback_stages": {"A": 0, "B": 0, "C": 0, "D": 0},
            "underfilled_requests": 0
        }
        
        self.logger.info("CandidateFetcher initialization completed")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the candidate fetcher."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("candidate_fetcher")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('logs/step3d2_cache.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _load_release_lock(self, path: str) -> Dict[str, Any]:
        """Load the release lock manifest."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load release lock: {e}")
            raise
    
    def _load_policy(self) -> Dict[str, Any]:
        """Load the provisional policy configuration."""
        policy_path = "data/hybrid/policy_provisional.json"
        try:
            with open(policy_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load policy: {e}")
            raise
    
    def _compute_version_hash(self) -> str:
        """Compute short hash of release lock for version pinning."""
        manifest_str = json.dumps(self.release_lock, sort_keys=True)
        return hashlib.sha256(manifest_str.encode()).hexdigest()[:8]
    
    def _load_artifacts(self):
        """Load all required artifacts."""
        self.logger.info("Loading artifacts...")
        
        artifacts = self.release_lock["artifacts"]
        
        # Load collaborative filtering artifacts
        self.user_factors = np.load(artifacts["step3b_collaborative"]["data/collaborative/user_factors_k20.npy"]["absolute_path"], mmap_mode='r')
        self.movie_factors = np.load(artifacts["step3b_collaborative"]["data/collaborative/movie_factors_k20.npy"]["absolute_path"], mmap_mode='r')
        
        # Load index mappings
        self.user_index_map = pd.read_parquet(artifacts["step3b_collaborative"]["data/collaborative/user_index_map.parquet"]["absolute_path"])
        self.movie_index_map = pd.read_parquet(artifacts["step3b_collaborative"]["data/collaborative/movie_index_map.parquet"]["absolute_path"])
        
        # Create lookup dictionaries
        self.user_id_to_idx = dict(zip(self.user_index_map['userId'], self.user_index_map['user_index']))
        self.movie_id_to_idx = dict(zip(self.movie_index_map['canonical_id'], self.movie_index_map['movie_index']))
        
        # Load content neighbors
        self.similarity_neighbors = pd.read_parquet(artifacts["step3a_content_based"]["data/similarity/movies_neighbors_k50.parquet"]["absolute_path"])
        
        # Create neighbor lookup
        self.neighbor_lookup = {}
        for _, row in self.similarity_neighbors.iterrows():
            movie_id = row['movie_id']
            if movie_id not in self.neighbor_lookup:
                self.neighbor_lookup[movie_id] = []
            self.neighbor_lookup[movie_id].append({
                'neighbor_id': row['neighbor_id'],
                'score': row['score'],
                'rank': row['rank']
            })
        
        # Load user activity
        self.user_activity = pd.read_parquet(artifacts["step3c_hybrid"]["data/derived/user_activity_snapshot.parquet"]["absolute_path"])
        self.user_activity_lookup = dict(zip(self.user_activity['user_index'], self.user_activity['ratings_count']))
        
        self.logger.info("All artifacts loaded successfully")
    
    def _canonicalize_filter_signature(self, genres: List[str] = None, 
                                      providers: List[str] = None,
                                      year_range: Tuple[int, int] = None) -> str:
        """Canonicalize filter signature for cache keys."""
        # Sort and join genres
        genre_str = ",".join(sorted(genres)) if genres else ""
        
        # Sort and join providers
        provider_str = ",".join(sorted(providers)) if providers else ""
        
        # Format year range
        year_str = f"{year_range[0]}-{year_range[1]}" if year_range else ""
        
        return f"F:{genre_str}|{provider_str}|{year_str}"
    
    def _get_user_bucket(self, user_id: str) -> str:
        """Determine user bucket based on rating count."""
        try:
            user_idx = self.user_id_to_idx.get(user_id)
            if user_idx is None:
                return 'cold'
            
            ratings_count = self.user_activity_lookup.get(user_idx, 0)
            
            if ratings_count <= self.policy["bucket_thresholds"]["cold"]["max_ratings"]:
                return 'cold'
            elif ratings_count <= self.policy["bucket_thresholds"]["light"]["max_ratings"]:
                return 'light'
            elif ratings_count <= self.policy["bucket_thresholds"]["medium"]["max_ratings"]:
                return 'medium'
            else:
                return 'heavy'
        except Exception as e:
            self.logger.warning(f"Error determining user bucket for {user_id}: {e}")
            return 'cold'
    
    def _get_cf_seeds(self, user_id: str, target_count: int = 800) -> List[str]:
        """Get CF seeds for user."""
        try:
            user_idx = self.user_id_to_idx.get(user_id)
            if user_idx is None or user_idx >= self.user_factors.shape[0]:
                return []
            
            user_vector = self.user_factors[user_idx]
            
            # Compute scores for all movies
            scores = np.dot(user_vector, self.movie_factors.T)
            
            # Get top movies
            top_indices = np.argsort(scores)[::-1][:target_count]
            
            # Convert to movie IDs
            movie_ids = []
            for idx in top_indices:
                if idx < len(self.movie_index_map):
                    movie_id = self.movie_index_map.iloc[idx]['canonical_id']
                    movie_ids.append(movie_id)
            
            return movie_ids
            
        except Exception as e:
            self.logger.warning(f"Error getting CF seeds for user {user_id}: {e}")
            return []
    
    def _expand_content_neighbors(self, seed_movies: List[str], k: int = 8) -> List[str]:
        """Expand content neighbors for seed movies."""
        neighbors = set()
        
        for movie_id in seed_movies:
            movie_neighbors = self.neighbor_lookup.get(movie_id, [])
            top_neighbors = sorted(movie_neighbors, key=lambda x: x['rank'])[:k]
            
            for neighbor in top_neighbors:
                neighbors.add(neighbor['neighbor_id'])
        
        return list(neighbors)
    
    def _apply_filters(self, candidates: List[str], genres: List[str] = None,
                      providers: List[str] = None, year_range: Tuple[int, int] = None) -> List[str]:
        """Apply filters to candidates."""
        # For now, return all candidates (filtering would require additional data)
        # In a real implementation, this would filter based on movie metadata
        return candidates
    
    def _get_popularity_candidates(self, genres: List[str] = None,
                                  providers: List[str] = None, 
                                  count: int = 100) -> List[str]:
        """Get popularity-based candidates for fallback."""
        # For now, return top movies from movie index
        # In a real implementation, this would use popularity data
        top_movies = self.movie_index_map['canonical_id'].head(count).tolist()
        return top_movies
    
    def get_candidates(self, user_id: str, K: int = 50, 
                      genres: List[str] = None, providers: List[str] = None,
                      year_range: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Get candidates for user with caching and fallback chain.
        
        Args:
            user_id: User identifier
            K: Number of candidates needed
            genres: List of genre filters
            providers: List of provider filters  
            year_range: (min_year, max_year) tuple
            
        Returns:
            Dictionary with candidates and metadata
        """
        start_time = time.time()
        self.metrics["requests"] += 1
        
        # Create filter signature
        filter_sig = self._canonicalize_filter_signature(genres, providers, year_range)
        
        # Check cache first
        cache_key = f"{user_id}|{filter_sig}"
        cached_result = self.cache.get("U", cache_key, self.version_hash)
        
        if cached_result is not None:
            self.metrics["cache_hits"] += 1
            self.logger.info(f"Cache hit for user {user_id}, filter {filter_sig}")
            # Update the cache_hit flag in the cached result
            cached_result["metadata"]["cache_hit"] = True
            return cached_result
        
        self.metrics["cache_misses"] += 1
        self.logger.info(f"Cache miss for user {user_id}, filter {filter_sig}")
        
        # Assemble candidates using fallback chain
        candidates = []
        provenance = []
        fallback_stage = "A"
        
        # Stage A: CF Seeds
        try:
            cf_seeds = self._get_cf_seeds(user_id, target_count=800)
            if cf_seeds:
                candidates.extend(cf_seeds)
                provenance.extend(["cf_seed"] * len(cf_seeds))
                self.metrics["fallback_stages"]["A"] += 1
                fallback_stage = "B"
        except Exception as e:
            self.logger.warning(f"CF seeds failed for user {user_id}: {e}")
        
        # Stage B: Content Expansion
        try:
            if candidates:
                content_neighbors = self._expand_content_neighbors(cf_seeds, k=8)
                candidates.extend(content_neighbors)
                provenance.extend(["content_neighbor"] * len(content_neighbors))
                self.metrics["fallback_stages"]["B"] += 1
                fallback_stage = "C"
        except Exception as e:
            self.logger.warning(f"Content expansion failed for user {user_id}: {e}")
        
        # Stage C: Filter-Aware Pruning
        try:
            filtered_candidates = self._apply_filters(candidates, genres, providers, year_range)
            if len(filtered_candidates) >= int(1.2 * K):
                candidates = filtered_candidates
                provenance = ["filtered"] * len(candidates)
                self.metrics["fallback_stages"]["C"] += 1
                fallback_stage = "D"
        except Exception as e:
            self.logger.warning(f"Filtering failed for user {user_id}: {e}")
        
        # Stage D: Popularity Backfill
        if len(candidates) < K:
            try:
                popularity_candidates = self._get_popularity_candidates(genres, providers, K)
                candidates.extend(popularity_candidates)
                provenance.extend(["popularity"] * len(popularity_candidates))
                self.metrics["fallback_stages"]["D"] += 1
            except Exception as e:
                self.logger.warning(f"Popularity backfill failed for user {user_id}: {e}")
        
        # Deduplicate while preserving order
        seen = set()
        deduped_candidates = []
        deduped_provenance = []
        
        for candidate, prov in zip(candidates, provenance):
            if candidate not in seen:
                seen.add(candidate)
                deduped_candidates.append(candidate)
                deduped_provenance.append(prov)
        
        # Take top K
        final_candidates = deduped_candidates[:K]
        final_provenance = deduped_provenance[:K]
        
        # Check if underfilled
        if len(final_candidates) < K:
            self.metrics["underfilled_requests"] += 1
        
        # Prepare result
        result = {
            "user_id": user_id,
            "candidates": final_candidates,
            "provenance": final_provenance,
            "metadata": {
                "total_candidates": len(final_candidates),
                "k_requested": K,
                "k_returned": len(final_candidates),
                "filter_signature": filter_sig,
                "fallback_stage": fallback_stage,
                "user_bucket": self._get_user_bucket(user_id),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "cache_hit": False
            }
        }
        
        # Cache result
        self.cache.put("U", cache_key, result, self.version_hash)
        
        # Update metrics
        self.metrics["total_latency_ms"] += result["metadata"]["processing_time_ms"]
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache.cache),
            "max_size": self.cache.max_size,
            "hit_ratio": self.metrics["cache_hits"] / max(self.metrics["requests"], 1),
            "total_requests": self.metrics["requests"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "avg_latency_ms": self.metrics["total_latency_ms"] / max(self.metrics["requests"], 1),
            "fallback_stages": self.metrics["fallback_stages"],
            "underfilled_requests": self.metrics["underfilled_requests"],
            "version_hash": self.version_hash
        }
    
    def warm_cache(self, warm_list_path: str = "data/runtime/cache_warm_list_3d2.csv"):
        """Warm the cache with pre-computed targets."""
        self.logger.info("Starting cache warm process...")
        
        if not os.path.exists(warm_list_path):
            self.logger.warning(f"Warm list not found: {warm_list_path}")
            return
        
        warm_start = time.time()
        warm_count = 0
        
        try:
            warm_df = pd.read_csv(warm_list_path)
            
            for _, row in warm_df.iterrows():
                user_id = str(row['user_id'])
                genres = eval(row['genres']) if pd.notna(row['genres']) else None
                providers = eval(row['providers']) if pd.notna(row['providers']) else None
                year_range = eval(row['year_range']) if pd.notna(row['year_range']) else None
                K = int(row['K'])
                
                # Get candidates (will be cached)
                self.get_candidates(user_id, K, genres, providers, year_range)
                warm_count += 1
                
                if warm_count % 100 == 0:
                    self.logger.info(f"Warmed {warm_count} entries...")
        
        except Exception as e:
            self.logger.error(f"Cache warm failed: {e}")
        
        warm_duration = time.time() - warm_start
        self.logger.info(f"Cache warm completed: {warm_count} entries in {warm_duration:.2f}s")
        
        return {
            "warm_count": warm_count,
            "warm_duration_seconds": warm_duration,
            "entries_per_second": warm_count / max(warm_duration, 0.001)
        }


def main():
    """CLI entrypoint for the candidate fetcher."""
    parser = argparse.ArgumentParser(description='Movie Recommendation Candidate Fetcher')
    parser.add_argument('--user-id', required=True, help='User ID to get candidates for')
    parser.add_argument('--k', type=int, default=50, help='Number of candidates (default: 50)')
    parser.add_argument('--genres', nargs='*', help='Genre filters')
    parser.add_argument('--providers', nargs='*', help='Provider filters')
    parser.add_argument('--year-min', type=int, help='Minimum year filter')
    parser.add_argument('--year-max', type=int, help='Maximum year filter')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    parser.add_argument('--warm', action='store_true', help='Warm the cache before processing')
    parser.add_argument('--stats', action='store_true', help='Show cache statistics')
    
    args = parser.parse_args()
    
    try:
        # Initialize fetcher
        fetcher = CandidateFetcher()
        
        # Warm cache if requested
        if args.warm:
            fetcher.warm_cache()
        
        # Show stats if requested
        if args.stats:
            stats = fetcher.get_cache_stats()
            print(json.dumps(stats, indent=2))
            return
        
        # Prepare filters
        year_range = None
        if args.year_min is not None and args.year_max is not None:
            year_range = (args.year_min, args.year_max)
        
        # Get candidates
        results = fetcher.get_candidates(
            args.user_id, 
            args.k, 
            args.genres, 
            args.providers, 
            year_range
        )
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(results, indent=2))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
