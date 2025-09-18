#!/usr/bin/env python3
"""
Step 3d.1 - Scoring Service (Stateless recommend())
Movie Recommendation Optimizer - Production Scoring Service

This module provides a stateless scoring service that generates top-K recommendations
for users using the hybrid scoring approach from Step 3c, with policy-based alpha
selection and robust error handling.
"""

import os
import sys
import json
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from scipy.stats import rankdata
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class MovieRecommendationScorer:
    """
    Stateless movie recommendation scorer implementing hybrid scoring logic.
    
    This class loads all required artifacts from the release lock and provides
    a deterministic recommend() function that matches offline scoring results.
    """
    
    def __init__(self, release_lock_path: str = "data/hybrid/release_lock_3d.json"):
        """Initialize the scorer with artifacts from release lock."""
        self.logger = self._setup_logging()
        self.logger.info("Initializing MovieRecommendationScorer")
        
        # Load release lock and policy
        self.release_lock = self._load_release_lock(release_lock_path)
        self.policy = self._load_policy()
        
        # Initialize artifact paths
        self._init_artifact_paths()
        
        # Load all required artifacts
        self._load_artifacts()
        
        # Initialize scoring parameters
        self._init_scoring_params()
        
        self.logger.info("Scorer initialization completed successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the scoring service."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("scorer")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('logs/step3d1_scorer.log')
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
    
    def _init_artifact_paths(self):
        """Initialize artifact paths from release lock."""
        artifacts = self.release_lock["artifacts"]
        
        # Step 3a - Content-based artifacts
        self.content_embedding_path = artifacts["step3a_content_based"]["data/features/composite/movies_embedding_v1.npy"]["absolute_path"]
        self.similarity_neighbors_path = artifacts["step3a_content_based"]["data/similarity/movies_neighbors_k50.parquet"]["absolute_path"]
        
        # Step 3b - Collaborative filtering artifacts
        self.user_factors_path = artifacts["step3b_collaborative"]["data/collaborative/user_factors_k20.npy"]["absolute_path"]
        self.movie_factors_path = artifacts["step3b_collaborative"]["data/collaborative/movie_factors_k20.npy"]["absolute_path"]
        self.user_index_map_path = artifacts["step3b_collaborative"]["data/collaborative/user_index_map.parquet"]["absolute_path"]
        self.movie_index_map_path = artifacts["step3b_collaborative"]["data/collaborative/movie_index_map.parquet"]["absolute_path"]
        
        # Step 3c - Hybrid artifacts
        self.candidates_dir = "data/hybrid/candidates"
        self.user_activity_path = artifacts["step3c_hybrid"]["data/derived/user_activity_snapshot.parquet"]["absolute_path"]
    
    def _load_artifacts(self):
        """Load all required artifacts into memory."""
        self.logger.info("Loading artifacts...")
        
        # Load collaborative filtering artifacts
        self.logger.info("Loading collaborative filtering artifacts...")
        self.user_factors = np.load(self.user_factors_path, mmap_mode='r')
        self.movie_factors = np.load(self.movie_factors_path, mmap_mode='r')
        
        # Load index mappings
        self.user_index_map = pd.read_parquet(self.user_index_map_path)
        self.movie_index_map = pd.read_parquet(self.movie_index_map_path)
        
        # Create lookup dictionaries for faster access
        self.user_id_to_idx = dict(zip(self.user_index_map['userId'], self.user_index_map['user_index']))
        self.movie_id_to_idx = dict(zip(self.movie_index_map['canonical_id'], self.movie_index_map['movie_index']))
        
        # Load content embeddings
        self.logger.info("Loading content embeddings...")
        self.content_embeddings = np.load(self.content_embedding_path, mmap_mode='r')
        
        # Load similarity neighbors
        self.logger.info("Loading similarity neighbors...")
        self.similarity_neighbors = pd.read_parquet(self.similarity_neighbors_path)
        
        # Create neighbor lookup for faster access
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
        
        # Load user activity snapshot
        self.user_activity = pd.read_parquet(self.user_activity_path)
        self.user_activity_lookup = dict(zip(self.user_activity['user_index'], self.user_activity['ratings_count']))
        
        self.logger.info("All artifacts loaded successfully")
    
    def _init_scoring_params(self):
        """Initialize scoring parameters from policy."""
        self.alpha_defaults = self.policy["alpha_defaults"]
        self.bucket_thresholds = self.policy["bucket_thresholds"]
        self.parameters = self.policy["parameters"]
        self.fallback_policies = self.policy["fallback_policies"]
        
        # Set random seed for deterministic behavior
        np.random.seed(42)
    
    def get_user_bucket(self, user_id: str) -> str:
        """Determine user bucket based on rating count."""
        try:
            user_idx = self.user_id_to_idx.get(user_id)
            if user_idx is None:
                return 'cold'  # New user
            
            ratings_count = self.user_activity_lookup.get(user_idx, 0)
            
            if ratings_count <= self.bucket_thresholds["cold"]["max_ratings"]:
                return 'cold'
            elif ratings_count <= self.bucket_thresholds["light"]["max_ratings"]:
                return 'light'
            elif ratings_count <= self.bucket_thresholds["medium"]["max_ratings"]:
                return 'medium'
            else:
                return 'heavy'
        except Exception as e:
            self.logger.warning(f"Error determining user bucket for {user_id}: {e}")
            return 'cold'
    
    def get_alpha_for_user(self, user_id: str) -> float:
        """Get alpha value for user based on bucket and policy."""
        bucket = self.get_user_bucket(user_id)
        
        # Check for active user override
        if self.policy.get("active_user_override_alpha") is not None:
            return self.policy["active_user_override_alpha"]
        
        return self.alpha_defaults.get(bucket, 0.5)
    
    def compute_cf_scores(self, user_id: str, candidate_movies: List[str]) -> np.ndarray:
        """Compute collaborative filtering scores for candidate movies."""
        try:
            user_idx = self.user_id_to_idx.get(user_id)
            if user_idx is None:
                self.logger.warning(f"User {user_id} not found in CF factors")
                return None
            
            if user_idx >= self.user_factors.shape[0]:
                self.logger.warning(f"User index {user_idx} out of bounds for CF factors")
                return None
            
            user_vector = self.user_factors[user_idx]
            movie_indices = []
            
            for movie_id in candidate_movies:
                movie_idx = self.movie_id_to_idx.get(movie_id)
                if movie_idx is not None and movie_idx < self.movie_factors.shape[0]:
                    movie_indices.append(movie_idx)
                else:
                    movie_indices.append(-1)  # Invalid movie
            
            if not movie_indices or all(idx == -1 for idx in movie_indices):
                return None
            
            # Compute dot products
            cf_scores = np.zeros(len(candidate_movies))
            for i, movie_idx in enumerate(movie_indices):
                if movie_idx != -1:
                    cf_scores[i] = np.dot(user_vector, self.movie_factors[movie_idx])
                else:
                    cf_scores[i] = 0.0
            
            return cf_scores
            
        except Exception as e:
            self.logger.error(f"Error computing CF scores for user {user_id}: {e}")
            return None
    
    def compute_content_scores(self, user_id: str, candidate_movies: List[str]) -> np.ndarray:
        """Compute content-based scores for candidate movies."""
        try:
            content_scores = np.zeros(len(candidate_movies))
            
            for i, movie_id in enumerate(candidate_movies):
                # Get top 10 neighbors for this movie
                neighbors = self.neighbor_lookup.get(movie_id, [])
                top_neighbors = sorted(neighbors, key=lambda x: x['rank'])[:10]
                
                if top_neighbors:
                    # Average score of top 10 neighbors
                    content_scores[i] = np.mean([n['score'] for n in top_neighbors])
                else:
                    content_scores[i] = 0.0
            
            return content_scores
            
        except Exception as e:
            self.logger.error(f"Error computing content scores for user {user_id}: {e}")
            return None
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores using min-max scaling with epsilon fallback."""
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score - min_score < 1e-8:  # Constant scores
            return np.full_like(scores, 0.5)  # Neutral score
        
        return (scores - min_score) / (max_score - min_score)
    
    def percentile_rank(self, scores: np.ndarray) -> np.ndarray:
        """Convert scores to percentile ranks."""
        if len(scores) == 0:
            return scores
        
        ranks = rankdata(scores, method='average')
        return ranks / len(scores)
    
    def load_user_candidates(self, user_id: str) -> Optional[pd.DataFrame]:
        """Load pre-computed candidates for user."""
        candidates_path = f"{self.candidates_dir}/user_{user_id}_candidates.parquet"
        
        if os.path.exists(candidates_path):
            try:
                return pd.read_parquet(candidates_path)
            except Exception as e:
                self.logger.warning(f"Error loading candidates for user {user_id}: {e}")
                return None
        else:
            self.logger.warning(f"No candidates found for user {user_id}")
            return None
    
    def recommend(self, user_id: str, K: int = 50) -> Dict[str, Any]:
        """
        Generate top-K recommendations for a user.
        
        Args:
            user_id: User identifier
            K: Number of recommendations to return (default 50)
            
        Returns:
            Dictionary containing recommendations and metadata
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Generating recommendations for user {user_id}, K={K}")
            
            # Load user candidates
            candidates_df = self.load_user_candidates(user_id)
            if candidates_df is None or len(candidates_df) == 0:
                self.logger.warning(f"No candidates available for user {user_id}")
                return self._fallback_recommendations(user_id, K)
            
            candidate_movies = candidates_df['canonical_id'].tolist()
            
            # Get alpha for user
            alpha = self.get_alpha_for_user(user_id)
            self.logger.info(f"Using alpha={alpha} for user {user_id}")
            
            # Compute CF scores
            cf_scores = self.compute_cf_scores(user_id, candidate_movies)
            if cf_scores is None:
                self.logger.warning(f"CF scores unavailable for user {user_id}, using content-only")
                alpha = 0.0
                cf_scores = np.zeros(len(candidate_movies))
            
            # Compute content scores
            content_scores = self.compute_content_scores(user_id, candidate_movies)
            if content_scores is None:
                self.logger.warning(f"Content scores unavailable for user {user_id}, using CF-only")
                alpha = 1.0
                content_scores = np.zeros(len(candidate_movies))
            
            # Normalize scores
            cf_normalized = self.normalize_scores(cf_scores)
            content_normalized = self.normalize_scores(content_scores)
            
            # Convert to percentile ranks
            cf_ranked = self.percentile_rank(cf_normalized)
            content_ranked = self.percentile_rank(content_normalized)
            
            # Compute hybrid scores
            hybrid_scores = alpha * content_ranked + (1 - alpha) * cf_ranked
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'canonical_id': candidate_movies,
                'hybrid_score': hybrid_scores,
                'content_score': content_ranked,
                'collab_score': cf_ranked,
                'alpha_used': alpha
            })
            
            # Sort by hybrid score (descending) and apply tie-breaking
            results_df = results_df.sort_values(['hybrid_score', 'canonical_id'], 
                                              ascending=[False, True]).reset_index(drop=True)
            
            # Take top K
            top_k = results_df.head(K)
            
            # Prepare response
            recommendations = top_k['canonical_id'].tolist()
            
            response = {
                'user_id': user_id,
                'recommendations': recommendations,
                'metadata': {
                    'alpha_used': alpha,
                    'total_candidates': len(candidate_movies),
                    'k_requested': K,
                    'k_returned': len(recommendations),
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'scoring_method': 'hybrid',
                    'user_bucket': self.get_user_bucket(user_id)
                },
                'scores': {
                    'hybrid_scores': top_k['hybrid_score'].tolist(),
                    'content_scores': top_k['content_score'].tolist(),
                    'collab_scores': top_k['collab_score'].tolist()
                }
            }
            
            self.logger.info(f"Generated {len(recommendations)} recommendations for user {user_id} in {response['metadata']['processing_time_ms']:.2f}ms")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return self._error_response(user_id, str(e))
    
    def _fallback_recommendations(self, user_id: str, K: int) -> Dict[str, Any]:
        """Generate fallback recommendations when no candidates available."""
        self.logger.warning(f"Using fallback recommendations for user {user_id}")
        
        # Simple fallback: return first K movies from movie index
        fallback_movies = self.movie_index_map['canonical_id'].head(K).tolist()
        
        return {
            'user_id': user_id,
            'recommendations': fallback_movies,
            'metadata': {
                'alpha_used': 0.0,
                'total_candidates': 0,
                'k_requested': K,
                'k_returned': len(fallback_movies),
                'processing_time_ms': 0.0,
                'scoring_method': 'fallback',
                'user_bucket': 'unknown'
            },
            'scores': {
                'hybrid_scores': [0.0] * len(fallback_movies),
                'content_scores': [0.0] * len(fallback_movies),
                'collab_scores': [0.0] * len(fallback_movies)
            }
        }
    
    def _error_response(self, user_id: str, error_msg: str) -> Dict[str, Any]:
        """Generate error response."""
        return {
            'user_id': user_id,
            'recommendations': [],
            'error': error_msg,
            'metadata': {
                'alpha_used': 0.0,
                'total_candidates': 0,
                'k_requested': 0,
                'k_returned': 0,
                'processing_time_ms': 0.0,
                'scoring_method': 'error',
                'user_bucket': 'unknown'
            },
            'scores': {
                'hybrid_scores': [],
                'content_scores': [],
                'collab_scores': []
            }
        }


def main():
    """CLI entrypoint for the scoring service."""
    parser = argparse.ArgumentParser(description='Movie Recommendation Scoring Service')
    parser.add_argument('--user-id', required=True, help='User ID to generate recommendations for')
    parser.add_argument('--k', type=int, default=50, help='Number of recommendations (default: 50)')
    parser.add_argument('--release-lock', default='data/hybrid/release_lock_3d.json', 
                       help='Path to release lock file')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    try:
        # Initialize scorer
        scorer = MovieRecommendationScorer(args.release_lock)
        
        # Generate recommendations
        results = scorer.recommend(args.user_id, args.k)
        
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
