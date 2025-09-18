#!/usr/bin/env python3
"""
Step 4.2.2 - Snapshot Generation (Side-by-Side Lists)
Movie Recommendation Optimizer - Case Study Snapshot Generator

This module generates side-by-side snapshots of recommendations from all three systems
(content, CF, hybrid_bucket-gate) for the case slate defined in Step 4.2.1.
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class CaseSnapshotGenerator:
    """
    Generates side-by-side snapshots for case study analysis.
    """
    
    def __init__(self, policy_path: str = "data/hybrid/policy_step4.json"):
        """Initialize the snapshot generator."""
        self.logger = self._setup_logging()
        self.logger.info("Initializing CaseSnapshotGenerator")
        
        # Load policy
        self.policy = self._load_policy(policy_path)
        
        # Load case slate data
        self.users_df = pd.read_csv("data/cases/users_case_slate.csv")
        self.anchors_df = pd.read_csv("data/cases/anchors_case_slate.csv")
        
        # Load movie metadata
        self.movie_master = pd.read_parquet("data/normalized/movies_master.parquet")
        self.movie_genres = pd.read_parquet("data/features/genres/movies_genres_multihot_full.parquet")
        
        # Load collaborative filtering artifacts
        self.user_factors = np.load("data/collaborative/user_factors_k20.npy", mmap_mode='r')
        self.movie_factors = np.load("data/collaborative/movie_factors_k20.npy", mmap_mode='r')
        self.user_index_map = pd.read_parquet("data/collaborative/user_index_map.parquet")
        self.movie_index_map = pd.read_parquet("data/collaborative/movie_index_map.parquet")
        
        # Load content-based artifacts
        self.content_embeddings = np.load("data/features/composite/movies_embedding_v1.npy", mmap_mode='r')
        self.similarity_neighbors = pd.read_parquet("data/similarity/movies_neighbors_k50.parquet")
        
        # Create lookup dictionaries
        self._create_lookups()
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        self.logger.info("Snapshot generator initialization completed")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the snapshot generator."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("snapshot_generator")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('logs/step4_cases_snapshots.log')
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
    
    def _load_policy(self, policy_path: str) -> Dict[str, Any]:
        """Load the policy configuration."""
        try:
            with open(policy_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load policy: {e}")
            raise
    
    def _create_lookups(self):
        """Create lookup dictionaries for faster access."""
        # User lookups
        self.user_id_to_idx = dict(zip(self.user_index_map['userId'], self.user_index_map['user_index']))
        self.user_idx_to_id = dict(zip(self.user_index_map['user_index'], self.user_index_map['userId']))
        
        # Movie lookups
        self.movie_id_to_idx = dict(zip(self.movie_index_map['canonical_id'], self.movie_index_map['movie_index']))
        self.movie_idx_to_id = dict(zip(self.movie_index_map['movie_index'], self.movie_index_map['canonical_id']))
        
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
        
        # Create movie metadata lookup
        self.movie_metadata = {}
        for _, row in self.movie_master.iterrows():
            self.movie_metadata[row['canonical_id']] = {
                'title': row['title'],
                'year': row['year'],
                'genres_str': row.get('genres_str', ''),
                'imdb_rating': row.get('imdb_rating', 0),
                'imdb_votes': row.get('imdb_votes', 0)
            }
    
    def get_user_bucket(self, user_id: str) -> str:
        """Determine user bucket based on rating count."""
        try:
            # Extract user index from user_id
            if user_id.startswith('cold_synth_'):
                return 'cold_synth'
            elif user_id.startswith('light_'):
                return 'light'
            elif user_id.startswith('medium_'):
                return 'medium'
            elif user_id.startswith('heavy_'):
                return 'heavy'
            else:
                return 'unknown'
        except Exception as e:
            self.logger.warning(f"Error determining user bucket for {user_id}: {e}")
            return 'unknown'
    
    def get_alpha_for_user(self, user_id: str) -> float:
        """Get alpha value for user based on bucket and policy."""
        bucket = self.get_user_bucket(user_id)
        
        # Map synthetic cold users to cold bucket
        if bucket == 'cold_synth':
            bucket = 'cold'
        
        return self.policy["alpha_map"].get(bucket, 0.5)
    
    def compute_cf_scores(self, user_id: str, candidate_movies: List[str]) -> np.ndarray:
        """Compute collaborative filtering scores for candidate movies."""
        try:
            # Extract user index from user_id
            if user_id.startswith('cold_synth_'):
                # Cold synthetic users have no CF signal
                return np.zeros(len(candidate_movies))
            
            user_idx = int(user_id.split('_')[1])
            
            if user_idx >= self.user_factors.shape[0]:
                self.logger.warning(f"User index {user_idx} out of bounds for CF factors")
                return np.zeros(len(candidate_movies))
            
            user_vector = self.user_factors[user_idx]
            movie_indices = []
            
            for movie_id in candidate_movies:
                movie_idx = self.movie_id_to_idx.get(movie_id)
                if movie_idx is not None and movie_idx < self.movie_factors.shape[0]:
                    movie_indices.append(movie_idx)
                else:
                    movie_indices.append(-1)  # Invalid movie
            
            if not movie_indices or all(idx == -1 for idx in movie_indices):
                return np.zeros(len(candidate_movies))
            
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
            return np.zeros(len(candidate_movies))
    
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
            return np.zeros(len(candidate_movies))
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores using min-max scaling."""
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score - min_score < 1e-8:  # Constant scores
            return np.full_like(scores, 0.5)  # Neutral score
        
        return (scores - min_score) / (max_score - min_score)
    
    def get_movie_info(self, movie_id: str) -> Dict[str, Any]:
        """Get movie information for display."""
        metadata = self.movie_metadata.get(movie_id, {})
        
        # Get genres from the genres dataframe
        movie_genre_row = self.movie_genres[self.movie_genres.index == movie_id]
        genres = []
        if not movie_genre_row.empty:
            for col in self.movie_genres.columns:
                if col.startswith('genre_') and movie_genre_row[col].iloc[0] == 1:
                    genre_name = col.replace('genre_', '').replace('_', ' ').title()
                    genres.append(genre_name)
        
        return {
            'canonical_id': movie_id,
            'title': metadata.get('title', 'Unknown'),
            'year': metadata.get('year', 0),
            'genres': ', '.join(genres[:3]),  # Top 3 genres
            'imdb_rating': metadata.get('imdb_rating', 0),
            'imdb_votes': metadata.get('imdb_votes', 0)
        }
    
    def generate_recommendations(self, user_id: str, system: str, K: int = 10) -> Dict[str, Any]:
        """Generate recommendations for a specific system."""
        start_time = time.time()
        
        try:
            # Get all available movies as candidates (simplified for case study)
            candidate_movies = self.movie_index_map['canonical_id'].tolist()[:1000]  # Limit for performance
            
            if system == 'content':
                # Content-only system
                scores = self.compute_content_scores(user_id, candidate_movies)
                alpha_used = 1.0
                
            elif system == 'cf':
                # CF-only system
                scores = self.compute_cf_scores(user_id, candidate_movies)
                alpha_used = 0.0
                
            elif system == 'hybrid_bg':
                # Hybrid bucket-gate system
                cf_scores = self.compute_cf_scores(user_id, candidate_movies)
                content_scores = self.compute_content_scores(user_id, candidate_movies)
                
                # Normalize scores
                cf_normalized = self.normalize_scores(cf_scores)
                content_normalized = self.normalize_scores(content_scores)
                
                # Get alpha for user
                alpha_used = self.get_alpha_for_user(user_id)
                
                # Blend scores
                scores = alpha_used * content_normalized + (1 - alpha_used) * cf_normalized
                
            else:
                raise ValueError(f"Unknown system: {system}")
            
            # Normalize final scores
            scores = self.normalize_scores(scores)
            
            # Get top K movies
            top_indices = np.argsort(scores)[::-1][:K]
            top_movies = [candidate_movies[i] for i in top_indices]
            top_scores = scores[top_indices]
            
            # Get movie information
            recommendations = []
            for i, (movie_id, score) in enumerate(zip(top_movies, top_scores)):
                movie_info = self.get_movie_info(movie_id)
                movie_info['rank'] = i + 1
                movie_info['score'] = float(score)
                recommendations.append(movie_info)
            
            return {
                'user_id': user_id,
                'system': system,
                'alpha_used': alpha_used,
                'recommendations': recommendations,
                'metadata': {
                    'total_candidates': len(candidate_movies),
                    'k_requested': K,
                    'k_returned': len(recommendations),
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'user_bucket': self.get_user_bucket(user_id)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations for user {user_id}, system {system}: {e}")
            return {
                'user_id': user_id,
                'system': system,
                'alpha_used': 0.0,
                'recommendations': [],
                'error': str(e),
                'metadata': {
                    'total_candidates': 0,
                    'k_requested': K,
                    'k_returned': 0,
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'user_bucket': self.get_user_bucket(user_id)
                }
            }
    
    def generate_case_snapshot(self, user_id: str, anchor_id: str, case_id: str) -> Dict[str, Any]:
        """Generate snapshot for a specific case."""
        self.logger.info(f"Generating snapshot for case {case_id}: user {user_id}, anchor {anchor_id}")
        
        # Generate recommendations for all three systems
        systems = ['content', 'cf', 'hybrid_bg']
        results = {}
        
        for system in systems:
            self.logger.info(f"Generating {system} recommendations for case {case_id}")
            results[system] = self.generate_recommendations(user_id, system, K=10)
        
        # Get anchor movie info
        anchor_info = self.get_movie_info(anchor_id)
        
        # Create case snapshot
        snapshot = {
            'case_id': case_id,
            'user_id': user_id,
            'anchor_id': anchor_id,
            'anchor_info': anchor_info,
            'user_bucket': self.get_user_bucket(user_id),
            'timestamp': datetime.now().isoformat(),
            'systems': results
        }
        
        return snapshot
    
    def save_snapshot(self, snapshot: Dict[str, Any], output_dir: str = "data/cases/snapshots"):
        """Save snapshot to JSON files."""
        case_id = snapshot['case_id']
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual system snapshots
        for system, data in snapshot['systems'].items():
            filename = f"{case_id}_{system}.json"
            filepath = output_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Saved {system} snapshot to {filepath}")
        
        # Save combined snapshot
        combined_filename = f"{case_id}_combined.json"
        combined_filepath = output_path / combined_filename
        
        with open(combined_filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        self.logger.info(f"Saved combined snapshot to {combined_filepath}")
    
    def create_triptych_visualization(self, snapshot: Dict[str, Any], output_dir: str = "docs/img/cases"):
        """Create triptych visualization for the case."""
        case_id = snapshot['case_id']
        user_id = snapshot['user_id']
        anchor_info = snapshot['anchor_info']
        user_bucket = snapshot['user_bucket']
        
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 12))
        fig.suptitle(f'Case {case_id}: {user_id} ({user_bucket}) - Anchor: {anchor_info["title"]} ({anchor_info["year"]})', 
                     fontsize=16, fontweight='bold')
        
        systems = ['content', 'cf', 'hybrid_bg']
        system_titles = ['Content-Based', 'Collaborative Filtering', 'Hybrid Bucket-Gate']
        
        for i, (system, title) in enumerate(zip(systems, system_titles)):
            ax = axes[i]
            recommendations = snapshot['systems'][system]['recommendations']
            alpha_used = snapshot['systems'][system]['alpha_used']
            
            # Set subplot title
            ax.set_title(f'{title}\n(α = {alpha_used:.2f})', fontsize=14, fontweight='bold')
            
            # Create recommendation list
            y_pos = np.arange(len(recommendations))
            
            # Plot bars for scores
            scores = [r['score'] for r in recommendations]
            bars = ax.barh(y_pos, scores, alpha=0.7, color=plt.cm.viridis(np.array(scores)))
            
            # Add movie titles and info
            for j, rec in enumerate(recommendations):
                title_text = f"{rec['rank']}. {rec['title'][:30]}{'...' if len(rec['title']) > 30 else ''}"
                year_text = f"({rec['year']})" if rec['year'] > 0 else ""
                genre_text = f" - {rec['genres'][:20]}{'...' if len(rec['genres']) > 20 else ''}"
                
                ax.text(0.01, j, f"{title_text} {year_text}{genre_text}", 
                       va='center', ha='left', fontsize=10, fontweight='bold')
                ax.text(0.95, j, f"{rec['score']:.3f}", 
                       va='center', ha='right', fontsize=9)
            
            # Customize subplot
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, len(recommendations) - 0.5)
            ax.set_xlabel('Score', fontsize=12)
            ax.set_ylabel('Rank', fontsize=12)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            
            # Remove y-axis ticks
            ax.set_yticks([])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{case_id}_triptych.png"
        filepath = output_path / filename
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved triptych visualization to {filepath}")
    
    def generate_all_snapshots(self):
        """Generate snapshots for all cases in the case slate."""
        self.logger.info("Starting snapshot generation for all cases")
        
        # Create output directories
        os.makedirs("data/cases/snapshots", exist_ok=True)
        os.makedirs("docs/img/cases", exist_ok=True)
        
        total_cases = 0
        successful_cases = 0
        
        # Generate snapshots for each user-anchor combination
        for _, user_row in self.users_df.iterrows():
            user_id = user_row['user_id']
            user_bucket = user_row['cohort']
            
            for _, anchor_row in self.anchors_df.iterrows():
                anchor_id = anchor_row['canonical_id']
                anchor_bucket = anchor_row['popularity_bucket']
                
                # Create case ID
                case_id = f"{user_id}_{anchor_id}"
                total_cases += 1
                
                try:
                    # Generate snapshot
                    snapshot = self.generate_case_snapshot(user_id, anchor_id, case_id)
                    
                    # Save snapshot
                    self.save_snapshot(snapshot)
                    
                    # Create visualization
                    self.create_triptych_visualization(snapshot)
                    
                    successful_cases += 1
                    self.logger.info(f"Successfully generated snapshot for case {case_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate snapshot for case {case_id}: {e}")
        
        self.logger.info(f"Snapshot generation completed: {successful_cases}/{total_cases} cases successful")
        return successful_cases, total_cases


def main():
    """CLI entrypoint for the snapshot generator."""
    parser = argparse.ArgumentParser(description='Case Study Snapshot Generator')
    parser.add_argument('--policy', default='data/hybrid/policy_step4.json', 
                       help='Path to policy file')
    parser.add_argument('--case-id', help='Generate snapshot for specific case ID')
    parser.add_argument('--user-id', help='User ID for specific case')
    parser.add_argument('--anchor-id', help='Anchor ID for specific case')
    parser.add_argument('--all', action='store_true', help='Generate all snapshots')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = CaseSnapshotGenerator(args.policy)
        
        if args.all:
            # Generate all snapshots
            successful, total = generator.generate_all_snapshots()
            print(f"Generated {successful}/{total} snapshots successfully")
            
        elif args.case_id and args.user_id and args.anchor_id:
            # Generate specific case
            snapshot = generator.generate_case_snapshot(args.user_id, args.anchor_id, args.case_id)
            generator.save_snapshot(snapshot)
            generator.create_triptych_visualization(snapshot)
            print(f"Generated snapshot for case {args.case_id}")
            
        else:
            print("Please specify --all or provide --case-id, --user-id, and --anchor-id")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


