#!/usr/bin/env python3
"""
Step 4.2.3 - Rationale Attribution & Evidence
Movie Recommendation Optimizer - Attribution Generator

This module generates clear "why recommended" rationales for each recommended item
using concrete evidence signals from content embeddings, CF factors, and policy decisions.
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
from datetime import datetime
import argparse

def safe_json_serialize(obj):
    """Safely serialize objects to JSON, handling NaN and other problematic values."""
    if isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif pd.isna(obj):
        return None
    else:
        return obj

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class AttributionGenerator:
    """
    Generates rationale attributions for recommendation items.
    """
    
    def __init__(self, policy_path: str = "data/hybrid/policy_step4.json"):
        """Initialize the attribution generator."""
        self.logger = self._setup_logging()
        self.logger.info("Initializing AttributionGenerator")
        
        # Load policy
        self.policy = self._load_policy(policy_path)
        
        # Load content-based artifacts
        self.content_embeddings = np.load("data/features/composite/movies_embedding_v1.npy", mmap_mode='r')
        self.similarity_neighbors = pd.read_parquet("data/similarity/movies_neighbors_k50.parquet")
        self.movie_features = pd.read_parquet("data/features/composite/movies_features_v1.parquet")
        self.movie_genres = pd.read_parquet("data/features/genres/movies_genres_multihot_full.parquet")
        
        # Load collaborative filtering artifacts
        self.user_factors = np.load("data/collaborative/user_factors_k20.npy", mmap_mode='r')
        self.movie_factors = np.load("data/collaborative/movie_factors_k20.npy", mmap_mode='r')
        self.user_index_map = pd.read_parquet("data/collaborative/user_index_map.parquet")
        self.movie_index_map = pd.read_parquet("data/collaborative/movie_index_map.parquet")
        
        # Load movie metadata
        self.movie_master = pd.read_parquet("data/normalized/movies_master.parquet")
        
        # Create lookup dictionaries
        self._create_lookups()
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        self.logger.info("Attribution generator initialization completed")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the attribution generator."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("attribution_generator")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('logs/step4_2_3_attributions.log')
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
        
        # Create genre lookup
        self.genre_lookup = {}
        for _, row in self.movie_genres.iterrows():
            movie_id = row.name
            genres = []
            for col in self.movie_genres.columns:
                if col.startswith('genre_') and row[col] == 1:
                    genre_name = col.replace('genre_', '').replace('_', ' ').title()
                    genres.append(genre_name)
            self.genre_lookup[movie_id] = genres
    
    def get_content_signals(self, movie_id: str, anchor_id: str) -> Dict[str, Any]:
        """Get content-based evidence signals for a movie."""
        try:
            # Get movie indices
            movie_idx = self.movie_id_to_idx.get(movie_id)
            anchor_idx = self.movie_id_to_idx.get(anchor_id)
            
            if movie_idx is None or anchor_idx is None:
                return {"error": "Movie or anchor not found in index"}
            
            # Compute cosine similarity
            movie_embedding = self.content_embeddings[movie_idx]
            anchor_embedding = self.content_embeddings[anchor_idx]
            cosine_similarity = float(np.dot(movie_embedding, anchor_embedding))
            
            # Get genre overlap
            movie_genres = set(self.genre_lookup.get(movie_id, []))
            anchor_genres = set(self.genre_lookup.get(anchor_id, []))
            genre_overlap = len(movie_genres.intersection(anchor_genres))
            genre_union = len(movie_genres.union(anchor_genres))
            genre_jaccard = genre_overlap / genre_union if genre_union > 0 else 0.0
            
            # Get similarity rank from neighbors
            neighbors = self.similarity_neighbors[self.similarity_neighbors['movie_id'] == anchor_id]
            movie_neighbors = neighbors[neighbors['neighbor_id'] == movie_id]
            similarity_rank = int(movie_neighbors['rank'].iloc[0]) if not movie_neighbors.empty else None
            
            return {
                "cosine_similarity": cosine_similarity,
                "genre_overlap": genre_overlap,
                "genre_jaccard": genre_jaccard,
                "movie_genres": list(movie_genres),
                "anchor_genres": list(anchor_genres),
                "similarity_rank": similarity_rank,
                "shared_genres": list(movie_genres.intersection(anchor_genres))
            }
            
        except Exception as e:
            self.logger.error(f"Error computing content signals for {movie_id}: {e}")
            return {"error": str(e)}
    
    def get_cf_signals(self, movie_id: str, user_id: str) -> Dict[str, Any]:
        """Get collaborative filtering evidence signals for a movie."""
        try:
            # Extract user index from user_id
            if user_id.startswith('cold_synth_'):
                return {"error": "Cold synthetic user - no CF signals available"}
            
            user_idx = int(user_id.split('_')[1])
            
            if user_idx >= self.user_factors.shape[0]:
                return {"error": "User index out of bounds"}
            
            # Get movie index
            movie_idx = self.movie_id_to_idx.get(movie_id)
            if movie_idx is None:
                return {"error": "Movie not found in CF index"}
            
            # Compute CF score
            user_vector = self.user_factors[user_idx]
            movie_vector = self.movie_factors[movie_idx]
            cf_score = float(np.dot(user_vector, movie_vector))
            
            # Get user factor norm
            user_norm = float(np.linalg.norm(user_vector))
            
            # Get movie factor norm
            movie_norm = float(np.linalg.norm(movie_vector))
            
            # Find similar users (users with similar factor vectors)
            user_similarities = np.dot(self.user_factors, user_vector)
            similar_users = np.argsort(user_similarities)[::-1][:10]  # Top 10 similar users
            similar_user_scores = user_similarities[similar_users]
            
            return {
                "cf_score": cf_score,
                "user_norm": user_norm,
                "movie_norm": movie_norm,
                "similar_users_count": len(similar_users),
                "top_similar_user_score": float(similar_user_scores[0]) if len(similar_user_scores) > 0 else 0.0,
                "avg_similar_user_score": float(np.mean(similar_user_scores)) if len(similar_user_scores) > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error computing CF signals for {movie_id}, user {user_id}: {e}")
            return {"error": str(e)}
    
    def get_policy_path(self, user_id: str, movie_id: str) -> Dict[str, Any]:
        """Get policy decision path for a recommendation."""
        try:
            # Determine user bucket
            if user_id.startswith('cold_synth_'):
                bucket = 'cold'
            elif user_id.startswith('light_'):
                bucket = 'light'
            elif user_id.startswith('medium_'):
                bucket = 'medium'
            elif user_id.startswith('heavy_'):
                bucket = 'heavy'
            else:
                bucket = 'unknown'
            
            # Get alpha value
            alpha = self.policy["alpha_map"].get(bucket, 0.5)
            
            # Check for overrides
            overrides_applied = []
            
            # Check long-tail override
            if bucket in ['cold', 'light'] and alpha < 0.5:
                overrides_applied.append("minimal_history_guardrail")
            
            # Check if movie is long-tail (simplified heuristic)
            movie_metadata = self.movie_metadata.get(movie_id, {})
            imdb_votes = movie_metadata.get('imdb_votes', 0)
            
            # Handle NaN values
            if pd.isna(imdb_votes) or imdb_votes is None:
                imdb_votes = 0
            
            if imdb_votes < 1000:  # Long-tail heuristic
                overrides_applied.append("long_tail_override")
            
            return {
                "user_bucket": bucket,
                "alpha_used": alpha,
                "overrides_applied": overrides_applied,
                "policy_version": self.policy.get("version", "unknown"),
                "fallback_used": len(overrides_applied) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error computing policy path for {user_id}, {movie_id}: {e}")
            return {"error": str(e)}
    
    def get_diversity_notes(self, movie_id: str, recommendations: List[Dict]) -> Dict[str, Any]:
        """Get diversity analysis for a movie in the context of recommendations."""
        try:
            # Get movie genres
            movie_genres = set(self.genre_lookup.get(movie_id, []))
            
            # Count genre diversity in recommendations
            all_genres = set()
            for rec in recommendations:
                rec_genres = set(self.genre_lookup.get(rec['canonical_id'], []))
                all_genres.update(rec_genres)
            
            # Check if movie adds diversity
            new_genres = movie_genres - all_genres
            diversity_contribution = len(new_genres) / len(movie_genres) if movie_genres else 0.0
            
            # Check if movie is from long-tail (diversity boost)
            movie_metadata = self.movie_metadata.get(movie_id, {})
            imdb_votes = movie_metadata.get('imdb_votes', 0)
            is_long_tail = imdb_votes < 1000
            
            return {
                "diversity_contribution": diversity_contribution,
                "new_genres_added": list(new_genres),
                "is_long_tail": is_long_tail,
                "total_genres_in_recs": len(all_genres),
                "movie_genre_count": len(movie_genres)
            }
            
        except Exception as e:
            self.logger.error(f"Error computing diversity notes for {movie_id}: {e}")
            return {"error": str(e)}
    
    def get_filter_signals(self, movie_id: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get filter match signals for a movie."""
        try:
            movie_metadata = self.movie_metadata.get(movie_id, {})
            
            filter_matches = {}
            
            # Year filter
            if filters and 'year_range' in filters:
                year_min, year_max = filters['year_range']
                movie_year = movie_metadata.get('year', 0)
                
                # Handle NaN values
                if pd.isna(movie_year) or movie_year is None:
                    movie_year = 0
                
                filter_matches['year_match'] = year_min <= movie_year <= year_max
                filter_matches['year_value'] = int(movie_year)
            else:
                filter_matches['year_match'] = True  # No year filter applied
                movie_year = movie_metadata.get('year', 0)
                if pd.isna(movie_year) or movie_year is None:
                    movie_year = 0
                filter_matches['year_value'] = int(movie_year)
            
            # Provider filter (simplified - would need provider data)
            if filters and 'providers' in filters:
                # For now, assume all movies match provider filter
                filter_matches['provider_match'] = True
            else:
                filter_matches['provider_match'] = True  # No provider filter applied
            
            return filter_matches
            
        except Exception as e:
            self.logger.error(f"Error computing filter signals for {movie_id}: {e}")
            return {"error": str(e)}
    
    def generate_attribution(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate attribution for a single case."""
        case_id = case_data['case_id']
        user_id = case_data['user_id']
        anchor_id = case_data['anchor_id']
        user_bucket = case_data['user_bucket']
        
        self.logger.info(f"Generating attribution for case {case_id}")
        
        # Get hybrid recommendations
        hybrid_recs = case_data['systems']['hybrid_bg']['recommendations']
        
        attributions = []
        
        for rec in hybrid_recs:
            movie_id = rec['canonical_id']
            
            # Get all evidence signals
            content_signals = self.get_content_signals(movie_id, anchor_id)
            cf_signals = self.get_cf_signals(movie_id, user_id)
            policy_path = self.get_policy_path(user_id, movie_id)
            diversity_notes = self.get_diversity_notes(movie_id, hybrid_recs)
            filter_signals = self.get_filter_signals(movie_id)
            
            # Create attribution entry
            attribution = {
                "movie_id": movie_id,
                "title": rec['title'],
                "rank": rec['rank'],
                "score": rec['score'],
                "content_signals": content_signals,
                "cf_signals": cf_signals,
                "policy_path": policy_path,
                "diversity_notes": diversity_notes,
                "filter_signals": filter_signals,
                "rationale_summary": self._generate_rationale_summary(
                    rec, content_signals, cf_signals, policy_path, diversity_notes
                )
            }
            
            attributions.append(attribution)
        
        return {
            "case_id": case_id,
            "user_id": user_id,
            "anchor_id": anchor_id,
            "user_bucket": user_bucket,
            "timestamp": datetime.now().isoformat(),
            "attributions": attributions
        }
    
    def _generate_rationale_summary(self, rec: Dict, content_signals: Dict, cf_signals: Dict, 
                                   policy_path: Dict, diversity_notes: Dict) -> str:
        """Generate a concise rationale summary for a recommendation."""
        try:
            title = rec['title']
            rank = rec['rank']
            
            # Start with basic info
            rationale = f"Ranked #{rank} because "
            
            # Add content signals
            if 'cosine_similarity' in content_signals and not content_signals.get('error'):
                cosine = content_signals['cosine_similarity']
                if cosine is not None and not (isinstance(cosine, float) and pd.isna(cosine)):
                    rationale += f"it has high content similarity ({cosine:.3f}) with the anchor movie"
                    
                    if content_signals.get('shared_genres'):
                        genres = ', '.join(content_signals['shared_genres'][:2])
                        rationale += f" and shares genres ({genres})"
                    
                    rationale += ". "
            
            # Add CF signals
            if 'cf_score' in cf_signals and not cf_signals.get('error'):
                cf_score = cf_signals['cf_score']
                if cf_score is not None and not (isinstance(cf_score, float) and pd.isna(cf_score)):
                    rationale += f"Collaborative filtering suggests it matches your taste profile (CF score: {cf_score:.3f}). "
            
            # Add policy context
            alpha = policy_path.get('alpha_used', 0.5)
            bucket = policy_path.get('user_bucket', 'unknown')
            
            if bucket == 'cold':
                rationale += f"As a new user, we're using content-heavy recommendations (α={alpha:.1f}) to help you discover movies. "
            elif bucket == 'heavy':
                rationale += f"As an experienced user, we're leveraging your rating history (α={alpha:.1f}) for personalized recommendations. "
            
            # Add diversity note
            if diversity_notes.get('is_long_tail'):
                rationale += "This is a lesser-known movie that adds diversity to your recommendations. "
            
            return rationale.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating rationale summary: {e}")
            return f"Recommended at rank #{rec['rank']} based on hybrid scoring."
    
    def save_attribution(self, attribution: Dict[str, Any], output_dir: str = "data/cases/attributions"):
        """Save attribution to JSON file."""
        case_id = attribution['case_id']
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{case_id}.json"
        filepath = output_path / filename
        
        # Use safe serialization to handle NaN values
        safe_attribution = safe_json_serialize(attribution)
        
        with open(filepath, 'w') as f:
            json.dump(safe_attribution, f, indent=2)
        
        self.logger.info(f"Saved attribution to {filepath}")
    
    def generate_why_templates(self, output_dir: str = "docs/cases"):
        """Generate reusable why templates."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        templates = {
            "content_based": {
                "high_similarity": "Recommended because it has high content similarity ({cosine:.3f}) with {anchor_title}.",
                "genre_match": "Appears due to shared genres ({genres}) with your anchor movie.",
                "semantic_similarity": "Selected based on semantic similarity in plot, themes, or style."
            },
            "collaborative": {
                "user_similarity": "Recommended because {similar_users_count} users with similar taste also rated it highly.",
                "cf_score": "Appears due to collaborative filtering score ({cf_score:.3f}) matching your preferences.",
                "neighbor_rating": "Users who liked similar movies also rated this highly."
            },
            "policy": {
                "cold_user": "As a new user, we're using content-heavy recommendations (α={alpha:.1f}) to help you discover movies.",
                "heavy_user": "As an experienced user, we're leveraging your rating history (α={alpha:.1f}) for personalized recommendations.",
                "light_user": "Based on your limited history, we're balancing content and collaborative signals (α={alpha:.1f}).",
                "medium_user": "Using a balanced approach (α={alpha:.1f}) combining your preferences with content similarity."
            },
            "diversity": {
                "long_tail": "This lesser-known movie adds diversity to your recommendations.",
                "genre_diversity": "Selected to provide genre diversity in your recommendations.",
                "popularity_balance": "Included to balance popular and niche content."
            },
            "filters": {
                "year_match": "Matches your year preference ({year_range}).",
                "provider_match": "Available on your preferred streaming platform.",
                "no_filters": "No specific filters applied to this recommendation."
            }
        }
        
        filepath = output_path / "why_templates.md"
        
        with open(filepath, 'w') as f:
            f.write("# Why Templates for Movie Recommendations\n\n")
            f.write("This document contains reusable sentence structures for generating recommendation rationales.\n\n")
            
            for category, template_dict in templates.items():
                f.write(f"## {category.replace('_', ' ').title()}\n\n")
                for key, template in template_dict.items():
                    f.write(f"**{key.replace('_', ' ').title()}**: {template}\n\n")
        
        self.logger.info(f"Saved why templates to {filepath}")
    
    def generate_case_why(self, attribution: Dict[str, Any], output_dir: str = "docs/cases"):
        """Generate human-readable why document for a case."""
        case_id = attribution['case_id']
        user_id = attribution['user_id']
        anchor_id = attribution['anchor_id']
        user_bucket = attribution['user_bucket']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{case_id}_why.md"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            f.write(f"# Why Recommendations for {case_id}\n\n")
            f.write(f"**User**: {user_id} ({user_bucket})\n")
            f.write(f"**Anchor Movie**: {self.movie_metadata.get(anchor_id, {}).get('title', 'Unknown')}\n\n")
            
            f.write("## Recommendation Rationales\n\n")
            
            for i, attr in enumerate(attribution['attributions'], 1):
                f.write(f"### {i}. {attr['title']} (Rank #{attr['rank']})\n\n")
                f.write(f"**Score**: {attr['score']:.3f}\n\n")
                f.write(f"**Rationale**: {attr['rationale_summary']}\n\n")
                
                # Add detailed evidence
                f.write("**Detailed Evidence**:\n\n")
                
                # Content signals
                if not attr['content_signals'].get('error'):
                    f.write("- **Content Similarity**: ")
                    f.write(f"Cosine similarity = {attr['content_signals'].get('cosine_similarity', 0):.3f}")
                    if attr['content_signals'].get('shared_genres'):
                        f.write(f", shares {len(attr['content_signals']['shared_genres'])} genres")
                    f.write("\n")
                
                # CF signals
                if not attr['cf_signals'].get('error'):
                    f.write("- **Collaborative Filtering**: ")
                    f.write(f"CF score = {attr['cf_signals'].get('cf_score', 0):.3f}")
                    f.write(f", {attr['cf_signals'].get('similar_users_count', 0)} similar users")
                    f.write("\n")
                
                # Policy path
                f.write("- **Policy Decision**: ")
                f.write(f"α = {attr['policy_path'].get('alpha_used', 0):.1f} ({attr['policy_path'].get('user_bucket', 'unknown')} user)")
                if attr['policy_path'].get('overrides_applied'):
                    f.write(f", overrides: {', '.join(attr['policy_path']['overrides_applied'])}")
                f.write("\n")
                
                f.write("\n---\n\n")
        
        self.logger.info(f"Saved case why document to {filepath}")
    
    def process_all_cases(self, snapshots_dir: str = "data/cases/snapshots"):
        """Process all case snapshots to generate attributions."""
        self.logger.info("Starting attribution generation for all cases")
        
        snapshots_path = Path(snapshots_dir)
        combined_files = list(snapshots_path.glob("*_combined.json"))
        
        successful_cases = 0
        total_cases = len(combined_files)
        
        for file_path in combined_files:
            try:
                # Load case data
                with open(file_path, 'r') as f:
                    case_data = json.load(f)
                
                # Generate attribution
                attribution = self.generate_attribution(case_data)
                
                # Save attribution
                self.save_attribution(attribution)
                
                # Generate case why document
                self.generate_case_why(attribution)
                
                successful_cases += 1
                self.logger.info(f"Successfully processed case {attribution['case_id']}")
                
            except Exception as e:
                self.logger.error(f"Failed to process case {file_path}: {e}")
        
        # Generate why templates
        self.generate_why_templates()
        
        self.logger.info(f"Attribution generation completed: {successful_cases}/{total_cases} cases successful")
        return successful_cases, total_cases


def main():
    """CLI entrypoint for the attribution generator."""
    parser = argparse.ArgumentParser(description='Attribution Generator')
    parser.add_argument('--policy', default='data/hybrid/policy_step4.json', 
                       help='Path to policy file')
    parser.add_argument('--case-id', help='Generate attribution for specific case ID')
    parser.add_argument('--snapshot-file', help='Path to specific snapshot file')
    parser.add_argument('--all', action='store_true', help='Process all cases')
    
    args = parser.parse_args()
    
    try:
        generator = AttributionGenerator(args.policy)
        
        if args.all:
            successful, total = generator.process_all_cases()
            print(f"Generated attributions for {successful}/{total} cases successfully")
            
        elif args.case_id and args.snapshot_file:
            with open(args.snapshot_file, 'r') as f:
                case_data = json.load(f)
            
            attribution = generator.generate_attribution(case_data)
            generator.save_attribution(attribution)
            generator.generate_case_why(attribution)
            print(f"Generated attribution for case {args.case_id}")
            
        else:
            print("Please specify --all or provide --case-id and --snapshot-file")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
