#!/usr/bin/env python3
"""
Step 4.2.4 - Red-Team & Error Taxonomy
Movie Recommendation Optimizer - Error Analysis and Mitigation

This module identifies common failure modes in recommendations, documents them
with reproducible recipes, and proposes concrete mitigations.
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
from collections import defaultdict, Counter
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class ErrorTaxonomyAnalyzer:
    """
    Analyzes recommendation failures and generates error taxonomy.
    """
    
    def __init__(self, policy_path: str = "data/hybrid/policy_step4.json"):
        """Initialize the error taxonomy analyzer."""
        self.logger = self._setup_logging()
        self.logger.info("Initializing ErrorTaxonomyAnalyzer")
        
        # Load policy
        self.policy = self._load_policy(policy_path)
        
        # Load movie metadata for analysis
        self.movie_master = pd.read_parquet("data/normalized/movies_master.parquet")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        self.logger.info("Error taxonomy analyzer initialization completed")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the error taxonomy analyzer."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("error_taxonomy")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('logs/step4_2_4_error_taxonomy.log')
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
    
    def analyze_all_cases(self, snapshots_dir: str = "data/cases/snapshots", 
                         attributions_dir: str = "data/cases/attributions") -> List[Dict[str, Any]]:
        """Analyze all cases for failure patterns."""
        self.logger.info("Starting comprehensive case analysis")
        
        snapshots_path = Path(snapshots_dir)
        attributions_path = Path(attributions_dir)
        
        snapshot_files = list(snapshots_path.glob("*_combined.json"))
        attribution_files = list(attributions_path.glob("*.json"))
        
        analyzed_cases = []
        
        for snapshot_file, attr_file in zip(snapshot_files, attribution_files):
            try:
                # Load case data
                with open(snapshot_file, 'r') as f:
                    snapshot = json.load(f)
                with open(attr_file, 'r') as f:
                    attribution = json.load(f)
                
                # Analyze case for failures
                case_analysis = self._analyze_single_case(snapshot, attribution)
                analyzed_cases.append(case_analysis)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze case {snapshot_file}: {e}")
        
        self.logger.info(f"Analyzed {len(analyzed_cases)} cases")
        return analyzed_cases
    
    def _analyze_single_case(self, snapshot: Dict[str, Any], attribution: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single case for failure patterns."""
        case_id = snapshot['case_id']
        user_bucket = snapshot['user_bucket']
        anchor_id = snapshot['anchor_id']
        anchor_info = snapshot['anchor_info']
        
        hybrid_recs = snapshot['systems']['hybrid_bg']['recommendations']
        attributions = attribution['attributions']
        
        # Initialize analysis
        analysis = {
            'case_id': case_id,
            'user_bucket': user_bucket,
            'anchor_id': anchor_id,
            'anchor_title': anchor_info['title'],
            'anchor_year': anchor_info['year'],
            'anchor_imdb_votes': anchor_info.get('imdb_votes', 0),
            'failures': [],
            'metrics': {}
        }
        
        # Analyze each failure type
        self._detect_popularity_bias(analysis, hybrid_recs, attributions)
        self._detect_redundancy(analysis, hybrid_recs, attributions)
        self._detect_temporal_drift(analysis, hybrid_recs, attributions)
        self._detect_provider_mismatch(analysis, hybrid_recs, attributions)
        self._detect_franchise_overfit(analysis, hybrid_recs, attributions)
        self._detect_niche_misfire(analysis, hybrid_recs, attributions)
        self._detect_long_tail_starvation(analysis, hybrid_recs, attributions)
        self._detect_cold_start_miss(analysis, hybrid_recs, attributions)
        
        return analysis
    
    def _detect_popularity_bias(self, analysis: Dict, recs: List[Dict], attrs: List[Dict]):
        """Detect popularity bias / head collapse."""
        imdb_votes = [rec.get('imdb_votes', 0) for rec in recs]
        imdb_votes = [v for v in imdb_votes if v > 0]
        
        if not imdb_votes:
            return
        
        # Calculate metrics
        avg_votes = np.mean(imdb_votes)
        median_votes = np.median(imdb_votes)
        high_pop_ratio = sum(1 for v in imdb_votes if v > 100000) / len(imdb_votes)
        long_tail_ratio = sum(1 for v in imdb_votes if v < 1000) / len(imdb_votes)
        
        analysis['metrics']['popularity'] = {
            'avg_imdb_votes': avg_votes,
            'median_imdb_votes': median_votes,
            'high_pop_ratio': high_pop_ratio,
            'long_tail_ratio': long_tail_ratio
        }
        
        # Detect failure conditions
        if high_pop_ratio > 0.7:  # 70%+ high popularity
            analysis['failures'].append({
                'type': 'popularity_bias',
                'severity': 'S2' if high_pop_ratio > 0.9 else 'S3',
                'symptoms': f'High popularity concentration: {high_pop_ratio:.1%} of recs have >100k votes',
                'evidence': f'avg_votes={avg_votes:.0f}, median={median_votes:.0f}',
                'repro_steps': f'User {analysis["user_bucket"]} with anchor {analysis["anchor_title"]} ({analysis["anchor_year"]})'
            })
        
        if long_tail_ratio < 0.2:  # <20% long-tail
            analysis['failures'].append({
                'type': 'long_tail_starvation',
                'severity': 'S3',
                'symptoms': f'Insufficient long-tail diversity: only {long_tail_ratio:.1%} of recs have <1k votes',
                'evidence': f'avg_votes={avg_votes:.0f}, median={median_votes:.0f}',
                'repro_steps': f'User {analysis["user_bucket"]} with anchor {analysis["anchor_title"]} ({analysis["anchor_year"]})'
            })
    
    def _detect_redundancy(self, analysis: Dict, recs: List[Dict], attrs: List[Dict]):
        """Detect redundancy / near-duplicates."""
        # Check for high cosine similarity between recommendations
        cosine_similarities = []
        for attr in attrs:
            cosine = attr['content_signals'].get('cosine_similarity', 0)
            if cosine > 0:
                cosine_similarities.append(cosine)
        
        if len(cosine_similarities) < 2:
            return
        
        # Calculate pairwise similarities (simplified)
        high_sim_pairs = 0
        total_pairs = 0
        
        for i in range(len(cosine_similarities)):
            for j in range(i+1, len(cosine_similarities)):
                total_pairs += 1
                if abs(cosine_similarities[i] - cosine_similarities[j]) < 0.1:  # Very similar scores
                    high_sim_pairs += 1
        
        if total_pairs > 0:
            redundancy_ratio = high_sim_pairs / total_pairs
            
            analysis['metrics']['redundancy'] = {
                'redundancy_ratio': redundancy_ratio,
                'avg_cosine': np.mean(cosine_similarities),
                'cosine_std': np.std(cosine_similarities)
            }
            
            if redundancy_ratio > 0.3:  # 30%+ similar pairs
                analysis['failures'].append({
                    'type': 'redundancy',
                    'severity': 'S3',
                    'symptoms': f'High redundancy: {redundancy_ratio:.1%} of recommendation pairs are very similar',
                    'evidence': f'avg_cosine={np.mean(cosine_similarities):.3f}, std={np.std(cosine_similarities):.3f}',
                    'repro_steps': f'User {analysis["user_bucket"]} with anchor {analysis["anchor_title"]} ({analysis["anchor_year"]})'
                })
    
    def _detect_temporal_drift(self, analysis: Dict, recs: List[Dict], attrs: List[Dict]):
        """Detect stale / temporal drift."""
        anchor_year = analysis['anchor_year']
        rec_years = [rec.get('year', 0) for rec in recs if rec.get('year', 0) > 0]
        
        if not rec_years or anchor_year <= 0:
            return
        
        # Calculate temporal metrics
        avg_rec_year = np.mean(rec_years)
        year_gap = abs(avg_rec_year - anchor_year)
        old_movies_ratio = sum(1 for y in rec_years if y < anchor_year - 10) / len(rec_years)
        
        analysis['metrics']['temporal'] = {
            'anchor_year': anchor_year,
            'avg_rec_year': avg_rec_year,
            'year_gap': year_gap,
            'old_movies_ratio': old_movies_ratio
        }
        
        if year_gap > 20:  # Large temporal gap
            analysis['failures'].append({
                'type': 'temporal_drift',
                'severity': 'S2' if year_gap > 30 else 'S3',
                'symptoms': f'Large temporal gap: {year_gap:.0f} years between anchor ({anchor_year}) and avg rec ({avg_rec_year:.0f})',
                'evidence': f'anchor_year={anchor_year}, avg_rec_year={avg_rec_year:.0f}, gap={year_gap:.0f}',
                'repro_steps': f'User {analysis["user_bucket"]} with anchor {analysis["anchor_title"]} ({analysis["anchor_year"]})'
            })
        
        if old_movies_ratio > 0.5:  # 50%+ old movies
            analysis['failures'].append({
                'type': 'stale_content',
                'severity': 'S3',
                'symptoms': f'Too many old movies: {old_movies_ratio:.1%} of recs are >10 years older than anchor',
                'evidence': f'anchor_year={anchor_year}, old_ratio={old_movies_ratio:.1%}',
                'repro_steps': f'User {analysis["user_bucket"]} with anchor {analysis["anchor_title"]} ({analysis["anchor_year"]})'
            })
    
    def _detect_provider_mismatch(self, analysis: Dict, recs: List[Dict], attrs: List[Dict]):
        """Detect provider mismatch (simplified - would need provider data)."""
        # For now, this is a placeholder since we don't have provider data
        # In a real implementation, this would check streaming availability
        pass
    
    def _detect_franchise_overfit(self, analysis: Dict, recs: List[Dict], attrs: List[Dict]):
        """Detect over-sequels / franchise overfit."""
        # Simple heuristic: check for similar titles or years
        anchor_title = analysis['anchor_title'].lower()
        anchor_year = analysis['anchor_year']
        
        franchise_indicators = []
        for rec in recs:
            rec_title = rec.get('title', '').lower()
            rec_year = rec.get('year', 0)
            
            # Check for common franchise patterns
            if any(word in rec_title for word in ['part', 'chapter', 'sequel', '2', '3', '4', '5']):
                if abs(rec_year - anchor_year) < 5:  # Within 5 years
                    franchise_indicators.append(rec['title'])
        
        if len(franchise_indicators) > 2:  # More than 2 franchise items
            analysis['failures'].append({
                'type': 'franchise_overfit',
                'severity': 'S3',
                'symptoms': f'Franchise overfit: {len(franchise_indicators)} franchise/sequel items recommended',
                'evidence': f'franchise_items={franchise_indicators[:3]}',  # Show first 3
                'repro_steps': f'User {analysis["user_bucket"]} with anchor {analysis["anchor_title"]} ({analysis["anchor_year"]})'
            })
    
    def _detect_niche_misfire(self, analysis: Dict, recs: List[Dict], attrs: List[Dict]):
        """Detect niche misfire (topic mismatch)."""
        # Check for very low content similarity
        cosine_scores = []
        for attr in attrs:
            cosine = attr['content_signals'].get('cosine_similarity', 0)
            if cosine > 0:
                cosine_scores.append(cosine)
        
        if cosine_scores:
            avg_cosine = np.mean(cosine_scores)
            low_sim_ratio = sum(1 for c in cosine_scores if c < 0.3) / len(cosine_scores)
            
            analysis['metrics']['content_similarity'] = {
                'avg_cosine': avg_cosine,
                'low_sim_ratio': low_sim_ratio
            }
            
            if avg_cosine < 0.4:  # Very low average similarity
                analysis['failures'].append({
                    'type': 'niche_misfire',
                    'severity': 'S2',
                    'symptoms': f'Low content similarity: avg cosine={avg_cosine:.3f}',
                    'evidence': f'avg_cosine={avg_cosine:.3f}, low_sim_ratio={low_sim_ratio:.1%}',
                    'repro_steps': f'User {analysis["user_bucket"]} with anchor {analysis["anchor_title"]} ({analysis["anchor_year"]})'
                })
    
    def _detect_long_tail_starvation(self, analysis: Dict, recs: List[Dict], attrs: List[Dict]):
        """Detect long-tail starvation (already covered in popularity bias)."""
        # This is already detected in _detect_popularity_bias
        pass
    
    def _detect_cold_start_miss(self, analysis: Dict, recs: List[Dict], attrs: List[Dict]):
        """Detect cold-start miss."""
        if analysis['user_bucket'] != 'cold_synth':
            return
        
        # Check if cold user got appropriate content-heavy recommendations
        alpha_values = [attr['policy_path'].get('alpha_used', 0) for attr in attrs]
        avg_alpha = np.mean(alpha_values) if alpha_values else 0
        
        analysis['metrics']['cold_start'] = {
            'avg_alpha': avg_alpha,
            'user_bucket': analysis['user_bucket']
        }
        
        if avg_alpha > 0.3:  # Should be content-heavy (low alpha)
            analysis['failures'].append({
                'type': 'cold_start_miss',
                'severity': 'S2',
                'symptoms': f'Cold user got non-content-heavy recs: avg alpha={avg_alpha:.2f} (should be <0.3)',
                'evidence': f'avg_alpha={avg_alpha:.2f}, user_bucket={analysis["user_bucket"]}',
                'repro_steps': f'Cold user {analysis["user_bucket"]} with anchor {analysis["anchor_title"]} ({analysis["anchor_year"]})'
            })
    
    def generate_error_taxonomy(self, analyzed_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive error taxonomy."""
        self.logger.info("Generating error taxonomy")
        
        # Aggregate failures by type
        failure_counts = Counter()
        severity_counts = Counter()
        cohort_failures = defaultdict(Counter)
        
        for case in analyzed_cases:
            for failure in case['failures']:
                failure_type = failure['type']
                severity = failure['severity']
                cohort = case['user_bucket']
                
                failure_counts[failure_type] += 1
                severity_counts[severity] += 1
                cohort_failures[cohort][failure_type] += 1
        
        # Define failure types with detection rules and mitigations
        failure_types = {
            'popularity_bias': {
                'description': 'Recommendations overly concentrated on popular/head items',
                'detection_rules': [
                    'High popularity ratio >70% (items with >100k IMDB votes)',
                    'Low long-tail ratio <20% (items with <1k IMDB votes)'
                ],
                'symptoms': [
                    'Recommendation list dominated by blockbusters',
                    'Lack of niche or independent films',
                    'Poor diversity in recommendation quality'
                ],
                'mitigations': [
                    'Increase diversity weight (λ_div)',
                    'Implement long-tail quota (min 30% <1k votes)',
                    'Add popularity penalty in scoring'
                ],
                'policy_knobs': ['λ_div', 'popularity_penalty', 'tail_quota'],
                'validation_checks': ['long_tail_ratio >= 0.3', 'high_pop_ratio <= 0.6']
            },
            'redundancy': {
                'description': 'Recommendations are too similar to each other',
                'detection_rules': [
                    'High cosine similarity between recommendation pairs',
                    'Low variance in content similarity scores'
                ],
                'symptoms': [
                    'Multiple very similar movies recommended',
                    'Lack of variety in recommendation styles',
                    'User sees repetitive content'
                ],
                'mitigations': [
                    'Implement MMR (Maximal Marginal Relevance)',
                    'Increase diversity penalty in scoring',
                    'Add genre diversity constraints'
                ],
                'policy_knobs': ['mmr_lambda', 'diversity_penalty', 'genre_diversity_weight'],
                'validation_checks': ['redundancy_ratio <= 0.2', 'cosine_variance >= 0.1']
            },
            'temporal_drift': {
                'description': 'Recommendations are temporally misaligned with user preferences',
                'detection_rules': [
                    'Large year gap between anchor and average recommendation',
                    'High ratio of very old movies relative to anchor'
                ],
                'symptoms': [
                    'Recommendations from wrong era',
                    'Mismatch between user temporal preferences and recs',
                    'Outdated content recommendations'
                ],
                'mitigations': [
                    'Add recency boost to scoring',
                    'Implement temporal alignment constraints',
                    'Weight recent movies higher'
                ],
                'policy_knobs': ['recency_boost', 'temporal_alignment_weight', 'year_gap_penalty'],
                'validation_checks': ['year_gap <= 15', 'old_movies_ratio <= 0.3']
            },
            'franchise_overfit': {
                'description': 'Over-recommendation of franchise/sequel content',
                'detection_rules': [
                    'High ratio of franchise/sequel items',
                    'Multiple items from same franchise'
                ],
                'symptoms': [
                    'Too many sequels/prequels recommended',
                    'Lack of original content variety',
                    'Franchise fatigue for users'
                ],
                'mitigations': [
                    'Add franchise penalty in scoring',
                    'Limit franchise items per recommendation list',
                    'Boost original content weight'
                ],
                'policy_knobs': ['franchise_penalty', 'max_franchise_items', 'original_content_boost'],
                'validation_checks': ['franchise_ratio <= 0.3', 'max_franchise_items <= 2']
            },
            'niche_misfire': {
                'description': 'Recommendations don\'t match user\'s niche interests',
                'detection_rules': [
                    'Low average content similarity scores',
                    'High ratio of low-similarity recommendations'
                ],
                'symptoms': [
                    'Recommendations feel random or irrelevant',
                    'Poor content-to-user matching',
                    'Low user engagement with recommendations'
                ],
                'mitigations': [
                    'Improve content similarity computation',
                    'Add niche-specific boosting',
                    'Enhance user preference modeling'
                ],
                'policy_knobs': ['content_similarity_weight', 'niche_boost', 'preference_modeling_weight'],
                'validation_checks': ['avg_cosine >= 0.5', 'low_sim_ratio <= 0.2']
            },
            'cold_start_miss': {
                'description': 'Cold users not getting appropriate content-heavy recommendations',
                'detection_rules': [
                    'Cold users getting high alpha values (>0.3)',
                    'Cold users getting CF-heavy recommendations'
                ],
                'symptoms': [
                    'New users see irrelevant recommendations',
                    'Cold start problem not properly handled',
                    'Poor onboarding experience'
                ],
                'mitigations': [
                    'Enforce content-heavy policy for cold users',
                    'Improve cold start content selection',
                    'Add content diversity for new users'
                ],
                'policy_knobs': ['cold_user_alpha_max', 'cold_start_content_weight', 'new_user_diversity'],
                'validation_checks': ['cold_user_alpha <= 0.25', 'content_recs_ratio >= 0.8']
            }
        }
        
        # Generate taxonomy summary
        taxonomy = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_cases_analyzed': len(analyzed_cases),
                'total_failures_detected': sum(failure_counts.values()),
                'failure_distribution': dict(failure_counts),
                'severity_distribution': dict(severity_counts)
            },
            'failure_types': failure_types,
            'cohort_analysis': dict(cohort_failures),
            'summary_statistics': {
                'most_common_failure': failure_counts.most_common(1)[0] if failure_counts else None,
                'highest_severity_failures': severity_counts.most_common(1)[0] if severity_counts else None,
                'cases_with_failures': sum(1 for case in analyzed_cases if case['failures']),
                'failure_rate': sum(1 for case in analyzed_cases if case['failures']) / len(analyzed_cases) if analyzed_cases else 0
            }
        }
        
        return taxonomy
    
    def generate_error_backlog(self, analyzed_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate error backlog with reproducible cases."""
        self.logger.info("Generating error backlog")
        
        backlog = []
        
        for case in analyzed_cases:
            if not case['failures']:
                continue
            
            # Determine anchor bucket
            anchor_votes = case['anchor_imdb_votes']
            if anchor_votes > 100000:
                anchor_bucket = 'head'
            elif anchor_votes > 10000:
                anchor_bucket = 'mid'
            else:
                anchor_bucket = 'long_tail'
            
            # Determine surface (simplified)
            surface = 'default'  # Would be determined by filters in real implementation
            
            for failure in case['failures']:
                backlog_item = {
                    'case_id': case['case_id'],
                    'cohort': case['user_bucket'],
                    'anchor_bucket': anchor_bucket,
                    'surface': surface,
                    'failure_type': failure['type'],
                    'symptoms': failure['symptoms'],
                    'evidence_refs': {
                        'snapshot_file': f"data/cases/snapshots/{case['case_id']}_combined.json",
                        'attribution_file': f"data/cases/attributions/{case['case_id']}.json",
                        'why_document': f"docs/cases/{case['case_id']}_why.md",
                        'triptych_image': f"docs/img/cases/{case['case_id']}_triptych.png"
                    },
                    'repro_steps': failure['repro_steps'],
                    'severity': failure['severity'],
                    'proposed_fix': self._get_proposed_fix(failure['type']),
                    'owner': 'recommendation_team',
                    'status': 'open',
                    'created_at': datetime.now().isoformat(),
                    'metrics': case['metrics'].get(failure['type'].split('_')[0], {})
                }
                
                backlog.append(backlog_item)
        
        return backlog
    
    def _get_proposed_fix(self, failure_type: str) -> str:
        """Get proposed fix for failure type."""
        fixes = {
            'popularity_bias': 'Implement long-tail quota (30% min) and diversity weighting',
            'redundancy': 'Add MMR diversity penalty and genre diversity constraints',
            'temporal_drift': 'Add recency boost and temporal alignment weighting',
            'franchise_overfit': 'Implement franchise penalty and limit franchise items per list',
            'niche_misfire': 'Improve content similarity computation and niche boosting',
            'cold_start_miss': 'Enforce content-heavy policy (α≤0.25) for cold users'
        }
        return fixes.get(failure_type, 'Review and adjust recommendation parameters')
    
    def generate_mitigation_matrix(self, taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mitigation matrix mapping failures to policy knobs."""
        self.logger.info("Generating mitigation matrix")
        
        matrix = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'policy_version': self.policy.get('version', 'unknown')
            },
            'mitigation_mappings': {}
        }
        
        for failure_type, details in taxonomy['failure_types'].items():
            matrix['mitigation_mappings'][failure_type] = {
                'description': details['description'],
                'policy_knobs': details['policy_knobs'],
                'proposed_values': self._get_proposed_values(failure_type),
                'validation_checks': details['validation_checks'],
                'implementation_priority': self._get_implementation_priority(failure_type)
            }
        
        return matrix
    
    def _get_proposed_values(self, failure_type: str) -> Dict[str, Any]:
        """Get proposed values for policy knobs."""
        values = {
            'popularity_bias': {
                'λ_div': 0.3,
                'popularity_penalty': 0.2,
                'tail_quota': 0.3
            },
            'redundancy': {
                'mmr_lambda': 0.7,
                'diversity_penalty': 0.15,
                'genre_diversity_weight': 0.25
            },
            'temporal_drift': {
                'recency_boost': 0.1,
                'temporal_alignment_weight': 0.2,
                'year_gap_penalty': 0.15
            },
            'franchise_overfit': {
                'franchise_penalty': 0.3,
                'max_franchise_items': 2,
                'original_content_boost': 0.2
            },
            'niche_misfire': {
                'content_similarity_weight': 0.4,
                'niche_boost': 0.15,
                'preference_modeling_weight': 0.3
            },
            'cold_start_miss': {
                'cold_user_alpha_max': 0.25,
                'cold_start_content_weight': 0.8,
                'new_user_diversity': 0.4
            }
        }
        return values.get(failure_type, {})
    
    def _get_implementation_priority(self, failure_type: str) -> str:
        """Get implementation priority for failure type."""
        priorities = {
            'cold_start_miss': 'P0',
            'popularity_bias': 'P1',
            'niche_misfire': 'P1',
            'temporal_drift': 'P2',
            'redundancy': 'P2',
            'franchise_overfit': 'P3'
        }
        return priorities.get(failure_type, 'P3')
    
    def save_taxonomy(self, taxonomy: Dict[str, Any], output_path: str = "docs/step4_error_taxonomy.md"):
        """Save error taxonomy to markdown file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("# Error Taxonomy for Movie Recommendation Optimizer\n\n")
            f.write(f"**Generated**: {taxonomy['metadata']['generated_at']}\n")
            f.write(f"**Cases Analyzed**: {taxonomy['metadata']['total_cases_analyzed']}\n")
            f.write(f"**Total Failures**: {taxonomy['metadata']['total_failures_detected']}\n\n")
            
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Failure Rate**: {taxonomy['summary_statistics']['failure_rate']:.1%}\n")
            f.write(f"- **Cases with Failures**: {taxonomy['summary_statistics']['cases_with_failures']}\n")
            if taxonomy['summary_statistics']['most_common_failure']:
                f.write(f"- **Most Common Failure**: {taxonomy['summary_statistics']['most_common_failure'][0]} ({taxonomy['summary_statistics']['most_common_failure'][1]} cases)\n")
            f.write("\n")
            
            f.write("## Failure Types\n\n")
            for failure_type, details in taxonomy['failure_types'].items():
                f.write(f"### {failure_type.replace('_', ' ').title()}\n\n")
                f.write(f"**Description**: {details['description']}\n\n")
                
                f.write("**Detection Rules**:\n")
                for rule in details['detection_rules']:
                    f.write(f"- {rule}\n")
                f.write("\n")
                
                f.write("**Symptoms**:\n")
                for symptom in details['symptoms']:
                    f.write(f"- {symptom}\n")
                f.write("\n")
                
                f.write("**Mitigations**:\n")
                for mitigation in details['mitigations']:
                    f.write(f"- {mitigation}\n")
                f.write("\n")
                
                f.write(f"**Policy Knobs**: {', '.join(details['policy_knobs'])}\n\n")
                f.write("**Validation Checks**:\n")
                for check in details['validation_checks']:
                    f.write(f"- {check}\n")
                f.write("\n---\n\n")
        
        self.logger.info(f"Saved error taxonomy to {output_file}")
    
    def save_backlog(self, backlog: List[Dict[str, Any]], output_path: str = "data/cases/error_backlog.json"):
        """Save error backlog to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(backlog, f, indent=2)
        
        self.logger.info(f"Saved error backlog to {output_file}")
    
    def save_mitigation_matrix(self, matrix: Dict[str, Any], output_path: str = "docs/step4_mitigation_matrix.md"):
        """Save mitigation matrix to markdown file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("# Mitigation Matrix for Movie Recommendation Optimizer\n\n")
            f.write(f"**Generated**: {matrix['metadata']['generated_at']}\n")
            f.write(f"**Policy Version**: {matrix['metadata']['policy_version']}\n\n")
            
            f.write("## Mitigation Mappings\n\n")
            for failure_type, details in matrix['mitigation_mappings'].items():
                f.write(f"### {failure_type.replace('_', ' ').title()}\n\n")
                f.write(f"**Description**: {details['description']}\n\n")
                
                f.write("**Policy Knobs & Proposed Values**:\n")
                for knob, value in details['proposed_values'].items():
                    f.write(f"- `{knob}`: {value}\n")
                f.write("\n")
                
                f.write("**Validation Checks**:\n")
                for check in details['validation_checks']:
                    f.write(f"- {check}\n")
                f.write("\n")
                
                f.write(f"**Implementation Priority**: {details['implementation_priority']}\n\n")
                f.write("---\n\n")
        
        self.logger.info(f"Saved mitigation matrix to {output_file}")
    
    def generate_evidence_index(self, backlog: List[Dict[str, Any]], output_path: str = "docs/img/cases/_evidence_index.md"):
        """Generate evidence index linking backlog items to files."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("# Evidence Index for Error Backlog\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n")
            f.write(f"**Total Backlog Items**: {len(backlog)}\n\n")
            
            f.write("## Backlog Items by Failure Type\n\n")
            
            # Group by failure type
            by_failure_type = defaultdict(list)
            for item in backlog:
                by_failure_type[item['failure_type']].append(item)
            
            for failure_type, items in by_failure_type.items():
                f.write(f"### {failure_type.replace('_', ' ').title()} ({len(items)} items)\n\n")
                
                for item in items:
                    f.write(f"#### {item['case_id']} (Severity: {item['severity']})\n\n")
                    f.write(f"**Cohort**: {item['cohort']} | **Anchor**: {item['anchor_bucket']} | **Surface**: {item['surface']}\n\n")
                    f.write(f"**Symptoms**: {item['symptoms']}\n\n")
                    f.write(f"**Evidence Files**:\n")
                    for file_type, file_path in item['evidence_refs'].items():
                        f.write(f"- **{file_type.replace('_', ' ').title()}**: `{file_path}`\n")
                    f.write(f"\n**Proposed Fix**: {item['proposed_fix']}\n\n")
                    f.write("---\n\n")
        
        self.logger.info(f"Saved evidence index to {output_file}")
    
    def run_full_analysis(self):
        """Run complete error taxonomy analysis."""
        self.logger.info("Starting full error taxonomy analysis")
        
        # Analyze all cases
        analyzed_cases = self.analyze_all_cases()
        
        # Generate taxonomy
        taxonomy = self.generate_error_taxonomy(analyzed_cases)
        self.save_taxonomy(taxonomy)
        
        # Generate backlog
        backlog = self.generate_error_backlog(analyzed_cases)
        self.save_backlog(backlog)
        
        # Generate mitigation matrix
        matrix = self.generate_mitigation_matrix(taxonomy)
        self.save_mitigation_matrix(matrix)
        
        # Generate evidence index
        self.generate_evidence_index(backlog)
        
        self.logger.info("Error taxonomy analysis completed")
        
        return {
            'analyzed_cases': len(analyzed_cases),
            'total_failures': sum(len(case['failures']) for case in analyzed_cases),
            'backlog_items': len(backlog),
            'failure_types': len(taxonomy['failure_types'])
        }


def main():
    """CLI entrypoint for the error taxonomy analyzer."""
    parser = argparse.ArgumentParser(description='Error Taxonomy Analyzer')
    parser.add_argument('--policy', default='data/hybrid/policy_step4.json', 
                       help='Path to policy file')
    parser.add_argument('--snapshots-dir', default='data/cases/snapshots',
                       help='Directory containing snapshot files')
    parser.add_argument('--attributions-dir', default='data/cases/attributions',
                       help='Directory containing attribution files')
    
    args = parser.parse_args()
    
    try:
        analyzer = ErrorTaxonomyAnalyzer(args.policy)
        results = analyzer.run_full_analysis()
        
        print(f"Error taxonomy analysis completed:")
        print(f"- Analyzed {results['analyzed_cases']} cases")
        print(f"- Detected {results['total_failures']} total failures")
        print(f"- Generated {results['backlog_items']} backlog items")
        print(f"- Identified {results['failure_types']} failure types")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


