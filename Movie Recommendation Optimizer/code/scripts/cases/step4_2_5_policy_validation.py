#!/usr/bin/env python3
"""
Step 4.2.5 - Policy Validation & Override Tuning
Movie Recommendation Optimizer - Policy Analysis and Tuning

This module validates the current bucket-gate + override policy behavior
and proposes targeted adjustments to improve robustness.
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

class PolicyValidator:
    """
    Validates policy behavior and proposes tuning adjustments.
    """
    
    def __init__(self, policy_path: str = "data/hybrid/policy_step4.json"):
        """Initialize the policy validator."""
        self.logger = self._setup_logging()
        self.logger.info("Initializing PolicyValidator")
        
        # Load current policy
        self.policy = self._load_policy(policy_path)
        
        # Load error backlog
        self.backlog = self._load_backlog()
        
        # Load mitigation matrix
        self.mitigation_matrix = self._load_mitigation_matrix()
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        self.logger.info("Policy validator initialization completed")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the policy validator."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("policy_validator")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('logs/step4_2_5_policy_validation.log')
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
        """Load the current policy configuration."""
        try:
            with open(policy_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load policy: {e}")
            raise
    
    def _load_backlog(self) -> List[Dict[str, Any]]:
        """Load error backlog."""
        try:
            with open('data/cases/error_backlog.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load backlog: {e}")
            return []
    
    def _load_mitigation_matrix(self) -> Dict[str, Any]:
        """Load mitigation matrix."""
        try:
            with open('docs/step4_mitigation_matrix.md', 'r') as f:
                # Parse markdown to extract mitigation mappings
                content = f.read()
                # This is a simplified parser - in practice would use proper markdown parsing
                return {"content": content}
        except Exception as e:
            self.logger.error(f"Failed to load mitigation matrix: {e}")
            return {}
    
    def analyze_policy_behavior(self) -> Dict[str, Any]:
        """Analyze current policy behavior across case studies."""
        self.logger.info("Analyzing policy behavior")
        
        # Load sample attributions for analysis
        attributions_dir = Path('data/cases/attributions')
        attribution_files = list(attributions_dir.glob('*.json'))
        
        analysis = {
            'metadata': {
                'analyzed_at': datetime.now().isoformat(),
                'policy_version': self.policy.get('version', 'unknown'),
                'total_cases': len(attribution_files)
            },
            'alpha_usage': Counter(),
            'override_triggers': Counter(),
            'cohort_behavior': defaultdict(list),
            'override_effectiveness': {},
            'failure_correlations': defaultdict(list),
            'specific_issues': []
        }
        
        # Analyze each case
        for attr_file in attribution_files[:50]:  # Sample 50 cases for analysis
            try:
                with open(attr_file, 'r') as f:
                    attribution = json.load(f)
                
                self._analyze_single_case(attribution, analysis)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze case {attr_file}: {e}")
        
        # Convert counters to regular dicts for JSON serialization
        analysis['alpha_usage'] = dict(analysis['alpha_usage'])
        analysis['override_triggers'] = dict(analysis['override_triggers'])
        analysis['cohort_behavior'] = dict(analysis['cohort_behavior'])
        
        return analysis
    
    def _analyze_single_case(self, attribution: Dict[str, Any], analysis: Dict[str, Any]):
        """Analyze a single case for policy behavior."""
        case_id = attribution['case_id']
        user_bucket = attribution['user_bucket']
        
        # Collect alpha values and overrides
        alphas = []
        overrides_used = set()
        
        for attr in attribution['attributions']:
            policy_path = attr['policy_path']
            alpha = policy_path.get('alpha_used', 0.5)
            overrides = policy_path.get('overrides_applied', [])
            
            alphas.append(alpha)
            overrides_used.update(overrides)
        
        if alphas:
            avg_alpha = np.mean(alphas)
            analysis['alpha_usage'][avg_alpha] += 1
            analysis['cohort_behavior'][user_bucket].append(avg_alpha)
        
        for override in overrides_used:
            analysis['override_triggers'][override] += 1
        
        # Check for specific policy issues
        self._check_cold_start_issues(case_id, user_bucket, alphas, analysis)
        self._check_long_tail_issues(case_id, user_bucket, overrides_used, analysis)
        self._check_alpha_consistency(case_id, user_bucket, alphas, analysis)
    
    def _check_cold_start_issues(self, case_id: str, user_bucket: str, alphas: List[float], analysis: Dict):
        """Check for cold start policy issues."""
        if user_bucket == 'cold_synth':
            avg_alpha = np.mean(alphas) if alphas else 0.5
            expected_alpha = self.policy['alpha_map'].get('cold', 0.2)
            
            if avg_alpha > expected_alpha + 0.1:  # Allow small tolerance
                analysis['specific_issues'].append({
                    'case_id': case_id,
                    'issue_type': 'cold_start_alpha_too_high',
                    'expected_alpha': expected_alpha,
                    'actual_alpha': avg_alpha,
                    'severity': 'S2'
                })
    
    def _check_long_tail_issues(self, case_id: str, user_bucket: str, overrides: set, analysis: Dict):
        """Check for long-tail policy issues."""
        if 'long_tail_override' in overrides:
            # Check if this case appears in backlog for long-tail starvation
            backlog_item = next((item for item in self.backlog 
                               if item['case_id'] == case_id and 
                               item['failure_type'] == 'long_tail_starvation'), None)
            
            if backlog_item:
                analysis['specific_issues'].append({
                    'case_id': case_id,
                    'issue_type': 'long_tail_override_ineffective',
                    'override_triggered': True,
                    'still_has_starvation': True,
                    'severity': 'S3'
                })
    
    def _check_alpha_consistency(self, case_id: str, user_bucket: str, alphas: List[float], analysis: Dict):
        """Check for alpha consistency within a case."""
        if len(set(alphas)) > 1:  # Multiple alpha values in same case
            analysis['specific_issues'].append({
                'case_id': case_id,
                'issue_type': 'alpha_inconsistency',
                'alphas_used': alphas,
                'user_bucket': user_bucket,
                'severity': 'S3'
            })
    
    def validate_override_effectiveness(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate effectiveness of current overrides."""
        self.logger.info("Validating override effectiveness")
        
        override_analysis = {
            'minimal_history_guardrail': {
                'triggered_cases': 0,
                'effective_cases': 0,
                'ineffective_cases': 0,
                'evidence': []
            },
            'long_tail_override': {
                'triggered_cases': 0,
                'effective_cases': 0,
                'ineffective_cases': 0,
                'evidence': []
            }
        }
        
        # Analyze minimal history guardrail
        cold_cases = [item for item in self.backlog if item['cohort'] == 'cold_synth']
        cold_start_misses = [item for item in cold_cases if item['failure_type'] == 'cold_start_miss']
        
        override_analysis['minimal_history_guardrail']['triggered_cases'] = len(cold_cases)
        override_analysis['minimal_history_guardrail']['ineffective_cases'] = len(cold_start_misses)
        override_analysis['minimal_history_guardrail']['effective_cases'] = len(cold_cases) - len(cold_start_misses)
        
        # Add evidence
        for item in cold_start_misses[:3]:  # Top 3 examples
            override_analysis['minimal_history_guardrail']['evidence'].append({
                'case_id': item['case_id'],
                'symptoms': item['symptoms'],
                'proposed_fix': item['proposed_fix']
            })
        
        # Analyze long-tail override
        long_tail_cases = [item for item in self.backlog if 'long_tail_override' in item.get('evidence_refs', {})]
        long_tail_starvation = [item for item in long_tail_cases if item['failure_type'] == 'long_tail_starvation']
        
        override_analysis['long_tail_override']['triggered_cases'] = len(long_tail_cases)
        override_analysis['long_tail_override']['ineffective_cases'] = len(long_tail_starvation)
        override_analysis['long_tail_override']['effective_cases'] = len(long_tail_cases) - len(long_tail_starvation)
        
        # Add evidence
        for item in long_tail_starvation[:3]:  # Top 3 examples
            override_analysis['long_tail_override']['evidence'].append({
                'case_id': item['case_id'],
                'symptoms': item['symptoms'],
                'proposed_fix': item['proposed_fix']
            })
        
        return override_analysis
    
    def generate_policy_findings(self, analysis: Dict[str, Any], override_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive policy findings."""
        self.logger.info("Generating policy findings")
        
        findings = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'policy_version': self.policy.get('version', 'unknown'),
                'analysis_summary': {
                    'total_cases_analyzed': analysis['metadata']['total_cases'],
                    'specific_issues_found': len(analysis['specific_issues']),
                    'override_effectiveness': override_analysis
                }
            },
            'override_validation': {},
            'parameter_assessment': {},
            'recommendations': [],
            'evidence_cases': []
        }
        
        # Validate each override
        for override_name, override_data in override_analysis.items():
            effectiveness_rate = override_data['effective_cases'] / max(override_data['triggered_cases'], 1)
            
            findings['override_validation'][override_name] = {
                'status': 'PASS' if effectiveness_rate >= 0.7 else 'FAIL',
                'effectiveness_rate': effectiveness_rate,
                'triggered_cases': override_data['triggered_cases'],
                'effective_cases': override_data['effective_cases'],
                'ineffective_cases': override_data['ineffective_cases'],
                'evidence': override_data['evidence']
            }
        
        # Assess current parameters
        findings['parameter_assessment'] = self._assess_parameters(analysis)
        
        # Generate recommendations
        findings['recommendations'] = self._generate_recommendations(analysis, override_analysis)
        
        # Collect evidence cases
        findings['evidence_cases'] = analysis['specific_issues'][:5]  # Top 5 issues
        
        return findings
    
    def _assess_parameters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current parameter values."""
        assessment = {}
        
        # Assess alpha values
        alpha_map = self.policy['alpha_map']
        cohort_behavior = analysis['cohort_behavior']
        
        for cohort, expected_alpha in alpha_map.items():
            if cohort in cohort_behavior:
                actual_alphas = cohort_behavior[cohort]
                avg_actual = np.mean(actual_alphas)
                std_actual = np.std(actual_alphas)
                
                assessment[f'{cohort}_alpha'] = {
                    'expected': expected_alpha,
                    'actual_avg': avg_actual,
                    'actual_std': std_actual,
                    'status': 'GOOD' if abs(avg_actual - expected_alpha) < 0.05 else 'NEEDS_TUNING',
                    'recommendation': self._get_alpha_recommendation(cohort, expected_alpha, avg_actual)
                }
        
        return assessment
    
    def _get_alpha_recommendation(self, cohort: str, expected: float, actual: float) -> str:
        """Get alpha value recommendation."""
        if abs(actual - expected) < 0.05:
            return "No change needed"
        elif actual > expected + 0.1:
            return f"Consider reducing to {expected - 0.05:.2f} for stricter content-heavy policy"
        else:
            return f"Consider increasing to {expected + 0.05:.2f} for more collaborative signals"
    
    def _generate_recommendations(self, analysis: Dict[str, Any], override_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate policy recommendations."""
        recommendations = []
        
        # Cold start recommendations
        cold_start_issues = len([item for item in self.backlog if item['failure_type'] == 'cold_start_miss'])
        if cold_start_issues > 0:
            recommendations.append({
                'priority': 'P0',
                'parameter': 'cold_user_alpha_max',
                'current_value': 0.2,
                'proposed_value': 0.15,
                'rationale': f'Reduce alpha to address {cold_start_issues} cold-start miss cases',
                'expected_impact': 'Reduce cold-start failures by 20-30%'
            })
        
        # Long-tail recommendations
        long_tail_issues = len([item for item in self.backlog if item['failure_type'] == 'long_tail_starvation'])
        if long_tail_issues > 0:
            recommendations.append({
                'priority': 'P1',
                'parameter': 'tail_quota',
                'current_value': 'not_implemented',
                'proposed_value': 0.3,
                'rationale': f'Implement long-tail quota to address {long_tail_issues} starvation cases',
                'expected_impact': 'Ensure 30% of recommendations are long-tail items'
            })
        
        # Redundancy recommendations
        redundancy_issues = len([item for item in self.backlog if item['failure_type'] == 'redundancy'])
        if redundancy_issues > 0:
            recommendations.append({
                'priority': 'P1',
                'parameter': 'mmr_lambda',
                'current_value': 'not_implemented',
                'proposed_value': 0.7,
                'rationale': f'Implement MMR diversity to address {redundancy_issues} redundancy cases',
                'expected_impact': 'Reduce redundant recommendations by 40-50%'
            })
        
        # Temporal drift recommendations
        temporal_issues = len([item for item in self.backlog if item['failure_type'] == 'temporal_drift'])
        if temporal_issues > 0:
            recommendations.append({
                'priority': 'P2',
                'parameter': 'recency_boost',
                'current_value': 'not_implemented',
                'proposed_value': 0.1,
                'rationale': f'Add recency boost to address {temporal_issues} temporal drift cases',
                'expected_impact': 'Improve temporal alignment by 25-35%'
            })
        
        return recommendations
    
    def generate_policy_proposals(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate policy proposals with new parameter values."""
        self.logger.info("Generating policy proposals")
        
        # Start with current policy
        proposals = self.policy.copy()
        
        # Apply recommended changes
        recommendations = findings['recommendations']
        
        # Update alpha values
        for rec in recommendations:
            if rec['parameter'] == 'cold_user_alpha_max':
                proposals['alpha_map']['cold'] = rec['proposed_value']
        
        # Add new parameters
        if 'new_parameters' not in proposals:
            proposals['new_parameters'] = {}
        
        for rec in recommendations:
            if rec['parameter'] not in ['cold_user_alpha_max']:  # Already handled above
                proposals['new_parameters'][rec['parameter']] = {
                    'value': rec['proposed_value'],
                    'priority': rec['priority'],
                    'rationale': rec['rationale'],
                    'expected_impact': rec['expected_impact']
                }
        
        # Update metadata
        proposals['version'] = '2.1'
        proposals['updated_at'] = datetime.now().isoformat()
        proposals['update_reason'] = 'Policy tuning based on case study analysis'
        
        return proposals
    
    def generate_diff_document(self, current_policy: Dict[str, Any], proposals: Dict[str, Any]) -> str:
        """Generate diff document showing changes."""
        diff_content = []
        diff_content.append("# Policy Step 4 Proposals - Changes Summary")
        diff_content.append("")
        diff_content.append(f"**Generated**: {datetime.now().isoformat()}")
        diff_content.append(f"**From Version**: {current_policy.get('version', 'unknown')}")
        diff_content.append(f"**To Version**: {proposals.get('version', 'unknown')}")
        diff_content.append("")
        
        # Alpha changes
        current_alphas = current_policy.get('alpha_map', {})
        proposed_alphas = proposals.get('alpha_map', {})
        
        diff_content.append("## Alpha Value Changes")
        diff_content.append("")
        for cohort in current_alphas:
            current_val = current_alphas[cohort]
            proposed_val = proposed_alphas.get(cohort, current_val)
            if current_val != proposed_val:
                diff_content.append(f"- **{cohort}**: {current_val} → {proposed_val}")
            else:
                diff_content.append(f"- **{cohort}**: {current_val} (no change)")
        diff_content.append("")
        
        # New parameters
        new_params = proposals.get('new_parameters', {})
        if new_params:
            diff_content.append("## New Parameters")
            diff_content.append("")
            for param_name, param_data in new_params.items():
                diff_content.append(f"- **{param_name}**: {param_data['value']}")
                diff_content.append(f"  - Priority: {param_data['priority']}")
                diff_content.append(f"  - Rationale: {param_data['rationale']}")
                diff_content.append(f"  - Expected Impact: {param_data['expected_impact']}")
                diff_content.append("")
        
        # Override rule changes
        diff_content.append("## Override Rule Assessment")
        diff_content.append("")
        diff_content.append("Current override rules remain unchanged, but effectiveness validated:")
        for rule_name, rule in current_policy.get('override_rules', {}).items():
            diff_content.append(f"- **{rule_name}**: {rule['description']}")
        diff_content.append("")
        
        return "\n".join(diff_content)
    
    def save_findings(self, findings: Dict[str, Any], output_path: str = "docs/policy_step4_case_findings.md"):
        """Save policy findings to markdown file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("# Policy Step 4 Case Findings\n\n")
            f.write(f"**Generated**: {findings['metadata']['generated_at']}\n")
            f.write(f"**Policy Version**: {findings['metadata']['policy_version']}\n")
            f.write(f"**Cases Analyzed**: {findings['metadata']['analysis_summary']['total_cases_analyzed']}\n\n")
            
            f.write("## Override Validation Results\n\n")
            for override_name, validation in findings['override_validation'].items():
                f.write(f"### {override_name.replace('_', ' ').title()}\n\n")
                f.write(f"**Status**: {validation['status']}\n")
                f.write(f"**Effectiveness Rate**: {validation['effectiveness_rate']:.1%}\n")
                f.write(f"**Triggered Cases**: {validation['triggered_cases']}\n")
                f.write(f"**Effective Cases**: {validation['effective_cases']}\n")
                f.write(f"**Ineffective Cases**: {validation['ineffective_cases']}\n\n")
                
                if validation['evidence']:
                    f.write("**Evidence Cases**:\n")
                    for evidence in validation['evidence']:
                        f.write(f"- {evidence['case_id']}: {evidence['symptoms']}\n")
                    f.write("\n")
            
            f.write("## Parameter Assessment\n\n")
            for param_name, assessment in findings['parameter_assessment'].items():
                f.write(f"### {param_name.replace('_', ' ').title()}\n\n")
                f.write(f"**Status**: {assessment['status']}\n")
                f.write(f"**Expected**: {assessment['expected']}\n")
                f.write(f"**Actual Average**: {assessment['actual_avg']:.3f}\n")
                f.write(f"**Actual Std**: {assessment['actual_std']:.3f}\n")
                f.write(f"**Recommendation**: {assessment['recommendation']}\n\n")
            
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(findings['recommendations'], 1):
                f.write(f"### {i}. {rec['parameter']} (Priority: {rec['priority']})\n\n")
                f.write(f"**Current Value**: {rec['current_value']}\n")
                f.write(f"**Proposed Value**: {rec['proposed_value']}\n")
                f.write(f"**Rationale**: {rec['rationale']}\n")
                f.write(f"**Expected Impact**: {rec['expected_impact']}\n\n")
        
        self.logger.info(f"Saved policy findings to {output_file}")
    
    def save_proposals(self, proposals: Dict[str, Any], output_path: str = "data/hybrid/policy_step4_proposals.json"):
        """Save policy proposals to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(proposals, f, indent=2)
        
        self.logger.info(f"Saved policy proposals to {output_file}")
    
    def save_diff(self, diff_content: str, output_path: str = "docs/policy_step4_proposals_diff.md"):
        """Save diff document to markdown file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(diff_content)
        
        self.logger.info(f"Saved diff document to {output_file}")
    
    def run_full_validation(self):
        """Run complete policy validation and tuning analysis."""
        self.logger.info("Starting full policy validation")
        
        # Analyze policy behavior
        analysis = self.analyze_policy_behavior()
        
        # Validate override effectiveness
        override_analysis = self.validate_override_effectiveness(analysis)
        
        # Generate findings
        findings = self.generate_policy_findings(analysis, override_analysis)
        self.save_findings(findings)
        
        # Generate proposals
        proposals = self.generate_policy_proposals(findings)
        self.save_proposals(proposals)
        
        # Generate diff document
        diff_content = self.generate_diff_document(self.policy, proposals)
        self.save_diff(diff_content)
        
        self.logger.info("Policy validation completed")
        
        return {
            'findings_generated': True,
            'proposals_generated': True,
            'diff_generated': True,
            'recommendations_count': len(findings['recommendations']),
            'override_validation': findings['override_validation']
        }


def main():
    """CLI entrypoint for the policy validator."""
    parser = argparse.ArgumentParser(description='Policy Validator')
    parser.add_argument('--policy', default='data/hybrid/policy_step4.json', 
                       help='Path to policy file')
    
    args = parser.parse_args()
    
    try:
        validator = PolicyValidator(args.policy)
        results = validator.run_full_validation()
        
        print(f"Policy validation completed:")
        print(f"- Findings generated: {results['findings_generated']}")
        print(f"- Proposals generated: {results['proposals_generated']}")
        print(f"- Diff generated: {results['diff_generated']}")
        print(f"- Recommendations: {results['recommendations_count']}")
        
        print("\nOverride Validation Results:")
        for override, validation in results['override_validation'].items():
            print(f"- {override}: {validation['status']} ({validation['effectiveness_rate']:.1%})")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


