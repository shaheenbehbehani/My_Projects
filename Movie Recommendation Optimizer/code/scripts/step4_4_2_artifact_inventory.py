#!/usr/bin/env python3
"""
Step 4.4.2 - Artifact Inventory & Validation
Movie Recommendation Optimizer - Create comprehensive artifact manifest

This module creates a machine-readable artifact manifest that catalogs all Step 4 outputs,
validates file existence and formats, and cross-checks policy references.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class ArtifactInventory:
    """
    Creates comprehensive artifact inventory for all Step 4 outputs.
    """
    
    def __init__(self, project_root: str = "/Users/shaheen/Desktop/Netflix"):
        """Initialize the artifact inventory."""
        self.project_root = Path(project_root)
        self.artifacts = {
            "step4_1_metrics": [],
            "step4_2_case_studies": [],
            "step4_3_edge_cases": [],
            "step4_4_1_final_report": [],
            "logs": [],
            "policies": [],
            "data": []
        }
        self.validation_results = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "missing_files": 0,
            "policy_references": {},
            "errors": []
        }
    
    def discover_artifacts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover all Step 4 artifacts in the project."""
        print("Discovering Step 4 artifacts...")
        
        # Step 4.1 - Offline Metrics
        self._discover_step4_1_artifacts()
        
        # Step 4.2 - Case Studies
        self._discover_step4_2_artifacts()
        
        # Step 4.3 - Edge Cases
        self._discover_step4_3_artifacts()
        
        # Step 4.4.1 - Final Report
        self._discover_step4_4_1_artifacts()
        
        # Logs
        self._discover_log_artifacts()
        
        # Policies
        self._discover_policy_artifacts()
        
        # Data files
        self._discover_data_artifacts()
        
        return self.artifacts
    
    def _discover_step4_1_artifacts(self):
        """Discover Step 4.1 offline metrics artifacts."""
        step4_1_patterns = [
            "docs/step4_summary.md",
            "docs/step4_eval_metrics.md",
            "docs/step4_metrics_framework.md",
            "docs/step4_cf_eval.md",
            "docs/step4_content_eval.md",
            "docs/step4_hybrid_eval.md",
            "docs/step4_stratified_analysis.md",
            "docs/README_snippet_step4.md",
            "data/eval/best_alpha_step4.json",
            "data/eval/cf_eval_coverage_metrics_*.png",
            "data/eval/cf_eval_coverage_metrics_*.png",
            "docs/img/step4_*.png"
        ]
        
        for pattern in step4_1_patterns:
            if "*" in pattern:
                # Handle glob patterns
                import glob
                matches = glob.glob(str(self.project_root / pattern))
                for match in matches:
                    self._add_artifact("step4_1_metrics", match)
            else:
                self._add_artifact("step4_1_metrics", pattern)
    
    def _discover_step4_2_artifacts(self):
        """Discover Step 4.2 case studies artifacts."""
        step4_2_patterns = [
            "docs/step4_case_studies.md",
            "docs/step4_case_checklist.md",
            "docs/step4_error_taxonomy.md",
            "docs/step4_mitigation_matrix.md",
            "docs/policy_step4_case_findings.md",
            "docs/policy_step4_diff.md",
            "docs/policy_step4_proposals_diff.md",
            "data/cases/anchors_case_slate.csv",
            "data/cases/users_case_slate.csv",
            "data/cases/error_backlog.json",
            "data/cases/snapshots/*.json",
            "data/cases/attributions/*.json",
            "docs/img/cases/*.png",
            "docs/cases/*.md"
        ]
        
        for pattern in step4_2_patterns:
            if "*" in pattern:
                import glob
                matches = glob.glob(str(self.project_root / pattern))
                for match in matches:
                    self._add_artifact("step4_2_case_studies", match)
            else:
                self._add_artifact("step4_2_case_studies", pattern)
    
    def _discover_step4_3_artifacts(self):
        """Discover Step 4.3 edge cases artifacts."""
        step4_3_patterns = [
            "docs/step4_edgecases.md",
            "docs/step4_edgecases_analysis.md",
            "docs/step4_edgecases_scenarios.md",
            "data/eval/edge_cases/README.md",
            "data/eval/edge_cases/users.sample.jsonl",
            "data/eval/edge_cases/items.sample.jsonl",
            "data/eval/edge_cases/scenarios.v1.json",
            "data/eval/edge_cases/results/*.json",
            "docs/img/edgecases/*.png",
            "logs/step4_3_edgecases_exec.log"
        ]
        
        for pattern in step4_3_patterns:
            if "*" in pattern:
                import glob
                matches = glob.glob(str(self.project_root / pattern))
                for match in matches:
                    self._add_artifact("step4_3_edge_cases", match)
            else:
                self._add_artifact("step4_3_edge_cases", pattern)
    
    def _discover_step4_4_1_artifacts(self):
        """Discover Step 4.4.1 final report artifacts."""
        step4_4_1_patterns = [
            "docs/step4_final_report.md",
            "docs/step4_3_2_execution_summary.md"
        ]
        
        for pattern in step4_4_1_patterns:
            self._add_artifact("step4_4_1_final_report", pattern)
    
    def _discover_log_artifacts(self):
        """Discover log artifacts."""
        log_patterns = [
            "logs/step4_1_eval.log",
            "logs/step4_2_cases.log",
            "logs/step4_3_edgecases_exec.log",
            "logs/step4_3_edgecases.todo"
        ]
        
        for pattern in log_patterns:
            self._add_artifact("logs", pattern)
    
    def _discover_policy_artifacts(self):
        """Discover policy artifacts."""
        policy_patterns = [
            "data/hybrid/policy_step4.json",
            "data/hybrid/policy_step4_proposals.json",
            "data/hybrid/policy_provisional.json",
            "data/experiments/bucket_config.json",
            "data/controls/runtime_toggles.example.json"
        ]
        
        for pattern in policy_patterns:
            self._add_artifact("policies", pattern)
    
    def _discover_data_artifacts(self):
        """Discover data artifacts."""
        data_patterns = [
            "data/eval/",
            "data/cases/",
            "data/features/",
            "data/hybrid/",
            "data/similarity/",
            "data/collaborative/"
        ]
        
        for pattern in data_patterns:
            if os.path.exists(self.project_root / pattern):
                self._add_artifact("data", pattern)
    
    def _add_artifact(self, category: str, file_path: str):
        """Add an artifact to the inventory."""
        full_path = self.project_root / file_path
        if full_path.exists():
            artifact = {
                "file_path": str(full_path.relative_to(self.project_root)),
                "absolute_path": str(full_path),
                "size_bytes": full_path.stat().st_size,
                "file_type": self._get_file_type(full_path),
                "purpose": self._get_file_purpose(full_path),
                "last_modified": datetime.fromtimestamp(full_path.stat().st_mtime).isoformat(),
                "exists": True
            }
            self.artifacts[category].append(artifact)
        else:
            # Track missing files
            artifact = {
                "file_path": file_path,
                "absolute_path": str(full_path),
                "size_bytes": 0,
                "file_type": self._get_file_type(Path(file_path)),
                "purpose": self._get_file_purpose(Path(file_path)),
                "last_modified": None,
                "exists": False
            }
            self.artifacts[category].append(artifact)
            self.validation_results["missing_files"] += 1
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type based on extension."""
        suffix = file_path.suffix.lower()
        if suffix == '.md':
            return 'MD'
        elif suffix == '.json':
            return 'JSON'
        elif suffix == '.png':
            return 'PNG'
        elif suffix == '.log':
            return 'LOG'
        elif suffix == '.csv':
            return 'CSV'
        elif suffix == '.jsonl':
            return 'JSONL'
        elif suffix == '.py':
            return 'PY'
        else:
            return 'OTHER'
    
    def _get_file_purpose(self, file_path: Path) -> str:
        """Determine file purpose based on path and name."""
        path_str = str(file_path).lower()
        
        if 'step4_summary' in path_str or 'step4_final_report' in path_str:
            return 'Executive Summary'
        elif 'metrics' in path_str or 'eval' in path_str:
            return 'Evaluation Metrics'
        elif 'case' in path_str:
            return 'Case Study Analysis'
        elif 'edge' in path_str:
            return 'Edge Case Testing'
        elif 'policy' in path_str:
            return 'Policy Configuration'
        elif 'log' in path_str:
            return 'Execution Log'
        elif 'img' in path_str or 'png' in path_str:
            return 'Visualization'
        elif 'data' in path_str:
            return 'Data File'
        else:
            return 'Documentation'
    
    def validate_artifacts(self) -> Dict[str, Any]:
        """Validate all artifacts for existence, size, and format."""
        print("Validating artifacts...")
        
        total_files = 0
        valid_files = 0
        invalid_files = 0
        
        for category, artifacts in self.artifacts.items():
            for artifact in artifacts:
                total_files += 1
                self.validation_results["total_files"] = total_files
                
                if artifact["exists"]:
                    if artifact["size_bytes"] > 0:
                        valid_files += 1
                        self.validation_results["valid_files"] = valid_files
                    else:
                        invalid_files += 1
                        self.validation_results["invalid_files"] = invalid_files
                        self.validation_results["errors"].append(f"Empty file: {artifact['file_path']}")
                else:
                    invalid_files += 1
                    self.validation_results["invalid_files"] = invalid_files
                    self.validation_results["errors"].append(f"Missing file: {artifact['file_path']}")
        
        return self.validation_results
    
    def cross_check_policy_references(self) -> Dict[str, Any]:
        """Cross-check policy references across all artifacts."""
        print("Cross-checking policy references...")
        
        policy_refs = {
            "alpha_values": ["0.15", "0.4", "0.6", "0.8"],
            "k_values": ["5", "10", "20", "50"],
            "policy_terms": ["bucket-gate", "hybrid", "cold", "light", "medium", "heavy"]
        }
        
        for category, artifacts in self.artifacts.items():
            category_refs = {
                "alpha_values": [],
                "k_values": [],
                "policy_terms": []
            }
            
            for artifact in artifacts:
                if artifact["exists"] and artifact["file_type"] in ["MD", "JSON", "CSV"]:
                    try:
                        content = self._read_file_content(artifact["absolute_path"])
                        if content:
                            for ref_type, refs in policy_refs.items():
                                for ref in refs:
                                    if ref in content:
                                        category_refs[ref_type].append({
                                            "file": artifact["file_path"],
                                            "reference": ref
                                        })
                    except Exception as e:
                        self.validation_results["errors"].append(f"Error reading {artifact['file_path']}: {e}")
            
            self.validation_results["policy_references"][category] = category_refs
        
        return self.validation_results["policy_references"]
    
    def _read_file_content(self, file_path: str) -> Optional[str]:
        """Read file content for policy reference checking."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None
    
    def generate_markdown_report(self, output_path: str = "docs/step4_artifact_inventory.md"):
        """Generate human-readable markdown inventory report."""
        report = f"""# Step 4.4.2: Artifact Inventory & Validation

**Generated**: {datetime.now().isoformat()}Z  
**Status**: ✅ COMPLETED  
**Total Artifacts**: {self.validation_results['total_files']}  
**Valid Files**: {self.validation_results['valid_files']}  
**Invalid Files**: {self.validation_results['invalid_files']}  
**Missing Files**: {self.validation_results['missing_files']}

## Executive Summary

This inventory catalogs all Step 4 outputs from the Movie Recommendation Optimizer project, including offline metrics, case studies, edge case testing, and the consolidated final report. The manifest provides comprehensive tracking of file locations, sizes, types, and purposes across all evaluation phases.

### Key Statistics
- **Total Artifacts**: {self.validation_results['total_files']}
- **Valid Files**: {self.validation_results['valid_files']} ({self.validation_results['valid_files']/self.validation_results['total_files']*100:.1f}%)
- **Invalid Files**: {self.validation_results['invalid_files']} ({self.validation_results['invalid_files']/self.validation_results['total_files']*100:.1f}%)
- **Missing Files**: {self.validation_results['missing_files']} ({self.validation_results['missing_files']/self.validation_results['total_files']*100:.1f}%)

## Artifact Inventory by Step

### Step 4.1 - Offline Metrics
**Purpose**: Evaluation framework and performance metrics  
**Total Files**: {len(self.artifacts['step4_1_metrics'])}

| File Path | Size (bytes) | Type | Purpose | Status |
|-----------|--------------|------|---------|--------|
"""

        for artifact in self.artifacts['step4_1_metrics']:
            status = "✅" if artifact['exists'] and artifact['size_bytes'] > 0 else "❌"
            report += f"| {artifact['file_path']} | {artifact['size_bytes']:,} | {artifact['file_type']} | {artifact['purpose']} | {status} |\n"

        report += f"""
### Step 4.2 - Case Studies
**Purpose**: Qualitative analysis and error taxonomy  
**Total Files**: {len(self.artifacts['step4_2_case_studies'])}

| File Path | Size (bytes) | Type | Purpose | Status |
|-----------|--------------|------|---------|--------|
"""

        for artifact in self.artifacts['step4_2_case_studies']:
            status = "✅" if artifact['exists'] and artifact['size_bytes'] > 0 else "❌"
            report += f"| {artifact['file_path']} | {artifact['size_bytes']:,} | {artifact['file_type']} | {artifact['purpose']} | {status} |\n"

        report += f"""
### Step 4.3 - Edge Cases
**Purpose**: Robustness testing and edge case validation  
**Total Files**: {len(self.artifacts['step4_3_edge_cases'])}

| File Path | Size (bytes) | Type | Purpose | Status |
|-----------|--------------|------|---------|--------|
"""

        for artifact in self.artifacts['step4_3_edge_cases']:
            status = "✅" if artifact['exists'] and artifact['size_bytes'] > 0 else "❌"
            report += f"| {artifact['file_path']} | {artifact['size_bytes']:,} | {artifact['file_type']} | {artifact['purpose']} | {status} |\n"

        report += f"""
### Step 4.4.1 - Final Report
**Purpose**: Consolidated documentation and executive summary  
**Total Files**: {len(self.artifacts['step4_4_1_final_report'])}

| File Path | Size (bytes) | Type | Purpose | Status |
|-----------|--------------|------|---------|--------|
"""

        for artifact in self.artifacts['step4_4_1_final_report']:
            status = "✅" if artifact['exists'] and artifact['size_bytes'] > 0 else "❌"
            report += f"| {artifact['file_path']} | {artifact['size_bytes']:,} | {artifact['file_type']} | {artifact['purpose']} | {status} |\n"

        report += f"""
### Supporting Files
**Purpose**: Logs, policies, and data files  
**Total Files**: {len(self.artifacts['logs']) + len(self.artifacts['policies']) + len(self.artifacts['data'])}

#### Logs
| File Path | Size (bytes) | Type | Purpose | Status |
|-----------|--------------|------|---------|--------|
"""

        for artifact in self.artifacts['logs']:
            status = "✅" if artifact['exists'] and artifact['size_bytes'] > 0 else "❌"
            report += f"| {artifact['file_path']} | {artifact['size_bytes']:,} | {artifact['file_type']} | {artifact['purpose']} | {status} |\n"

        report += f"""
#### Policies
| File Path | Size (bytes) | Type | Purpose | Status |
|-----------|--------------|------|---------|--------|
"""

        for artifact in self.artifacts['policies']:
            status = "✅" if artifact['exists'] and artifact['size_bytes'] > 0 else "❌"
            report += f"| {artifact['file_path']} | {artifact['size_bytes']:,} | {artifact['file_type']} | {artifact['purpose']} | {status} |\n"

        report += f"""
#### Data Files
| File Path | Size (bytes) | Type | Purpose | Status |
|-----------|--------------|------|---------|--------|
"""

        for artifact in self.artifacts['data']:
            status = "✅" if artifact['exists'] and artifact['size_bytes'] > 0 else "❌"
            report += f"| {artifact['file_path']} | {artifact['size_bytes']:,} | {artifact['file_type']} | {artifact['purpose']} | {status} |\n"

        report += f"""
## Policy Reference Validation

### Alpha Values (α={{0.15, 0.4, 0.6, 0.8}})
"""

        for category, refs in self.validation_results['policy_references'].items():
            alpha_refs = refs.get('alpha_values', [])
            if alpha_refs:
                report += f"\n#### {category.replace('_', ' ').title()}\n"
                for ref in alpha_refs:
                    report += f"- **{ref['file']}**: Contains reference to α={ref['reference']}\n"

        report += f"""
### K Values ({{5, 10, 20, 50}})
"""

        for category, refs in self.validation_results['policy_references'].items():
            k_refs = refs.get('k_values', [])
            if k_refs:
                report += f"\n#### {category.replace('_', ' ').title()}\n"
                for ref in k_refs:
                    report += f"- **{ref['file']}**: Contains reference to K={ref['reference']}\n"

        report += f"""
## Validation Results

### File Validation Summary
- **Total Files Checked**: {self.validation_results['total_files']}
- **Valid Files**: {self.validation_results['valid_files']} ({self.validation_results['valid_files']/self.validation_results['total_files']*100:.1f}%)
- **Invalid Files**: {self.validation_results['invalid_files']} ({self.validation_results['invalid_files']/self.validation_results['total_files']*100:.1f}%)
- **Missing Files**: {self.validation_results['missing_files']} ({self.validation_results['missing_files']/self.validation_results['total_files']*100:.1f}%)

### Validation Errors
"""

        if self.validation_results['errors']:
            for error in self.validation_results['errors']:
                report += f"- ❌ {error}\n"
        else:
            report += "- ✅ No validation errors found\n"

        report += f"""
### Policy Reference Coverage
- **Alpha Values**: Found in {sum(1 for refs in self.validation_results['policy_references'].values() if refs.get('alpha_values'))} categories
- **K Values**: Found in {sum(1 for refs in self.validation_results['policy_references'].values() if refs.get('k_values'))} categories
- **Policy Terms**: Found in {sum(1 for refs in self.validation_results['policy_references'].values() if refs.get('policy_terms'))} categories

## Machine-Readable Manifest

The complete machine-readable manifest is available at: `data/step4_artifact_manifest.json`

## Next Steps

This inventory is ready for **Step 4.4.3 (Handoff Package Creation)** with comprehensive artifact tracking and validation results.

---

**Generated by**: Claude (Anthropic)  
**Project**: Movie Recommendation Optimizer  
**Phase**: Step 4.4.2 - Artifact Inventory & Validation  
**Version**: 1.0  
**Status**: ✅ COMPLETED
"""

        # Write markdown report
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Markdown report generated: {output_path}")
        return report
    
    def generate_json_manifest(self, output_path: str = "data/step4_artifact_manifest.json"):
        """Generate machine-readable JSON manifest."""
        manifest = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "project": "Movie Recommendation Optimizer",
                "phase": "Step 4.4.2 - Artifact Inventory & Validation",
                "version": "1.0",
                "status": "COMPLETED"
            },
            "summary": {
                "total_artifacts": self.validation_results['total_files'],
                "valid_files": self.validation_results['valid_files'],
                "invalid_files": self.validation_results['invalid_files'],
                "missing_files": self.validation_results['missing_files'],
                "validation_success_rate": self.validation_results['valid_files'] / self.validation_results['total_files'] if self.validation_results['total_files'] > 0 else 0
            },
            "artifacts": self.artifacts,
            "validation_results": self.validation_results,
            "policy_references": self.validation_results['policy_references']
        }
        
        # Write JSON manifest
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"JSON manifest generated: {output_path}")
        return manifest


def main():
    """CLI entrypoint for the artifact inventory."""
    parser = argparse.ArgumentParser(description='Step 4.4.2 - Artifact Inventory & Validation')
    parser.add_argument('--project-root', default='/Users/shaheen/Desktop/Netflix',
                       help='Project root directory')
    parser.add_argument('--output-md', default='docs/step4_artifact_inventory.md',
                       help='Output markdown report path')
    parser.add_argument('--output-json', default='data/step4_artifact_manifest.json',
                       help='Output JSON manifest path')
    
    args = parser.parse_args()
    
    try:
        # Initialize inventory
        inventory = ArtifactInventory(args.project_root)
        
        # Discover artifacts
        artifacts = inventory.discover_artifacts()
        
        # Validate artifacts
        validation = inventory.validate_artifacts()
        
        # Cross-check policy references
        policy_refs = inventory.cross_check_policy_references()
        
        # Generate reports
        md_report = inventory.generate_markdown_report(args.output_md)
        json_manifest = inventory.generate_json_manifest(args.output_json)
        
        print(f"Artifact inventory completed successfully!")
        print(f"Total artifacts: {validation['total_files']}")
        print(f"Valid files: {validation['valid_files']}")
        print(f"Invalid files: {validation['invalid_files']}")
        print(f"Missing files: {validation['missing_files']}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

