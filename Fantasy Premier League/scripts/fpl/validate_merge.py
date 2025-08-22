#!/usr/bin/env python3
"""
FPL Merge Validation Script - Step 3b

This script validates the merged dataset created in Step 3a to ensure data quality,
coverage, and schema integrity.

Validation Requirements:
1. Coverage: ≥95% of FPL players linked
2. Uniqueness: No duplicate (player_id, gameweek) rows
3. Foreign Keys: All team_id values exist in teams_processed
4. Schema: Required columns exist with correct dtypes
5. Gameweek ranges: 1-38
6. Missing values audit for key fields

Outputs:
- outputs/fpl_validation.json: Machine-readable validation report
- outputs/fpl_validation.md: Human-readable validation summary
- Console logging with quick inspection summary
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FPLMergeValidator:
    """Validates the merged FPL dataset for quality, coverage, and integrity."""
    
    def __init__(self, data_dir: str = "data", outputs_dir: str = "outputs"):
        self.data_dir = Path(data_dir)
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Validation results storage
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PENDING',
            'coverage_percentage': 0.0,
            'validation_checks': {},
            'coverage_tables': {},
            'null_summary': {},
            'warnings': [],
            'errors': []
        }
        
        # Required columns for schema validation
        self.required_columns = [
            'player_id', 'player_name', 'team_id', 'position', 
            'price', 'availability_status', 'gameweek', 'xP'
        ]
        
        # Numeric columns for dtype validation
        self.numeric_columns = [
            'price', 'xP', 'weekly_wages', 'annual_wages', 'club_value',
            'stadium_capacity', 'avg_attendance'
        ]
        
        # Key fields for null audit
        self.key_fields = [
            'player_id', 'team_id', 'position', 'price', 'xP', 'gameweek'
        ]
        
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the merged dataset and reference datasets."""
        logger.info("Loading datasets for validation...")
        
        # Load merged dataset
        merged_path = self.data_dir / "FPL" / "processed" / "fpl_merged.parquet"
        if not merged_path.exists():
            raise FileNotFoundError(f"Merged dataset not found: {merged_path}")
        
        merged_data = pd.read_parquet(merged_path)
        logger.info(f"Loaded merged dataset: {merged_data.shape}")
        
        # Load reference datasets
        teams_path = self.data_dir / "FPL" / "processed" / "teams_processed.parquet"
        if not teams_path.exists():
            raise FileNotFoundError(f"Teams dataset not found: {teams_path}")
        
        teams_data = pd.read_parquet(teams_path)
        logger.info(f"Loaded teams dataset: {teams_data.shape}")
        
        return merged_data, teams_data
    
    def check_coverage(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Check coverage metrics for players and teams."""
        logger.info("Checking coverage metrics...")
        
        # Total FPL players (from projections)
        total_fpl_players = merged_data['player_id'].nunique()
        
        # Players with complete data (no nulls in key fields)
        complete_players = merged_data.dropna(subset=self.key_fields)['player_id'].nunique()
        
        # Coverage percentage
        coverage_pct = (complete_players / total_fpl_players) * 100 if total_fpl_players > 0 else 0
        
        # Check if coverage meets requirement (≥95%)
        coverage_meets_requirement = coverage_pct >= 95
        
        coverage_results = {
            'total_fpl_players': total_fpl_players,
            'complete_players': complete_players,
            'coverage_percentage': coverage_pct,
            'meets_requirement': coverage_meets_requirement,
            'requirement_threshold': 95.0
        }
        
        # Store in validation results
        self.validation_results['coverage_percentage'] = coverage_pct
        self.validation_results['validation_checks']['coverage'] = coverage_results
        
        return coverage_results
    
    def check_uniqueness(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate (player_id, gameweek) rows."""
        logger.info("Checking for duplicate player-gameweek combinations...")
        
        # Check for duplicates
        duplicate_check = merged_data.duplicated(subset=['player_id', 'gameweek'], keep=False)
        duplicate_count = duplicate_check.sum()
        
        # Get duplicate details if any exist
        duplicates = None
        if duplicate_count > 0:
            duplicates = merged_data[duplicate_check][['player_id', 'player_name', 'gameweek']].head(10)
        
        uniqueness_results = {
            'duplicate_count': duplicate_count,
            'has_duplicates': duplicate_count > 0,
            'duplicate_examples': duplicates.to_dict('records') if duplicates is not None else None
        }
        
        self.validation_results['validation_checks']['uniqueness'] = uniqueness_results
        return uniqueness_results
    
    def check_foreign_keys(self, merged_data: pd.DataFrame, teams_data: pd.DataFrame) -> Dict[str, Any]:
        """Verify all team_id values exist in teams_processed."""
        logger.info("Checking foreign key integrity...")
        
        # Get unique team IDs from merged data
        merged_team_ids = set(merged_data['team_id'].unique())
        reference_team_ids = set(teams_data['team_id'].unique())
        
        # Find missing team IDs
        missing_team_ids = merged_team_ids - reference_team_ids
        
        # Check if all foreign keys are valid
        all_keys_valid = len(missing_team_ids) == 0
        
        foreign_key_results = {
            'merged_team_count': len(merged_team_ids),
            'reference_team_count': len(reference_team_ids),
            'missing_team_ids': list(missing_team_ids),
            'all_keys_valid': all_keys_valid
        }
        
        self.validation_results['validation_checks']['foreign_keys'] = foreign_key_results
        return foreign_key_results
    
    def check_schema(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Check schema requirements and column types."""
        logger.info("Checking schema requirements...")
        
        # Check required columns exist
        existing_columns = set(merged_data.columns)
        missing_columns = set(self.required_columns) - existing_columns
        all_required_columns_exist = len(missing_columns) == 0
        
        # Check numeric column dtypes
        dtype_issues = []
        for col in self.numeric_columns:
            if col in merged_data.columns:
                if not pd.api.types.is_numeric_dtype(merged_data[col]):
                    dtype_issues.append(f"{col}: {merged_data[col].dtype}")
        
        # Check gameweek range
        gameweek_range_valid = True
        if 'gameweek' in merged_data.columns:
            min_gw = merged_data['gameweek'].min()
            max_gw = merged_data['gameweek'].max()
            gameweek_range_valid = 1 <= min_gw <= max_gw <= 38
            gameweek_range = {'min': min_gw, 'max': max_gw}
        else:
            gameweek_range = None
        
        schema_results = {
            'all_required_columns_exist': all_required_columns_exist,
            'missing_columns': list(missing_columns),
            'dtype_issues': dtype_issues,
            'gameweek_range_valid': gameweek_range_valid,
            'gameweek_range': gameweek_range,
            'total_columns': len(merged_data.columns),
            'required_columns_count': len(self.required_columns)
        }
        
        self.validation_results['validation_checks']['schema'] = schema_results
        return schema_results
    
    def generate_coverage_tables(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate per-team and per-position coverage tables."""
        logger.info("Generating coverage tables...")
        
        # Per-team coverage
        team_coverage = merged_data.groupby('team_id').agg({
            'player_id': 'nunique',
            'position': 'count'
        }).reset_index()
        
        # Add team names
        teams_path = self.data_dir / "FPL" / "processed" / "teams_processed.parquet"
        teams_data = pd.read_parquet(teams_path)
        team_coverage = team_coverage.merge(
            teams_data[['team_id', 'team_name']], 
            on='team_id', 
            how='left'
        )
        
        # Per-position coverage
        position_coverage = merged_data.groupby('position').agg({
            'player_id': 'nunique',
            'team_id': 'nunique'
        }).reset_index()
        
        coverage_tables = {
            'per_team': team_coverage.to_dict('records'),
            'per_position': position_coverage.to_dict('records')
        }
        
        self.validation_results['coverage_tables'] = coverage_tables
        return coverage_tables
    
    def audit_null_values(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Audit null values in key fields."""
        logger.info("Auditing null values...")
        
        null_summary = {}
        for field in self.key_fields:
            if field in merged_data.columns:
                null_count = merged_data[field].isnull().sum()
                total_count = len(merged_data)
                null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
                
                null_summary[field] = {
                    'null_count': null_count,
                    'total_count': total_count,
                    'null_percentage': null_percentage
                }
        
        # Overall null summary
        overall_nulls = merged_data[self.key_fields].isnull().sum().sum()
        total_possible_nulls = len(merged_data) * len(self.key_fields)
        overall_null_percentage = (overall_nulls / total_possible_nulls) * 100 if total_possible_nulls > 0 else 0
        
        null_summary['overall'] = {
            'total_nulls': overall_nulls,
            'total_possible_nulls': total_possible_nulls,
            'overall_null_percentage': overall_null_percentage
        }
        
        self.validation_results['null_summary'] = null_summary
        return null_summary
    
    def check_unresolved_entities(self) -> Dict[str, Any]:
        """Check if there are any unresolved entities from Step 3a."""
        logger.info("Checking for unresolved entities...")
        
        unmatched_path = self.outputs_dir / "fpl_unmatched.csv"
        
        if unmatched_path.exists():
            unmatched_data = pd.read_csv(unmatched_path)
            unresolved_count = len(unmatched_data)
            has_unresolved = unresolved_count > 0
        else:
            unresolved_count = 0
            has_unresolved = False
            unmatched_data = None
        
        unresolved_results = {
            'has_unresolved_entities': has_unresolved,
            'unresolved_count': unresolved_count,
            'unresolved_file_path': str(unmatched_path) if unmatched_path.exists() else None,
            'unresolved_entities': unmatched_data.to_dict('records') if unmatched_data is not None else None
        }
        
        self.validation_results['validation_checks']['unresolved_entities'] = unresolved_results
        return unresolved_results
    
    def determine_overall_status(self) -> str:
        """Determine overall validation status."""
        checks = self.validation_results['validation_checks']
        
        # Check critical requirements
        coverage_ok = checks.get('coverage', {}).get('meets_requirement', False)
        no_duplicates = not checks.get('uniqueness', {}).get('has_duplicates', True)
        all_keys_valid = checks.get('foreign_keys', {}).get('all_keys_valid', False)
        schema_ok = checks.get('schema', {}).get('all_required_columns_exist', False)
        
        if coverage_ok and no_duplicates and all_keys_valid and schema_ok:
            status = 'PASSED'
        elif not coverage_ok or not no_duplicates:
            status = 'FAILED'
        else:
            status = 'WARNING'
        
        self.validation_results['overall_status'] = status
        return status
    
    def generate_warnings_and_errors(self) -> None:
        """Generate warnings and errors based on validation results."""
        checks = self.validation_results['validation_checks']
        
        # Coverage warnings
        if not checks.get('coverage', {}).get('meets_requirement', False):
            self.validation_results['errors'].append(
                f"Coverage requirement not met: {self.validation_results['coverage_percentage']:.1f}% < 95%"
            )
        
        # Duplicate warnings
        if checks.get('uniqueness', {}).get('has_duplicates', False):
            self.validation_results['errors'].append(
                f"Duplicate player-gameweek rows found: {checks['uniqueness']['duplicate_count']}"
            )
        
        # Foreign key warnings
        if not checks.get('foreign_keys', {}).get('all_keys_valid', False):
            missing_ids = checks['foreign_keys']['missing_team_ids']
            self.validation_results['warnings'].append(
                f"Invalid team IDs found: {missing_ids}"
            )
        
        # Schema warnings
        if not checks.get('schema', {}).get('all_required_columns_exist', False):
            missing_cols = checks['schema']['missing_columns']
            self.validation_results['warnings'].append(
                f"Missing required columns: {missing_cols}"
            )
        
        # Null value warnings
        null_summary = self.validation_results['null_summary']
        for field, stats in null_summary.items():
            if field != 'overall' and stats['null_percentage'] > 10:
                self.validation_results['warnings'].append(
                    f"High null percentage in {field}: {stats['null_percentage']:.1f}%"
                )
    
    def save_json_report(self) -> None:
        """Save machine-readable validation report to JSON."""
        json_path = self.outputs_dir / "fpl_validation.json"
        
        with open(json_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Saved JSON validation report to {json_path}")
    
    def save_markdown_report(self) -> None:
        """Save human-readable validation summary to Markdown."""
        md_path = self.outputs_dir / "fpl_validation.md"
        
        # Generate markdown content
        md_content = self._generate_markdown_content()
        
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        logger.info(f"Saved Markdown validation report to {md_path}")
    
    def _generate_markdown_content(self) -> str:
        """Generate markdown content for the validation report."""
        results = self.validation_results
        checks = results['validation_checks']
        
        # Status emoji
        status_emoji = {
            'PASSED': '✅',
            'WARNING': '⚠️',
            'FAILED': '❌'
        }
        
        content = f"""# FPL Merge Validation Report - Step 3b

**Generated**: {results['timestamp']}  
**Overall Status**: {status_emoji.get(results['overall_status'], '❓')} {results['overall_status']}

## Executive Summary

- **Coverage**: {results['coverage_percentage']:.1f}% {'✅' if results['coverage_percentage'] >= 95 else '❌'}
- **Duplicates**: {'None found ✅' if not checks.get('uniqueness', {}).get('has_duplicates', False) else f"{checks['uniqueness']['duplicate_count']} found ❌"}
- **Schema**: {'Valid ✅' if checks.get('schema', {}).get('all_required_columns_exist', False) else 'Issues found ❌'}

## Coverage Analysis

**Total FPL Players**: {checks.get('coverage', {}).get('total_fpl_players', 'N/A')}  
**Complete Players**: {checks.get('coverage', {}).get('complete_players', 'N/A')}  
**Coverage Percentage**: {results['coverage_percentage']:.1f}%  
**Requirement Met**: {'✅ Yes' if results['coverage_percentage'] >= 95 else '❌ No (≥95% required)'}

## Validation Checks

### Uniqueness
- **Duplicate Rows**: {checks.get('uniqueness', {}).get('duplicate_count', 'N/A')}
- **Status**: {'✅ Passed' if not checks.get('uniqueness', {}).get('has_duplicates', False) else '❌ Failed'}

### Foreign Keys
- **Valid Team IDs**: {'✅ All valid' if checks.get('foreign_keys', {}).get('all_keys_valid', False) else '❌ Issues found'}
- **Missing Team IDs**: {checks.get('foreign_keys', {}).get('missing_team_ids', 'None')}

### Schema
- **Required Columns**: {'✅ All present' if checks.get('schema', {}).get('all_required_columns_exist', False) else '❌ Missing columns'}
- **Missing Columns**: {checks.get('schema', {}).get('missing_columns', 'None')}
- **Gameweek Range**: {checks.get('schema', {}).get('gameweek_range', 'N/A')}
- **Range Valid**: {'✅ Yes (1-38)' if checks.get('schema', {}).get('gameweek_range_valid', False) else '❌ No'}

## Coverage Tables

### Per-Team Coverage
| Team ID | Team Name | Player Count |
|---------|-----------|--------------|
"""
        
        # Add team coverage table
        for team in results['coverage_tables'].get('per_team', []):
            content += f"| {team.get('team_id', 'N/A')} | {team.get('team_name', 'N/A')} | {team.get('player_id', 'N/A')} |\n"
        
        content += """
### Per-Position Coverage
| Position | Player Count | Team Count |
|----------|--------------|------------|
"""
        
        # Add position coverage table
        for pos in results['coverage_tables'].get('per_position', []):
            content += f"| {pos.get('position', 'N/A')} | {pos.get('player_id', 'N/A')} | {pos.get('team_id', 'N/A')} |\n"
        
        content += f"""
## Null Value Summary

| Field | Null Count | Total Count | Null % |
|-------|------------|-------------|---------|
"""
        
        # Add null summary table
        for field, stats in results['null_summary'].items():
            if field != 'overall':
                content += f"| {field} | {stats.get('null_count', 'N/A')} | {stats.get('total_count', 'N/A')} | {stats.get('null_percentage', 'N/A'):.1f}% |\n"
        
        # Add warnings and errors
        if results['warnings']:
            content += "\n## ⚠️ Warnings\n"
            for warning in results['warnings']:
                content += f"- {warning}\n"
        
        if results['errors']:
            content += "\n## ❌ Errors\n"
            for error in results['errors']:
                content += f"- {error}\n"
        
        # Add unresolved entities info
        unresolved_check = checks.get('unresolved_entities', {})
        if unresolved_check.get('has_unresolved_entities', False):
            content += f"\n## 📋 Unresolved Entities\n"
            content += f"- **File**: {unresolved_check.get('unresolved_file_path', 'N/A')}\n"
            content += f"- **Count**: {unresolved_check.get('unresolved_count', 'N/A')}\n"
        
        content += f"""
## Next Steps

This validation report indicates the merged dataset is {'ready for use' if results['overall_status'] == 'PASSED' else 'needs attention'}.

**Recommendations:**
- {'✅ Proceed to next step' if results['overall_status'] == 'PASSED' else '❌ Fix critical issues first'}
- {'⚠️ Review warnings' if results['warnings'] else '✅ No warnings to address'}
- {'📊 Monitor data quality' if results['overall_status'] == 'WARNING' else ''}

---
*Generated by FPL Merge Validator v1.0*
"""
        
        return content
    
    def log_console_summary(self) -> None:
        """Log validation summary to console for quick inspection."""
        results = self.validation_results
        checks = results['validation_checks']
        
        logger.info("=" * 60)
        logger.info("FPL MERGE VALIDATION SUMMARY - STEP 3B")
        logger.info("=" * 60)
        
        # Overall status
        status_emoji = {'PASSED': '✅', 'WARNING': '⚠️', 'FAILED': '❌'}
        logger.info(f"Overall Status: {status_emoji.get(results['overall_status'], '❓')} {results['overall_status']}")
        
        # Coverage
        coverage_ok = "✅" if results['coverage_percentage'] >= 95 else "❌"
        logger.info(f"Coverage: {coverage_ok} {results['coverage_percentage']:.1f}% (≥95% required)")
        
        # Duplicates
        dup_ok = "✅" if not checks.get('uniqueness', {}).get('has_duplicates', False) else "❌"
        dup_count = checks.get('uniqueness', {}).get('duplicate_count', 0)
        logger.info(f"Duplicates: {dup_ok} {dup_count} duplicate rows found")
        
        # Schema
        schema_ok = "✅" if checks.get('schema', {}).get('all_required_columns_exist', False) else "❌"
        logger.info(f"Schema: {schema_ok} All required columns present")
        
        # Foreign keys
        fk_ok = "✅" if checks.get('foreign_keys', {}).get('all_keys_valid', False) else "❌"
        logger.info(f"Foreign Keys: {fk_ok} All team IDs valid")
        
        # Warnings and errors
        if results['warnings']:
            logger.info(f"⚠️  Warnings: {len(results['warnings'])}")
            for warning in results['warnings'][:3]:  # Show first 3
                logger.info(f"   - {warning}")
            if len(results['warnings']) > 3:
                logger.info(f"   ... and {len(results['warnings']) - 3} more")
        
        if results['errors']:
            logger.info(f"❌ Errors: {len(results['errors'])}")
            for error in results['errors']:
                logger.info(f"   - {error}")
        
        logger.info("=" * 60)
    
    def run(self) -> None:
        """Run the complete validation process."""
        try:
            logger.info("Starting FPL Merge Validation (Step 3b)")
            
            # Load datasets
            merged_data, teams_data = self.load_datasets()
            
            # Run validation checks
            self.check_coverage(merged_data)
            self.check_uniqueness(merged_data)
            self.check_foreign_keys(merged_data, teams_data)
            self.check_schema(merged_data)
            self.generate_coverage_tables(merged_data)
            self.audit_null_values(merged_data)
            self.check_unresolved_entities()
            
            # Determine overall status
            self.determine_overall_status()
            
            # Generate warnings and errors
            self.generate_warnings_and_errors()
            
            # Save reports
            self.save_json_report()
            self.save_markdown_report()
            
            # Log console summary
            self.log_console_summary()
            
            # Exit with appropriate status
            if self.validation_results['overall_status'] == 'FAILED':
                logger.error("Validation failed - exiting with error status")
                sys.exit(1)
            elif self.validation_results['overall_status'] == 'WARNING':
                logger.warning("Validation completed with warnings")
            else:
                logger.info("Validation completed successfully!")
                
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            raise


def main():
    """Main entry point."""
    validator = FPLMergeValidator()
    validator.run()


if __name__ == "__main__":
    main() 