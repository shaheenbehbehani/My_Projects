#!/usr/bin/env python3
"""
Feature Validation

Validates the final feature store with quality checks:
- Row count matches fixtures teams
- Feature value ranges
- Probability constraints
- Missing value analysis

Appends validation results to reports/features_summary.md
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureValidator:
    """Validates the final feature store for quality and consistency."""
    
    def __init__(self, processed_dir: Path, reports_dir: Path):
        """Initialize the feature validator."""
        self.processed_dir = Path(processed_dir)
        self.reports_dir = Path(reports_dir)
        
        # Validation results
        self.validation_results = []
        self.warnings = []
        self.errors = []
    
    def add_validation(self, check_name: str, status: str, message: str, details: str = ""):
        """Add a validation result."""
        self.validation_results.append({
            'check': check_name,
            'status': status,
            'message': message,
            'details': details
        })
        
        if status == 'ERROR':
            self.errors.append(f"{check_name}: {message}")
            logger.error(f"❌ {check_name}: {message}")
        elif status == 'WARNING':
            self.warnings.append(f"{check_name}: {message}")
            logger.warning(f"⚠️ {check_name}: {message}")
        else:
            logger.info(f"✅ {check_name}: {message}")
    
    def load_feature_store(self) -> pl.DataFrame:
        """Load the final feature store."""
        logger.info("Loading feature store for validation...")
        
        feature_file = self.processed_dir / "features_2025_26.parquet"
        if not feature_file.exists():
            raise FileNotFoundError("Feature store not found - run build_feature_store.py first")
        
        try:
            df = pl.read_parquet(feature_file)
            logger.info(f"Loaded feature store: {df.height} teams, {df.width} features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load feature store: {e}")
            raise
    
    def load_fixtures_for_comparison(self) -> pl.DataFrame:
        """Load fixtures for team count validation."""
        logger.info("Loading fixtures for comparison...")
        
        fixtures_file = self.processed_dir / "fixtures_2025_26.parquet"
        if not fixtures_file.exists():
            raise FileNotFoundError("Fixtures file not found")
        
        try:
            df = pl.read_parquet(fixtures_file)
            logger.info(f"Loaded fixtures: {df.height} matches")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load fixtures: {e}")
            raise
    
    def validate_row_count(self, features_df: pl.DataFrame, fixtures_df: pl.DataFrame):
        """Validate that feature store has correct number of teams."""
        logger.info("Validating row count...")
        
        # Get unique teams from fixtures
        home_teams = fixtures_df['home_team'].unique()
        away_teams = fixtures_df['away_team'].unique()
        expected_teams = pl.concat([home_teams, away_teams]).unique().len()
        
        actual_teams = features_df.height
        
        if actual_teams == expected_teams:
            self.add_validation(
                "Row Count",
                "PASS",
                f"Correct number of teams: {actual_teams}",
                f"Matches expected count from fixtures"
            )
        else:
            self.add_validation(
                "Row Count",
                "ERROR",
                f"Team count mismatch: {actual_teams} vs expected {expected_teams}",
                f"Feature store should have one row per team in fixtures"
            )
    
    def validate_team_identifiers(self, features_df: pl.DataFrame):
        """Validate team identifier column."""
        logger.info("Validating team identifiers...")
        
        if 'team' not in features_df.columns:
            self.add_validation(
                "Team Column",
                "ERROR",
                "Missing 'team' column",
                "Feature store must have team identifier column"
            )
            return
        
        # Check for null teams
        null_teams = features_df['team'].null_count()
        if null_teams > 0:
            self.add_validation(
                "Team Identifiers",
                "ERROR",
                f"{null_teams} null team identifiers",
                "All teams must have valid identifiers"
            )
        
        # Check for duplicate teams
        unique_teams = features_df['team'].n_unique()
        total_teams = features_df.height
        
        if unique_teams != total_teams:
            duplicates = total_teams - unique_teams
            self.add_validation(
                "Team Uniqueness",
                "ERROR",
                f"{duplicates} duplicate teams found",
                "Each team should appear exactly once"
            )
        else:
            self.add_validation(
                "Team Uniqueness",
                "PASS",
                "All teams are unique",
                f"{unique_teams} unique teams"
            )
    
    def validate_probability_columns(self, features_df: pl.DataFrame):
        """Validate probability columns are in valid range."""
        logger.info("Validating probability columns...")
        
        probability_cols = [col for col in features_df.columns if 'prob' in col.lower()]
        
        for col in probability_cols:
            if col in features_df.columns:
                min_val = features_df[col].min()
                max_val = features_df[col].max()
                sum_val = features_df[col].sum()
                
                # Check range [0, 1]
                if min_val < 0 or max_val > 1:
                    self.add_validation(
                        f"Probability Range ({col})",
                        "ERROR",
                        f"Values outside [0,1]: min={min_val:.6f}, max={max_val:.6f}",
                        "Probabilities must be between 0 and 1"
                    )
                else:
                    self.add_validation(
                        f"Probability Range ({col})",
                        "PASS",
                        f"Valid range: [{min_val:.6f}, {max_val:.6f}]",
                        ""
                    )
                
                # Check if probabilities sum to approximately 1 (for title probabilities)
                if 'title' in col.lower():
                    if abs(sum_val - 1.0) > 0.01:
                        self.add_validation(
                            f"Probability Sum ({col})",
                            "WARNING",
                            f"Sum deviates from 1.0: {sum_val:.6f}",
                            "Title probabilities should sum to 1.0"
                        )
                    else:
                        self.add_validation(
                            f"Probability Sum ({col})",
                            "PASS",
                            f"Sum close to 1.0: {sum_val:.6f}",
                            ""
                        )
    
    def validate_feature_ranges(self, features_df: pl.DataFrame):
        """Validate feature values are in reasonable ranges."""
        logger.info("Validating feature ranges...")
        
        # Define expected ranges for key features
        range_checks = {
            'elo_pre': (1000, 2000),
            'market_value_eur': (1_000_000, 2_000_000_000),
            'annual_wages_gbp': (1_000_000, 500_000_000),
            'avg_attendance': (5_000, 100_000),
            'capacity': (10_000, 150_000),
            'manager_tenure_days': (0, 10_000),
            'squad_goals_90': (0.5, 4.0),
            'squad_xg_90': (0.5, 4.0),
            'avg_xg_per_shot': (0.05, 0.3),
            'shots_per_game': (5, 25),
            'injury_burden_per_1000': (0, 200)
        }
        
        for col, (min_expected, max_expected) in range_checks.items():
            if col in features_df.columns:
                min_val = features_df[col].min()
                max_val = features_df[col].max()
                
                if min_val < min_expected or max_val > max_expected:
                    self.add_validation(
                        f"Range Check ({col})",
                        "WARNING",
                        f"Values outside expected range: [{min_val:.1f}, {max_val:.1f}] vs [{min_expected}, {max_expected}]",
                        "Values may indicate data quality issues"
                    )
                else:
                    self.add_validation(
                        f"Range Check ({col})",
                        "PASS",
                        f"Values in expected range: [{min_val:.1f}, {max_val:.1f}]",
                        ""
                    )
    
    def validate_missing_values(self, features_df: pl.DataFrame):
        """Validate missing value patterns."""
        logger.info("Validating missing values...")
        
        total_cells = features_df.height * (features_df.width - 1)  # Exclude team column
        total_nulls = sum(features_df[col].null_count() for col in features_df.columns if col != 'team')
        
        completeness = (total_cells - total_nulls) / total_cells if total_cells > 0 else 1.0
        
        if completeness < 0.95:
            self.add_validation(
                "Data Completeness",
                "WARNING",
                f"Low completeness: {completeness*100:.1f}%",
                f"{total_nulls:,} missing values out of {total_cells:,} cells"
            )
        else:
            self.add_validation(
                "Data Completeness",
                "PASS",
                f"Good completeness: {completeness*100:.1f}%",
                f"{total_nulls:,} missing values out of {total_cells:,} cells"
            )
        
        # Check for columns with high missing rates
        for col in features_df.columns:
            if col != 'team':
                null_count = features_df[col].null_count()
                null_rate = null_count / features_df.height
                
                if null_rate > 0.2:  # More than 20% missing
                    self.add_validation(
                        f"Column Completeness ({col})",
                        "WARNING",
                        f"High missing rate: {null_rate*100:.1f}%",
                        f"{null_count} out of {features_df.height} values missing"
                    )
    
    def validate_derived_features(self, features_df: pl.DataFrame):
        """Validate derived feature calculations."""
        logger.info("Validating derived features...")
        
        # Check wage-to-value ratio
        if all(col in features_df.columns for col in ['wage_to_value_ratio', 'annual_wages_gbp', 'market_value_eur']):
            # Recalculate and compare
            expected_ratio = (features_df['annual_wages_gbp'] * 1.15) / features_df['market_value_eur']
            actual_ratio = features_df['wage_to_value_ratio']
            
            max_diff = (expected_ratio - actual_ratio).abs().max()
            
            if max_diff > 0.001:
                self.add_validation(
                    "Wage-to-Value Ratio",
                    "WARNING",
                    f"Calculation discrepancy: max diff {max_diff:.6f}",
                    "Derived feature may not match expected calculation"
                )
            else:
                self.add_validation(
                    "Wage-to-Value Ratio",
                    "PASS",
                    "Calculation correct",
                    ""
                )
        
        # Check stadium utilization
        if all(col in features_df.columns for col in ['stadium_utilization', 'avg_attendance', 'capacity']):
            expected_util = features_df['avg_attendance'] / features_df['capacity']
            actual_util = features_df['stadium_utilization']
            
            max_diff = (expected_util - actual_util).abs().max()
            
            if max_diff > 0.001:
                self.add_validation(
                    "Stadium Utilization",
                    "WARNING",
                    f"Calculation discrepancy: max diff {max_diff:.6f}",
                    "Derived feature may not match expected calculation"
                )
            else:
                self.add_validation(
                    "Stadium Utilization",
                    "PASS",
                    "Calculation correct",
                    ""
                )
    
    def validate_rankings(self, features_df: pl.DataFrame):
        """Validate ranking columns."""
        logger.info("Validating ranking columns...")
        
        ranking_cols = [col for col in features_df.columns if 'rank' in col.lower()]
        
        for col in ranking_cols:
            if col in features_df.columns:
                min_rank = features_df[col].min()
                max_rank = features_df[col].max()
                unique_ranks = features_df[col].n_unique()
                
                # Rankings should start at 1 and go up to team count
                expected_max = features_df.height
                
                if min_rank != 1 or max_rank != expected_max:
                    self.add_validation(
                        f"Ranking Range ({col})",
                        "WARNING",
                        f"Unexpected range: [{min_rank}, {max_rank}] vs [1, {expected_max}]",
                        "Rankings should span from 1 to number of teams"
                    )
                else:
                    self.add_validation(
                        f"Ranking Range ({col})",
                        "PASS",
                        f"Correct range: [1, {expected_max}]",
                        ""
                    )
    
    def generate_validation_report(self) -> str:
        """Generate validation report section."""
        logger.info("Generating validation report...")
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Count results by status
        passed = len([r for r in self.validation_results if r['status'] == 'PASS'])
        warnings = len([r for r in self.validation_results if r['status'] == 'WARNING'])
        errors = len([r for r in self.validation_results if r['status'] == 'ERROR'])
        
        # Overall status
        if errors > 0:
            overall_status = "❌ FAILED"
        elif warnings > 0:
            overall_status = "⚠️ WARNINGS"
        else:
            overall_status = "✅ PASSED"
        
        report = f"""
---

## Validation Results
*Validated: {timestamp}*

### Summary

**Overall Status**: {overall_status}  
**Total Checks**: {len(self.validation_results)}  
**Passed**: {passed} ✅  
**Warnings**: {warnings} ⚠️  
**Errors**: {errors} ❌

### Detailed Results

"""
        
        # Group results by status
        for status in ['ERROR', 'WARNING', 'PASS']:
            status_results = [r for r in self.validation_results if r['status'] == status]
            
            if status_results:
                icon = {'ERROR': '❌', 'WARNING': '⚠️', 'PASS': '✅'}[status]
                report += f"#### {status} {icon}\n\n"
                
                for result in status_results:
                    report += f"**{result['check']}**: {result['message']}\n"
                    if result['details']:
                        report += f"  - {result['details']}\n"
                    report += "\n"
        
        # Recommendations
        report += "### Recommendations\n\n"
        
        if errors > 0:
            report += "**Critical Issues Found** ❌\n"
            report += "- Fix all errors before proceeding to modeling\n"
            report += "- Review data sources and feature engineering logic\n"
            report += "- Consider re-running feature pipeline components\n\n"
        
        if warnings > 0:
            report += "**Warnings Detected** ⚠️\n"
            report += "- Review warnings for potential data quality issues\n"
            report += "- Consider if values are reasonable for domain\n"
            report += "- Document any expected deviations\n\n"
        
        if errors == 0 and warnings == 0:
            report += "**All Checks Passed** ✅\n"
            report += "- Feature store ready for modeling\n"
            report += "- High data quality confirmed\n"
            report += "- Proceed to Phase 4 with confidence\n\n"
        
        report += "---\n\n"
        report += "*Validation completed by Premier League Data Pipeline Feature Validator*\n"
        
        return report
    
    def append_to_summary_report(self, validation_report: str):
        """Append validation results to features summary report."""
        logger.info("Appending validation results to summary report...")
        
        summary_file = self.reports_dir / "features_summary.md"
        
        if summary_file.exists():
            # Read existing report
            with open(summary_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            
            # Append validation results
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(existing_content)
                f.write(validation_report)
            
            logger.info(f"✅ Validation results appended to {summary_file}")
        else:
            logger.warning("Summary report not found - creating new validation report")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("# Feature Validation Report\n")
                f.write(validation_report)
    
    def validate_feature_store(self) -> bool:
        """Run complete feature store validation."""
        logger.info("Starting feature store validation...")
        
        try:
            # Load data
            features_df = self.load_feature_store()
            fixtures_df = self.load_fixtures_for_comparison()
            
            # Run validation checks
            self.validate_row_count(features_df, fixtures_df)
            self.validate_team_identifiers(features_df)
            self.validate_probability_columns(features_df)
            self.validate_feature_ranges(features_df)
            self.validate_missing_values(features_df)
            self.validate_derived_features(features_df)
            self.validate_rankings(features_df)
            
            # Generate and save report
            validation_report = self.generate_validation_report()
            self.append_to_summary_report(validation_report)
            
            # Log summary
            passed = len([r for r in self.validation_results if r['status'] == 'PASS'])
            warnings = len([r for r in self.validation_results if r['status'] == 'WARNING'])
            errors = len([r for r in self.validation_results if r['status'] == 'ERROR'])
            
            logger.info(f"Validation complete: {passed} passed, {warnings} warnings, {errors} errors")
            
            return errors == 0
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    
    try:
        validator = FeatureValidator(processed_dir, reports_dir)
        success = validator.validate_feature_store()
        
        if success:
            print(f"\n✅ Feature Validation Complete!")
            print(f"📄 Results appended to: {reports_dir / 'features_summary.md'}")
            return 0
        else:
            print(f"\n⚠️ Feature Validation Issues Found!")
            print(f"📄 See details in: {reports_dir / 'features_summary.md'}")
            return 0  # Always return 0 as this is a report, not a hard fail
        
    except Exception as e:
        logger.error(f"Failed to validate features: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 