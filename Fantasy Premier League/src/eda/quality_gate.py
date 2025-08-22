#!/usr/bin/env python3
"""
Data Quality Gate Checker

Performs quality checks on Premier League data and generates a gates report.
Checks xG coverage, manager tenure issues, and other data quality metrics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import re

import polars as pl

# Import our normalization module
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.normalization import canonicalize_frame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QualityGateChecker:
    """Performs data quality gate checks for Premier League dataset."""
    
    def __init__(self, data_raw: Path, reports_dir: Path):
        """Initialize the quality gate checker."""
        self.data_raw = Path(data_raw)
        self.reports_dir = Path(reports_dir)
        
        # Ensure directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Track quality check results
        self.checks = []
        self.warnings = []
        self.passed_checks = 0
        self.failed_checks = 0
        
    def add_check(self, name: str, status: str, message: str, details: str = ""):
        """Add a quality check result."""
        self.checks.append({
            'name': name,
            'status': status,
            'message': message,
            'details': details
        })
        
        if status == 'PASS':
            self.passed_checks += 1
        elif status == 'FAIL':
            self.failed_checks += 1
    
    def check_xg_coverage(self) -> bool:
        """Check xG data coverage in possession/stats files."""
        logger.info("Checking xG data coverage...")
        
        try:
            # Look for possession or team stats files
            possession_files = [
                self.data_raw / "Possession data 24-25.csv",
                self.data_raw / "Team Stats.csv",
                self.data_raw / "team_stats.csv"
            ]
            
            stats_df = None
            used_file = None
            
            for file_path in possession_files:
                if file_path.exists():
                    try:
                        stats_df = pl.read_csv(file_path, ignore_errors=True)
                        used_file = file_path
                        logger.info(f"Loaded stats from: {file_path.name}")
                        break
                    except Exception as e:
                        logger.warning(f"Could not load {file_path.name}: {e}")
            
            if stats_df is None:
                self.add_check(
                    "xG Coverage",
                    "SKIP",
                    "No possession/stats files found for xG analysis",
                    "Looked for: Possession data 24-25.csv, Team Stats.csv, team_stats.csv"
                )
                return True
            
            # Look for xG columns
            xg_cols = []
            for col in stats_df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['xg', 'expected_goals', 'expected goals']):
                    xg_cols.append(col)
            
            if not xg_cols:
                self.add_check(
                    "xG Coverage",
                    "SKIP",
                    "No xG columns found in stats data",
                    f"Available columns: {stats_df.columns[:10]}"
                )
                return True
            
            # Check coverage for each xG column
            total_rows = stats_df.height
            coverage_results = []
            
            for col in xg_cols:
                non_null_count = total_rows - stats_df[col].null_count()
                coverage_pct = (non_null_count / total_rows) * 100 if total_rows > 0 else 0
                coverage_results.append((col, coverage_pct, non_null_count))
            
            # Overall xG coverage (any xG column has data)
            has_any_xg = stats_df.select([
                pl.any_horizontal([pl.col(col).is_not_null() for col in xg_cols])
            ]).to_series().sum()
            
            overall_coverage = (has_any_xg / total_rows) * 100 if total_rows > 0 else 0
            
            # Quality gate: ≥90% coverage
            if overall_coverage >= 90.0:
                status = "PASS"
                message = f"xG coverage: {overall_coverage:.1f}% (≥90% required)"
            else:
                status = "FAIL"
                message = f"xG coverage: {overall_coverage:.1f}% (below 90% threshold)"
            
            details = f"File: {used_file.name}\n"
            details += f"Total records: {total_rows:,}\n"
            details += "Column coverage:\n"
            for col, pct, count in coverage_results:
                details += f"  • {col}: {pct:.1f}% ({count:,}/{total_rows:,})\n"
            
            self.add_check("xG Coverage", status, message, details)
            
            return status == "PASS"
            
        except Exception as e:
            self.add_check(
                "xG Coverage",
                "ERROR",
                f"Failed to check xG coverage: {str(e)}",
                ""
            )
            return False
    
    def check_manager_tenure(self) -> bool:
        """Check manager tenure data for missing or short tenures."""
        logger.info("Checking manager tenure data...")
        
        try:
            managers_file = self.data_raw / "Premier League Managers.csv"
            
            if not managers_file.exists():
                self.add_check(
                    "Manager Tenure",
                    "SKIP",
                    "Premier League Managers.csv not found",
                    ""
                )
                return True
            
            managers_df = pl.read_csv(managers_file, ignore_errors=True)
            logger.info(f"Loaded managers data: {managers_df.height} records")
            
            # Canonicalize team names
            team_cols = []
            for col in ['Club', 'Team', 'club', 'team']:
                if col in managers_df.columns:
                    team_cols.append(col)
            
            if team_cols:
                managers_df = canonicalize_frame(managers_df, team_cols)
            
            # Find date columns
            from_col = None
            until_col = None
            
            for col in managers_df.columns:
                col_lower = col.lower()
                if 'from' in col_lower and not from_col:
                    from_col = col
                elif any(term in col_lower for term in ['until', 'to', 'end']) and not until_col:
                    until_col = col
            
            if not from_col:
                self.add_check(
                    "Manager Tenure",
                    "SKIP",
                    "No 'from' date column found in managers data",
                    f"Available columns: {managers_df.columns}"
                )
                return True
            
            # Get current managers (latest appointment per club)
            team_col = team_cols[0] if team_cols else 'Club'
            
            # Parse from dates
            try:
                managers_df = managers_df.with_columns([
                    pl.col(from_col).str.strptime(pl.Date, format='%d/%m/%Y', strict=False).alias('from_date')
                ])
            except:
                try:
                    managers_df = managers_df.with_columns([
                        pl.col(from_col).str.strptime(pl.Date, format='%Y-%m-%d', strict=False).alias('from_date')
                    ])
                except:
                    self.add_check(
                        "Manager Tenure",
                        "ERROR",
                        "Could not parse manager 'from' dates",
                        f"Sample values: {managers_df[from_col].limit(5).to_list()}"
                    )
                    return False
            
            # Find current manager per club (most recent appointment)
            current_managers = (
                managers_df
                .filter(pl.col('from_date').is_not_null())
                .sort(['from_date'], descending=True)
                .group_by(team_col)
                .agg([
                    pl.col('from_date').first().alias('latest_from'),
                    pl.col('Manager').first().alias('current_manager')
                ])
            )
            
            # Calculate tenure days (from appointment to now)
            today = datetime.now().date()
            current_managers = current_managers.with_columns([
                (pl.lit(today) - pl.col('latest_from')).dt.total_days().alias('tenure_days')
            ])
            
            # Identify issues
            missing_managers = []
            short_tenure_managers = []
            
            for row in current_managers.iter_rows(named=True):
                club = row[team_col]
                manager = row['current_manager']
                tenure_days = row['tenure_days']
                
                if manager is None or manager == "":
                    missing_managers.append(club)
                elif tenure_days < 30:
                    short_tenure_managers.append((club, manager, tenure_days))
            
            # Quality assessment
            total_clubs = current_managers.height
            issues_count = len(missing_managers) + len(short_tenure_managers)
            
            if issues_count == 0:
                status = "PASS"
                message = f"All {total_clubs} clubs have stable manager appointments (≥30 days)"
            else:
                status = "WARN"
                message = f"{issues_count}/{total_clubs} clubs have manager tenure issues"
            
            details = f"Total clubs analyzed: {total_clubs}\n"
            if missing_managers:
                details += f"Missing current manager:\n"
                for club in missing_managers:
                    details += f"  • {club}\n"
            
            if short_tenure_managers:
                details += f"Short tenure (<30 days):\n"
                for club, manager, days in short_tenure_managers:
                    details += f"  • {club}: {manager} ({days} days)\n"
            
            if issues_count == 0:
                details += "No manager tenure issues detected."
            
            self.add_check("Manager Tenure", status, message, details)
            
            return status != "FAIL"
            
        except Exception as e:
            self.add_check(
                "Manager Tenure",
                "ERROR",
                f"Failed to check manager tenure: {str(e)}",
                ""
            )
            return False
    
    def check_data_completeness(self) -> bool:
        """Check overall data completeness and file presence."""
        logger.info("Checking data completeness...")
        
        try:
            # Required files for full analysis
            required_files = {
                'Historical Matches': list(self.data_raw.glob("E0*.csv")),
                'Club Values': self.data_raw / "Club Value.csv",
                'Club Wages': self.data_raw / "Club wages.csv",
                'Attendance Data': self.data_raw / "Attendance Data.csv",
                'Managers Data': self.data_raw / "Premier League Managers.csv"
            }
            
            missing_files = []
            present_files = []
            
            for category, file_path in required_files.items():
                if isinstance(file_path, list):
                    if file_path:  # Historical matches (multiple files)
                        present_files.append(f"{category}: {len(file_path)} files")
                    else:
                        missing_files.append(category)
                else:
                    if file_path.exists():
                        present_files.append(f"{category}: {file_path.name}")
                    else:
                        missing_files.append(category)
            
            # Quality assessment
            if not missing_files:
                status = "PASS"
                message = "All required data files present"
            elif len(missing_files) <= 1:
                status = "WARN"
                message = f"{len(missing_files)} optional file(s) missing"
            else:
                status = "FAIL"
                message = f"{len(missing_files)} required files missing"
            
            details = "Present files:\n"
            for file_info in present_files:
                details += f"  ✅ {file_info}\n"
            
            if missing_files:
                details += "\nMissing files:\n"
                for file_cat in missing_files:
                    details += f"  ❌ {file_cat}\n"
            
            self.add_check("Data Completeness", status, message, details)
            
            return status != "FAIL"
            
        except Exception as e:
            self.add_check(
                "Data Completeness",
                "ERROR",
                f"Failed to check data completeness: {str(e)}",
                ""
            )
            return False
    
    def generate_quality_report(self) -> Path:
        """Generate comprehensive quality gates report."""
        logger.info("Generating quality gates report...")
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_content = f"""# Data Quality Gates Report
*Generated: {timestamp}*

## Summary

**Total Checks**: {len(self.checks)}  
**Passed**: {self.passed_checks} ✅  
**Failed**: {self.failed_checks} ❌  
**Warnings/Skipped**: {len(self.checks) - self.passed_checks - self.failed_checks} ⚠️

## Quality Gate Results

"""
        
        # Add individual check results
        for check in self.checks:
            status_icon = {
                'PASS': '✅',
                'FAIL': '❌',
                'WARN': '⚠️',
                'SKIP': '⏭️',
                'ERROR': '🔥'
            }.get(check['status'], '❓')
            
            report_content += f"### {check['name']} {status_icon}\n\n"
            report_content += f"**Status**: {check['status']}\n\n"
            report_content += f"**Result**: {check['message']}\n\n"
            
            if check['details']:
                report_content += f"**Details**:\n```\n{check['details']}\n```\n\n"
            
            report_content += "---\n\n"
        
        # Add recommendations
        report_content += """## Recommendations

### High Priority
- Ensure xG data coverage ≥90% for advanced analytics
- Verify manager appointments are current and stable
- Complete any missing data files before Phase 3

### Medium Priority
- Validate data quality across all historical seasons
- Check for systematic biases in betting odds data
- Verify team name canonicalization coverage

### Low Priority
- Add automated data quality monitoring
- Implement data freshness checks
- Create data lineage documentation

## Quality Gates Summary

| Gate | Threshold | Status |
|------|-----------|--------|
"""
        
        # Add gates summary table
        for check in self.checks:
            status_symbol = "✅" if check['status'] == 'PASS' else "❌" if check['status'] == 'FAIL' else "⚠️"
            report_content += f"| {check['name']} | See details | {status_symbol} {check['status']} |\n"
        
        report_content += f"""

---

## Next Steps

1. **Address Failed Gates**: Resolve any failed quality checks before proceeding
2. **Review Warnings**: Investigate warnings and determine if action is needed  
3. **Phase 3 Readiness**: Ensure all critical data quality requirements are met

**Overall Assessment**: {"READY FOR PHASE 3" if self.failed_checks == 0 else "NEEDS ATTENTION BEFORE PHASE 3"}

---

*This report was generated automatically by the Premier League Data Pipeline Quality Gate Checker.*
"""
        
        # Write report
        report_path = self.reports_dir / "quality_gates.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"✅ Quality gates report saved: {report_path}")
        return report_path
    
    def run_quality_checks(self) -> Dict[str, any]:
        """Run all quality checks and generate report."""
        logger.info("Starting quality gate checks...")
        
        results = {
            'checks_run': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'report_path': None,
            'overall_status': 'UNKNOWN'
        }
        
        try:
            # Run all quality checks
            self.check_data_completeness()
            self.check_xg_coverage()
            self.check_manager_tenure()
            
            # Generate report
            results['report_path'] = self.generate_quality_report()
            
            # Calculate results
            results['checks_run'] = len(self.checks)
            results['passed'] = self.passed_checks
            results['failed'] = self.failed_checks
            results['warnings'] = len(self.checks) - self.passed_checks - self.failed_checks
            
            # Determine overall status
            if self.failed_checks > 0:
                results['overall_status'] = 'FAILED'
            elif results['warnings'] > 0:
                results['overall_status'] = 'WARNING'
            else:
                results['overall_status'] = 'PASSED'
            
            logger.info(f"✅ Quality checks complete: {results['overall_status']}")
            
        except Exception as e:
            error_msg = f"Quality checks failed: {e}"
            logger.error(error_msg)
            results['overall_status'] = 'ERROR'
        
        return results


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    data_raw = project_root / "data" / "raw"
    reports_dir = project_root / "reports"
    
    try:
        checker = QualityGateChecker(data_raw, reports_dir)
        results = checker.run_quality_checks()
        
        print(f"\n🚧 Quality Gates Analysis Complete!")
        print(f"📊 Checks Run: {results['checks_run']}")
        print(f"✅ Passed: {results['passed']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"⚠️ Warnings: {results['warnings']}")
        print(f"🎯 Overall Status: {results['overall_status']}")
        
        if results['report_path']:
            print(f"📄 Quality Report: {results['report_path']}")
        
        # Always return 0 (this is a report, not a hard fail)
        return 0
        
    except Exception as e:
        logger.error(f"Failed to run quality checks: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 