#!/usr/bin/env python3
"""
CSV/TSV Data Profiler using Polars

This script analyzes CSV and TSV files in the data/raw directory and generates
a comprehensive Markdown report with data quality insights.

Usage:
    python src/profile_csvs.py [--sample N]

Arguments:
    --sample N    Sample N rows from each file (useful for large files)
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProfiler:
    """Profiles CSV/TSV files and generates quality reports."""
    
    def __init__(self, sample_size: Optional[int] = None):
        self.sample_size = sample_size
        self.data_dir = Path("data/raw")
        self.report_dir = Path("reports")
        self.issues = []
        
    def find_csv_files(self) -> List[Path]:
        """Find all CSV and TSV files in the data directory."""
        if not self.data_dir.exists():
            logger.error(f"Data directory {self.data_dir} does not exist")
            return []
            
        csv_files = []
        for ext in ['*.csv', '*.tsv']:
            csv_files.extend(self.data_dir.glob(ext))
            csv_files.extend(self.data_dir.rglob(ext))  # Include subdirectories
            
        # Remove duplicates and sort
        csv_files = sorted(list(set(csv_files)))
        logger.info(f"Found {len(csv_files)} CSV/TSV files")
        return csv_files
    
    def detect_separator(self, file_path: Path) -> str:
        """Detect separator based on file extension."""
        return '\t' if file_path.suffix.lower() == '.tsv' else ','
    
    def analyze_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single CSV/TSV file and return profiling data."""
        try:
            logger.info(f"Analyzing {file_path}")
            
            separator = self.detect_separator(file_path)
            
            # Read file with polars in lazy mode
            df_lazy = pl.scan_csv(
                file_path,
                separator=separator,
                infer_schema_length=5000,
                ignore_errors=True
            )
            
            # Sample if requested
            if self.sample_size:
                df_lazy = df_lazy.limit(self.sample_size)
            
            # Collect to compute statistics
            df = df_lazy.collect()
            
            if df.height == 0:
                logger.warning(f"File {file_path} is empty")
                return None
                
            row_count = df.height
            column_count = df.width
            
            # Skip distinct value calculation for wide datasets
            calculate_distinct = column_count <= 200
            
            # Analyze each column
            columns_info = []
            for col_name in df.columns:
                col_data = df[col_name]
                dtype = str(col_data.dtype)
                
                # Calculate null percentage
                null_count = col_data.null_count()
                null_percentage = (null_count / row_count) * 100 if row_count > 0 else 0
                
                # Calculate distinct values (if not too many columns)
                distinct_count = None
                if calculate_distinct:
                    try:
                        distinct_count = col_data.n_unique()
                    except Exception as e:
                        logger.warning(f"Could not calculate distinct values for {col_name}: {e}")
                
                # Detect issues
                notes = self._detect_column_issues(col_name, col_data, dtype, null_percentage, file_path)
                
                columns_info.append({
                    'name': col_name,
                    'dtype': dtype,
                    'null_percentage': null_percentage,
                    'distinct_count': distinct_count,
                    'notes': notes
                })
            
            return {
                'file_path': file_path,
                'row_count': row_count,
                'column_count': column_count,
                'columns': columns_info,
                'calculate_distinct': calculate_distinct
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            self.issues.append(f"**Error processing {file_path}**: {str(e)}")
            return None
    
    def _detect_column_issues(self, col_name: str, col_data: pl.Series, dtype: str, 
                            null_percentage: float, file_path: Path) -> List[str]:
        """Detect potential issues with a column."""
        notes = []
        
        # High null percentage
        if null_percentage > 20:
            issue = f"**High null values**: Column `{col_name}` in `{file_path.name}` has {null_percentage:.1f}% null values"
            self.issues.append(issue)
            notes.append("High nulls")
        
        # Check for currency symbols in numeric-looking columns
        if dtype in ['Utf8', 'String'] and not col_data.is_null().all():
            # Get a sample of non-null values
            sample_values = col_data.drop_nulls().limit(100).to_list()
            sample_str = ' '.join(str(v) for v in sample_values if v is not None)
            
            # Check for currency symbols
            currency_pattern = r'[£€$¥¢₹₽]'
            if re.search(currency_pattern, sample_str):
                # Check if it might be numeric (contains digits)
                if re.search(r'\d', sample_str):
                    issue = f"**Currency in text column**: Column `{col_name}` in `{file_path.name}` contains currency symbols but is parsed as text"
                    self.issues.append(issue)
                    notes.append("Currency symbols")
            
            # Check for date patterns
            date_patterns = [
                r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
                r'\b\d{2}/\d{2}/\d{4}\b',  # DD/MM/YYYY or MM/DD/YYYY
                r'\b\d{1,2}/\d{1,2}/\d{4}\b'  # D/M/YYYY variants
            ]
            
            for pattern in date_patterns:
                if re.search(pattern, sample_str):
                    issue = f"**Potential date column**: Column `{col_name}` in `{file_path.name}` contains date-like patterns but is parsed as text"
                    self.issues.append(issue)
                    notes.append("Date-like")
                    break
        
        return notes
    
    def generate_markdown_report(self, profiles: List[Dict[str, Any]]) -> str:
        """Generate a Markdown report from the profiling data."""
        report_lines = [
            "# Data Profile Report",
            "",
            f"This report analyzes {len(profiles)} CSV/TSV files from the `data/raw` directory.",
            "",
        ]
        
        if self.sample_size:
            report_lines.extend([
                f"**Note**: Analysis was performed on a sample of {self.sample_size:,} rows per file.",
                ""
            ])
        
        # Table of contents
        report_lines.extend([
            "## Table of Contents",
            ""
        ])
        
        for i, profile in enumerate(profiles, 1):
            file_name = profile['file_path'].name
            report_lines.append(f"{i}. [{file_name}](#{file_name.lower().replace('.', '').replace(' ', '-')})")
        
        if self.issues:
            report_lines.append(f"{len(profiles) + 1}. [Data Quality Issues](#data-quality-issues)")
        
        report_lines.append("")
        
        # Individual file sections
        for profile in profiles:
            file_path = profile['file_path']
            file_name = file_path.name
            
            report_lines.extend([
                f"## {file_name}",
                "",
                f"**File Path**: `{file_path}`  ",
                f"**Rows**: {profile['row_count']:,}  ",
                f"**Columns**: {profile['column_count']}  ",
                ""
            ])
            
            if not profile['calculate_distinct']:
                report_lines.extend([
                    "*Note: Distinct value calculation skipped (>200 columns)*",
                    ""
                ])
            
            # Column table
            report_lines.extend([
                "| Column Name | Data Type | % Null | # Distinct | Notes |",
                "|-------------|-----------|--------|------------|-------|"
            ])
            
            for col in profile['columns']:
                name = col['name']
                dtype = col['dtype']
                null_pct = f"{col['null_percentage']:.1f}%"
                distinct = str(col['distinct_count']) if col['distinct_count'] is not None else "N/A"
                notes = ", ".join(col['notes']) if col['notes'] else "-"
                
                report_lines.append(f"| {name} | {dtype} | {null_pct} | {distinct} | {notes} |")
            
            report_lines.extend(["", "---", ""])
        
        # Issues section
        if self.issues:
            report_lines.extend([
                "## Data Quality Issues",
                "",
                "The following potential data quality issues were detected:",
                ""
            ])
            
            for issue in self.issues:
                report_lines.append(f"- {issue}")
            
            report_lines.append("")
        else:
            report_lines.extend([
                "## Data Quality Issues",
                "",
                "No significant data quality issues were detected! 🎉",
                ""
            ])
        
        # Footer
        report_lines.extend([
            "---",
            "",
            "*Report generated by Data Profiler using Polars*"
        ])
        
        return "\n".join(report_lines)
    
    def run(self) -> None:
        """Run the complete profiling process."""
        logger.info("Starting data profiling...")
        
        # Find files
        csv_files = self.find_csv_files()
        if not csv_files:
            logger.warning("No CSV/TSV files found")
            return
        
        # Analyze files
        profiles = []
        for file_path in csv_files:
            profile = self.analyze_file(file_path)
            if profile:
                profiles.append(profile)
        
        if not profiles:
            logger.error("No files could be successfully analyzed")
            return
        
        # Generate report
        report_content = self.generate_markdown_report(profiles)
        
        # Ensure reports directory exists
        self.report_dir.mkdir(exist_ok=True)
        
        # Write report
        report_path = self.report_dir / "data_profile.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report written to {report_path}")
        logger.info(f"Analyzed {len(profiles)} files successfully")
        if self.issues:
            logger.warning(f"Found {len(self.issues)} potential data quality issues")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Profile CSV/TSV files and generate a data quality report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/profile_csvs.py                    # Process all rows
  python src/profile_csvs.py --sample 250000    # Sample 250,000 rows per file
        """
    )
    
    parser.add_argument(
        '--sample', 
        type=int, 
        metavar='N',
        help='Sample N rows from each file (useful for large files)'
    )
    
    args = parser.parse_args()
    
    # Validate sample size
    if args.sample is not None and args.sample <= 0:
        parser.error("Sample size must be a positive integer")
    
    # Run profiler
    profiler = DataProfiler(sample_size=args.sample)
    profiler.run()


if __name__ == "__main__":
    main() 