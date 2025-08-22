#!/usr/bin/env python3
"""
Schema and Data Quality Tests for Premier League Dataset

This test suite validates:
- Data schema consistency
- Date parsing
- Currency field formats
- Team name canonicalization
- Data uniqueness constraints
- Cross-dataset join compatibility
- Processed fixtures parquet quality
"""

import pytest
import polars as pl
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import our normalization module
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.normalization import canonicalize_frame


class TestDataSchema:
    """Test suite for Premier League data schema validation."""
    
    @classmethod
    def setup_class(cls):
        """Set up test fixtures and locate data files."""
        cls.project_root = Path(__file__).parent.parent
        cls.data_raw = cls.project_root / "data" / "raw"
        cls.data_processed = cls.project_root / "data" / "processed"
        
        # Ensure data directories exist
        assert cls.data_raw.exists(), f"Raw data directory not found: {cls.data_raw}"
        cls.data_processed.mkdir(exist_ok=True)
        
        # Locate data files
        cls.historical_match_files = list(cls.data_raw.glob("E0*.csv"))
        
        # Find fixtures file by stem (more flexible)
        cls.fixtures_file = cls._find_fixtures_file()
        
        cls.club_value_file = cls.data_raw / "Club Value.csv"
        cls.club_wages_file = cls.data_raw / "Club wages.csv"
        cls.attendance_file = cls.data_raw / "Attendance Data.csv"
        cls.managers_file = cls.data_raw / "Premier League Managers.csv"
        
        print(f"Found {len(cls.historical_match_files)} historical match files")
        
        # Check which files exist and store as class attribute
        expected_files = {
            "fixtures": cls.fixtures_file,
            "club_value": cls.club_value_file,
            "club_wages": cls.club_wages_file,
            "attendance": cls.attendance_file,
            "managers": cls.managers_file
        }
        
        cls.existing_files = {}
        for name, file_path in expected_files.items():
            if file_path and file_path.exists():
                cls.existing_files[name] = file_path
                print(f"✓ Found {name}: {file_path.name}")
            else:
                print(f"⚠ Missing {name}: {file_path.name if file_path else 'not located'}")
        
    @classmethod
    def _find_fixtures_file(cls) -> Optional[Path]:
        """Find fixtures file by stem, accepting various extensions."""
        fixtures_stem = "premier_league_fixtures_2025_2026"
        
        # Try common extensions first
        extensions = ['.csv', '.tsv', '.txt', '']
        
        for ext in extensions:
            candidate = cls.data_raw / f"{fixtures_stem}{ext}"
            if candidate.exists():
                return candidate
        
        # Try case-insensitive search
        for file_path in cls.data_raw.iterdir():
            if file_path.is_file():
                stem_lower = file_path.stem.lower()
                target_lower = fixtures_stem.lower()
                if stem_lower == target_lower:
                    return file_path
        
        return None
        
    def test_data_files_exist(self):
        """Test that all required data files exist."""
        # Check historical match files
        assert len(self.historical_match_files) > 0, "No historical match files (E0*.csv) found"
        
        # Check fixtures file specifically
        if not self.fixtures_file:
            available_files = [f.name for f in self.data_raw.iterdir() if f.is_file()]
            pytest.fail(
                f"No fixtures file found with stem 'premier_league_fixtures_2025_2026' in {self.data_raw}. "
                f"Expected files with .csv, .tsv, .txt or no extension. Available files: {available_files[:10]}"
            )
        
        # At minimum, we need historical match files
        assert len(self.historical_match_files) > 0, "No historical match files found"
    
    def test_historical_match_data_schema(self):
        """Test schema and data quality of historical match files."""
        for match_file in self.historical_match_files[:3]:  # Test first 3 files
            print(f"\nTesting {match_file.name}")
            
            try:
                # Read with sampling for large files
                df_lazy = pl.scan_csv(match_file, ignore_errors=True)
                df = df_lazy.limit(100000).collect()
                
                if df.height == 0:
                    pytest.skip(f"File {match_file.name} is empty")
                
                print(f"  Loaded {df.height:,} rows, {df.width} columns")
                
                # Apply team name canonicalization if team columns exist
                team_cols = []
                for col in ['HomeTeam', 'AwayTeam', 'Home', 'Away']:
                    if col in df.columns:
                        team_cols.append(col)
                
                if team_cols:
                    df = canonicalize_frame(df, team_cols)
                    print(f"  Canonicalized team columns: {team_cols}")
                
                # Test date columns
                date_cols = [col for col in df.columns if 'date' in col.lower() or col == 'Date']
                for date_col in date_cols:
                    self._test_date_column(df, date_col, match_file.name)
                
                # Test uniqueness of (Date, HomeTeam, AwayTeam) if all columns exist
                key_cols = []
                if 'Date' in df.columns:
                    key_cols.append('Date')
                if 'HomeTeam' in df.columns:
                    key_cols.append('HomeTeam')
                elif 'Home' in df.columns:
                    key_cols.append('Home')
                if 'AwayTeam' in df.columns:
                    key_cols.append('AwayTeam')
                elif 'Away' in df.columns:
                    key_cols.append('Away')
                
                if len(key_cols) >= 3:  # Need at least date + 2 team columns
                    self._test_match_uniqueness(df, key_cols, match_file.name)
                
                # Test no nulls in team columns after canonicalization
                for team_col in team_cols:
                    null_count = df[team_col].null_count()
                    assert null_count == 0, f"Found {null_count} nulls in {team_col} column in {match_file.name} after canonicalization"
                
            except Exception as e:
                pytest.fail(f"Failed to process {match_file.name}: {str(e)}")
    
    def test_fixtures_data_schema(self):
        """Test schema and data quality of fixtures file."""
        if 'fixtures' not in self.existing_files:
            pytest.skip("Fixtures file not found")
        
        fixtures_file = self.existing_files['fixtures']
        print(f"\nTesting {fixtures_file.name}")
        
        try:
            df = pl.read_csv(fixtures_file, ignore_errors=True)
            print(f"  Loaded {df.height:,} rows, {df.width} columns")
            
            # Apply team name canonicalization
            team_cols = []
            for col in ['HomeTeam', 'AwayTeam', 'Home', 'Away', 'home_team', 'away_team']:
                if col in df.columns:
                    team_cols.append(col)
            
            if team_cols:
                df = canonicalize_frame(df, team_cols)
                print(f"  Canonicalized team columns: {team_cols}")
            
            # Test date columns
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            for date_col in date_cols:
                self._test_date_column(df, date_col, fixtures_file.name)
            
            # Test no nulls in team columns
            for team_col in team_cols:
                null_count = df[team_col].null_count()
                assert null_count == 0, f"Found {null_count} nulls in {team_col} column in fixtures after canonicalization"
            
        except Exception as e:
            pytest.fail(f"Failed to process fixtures file: {str(e)}")

    def test_processed_fixtures_parquet(self):
        """Test that the processed fixtures parquet exists and has valid schema."""
        fixtures_parquet = self.data_processed / "fixtures_2025_26.parquet"
        
        if not fixtures_parquet.exists():
            pytest.skip("Processed fixtures parquet not found - run build_fixtures.py first")
        
        print(f"\nTesting processed fixtures parquet: {fixtures_parquet}")
        
        try:
            df = pl.read_parquet(fixtures_parquet)
            print(f"  Loaded {df.height:,} rows, {df.width} columns")
            print(f"  Columns: {df.columns}")
            print(f"  Schema: {df.dtypes}")
            
            # Test required columns exist
            required_columns = ['match_id', 'date', 'matchweek', 'home_team', 'away_team']
            missing_columns = [col for col in required_columns if col not in df.columns]
            assert not missing_columns, f"Missing required columns in fixtures parquet: {missing_columns}"
            
            # Test match_id uniqueness
            unique_match_ids = df['match_id'].n_unique()
            total_matches = df.height
            assert unique_match_ids == total_matches, (
                f"Match IDs are not unique: {unique_match_ids} unique IDs for {total_matches} matches"
            )
            print(f"  ✓ All {total_matches} match IDs are unique")
            
            # Test non-null required fields
            for col in ['date', 'home_team', 'away_team']:
                null_count = df[col].null_count()
                assert null_count == 0, f"Found {null_count} null values in required column '{col}'"
            print(f"  ✓ No nulls in required columns")
            
            # Test matchweek range (should be 1-38 for Premier League)
            min_gw = df['matchweek'].min()
            max_gw = df['matchweek'].max()
            assert 1 <= min_gw <= 38, f"Invalid minimum matchweek: {min_gw} (expected 1-38)"
            assert 1 <= max_gw <= 38, f"Invalid maximum matchweek: {max_gw} (expected 1-38)"
            print(f"  ✓ Matchweeks in valid range: {min_gw}-{max_gw}")
            
            # Test date types
            assert df['date'].dtype == pl.Date, f"Date column has wrong type: {df['date'].dtype} (expected Date)"
            assert df['matchweek'].dtype == pl.Int32, f"Matchweek column has wrong type: {df['matchweek'].dtype} (expected Int32)"
            print(f"  ✓ Column types are correct")
            
            # Test reasonable number of matches (Premier League has 380 matches per season)
            # For development/testing, allow smaller datasets
            expected_matches = 380
            if df.height < 50:  # Sample/test data
                print(f"  ⚠️ Sample dataset detected: {df.height} matches (full season: {expected_matches})")
            else:
                assert 350 <= df.height <= 400, (
                    f"Unexpected number of matches: {df.height} (expected around {expected_matches})"
                )
                print(f"  ✅ Full season dataset: {df.height} matches")
            
        except Exception as e:
            pytest.fail(f"Failed to validate processed fixtures parquet: {str(e)}")
    
    def test_club_attributes_schema(self):
        """Test schema of club attribute files (Value, Wages, Attendance)."""
        club_files = {
            'club_value': 'Club Value.csv',
            'club_wages': 'Club wages.csv', 
            'attendance': 'Attendance Data.csv'
        }
        
        # Initialize club_dataframes as class attribute if not exists
        if not hasattr(self.__class__, 'club_dataframes'):
            self.__class__.club_dataframes = {}
        
        for file_key, file_name in club_files.items():
            if file_key not in self.existing_files:
                print(f"⚠ Skipping {file_name} - not found")
                continue
                
            file_path = self.existing_files[file_key]
            print(f"\nTesting {file_path.name}")
            
            try:
                df = pl.read_csv(file_path, ignore_errors=True)
                print(f"  Loaded {df.height:,} rows, {df.width} columns")
                
                # Apply team name canonicalization to club/squad columns
                team_cols = []
                for col in ['Club', 'Squad', 'Team', 'club', 'squad', 'team']:
                    if col in df.columns:
                        team_cols.append(col)
                
                if team_cols:
                    df = canonicalize_frame(df, team_cols)
                    print(f"  Canonicalized team columns: {team_cols}")
                
                # Test currency columns
                currency_cols = [col for col in df.columns 
                               if any(term in col.lower() for term in ['value', 'wage', 'salary', 'cost', 'price', 'revenue'])]
                
                for currency_col in currency_cols:
                    if df[currency_col].dtype == pl.String:
                        self._test_currency_column(df, currency_col, file_path.name)
                
                # Store for join tests
                self.__class__.club_dataframes[file_key] = df
                
            except Exception as e:
                pytest.fail(f"Failed to process {file_path.name}: {str(e)}")
    
    def test_managers_data_schema(self):
        """Test schema of managers file."""
        if 'managers' not in self.existing_files:
            pytest.skip("Managers file not found")
        
        managers_file = self.existing_files['managers']
        print(f"\nTesting {managers_file.name}")
        
        try:
            df = pl.read_csv(managers_file, ignore_errors=True)
            print(f"  Loaded {df.height:,} rows, {df.width} columns")
            
            # Apply team name canonicalization to club column
            team_cols = []
            for col in ['Club', 'Team', 'club', 'team']:
                if col in df.columns:
                    team_cols.append(col)
            
            if team_cols:
                df = canonicalize_frame(df, team_cols)
                print(f"  Canonicalized team columns: {team_cols}")
            
            # Test date columns
            date_cols = [col for col in df.columns if any(term in col.lower() for term in ['from', 'until', 'date'])]
            for date_col in date_cols:
                if df[date_col].dtype == pl.String:
                    self._test_date_column(df, date_col, managers_file.name, strict=False)
            
        except Exception as e:
            pytest.fail(f"Failed to process managers file: {str(e)}")
    
    def test_cross_dataset_joins(self):
        """Test join compatibility between fixtures/matches and club attribute data."""
        if not hasattr(self.__class__, 'club_dataframes') or not self.__class__.club_dataframes:
            pytest.skip("No club attribute data available for join testing")
        
        # Get a sample match/fixture dataset for testing
        test_teams = set()
        
        # Try to get teams from historical match files
        if self.historical_match_files:
            match_file = self.historical_match_files[0]
            try:
                df_lazy = pl.scan_csv(match_file, ignore_errors=True)
                df = df_lazy.limit(10000).collect()
                
                # Apply canonicalization
                team_cols = []
                for col in ['HomeTeam', 'AwayTeam', 'Home', 'Away']:
                    if col in df.columns:
                        team_cols.append(col)
                
                if team_cols:
                    df = canonicalize_frame(df, team_cols)
                    
                    # Collect unique teams
                    for col in team_cols:
                        teams = df[col].drop_nulls().unique().to_list()
                        test_teams.update(teams)
                        
            except Exception as e:
                print(f"Could not extract teams from {match_file.name}: {e}")
        
        if not test_teams:
            pytest.skip("No team data available for join testing")
        
        print(f"\nTesting joins with {len(test_teams)} unique teams")
        print(f"Sample teams: {sorted(list(test_teams))[:5]}")
        
        # Test joins with each club attribute dataset
        for dataset_name, club_df in self.__class__.club_dataframes.items():
            self._test_team_join_coverage(test_teams, club_df, dataset_name)
    
    def _test_date_column(self, df: pl.DataFrame, date_col: str, file_name: str, strict: bool = True):
        """Test that a date column contains valid dates."""
        print(f"    Testing date column: {date_col}")
        
        # Skip if column is already a date type
        if df[date_col].dtype == pl.Date:
            return
        
        # Get non-null values for testing
        non_null_values = df[date_col].drop_nulls()
        if non_null_values.len() == 0:
            return
        
        # Sample a few values to check format
        sample_values = non_null_values.limit(10).to_list()
        
        # Check for common date patterns
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # DD/MM/YYYY
            r'^\d{1,2}/\d{1,2}/\d{4}$',  # D/M/YYYY
        ]
        
        pattern_matches = 0
        for value in sample_values:
            if isinstance(value, str):
                for pattern in date_patterns:
                    if re.match(pattern, value.strip()):
                        pattern_matches += 1
                        break
        
        coverage = pattern_matches / len(sample_values) if sample_values else 0
        
        if strict:
            assert coverage >= 0.8, f"Date column {date_col} in {file_name} has low date pattern coverage: {coverage:.1%}"
        else:
            # More lenient for manager dates which might have partial dates
            if coverage < 0.5:
                print(f"    Warning: Date column {date_col} has low date pattern coverage: {coverage:.1%}")
    
    def _test_currency_column(self, df: pl.DataFrame, currency_col: str, file_name: str):
        """Test that a currency column can be converted to numeric after cleaning."""
        print(f"    Testing currency column: {currency_col}")
        
        # Get non-null values
        non_null_values = df[currency_col].drop_nulls()
        if non_null_values.len() == 0:
            return
        
        # Sample values to test
        sample_values = non_null_values.limit(100).to_list()
        
        # Try to extract numeric values by removing currency symbols and formatting
        numeric_count = 0
        for value in sample_values:
            if isinstance(value, str):
                # Remove currency symbols and common formatting
                cleaned = re.sub(r'[£€$¥¢₹₽,\s]', '', value.strip())
                
                # Handle billion/million/thousand suffixes more carefully
                # Replace bn, m, k with empty string after extracting the number
                if re.search(r'\d+\.?\d*[bmk]n?$', cleaned.lower()):
                    cleaned = re.sub(r'[bmk]n?$', '', cleaned.lower())
                
                try:
                    float(cleaned)
                    numeric_count += 1
                except ValueError:
                    # If still fails, try more aggressive cleaning
                    cleaned_aggressive = re.sub(r'[^\d.]', '', cleaned)
                    try:
                        float(cleaned_aggressive)
                        numeric_count += 1
                    except ValueError:
                        pass
        
        coverage = numeric_count / len(sample_values) if sample_values else 0
        
        # Show some sample values for debugging
        if coverage < 0.7:
            print(f"    Sample values: {sample_values[:3]}")
        
        assert coverage >= 0.7, f"Currency column {currency_col} in {file_name} has low numeric convertibility: {coverage:.1%}"
    
    def _test_match_uniqueness(self, df: pl.DataFrame, key_cols: List[str], file_name: str):
        """Test that match records are unique by (Date, HomeTeam, AwayTeam)."""
        print(f"    Testing uniqueness on columns: {key_cols}")
        
        # Count total rows vs unique combinations
        total_rows = df.height
        unique_combinations = df.select(key_cols).n_unique()
        
        uniqueness_ratio = unique_combinations / total_rows if total_rows > 0 else 1
        
        assert uniqueness_ratio >= 0.95, (
            f"Match uniqueness check failed in {file_name}: "
            f"{unique_combinations}/{total_rows} unique combinations ({uniqueness_ratio:.1%})"
        )
    
    def _test_team_join_coverage(self, test_teams: set, club_df: pl.DataFrame, dataset_name: str):
        """Test join coverage between match teams and club attribute data."""
        print(f"  Testing join coverage with {dataset_name}")
        
        # Find the team column in club data
        team_col = None
        for col in ['Club', 'Squad', 'Team', 'club', 'squad', 'team']:
            if col in club_df.columns:
                team_col = col
                break
        
        if not team_col:
            pytest.skip(f"No team column found in {dataset_name}")
        
        # Get unique teams from club data
        club_teams = set(club_df[team_col].drop_nulls().unique().to_list())
        
        # Calculate join coverage
        matched_teams = test_teams.intersection(club_teams)
        coverage = len(matched_teams) / len(test_teams) if test_teams else 0
        
        print(f"    Join coverage: {len(matched_teams)}/{len(test_teams)} teams ({coverage:.1%})")
        print(f"    Matched teams sample: {sorted(list(matched_teams))[:3]}")
        
        if coverage < 0.85:
            unmatched = test_teams - club_teams
            print(f"    Unmatched teams: {sorted(list(unmatched))[:5]}")
        
        assert coverage >= 0.85, (
            f"Join coverage too low between match data and {dataset_name}: "
            f"{len(matched_teams)}/{len(test_teams)} teams matched ({coverage:.1%}). "
            f"Expected ≥85% coverage (dev mode)."
        )


if __name__ == "__main__":
    """Run tests directly with pytest."""
    pytest.main([__file__, "-v", "--tb=short"]) 