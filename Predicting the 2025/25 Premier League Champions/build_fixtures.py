#!/usr/bin/env python3
"""
Premier League Fixtures Builder

Locates and processes Premier League 2025/26 fixtures into a clean, canonical format.
Handles various file formats and missing extensions, applies team name canonicalization,
and creates a structured parquet file for modeling.

Usage:
    python src/build_fixtures.py --in data/raw --out data/processed
"""

import argparse
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

# Import our normalization module
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.normalization import canonicalize_team

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FixturesBuilder:
    """Builds clean Premier League fixtures from raw data."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        """Initialize the fixtures builder."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.fixtures_stem = "premier_league_fixtures_2025_2026"
        
        # Ensure directories exist
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_fixtures_file(self) -> Path:
        """
        Find fixtures file by stem, accepting various extensions or no extension.
        
        Returns:
            Path to the fixtures file
            
        Raises:
            FileNotFoundError: If no fixtures file is found
        """
        logger.info(f"Looking for fixtures file with stem: {self.fixtures_stem}")
        
        # Try common extensions first
        extensions = ['.csv', '.tsv', '.txt', '']
        
        for ext in extensions:
            candidate = self.input_dir / f"{self.fixtures_stem}{ext}"
            if candidate.exists():
                logger.info(f"Found fixtures file: {candidate}")
                return candidate
        
        # Try case-insensitive search
        for file_path in self.input_dir.iterdir():
            if file_path.is_file():
                stem_lower = file_path.stem.lower()
                target_lower = self.fixtures_stem.lower()
                if stem_lower == target_lower:
                    logger.info(f"Found fixtures file (case-insensitive): {file_path}")
                    return file_path
        
        # List available files for debugging
        available_files = [f.name for f in self.input_dir.iterdir() if f.is_file()]
        raise FileNotFoundError(
            f"No fixtures file found with stem '{self.fixtures_stem}' in {self.input_dir}. "
            f"Available files: {available_files}"
        )
    
    def detect_delimiter(self, file_path: Path) -> str:
        """
        Detect delimiter for CSV-like files.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected delimiter
        """
        if file_path.suffix.lower() == '.tsv':
            return '\t'
        elif file_path.suffix.lower() in ['.csv', '.txt']:
            return ','
        else:
            # No extension - try to detect by examining content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                
                # Count delimiters to guess format
                comma_count = first_line.count(',')
                tab_count = first_line.count('\t')
                semicolon_count = first_line.count(';')
                pipe_count = first_line.count('|')
                
                # Choose delimiter with highest count
                delim_counts = [(',', comma_count), ('\t', tab_count), (';', semicolon_count), ('|', pipe_count)]
                best_delim = max(delim_counts, key=lambda x: x[1])
                
                if best_delim[1] > 0:
                    logger.info(f"Detected delimiter: '{best_delim[0]}' (count: {best_delim[1]})")
                    return best_delim[0]
                else:
                    logger.warning("No clear delimiter detected, defaulting to comma")
                    return ','
                    
            except Exception as e:
                logger.warning(f"Error detecting delimiter: {e}, defaulting to comma")
                return ','
    
    def load_fixtures_data(self, file_path: Path) -> pl.DataFrame:
        """
        Load fixtures data from file with robust error handling.
        
        Args:
            file_path: Path to fixtures file
            
        Returns:
            Raw fixtures DataFrame
        """
        logger.info(f"Loading fixtures data from: {file_path}")
        
        delimiter = self.detect_delimiter(file_path)
        
        try:
            # Try scan_csv first for efficiency
            if file_path.stat().st_size > 10_000:  # Use lazy loading for larger files
                df = pl.scan_csv(
                    file_path,
                    separator=delimiter,
                    ignore_errors=True,
                    infer_schema_length=1000
                ).collect()
            else:
                df = pl.read_csv(
                    file_path,
                    separator=delimiter,
                    ignore_errors=True,
                    infer_schema_length=1000
                )
                
            logger.info(f"Loaded {df.height} rows, {df.width} columns")
            logger.info(f"Columns: {df.columns}")
            
            if df.height == 0:
                raise ValueError("Fixtures file is empty")
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to load with delimiter '{delimiter}': {e}")
            
            # Fallback: try other common delimiters
            fallback_delimiters = [',', '\t', ';', '|']
            for fallback_delim in fallback_delimiters:
                if fallback_delim == delimiter:
                    continue
                    
                try:
                    logger.info(f"Trying fallback delimiter: '{fallback_delim}'")
                    df = pl.read_csv(
                        file_path,
                        separator=fallback_delim,
                        ignore_errors=True
                    )
                    
                    if df.height > 0:
                        logger.info(f"Success with delimiter '{fallback_delim}': {df.height} rows")
                        return df
                        
                except Exception:
                    continue
            
            raise ValueError(f"Could not load fixtures file with any delimiter: {e}")
    
    def standardize_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Standardize column names to expected format.
        
        Args:
            df: Raw fixtures DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        logger.info("Standardizing column names")
        
        # Mapping of possible column names to standard names
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower().strip()
            
            if any(term in col_lower for term in ['date', 'game_date', 'match_date']):
                column_mapping[col] = 'date'
            elif any(term in col_lower for term in ['time', 'kick_off', 'kickoff', 'start_time']):
                column_mapping[col] = 'time'
            elif any(term in col_lower for term in ['home', 'home_team', 'hometeam']):
                column_mapping[col] = 'home_team'
            elif any(term in col_lower for term in ['away', 'away_team', 'awayteam']):
                column_mapping[col] = 'away_team'
            elif any(term in col_lower for term in ['venue', 'stadium', 'ground']):
                column_mapping[col] = 'venue'
            elif any(term in col_lower for term in ['week', 'round', 'gameweek', 'matchweek', 'gw']):
                column_mapping[col] = 'matchweek'
        
        # Apply column renaming
        if column_mapping:
            logger.info(f"Column mapping: {column_mapping}")
            df = df.rename(column_mapping)
        
        # Check for required columns
        required_cols = ['date', 'home_team', 'away_team']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.error(f"Available columns: {df.columns}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Standardized columns: {df.columns}")
        return df
    
    def clean_and_parse_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean and parse date column to YYYY-MM-DD format.
        
        Args:
            df: DataFrame with date column
            
        Returns:
            DataFrame with parsed date column
        """
        logger.info("Parsing date column")
        
        # Sample some date values to understand format
        sample_dates = df['date'].drop_nulls().limit(5).to_list()
        logger.info(f"Sample date values: {sample_dates}")
        
        try:
            # Try multiple date parsing strategies
            df = df.with_columns([
                pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d', strict=False).alias('parsed_date')
            ])
            
            # If that didn't work, try other common formats
            if df['parsed_date'].null_count() == df.height:
                logger.info("Trying alternative date formats")
                
                # Try DD/MM/YYYY format
                df = df.with_columns([
                    pl.col('date').str.strptime(pl.Date, format='%d/%m/%Y', strict=False).alias('parsed_date')
                ])
                
                # Try MM/DD/YYYY format
                if df['parsed_date'].null_count() == df.height:
                    df = df.with_columns([
                        pl.col('date').str.strptime(pl.Date, format='%m/%d/%Y', strict=False).alias('parsed_date')
                    ])
            
            # Check parsing success
            null_dates = df['parsed_date'].null_count()
            if null_dates > 0:
                logger.warning(f"Could not parse {null_dates} date values")
                
            # Replace original date column
            df = df.drop('date').rename({'parsed_date': 'date'})
            
            logger.info(f"Successfully parsed {df.height - null_dates} dates")
            return df
            
        except Exception as e:
            logger.error(f"Date parsing failed: {e}")
            raise ValueError(f"Could not parse date column: {e}")
    
    def clean_time_column(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean and parse time column to HH:MM format.
        
        Args:
            df: DataFrame with optional time column
            
        Returns:
            DataFrame with cleaned kickoff_local column
        """
        if 'time' not in df.columns:
            logger.info("No time column found, setting kickoff_local to null")
            return df.with_columns([
                pl.lit(None, dtype=pl.String).alias('kickoff_local')
            ])
        
        logger.info("Cleaning time column")
        
        # Sample some time values
        sample_times = df['time'].drop_nulls().limit(5).to_list()
        logger.info(f"Sample time values: {sample_times}")
        
        try:
            # Clean time strings - remove timezone info, standardize format
            df = df.with_columns([
                pl.col('time')
                .str.replace_all(r'\s*(UK|GMT|BST|UTC)\s*', '', literal=False)
                .str.replace_all(r'[^\d:]', '')
                .str.extract(r'(\d{1,2}:\d{2})', 1)
                .alias('kickoff_local')
            ])
            
            # Validate time format and pad if needed
            df = df.with_columns([
                pl.when(pl.col('kickoff_local').str.len_chars() == 4)
                .then(pl.concat_str([pl.lit('0'), pl.col('kickoff_local')]))
                .otherwise(pl.col('kickoff_local'))
                .alias('kickoff_local')
            ])
            
            # Drop original time column
            df = df.drop('time')
            
            valid_times = df['kickoff_local'].drop_nulls().len()
            logger.info(f"Successfully parsed {valid_times} time values")
            
            return df
            
        except Exception as e:
            logger.warning(f"Time parsing failed: {e}, setting to null")
            return df.drop('time').with_columns([
                pl.lit(None, dtype=pl.String).alias('kickoff_local')
            ])
    
    def canonicalize_team_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply team name canonicalization.
        
        Args:
            df: DataFrame with team columns
            
        Returns:
            DataFrame with canonicalized team names
        """
        logger.info("Canonicalizing team names")
        
        # Apply canonicalization to team columns
        df = df.with_columns([
            pl.col('home_team').map_elements(canonicalize_team, return_dtype=pl.String).alias('home_team'),
            pl.col('away_team').map_elements(canonicalize_team, return_dtype=pl.String).alias('away_team')
        ])
        
        # Log unique teams
        all_teams = set(df['home_team'].unique().to_list() + df['away_team'].unique().to_list())
        logger.info(f"Found {len(all_teams)} unique teams after canonicalization")
        logger.info(f"Sample teams: {sorted(list(all_teams))[:5]}")
        
        return df
    
    def infer_matchweek(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Infer matchweek if not present.
        
        Args:
            df: DataFrame with dates and teams
            
        Returns:
            DataFrame with matchweek column
        """
        if 'matchweek' in df.columns:
            logger.info("Matchweek column already present")
            return df
        
        logger.info("Inferring matchweek from fixture order")
        
        # Sort by date to get chronological order
        df = df.sort('date')
        
        # Track gameweek for each team
        team_gameweeks = {}
        matchweeks = []
        
        for row in df.iter_rows(named=True):
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get next gameweek for each team
            home_gw = team_gameweeks.get(home_team, 0) + 1
            away_gw = team_gameweeks.get(away_team, 0) + 1
            
            # Matchweek is the maximum of both teams' gameweeks
            matchweek = max(home_gw, away_gw)
            matchweeks.append(matchweek)
            
            # Update team gameweeks
            team_gameweeks[home_team] = home_gw
            team_gameweeks[away_team] = away_gw
        
        # Add matchweek column
        df = df.with_columns([
            pl.Series('matchweek', matchweeks, dtype=pl.Int32)
        ])
        
        max_gw = max(matchweeks)
        logger.info(f"Inferred matchweeks 1-{max_gw}")
        
        return df
    
    def generate_match_ids(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate stable match IDs using SHA1 hash.
        
        Args:
            df: DataFrame with match data
            
        Returns:
            DataFrame with match_id column
        """
        logger.info("Generating match IDs")
        
        def create_match_id(date_val, home_team, away_team, kickoff_local):
            """Create a stable match ID from match details."""
            # Convert date to string if it's a date object
            if hasattr(date_val, 'strftime'):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val)
            
            # Handle null kickoff time
            time_str = kickoff_local if kickoff_local is not None else '15:00'
            
            # Create stable identifier
            match_string = f"{date_str}|{home_team}|{away_team}|{time_str}"
            
            # Generate SHA1 hash
            return hashlib.sha1(match_string.encode('utf-8')).hexdigest()[:12]
        
        # Apply match ID generation
        df = df.with_columns([
            pl.struct(['date', 'home_team', 'away_team', 'kickoff_local'])
            .map_elements(
                lambda x: create_match_id(x['date'], x['home_team'], x['away_team'], x['kickoff_local']),
                return_dtype=pl.String
            )
            .alias('match_id')
        ])
        
        # Check for duplicate match IDs
        unique_ids = df['match_id'].n_unique()
        total_matches = df.height
        
        if unique_ids != total_matches:
            logger.warning(f"Found {total_matches - unique_ids} duplicate match IDs")
        else:
            logger.info(f"Generated {unique_ids} unique match IDs")
        
        return df
    
    def finalize_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Finalize the output schema with proper column order and types.
        
        Args:
            df: DataFrame with all processed columns
            
        Returns:
            Final DataFrame with clean schema
        """
        logger.info("Finalizing output schema")
        
        # Define final column order
        final_columns = ['match_id', 'date', 'matchweek', 'home_team', 'away_team', 'kickoff_local']
        
        # Add venue if available
        if 'venue' in df.columns:
            final_columns.append('venue')
        
        # Select and order columns
        df = df.select(final_columns)
        
        # Ensure proper data types
        df = df.with_columns([
            pl.col('match_id').cast(pl.String),
            pl.col('date').cast(pl.Date),
            pl.col('matchweek').cast(pl.Int32),
            pl.col('home_team').cast(pl.String),
            pl.col('away_team').cast(pl.String),
            pl.col('kickoff_local').cast(pl.String)
        ])
        
        if 'venue' in df.columns:
            df = df.with_columns([
                pl.col('venue').cast(pl.String)
            ])
        
        logger.info(f"Final schema: {df.dtypes}")
        return df
    
    def build_fixtures(self) -> Path:
        """
        Main method to build clean fixtures parquet.
        
        Returns:
            Path to the output parquet file
        """
        logger.info("Starting fixtures build process")
        
        # Step 1: Find fixtures file
        fixtures_file = self.find_fixtures_file()
        
        # Step 2: Load raw data
        df = self.load_fixtures_data(fixtures_file)
        
        # Step 3: Standardize column names
        df = self.standardize_column_names(df)
        
        # Step 4: Parse dates
        df = self.clean_and_parse_dates(df)
        
        # Step 5: Clean time column
        df = self.clean_time_column(df)
        
        # Step 6: Canonicalize team names
        df = self.canonicalize_team_names(df)
        
        # Step 7: Infer matchweek if needed
        df = self.infer_matchweek(df)
        
        # Step 8: Generate match IDs
        df = self.generate_match_ids(df)
        
        # Step 9: Finalize schema
        df = self.finalize_schema(df)
        
        # Step 10: Save parquet
        output_path = self.output_dir / "fixtures_2025_26.parquet"
        df.write_parquet(output_path)
        
        logger.info(f"✅ Fixtures parquet saved: {output_path}")
        logger.info(f"📊 Final dataset: {df.height} matches, {df.width} columns")
        
        # Log sample of final data
        logger.info("Sample fixtures:")
        print(df.head(3))
        
        return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build clean Premier League fixtures parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/build_fixtures.py --in data/raw --out data/processed
  python src/build_fixtures.py --in /path/to/raw --out /path/to/processed
        """
    )
    
    parser.add_argument(
        '--in', '--input',
        dest='input_dir',
        type=str,
        default='data/raw',
        help='Input directory containing fixtures file (default: data/raw)'
    )
    
    parser.add_argument(
        '--out', '--output',
        dest='output_dir',
        type=str,
        default='data/processed',
        help='Output directory for parquet file (default: data/processed)'
    )
    
    args = parser.parse_args()
    
    try:
        builder = FixturesBuilder(args.input_dir, args.output_dir)
        output_path = builder.build_fixtures()
        
        print(f"\n🎉 Success! Fixtures built at: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to build fixtures: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 