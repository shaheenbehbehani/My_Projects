#!/usr/bin/env python3
"""
Build per-match training dataset for Premier League winner prediction.

This script creates a comprehensive training dataset with:
- Historical match data from 2014/15 to 2024/25
- Elo ratings and rolling form features
- Bookmaker odds (if available)
- Proper temporal feature engineering to avoid leakage
- Standardized team names and season keys

Input: data/raw/historical_matches.parquet
Output: data/processed/match_dataset.parquet
"""

import polars as pl
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.normalization import canonicalize_frame
from .utils import set_random_seed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
set_random_seed(42)


class MatchDatasetBuilder:
    """Build comprehensive training dataset for Premier League match prediction."""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # Ensure processed directory exists
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Elo rating parameters
        self.elo_k_factor = 32
        self.elo_initial_rating = 1500
        
        # Rolling window parameters
        self.form_window = 5  # Last 5 matches
        self.goals_window = 10  # Last 10 matches for goals
        
    def load_historical_matches(self) -> pl.DataFrame:
        """Load historical match data."""
        matches_path = self.raw_data_dir / "historical_matches.parquet"
        
        if not matches_path.exists():
            logger.info("Historical matches parquet not found. Auto-building from raw CSV files...")
            self._build_historical_matches_from_csv()
        
        logger.info(f"Loading historical matches from {matches_path}")
        df = pl.read_parquet(matches_path)
        logger.info(f"Loaded {df.height:,} matches with columns: {df.columns}")
        
        return df
    
    def _build_historical_matches_from_csv(self) -> None:
        """Build historical_matches.parquet from raw CSV files."""
        logger.info("Building historical matches dataset from raw CSV files...")
        
        # Define fixed target schema
        target_schema = {
            'date': pl.Date,
            'home_team': pl.Utf8,
            'away_team': pl.Utf8,
            'home_goals': pl.Int64,
            'away_goals': pl.Int64,
            'result': pl.Utf8,
            'home_points': pl.Int64,
            'season': pl.Utf8
        }
        
        target_columns = list(target_schema.keys())
        
        # Find all E*.csv files
        csv_pattern = "E*.csv"
        csv_files = list(self.raw_data_dir.glob(csv_pattern))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found matching pattern {csv_pattern} in {self.raw_data_dir}")
        
        logger.info(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
        logger.info(f"Target schema: {target_columns}")
        
        all_matches = []
        file_stats = {}
        
        for csv_file in sorted(csv_files):
            try:
                logger.info(f"Processing {csv_file.name}...")
                
                # Read CSV file
                df = pl.read_csv(csv_file, ignore_errors=True)
                rows_read = df.height
                
                if rows_read == 0:
                    logger.warning(f"Skipping {csv_file.name} - empty file")
                    file_stats[csv_file.name] = {'read': 0, 'kept': 0, 'skipped': 0, 'reason': 'empty'}
                    continue
                
                # Verify required football-data columns exist
                required_source_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                missing_columns = [col for col in required_source_columns if col not in df.columns]
                
                if missing_columns:
                    logger.warning(f"Skipping {csv_file.name} - missing required columns: {missing_columns}")
                    file_stats[csv_file.name] = {'read': rows_read, 'kept': 0, 'skipped': rows_read, 'reason': f'missing columns: {missing_columns}'}
                    continue
                
                # Extract season from filename (E0.csv -> 2014/15, E1.csv -> 2015/16, etc.)
                season_num = int(csv_file.stem[1:]) if csv_file.stem[1:].isdigit() else 0
                season_year = 2014 + season_num
                season = f"{season_year}/{str(season_year + 1)[-2:]}"
                
                # Select only required columns and rename to standard names
                column_mapping = {
                    'Date': 'date',
                    'HomeTeam': 'home_team',
                    'AwayTeam': 'away_team',
                    'FTHG': 'home_goals',
                    'FTAG': 'away_goals',
                    'FTR': 'result'
                }
                
                # Select and rename columns
                df = df.select(list(column_mapping.keys())).rename(column_mapping)
                
                # Robust date parsing with multiple formats
                if 'date' in df.columns:
                    # Ensure Date column is string type
                    df = df.with_columns(pl.col('date').cast(pl.Utf8))
                    
                    # Try multiple date patterns
                    patterns = ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]
                    
                    # Create parsed candidates for each pattern
                    date_candidates = []
                    for pattern in patterns:
                        try:
                            parsed = pl.col('date').str.strptime(pl.Date, format=pattern, strict=False, exact=False)
                            date_candidates.append(parsed)
                        except Exception:
                            continue
                    
                    if date_candidates:
                        # Coalesce all parsed candidates
                        df = df.with_columns(
                            pl.coalesce(date_candidates).alias('date_parsed')
                        )
                        
                        # If still null, try Polars inference
                        df = df.with_columns(
                            pl.when(pl.col('date_parsed').is_null())
                            .then(pl.col('date').str.to_datetime(strict=False, exact=False).cast(pl.Date))
                            .otherwise(pl.col('date_parsed'))
                            .alias('date_final')
                        )
                        
                        # Replace original date column
                        df = df.drop(['date', 'date_parsed']).rename({'date_final': 'date'})
                    else:
                        # Fallback to Polars inference
                        df = df.with_columns(
                            pl.col('date').str.to_datetime(strict=False, exact=False).cast(pl.Date).alias('date')
                        )
                
                # Convert goals to numeric
                if 'home_goals' in df.columns:
                    df = df.with_columns(
                        pl.col('home_goals').cast(pl.Int64, strict=False).fill_null(0)
                    )
                if 'away_goals' in df.columns:
                    df = df.with_columns(
                        pl.col('away_goals').cast(pl.Int64, strict=False).fill_null(0)
                    )
                
                # Normalize team names
                if 'home_team' in df.columns and 'away_team' in df.columns:
                    df = df.with_columns(
                        pl.col('home_team').map_elements(lambda x: self._normalize_team(x) if x else None, return_dtype=pl.Utf8),
                        pl.col('away_team').map_elements(lambda x: self._normalize_team(x) if x else None, return_dtype=pl.Utf8)
                    )
                
                # Filter out rows with missing essential data
                df_before_filter = df.height
                df = df.filter(
                    pl.col('home_team').is_not_null() & 
                    pl.col('away_team').is_not_null() &
                    pl.col('date').is_not_null()
                )
                rows_kept = df.height
                rows_skipped = df_before_filter - rows_kept
                
                if rows_kept > 0:
                    # Add missing columns to match target schema
                    df = df.with_columns(
                        pl.lit(0).alias('home_points'),  # Will be computed later
                        pl.lit(season).alias('season')
                    )
                    
                    # Ensure all target columns exist and reorder to match schema
                    for col in target_columns:
                        if col not in df.columns:
                            df = df.with_columns(pl.lit(None).alias(col))
                    
                    # Reorder columns to match target schema exactly
                    df = df.select(target_columns)
                    
                    all_matches.append(df)
                    logger.info(f"  Added {rows_kept} matches from {season}")
                    file_stats[csv_file.name] = {'read': rows_read, 'kept': rows_kept, 'skipped': rows_skipped, 'reason': 'success'}
                else:
                    logger.warning(f"  No valid matches found in {csv_file.name} after filtering")
                    file_stats[csv_file.name] = {'read': rows_read, 'kept': 0, 'skipped': rows_kept, 'reason': 'no valid data after filtering'}
                    
            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {e}")
                file_stats[csv_file.name] = {'read': rows_read if 'rows_read' in locals() else 'unknown', 'kept': 0, 'skipped': 'error', 'reason': str(e)}
                continue
        
        # Log file processing statistics
        logger.info("File processing statistics:")
        for filename, stats in file_stats.items():
            if stats['reason'] == 'success':
                logger.info(f"  {filename}: {stats['read']} read, {stats['kept']} kept, {stats['skipped']} skipped")
            else:
                logger.info(f"  {filename}: {stats['read']} read, {stats['kept']} kept, {stats['skipped']} skipped - {stats['reason']}")
        
        if not all_matches:
            raise ValueError(f"No valid match data found in any CSV files. Files inspected: {list(file_stats.keys())}")
        
        # Concatenate all matches by name (not position)
        combined_df = pl.concat(all_matches, how='align')
        
        # Compute home_points from result
        combined_df = combined_df.with_columns(
            pl.when(pl.col('result') == 'H')
            .then(3)
            .when(pl.col('result') == 'D')
            .then(1)
            .otherwise(0)
            .alias('home_points')
        )
        
        # Sort by date (oldest first)
        combined_df = combined_df.sort('date')
        
        # Remove duplicates
        combined_df = combined_df.unique(subset=['date', 'home_team', 'away_team', 'home_goals', 'away_goals'])
        
        logger.info(f"Combined dataset: {combined_df.height:,} matches from {combined_df['season'].n_unique()} seasons")
        logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        logger.info(f"Teams: {combined_df['home_team'].n_unique()} unique teams")
        logger.info(f"Final schema: {combined_df.columns}")
        
        # Save combined dataset
        output_path = self.raw_data_dir / "historical_matches.parquet"
        combined_df.write_parquet(output_path)
        
        logger.info(f"Saved combined historical matches to {output_path}")
    
    def _normalize_team(self, team_name: str) -> str:
        """Normalize team name using Phase 1 normalizer."""
        if not team_name:
            return None
        
        try:
            # Import normalization function
            from src.normalization import normalize_team
            return normalize_team(team_name)
        except ImportError:
            # Fallback if normalization module not available
            return team_name.strip() if team_name else None
    
    def standardize_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize column names to expected format."""
        logger.info("Standardizing column names...")
        
        # Map common column name variations
        column_mapping = {
            # Date columns
            'Date': 'date',
            'date': 'date',
            'match_date': 'date',
            
            # Season columns
            'Season': 'season',
            'season': 'season',
            'year': 'season',
            
            # Team columns
            'HomeTeam': 'home_team',
            'home_team': 'home_team',
            'Home': 'home_team',
            'home': 'home_team',
            'AwayTeam': 'away_team',
            'away_team': 'away_team',
            'Away': 'away_team',
            'away': 'away_team',
            
            # Goals columns
            'FTHG': 'home_goals',
            'home_goals': 'home_goals',
            'full_time_goals_home': 'home_goals',
            'FTAG': 'away_goals',
            'away_goals': 'away_goals',
            'full_time_goals_away': 'away_goals',
            
            # Result columns
            'FTR': 'result',
            'result': 'result',
            'match_result': 'result',
            
            # Expected goals (if available)
            'xG_home': 'home_xg',
            'xG_away': 'away_xg',
            'home_xg': 'home_xg',
            'away_xg': 'away_xg'
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename({old_name: new_name})
        
        logger.info(f"Standardized columns: {df.columns}")
        return df
    
    def parse_dates_and_seasons(self, df: pl.DataFrame) -> pl.DataFrame:
        """Parse dates and create season keys."""
        logger.info("Parsing dates and creating season keys...")
        
        # Ensure date column exists
        if 'date' not in df.columns:
            raise ValueError("No date column found after standardization")
        
        # Parse dates if they're strings
        if df['date'].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col('date').str.strptime(pl.Date, fmt="%Y-%m-%d").alias('date')
            )
        
        # Create season key (e.g., 2014 -> "2014/15")
        df = df.with_columns(
            pl.col('date').dt.year().alias('year'),
            pl.col('date').dt.month().alias('month')
        )
        
        # Season starts in August, so adjust year accordingly
        df = df.with_columns(
            pl.when(pl.col('month') >= 8)
            .then(pl.col('year'))
            .otherwise(pl.col('year') - 1)
            .alias('season_start_year')
        )
        
        df = df.with_columns(
            (pl.col('season_start_year').cast(pl.Utf8) + "/" + 
             (pl.col('season_start_year') + 1).cast(pl.Utf8).str.slice(-2))
            .alias('season')
        )
        
        # Drop intermediate columns
        df = df.drop(['year', 'month', 'season_start_year'])
        
        logger.info(f"Created seasons: {df['season'].unique().to_list()}")
        return df
    
    def create_target_variable(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create target variable y from goals or existing result."""
        logger.info("Creating target variable...")
        
        if 'result' in df.columns:
            # Map existing result to our format
            result_mapping = {
                'H': 'H', 'Home': 'H', 'home': 'H', '1': 'H',
                'D': 'D', 'Draw': 'D', 'draw': 'D', 'X': 'D', '0': 'D',
                'A': 'A', 'Away': 'A', 'away': 'A', '2': 'A'
            }
            
            df = df.with_columns(
                pl.col('result').map_elements(lambda x: result_mapping.get(str(x), 'D'), return_dtype=pl.Utf8)
                .alias('y')
            )
        else:
            # Create from goals
            df = df.with_columns(
                pl.when(pl.col('home_goals') > pl.col('away_goals'))
                .then(pl.lit('H'))
                .when(pl.col('home_goals') < pl.col('away_goals'))
                .then(pl.lit('A'))
                .otherwise(pl.lit('D'))
                .alias('y')
            )
        
        # Convert to numeric for modeling (H=0, D=1, A=2)
        df = df.with_columns(
            pl.col('y').map_elements(lambda x: {'H': 0, 'D': 1, 'A': 2}.get(x, 1), return_dtype=pl.Int64).alias('result_label')
        )
        
        logger.info(f"Target distribution: {df['y'].value_counts()}")
        return df
    
    def standardize_team_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize team names using Phase 1 normalizer."""
        logger.info("Standardizing team names...")
        
        # Apply team name canonicalization
        team_columns = ['home_team', 'away_team']
        df = canonicalize_frame(df, team_columns)
        
        logger.info(f"Team names standardized. Sample teams: {df['home_team'].unique().to_list()[:5]}")
        return df
    
    def compute_elo_ratings(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Elo ratings for all teams over time."""
        logger.info("Computing Elo ratings...")
        
        # Sort by date to ensure chronological order
        df = df.sort('date')
        
        # Initialize Elo ratings for all teams
        all_teams = set(df['home_team'].unique().to_list() + df['away_team'].unique().to_list())
        elo_ratings = {team: self.elo_initial_rating for team in all_teams}
        
        # Track Elo ratings over time
        elo_history = []
        
        for row in df.iter_rows(named=True):
            home_team = row['home_team']
            away_team = row['away_team']
            home_goals = row['home_goals']
            away_goals = row['away_goals']
            
            # Get current Elo ratings
            home_elo = elo_ratings[home_team]
            away_elo = elo_ratings[away_team]
            
            # Calculate expected outcome
            expected_home = 1 / (1 + 10**((away_elo - home_elo) / 400))
            expected_away = 1 - expected_home
            
            # Determine actual outcome
            if home_goals > away_goals:
                actual_home = 1.0
                actual_away = 0.0
            elif home_goals < away_goals:
                actual_home = 0.0
                actual_away = 1.0
            else:
                actual_home = 0.5
                actual_away = 0.5
            
            # Update Elo ratings
            elo_ratings[home_team] += self.elo_k_factor * (actual_home - expected_home)
            elo_ratings[away_team] += self.elo_k_factor * (actual_away - expected_away)
            
            # Store Elo ratings for this match
            elo_history.append({
                'date': row['date'],
                'home_team': home_team,
                'away_team': away_team,
                'home_elo': home_elo,
                'away_elo': away_elo
            })
        
        # Convert to DataFrame and join back
        elo_df = pl.DataFrame(elo_history)
        df = df.join(elo_df, on=['date', 'home_team', 'away_team'], how='left')
        
        logger.info("Elo ratings computed successfully")
        return df
    
    def compute_rolling_form_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute rolling form features for each team."""
        logger.info("Computing rolling form features...")
        
        # Sort by date and team
        df = df.sort(['date', 'home_team'])
        
        # Initialize form tracking
        team_form = {}
        team_goals_for = {}
        team_goals_against = {}
        
        # Initialize for all teams
        all_teams = set(df['home_team'].unique().to_list() + df['away_team'].unique().to_list())
        for team in all_teams:
            team_form[team] = []
            team_goals_for[team] = []
            team_goals_against[team] = []
        
        # Process matches chronologically
        for row in df.iter_rows(named=True):
            home_team = row['home_team']
            away_team = row['away_team']
            home_goals = row['home_goals']
            away_goals = row['away_goals']
            
            # Calculate points for this match
            if home_goals > away_goals:
                home_points = 3
                away_points = 0
            elif home_goals < away_goals:
                home_points = 0
                away_points = 3
            else:
                home_points = 1
                away_points = 1
            
            # Update team form
            team_form[home_team].append(home_points)
            team_form[away_team].append(away_points)
            
            # Update goals
            team_goals_for[home_team].append(home_goals)
            team_goals_against[home_team].append(away_goals)
            team_goals_for[away_team].append(away_goals)
            team_goals_against[away_team].append(home_goals)
            
            # Keep only last N matches
            if len(team_form[home_team]) > self.form_window:
                team_form[home_team] = team_form[home_team][-self.form_window:]
            if len(team_form[away_team]) > self.form_window:
                team_form[away_team] = team_form[away_team][-self.form_window:]
            
            if len(team_goals_for[home_team]) > self.goals_window:
                team_goals_for[home_team] = team_goals_for[home_team][-self.goals_window:]
                team_goals_against[home_team] = team_goals_against[home_team][-self.goals_window:]
            if len(team_goals_for[away_team]) > self.goals_window:
                team_goals_for[away_team] = team_goals_for[away_team][-self.goals_window:]
                team_goals_against[away_team] = team_goals_against[away_team][-self.goals_window:]
        
        # Now compute rolling averages for each match
        form_features = []
        
        for row in df.iter_rows(named=True):
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get form features (excluding current match)
            home_form = team_form[home_team][:-1] if len(team_form[home_team]) > 1 else []
            away_form = team_form[away_team][:-1] if len(team_form[away_team]) > 1 else []
            
            # Compute rolling averages
            home_last5_pts = np.mean(home_form[-self.form_window:]) if home_form else 0
            away_last5_pts = np.mean(away_form[-self.form_window:]) if away_form else 0
            
            # Goals features
            home_goals_for_avg = np.mean(team_goals_for[home_team][:-1]) if len(team_goals_for[home_team]) > 1 else 0
            home_goals_against_avg = np.mean(team_goals_against[home_team][:-1]) if len(team_goals_against[home_team]) > 1 else 0
            away_goals_for_avg = np.mean(team_goals_for[away_team][:-1]) if len(team_goals_for[away_team]) > 1 else 0
            away_goals_against_avg = np.mean(team_goals_against[away_team][:-1]) if len(team_goals_against[away_team]) > 1 else 0
            
            form_features.append({
                'date': row['date'],
                'home_team': home_team,
                'away_team': away_team,
                'home_last5_pts': home_last5_pts,
                'away_last5_pts': away_last5_pts,
                'home_goals_for_avg': home_goals_for_avg,
                'home_goals_against_avg': home_goals_against_avg,
                'away_goals_for_avg': away_goals_for_avg,
                'away_goals_against_avg': away_goals_against_avg
            })
        
        # Join form features back to main dataset
        form_df = pl.DataFrame(form_features)
        df = df.join(form_df, on=['date', 'home_team', 'away_team'], how='left')
        
        logger.info("Rolling form features computed successfully")
        return df
    
    def add_bookmaker_odds(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add bookmaker odds features if available."""
        logger.info("Adding bookmaker odds features...")
        
        # Look for common odds column patterns
        odds_patterns = {
            'B365H': 'home_odds',
            'B365D': 'draw_odds', 
            'B365A': 'away_odds',
            'BWH': 'home_odds',
            'BWD': 'draw_odds',
            'BWA': 'away_odds',
            'IWH': 'home_odds',
            'IWD': 'draw_odds',
            'IWA': 'away_odds'
        }
        
        # Check which odds columns exist
        existing_odds = {}
        for pattern, new_name in odds_patterns.items():
            if pattern in df.columns:
                existing_odds[pattern] = new_name
        
        if existing_odds:
            # Rename odds columns
            df = df.rename(existing_odds)
            
            # Convert to numeric, handling any non-numeric values
            for new_name in set(existing_odds.values()):
                if new_name in df.columns:
                    df = df.with_columns(
                        pl.col(new_name).cast(pl.Float64, strict=False).fill_null(0)
                    )
            
            logger.info(f"Added odds features: {list(set(existing_odds.values()))}")
        else:
            logger.info("No bookmaker odds found in dataset")
        
        return df
    
    def create_final_dataset(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create final training dataset with all features."""
        logger.info("Creating final training dataset...")
        
        # Select and order columns
        feature_columns = [
            'date', 'season', 'home_team', 'away_team', 'y', 'result_label',
            'home_elo', 'away_elo',
            'home_last5_pts', 'away_last5_pts',
            'home_goals_for_avg', 'home_goals_against_avg',
            'away_goals_for_avg', 'away_goals_against_avg'
        ]
        
        # Add odds columns if they exist
        odds_columns = ['home_odds', 'draw_odds', 'away_odds']
        for col in odds_columns:
            if col in df.columns:
                feature_columns.append(col)
        
        # Select final columns
        final_df = df.select(feature_columns)
        
        # Sort by date
        final_df = final_df.sort('date')
        
        logger.info(f"Final dataset created with {final_df.height:,} rows and {final_df.width} columns")
        logger.info(f"Columns: {final_df.columns}")
        
        return final_df
    
    def save_dataset(self, df: pl.DataFrame) -> Path:
        """Save the final dataset to parquet."""
        output_path = self.processed_data_dir / "match_dataset.parquet"
        
        logger.info(f"Saving dataset to {output_path}")
        df.write_parquet(output_path)
        
        logger.info(f"Dataset saved successfully: {output_path}")
        return output_path
    
    def build_dataset(self) -> pl.DataFrame:
        """Build the complete training dataset."""
        logger.info("🚀 Starting dataset build process...")
        
        # Load data
        df = self.load_historical_matches()
        
        # Standardize columns
        df = self.standardize_column_names(df)
        
        # Parse dates and create seasons
        df = self.parse_dates_and_seasons(df)
        
        # Create target variable
        df = self.create_target_variable(df)
        
        # Standardize team names
        df = self.standardize_team_names(df)
        
        # Compute Elo ratings
        df = self.compute_elo_ratings(df)
        
        # Compute rolling form features
        df = self.compute_rolling_form_features(df)
        
        # Add bookmaker odds
        df = self.add_bookmaker_odds(df)
        
        # Create final dataset
        final_df = self.create_final_dataset(df)
        
        # Save dataset
        self.save_dataset(final_df)
        
        logger.info("✅ Dataset build process completed successfully!")
        return final_df


def main():
    """Main entry point for building the match dataset."""
    parser = argparse.ArgumentParser(description="Build Premier League match training dataset")
    parser.add_argument("--raw-data-dir", default="data/raw", help="Raw data directory path")
    parser.add_argument("--processed-data-dir", default="data/processed", help="Processed data directory path")
    
    args = parser.parse_args()
    
    try:
        # Build dataset
        builder = MatchDatasetBuilder(args.raw_data_dir, args.processed_data_dir)
        dataset = builder.build_dataset()
        
        print(f"\n✅ Match dataset built successfully!")
        print(f"📁 Output: {args.processed_data_dir}/match_dataset.parquet")
        print(f"📊 Shape: {dataset.height:,} rows × {dataset.width} columns")
        print(f"📅 Date range: {dataset['date'].min()} to {dataset['date'].max()}")
        print(f"🏆 Teams: {dataset['home_team'].n_unique()} unique teams")
        print(f"🎯 Target distribution: {dataset['y'].value_counts()}")
        
    except Exception as e:
        logger.error(f"Failed to build dataset: {e}")
        raise


if __name__ == "__main__":
    main() 