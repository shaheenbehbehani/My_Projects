#!/usr/bin/env python3
"""
Season Form Features Builder

Builds season-to-date and rolling form features from historical matches:
- Rolling 5-match xG differential and points per game
- Elo-style team ratings with home advantage
- Pre-season baselines for 2025/26 predictions

Output: data/processed/features/form_history.parquet
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import math

import polars as pl
import numpy as np

# Import our normalization module
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.normalization import canonicalize_frame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SeasonFormBuilder:
    """Builds season form and Elo rating features from historical match data."""
    
    # Elo rating parameters
    DEFAULT_ELO = 1500
    K_FACTOR = 20
    HOME_ADVANTAGE = 60
    
    def __init__(self, data_raw: Path, output_dir: Path):
        """Initialize the season form builder."""
        self.data_raw = Path(data_raw)
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track team statistics
        self.team_elos = {}
        self.team_form_history = []
        
    def load_historical_matches(self) -> Optional[pl.DataFrame]:
        """Load and combine historical match data."""
        logger.info("Loading historical match data...")
        
        historical_files = list(self.data_raw.glob("E0*.csv"))
        if not historical_files:
            logger.warning("No historical match files (E0*.csv) found")
            return None
        
        logger.info(f"Found {len(historical_files)} historical match files")
        
        all_matches = []
        
        for match_file in historical_files:
            try:
                # Load file
                df = pl.read_csv(match_file, ignore_errors=True)
                logger.info(f"Loaded {match_file.name}: {df.height:,} rows")
                
                # Add season identifier from filename
                season = self._extract_season_from_filename(match_file.name)
                df = df.with_columns([pl.lit(season).alias('season')])
                
                # Canonicalize team names
                team_cols = []
                for col in ['HomeTeam', 'AwayTeam', 'Home', 'Away']:
                    if col in df.columns:
                        team_cols.append(col)
                
                if team_cols:
                    df = canonicalize_frame(df, team_cols)
                
                # Standardize column names
                df = self._standardize_match_columns(df)
                
                all_matches.append(df)
                
            except Exception as e:
                logger.error(f"Failed to load {match_file.name}: {e}")
                continue
        
        if not all_matches:
            logger.error("No match files could be loaded")
            return None
        
        # Combine all seasons
        combined_df = pl.concat(all_matches, how="diagonal_relaxed")
        logger.info(f"Combined dataset: {combined_df.height:,} total matches")
        
        return combined_df
    
    def _extract_season_from_filename(self, filename: str) -> str:
        """Extract season identifier from filename."""
        # E0.csv, E0 (1).csv, E0 (2).csv etc.
        if 'E0.csv' in filename:
            return '2023-24'  # Most recent
        elif 'E0 (1)' in filename:
            return '2022-23'
        elif 'E0 (2)' in filename:
            return '2021-22'
        else:
            return 'unknown'
    
    def _standardize_match_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize match data column names."""
        # Map columns to standard names
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower()
            
            if col_lower in ['date', 'matchdate', 'match_date']:
                column_mapping[col] = 'date'
            elif col_lower in ['hometeam', 'home', 'home_team']:
                column_mapping[col] = 'home_team'
            elif col_lower in ['awayteam', 'away', 'away_team']:
                column_mapping[col] = 'away_team'
            elif col_lower in ['fthg', 'home_goals', 'hg']:
                column_mapping[col] = 'home_goals'
            elif col_lower in ['ftag', 'away_goals', 'ag']:
                column_mapping[col] = 'away_goals'
            elif col_lower in ['ftr', 'result', 'match_result']:
                column_mapping[col] = 'result'
        
        if column_mapping:
            df = df.rename(column_mapping)
        
        return df
    
    def parse_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Parse date column to proper date format."""
        if 'date' not in df.columns:
            logger.warning("No date column found - using sequence numbers")
            return df.with_columns([
                pl.int_range(len(df)).alias('match_sequence')
            ])
        
        try:
            # Try multiple date formats
            df = df.with_columns([
                pl.col('date').str.strptime(pl.Date, format='%d/%m/%Y', strict=False).alias('parsed_date')
            ])
            
            # If that fails, try other formats
            if df['parsed_date'].null_count() == df.height:
                df = df.with_columns([
                    pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d', strict=False).alias('parsed_date')
                ])
            
            # Replace original date
            df = df.drop('date').rename({'parsed_date': 'date'})
            
            # Sort by date
            df = df.sort('date')
            
            logger.info(f"Successfully parsed {df['date'].drop_nulls().len()} dates")
            
        except Exception as e:
            logger.warning(f"Date parsing failed: {e}")
            # Use sequence as fallback
            df = df.with_columns([
                pl.int_range(len(df)).alias('match_sequence')
            ])
        
        return df
    
    def calculate_expected_score(self, elo_home: float, elo_away: float) -> Tuple[float, float]:
        """Calculate expected scores using Elo formula."""
        # Adjust for home advantage
        elo_home_adj = elo_home + self.HOME_ADVANTAGE
        
        # Calculate expected scores
        expected_home = 1 / (1 + 10**((elo_away - elo_home_adj) / 400))
        expected_away = 1 - expected_home
        
        return expected_home, expected_away
    
    def update_elo_ratings(self, home_team: str, away_team: str, 
                          result: str, home_goals: int, away_goals: int) -> Tuple[float, float]:
        """Update Elo ratings based on match result."""
        # Get current ratings
        elo_home = self.team_elos.get(home_team, self.DEFAULT_ELO)
        elo_away = self.team_elos.get(away_team, self.DEFAULT_ELO)
        
        # Calculate expected scores
        expected_home, expected_away = self.calculate_expected_score(elo_home, elo_away)
        
        # Determine actual scores
        if result == 'H':  # Home win
            actual_home, actual_away = 1.0, 0.0
        elif result == 'A':  # Away win
            actual_home, actual_away = 0.0, 1.0
        else:  # Draw
            actual_home, actual_away = 0.5, 0.5
        
        # Goal difference multiplier for larger updates
        goal_diff = abs(home_goals - away_goals)
        multiplier = math.log(goal_diff + 1) + 1
        
        # Update ratings
        rating_change_home = self.K_FACTOR * multiplier * (actual_home - expected_home)
        rating_change_away = self.K_FACTOR * multiplier * (actual_away - expected_away)
        
        new_elo_home = elo_home + rating_change_home
        new_elo_away = elo_away + rating_change_away
        
        # Store updated ratings
        self.team_elos[home_team] = new_elo_home
        self.team_elos[away_team] = new_elo_away
        
        return new_elo_home, new_elo_away
    
    def calculate_rolling_metrics(self, matches_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate rolling form metrics for each team."""
        logger.info("Calculating rolling form metrics...")
        
        # Parse dates first
        matches_df = self.parse_dates(matches_df)
        
        # Initialize Elo ratings for all teams
        all_teams = set()
        for col in ['home_team', 'away_team']:
            if col in matches_df.columns:
                teams = matches_df[col].drop_nulls().unique().to_list()
                all_teams.update(teams)
        
        logger.info(f"Found {len(all_teams)} unique teams")
        
        # Initialize Elo ratings
        for team in all_teams:
            self.team_elos[team] = self.DEFAULT_ELO
        
        # Process matches chronologically
        team_match_history = {team: [] for team in all_teams}
        form_records = []
        
        for i, row in enumerate(matches_df.iter_rows(named=True)):
            try:
                home_team = row['home_team']
                away_team = row['away_team']
                
                if not home_team or not away_team:
                    continue
                
                home_goals = row.get('home_goals', 0) or 0
                away_goals = row.get('away_goals', 0) or 0
                result = row.get('result', 'D')
                match_date = row.get('date')
                season = row.get('season', 'unknown')
                
                # Get pre-match Elo ratings
                elo_home_pre = self.team_elos.get(home_team, self.DEFAULT_ELO)
                elo_away_pre = self.team_elos.get(away_team, self.DEFAULT_ELO)
                
                # Calculate rolling metrics for both teams
                home_form = self._calculate_team_rolling_form(team_match_history[home_team])
                away_form = self._calculate_team_rolling_form(team_match_history[away_team])
                
                # Record pre-match form
                form_records.append({
                    'team': home_team,
                    'date': match_date,
                    'season': season,
                    'elo_pre': elo_home_pre,
                    'roll5_xg_diff': home_form['xg_diff'],
                    'roll5_ppg': home_form['ppg'],
                    'is_home': True
                })
                
                form_records.append({
                    'team': away_team,
                    'date': match_date,
                    'season': season,
                    'elo_pre': elo_away_pre,
                    'roll5_xg_diff': away_form['xg_diff'],
                    'roll5_ppg': away_form['ppg'],
                    'is_home': False
                })
                
                # Update Elo ratings
                new_elo_home, new_elo_away = self.update_elo_ratings(
                    home_team, away_team, result, home_goals, away_goals
                )
                
                # Add match to team histories
                home_match = {
                    'date': match_date,
                    'goals_for': home_goals,
                    'goals_against': away_goals,
                    'points': 3 if result == 'H' else (1 if result == 'D' else 0),
                    'xg_for': row.get('home_xg', home_goals),  # Fallback to goals if no xG
                    'xg_against': row.get('away_xg', away_goals)
                }
                
                away_match = {
                    'date': match_date,
                    'goals_for': away_goals,
                    'goals_against': home_goals,
                    'points': 3 if result == 'A' else (1 if result == 'D' else 0),
                    'xg_for': row.get('away_xg', away_goals),
                    'xg_against': row.get('home_xg', home_goals)
                }
                
                team_match_history[home_team].append(home_match)
                team_match_history[away_team].append(away_match)
                
                # Keep only last 10 matches for rolling calculations
                team_match_history[home_team] = team_match_history[home_team][-10:]
                team_match_history[away_team] = team_match_history[away_team][-10:]
                
            except Exception as e:
                logger.warning(f"Error processing match {i}: {e}")
                continue
        
        # Convert to DataFrame
        form_df = pl.DataFrame(form_records)
        logger.info(f"Generated {form_df.height:,} form records")
        
        return form_df
    
    def _calculate_team_rolling_form(self, match_history: List[Dict]) -> Dict[str, float]:
        """Calculate rolling form metrics for a team."""
        if len(match_history) < 1:
            return {'xg_diff': 0.0, 'ppg': 0.0}
        
        # Take last 5 matches
        recent_matches = match_history[-5:]
        
        if not recent_matches:
            return {'xg_diff': 0.0, 'ppg': 0.0}
        
        # Calculate metrics
        total_points = sum(match['points'] for match in recent_matches)
        total_xg_for = sum(match['xg_for'] for match in recent_matches)
        total_xg_against = sum(match['xg_against'] for match in recent_matches)
        
        ppg = total_points / len(recent_matches)
        xg_diff = (total_xg_for - total_xg_against) / len(recent_matches)
        
        return {
            'xg_diff': round(xg_diff, 3),
            'ppg': round(ppg, 3)
        }
    
    def get_pre_season_baselines(self, form_df: pl.DataFrame, cutoff_date: str = '2025-08-01') -> pl.DataFrame:
        """Extract pre-season baselines for 2025/26."""
        logger.info(f"Extracting pre-season baselines before {cutoff_date}")
        
        try:
            cutoff = datetime.strptime(cutoff_date, '%Y-%m-%d').date()
            
            # Filter to matches before cutoff
            pre_cutoff = form_df.filter(
                pl.col('date') < cutoff
            )
            
            if pre_cutoff.height == 0:
                logger.warning("No data before cutoff date - using latest available")
                pre_cutoff = form_df
            
            # Get latest form for each team
            baselines = (
                pre_cutoff
                .sort(['team', 'date'])
                .group_by('team')
                .agg([
                    pl.col('elo_pre').last().alias('elo_baseline'),
                    pl.col('roll5_xg_diff').last().alias('xg_diff_baseline'),
                    pl.col('roll5_ppg').last().alias('ppg_baseline'),
                    pl.col('date').last().alias('last_match_date')
                ])
            )
            
            logger.info(f"Generated baselines for {baselines.height} teams")
            
            return baselines
            
        except Exception as e:
            logger.error(f"Failed to generate baselines: {e}")
            
            # Fallback: use final Elo ratings
            baseline_data = []
            for team, elo in self.team_elos.items():
                baseline_data.append({
                    'team': team,
                    'elo_baseline': elo,
                    'xg_diff_baseline': 0.0,
                    'ppg_baseline': 1.0,
                    'last_match_date': None
                })
            
            return pl.DataFrame(baseline_data)
    
    def build_form_features(self) -> Path:
        """Build complete form features dataset."""
        logger.info("Starting form features build...")
        
        # Load historical matches
        matches_df = self.load_historical_matches()
        if matches_df is None:
            raise ValueError("No historical match data available")
        
        # Calculate rolling metrics and Elo ratings
        form_df = self.calculate_rolling_metrics(matches_df)
        
        # Get pre-season baselines
        baselines_df = self.get_pre_season_baselines(form_df)
        
        # Save complete form history
        form_output = self.output_dir / "form_history.parquet"
        form_df.write_parquet(form_output)
        logger.info(f"✅ Form history saved: {form_output}")
        
        # Save pre-season baselines
        baselines_output = self.output_dir / "form_baselines_2025_26.parquet"
        baselines_df.write_parquet(baselines_output)
        logger.info(f"✅ Pre-season baselines saved: {baselines_output}")
        
        # Log final Elo ratings
        logger.info("Final Elo ratings (top 10):")
        sorted_elos = sorted(self.team_elos.items(), key=lambda x: x[1], reverse=True)
        for i, (team, elo) in enumerate(sorted_elos[:10]):
            logger.info(f"  {i+1:2d}. {team}: {elo:.1f}")
        
        return form_output


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    data_raw = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed" / "features"
    
    try:
        builder = SeasonFormBuilder(data_raw, output_dir)
        output_path = builder.build_form_features()
        
        print(f"\n🏁 Season Form Features Complete!")
        print(f"📊 Output: {output_path}")
        print(f"📊 Baselines: {output_path.parent / 'form_baselines_2025_26.parquet'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to build form features: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 