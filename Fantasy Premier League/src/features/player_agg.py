#!/usr/bin/env python3
"""
Player Aggregation Features Builder

Aggregates squad-level strengths from individual player data:
- Player season stats (goals, xG, xA, key passes, minutes)
- Shot quality metrics (avg xG per shot, shots per game)
- Uses latest 2 seasons with 60/40 weighting

Output: data/processed/features/player_agg.parquet
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import polars as pl
import numpy as np

# Import our normalization module
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.normalization import canonicalize_frame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlayerAggBuilder:
    """Builds squad-level player aggregation features."""
    
    def __init__(self, data_raw: Path, output_dir: Path):
        """Initialize the player aggregation builder."""
        self.data_raw = Path(data_raw)
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Weighting for multi-season data (most recent gets higher weight)
        self.season_weights = [0.6, 0.4]  # Current season, previous season
    
    def load_player_season_data(self) -> Optional[pl.DataFrame]:
        """Load and combine player season statistics."""
        logger.info("Loading player season data...")
        
        # Look for player season files
        player_files = list(self.data_raw.glob("players_epl_*.csv"))
        if not player_files:
            logger.warning("No player season files (players_epl_*.csv) found")
            return None
        
        logger.info(f"Found {len(player_files)} player season files")
        
        all_players = []
        
        for player_file in player_files:
            try:
                # Extract season from filename
                season = self._extract_season_from_filename(player_file.name)
                
                df = pl.read_csv(player_file, ignore_errors=True)
                logger.info(f"Loaded {player_file.name}: {df.height:,} players")
                
                # Add season identifier
                df = df.with_columns([pl.lit(season).alias('season')])
                
                # Standardize column names
                df = self._standardize_player_columns(df)
                
                # Find and canonicalize team column
                team_col = self._find_team_column(df)
                if team_col:
                    df = canonicalize_frame(df, [team_col])
                    df = df.rename({team_col: 'team'})
                else:
                    logger.warning(f"No team column found in {player_file.name}")
                    continue
                
                all_players.append(df)
                
            except Exception as e:
                logger.error(f"Failed to load {player_file.name}: {e}")
                continue
        
        if not all_players:
            logger.error("No player season files could be loaded")
            return None
        
        # Combine all seasons
        combined_df = pl.concat(all_players, how="diagonal_relaxed")
        logger.info(f"Combined player data: {combined_df.height:,} player records")
        
        return combined_df
    
    def load_shots_data(self) -> Optional[pl.DataFrame]:
        """Load and process shot-level data."""
        logger.info("Loading shots data...")
        
        # Look for shots files
        shots_files = list(self.data_raw.glob("shots_epl_*.csv"))
        if not shots_files:
            logger.warning("No shots files (shots_epl_*.csv) found")
            return None
        
        logger.info(f"Found {len(shots_files)} shots files")
        
        all_shots = []
        
        for shots_file in shots_files:
            try:
                # Extract season from filename
                season = self._extract_season_from_filename(shots_file.name)
                
                df = pl.read_csv(shots_file, ignore_errors=True)
                logger.info(f"Loaded {shots_file.name}: {df.height:,} shots")
                
                # Add season identifier
                df = df.with_columns([pl.lit(season).alias('season')])
                
                # Standardize column names
                df = self._standardize_shots_columns(df)
                
                # Find and canonicalize team column
                team_col = self._find_team_column(df)
                if team_col:
                    df = canonicalize_frame(df, [team_col])
                    df = df.rename({team_col: 'team'})
                else:
                    logger.warning(f"No team column found in {shots_file.name}")
                    continue
                
                all_shots.append(df)
                
            except Exception as e:
                logger.error(f"Failed to load {shots_file.name}: {e}")
                continue
        
        if not all_shots:
            logger.warning("No shots files could be loaded")
            return None
        
        # Combine all seasons
        combined_df = pl.concat(all_shots, how="diagonal_relaxed")
        logger.info(f"Combined shots data: {combined_df.height:,} shot records")
        
        return combined_df
    
    def _extract_season_from_filename(self, filename: str) -> str:
        """Extract season from filename."""
        # Look for patterns like 14-15, 2014-15, etc.
        season_match = re.search(r'(\d{2,4}[-_]\d{2})', filename)
        if season_match:
            season_str = season_match.group(1)
            # Normalize to YYYY-YY format
            if len(season_str.split('-')[0]) == 2:
                year1 = '20' + season_str.split('-')[0]
                year2 = season_str.split('-')[1]
                return f"{year1}-{year2}"
            else:
                return season_str
        else:
            return 'unknown'
    
    def _find_team_column(self, df: pl.DataFrame) -> Optional[str]:
        """Find the team column in a DataFrame."""
        possible_team_cols = ['team', 'Team', 'club', 'Club', 'squad', 'Squad']
        
        for col in possible_team_cols:
            if col in df.columns:
                return col
        
        return None
    
    def _standardize_player_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize player data column names."""
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Goals
            if col_lower in ['goals', 'goal', 'g']:
                column_mapping[col] = 'goals'
            # Expected goals
            elif col_lower in ['xg', 'expectedgoals', 'expected_goals']:
                column_mapping[col] = 'xg'
            # Expected assists
            elif col_lower in ['xa', 'xassists', 'expected_assists']:
                column_mapping[col] = 'xa'
            # Assists
            elif col_lower in ['assists', 'assist', 'a']:
                column_mapping[col] = 'assists'
            # Key passes
            elif 'key' in col_lower and 'pass' in col_lower:
                column_mapping[col] = 'key_passes'
            # Minutes
            elif col_lower in ['minutes', 'mins', 'min']:
                column_mapping[col] = 'minutes'
            # Shots
            elif col_lower in ['shots', 'shot']:
                column_mapping[col] = 'shots'
        
        if column_mapping:
            df = df.rename(column_mapping)
        
        return df
    
    def _standardize_shots_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize shots data column names."""
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Expected goals
            if col_lower in ['xg', 'expectedgoals', 'expected_goals']:
                column_mapping[col] = 'shot_xg'
            # Match ID
            elif 'match' in col_lower and 'id' in col_lower:
                column_mapping[col] = 'match_id'
        
        if column_mapping:
            df = df.rename(column_mapping)
        
        return df
    
    def aggregate_player_stats(self, players_df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate player stats to team level."""
        logger.info("Aggregating player stats to team level...")
        
        # Define columns to aggregate
        agg_columns = []
        
        # Sum columns (totals)
        sum_cols = ['goals', 'assists', 'shots', 'key_passes', 'minutes']
        for col in sum_cols:
            if col in players_df.columns:
                agg_columns.append(pl.col(col).sum().alias(f'squad_{col}'))
        
        # Mean columns for quality metrics
        mean_cols = ['xg', 'xa']
        for col in mean_cols:
            if col in players_df.columns:
                agg_columns.append(pl.col(col).sum().alias(f'squad_{col}'))
        
        if not agg_columns:
            logger.warning("No valid columns found for aggregation")
            return pl.DataFrame({'team': [], 'season': []})
        
        # Aggregate by team and season
        team_stats = (
            players_df
            .group_by(['team', 'season'])
            .agg(agg_columns + [pl.count().alias('squad_size')])
        )
        
        # Calculate per-90 minute stats where minutes are available
        if 'squad_minutes' in team_stats.columns:
            per_90_stats = []
            
            for col in ['squad_goals', 'squad_assists', 'squad_shots', 'squad_key_passes', 'squad_xg', 'squad_xa']:
                if col in team_stats.columns:
                    stat_name = col.replace('squad_', '')
                    per_90_col = f'{stat_name}_90'
                    per_90_stats.append(
                        (pl.col(col) / pl.col('squad_minutes') * 90).alias(per_90_col)
                    )
            
            if per_90_stats:
                team_stats = team_stats.with_columns(per_90_stats)
        
        logger.info(f"Aggregated stats for {team_stats.height} team-seasons")
        
        return team_stats
    
    def aggregate_shot_quality(self, shots_df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate shot quality metrics to team level."""
        logger.info("Aggregating shot quality metrics...")
        
        if 'shot_xg' not in shots_df.columns:
            logger.warning("No shot xG column found in shots data")
            return pl.DataFrame({'team': [], 'season': []})
        
        # Aggregate by team and season
        shot_stats = (
            shots_df
            .group_by(['team', 'season'])
            .agg([
                pl.col('shot_xg').mean().alias('avg_xg_per_shot'),
                pl.col('shot_xg').count().alias('total_shots'),
                pl.col('shot_xg').sum().alias('total_shot_xg')
            ])
        )
        
        # Calculate shots per game (approximate)
        # Assume ~38 games per season
        shot_stats = shot_stats.with_columns([
            (pl.col('total_shots') / 38).alias('shots_per_game')
        ])
        
        logger.info(f"Aggregated shot quality for {shot_stats.height} team-seasons")
        
        return shot_stats
    
    def apply_season_weights(self, team_stats: pl.DataFrame) -> pl.DataFrame:
        """Apply temporal weighting to multi-season data."""
        logger.info("Applying season weights...")
        
        # Get available seasons sorted by recency
        seasons = sorted(team_stats['season'].unique().to_list(), reverse=True)
        logger.info(f"Available seasons: {seasons}")
        
        if len(seasons) < 2:
            logger.info("Only one season available - no weighting applied")
            return team_stats
        
        # Take the most recent 2 seasons
        recent_seasons = seasons[:2]
        logger.info(f"Using seasons: {recent_seasons} with weights {self.season_weights}")
        
        # Filter to recent seasons
        recent_data = team_stats.filter(pl.col('season').is_in(recent_seasons))
        
        # Add weights
        weight_map = {season: weight for season, weight in zip(recent_seasons, self.season_weights)}
        
        conditions = []
        for season, weight in weight_map.items():
            conditions.append(
                pl.when(pl.col('season') == season).then(weight)
            )
        
        weighted_data = recent_data.with_columns([
            pl.concat_list(conditions).first().alias('season_weight')
        ])
        
        # Calculate weighted averages by team
        numeric_cols = []
        for col in weighted_data.columns:
            if col not in ['team', 'season', 'season_weight'] and weighted_data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                numeric_cols.append(col)
        
        if not numeric_cols:
            logger.warning("No numeric columns found for weighting")
            return recent_data.group_by('team').first()
        
        # Weight and aggregate
        agg_exprs = []
        for col in numeric_cols:
            weighted_col = f'weighted_{col}'
            agg_exprs.append(
                (pl.col(col) * pl.col('season_weight')).sum().alias(weighted_col)
            )
        
        # Also sum the weights for normalization
        agg_exprs.append(pl.col('season_weight').sum().alias('total_weight'))
        
        weighted_teams = (
            weighted_data
            .group_by('team')
            .agg(agg_exprs)
        )
        
        # Normalize by total weight
        final_cols = [pl.col('team')]
        for col in numeric_cols:
            weighted_col = f'weighted_{col}'
            if weighted_col in weighted_teams.columns:
                final_cols.append(
                    (pl.col(weighted_col) / pl.col('total_weight')).alias(col)
                )
        
        result = weighted_teams.select(final_cols)
        
        logger.info(f"Applied weights to {result.height} teams")
        
        return result
    
    def build_player_agg_features(self) -> Path:
        """Build complete player aggregation features."""
        logger.info("Starting player aggregation features build...")
        
        # Load player and shots data
        players_df = self.load_player_season_data()
        shots_df = self.load_shots_data()
        
        feature_dfs = []
        
        # Process player stats
        if players_df is not None:
            team_stats = self.aggregate_player_stats(players_df)
            if team_stats.height > 0:
                weighted_stats = self.apply_season_weights(team_stats)
                feature_dfs.append(weighted_stats)
        
        # Process shot quality
        if shots_df is not None:
            shot_stats = self.aggregate_shot_quality(shots_df)
            if shot_stats.height > 0:
                weighted_shots = self.apply_season_weights(shot_stats)
                feature_dfs.append(weighted_shots)
        
        if not feature_dfs:
            logger.warning("No player data available - creating empty output")
            empty_df = pl.DataFrame({
                'team': pl.Series([], dtype=pl.String),
                'squad_goals_90': pl.Series([], dtype=pl.Float64),
                'squad_xg_90': pl.Series([], dtype=pl.Float64),
                'squad_xa_90': pl.Series([], dtype=pl.Float64),
                'avg_xg_per_shot': pl.Series([], dtype=pl.Float64),
                'shots_per_game': pl.Series([], dtype=pl.Float64)
            })
            
            output_path = self.output_dir / "player_agg.parquet"
            empty_df.write_parquet(output_path)
            return output_path
        
        # Combine all feature DataFrames
        combined_df = feature_dfs[0]
        for df in feature_dfs[1:]:
            combined_df = combined_df.join(df, on='team', how='outer')
        
        # Fill missing values with league averages or defaults
        logger.info("Filling missing values with defaults...")
        
        default_values = {
            'squad_goals_90': 1.5,
            'squad_xg_90': 1.4,
            'squad_xa_90': 1.2,
            'avg_xg_per_shot': 0.1,
            'shots_per_game': 12.0
        }
        
        for col, default_val in default_values.items():
            if col in combined_df.columns:
                null_count = combined_df[col].null_count()
                if null_count > 0:
                    logger.info(f"  Filling {null_count} null values in {col} with {default_val}")
                    combined_df = combined_df.with_columns([
                        pl.col(col).fill_null(default_val)
                    ])
        
        # Save output
        output_path = self.output_dir / "player_agg.parquet"
        combined_df.write_parquet(output_path)
        
        logger.info(f"✅ Player aggregation features saved: {output_path}")
        logger.info(f"📊 Final dataset: {combined_df.height} teams, {combined_df.width} features")
        
        # Log feature summary
        logger.info("Feature summary:")
        for col in combined_df.columns:
            if col != 'team':
                non_null = combined_df[col].drop_nulls().len()
                mean_val = combined_df[col].mean()
                logger.info(f"  {col}: {non_null}/{combined_df.height} non-null, mean={mean_val:.3f}")
        
        return output_path


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    data_raw = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed" / "features"
    
    try:
        builder = PlayerAggBuilder(data_raw, output_dir)
        output_path = builder.build_player_agg_features()
        
        print(f"\n👥 Player Aggregation Features Complete!")
        print(f"📊 Output: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to build player aggregation features: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 