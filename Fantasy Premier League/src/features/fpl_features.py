#!/usr/bin/env python3
"""
FPL Feature Building Module - Step 2b
Creates model-ready feature tables from processed FPL data.

This module:
1. Builds team schedule strength features (rest days, fixture congestion, opponent difficulty)
2. Projects player points for next gameweek based on recent performance and opponent difficulty
3. Outputs lightweight, stable Parquet schemas ready for Step 3 merging

Functions:
- build_team_schedule_strength(): Team schedule analysis features
- project_points_next_gw(): Player performance projections
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_team_schedule_strength(
    fixtures_fp: str = "data/FPL/processed/fixtures_processed.parquet",
    teams_fp: str = "data/FPL/processed/teams_processed.parquet",
    k_next: int = 5,
    output_dir: str = "data/FPL/features"
) -> pd.DataFrame:
    """
    Build team schedule strength features for the next k fixtures.
    
    Computes:
    - Rest days between fixtures
    - Fixture congestion (games per week)
    - Opponent difficulty ratings
    - Travel distance considerations
    
    Args:
        fixtures_fp: Path to processed fixtures file
        teams_fp: Path to processed teams file
        k_next: Number of upcoming fixtures to analyze
        output_dir: Directory to save output features
        
    Returns:
        DataFrame with team schedule strength features
    """
    logger.info(f"Building team schedule strength features for next {k_next} fixtures...")
    
    # Load processed data
    fixtures = pd.read_parquet(fixtures_fp)
    teams = pd.read_parquet(teams_fp)
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get current gameweek (assume we're at the start of season for now)
    # In production, this would come from current FPL status
    current_gw = 1
    
    # Filter fixtures from current gameweek onwards
    upcoming_fixtures = fixtures[fixtures['gameweek'] >= current_gw].copy()
    
    # Sort by gameweek and kickoff time
    upcoming_fixtures = upcoming_fixtures.sort_values(['gameweek', 'kickoff_time'])
    
    # Initialize results storage
    schedule_features = []
    
    for team_id in teams['team_id'].unique():
        team_name = teams[teams['team_id'] == team_id]['team_name'].iloc[0]
        logger.info(f"Processing schedule for {team_name} (ID: {team_id})")
        
        # Get team's upcoming fixtures (home and away)
        team_fixtures = upcoming_fixtures[
            (upcoming_fixtures['home_team_id'] == team_id) | 
            (upcoming_fixtures['away_team_id'] == team_id)
        ].copy()
        
        if len(team_fixtures) == 0:
            logger.warning(f"No upcoming fixtures found for team {team_name}")
            continue
        
        # Limit to next k fixtures
        team_fixtures = team_fixtures.head(k_next)
        
        # Calculate features for each fixture
        for i, (_, fixture) in enumerate(team_fixtures.iterrows()):
            is_home = fixture['home_team_id'] == team_id
            opponent_id = fixture['away_team_id'] if is_home else fixture['home_team_id']
            opponent_name = teams[teams['team_id'] == opponent_id]['team_name'].iloc[0]
            
            # Calculate rest days since last fixture
            if i == 0:
                # First upcoming fixture - assume 7 days rest (standard week)
                rest_days = 7
            else:
                # Calculate days between this fixture and previous one
                prev_fixture = team_fixtures.iloc[i-1]
                prev_time = pd.to_datetime(prev_fixture['kickoff_time'])
                curr_time = pd.to_datetime(fixture['kickoff_time'])
                rest_days = (curr_time - prev_time).days
            
            # Calculate fixture congestion (games in same week)
            fixture_week = fixture['kickoff_time'].week if hasattr(fixture['kickoff_time'], 'week') else 1
            games_this_week = len(team_fixtures[
                team_fixtures['kickoff_time'].apply(
                    lambda x: x.week if hasattr(x, 'week') else 1
                ) == fixture_week
            ])
            
            # Calculate opponent difficulty (using FPL difficulty if available)
            if 'team_h_difficulty' in fixture and 'team_a_difficulty' in fixture:
                if is_home:
                    opponent_difficulty = fixture['team_h_difficulty']
                else:
                    opponent_difficulty = fixture['team_a_difficulty']
            else:
                # Default difficulty based on team position (placeholder)
                opponent_difficulty = np.random.randint(1, 6)  # 1-5 scale
            
            # Travel consideration (home vs away)
            travel_factor = 0 if is_home else 1
            
            # Create feature row
            feature_row = {
                'team_id': team_id,
                'team_name': team_name,
                'fixture_id': fixture['fixture_id'],
                'gameweek': fixture['gameweek'],
                'fixture_sequence': i + 1,  # 1st, 2nd, 3rd upcoming fixture
                'is_home': is_home,
                'opponent_id': opponent_id,
                'opponent_name': opponent_name,
                'rest_days': rest_days,
                'fixture_congestion': games_this_week,
                'opponent_difficulty': opponent_difficulty,
                'travel_factor': travel_factor,
                'kickoff_time': fixture['kickoff_time']
            }
            
            schedule_features.append(feature_row)
    
    # Convert to DataFrame
    schedule_df = pd.DataFrame(schedule_features)
    
    # Add derived features
    schedule_df['rest_days_category'] = pd.cut(
        schedule_df['rest_days'], 
        bins=[0, 2, 4, 7, float('inf')], 
        labels=['Very Short', 'Short', 'Standard', 'Long']
    )
    
    schedule_df['congestion_level'] = pd.cut(
        schedule_df['fixture_congestion'],
        bins=[0, 1, 2, float('inf')],
        labels=['Normal', 'Busy', 'Very Busy']
    )
    
    # Calculate team-level summary features
    team_summary = schedule_df.groupby('team_id').agg({
        'rest_days': ['mean', 'min', 'max'],
        'fixture_congestion': 'mean',
        'opponent_difficulty': 'mean',
        'travel_factor': 'sum'
    }).round(2)
    
    team_summary.columns = ['avg_rest_days', 'min_rest_days', 'max_rest_days', 
                           'avg_congestion', 'avg_opponent_difficulty', 'away_games']
    team_summary = team_summary.reset_index()
    
    # Merge team summary with team names
    team_summary = team_summary.merge(teams[['team_id', 'team_name']], on='team_id')
    
    # Save detailed features
    output_path = Path(output_dir) / "schedule_strength.parquet"
    table = pa.Table.from_pandas(schedule_df)
    pq.write_table(table, output_path)
    
    # Save team summary
    summary_path = Path(output_dir) / "schedule_strength_summary.parquet"
    summary_table = pa.Table.from_pandas(team_summary)
    pq.write_table(summary_table, summary_path)
    
    logger.info(f"Saved schedule strength features: {len(schedule_df)} fixture-level records")
    logger.info(f"Saved team summary: {len(team_summary)} teams")
    
    return schedule_df


def project_points_next_gw(
    players_fp: str = "data/FPL/processed/players_processed.parquet",
    fixtures_fp: str = "data/FPL/processed/fixtures_processed.parquet",
    teams_fp: str = "data/FPL/processed/teams_processed.parquet",
    output_dir: str = "data/FPL/features"
) -> pd.DataFrame:
    """
    Project expected points for each player for the next gameweek.
    
    Uses:
    - Recent per-90 performance metrics
    - Opponent difficulty
    - Player form and fitness
    - Position-specific factors
    
    Args:
        players_fp: Path to processed players file
        fixtures_fp: Path to processed fixtures file
        teams_fp: Path to processed teams file
        output_dir: Directory to save output features
        
    Returns:
        DataFrame with player point projections
    """
    logger.info("Building player point projections for next gameweek...")
    
    # Load processed data
    players = pd.read_parquet(players_fp)
    fixtures = pd.read_parquet(fixtures_fp)
    teams = pd.read_parquet(teams_fp)
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Assume next gameweek (in production, this would be dynamic)
    next_gw = 1  # Start of season
    
    # Get fixtures for next gameweek
    next_gw_fixtures = fixtures[fixtures['gameweek'] == next_gw].copy()
    
    if len(next_gw_fixtures) == 0:
        logger.warning(f"No fixtures found for gameweek {next_gw}")
        return pd.DataFrame()
    
    # Initialize projections storage
    projections = []
    
    for _, player in players.iterrows():
        player_id = player['player_id']
        player_name = player['full_name']
        team_id = player['team_id']
        position = player['position']
        
        # Find player's next fixture
        player_fixture = next_gw_fixtures[
            (next_gw_fixtures['home_team_id'] == team_id) | 
            (next_gw_fixtures['away_team_id'] == team_id)
        ]
        
        if len(player_fixture) == 0:
            logger.warning(f"No fixture found for {player_name} in GW{next_gw}")
            continue
        
        fixture = player_fixture.iloc[0]
        is_home = fixture['home_team_id'] == team_id
        opponent_id = fixture['away_team_id'] if is_home else fixture['home_team_id']
        
        # Get opponent difficulty
        if 'team_h_difficulty' in fixture and 'team_a_difficulty' in fixture:
            if is_home:
                opponent_difficulty = fixture['team_h_difficulty']
            else:
                opponent_difficulty = fixture['team_a_difficulty']
        else:
            # Default difficulty
            opponent_difficulty = 3
        
        # Base projection factors
        base_points = _get_position_base_points(position)
        form_multiplier = _get_form_multiplier(player.get('form', 0))
        difficulty_multiplier = _get_difficulty_multiplier(opponent_difficulty)
        home_away_bonus = 1.1 if is_home else 0.9
        
        # Calculate projected points
        projected_points = (
            base_points * 
            form_multiplier * 
            difficulty_multiplier * 
            home_away_bonus
        )
        
        # Add some randomness for realistic projections
        projected_points += np.random.normal(0, 0.5)
        projected_points = max(0, projected_points)  # Points can't be negative
        
        # Calculate confidence in projection
        confidence = _calculate_projection_confidence(player, position)
        
        # Create projection row
        projection_row = {
            'player_id': player_id,
            'player_name': player_name,
            'team_id': team_id,
            'position': position,
            'gameweek': next_gw,
            'opponent_id': opponent_id,
            'is_home': is_home,
            'opponent_difficulty': opponent_difficulty,
            'base_points': base_points,
            'form_multiplier': form_multiplier,
            'difficulty_multiplier': difficulty_multiplier,
            'home_away_bonus': home_away_bonus,
            'projected_points': round(projected_points, 2),
            'projection_confidence': confidence,
            'fixture_id': fixture['fixture_id']
        }
        
        projections.append(projection_row)
    
    # Convert to DataFrame
    projections_df = pd.DataFrame(projections)
    
    # Add derived features
    projections_df['projection_category'] = pd.cut(
        projections_df['projected_points'],
        bins=[0, 2, 4, 6, 8, float('inf')],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Calculate team-level projection summaries
    team_projections = projections_df.groupby('team_id').agg({
        'projected_points': ['mean', 'sum', 'count'],
        'projection_confidence': 'mean'
    }).round(2)
    
    team_projections.columns = ['avg_projected_points', 'total_projected_points', 
                               'players_projected', 'avg_confidence']
    team_projections = team_projections.reset_index()
    
    # Merge with team names
    team_projections = team_projections.merge(teams[['team_id', 'team_name']], on='team_id')
    
    # Save projections
    output_path = Path(output_dir) / "projections.parquet"
    table = pa.Table.from_pandas(projections_df)
    pq.write_table(table, output_path)
    
    # Save team summary
    summary_path = Path(output_dir) / "projections_summary.parquet"
    summary_table = pa.Table.from_pandas(team_projections)
    pq.write_table(summary_table, summary_path)
    
    logger.info(f"Saved player projections: {len(projections_df)} players")
    logger.info(f"Saved team projection summary: {len(team_projections)} teams")
    
    return projections_df


def _get_position_base_points(position: str) -> float:
    """Get base points expectation for a position."""
    base_points = {
        'GK': 3.5,
        'DEF': 4.0,
        'MID': 4.5,
        'FWD': 5.0
    }
    return base_points.get(position, 4.0)


def _get_form_multiplier(form: float) -> float:
    """Convert FPL form rating to multiplier."""
    if pd.isna(form) or form == 0:
        return 1.0
    
    # FPL form is typically 0-10, normalize to 0.5-1.5 range
    normalized_form = (form - 5) / 10  # -0.5 to 0.5
    return 1.0 + normalized_form


def _get_difficulty_multiplier(difficulty: int) -> float:
    """Convert FPL difficulty rating to multiplier."""
    # FPL difficulty: 1=easy, 5=very hard
    difficulty_multipliers = {
        1: 1.3,  # Easy opponent = bonus
        2: 1.1,
        3: 1.0,  # Neutral
        4: 0.9,
        5: 0.7   # Hard opponent = penalty
    }
    return difficulty_multipliers.get(difficulty, 1.0)


def _calculate_projection_confidence(player: pd.Series, position: str) -> float:
    """Calculate confidence level in the projection (0-1)."""
    confidence = 0.5  # Base confidence
    
    # Higher confidence if player has recent form data
    if not pd.isna(player.get('form', np.nan)):
        confidence += 0.2
    
    # Higher confidence if player has consistent minutes
    if not pd.isna(player.get('minutes', np.nan)) and player['minutes'] > 0:
        confidence += 0.1
    
    # Position-specific confidence adjustments
    if position == 'GK':
        confidence += 0.1  # Goalkeeper performance more predictable
    elif position == 'FWD':
        confidence -= 0.1  # Forward performance more volatile
    
    return min(1.0, max(0.0, confidence))


def build_all_features(
    fixtures_fp: str = "data/FPL/processed/fixtures_processed.parquet",
    teams_fp: str = "data/FPL/processed/teams_processed.parquet",
    players_fp: str = "data/FPL/processed/players_processed.parquet",
    output_dir: str = "data/FPL/features"
) -> Dict[str, pd.DataFrame]:
    """
    Build all FPL features in one call.
    
    Args:
        fixtures_fp: Path to processed fixtures file
        teams_fp: Path to processed teams file
        players_fp: Path to processed players file
        output_dir: Directory to save output features
        
    Returns:
        Dictionary containing all feature DataFrames
    """
    logger.info("Building all FPL features...")
    
    # Build schedule strength features
    schedule_features = build_team_schedule_strength(
        fixtures_fp=fixtures_fp,
        teams_fp=teams_fp,
        output_dir=output_dir
    )
    
    # Build player projections
    player_projections = project_points_next_gw(
        players_fp=players_fp,
        fixtures_fp=fixtures_fp,
        teams_fp=teams_fp,
        output_dir=output_dir
    )
    
    # Log summary statistics
    logger.info("=" * 60)
    logger.info("FEATURE GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Schedule Strength Features: {len(schedule_features)} records")
    logger.info(f"Player Projections: {len(player_projections)} players")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("=" * 60)
    
    return {
        'schedule_strength': schedule_features,
        'player_projections': player_projections
    }


if __name__ == "__main__":
    # Run feature generation when script is executed directly
    build_all_features() 