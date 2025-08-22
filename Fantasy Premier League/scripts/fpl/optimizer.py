#!/usr/bin/env python3
"""
FPL Squad Optimizer using Integer Linear Programming (ILP)

This script optimizes Fantasy Premier League squads using PuLP to solve
the ILP problem with constraints for squad size, budget, positions, and club caps.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pulp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BUDGET_LIMIT = 1000.0  # Adjusted based on actual price data scale
SQUAD_SIZE = 15
POSITION_QUOTAS = {
    'GK': 2,
    'DEF': 5,
    'MID': 5,
    'FWD': 3
}
DEFAULT_CLUB_CAP = 3


def load_fpl_data(data_path: str) -> pd.DataFrame:
    """Load FPL features data from parquet file."""
    try:
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} player records from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        sys.exit(1)


def load_team_names(teams_path: str) -> Dict[int, str]:
    """Load team names mapping from teams processed data."""
    try:
        teams_df = pd.read_parquet(teams_path)
        team_mapping = dict(zip(teams_df['team_id'], teams_df['team_name']))
        logger.info(f"Loaded {len(team_mapping)} team names")
        return team_mapping
    except Exception as e:
        logger.warning(f"Failed to load team names: {e}")
        return {}


def load_players_ref(players_path: str) -> pd.DataFrame:
    """Load players reference data and derive canonical player names."""
    try:
        players_df = pd.read_parquet(players_path)
        
        # Derive canonical player_name
        if 'player_name' in players_df.columns:
            players_df['canonical_name'] = players_df['player_name']
        elif 'full_name' in players_df.columns:
            players_df['canonical_name'] = players_df['full_name']
        elif 'first_name' in players_df.columns and 'last_name' in players_df.columns:
            players_df['canonical_name'] = players_df['first_name'] + ' ' + players_df['last_name']
        else:
            # Fallback to display_name if available
            players_df['canonical_name'] = players_df.get('display_name', 'Unknown Player')
        
        # Keep only required columns
        result_df = players_df[['player_id', 'canonical_name', 'team_id']].copy()
        result_df = result_df.rename(columns={'canonical_name': 'player_name'})
        
        logger.info(f"Loaded {len(result_df)} player references from {players_path}")
        return result_df
    except Exception as e:
        logger.error(f"Failed to load players reference from {players_path}: {e}")
        sys.exit(1)


def filter_data_by_gameweek(df: pd.DataFrame, target_gw: int) -> pd.DataFrame:
    """Filter data to target gameweek."""
    available_gws = sorted(df['gameweek'].unique())
    
    if target_gw not in available_gws:
        logger.error(f"Gameweek {target_gw} not available. Available gameweeks: {available_gws}")
        sys.exit(1)
    
    filtered_df = df[df['gameweek'] == target_gw].copy()
    logger.info(f"Filtered to {len(filtered_df)} players for gameweek {target_gw}")
    return filtered_df


def apply_price_filters(df: pd.DataFrame, min_price: Optional[float], max_price: Optional[float]) -> pd.DataFrame:
    """Apply optional price filters."""
    if min_price is not None:
        df = df[df['price'] >= min_price]
        logger.info(f"Applied min price filter: £{min_price}m, {len(df)} players remaining")
    
    if max_price is not None:
        df = df[df['price'] <= max_price]
        logger.info(f"Applied max price filter: £{max_price}m, {len(df)} players remaining")
    
    return df


def create_optimization_model(df: pd.DataFrame, club_cap: int) -> Tuple[pulp.LpProblem, Dict[int, pulp.LpVariable]]:
    """Create the ILP optimization model."""
    # Create the optimization problem
    prob = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)
    
    # Create binary decision variables for each player
    player_vars = {}
    for _, player in df.iterrows():
        player_vars[player['player_id']] = pulp.LpVariable(
            f"player_{player['player_id']}", 
            cat=pulp.LpBinary
        )
    
    # Objective: maximize total projected points
    prob += pulp.lpSum(
        player_vars[player['player_id']] * player['xP'] 
        for _, player in df.iterrows()
    )
    
    # Constraint 1: Squad size = 15
    prob += pulp.lpSum(player_vars.values()) == SQUAD_SIZE
    
    # Constraint 2: Budget constraint
    prob += pulp.lpSum(
        player_vars[player['player_id']] * player['price'] 
        for _, player in df.iterrows()
    ) <= BUDGET_LIMIT
    
    # Constraint 3: Position quotas
    for position, quota in POSITION_QUOTAS.items():
        prob += pulp.lpSum(
            player_vars[player['player_id']] 
            for _, player in df.iterrows() 
            if player['position'] == position
        ) == quota
    
    # Constraint 4: Club cap (max 3 players per club)
    for team_id in df['team_id'].unique():
        prob += pulp.lpSum(
            player_vars[player['player_id']] 
            for _, player in df.iterrows() 
            if player['team_id'] == team_id
        ) <= club_cap
    
    return prob, player_vars


def solve_optimization(prob: pulp.LpProblem) -> bool:
    """Solve the optimization problem."""
    logger.info("Solving optimization problem...")
    
    # Use CBC solver (default with PuLP)
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    
    if prob.status == pulp.LpStatusOptimal:
        logger.info("Optimization completed successfully!")
        return True
    elif prob.status == pulp.LpStatusInfeasible:
        logger.error("Problem is infeasible - no solution exists with current constraints")
        return False
    elif prob.status == pulp.LpStatusUnbounded:
        logger.error("Problem is unbounded")
        return False
    else:
        logger.error(f"Optimization failed with status: {prob.status}")
        return False


def extract_solution(df: pd.DataFrame, player_vars: Dict[int, pulp.LpVariable]) -> pd.DataFrame:
    """Extract the selected players from the solution."""
    selected_players = []
    
    for _, player in df.iterrows():
        player_id = player['player_id']
        if player_vars[player_id].value() == 1:
            selected_players.append(player)
    
    solution_df = pd.DataFrame(selected_players)
    solution_df = solution_df.sort_values('xP', ascending=False)
    
    return solution_df


def create_output_files(solution_df: pd.DataFrame, team_names: Dict[int, str], output_dir: str):
    """Create the three output CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Prepare data with team names
    output_data = solution_df.copy()
    output_data['team_name'] = output_data['team_id'].map(team_names).fillna(f"Team_{output_data['team_id']}")
    
    # Select columns for output - ensure player_name exists
    squad_columns = ['player_id', 'player_name', 'team_name', 'position', 'price', 'xP']
    
    # Validate that player_name column exists and has no missing values
    if 'player_name' not in output_data.columns:
        logger.error("player_name column not found in solution data")
        sys.exit(1)
    
    missing_names = output_data['player_name'].isna().sum()
    if missing_names > 0:
        logger.error(f"Found {missing_names} rows with missing player names")
        sys.exit(1)
    
    # Squad CSV (all 15 players)
    squad_df = output_data[squad_columns].copy()
    squad_path = output_path / 'fpl_squad.csv'
    squad_df.to_csv(squad_path, index=False)
    logger.info(f"Squad saved to {squad_path}")
    
    # Starting XI CSV (top 11 by projected points)
    starting_xi_df = output_data.head(11)[squad_columns].copy()
    starting_xi_path = output_path / 'fpl_starting_xi.csv'
    starting_xi_df.to_csv(starting_xi_path, index=False)
    logger.info(f"Starting XI saved to {starting_xi_path}")
    
    # Bench CSV (remaining 4 players ordered by projected points descending)
    bench_df = output_data.tail(4)[squad_columns].copy()
    bench_path = output_path / 'fpl_bench.csv'
    bench_df.to_csv(bench_path, index=False)
    logger.info(f"Bench saved to {bench_path}")


def print_summary(solution_df: pd.DataFrame, team_names: Dict[int, str]):
    """Print optimization summary to console."""
    total_cost = solution_df['price'].sum()
    total_points = solution_df['xP'].sum()
    
    print("\n" + "="*60)
    print("FPL SQUAD OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Total Squad Cost: £{total_cost:.1f}m")
    print(f"Total Projected Points: {total_points:.2f}")
    print(f"Budget Remaining: £{BUDGET_LIMIT - total_cost:.1f}m")
    
    # Position breakdown
    print("\nPosition Breakdown:")
    for position, quota in POSITION_QUOTAS.items():
        count = len(solution_df[solution_df['position'] == position])
        print(f"  {position}: {count}/{quota}")
    
    # Club breakdown
    print("\nClub Breakdown:")
    club_counts = solution_df['team_id'].value_counts()
    for team_id, count in club_counts.items():
        team_name = team_names.get(team_id, f"Team_{team_id}")
        status = " (AT CAP)" if count == 3 else ""
        print(f"  {team_name}: {count} players{status}")
    
    # Top performers
    print("\nTop 5 Projected Point Scorers:")
    top_5 = solution_df.head(5)
    for _, player in top_5.iterrows():
        team_name = team_names.get(player['team_id'], f"Team_{player['team_id']}")
        player_name = player.get('player_name', f"Player_{player['player_id']}")
        print(f"  {player['position']} - {player_name} ({team_name}) - £{player['price']}m - {player['xP']:.2f} pts")
    
    print("="*60)


def validate_solution(solution_df: pd.DataFrame, club_cap: int) -> bool:
    """Validate that the solution meets all constraints."""
    # Check squad size
    if len(solution_df) != SQUAD_SIZE:
        logger.error(f"Squad size mismatch: {len(solution_df)} != {SQUAD_SIZE}")
        return False
    
    # Check budget
    total_cost = solution_df['price'].sum()
    if total_cost > BUDGET_LIMIT:
        logger.error(f"Budget exceeded: £{total_cost:.1f}m > £{BUDGET_LIMIT}m")
        return False
    
    # Check position quotas
    for position, quota in POSITION_QUOTAS.items():
        count = len(solution_df[solution_df['position'] == position])
        if count != quota:
            logger.error(f"Position quota mismatch for {position}: {count} != {quota}")
            return False
    
    # Check club cap
    club_counts = solution_df['team_id'].value_counts()
    if (club_counts > club_cap).any():
        logger.error(f"Club cap exceeded: {club_counts[club_counts > club_cap]}")
        return False
    
    logger.info("Solution validation passed")
    return True


def main():
    """Main function to run the FPL squad optimizer."""
    parser = argparse.ArgumentParser(description='FPL Squad Optimizer using ILP')
    parser.add_argument('--gw', type=int, help='Target gameweek (default: latest available)')
    parser.add_argument('--min-price', type=float, help='Minimum player price filter')
    parser.add_argument('--max-price', type=float, help='Maximum player price filter')
    parser.add_argument('--allow-club-cap', type=int, default=DEFAULT_CLUB_CAP, 
                       help=f'Maximum players per club (default: {DEFAULT_CLUB_CAP})')
    parser.add_argument('--data-path', default='data/FPL/processed/fpl_features_model.parquet',
                       help='Path to FPL features data')
    parser.add_argument('--teams-path', default='data/FPL/processed/teams_processed.parquet',
                       help='Path to teams data')
    parser.add_argument('--players-path', default='data/FPL/processed/players_processed.parquet',
                       help='Path to players data')
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    # Load data
    df = load_fpl_data(args.data_path)
    team_names = load_team_names(args.teams_path)
    players_ref = load_players_ref(args.players_path)
    
    # Determine target gameweek
    if args.gw is None:
        target_gw = max(df['gameweek'].unique())
        logger.info(f"No gameweek specified, using latest available: {target_gw}")
    else:
        target_gw = args.gw
    
    # Filter data
    df = filter_data_by_gameweek(df, target_gw)
    df = apply_price_filters(df, args.min_price, args.max_price)
    
    # Merge player names
    df = df.merge(players_ref[['player_id', 'player_name']], on='player_id', how='left')
    
    # Check for missing player names
    missing_names = df['player_name'].isna().sum()
    total_rows = len(df)
    missing_pct = (missing_names / total_rows) * 100 if total_rows > 0 else 0
    
    if missing_pct > 1.0:
        logger.warning(f"Missing player names: {missing_names}/{total_rows} ({missing_pct:.1f}%)")
        if missing_names > 0:
            missing_ids = df[df['player_name'].isna()]['player_id'].head(3).tolist()
            logger.warning(f"Example missing IDs: {missing_ids}")
            logger.warning("Consider running 'make fpl-bootstrap' to refresh players, then 'make fpl-featurize' and re-run optimization")
    else:
        logger.info(f"Player names merged successfully: {missing_names}/{total_rows} missing ({missing_pct:.1f}%)")
    
    # Check if we have enough players after filtering
    if len(df) < SQUAD_SIZE:
        logger.error(f"Not enough players available after filtering: {len(df)} < {SQUAD_SIZE}")
        sys.exit(1)
    
    # Create and solve optimization model
    prob, player_vars = create_optimization_model(df, args.allow_club_cap)
    
    if not solve_optimization(prob):
        logger.error("Optimization failed")
        sys.exit(1)
    
    # Extract solution
    solution_df = extract_solution(df, player_vars)
    
    # Validate solution
    if not validate_solution(solution_df, args.allow_club_cap):
        logger.error("Solution validation failed")
        sys.exit(1)
    
    # Create output files
    create_output_files(solution_df, team_names, args.output_dir)
    
    # Print summary
    print_summary(solution_df, team_names)


if __name__ == "__main__":
    main() 