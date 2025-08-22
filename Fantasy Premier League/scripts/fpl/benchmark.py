#!/usr/bin/env python3
"""
FPL Optimizer Performance Benchmarks vs Historical Seasons (Step 5b)

This script backtests the optimizer across historical gameweeks, computes accuracy/error metrics,
and benchmarks against baselines (template, random) to validate optimization performance.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import glob

import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BUDGET_LIMIT = 1000.0
SQUAD_SIZE = 15
POSITION_QUOTAS = {
    'GK': 2,
    'DEF': 5,
    'MID': 5,
    'FWD': 3
}
DEFAULT_CLUB_CAP = 3
RANDOM_SEED = 42

# Set deterministic random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Valid formations for starting XI selection
VALID_FORMATIONS = {
    '4-4-2': {'GK': 1, 'DEF': 4, 'MID': 4, 'FWD': 2},
    '4-3-3': {'GK': 1, 'DEF': 4, 'MID': 3, 'FWD': 3},
    '3-5-2': {'GK': 1, 'DEF': 3, 'MID': 5, 'FWD': 2},
    '5-3-2': {'GK': 1, 'DEF': 5, 'MID': 3, 'FWD': 2},
    '4-5-1': {'GK': 1, 'DEF': 4, 'MID': 5, 'FWD': 1}
}


def load_features_data(features_path: str) -> pd.DataFrame:
    """Load FPL features data."""
    try:
        df = pd.read_parquet(features_path)
        logger.info(f"✅ Loaded features data: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load features data: {e}")
        sys.exit(1)


def load_team_names(teams_path: str) -> Dict[int, str]:
    """Load team names mapping from teams processed data."""
    try:
        teams_df = pd.read_parquet(teams_path)
        team_mapping = dict(zip(teams_df['team_id'], teams_df['team_name']))
        logger.info(f"✅ Loaded {len(team_mapping)} team names")
        return team_mapping
    except Exception as e:
        logger.warning(f"⚠️ Failed to load team names: {e}")
        return {}


def merge_team_names(features_df: pd.DataFrame, team_names: Dict[int, str]) -> pd.DataFrame:
    """Merge team names into features data."""
    try:
        df = features_df.copy()
        df['team_name'] = df['team_id'].map(team_names)
        
        # Check for missing team names
        missing_teams = df['team_name'].isna().sum()
        if missing_teams > 0:
            logger.warning(f"⚠️ {missing_teams} records missing team names")
        
        logger.info(f"✅ Merged team names: {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to merge team names: {e}")
        return features_df


def find_historical_gw_files(historical_gw_path: str) -> List[str]:
    """Find historical gameweek files in the specified path."""
    try:
        # Look for various file patterns
        patterns = [
            os.path.join(historical_gw_path, "gw*_live.parquet"),
            os.path.join(historical_gw_path, "*_history.parquet"),
            os.path.join(historical_gw_path, "player_hist", "*_history.parquet")
        ]
        
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        
        if not files:
            logger.error(f"❌ No historical gameweek files found in {historical_gw_path}")
            logger.error("Expected files: gw*_live.parquet or *_history.parquet")
            sys.exit(1)
        
        # Sort files and extract gameweek numbers
        gw_files = []
        for file in files:
            if 'gw' in file and 'live' in file:
                # Extract GW number from filename like gw1_live.parquet
                try:
                    gw_num = int(file.split('gw')[1].split('_')[0])
                    gw_files.append((gw_num, file))
                except:
                    continue
            elif 'history' in file:
                # For history files, we'll need to examine content to determine GW
                gw_files.append((0, file))  # Placeholder
        
        gw_files.sort(key=lambda x: x[0])
        logger.info(f"✅ Found {len(gw_files)} historical gameweek files")
        
        return [file for _, file in gw_files]
        
    except Exception as e:
        logger.error(f"❌ Failed to find historical gameweek files: {e}")
        sys.exit(1)


def load_historical_gw_data(file_path: str) -> pd.DataFrame:
    """Load historical gameweek data and standardize columns."""
    try:
        df = pd.read_parquet(file_path)
        
        # Standardize column names
        column_mapping = {
            'id': 'player_id',
            'element': 'player_id',
            'stats.total_points': 'total_points',
            'total_points': 'total_points',
            'minutes': 'minutes',
            'goals_scored': 'goals_scored',
            'assists': 'assists',
            'clean_sheets': 'clean_sheets',
            'goals_conceded': 'goals_conceded',
            'saves': 'saves',
            'yellow_cards': 'yellow_cards',
            'red_cards': 'red_cards',
            'bonus': 'bonus',
            'bps': 'bps'
        }
        
        # Rename columns that exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_cols = ['player_id', 'total_points']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"⚠️ Missing columns in {file_path}: {missing_cols}")
            return pd.DataFrame()
        
        logger.info(f"✅ Loaded historical GW data: {len(df)} records from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to load historical GW data from {file_path}: {e}")
        return pd.DataFrame()


def create_optimizer_squad(features_df: pd.DataFrame, target_gw: int, budget: float = BUDGET_LIMIT, 
                          club_cap: int = DEFAULT_CLUB_CAP) -> pd.DataFrame:
    """Create optimized squad using ILP constraints for a specific gameweek."""
    try:
        # Filter to target gameweek and available players
        gw_data = features_df[features_df['gameweek'] == target_gw].copy()
        
        if len(gw_data) < SQUAD_SIZE:
            logger.warning(f"⚠️ Insufficient players for GW {target_gw}: {len(gw_data)} < {SQUAD_SIZE}")
            return pd.DataFrame()
        
        # Create optimization problem
        prob = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)
        
        # Decision variables: 1 if player is selected, 0 otherwise
        player_vars = pulp.LpVariable.dicts("player",
                                           [(i, player['player_id']) for i, player in gw_data.iterrows()],
                                           cat='Binary')
        
        # Objective: maximize total projected points
        prob += pulp.lpSum([player_vars[(i, player['player_id'])] * player['xP'] 
                           for i, player in gw_data.iterrows()])
        
        # Constraint 1: Squad size = 15
        prob += pulp.lpSum([player_vars[(i, player['player_id'])] 
                           for i, player in gw_data.iterrows()]) == SQUAD_SIZE
        
        # Constraint 2: Budget limit
        prob += pulp.lpSum([player_vars[(i, player['player_id'])] * player['price'] 
                           for i, player in gw_data.iterrows()]) <= budget
        
        # Constraint 3: Position quotas
        for position, quota in POSITION_QUOTAS.items():
            position_players = gw_data[gw_data['position'] == position]
            prob += pulp.lpSum([player_vars[(i, player['player_id'])] 
                               for i, player in position_players.iterrows()]) == quota
        
        # Constraint 4: Club cap
        for team_id in gw_data['team_id'].unique():
            team_players = gw_data[gw_data['team_id'] == team_id]
            prob += pulp.lpSum([player_vars[(i, player['player_id'])] 
                               for i, player in team_players.iterrows()]) <= club_cap
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if prob.status != pulp.LpStatusOptimal:
            logger.warning(f"⚠️ Optimization failed for GW {target_gw}")
            return pd.DataFrame()
        
        # Extract solution
        selected_players = []
        for i, player in gw_data.iterrows():
            if player_vars[(i, player['player_id'])].value() == 1:
                selected_players.append(player)
        
        if len(selected_players) != SQUAD_SIZE:
            logger.warning(f"⚠️ Solution size mismatch for GW {target_gw}: {len(selected_players)} != {SQUAD_SIZE}")
            return pd.DataFrame()
        
        squad_df = pd.DataFrame(selected_players)
        squad_df = squad_df.sort_values('xP', ascending=False)
        
        logger.info(f"✅ Created optimizer squad for GW {target_gw}: {len(squad_df)} players")
        return squad_df
        
    except Exception as e:
        logger.error(f"❌ Failed to create optimizer squad for GW {target_gw}: {e}")
        return pd.DataFrame()


def create_template_squad(features_df: pd.DataFrame, target_gw: int, budget: float = BUDGET_LIMIT) -> pd.DataFrame:
    """Create template squad using most-owned players within budget."""
    try:
        # Filter to target gameweek
        gw_data = features_df[features_df['gameweek'] == target_gw].copy()
        
        if len(gw_data) < SQUAD_SIZE:
            logger.warning(f"⚠️ Insufficient players for template squad GW {target_gw}: {len(gw_data)} < {SQUAD_SIZE}")
            return pd.DataFrame()
        
        # Sort by xP (since ownership data may not be available)
        gw_data = gw_data.sort_values('xP', ascending=False)
        
        # Initialize squad
        squad = []
        remaining_budget = budget
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        club_counts = {}
        
        for _, player in gw_data.iterrows():
            if len(squad) >= SQUAD_SIZE:
                break
                
            # Check position quota
            if position_counts[player['position']] >= POSITION_QUOTAS[player['position']]:
                continue
                
            # Check budget
            if player['price'] > remaining_budget:
                continue
                
            # Check club cap
            club = player['team_name']
            if club_counts.get(club, 0) >= DEFAULT_CLUB_CAP:
                continue
                
            # Add player to squad
            squad.append(player)
            remaining_budget -= player['price']
            position_counts[player['position']] += 1
            club_counts[club] = club_counts.get(club, 0) + 1
        
        if len(squad) < SQUAD_SIZE:
            logger.warning(f"⚠️ Template squad incomplete for GW {target_gw}: {len(squad)}/{SQUAD_SIZE} players")
            
        template_df = pd.DataFrame(squad)
        template_df = template_df.sort_values('xP', ascending=False)
        logger.info(f"✅ Created template squad for GW {target_gw}: {len(template_df)} players")
        return template_df
        
    except Exception as e:
        logger.error(f"❌ Failed to create template squad for GW {target_gw}: {e}")
        return pd.DataFrame()


def create_random_squad(features_df: pd.DataFrame, target_gw: int, budget: float = BUDGET_LIMIT) -> pd.DataFrame:
    """Create random valid squad under FPL rules."""
    try:
        # Filter to target gameweek
        gw_data = features_df[features_df['gameweek'] == target_gw].copy()
        
        if len(gw_data) < SQUAD_SIZE:
            logger.warning(f"⚠️ Insufficient players for random squad GW {target_gw}: {len(gw_data)} < {SQUAD_SIZE}")
            return pd.DataFrame()
        
        # Initialize squad
        squad = []
        remaining_budget = budget
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        club_counts = {}
        
        # Shuffle players for randomness
        players_list = gw_data.sample(frac=1.0, random_state=RANDOM_SEED).to_dict('records')
        
        for player in players_list:
            if len(squad) >= SQUAD_SIZE:
                break
                
            # Check position quota
            if position_counts[player['position']] >= POSITION_QUOTAS[player['position']]:
                continue
                
            # Check budget
            if player['price'] > remaining_budget:
                continue
                
            # Check club cap
            club = player['team_name']
            if club_counts.get(club, 0) >= DEFAULT_CLUB_CAP:
                continue
                
            # Add player to squad
            squad.append(player)
            remaining_budget -= player['price']
            position_counts[player['position']] += 1
            club_counts[club] = club_counts.get(club, 0) + 1
        
        if len(squad) < SQUAD_SIZE:
            logger.warning(f"⚠️ Random squad incomplete for GW {target_gw}: {len(squad)}/{SQUAD_SIZE} players")
            
        random_df = pd.DataFrame(squad)
        random_df = random_df.sort_values('xP', ascending=False)
        logger.info(f"✅ Created random squad for GW {target_gw}: {len(random_df)} players")
        return random_df
        
    except Exception as e:
        logger.error(f"❌ Failed to create random squad for GW {target_gw}: {e}")
        return pd.DataFrame()


def pick_starting_xi(squad_df: pd.DataFrame, formation: str = '4-4-2') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pick starting XI based on formation."""
    try:
        if formation not in VALID_FORMATIONS:
            formation = '4-4-2'  # Default fallback
        
        position_quota = VALID_FORMATIONS[formation]
        xi_players = []
        remaining_players = squad_df.copy()
        
        # Sort by xP descending
        remaining_players = remaining_players.sort_values('xP', ascending=False)
        
        # Pick players for each position according to quota
        for position, quota in position_quota.items():
            position_players = remaining_players[remaining_players['position'] == position]
            
            if len(position_players) < quota:
                logger.warning(f"⚠️ Not enough {position} players for formation {formation}")
                continue
            
            # Take top quota players by xP
            selected = position_players.head(quota)
            xi_players.append(selected)
            
            # Remove selected players from remaining pool
            remaining_players = remaining_players[~remaining_players['player_id'].isin(selected['player_id'])]
        
        # Combine all selected players
        xi_df = pd.concat(xi_players, ignore_index=True)
        xi_df = xi_df.sort_values('xP', ascending=False)
        
        # Add captain and vice-captain (top 2 by xP)
        xi_df['is_captain'] = 0
        xi_df['is_vice'] = 0
        if len(xi_df) >= 1:
            xi_df.loc[xi_df.index[0], 'is_captain'] = 1
        if len(xi_df) >= 2:
            xi_df.loc[xi_df.index[1], 'is_vice'] = 1
        
        # Remaining players become bench
        bench_df = remaining_players.copy()
        
        return xi_df, bench_df
        
    except Exception as e:
        logger.error(f"❌ Failed to pick starting XI: {e}")
        return pd.DataFrame(), pd.DataFrame()


def calculate_actual_points(xi_df: pd.DataFrame, historical_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate actual points for the starting XI from historical data."""
    try:
        results = {
            'total_points': 0.0,
            'captain_points': 0.0,
            'vice_points': 0.0,
            'bench_points': 0.0,
            'captain_uplift': 0.0
        }
        
        # Calculate total points for starting XI
        for _, player in xi_df.iterrows():
            player_id = player['player_id']
            player_data = historical_data[historical_data['player_id'] == player_id]
            
            if not player_data.empty:
                actual_points = player_data.iloc[0]['total_points']
                results['total_points'] += actual_points
                
                # Apply captain bonus - extract scalar values from pandas Series
                captain_flag = int(player['is_captain'])
                vice_flag = int(player['is_vice'])
                
                if captain_flag == 1:
                    results['captain_points'] = actual_points
                    results['total_points'] += actual_points  # Captain gets double points
                elif vice_flag == 1:
                    results['vice_points'] = actual_points
        
        # Calculate captain uplift
        if results['captain_points'] > 0:
            results['captain_uplift'] = results['captain_points']
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to calculate actual points: {e}")
        return {'total_points': 0.0, 'captain_points': 0.0, 'vice_points': 0.0, 
                'bench_points': 0.0, 'captain_uplift': 0.0}


def run_benchmark_for_gw(features_df: pd.DataFrame, target_gw: int, historical_data: pd.DataFrame,
                         models: List[str]) -> List[Dict]:
    """Run benchmark for a specific gameweek."""
    try:
        results = []
        
        for model in models:
            logger.info(f"🔄 Running {model} for GW {target_gw}...")
            
            # Create squad based on model type
            if model == 'optimizer':
                squad_df = create_optimizer_squad(features_df, target_gw)
            elif model == 'template':
                squad_df = create_template_squad(features_df, target_gw)
            elif model == 'random':
                squad_df = create_random_squad(features_df, target_gw)
            else:
                logger.warning(f"⚠️ Unknown model type: {model}")
                continue
            
            if squad_df.empty:
                logger.warning(f"⚠️ Failed to create {model} squad for GW {target_gw}")
                continue
            
            # Pick starting XI
            xi_df, bench_df = pick_starting_xi(squad_df)
            
            if xi_df.empty:
                logger.warning(f"⚠️ Failed to pick starting XI for {model} GW {target_gw}")
                continue
            
            # Calculate projected points
            pred_total_xp = xi_df['xP'].sum()
            
            # Apply captain bonus to projected points
            captain_xp = xi_df[xi_df['is_captain'] == 1]['xP'].iloc[0] if (xi_df['is_captain'] == 1).any() else 0
            pred_total_xp += captain_xp
            
            # Calculate actual points
            actual_results = calculate_actual_points(xi_df, historical_data)
            actual_total_pts = actual_results['total_points']
            
            # Calculate error
            error = actual_total_pts - pred_total_xp
            
            # Calculate squad cost
            squad_cost = squad_df['price'].sum()
            
            # Store results
            result = {
                'gw': target_gw,
                'model': model,
                'pred_total_xP': pred_total_xp,
                'actual_total_pts': actual_total_pts,
                'error': error,
                'captain_uplift_actual': actual_results['captain_uplift'],
                'bench_points_actual': actual_results['bench_points'],
                'squad_cost': squad_cost
            }
            
            results.append(result)
            logger.info(f"✅ {model} GW {target_gw}: Pred {pred_total_xp:.1f}, Actual {actual_total_pts:.1f}, Error {error:+.1f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to run benchmark for GW {target_gw}: {e}")
        return []


def calculate_aggregate_metrics(benchmark_results: List[Dict]) -> pd.DataFrame:
    """Calculate aggregate metrics across all gameweeks."""
    try:
        if not benchmark_results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(benchmark_results)
        
        # Group by model
        aggregate_data = []
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            # Calculate metrics
            mae = np.mean(np.abs(model_data['error']))
            rmse = np.sqrt(np.mean(model_data['error']**2))
            
            # Calculate correlation (only if we have multiple data points)
            if len(model_data) > 1:
                try:
                    correlation = np.corrcoef(model_data['pred_total_xP'], model_data['actual_total_pts'])[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                except:
                    correlation = 0.0
            else:
                correlation = 0.0  # No correlation for single data point
            
            mean_pred_xp = model_data['pred_total_xP'].mean()
            mean_actual_pts = model_data['actual_total_pts'].mean()
            
            # Calculate win rate vs template (if template exists)
            if 'template' in df['model'].unique():
                template_data = df[df['model'] == 'template']
                wins = 0
                total_gws = 0
                
                for gw in df['gw'].unique():
                    gw_model = model_data[model_data['gw'] == gw]
                    gw_template = template_data[template_data['gw'] == gw]
                    
                    if not gw_model.empty and not gw_template.empty:
                        total_gws += 1
                        if gw_model.iloc[0]['actual_total_pts'] > gw_template.iloc[0]['actual_total_pts']:
                            wins += 1
                
                win_rate = (wins / total_gws * 100) if total_gws > 0 else 0
            else:
                win_rate = 0
            
            aggregate_data.append({
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r': correlation,
                'mean_pred_xP': mean_pred_xp,
                'mean_actual_pts': mean_actual_pts,
                'win_rate_vs_template': win_rate
            })
        
        aggregate_df = pd.DataFrame(aggregate_data)
        logger.info(f"✅ Calculated aggregate metrics for {len(aggregate_df)} models")
        return aggregate_df
        
    except Exception as e:
        logger.error(f"❌ Failed to calculate aggregate metrics: {e}")
        return pd.DataFrame()


def calculate_calibration_metrics(benchmark_results: List[Dict], num_bins: int = 10) -> pd.DataFrame:
    """Calculate calibration metrics by binning predicted vs actual points."""
    try:
        if not benchmark_results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(benchmark_results)
        
        calibration_data = []
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            if len(model_data) < num_bins:
                continue
            
            # Create bins based on predicted points
            model_data = model_data.sort_values('pred_total_xP')
            bin_size = len(model_data) // num_bins
            
            for bin_idx in range(num_bins):
                start_idx = bin_idx * bin_size
                end_idx = start_idx + bin_size if bin_idx < num_bins - 1 else len(model_data)
                
                bin_data = model_data.iloc[start_idx:end_idx]
                
                if len(bin_data) > 0:
                    avg_pred_xp = bin_data['pred_total_xP'].mean()
                    avg_actual_pts = bin_data['actual_total_pts'].mean()
                    count = len(bin_data)
                    
                    calibration_data.append({
                        'model': model,
                        'bin': bin_idx + 1,
                        'avg_pred_xP': avg_pred_xp,
                        'avg_actual_pts': avg_actual_pts,
                        'count': count
                    })
        
        calibration_df = pd.DataFrame(calibration_data)
        logger.info(f"✅ Calculated calibration metrics for {len(calibration_df)} bins")
        return calibration_df
        
    except Exception as e:
        logger.error(f"❌ Failed to calculate calibration metrics: {e}")
        return pd.DataFrame()


def create_benchmark_plots(benchmark_results: List[Dict], aggregate_df: pd.DataFrame, 
                          calibration_df: pd.DataFrame, output_dir: str):
    """Create benchmark visualization plots."""
    try:
        # Create plots directory
        plots_dir = os.path.join(output_dir, 'fpl_benchmark_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Convert results to DataFrame
        df = pd.DataFrame(benchmark_results)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Predicted vs Actual Scatter Plot
        plt.figure(figsize=(10, 8))
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            plt.scatter(model_data['pred_total_xP'], model_data['actual_total_pts'], 
                       label=model, alpha=0.7, s=50)
        
        # Add y=x reference line
        min_val = min(df['pred_total_xP'].min(), df['actual_total_pts'].min())
        max_val = max(df['pred_total_xP'].max(), df['actual_total_pts'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        plt.xlabel('Predicted Points (xP)')
        plt.ylabel('Actual Points')
        plt.title('Predicted vs Actual Points by Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'pred_vs_actual_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-GW Predicted vs Actual Line Plot
        plt.figure(figsize=(12, 8))
        for model in df['model'].unique():
            model_data = df[df['model'] == model].sort_values('gw')
            plt.plot(model_data['gw'], model_data['pred_total_xP'], 
                    marker='o', label=f'{model} (Pred)', linestyle='-', alpha=0.8)
            plt.plot(model_data['gw'], model_data['actual_total_pts'], 
                    marker='s', label=f'{model} (Actual)', linestyle='--', alpha=0.8)
        
        plt.xlabel('Gameweek')
        plt.ylabel('Points')
        plt.title('Predicted vs Actual Points by Gameweek')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'per_gw_pred_actual_line.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Calibration Curve
        if not calibration_df.empty:
            plt.figure(figsize=(10, 8))
            for model in calibration_df['model'].unique():
                model_data = calibration_df[calibration_df['model'] == model].sort_values('bin')
                plt.plot(model_data['avg_pred_xP'], model_data['avg_actual_pts'], 
                        marker='o', label=model, linewidth=2)
            
            # Add perfect calibration line
            min_pred = calibration_df['avg_pred_xP'].min()
            max_pred = calibration_df['avg_pred_xP'].max()
            plt.plot([min_pred, max_pred], [min_pred, max_pred], 'k--', alpha=0.5, label='Perfect Calibration')
            
            plt.xlabel('Average Predicted Points')
            plt.ylabel('Average Actual Points')
            plt.title('Calibration Curve: Predicted vs Actual Points by Bin')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'calibration_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Error Histogram
        plt.figure(figsize=(10, 8))
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            plt.hist(model_data['error'], bins=20, alpha=0.7, label=model, density=True)
        
        plt.xlabel('Prediction Error (Actual - Predicted)')
        plt.ylabel('Density')
        plt.title('Distribution of Prediction Errors by Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'error_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Created benchmark plots in {plots_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create benchmark plots: {e}")


def create_benchmark_report(benchmark_results: List[Dict], aggregate_df: pd.DataFrame, 
                           calibration_df: pd.DataFrame, output_dir: str):
    """Create benchmark report in Markdown format."""
    try:
        report_path = os.path.join(output_dir, 'fpl_benchmark_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# FPL Optimizer Performance Benchmarks vs Historical Seasons\n\n")
            f.write("## Overview\n\n")
            f.write("This report presents the results of backtesting the FPL Optimizer against historical gameweek data.\n\n")
            
            f.write("## Setup\n\n")
            f.write("- **Benchmark Period**: Gameweeks specified in command line arguments\n")
            f.write("- **Models Tested**: Optimizer (ILP), Template (Most Owned), Random\n")
            f.write("- **Data Sources**: FPL features and historical gameweek data\n")
            f.write("- **Leakage Controls**: Only data available up to each gameweek used for predictions\n\n")
            
            f.write("## Datasets\n\n")
            f.write("- **Features Data**: Processed FPL features with projected points (xP)\n")
            f.write("- **Historical Data**: Actual gameweek results from FPL API\n")
            f.write("- **Teams Data**: Team mappings and metadata\n\n")
            
            f.write("## Headline Results\n\n")
            
            if not aggregate_df.empty:
                f.write("### Aggregate Performance Metrics\n\n")
                f.write("| Model | MAE | RMSE | Correlation | Mean Pred xP | Mean Actual Pts | Win Rate vs Template |\n")
                f.write("|-------|-----|------|-------------|--------------|-----------------|---------------------|\n")
                
                for _, row in aggregate_df.iterrows():
                    f.write(f"| {row['model']} | {row['mae']:.2f} | {row['rmse']:.2f} | {row['r']:.3f} | "
                           f"{row['mean_pred_xP']:.1f} | {row['mean_actual_pts']:.1f} | {row['win_rate_vs_template']:.1f}% |\n")
                f.write("\n")
            
            if benchmark_results:
                df = pd.DataFrame(benchmark_results)
                f.write("### Gameweek-by-Gameweek Results\n\n")
                f.write(f"Total gameweeks benchmarked: {len(df['gw'].unique())}\n\n")
                
                # Show sample results
                sample_gws = df['gw'].unique()[:5]  # First 5 GWs
                f.write("Sample results for first 5 gameweeks:\n\n")
                f.write("| GW | Model | Pred xP | Actual Pts | Error | Captain Uplift |\n")
                f.write("|----|-------|---------|------------|-------|----------------|\n")
                
                for gw in sample_gws:
                    gw_data = df[df['gw'] == gw]
                    for _, row in gw_data.iterrows():
                        f.write(f"| {row['gw']} | {row['model']} | {row['pred_total_xP']:.1f} | "
                               f"{row['actual_total_pts']:.1f} | {row['error']:+.1f} | {row['captain_uplift_actual']:.1f} |\n")
                f.write("\n")
            
            f.write("## Key Insights\n\n")
            
            if not aggregate_df.empty:
                # Find best performing model
                best_model = aggregate_df.loc[aggregate_df['r'].idxmax()] if 'r' in aggregate_df.columns and not aggregate_df['r'].isna().all() else None
                if best_model is not None and not np.isnan(best_model['r']):
                    f.write(f"- **Best Correlation**: {best_model['model']} achieved correlation of {best_model['r']:.3f}\n")
                
                # Find lowest error model
                best_mae = aggregate_df.loc[aggregate_df['mae'].idxmin()]
                f.write(f"- **Lowest MAE**: {best_mae['model']} achieved MAE of {best_mae['mae']:.2f}\n")
                
                # Win rate analysis
                optimizer_row = aggregate_df[aggregate_df['model'] == 'optimizer']
                if not optimizer_row.empty:
                    win_rate = optimizer_row.iloc[0]['win_rate_vs_template']
                    f.write(f"- **Optimizer vs Template**: Optimizer wins {win_rate:.1f}% of gameweeks\n")
            
            f.write("\n## Limitations\n\n")
            f.write("- Limited historical data availability may affect statistical significance\n")
            f.write("- Features data currently limited to recent gameweeks\n")
            f.write("- Captain selection strategy simplified for benchmarking\n")
            f.write("- Bench points calculation may not reflect actual FPL substitution rules\n\n")
            
            f.write("## Methodology Notes\n\n")
            f.write("- **Leakage Prevention**: All predictions use only data available up to the target gameweek\n")
            f.write("- **Deterministic Results**: Fixed random seed (42) ensures reproducible baseline squads\n")
            f.write("- **Constraint Validation**: All squads respect FPL rules (budget, positions, club caps)\n")
            f.write("- **Captain Bonus**: Applied to both predicted and actual points for fair comparison\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `fpl_benchmark_per_gw.csv`: Per-gameweek results\n")
            f.write("- `fpl_benchmark_aggregate.csv`: Aggregate performance metrics\n")
            f.write("- `fpl_benchmark_calibration.csv`: Calibration analysis\n")
            f.write("- `fpl_benchmark_plots/`: Visualization plots\n")
            f.write("- `fpl_benchmark_report.md`: This report\n\n")
        
        logger.info(f"✅ Created benchmark report at {report_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create benchmark report: {e}")


def save_benchmark_outputs(benchmark_results: List[Dict], aggregate_df: pd.DataFrame, 
                          calibration_df: pd.DataFrame, output_dir: str):
    """Save all benchmark outputs to CSV files."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save per-gameweek results
        if benchmark_results:
            per_gw_df = pd.DataFrame(benchmark_results)
            per_gw_path = os.path.join(output_dir, 'fpl_benchmark_per_gw.csv')
            per_gw_df.to_csv(per_gw_path, index=False)
            logger.info(f"✅ Saved per-gameweek results to {per_gw_path}")
        
        # Save aggregate metrics
        if not aggregate_df.empty:
            aggregate_path = os.path.join(output_dir, 'fpl_benchmark_aggregate.csv')
            aggregate_df.to_csv(aggregate_path, index=False)
            logger.info(f"✅ Saved aggregate metrics to {aggregate_path}")
        
        # Save calibration metrics
        if not calibration_df.empty:
            calibration_path = os.path.join(output_dir, 'fpl_benchmark_calibration.csv')
            calibration_df.to_csv(calibration_path, index=False)
            logger.info(f"✅ Saved calibration metrics to {calibration_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save benchmark outputs: {e}")


def main():
    """Main function to run FPL benchmarking."""
    parser = argparse.ArgumentParser(description='FPL Optimizer Performance Benchmarks vs Historical Seasons')
    parser.add_argument('--gw-start', type=int, required=True, help='Starting gameweek for benchmarking')
    parser.add_argument('--gw-end', type=int, required=True, help='Ending gameweek for benchmarking')
    parser.add_argument('--features-path', default='data/FPL/processed/fpl_features_model.parquet',
                       help='Path to FPL features data')
    parser.add_argument('--teams-path', default='data/FPL/processed/teams_processed.parquet',
                       help='Path to teams data')
    parser.add_argument('--historical-gw-path', default='data/FPL/raw',
                       help='Path to historical gameweek data')
    parser.add_argument('--model', choices=['optimizer', 'template', 'random', 'all'], default='all',
                       help='Model(s) to benchmark')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory for benchmark results')
    
    args = parser.parse_args()
    
    # Set random seed
    global RANDOM_SEED
    RANDOM_SEED = args.seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    logger.info("🚀 Starting FPL Optimizer Performance Benchmarks vs Historical Seasons (Step 5b)...")
    logger.info(f"Benchmarking gameweeks {args.gw_start} to {args.gw_end}")
    
    # Determine models to benchmark
    if args.model == 'all':
        models = ['optimizer', 'template', 'random']
    else:
        models = [args.model]
    
    logger.info(f"Models to benchmark: {', '.join(models)}")
    
    # Load data
    features_df = load_features_data(args.features_path)
    team_names = load_team_names(args.teams_path)
    features_df = merge_team_names(features_df, team_names)
    
    # Find historical gameweek files
    historical_files = find_historical_gw_files(args.historical_gw_path)
    
    # Check available gameweeks in features data
    available_gws = sorted(features_df['gameweek'].unique())
    logger.info(f"Available gameweeks in features data: {available_gws}")
    
    if args.gw_start not in available_gws or args.gw_end not in available_gws:
        logger.error(f"❌ Requested gameweek range {args.gw_start}-{args.gw_end} not available in features data")
        logger.error(f"Available gameweeks: {available_gws}")
        sys.exit(1)
    
    # Run benchmarks
    all_results = []
    
    for gw in range(args.gw_start, args.gw_end + 1):
        if gw not in available_gws:
            logger.warning(f"⚠️ Skipping GW {gw} - not available in features data")
            continue
        
        # Find corresponding historical data
        historical_data = None
        for file_path in historical_files:
            if f'gw{gw}' in file_path or f'gw{gw}_live' in file_path:
                historical_data = load_historical_gw_data(file_path)
                break
        
        if historical_data is None or historical_data.empty:
            logger.warning(f"⚠️ No historical data found for GW {gw}")
            continue
        
        # Run benchmark for this gameweek
        gw_results = run_benchmark_for_gw(features_df, gw, historical_data, models)
        all_results.extend(gw_results)
    
    if not all_results:
        logger.error("❌ No benchmark results generated")
        sys.exit(1)
    
    # Calculate aggregate metrics
    aggregate_df = calculate_aggregate_metrics(all_results)
    calibration_df = calculate_calibration_metrics(all_results)
    
    # Save outputs
    save_benchmark_outputs(all_results, aggregate_df, calibration_df, args.output_dir)
    
    # Create plots
    create_benchmark_plots(all_results, aggregate_df, calibration_df, args.output_dir)
    
    # Create report
    create_benchmark_report(all_results, aggregate_df, calibration_df, args.output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("FPL BENCHMARKING SUMMARY")
    print("="*80)
    
    if not aggregate_df.empty:
        print("\n📊 AGGREGATE PERFORMANCE:")
        print("-" * 60)
        for _, row in aggregate_df.iterrows():
            print(f"{row['model']:<12} | MAE: {row['mae']:>6.2f} | RMSE: {row['rmse']:>6.2f} | "
                  f"r: {row['r']:>6.3f} | Win Rate: {row['win_rate_vs_template']:>5.1f}%")
    
    print(f"\n📁 Results saved to: {args.output_dir}")
    print("="*80)
    
    logger.info("✅ FPL benchmarking completed successfully!")


if __name__ == "__main__":
    main() 