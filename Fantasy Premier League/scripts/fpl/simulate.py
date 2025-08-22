#!/usr/bin/env python3
"""
FPL Simulation & Scenario Testing (Step 4c)

This script simulates FPL performance across multiple gameweeks with various scenarios
including injuries, fixture congestion, and chip usage.
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Import rotation functions
from .rotation import (
    VALID_FORMATIONS, pick_starting_xi, choose_captains, order_bench,
    validate_outputs, write_outputs
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Scenario definitions
AVAILABLE_SCENARIOS = [
    'baseline', 'injury_absences', 'low_minutes_downgrade', 
    'fixture_congestion', 'triple_captain_one_gw', 'bench_boost_one_gw'
]


def load_squad(squad_path: str) -> pd.DataFrame:
    """Load the FPL squad from CSV."""
    try:
        df = pd.read_csv(squad_path)
        logger.info(f"Loaded squad from {squad_path}: {len(df)} players")
        return df
    except Exception as e:
        logger.error(f"Failed to load squad from {squad_path}: {e}")
        sys.exit(1)


def load_features(features_path: str, gw_start: int, gw_end: int) -> pd.DataFrame:
    """Load FPL features data for the specified gameweek range."""
    try:
        df = pd.read_parquet(features_path)
        # Filter to target gameweek range
        df = df[df['gameweek'].between(gw_start, gw_end)]
        logger.info(f"Loaded features for GWs {gw_start}-{gw_end}: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Failed to load features from {features_path}: {e}")
        sys.exit(1)


def simulate_baseline(squad_df: pd.DataFrame, features_df: pd.DataFrame, 
                     formation: str, captain_strategy: str, bench_strategy: str, 
                     keep_gk_last: bool) -> Tuple[float, float, float, float, str, int]:
    """Simulate baseline scenario with no modifications."""
    # Get current GW data
    current_gw = features_df['gameweek'].iloc[0]
    gw_features = features_df[features_df['gameweek'] == current_gw]
    
    # Select only the additional features we need, avoiding conflicts with squad columns
    feature_cols = ['player_id', 'gameweek', 'minutes_flag_available', 'consistency_g6', 
                   'xp_rolling_mean_g3', 'fixture_congestion_lag1']
    # Only include columns that exist in the features data
    available_feature_cols = [col for col in feature_cols if col in gw_features.columns]
    
    # Merge squad with current GW features, keeping squad columns as primary
    gw_squad = squad_df.merge(gw_features[available_feature_cols], on='player_id', how='left')
    
    # Pick starting XI and bench
    xi_df, bench_df, actual_formation = pick_starting_xi(gw_squad, formation, {})
    
    # Choose captain and vice
    xi_df = choose_captains(xi_df, captain_strategy)
    
    # Order bench
    bench_df = order_bench(bench_df, bench_strategy, keep_gk_last)
    
    # Calculate totals
    xi_total = xi_df['xP'].sum()
    bench_total = bench_df['xP'].sum()
    squad_total = xi_total + bench_total
    
    # Get captain info
    captain = xi_df[xi_df['is_captain'] == 1].iloc[0]
    captain_bonus = captain['xP']  # Captain gets double points
    
    return xi_total, bench_total, squad_total, captain_bonus, actual_formation, current_gw


def simulate_injury_absences(squad_df: pd.DataFrame, features_df: pd.DataFrame,
                            formation: str, captain_strategy: str, bench_strategy: str,
                            keep_gk_last: bool, absence_rate: float, captain_absence_rate: float,
                            seed: int) -> Tuple[float, float, float, float, str, int]:
    """Simulate injury absences scenario."""
    random.seed(seed)
    
    current_gw = features_df['gameweek'].iloc[0]
    gw_features = features_df[features_df['gameweek'] == current_gw]
    
    # Select only the additional features we need, avoiding conflicts with squad columns
    feature_cols = ['player_id', 'gameweek', 'minutes_flag_available', 'consistency_g6', 
                   'xp_rolling_mean_g3', 'fixture_congestion_lag1']
    # Only include columns that exist in the features data
    available_feature_cols = [col for col in feature_cols if col in gw_features.columns]
    
    # Merge squad with current GW features, keeping squad columns as primary
    gw_squad = squad_df.merge(gw_features[available_feature_cols], on='player_id', how='left')
    
    # Pick starting XI and bench
    xi_df, bench_df, actual_formation = pick_starting_xi(gw_squad, formation, {})
    
    # Choose captain and vice
    xi_df = choose_captains(xi_df, captain_strategy)
    
    # Order bench
    bench_df = order_bench(bench_df, bench_strategy, keep_gk_last)
    
    # Simulate absences
    xi_df['absent'] = [random.random() < absence_rate for _ in range(len(xi_df))]
    
    # Special handling for captain absence
    captain_idx = xi_df[xi_df['is_captain'] == 1].index[0]
    if random.random() < captain_absence_rate:
        xi_df.loc[captain_idx, 'absent'] = True
    
    # Apply autosubs
    absent_starters = xi_df[xi_df['absent']]
    available_bench = bench_df[~bench_df['player_id'].isin(absent_starters['player_id'])]
    
    # Replace absent starters with bench players
    for _, absent_starter in absent_starters.iterrows():
        if len(available_bench) > 0:
            # Find best bench replacement for the position
            position = absent_starter['position']
            position_bench = available_bench[available_bench['position'] == position]
            
            if len(position_bench) > 0:
                replacement = available_bench.iloc[0]
                # Update XI with replacement
                xi_df.loc[absent_starter.name, 'player_id'] = replacement['player_id']
                xi_df.loc[absent_starter.name, 'xP'] = replacement['xP']
                xi_df.loc[absent_starter.name, 'absent'] = False
                
                # Remove replacement from available bench
                available_bench = available_bench[available_bench['player_id'] != replacement['player_id']]
    
    # Calculate totals (only non-absent players)
    active_xi = xi_df[~xi_df['absent']]
    xi_total = active_xi['xP'].sum()
    bench_total = available_bench['xP'].sum()
    squad_total = xi_total + bench_total
    
    # Captain bonus (captain gets double if present, vice gets double if captain absent)
    captain = active_xi[active_xi['is_captain'] == 1]
    if len(captain) > 0:
        captain_bonus = captain.iloc[0]['xP']
    else:
        # Captain absent, vice gets double
        vice = active_xi[active_xi['is_vice'] == 1]
        captain_bonus = vice.iloc[0]['xP'] if len(vice) > 0 else 0
    
    return xi_total, bench_total, squad_total, captain_bonus, actual_formation, current_gw


def simulate_low_minutes_downgrade(squad_df: pd.DataFrame, features_df: pd.DataFrame,
                                  formation: str, captain_strategy: str, bench_strategy: str,
                                  keep_gk_last: bool, low_minutes_mult: float) -> Tuple[float, float, float, float, str, int]:
    """Simulate low minutes downgrade scenario."""
    current_gw = features_df['gameweek'].iloc[0]
    gw_features = features_df[features_df['gameweek'] == current_gw]
    
    # Select only the additional features we need, avoiding conflicts with squad columns
    feature_cols = ['player_id', 'gameweek', 'minutes_flag_available', 'consistency_g6', 
                   'xp_rolling_mean_g3', 'fixture_congestion_lag1']
    # Only include columns that exist in the features data
    available_feature_cols = [col for col in feature_cols if col in gw_features.columns]
    
    # Merge squad with current GW features, keeping squad columns as primary
    gw_squad = squad_df.merge(gw_features[available_feature_cols], on='player_id', how='left')
    
    # Apply low minutes penalty
    if 'minutes_flag_available' in gw_squad.columns:
        gw_squad.loc[gw_squad['minutes_flag_available'] == 0, 'xP'] *= low_minutes_mult
        logger.info(f"Applied low minutes penalty (x{low_minutes_mult}) to players with minutes_flag_available=0")
    
    # Pick starting XI and bench
    xi_df, bench_df, actual_formation = pick_starting_xi(gw_squad, formation, {})
    
    # Choose captain and vice
    xi_df = choose_captains(xi_df, captain_strategy)
    
    # Order bench
    bench_df = order_bench(bench_df, bench_strategy, keep_gk_last)
    
    # Calculate totals
    xi_total = xi_df['xP'].sum()
    bench_total = bench_df['xP'].sum()
    squad_total = xi_total + bench_total
    
    # Captain bonus
    captain = xi_df[xi_df['is_captain'] == 1].iloc[0]
    captain_bonus = captain['xP']
    
    return xi_total, bench_total, squad_total, captain_bonus, actual_formation, current_gw


def simulate_fixture_congestion(squad_df: pd.DataFrame, features_df: pd.DataFrame,
                               formation: str, captain_strategy: str, bench_strategy: str,
                               keep_gk_last: bool, congestion_penalty: float) -> Tuple[float, float, float, float, str, int]:
    """Simulate fixture congestion scenario."""
    current_gw = features_df['gameweek'].iloc[0]
    gw_features = features_df[features_df['gameweek'] == current_gw]
    
    # Select only the additional features we need, avoiding conflicts with squad columns
    feature_cols = ['player_id', 'gameweek', 'minutes_flag_available', 'consistency_g6', 
                   'xp_rolling_mean_g3', 'fixture_congestion_lag1']
    # Only include columns that exist in the features data
    available_feature_cols = [col for col in feature_cols if col in gw_features.columns]
    
    # Merge squad with current GW features, keeping squad columns as primary
    gw_squad = squad_df.merge(gw_features[available_feature_cols], on='player_id', how='left')
    
    # Apply congestion penalty
    if 'fixture_congestion_lag1' in gw_squad.columns:
        congested_players = gw_squad[gw_squad['fixture_congestion_lag1'] == True]
        if len(congested_players) > 0:
            gw_squad.loc[gw_squad['fixture_congestion_lag1'] == True, 'xP'] *= congestion_penalty
            logger.info(f"Applied congestion penalty (x{congestion_penalty}) to {len(congested_players)} players")
    
    # Pick starting XI and bench
    xi_df, bench_df, actual_formation = pick_starting_xi(gw_squad, formation, {})
    
    # Choose captain and vice
    xi_df = choose_captains(xi_df, captain_strategy)
    
    # Order bench
    bench_df = order_bench(bench_df, bench_strategy, keep_gk_last)
    
    # Calculate totals
    xi_total = xi_df['xP'].sum()
    bench_total = bench_df['xP'].sum()
    squad_total = xi_total + bench_total
    
    # Captain bonus
    captain = xi_df[xi_df['is_captain'] == 1].iloc[0]
    captain_bonus = captain['xP']
    
    return xi_total, bench_total, squad_total, captain_bonus, actual_formation, current_gw


def simulate_triple_captain(squad_df: pd.DataFrame, features_df: pd.DataFrame,
                           formation: str, captain_strategy: str, bench_strategy: str,
                           keep_gk_last: bool) -> Tuple[float, float, float, float, str, int, int]:
    """Simulate triple captain scenario - find best GW and apply 3x bonus."""
    best_gw = None
    best_captain_xp = -1
    
    # Find GW with highest captain xP
    for gw in features_df['gameweek'].unique():
        gw_features = features_df[features_df['gameweek'] == gw]
        
        # Select only the additional features we need, avoiding conflicts with squad columns
        feature_cols = ['player_id', 'gameweek', 'minutes_flag_available', 'consistency_g6', 
                       'xp_rolling_mean_g3', 'fixture_congestion_lag1']
        # Only include columns that exist in the features data
        available_feature_cols = [col for col in feature_cols if col in gw_features.columns]
        
        # Merge squad with current GW features, keeping squad columns as primary
        gw_squad = squad_df.merge(gw_features[available_feature_cols], on='player_id', how='left')
        
        # Pick XI and captain for this GW
        xi_df, _, _ = pick_starting_xi(gw_squad, formation, {})
        xi_df = choose_captains(xi_df, captain_strategy)
        
        captain = xi_df[xi_df['is_captain'] == 1].iloc[0]
        if captain['xP'] > best_captain_xp:
            best_captain_xp = captain['xP']
            best_gw = gw
    
    # Now simulate the best GW with triple captain
    best_gw_features = features_df[features_df['gameweek'] == best_gw]
    
    # Select only the additional features we need, avoiding conflicts with squad columns
    feature_cols = ['player_id', 'gameweek', 'minutes_flag_available', 'consistency_g6', 
                   'xp_rolling_mean_g3', 'fixture_congestion_lag1']
    # Only include columns that exist in the features data
    available_feature_cols = [col for col in feature_cols if col in best_gw_features.columns]
    
    # Merge squad with current GW features, keeping squad columns as primary
    best_gw_squad = squad_df.merge(best_gw_features[available_feature_cols], on='player_id', how='left')
    
    # Pick XI and captain
    xi_df, bench_df, actual_formation = pick_starting_xi(best_gw_squad, formation, {})
    xi_df = choose_captains(xi_df, captain_strategy)
    
    # Order bench
    bench_df = order_bench(bench_df, bench_strategy, keep_gk_last)
    
    # Calculate totals with triple captain
    xi_total = xi_df['xP'].sum()
    bench_total = bench_df['xP'].sum()
    squad_total = xi_total + bench_total
    
    # Triple captain bonus (3x instead of 2x)
    captain = xi_df[xi_df['is_captain'] == 1].iloc[0]
    captain_bonus = captain['xP'] * 2  # Additional 2x on top of base xP
    
    return xi_total, bench_total, squad_total, captain_bonus, actual_formation, best_gw, best_gw


def simulate_bench_boost(squad_df: pd.DataFrame, features_df: pd.DataFrame,
                         formation: str, captain_strategy: str, bench_strategy: str,
                         keep_gk_last: bool) -> Tuple[float, float, float, float, str, int, int]:
    """Simulate bench boost scenario - find best GW and add bench to XI."""
    best_gw = None
    best_bench_xp = -1
    
    # Find GW with highest bench xP
    for gw in features_df['gameweek'].unique():
        gw_features = features_df[features_df['gameweek'] == gw]
        
        # Select only the additional features we need, avoiding conflicts with squad columns
        feature_cols = ['player_id', 'gameweek', 'minutes_flag_available', 'consistency_g6', 
                       'xp_rolling_mean_g3', 'fixture_congestion_lag1']
        # Only include columns that exist in the features data
        available_feature_cols = [col for col in feature_cols if col in gw_features.columns]
        
        # Merge squad with current GW features, keeping squad columns as primary
        gw_squad = squad_df.merge(gw_features[available_feature_cols], on='player_id', how='left')
        
        # Pick XI and bench for this GW
        xi_df, bench_df, _ = pick_starting_xi(gw_squad, formation, {})
        
        bench_total = bench_df['xP'].sum()
        if bench_total > best_bench_xp:
            best_bench_xp = bench_total
            best_gw = gw
    
    # Now simulate the best GW with bench boost
    best_gw_features = features_df[features_df['gameweek'] == best_gw]
    
    # Select only the additional features we need, avoiding conflicts with squad columns
    feature_cols = ['player_id', 'gameweek', 'minutes_flag_available', 'consistency_g6', 
                   'xp_rolling_mean_g3', 'fixture_congestion_lag1']
    # Only include columns that exist in the features data
    available_feature_cols = [col for col in feature_cols if col in best_gw_features.columns]
    
    # Merge squad with current GW features, keeping squad columns as primary
    best_gw_squad = squad_df.merge(best_gw_features[available_feature_cols], on='player_id', how='left')
    
    # Pick XI and captain
    xi_df, bench_df, actual_formation = pick_starting_xi(best_gw_squad, formation, {})
    xi_df = choose_captains(xi_df, captain_strategy)
    
    # Order bench
    bench_df = order_bench(bench_df, bench_strategy, keep_gk_last)
    
    # Calculate totals with bench boost (bench points added to XI)
    xi_total = xi_df['xP'].sum() + bench_df['xP'].sum()
    bench_total = 0  # Bench points are now part of XI
    squad_total = xi_total
    
    # Captain bonus
    captain = xi_df[xi_df['is_captain'] == 1].iloc[0]
    captain_bonus = captain['xP']
    
    return xi_total, bench_total, squad_total, captain_bonus, actual_formation, best_gw, best_gw


def run_simulation(squad_df: pd.DataFrame, features_df: pd.DataFrame, scenario: str,
                  formation: str, captain_strategy: str, bench_strategy: str, keep_gk_last: bool,
                  **kwargs) -> Tuple[float, float, float, float, str, int, Optional[int]]:
    """Run simulation for a specific scenario."""
    if scenario == 'baseline':
        return simulate_baseline(squad_df, features_df, formation, captain_strategy, 
                               bench_strategy, keep_gk_last)
    elif scenario == 'injury_absences':
        return simulate_injury_absences(squad_df, features_df, formation, captain_strategy,
                                      bench_strategy, keep_gk_last, 
                                      kwargs.get('absence_rate', 0.05),
                                      kwargs.get('captain_absence_rate', 0.02),
                                      kwargs.get('seed', 42))
    elif scenario == 'low_minutes_downgrade':
        return simulate_low_minutes_downgrade(squad_df, features_df, formation, captain_strategy,
                                            bench_strategy, keep_gk_last,
                                            kwargs.get('low_minutes_mult', 0.8))
    elif scenario == 'fixture_congestion':
        return simulate_fixture_congestion(squad_df, features_df, formation, captain_strategy,
                                         bench_strategy, keep_gk_last,
                                         kwargs.get('congestion_penalty', 0.9))
    elif scenario == 'triple_captain_one_gw':
        result = simulate_triple_captain(squad_df, features_df, formation, captain_strategy,
                                       bench_strategy, keep_gk_last)
        return result[:-1] + (result[-1],)  # Return chip_gw as last element
    elif scenario == 'bench_boost_one_gw':
        result = simulate_bench_boost(squad_df, features_df, formation, captain_strategy,
                                    bench_strategy, keep_gk_last)
        return result[:-1] + (result[-1],)  # Return chip_gw as last element
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def save_results(summary_results: List[Dict], per_gw_results: List[Dict], 
                scenario_config: Dict, output_dir: str):
    """Save simulation results to files."""
    output_path = Path(output_dir) / 'sims'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary_df = pd.DataFrame(summary_results)
    summary_path = output_path / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")
    
    # Save per-GW results
    per_gw_df = pd.DataFrame(per_gw_results)
    per_gw_path = output_path / 'per_gw.csv'
    per_gw_df.to_csv(per_gw_path, index=False)
    logger.info(f"Per-GW results saved to {per_gw_path}")
    
    # Save scenario config
    config_path = output_path / 'scenario_config.json'
    with open(config_path, 'w') as f:
        json.dump(scenario_config, f, indent=2)
    logger.info(f"Scenario config saved to {config_path}")


def main():
    """Main function to run the FPL simulation."""
    parser = argparse.ArgumentParser(description='FPL Simulation & Scenario Testing (Step 4c)')
    parser.add_argument('--gw-start', type=int, required=True, help='Starting gameweek')
    parser.add_argument('--gw-end', type=int, required=True, help='Ending gameweek')
    parser.add_argument('--squad-path', default='outputs/fpl_squad.csv',
                       help='Path to squad CSV (default: outputs/fpl_squad.csv)')
    parser.add_argument('--features-path', default='data/FPL/processed/fpl_features_model.parquet',
                       help='Path to features data (default: data/FPL/processed/fpl_features_model.parquet)')
    parser.add_argument('--enable', nargs='+', default=['baseline'],
                       choices=AVAILABLE_SCENARIOS,
                       help='Scenarios to enable (default: baseline)')
    
    # Scenario-specific parameters
    parser.add_argument('--starter-absence-rate', type=float, default=0.05,
                       help='Starter absence rate for injury scenario (default: 0.05)')
    parser.add_argument('--captain-absence-rate', type=float, default=0.02,
                       help='Captain absence rate for injury scenario (default: 0.02)')
    parser.add_argument('--low-minutes-mult', type=float, default=0.8,
                       help='Low minutes multiplier (default: 0.8)')
    parser.add_argument('--congestion-penalty', type=float, default=0.9,
                       help='Fixture congestion penalty (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    # Rotation pass-through arguments
    parser.add_argument('--formation', default='auto',
                       choices=['auto'] + list(VALID_FORMATIONS.keys()),
                       help='Formation to use (default: auto)')
    parser.add_argument('--captain-strategy', default='top_xp',
                       choices=['top_xp', 'risk_adjusted'],
                       help='Captain selection strategy (default: top_xp)')
    parser.add_argument('--bench-strategy', default='xp',
                       choices=['xp', 'low_minutes_first'],
                       help='Bench ordering strategy (default: xp)')
    parser.add_argument('--bench-keep-gk-last', default=True, type=lambda x: x.lower() == 'true',
                       help='Keep GK last on bench (default: true)')
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory (default: outputs)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load data
    squad_df = load_squad(args.squad_path)
    features_df = load_features(args.features_path, args.gw_start, args.gw_end)
    
    # Validate gameweek range
    available_gws = sorted(features_df['gameweek'].unique())
    if args.gw_start not in available_gws or args.gw_end not in available_gws:
        logger.error(f"Gameweek range {args.gw_start}-{args.gw_end} not available. Available: {available_gws}")
        sys.exit(1)
    
    # Prepare scenario config
    scenario_config = {
        'timestamp': datetime.now().isoformat(),
        'gw_start': args.gw_start,
        'gw_end': args.gw_end,
        'enabled_scenarios': args.enable,
        'starter_absence_rate': args.starter_absence_rate,
        'captain_absence_rate': args.captain_absence_rate,
        'low_minutes_mult': args.low_minutes_mult,
        'congestion_penalty': args.congestion_penalty,
        'seed': args.seed,
        'formation': args.formation,
        'captain_strategy': args.captain_strategy,
        'bench_strategy': args.bench_strategy,
        'bench_keep_gk_last': args.bench_keep_gk_last
    }
    
    # Run simulations
    summary_results = []
    per_gw_results = []
    
    for scenario in args.enable:
        logger.info(f"Running scenario: {scenario}")
        
        # Run simulation for each GW
        scenario_summary = {
            'scenario': scenario,
            'gw_start': args.gw_start,
            'gw_end': args.gw_end,
            'xi_total_xp_sum': 0,
            'squad_total_xp_sum': 0,
            'bench_total_xp_sum': 0,
            'captain_bonus_sum': 0,
            'chip_used': scenario in ['triple_captain_one_gw', 'bench_boost_one_gw'],
            'chip_gw': None
        }
        
        for gw in range(args.gw_start, args.gw_end + 1):
            gw_features = features_df[features_df['gameweek'] == gw]
            if len(gw_features) == 0:
                continue
            
            # Run simulation for this GW
            try:
                if scenario in ['triple_captain_one_gw', 'bench_boost_one_gw']:
                    xi_total, bench_total, squad_total, captain_bonus, formation, current_gw, chip_gw = \
                        run_simulation(squad_df, gw_features, scenario, args.formation, 
                                     args.captain_strategy, args.bench_strategy, args.bench_keep_gk_last,
                                     absence_rate=args.starter_absence_rate,
                                     captain_absence_rate=args.captain_absence_rate,
                                     low_minutes_mult=args.low_minutes_mult,
                                     congestion_penalty=args.congestion_penalty,
                                     seed=args.seed)
                    scenario_summary['chip_gw'] = chip_gw
                else:
                    xi_total, bench_total, squad_total, captain_bonus, formation, current_gw = \
                        run_simulation(squad_df, gw_features, scenario, args.formation, 
                                     args.captain_strategy, args.bench_strategy, args.bench_keep_gk_last,
                                     absence_rate=args.starter_absence_rate,
                                     captain_absence_rate=args.captain_absence_rate,
                                     low_minutes_mult=args.low_minutes_mult,
                                     congestion_penalty=args.congestion_penalty,
                                     seed=args.seed)
                
                # Accumulate totals
                scenario_summary['xi_total_xp_sum'] += xi_total
                scenario_summary['squad_total_xp_sum'] += squad_total
                scenario_summary['bench_total_xp_sum'] += bench_total
                scenario_summary['captain_bonus_sum'] += captain_bonus
                
                # Record per-GW results
                per_gw_results.append({
                    'scenario': scenario,
                    'gw': current_gw,
                    'formation': formation,
                    'xi_total_xp': xi_total,
                    'bench_total_xp': bench_total,
                    'captain_id': None,  # TODO: Extract from simulation
                    'captain_name': None,  # TODO: Extract from simulation
                    'captain_xP': captain_bonus,
                    'vice_id': None,  # TODO: Extract from simulation
                    'vice_name': None,  # TODO: Extract from simulation
                    'used_triple_captain': scenario == 'triple_captain_one_gw' and current_gw == scenario_summary['chip_gw'],
                    'used_bench_boost': scenario == 'bench_boost_one_gw' and current_gw == scenario_summary['chip_gw']
                })
                
            except Exception as e:
                logger.error(f"Failed to simulate {scenario} for GW {gw}: {e}")
                continue
        
        summary_results.append(scenario_summary)
    
    # Save results
    save_results(summary_results, per_gw_results, scenario_config, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("FPL SIMULATION SUMMARY")
    print("="*60)
    print(f"Gameweeks: {args.gw_start}-{args.gw_end}")
    print(f"Scenarios: {', '.join(args.enable)}")
    print(f"Results saved to: {args.output_dir}/sims/")
    
    for result in summary_results:
        print(f"\n{result['scenario'].upper()}:")
        print(f"  XI Total xP: {result['xi_total_xp_sum']:.2f}")
        print(f"  Squad Total xP: {result['squad_total_xp_sum']:.2f}")
        print(f"  Captain Bonus: {result['captain_bonus_sum']:.2f}")
        if result['chip_used']:
            print(f"  Chip Used: {result['scenario']} in GW {result['chip_gw']}")
    
    print("="*60)


if __name__ == "__main__":
    main() 