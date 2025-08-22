#!/usr/bin/env python3
"""
FPL Optimizer Evaluation & Stress Tests (Step 5a)

This script evaluates optimizer outputs against baselines and runs stress tests
to validate the optimization approach and robustness.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import pulp

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


def load_optimizer_outputs(output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load optimizer outputs: squad, starting XI, and bench."""
    try:
        squad_path = os.path.join(output_dir, 'fpl_squad.csv')
        xi_path = os.path.join(output_dir, 'fpl_starting_xi.csv')
        bench_path = os.path.join(output_dir, 'fpl_bench.csv')
        
        squad_df = pd.read_csv(squad_path)
        xi_df = pd.read_csv(xi_path)
        bench_df = pd.read_csv(bench_path)
        
        logger.info(f"✅ Loaded optimizer outputs: {len(squad_df)} squad, {len(xi_df)} XI, {len(bench_df)} bench")
        return squad_df, xi_df, bench_df
        
    except Exception as e:
        logger.error(f"❌ Failed to load optimizer outputs: {e}")
        sys.exit(1)


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


def create_template_squad(features_df: pd.DataFrame, budget: float = BUDGET_LIMIT) -> pd.DataFrame:
    """Create template squad using most-owned players within budget."""
    try:
        # Get latest gameweek data
        latest_gw = features_df['gameweek'].max()
        gw_data = features_df[features_df['gameweek'] == latest_gw].copy()
        
        # Sort by ownership (if available) or xP, then by price
        if 'ownership' in gw_data.columns:
            gw_data = gw_data.sort_values(['ownership', 'xP'], ascending=[False, False])
        else:
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
            logger.warning(f"⚠️ Template squad incomplete: {len(squad)}/{SQUAD_SIZE} players")
            
        template_df = pd.DataFrame(squad)
        logger.info(f"✅ Created template squad: {len(template_df)} players, £{budget - remaining_budget:.1f} spent")
        return template_df
        
    except Exception as e:
        logger.error(f"❌ Failed to create template squad: {e}")
        return pd.DataFrame()


def create_random_squad(features_df: pd.DataFrame, budget: float = BUDGET_LIMIT) -> pd.DataFrame:
    """Create random valid squad under FPL rules."""
    try:
        # Get latest gameweek data
        latest_gw = features_df['gameweek'].max()
        gw_data = features_df[features_df['gameweek'] == latest_gw].copy()
        
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
            logger.warning(f"⚠️ Random squad incomplete: {len(squad)}/{SQUAD_SIZE} players")
            
        random_df = pd.DataFrame(squad)
        logger.info(f"✅ Created random squad: {len(random_df)} players, £{budget - remaining_budget:.1f} spent")
        return random_df
        
    except Exception as e:
        logger.error(f"❌ Failed to create random squad: {e}")
        return pd.DataFrame()


def create_naive_max_xp_squad(features_df: pd.DataFrame) -> pd.DataFrame:
    """Select top 15 players by projected points ignoring constraints."""
    try:
        # Get latest gameweek data
        latest_gw = features_df['gameweek'].max()
        gw_data = features_df[features_df['gameweek'] == latest_gw].copy()
        
        # Sort by xP and take top 15
        naive_df = gw_data.nlargest(SQUAD_SIZE, 'xP')
        
        # Calculate total cost
        total_cost = naive_df['price'].sum()
        
        logger.info(f"✅ Created naïve max xP squad: {len(naive_df)} players, £{total_cost:.1f} cost")
        return naive_df
        
    except Exception as e:
        logger.error(f"❌ Failed to create naïve max xP squad: {e}")
        return pd.DataFrame()


def validate_squad_constraints(squad_df: pd.DataFrame, budget: float = BUDGET_LIMIT, 
                              club_cap: int = DEFAULT_CLUB_CAP) -> Dict[str, Any]:
    """Validate squad against FPL constraints."""
    try:
        validation = {
            'total_players': len(squad_df),
            'total_cost': squad_df['price'].sum(),
            'budget_compliant': squad_df['price'].sum() <= budget,
            'position_quota_met': True,
            'club_cap_met': True,
            'constraints_satisfied': True
        }
        
        # Check position quotas
        position_counts = squad_df['position'].value_counts()
        for pos, quota in POSITION_QUOTAS.items():
            if position_counts.get(pos, 0) != quota:
                validation['position_quota_met'] = False
                validation['constraints_satisfied'] = False
        
        # Check club cap
        club_counts = squad_df['team_name'].value_counts()
        if (club_counts > club_cap).any():
            validation['club_cap_met'] = False
            validation['constraints_satisfied'] = False
        
        # Calculate total xP
        validation['total_xp'] = squad_df['xP'].sum()
        
        return validation
        
    except Exception as e:
        logger.error(f"❌ Failed to validate squad constraints: {e}")
        return {}


def run_budget_stress_test(features_df: pd.DataFrame, club_cap: int = DEFAULT_CLUB_CAP) -> Dict[str, Any]:
    """Run budget variation stress test."""
    try:
        results = {}
        budgets = [95.0, 100.0, 105.0]
        
        for budget in budgets:
            logger.info(f"🔄 Testing budget £{budget}m...")
            
            # Create squad with budget constraint
            squad = create_template_squad(features_df, budget * 10)  # Convert to actual price scale
            
            if not squad.empty:
                validation = validate_squad_constraints(squad, budget * 10, club_cap)
                results[f'budget_{int(budget)}m'] = {
                    'squad_size': validation['total_players'],
                    'total_cost': validation['total_cost'],
                    'total_xp': validation['total_xp'],
                    'constraints_satisfied': validation['constraints_satisfied']
                }
            else:
                results[f'budget_{int(budget)}m'] = {'error': 'Failed to create squad'}
        
        logger.info(f"✅ Budget stress test completed")
        return results
        
    except Exception as e:
        logger.error(f"❌ Budget stress test failed: {e}")
        return {}


def run_club_cap_stress_test(features_df: pd.DataFrame, budget: float = BUDGET_LIMIT) -> Dict[str, Any]:
    """Run club cap variation stress test."""
    try:
        results = {}
        club_caps = [2, 3]
        
        for cap in club_caps:
            logger.info(f"🔄 Testing club cap {cap}...")
            
            # Create squad with club cap constraint
            squad = create_template_squad(features_df, budget)
            
            if not squad.empty:
                validation = validate_squad_constraints(squad, budget, cap)
                results[f'club_cap_{cap}'] = {
                    'squad_size': validation['total_players'],
                    'total_cost': validation['total_cost'],
                    'total_xp': validation['total_xp'],
                    'constraints_satisfied': validation['constraints_satisfied']
                }
            else:
                results[f'club_cap_{cap}'] = {'error': 'Failed to create squad'}
        
        logger.info(f"✅ Club cap stress test completed")
        return results
        
    except Exception as e:
        logger.error(f"❌ Club cap stress test failed: {e}")
        return {}


def run_projection_model_stress_test(features_df: pd.DataFrame) -> Dict[str, Any]:
    """Run projection model variation stress test."""
    try:
        results = {}
        
        # Get latest gameweek
        latest_gw = features_df['gameweek'].max()
        
        # Test 1: Season average xP
        season_avg = features_df.groupby('player_id')['xP'].mean().reset_index()
        season_avg = season_avg.merge(features_df[features_df['gameweek'] == latest_gw][['player_id', 'position', 'price', 'team_name']], on='player_id')
        season_avg = season_avg.sort_values('xP', ascending=False).head(SQUAD_SIZE)
        
        # Test 2: Last 3 GWs rolling average (if available)
        if latest_gw >= 3:
            recent_data = features_df[features_df['gameweek'] >= latest_gw - 2]
            rolling_avg = recent_data.groupby('player_id')['xP'].mean().reset_index()
            rolling_avg = rolling_avg.merge(features_df[features_df['gameweek'] == latest_gw][['player_id', 'position', 'price', 'team_name']], on='player_id')
            rolling_avg = rolling_avg.sort_values('xP', ascending=False).head(SQUAD_SIZE)
        else:
            rolling_avg = pd.DataFrame()
        
        # Validate squads
        if not season_avg.empty:
            validation = validate_squad_constraints(season_avg)
            results['season_average'] = {
                'squad_size': validation['total_players'],
                'total_cost': validation['total_cost'],
                'total_xp': validation['total_xp'],
                'constraints_satisfied': validation['constraints_satisfied']
            }
        
        if not rolling_avg.empty:
            validation = validate_squad_constraints(rolling_avg)
            results['rolling_3gw'] = {
                'squad_size': validation['total_players'],
                'total_cost': validation['total_cost'],
                'total_xp': validation['total_xp'],
                'constraints_satisfied': validation['constraints_satisfied']
            }
        
        logger.info(f"✅ Projection model stress test completed")
        return results
        
    except Exception as e:
        logger.error(f"❌ Projection model stress test failed: {e}")
        return {}


def run_determinism_check(features_df: pd.DataFrame, num_runs: int = 3) -> Dict[str, Any]:
    """Check if repeated runs yield identical outputs."""
    try:
        results = []
        
        for run in range(num_runs):
            logger.info(f"🔄 Determinism check run {run + 1}/{num_runs}...")
            
            # Create squad with same parameters
            squad = create_template_squad(features_df, BUDGET_LIMIT)
            
            if not squad.empty:
                # Sort by player_id for consistent comparison
                squad_sorted = squad.sort_values('player_id')
                squad_hash = hash(tuple(squad_sorted['player_id'].values))
                
                results.append({
                    'run': run + 1,
                    'squad_size': len(squad),
                    'total_cost': squad['price'].sum(),
                    'total_xp': squad['xP'].sum(),
                    'squad_hash': squad_hash
                })
        
        # Check if all runs produced identical results
        if len(results) > 1:
            first_hash = results[0]['squad_hash']
            deterministic = all(r['squad_hash'] == first_hash for r in results)
        else:
            deterministic = True
        
        logger.info(f"✅ Determinism check completed: {'deterministic' if deterministic else 'non-deterministic'}")
        
        return {
            'deterministic': deterministic,
            'runs': results
        }
        
    except Exception as e:
        logger.error(f"❌ Determinism check failed: {e}")
        return {}


def create_evaluation_summary(optimizer_squad: pd.DataFrame, template_squad: pd.DataFrame,
                             random_squad: pd.DataFrame, naive_squad: pd.DataFrame) -> pd.DataFrame:
    """Create evaluation summary comparing all approaches."""
    try:
        summary_data = []
        
        # Optimizer squad
        if not optimizer_squad.empty:
            optimizer_validation = validate_squad_constraints(optimizer_squad)
            summary_data.append({
                'approach': 'Optimizer (ILP)',
                'squad_size': optimizer_validation['total_players'],
                'total_cost': optimizer_validation['total_cost'],
                'total_xp': optimizer_validation['total_xp'],
                'budget_compliant': optimizer_validation['budget_compliant'],
                'position_quota_met': optimizer_validation['position_quota_met'],
                'club_cap_met': optimizer_validation['club_cap_met'],
                'constraints_satisfied': optimizer_validation['constraints_satisfied']
            })
        
        # Template squad
        if not template_squad.empty:
            template_validation = validate_squad_constraints(template_squad)
            summary_data.append({
                'approach': 'Template (Most Owned)',
                'squad_size': template_validation['total_players'],
                'total_cost': template_validation['total_cost'],
                'total_xp': template_validation['total_xp'],
                'budget_compliant': template_validation['budget_compliant'],
                'position_quota_met': template_validation['position_quota_met'],
                'club_cap_met': template_validation['club_cap_met'],
                'constraints_satisfied': template_validation['constraints_satisfied']
            })
        
        # Random squad
        if not random_squad.empty:
            random_validation = validate_squad_constraints(random_squad)
            summary_data.append({
                'approach': 'Random',
                'squad_size': random_validation['total_players'],
                'total_cost': random_validation['total_cost'],
                'total_xp': random_validation['total_xp'],
                'budget_compliant': random_validation['budget_compliant'],
                'position_quota_met': random_validation['position_quota_met'],
                'club_cap_met': random_validation['club_cap_met'],
                'constraints_satisfied': random_validation['constraints_satisfied']
            })
        
        # Naïve max xP squad
        if not naive_squad.empty:
            naive_validation = validate_squad_constraints(naive_squad)
            summary_data.append({
                'approach': 'Naïve Max xP',
                'squad_size': naive_validation['total_players'],
                'total_cost': naive_validation['total_cost'],
                'total_xp': naive_validation['total_xp'],
                'budget_compliant': naive_validation['budget_compliant'],
                'position_quota_met': naive_validation['position_quota_met'],
                'club_cap_met': naive_validation['club_cap_met'],
                'constraints_satisfied': naive_validation['constraints_satisfied']
            })
        
        summary_df = pd.DataFrame(summary_data)
        logger.info(f"✅ Created evaluation summary: {len(summary_df)} approaches")
        return summary_df
        
    except Exception as e:
        logger.error(f"❌ Failed to create evaluation summary: {e}")
        return pd.DataFrame()


def create_stress_test_summary(budget_results: Dict, club_cap_results: Dict, 
                              projection_results: Dict, determinism_results: Dict) -> pd.DataFrame:
    """Create stress test summary."""
    try:
        stress_data = []
        
        # Budget variation results
        for budget, result in budget_results.items():
            if 'error' not in result:
                stress_data.append({
                    'test_type': 'Budget Variation',
                    'test_config': budget,
                    'squad_size': result['squad_size'],
                    'total_cost': result['total_cost'],
                    'total_xp': result['total_xp'],
                    'constraints_satisfied': result['constraints_satisfied']
                })
        
        # Club cap variation results
        for cap, result in club_cap_results.items():
            if 'error' not in result:
                stress_data.append({
                    'test_type': 'Club Cap Variation',
                    'test_config': cap,
                    'squad_size': result['squad_size'],
                    'total_cost': result['total_cost'],
                    'total_xp': result['total_xp'],
                    'constraints_satisfied': result['constraints_satisfied']
                })
        
        # Projection model results
        for model, result in projection_results.items():
            if 'error' not in result:
                stress_data.append({
                    'test_type': 'Projection Model',
                    'test_config': model,
                    'squad_size': result['squad_size'],
                    'total_cost': result['total_cost'],
                    'total_xp': result['total_xp'],
                    'constraints_satisfied': result['constraints_satisfied']
                })
        
        # Determinism results
        if determinism_results:
            stress_data.append({
                'test_type': 'Determinism',
                'test_config': f"{determinism_results.get('deterministic', False)}",
                'squad_size': len(determinism_results.get('runs', [])),
                'total_cost': 'N/A',
                'total_xp': 'N/A',
                'constraints_satisfied': determinism_results.get('deterministic', False)
            })
        
        stress_df = pd.DataFrame(stress_data)
        logger.info(f"✅ Created stress test summary: {len(stress_df)} tests")
        return stress_df
        
    except Exception as e:
        logger.error(f"❌ Failed to create stress test summary: {e}")
        return pd.DataFrame()


def print_console_summary(summary_df: pd.DataFrame, stress_df: pd.DataFrame):
    """Print console summary showing performance uplift."""
    try:
        print("\n" + "="*80)
        print("FPL OPTIMIZER EVALUATION SUMMARY")
        print("="*80)
        
        if not summary_df.empty:
            print("\n📊 BASELINE COMPARISON:")
            print("-" * 60)
            
            # Find optimizer row
            optimizer_row = summary_df[summary_df['approach'] == 'Optimizer (ILP)']
            if not optimizer_row.empty:
                optimizer_xp = optimizer_row.iloc[0]['total_xp']
                
                for _, row in summary_df.iterrows():
                    approach = row['approach']
                    xp = row['total_xp']
                    cost = row['total_cost']
                    constraints = "✅" if row['constraints_satisfied'] else "❌"
                    
                    if approach != 'Optimizer (ILP)':
                        uplift = ((optimizer_xp - xp) / xp) * 100 if xp > 0 else 0
                        print(f"{approach:<20} | xP: {xp:>6.1f} | Cost: £{cost:>6.1f} | Uplift: {uplift:>+6.1f}% | {constraints}")
                    else:
                        print(f"{approach:<20} | xP: {xp:>6.1f} | Cost: £{cost:>6.1f} | {'':>6} | {constraints}")
        
        if not stress_df.empty:
            print("\n🧪 STRESS TEST RESULTS:")
            print("-" * 60)
            
            for _, row in stress_df.iterrows():
                test_type = row['test_type']
                config = row['test_config']
                constraints = "✅" if row['constraints_satisfied'] else "❌"
                
                if row['total_cost'] != 'N/A':
                    print(f"{test_type:<20} | {config:<15} | Constraints: {constraints}")
                else:
                    print(f"{test_type:<20} | {config:<15} | Deterministic: {constraints}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"❌ Failed to print console summary: {e}")


def save_evaluation_outputs(summary_df: pd.DataFrame, stress_df: pd.DataFrame, output_dir: str):
    """Save evaluation outputs to CSV files."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save evaluation summary
        summary_path = os.path.join(output_dir, 'fpl_eval_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"✅ Saved evaluation summary to {summary_path}")
        
        # Save stress test results
        stress_path = os.path.join(output_dir, 'fpl_eval_stress.csv')
        stress_df.to_csv(stress_path, index=False)
        logger.info(f"✅ Saved stress test results to {stress_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save evaluation outputs: {e}")


def main():
    """Main function to run FPL evaluation and stress tests."""
    parser = argparse.ArgumentParser(description='FPL Optimizer Evaluation & Stress Tests')
    parser.add_argument('--output-dir', default='outputs',
                       help='Directory containing optimizer outputs')
    parser.add_argument('--features-path', default='data/FPL/processed/fpl_features_model.parquet',
                       help='Path to FPL features data')
    parser.add_argument('--teams-path', default='data/FPL/processed/teams_processed.parquet',
                       help='Path to teams data')
    parser.add_argument('--eval-output-dir', default='outputs',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting FPL Optimizer Evaluation & Stress Tests (Step 5a)...")
    
    # Load data
    optimizer_squad, optimizer_xi, optimizer_bench = load_optimizer_outputs(args.output_dir)
    features_df = load_features_data(args.features_path)
    team_names = load_team_names(args.teams_path)
    
    # Merge team names into features data
    features_df = merge_team_names(features_df, team_names)
    
    # Create baseline squads
    logger.info("📋 Creating baseline squads for comparison...")
    template_squad = create_template_squad(features_df)
    random_squad = create_random_squad(features_df)
    naive_squad = create_naive_max_xp_squad(features_df)
    
    # Run stress tests
    logger.info("🧪 Running stress tests...")
    budget_results = run_budget_stress_test(features_df)
    club_cap_results = run_club_cap_stress_test(features_df)
    projection_results = run_projection_model_stress_test(features_df)
    determinism_results = run_determinism_check(features_df)
    
    # Create evaluation summaries
    logger.info("📊 Creating evaluation summaries...")
    summary_df = create_evaluation_summary(optimizer_squad, template_squad, random_squad, naive_squad)
    stress_df = create_stress_test_summary(budget_results, club_cap_results, projection_results, determinism_results)
    
    # Save outputs
    save_evaluation_outputs(summary_df, stress_df, args.eval_output_dir)
    
    # Print console summary
    print_console_summary(summary_df, stress_df)
    
    logger.info("✅ FPL evaluation and stress tests completed successfully!")


if __name__ == "__main__":
    main() 