#!/usr/bin/env python3
"""
FPL Squad Rotation Script (Step 4b)

This script takes the optimized squad from Step 4a and produces:
- Starting XI with formation selection
- Captain and vice-captain assignment
- Bench ordering
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Valid formations and their position requirements
VALID_FORMATIONS = {
    '343': {'GK': 1, 'DEF': 3, 'MID': 4, 'FWD': 3},
    '352': {'GK': 1, 'DEF': 3, 'MID': 5, 'FWD': 2},
    '442': {'GK': 1, 'DEF': 4, 'MID': 4, 'FWD': 2},
    '451': {'GK': 1, 'DEF': 4, 'MID': 5, 'FWD': 1},
    '433': {'GK': 1, 'DEF': 4, 'MID': 3, 'FWD': 3},
    '532': {'GK': 1, 'DEF': 5, 'MID': 3, 'FWD': 2},
    '541': {'GK': 1, 'DEF': 5, 'MID': 4, 'FWD': 1}
}


def load_squad(squad_path: str) -> pd.DataFrame:
    """Load the FPL squad from CSV."""
    try:
        df = pd.read_csv(squad_path)
        logger.info(f"Loaded squad from {squad_path}: {len(df)} players")
        return df
    except Exception as e:
        logger.error(f"Failed to load squad from {squad_path}: {e}")
        sys.exit(1)


def maybe_merge_features(squad_df: pd.DataFrame, features_path: Optional[str]) -> pd.DataFrame:
    """Merge additional features if features_path is provided."""
    if features_path is None:
        return squad_df
    
    try:
        features_df = pd.read_parquet(features_path)
        # Filter to same gameweek if available
        if 'gameweek' in features_df.columns:
            gameweek = squad_df.get('gameweek', 1).iloc[0] if 'gameweek' in squad_df.columns else 1
            features_df = features_df[features_df['gameweek'] == gameweek]
        
        # Merge on player_id
        merged_df = squad_df.merge(
            features_df[['player_id', 'minutes_flag_available', 'consistency_g6', 'xp_rolling_mean_g3']], 
            on='player_id', 
            how='left'
        )
        logger.info(f"Merged features from {features_path}")
        return merged_df
    except Exception as e:
        logger.warning(f"Failed to merge features: {e}")
        return squad_df


def _pick_auto_formation(squad_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Automatically select the best formation by evaluating all valid options."""
    best_formation = None
    best_xi = None
    best_bench = None
    best_total_xp = -1
    
    formation_results = []
    
    for formation_name, position_quota in VALID_FORMATIONS.items():
        try:
            xi_df, bench_df = _pick_fixed_formation(squad_df, formation_name)
            total_xp = xi_df['xP'].sum()
            formation_results.append({
                'formation': formation_name,
                'total_xp': total_xp,
                'xi_players': len(xi_df)
            })
            
            if total_xp > best_total_xp:
                best_total_xp = total_xp
                best_formation = formation_name
                best_xi = xi_df
                best_bench = bench_df
                
        except Exception as e:
            logger.warning(f"Formation {formation_name} failed: {e}")
            continue
    
    # Print formation comparison table
    if formation_results:
        print("\nFormation Analysis:")
        print("Formation | Total xP | Status")
        print("-" * 25)
        for result in sorted(formation_results, key=lambda x: x['total_xp'], reverse=True):
            status = "✓ SELECTED" if result['formation'] == best_formation else ""
            print(f"{result['formation']:9} | {result['total_xp']:8.2f} | {status}")
    
    if best_formation is None:
        raise ValueError("No valid formation could be selected")
    
    logger.info(f"Auto-selected formation: {best_formation} with {best_total_xp:.2f} total xP")
    return best_xi, best_bench, best_formation


def pick_starting_xi(squad_df: pd.DataFrame, formation: str, strategy_opts: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Pick the starting XI based on formation and strategy."""
    if formation == 'auto':
        return _pick_auto_formation(squad_df)
    else:
        xi_df, bench_df = _pick_fixed_formation(squad_df, formation)
        return xi_df, bench_df, formation


def _pick_fixed_formation(squad_df: pd.DataFrame, formation: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pick starting XI for a specific formation."""
    if formation not in VALID_FORMATIONS:
        raise ValueError(f"Invalid formation: {formation}. Valid options: {list(VALID_FORMATIONS.keys())}")
    
    position_quota = VALID_FORMATIONS[formation]
    xi_players = []
    remaining_players = squad_df.copy()
    
    # Sort by xP descending, then by player_name ascending for tie-breaking
    remaining_players = remaining_players.sort_values(['xP', 'player_name'], ascending=[False, True])
    
    # Pick players for each position according to quota
    for position, quota in position_quota.items():
        position_players = remaining_players[remaining_players['position'] == position]
        
        if len(position_players) < quota:
            raise ValueError(f"Not enough {position} players available. Need {quota}, have {len(position_players)}")
        
        # Take top quota players by xP
        selected = position_players.head(quota)
        xi_players.append(selected)
        
        # Remove selected players from remaining pool
        remaining_players = remaining_players[~remaining_players['player_id'].isin(selected['player_id'])]
    
    # Combine all selected players
    xi_df = pd.concat(xi_players, ignore_index=True)
    xi_df = xi_df.sort_values(['xP', 'player_name'], ascending=[False, True])
    
    # Add order column (1-11)
    xi_df['order'] = range(1, 12)
    
    # Remaining players become bench
    bench_df = remaining_players.copy()
    
    return xi_df, bench_df


def choose_captains(xi_df: pd.DataFrame, captain_strategy: str) -> pd.DataFrame:
    """Choose captain and vice-captain based on strategy."""
    xi_df = xi_df.copy()
    
    # Initialize captain/vice columns
    xi_df['is_captain'] = 0
    xi_df['is_vice'] = 0
    
    if captain_strategy == 'risk_adjusted':
        # Check if we have consistency data
        if 'consistency_g6' in xi_df.columns and not xi_df['consistency_g6'].isna().all():
            # Use consistency-based risk score
            xi_df['risk_score'] = xi_df['xP'] * (0.5 + 0.5 * xi_df['consistency_g6'].fillna(0))
            xi_df = xi_df.sort_values('risk_score', ascending=False)
            logger.info("Using consistency-based risk adjustment for captain selection")
        elif 'xp_rolling_mean_g3' in xi_df.columns and not xi_df['xp_rolling_mean_g3'].isna().all():
            # Use rolling mean as risk score
            xi_df['risk_score'] = xi_df['xp_rolling_mean_g3'].fillna(xi_df['xP'])
            xi_df = xi_df.sort_values('risk_score', ascending=False)
            logger.info("Using rolling mean for captain selection")
        else:
            # Fall back to top xP
            xi_df = xi_df.sort_values(['xP', 'player_name'], ascending=[False, True])
            logger.info("Falling back to top xP for captain selection")
    else:
        # Default: top xP strategy
        xi_df = xi_df.sort_values(['xP', 'player_name'], ascending=[False, True])
    
    # Assign captain and vice
    xi_df.loc[xi_df.index[0], 'is_captain'] = 1
    xi_df.loc[xi_df.index[1], 'is_vice'] = 1
    
    # Re-sort by original order
    xi_df = xi_df.sort_values('order')
    
    return xi_df


def order_bench(bench_df: pd.DataFrame, bench_strategy: str, keep_gk_last: bool) -> pd.DataFrame:
    """Order the bench players."""
    bench_df = bench_df.copy()
    
    if bench_strategy == 'low_minutes_first' and 'minutes_flag_available' in bench_df.columns:
        # Put low availability players first
        bench_df['low_minutes'] = bench_df['minutes_flag_available'] == 0
        bench_df = bench_df.sort_values(['low_minutes', 'xP', 'player_name'], ascending=[False, False, True])
        logger.info("Using low minutes first bench strategy")
    else:
        # Default: sort by xP descending, then player_name ascending
        bench_df = bench_df.sort_values(['xP', 'player_name'], ascending=[False, True])
    
    # Ensure GK is last if keep_gk_last is True
    if keep_gk_last:
        gk_players = bench_df[bench_df['position'] == 'GK']
        outfield_players = bench_df[bench_df['position'] != 'GK']
        
        if len(gk_players) > 0:
            # Reorder: outfield first, then GK
            bench_df = pd.concat([outfield_players, gk_players], ignore_index=True)
    
    # Add order column (1-4)
    bench_df['order'] = range(1, 5)
    
    return bench_df


def write_outputs(xi_df: pd.DataFrame, bench_df: pd.DataFrame, formation: str, output_dir: str):
    """Write the output CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Prepare starting XI output
    xi_output_cols = ['order', 'player_id', 'player_name', 'team_name', 'position', 'price', 'xP', 'is_captain', 'is_vice']
    xi_output = xi_df[xi_output_cols].copy()
    xi_output['formation'] = formation
    
    # Write starting XI
    xi_path = output_path / 'fpl_starting_xi.csv'
    xi_output.to_csv(xi_path, index=False)
    logger.info(f"Starting XI saved to {xi_path}")
    
    # Prepare bench output
    bench_output_cols = ['order', 'player_id', 'player_name', 'team_name', 'position', 'price', 'xP']
    bench_output = bench_df[bench_output_cols].copy()
    
    # Write bench
    bench_path = output_path / 'fpl_bench.csv'
    bench_output.to_csv(bench_path, index=False)
    logger.info(f"Bench saved to {bench_path}")


def validate_outputs(xi_df: pd.DataFrame, bench_df: pd.DataFrame, formation: str) -> bool:
    """Validate that the outputs meet all requirements."""
    # Check exactly 11 starters and 4 bench
    if len(xi_df) != 11:
        logger.error(f"Expected 11 starters, got {len(xi_df)}")
        return False
    
    if len(bench_df) != 4:
        logger.error(f"Expected 4 bench players, got {len(bench_df)}")
        return False
    
    # Check formation quotas
    position_quota = VALID_FORMATIONS[formation]
    for position, quota in position_quota.items():
        count = len(xi_df[xi_df['position'] == position])
        if count != quota:
            logger.error(f"Formation {formation} requires {quota} {position}, got {count}")
            return False
    
    # Check exactly 1 GK in starters
    gk_count = len(xi_df[xi_df['position'] == 'GK'])
    if gk_count != 1:
        logger.error(f"Expected 1 GK in starters, got {gk_count}")
        return False
    
    # Check no duplicates between XI and bench
    xi_ids = set(xi_df['player_id'])
    bench_ids = set(bench_df['player_id'])
    if xi_ids.intersection(bench_ids):
        logger.error("Duplicate players found between XI and bench")
        return False
    
    # Check captain and vice are assigned
    if xi_df['is_captain'].sum() != 1:
        logger.error("Exactly one captain must be assigned")
        return False
    
    if xi_df['is_vice'].sum() != 1:
        logger.error("Exactly one vice-captain must be assigned")
        return False
    
    # Check captain and vice are different players
    captain_idx = xi_df[xi_df['is_captain'] == 1].index[0]
    vice_idx = xi_df[xi_df['is_vice'] == 1].index[0]
    if captain_idx == vice_idx:
        logger.error("Captain and vice-captain must be different players")
        return False
    
    logger.info("Output validation passed")
    return True


def print_summary(xi_df: pd.DataFrame, bench_df: pd.DataFrame, formation: str):
    """Print console summary."""
    total_xi_xp = xi_df['xP'].sum()
    total_squad_xp = total_xi_xp + bench_df['xP'].sum()
    
    # Get captain and vice info
    captain = xi_df[xi_df['is_captain'] == 1].iloc[0]
    vice = xi_df[xi_df['is_vice'] == 1].iloc[0]
    
    print("\n" + "="*60)
    print("FPL ROTATION SUMMARY")
    print("="*60)
    print(f"Formation: {formation}")
    print(f"Captain: {captain['player_name']} ({captain['position']} - {captain['team_name']}) - {captain['xP']:.2f} xP")
    print(f"Vice: {vice['player_name']} ({vice['position']} - {vice['team_name']}) - {vice['xP']:.2f} xP")
    print(f"Starters Total xP: {total_xi_xp:.2f}")
    print(f"Full Squad Total xP: {total_squad_xp:.2f}")
    
    # Position breakdown
    pos_counts = xi_df['position'].value_counts()
    print(f"\nFormation Breakdown:")
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        count = pos_counts.get(pos, 0)
        print(f"  {pos}: {count}")
    
    # Club breakdown for starters
    club_counts = xi_df['team_name'].value_counts()
    print(f"\nClub Breakdown (Starters):")
    for club, count in club_counts.items():
        status = " (3+ PLAYERS)" if count >= 3 else ""
        print(f"  {club}: {count} players{status}")
    
    # Bench order
    print(f"\nBench Order:")
    for _, player in bench_df.iterrows():
        print(f"  Bench{player['order']}: {player['player_name']} ({player['position']} - {player['team_name']}) - {player['xP']:.2f} xP")
    
    print("="*60)


def main():
    """Main function to run the FPL rotation script."""
    parser = argparse.ArgumentParser(description='FPL Squad Rotation (Step 4b)')
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
    parser.add_argument('--squad-path', default='outputs/fpl_squad.csv',
                       help='Path to squad CSV (default: outputs/fpl_squad.csv)')
    parser.add_argument('--features-path', 
                       help='Optional path to features parquet for advanced strategies')
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory (default: outputs)')
    
    args = parser.parse_args()
    
    # Load squad
    squad_df = load_squad(args.squad_path)
    
    # Merge features if provided
    squad_df = maybe_merge_features(squad_df, args.features_path)
    
    # Pick starting XI and bench
    try:
        xi_df, bench_df, formation = pick_starting_xi(squad_df, args.formation, {})
    except Exception as e:
        logger.error(f"Failed to pick starting XI: {e}")
        sys.exit(1)
    
    # Choose captain and vice
    xi_df = choose_captains(xi_df, args.captain_strategy)
    
    # Order bench
    bench_df = order_bench(bench_df, args.bench_strategy, args.bench_keep_gk_last)
    
    # Validate outputs
    if not validate_outputs(xi_df, bench_df, formation):
        logger.error("Output validation failed")
        sys.exit(1)
    
    # Write outputs
    write_outputs(xi_df, bench_df, formation, args.output_dir)
    
    # Print summary
    print_summary(xi_df, bench_df, formation)


if __name__ == "__main__":
    main() 