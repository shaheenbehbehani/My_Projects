#!/usr/bin/env python3
"""
Simulate Premier League 2025/26 season using calibrated prediction model.

This script uses the calibrated model to generate match outcome probabilities
for each fixture and runs Monte Carlo simulations to calculate expected points,
title probability, and top-4 probability for each team.

Input: models/match_model_calibrated.pkl, data/processed/fixtures_2025_26.parquet
Outputs: outputs/sim_summary_2025_26.parquet, outputs/sim_summary_2025_26.csv
"""

import polars as pl
import numpy as np
import pandas as pd
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .utils import set_random_seed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
set_random_seed(42)


class SeasonSimulator:
    """Simulate Premier League season using calibrated prediction model."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models", outputs_dir: str = "outputs"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.outputs_dir = Path(outputs_dir)
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulation parameters
        self.feature_exclusions = ['date', 'season', 'home_team', 'away_team', 'y', 'result_label']
        
    def load_calibrated_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Load the calibrated model and feature information."""
        import joblib
        
        model_path = self.models_dir / "match_model_calibrated.pkl"
        feature_path = self.models_dir / "feature_info.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Calibrated model not found: {model_path}")
        
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature info not found: {feature_path}")
        
        # Load model
        model = joblib.load(model_path)
        logger.info(f"Loaded calibrated model: {type(model).__name__}")
        
        # Load feature info
        feature_info = joblib.load(feature_path)
        logger.info(f"Loaded feature info: {feature_info['n_features']} features")
        
        return model, feature_info
    
    def load_fixtures(self) -> pl.DataFrame:
        """Load the 2025/26 fixtures."""
        fixtures_path = self.processed_dir / "fixtures_2025_26.parquet"
        
        if not fixtures_path.exists():
            raise FileNotFoundError(f"Fixtures not found: {fixtures_path}")
        
        logger.info(f"Loading fixtures from {fixtures_path}")
        df = pl.read_parquet(fixtures_path)
        logger.info(f"Loaded {df.height:,} fixtures")
        
        return df
    
    def load_match_dataset(self) -> pl.DataFrame:
        """Load the match dataset to compute features for fixtures."""
        dataset_path = self.processed_dir / "match_dataset.parquet"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Match dataset not found: {dataset_path}")
        
        logger.info(f"Loading match dataset from {dataset_path}")
        df = pl.read_parquet(dataset_path)
        logger.info(f"Loaded {df.height:,} matches")
        
        return df
    
    def compute_fixture_features(self, fixtures: pl.DataFrame, 
                               historical_matches: pl.DataFrame) -> pl.DataFrame:
        """Compute features for each fixture using historical data."""
        logger.info("Computing features for fixtures...")
        
        # Get the latest Elo ratings and form features from historical data
        latest_data = historical_matches.sort('date').group_by('home_team').tail(1)
        
        # Create a lookup table for team features
        team_features = {}
        
        for row in latest_data.iter_rows(named=True):
            team = row['home_team']
            team_features[team] = {
                'elo': row['home_elo'],
                'last5_pts': row['home_last5_pts'],
                'goals_for_avg': row['home_goals_for_avg'],
                'goals_against_avg': row['home_goals_against_avg']
            }
        
        # Also get away team features
        away_latest = historical_matches.sort('date').group_by('away_team').tail(1)
        for row in away_latest.iter_rows(named=True):
            team = row['away_team']
            if team not in team_features:
                team_features[team] = {
                    'elo': row['away_elo'],
                    'last5_pts': row['away_last5_pts'],
                    'goals_for_avg': row['away_goals_for_avg'],
                    'goals_against_avg': row['away_goals_against_avg']
                }
        
        # Add features to fixtures
        fixtures_with_features = []
        
        for fixture in fixtures.iter_rows(named=True):
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            
            # Get team features
            home_features = team_features.get(home_team, {})
            away_features = team_features.get(away_team, {})
            
            # Create feature row
            feature_row = {
                'match_id': fixture['match_id'],
                'date': fixture['date'],
                'matchweek': fixture['matchweek'],
                'home_team': home_team,
                'away_team': away_team,
                'home_elo': home_features.get('elo', 1500),
                'away_elo': away_features.get('elo', 1500),
                'home_last5_pts': home_features.get('last5_pts', 0),
                'away_last5_pts': away_features.get('last5_pts', 0),
                'home_goals_for_avg': home_features.get('goals_for_avg', 0),
                'home_goals_against_avg': home_features.get('goals_against_avg', 0),
                'away_goals_for_avg': away_features.get('goals_for_avg', 0),
                'away_goals_against_avg': away_features.get('goals_against_avg', 0)
            }
            
            fixtures_with_features.append(feature_row)
        
        # Convert to DataFrame
        fixtures_df = pl.DataFrame(fixtures_with_features)
        
        logger.info(f"Computed features for {fixtures_df.height} fixtures")
        return fixtures_df
    
    def predict_match_outcomes(self, model: Any, fixtures: pl.DataFrame, 
                              feature_names: List[str]) -> np.ndarray:
        """Predict match outcome probabilities for all fixtures."""
        logger.info("Predicting match outcomes...")
        
        # Prepare features
        feature_df = fixtures.select(feature_names)
        X = feature_df.to_pandas()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X)
        
        logger.info(f"Generated predictions for {len(y_pred_proba)} fixtures")
        return y_pred_proba
    
    def run_monte_carlo_simulations(self, fixtures: pl.DataFrame, 
                                  match_probabilities: np.ndarray, 
                                  n_simulations: int = 10000) -> List[Dict[str, Any]]:
        """Run Monte Carlo simulations to generate season outcomes."""
        logger.info(f"Running {n_simulations:,} Monte Carlo simulations...")
        
        simulation_results = []
        
        for sim in range(n_simulations):
            if sim % 1000 == 0:
                logger.info(f"  Simulation {sim:,}/{n_simulations:,}")
            
            # Simulate season
            season_outcome = self._simulate_single_season(fixtures, match_probabilities)
            simulation_results.append(season_outcome)
        
        logger.info("Monte Carlo simulations complete")
        return simulation_results
    
    def _simulate_single_season(self, fixtures: pl.DataFrame, 
                               match_probabilities: np.ndarray) -> Dict[str, Any]:
        """Simulate a single season and return final standings."""
        # Initialize team points
        team_points = {}
        
        # Simulate each match
        for i, fixture in enumerate(fixtures.iter_rows(named=True)):
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            
            # Get match probabilities
            probs = match_probabilities[i]
            
            # Sample outcome based on probabilities
            outcome = np.random.choice(3, p=probs)  # 0=Home, 1=Draw, 2=Away
            
            # Initialize team points if needed
            if home_team not in team_points:
                team_points[home_team] = 0
            if away_team not in team_points:
                team_points[away_team] = 0
            
            # Award points based on outcome
            if outcome == 0:  # Home win
                team_points[home_team] += 3
            elif outcome == 2:  # Away win
                team_points[away_team] += 3
            else:  # Draw
                team_points[home_team] += 1
                team_points[away_team] += 1
        
        # Create standings
        standings = [
            {'team': team, 'points': points}
            for team, points in team_points.items()
        ]
        
        # Sort by points (descending)
        standings.sort(key=lambda x: x['points'], reverse=True)
        
        # Add positions
        for i, team_standing in enumerate(standings):
            team_standing['position'] = i + 1
        
        return {
            'simulation_id': len(simulation_results) if 'simulation_results' in locals() else 0,
            'standings': standings
        }
    
    def analyze_simulation_results(self, simulation_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyze Monte Carlo simulation results."""
        logger.info("Analyzing simulation results...")
        
        # Initialize team statistics
        team_stats = {}
        
        # Process each simulation
        for sim_result in simulation_results:
            standings = sim_result['standings']
            
            for team_standing in standings:
                team = team_standing['team']
                position = team_standing['position']
                points = team_standing['points']
                
                if team not in team_stats:
                    team_stats[team] = {
                        'total_points': 0,
                        'position_counts': {},
                        'title_wins': 0,
                        'top4_finishes': 0,
                        'simulations': 0
                    }
                
                # Update statistics
                team_stats[team]['total_points'] += points
                team_stats[team]['simulations'] += 1
                
                # Update position counts
                if position not in team_stats[team]['position_counts']:
                    team_stats[team]['position_counts'][position] = 0
                team_stats[team]['position_counts'][position] += 1
                
                # Update title and top-4 counts
                if position == 1:
                    team_stats[team]['title_wins'] += 1
                if position <= 4:
                    team_stats[team]['top4_finishes'] += 1
        
        # Calculate final statistics
        summary_data = []
        n_simulations = len(simulation_results)
        
        for team, stats in team_stats.items():
            expected_points = stats['total_points'] / stats['simulations']
            title_probability = stats['title_wins'] / n_simulations
            top4_probability = stats['top4_finishes'] / n_simulations
            
            # Most common position
            most_common_position = max(stats['position_counts'].items(), 
                                     key=lambda x: x[1])[0]
            
            summary_data.append({
                'team': team,
                'expected_points': expected_points,
                'title_probability': title_probability,
                'top4_probability': top4_probability,
                'most_common_position': most_common_position,
                'title_wins': stats['title_wins'],
                'top4_finishes': stats['top4_finishes'],
                'simulations': stats['simulations']
            })
        
        # Convert to DataFrame and sort by expected points
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('expected_points', ascending=False).reset_index(drop=True)
        
        # Add final rank
        summary_df['rank'] = range(1, len(summary_df) + 1)
        
        logger.info(f"Analysis complete for {len(summary_df)} teams")
        return summary_df
    
    def save_simulation_results(self, summary_df: pd.DataFrame) -> Tuple[Path, Path]:
        """Save simulation results to parquet and CSV files."""
        logger.info("Saving simulation results...")
        
        # Save to parquet
        parquet_path = self.outputs_dir / "sim_summary_2025_26.parquet"
        summary_df.to_parquet(parquet_path, index=False)
        logger.info(f"Simulation summary saved to {parquet_path}")
        
        # Save to CSV
        csv_path = self.outputs_dir / "sim_summary_2025_26.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Simulation summary saved to {csv_path}")
        
        return parquet_path, csv_path
    
    def print_simulation_summary(self, summary_df: pd.DataFrame):
        """Print a summary of simulation results to console."""
        print(f"\n🏆 Premier League 2025/26 Season Simulation Results")
        print(f"=" * 60)
        
        # Top 5 teams
        print(f"\n📊 Top 5 Teams (by Expected Points):")
        print(f"{'Rank':<4} {'Team':<20} {'Exp Points':<12} {'Title %':<8} {'Top 4 %':<8}")
        print(f"-" * 60)
        
        for _, row in summary_df.head().iterrows():
            print(f"{row['rank']:<4} {row['team']:<20} {row['expected_points']:<12.1f} "
                  f"{row['title_probability']:<8.1%} {row['top4_probability']:<8.1%}")
        
        # Title favorites
        title_favorites = summary_df[summary_df['title_probability'] > 0.05].head(3)
        if not title_favorites.empty:
            print(f"\n👑 Title Favorites:")
            for _, row in title_favorites.iterrows():
                print(f"   {row['team']}: {row['title_probability']:.1%}")
        
        # Top 4 contenders
        top4_contenders = summary_df[summary_df['top4_probability'] > 0.3].head(6)
        if not top4_contenders.empty:
            print(f"\n🔝 Top 4 Contenders:")
            for _, row in top4_contenders.iterrows():
                print(f"   {row['team']}: {row['top4_probability']:.1%}")
        
        print(f"\n📁 Results saved to outputs/sim_summary_2025_26.parquet and .csv")
    
    def run_simulation(self, n_simulations: int = 10000) -> pd.DataFrame:
        """Run the complete season simulation pipeline."""
        logger.info("🚀 Starting season simulation pipeline...")
        
        # Load calibrated model
        model, feature_info = self.load_calibrated_model()
        
        # Load fixtures
        fixtures = self.load_fixtures()
        
        # Load historical data for feature computation
        historical_matches = self.load_match_dataset()
        
        # Compute features for fixtures
        fixtures_with_features = self.compute_fixture_features(fixtures, historical_matches)
        
        # Predict match outcomes
        match_probabilities = self.predict_match_outcomes(
            model, fixtures_with_features, feature_info['feature_names']
        )
        
        # Run Monte Carlo simulations
        simulation_results = self.run_monte_carlo_simulations(
            fixtures_with_features, match_probabilities, n_simulations
        )
        
        # Analyze results
        summary_df = self.analyze_simulation_results(simulation_results)
        
        # Save results
        parquet_path, csv_path = self.save_simulation_results(summary_df)
        
        # Print summary
        self.print_simulation_summary(summary_df)
        
        logger.info("✅ Season simulation pipeline completed successfully!")
        return summary_df


def main():
    """Main entry point for season simulation."""
    parser = argparse.ArgumentParser(description="Simulate Premier League 2025/26 season")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--models-dir", default="models", help="Models directory path")
    parser.add_argument("--outputs-dir", default="outputs", help="Outputs directory path")
    parser.add_argument("--n-sims", type=int, default=10000, help="Number of Monte Carlo simulations")
    
    args = parser.parse_args()
    
    try:
        # Run simulation
        simulator = SeasonSimulator(args.data_dir, args.models_dir, args.outputs_dir)
        summary_df = simulator.run_simulation(args.n_sims)
        
        print(f"\n✅ Season simulation complete!")
        print(f"📊 Simulated {args.n_sims:,} seasons")
        print(f"🏆 {len(summary_df)} teams analyzed")
        
    except Exception as e:
        logger.error(f"Failed to run simulation: {e}")
        raise


if __name__ == "__main__":
    main() 