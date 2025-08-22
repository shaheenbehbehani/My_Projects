#!/usr/bin/env python3
"""
Bookmaker Priors Features Builder

Computes pre-season club-level priors for title probability:
- From outright odds (if available)
- Heuristic fallback using Elo and market value
- Prevents data leakage by using only pre-season information

Output: data/processed/features/bookmaker_priors_2025_26.parquet
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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


class BookmakerPriorsBuilder:
    """Builds bookmaker prior probability features."""
    
    def __init__(self, data_raw: Path, features_dir: Path, output_dir: Path):
        """Initialize the bookmaker priors builder."""
        self.data_raw = Path(data_raw)
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_outright_odds(self) -> Optional[pl.DataFrame]:
        """Load outright title odds if available."""
        logger.info("Looking for outright odds data...")
        
        # Look for various outright odds files
        odds_files = [
            self.data_raw / "outright_odds_2025_26.csv",
            self.data_raw / "title_odds.csv",
            self.data_raw / "season_odds.csv"
        ]
        
        for odds_file in odds_files:
            if odds_file.exists():
                try:
                    df = pl.read_csv(odds_file, ignore_errors=True)
                    logger.info(f"Found outright odds: {odds_file.name} ({df.height} teams)")
                    
                    # Find team column
                    team_col = None
                    for col in ['Team', 'Club', 'team', 'club']:
                        if col in df.columns:
                            team_col = col
                            break
                    
                    if team_col:
                        # Canonicalize team names
                        df = canonicalize_frame(df, [team_col])
                        df = df.rename({team_col: 'team'})
                        return df
                    
                except Exception as e:
                    logger.warning(f"Failed to load {odds_file.name}: {e}")
        
        logger.info("No outright odds files found")
        return None
    
    def load_pre_season_elo(self) -> Optional[pl.DataFrame]:
        """Load pre-season Elo ratings."""
        logger.info("Loading pre-season Elo ratings...")
        
        elo_file = self.features_dir / "form_baselines_2025_26.parquet"
        if not elo_file.exists():
            logger.warning("Pre-season Elo baselines not found")
            return None
        
        try:
            df = pl.read_parquet(elo_file)
            logger.info(f"Loaded pre-season Elo for {df.height} teams")
            return df.select(['team', 'elo_baseline'])
        
        except Exception as e:
            logger.error(f"Failed to load pre-season Elo: {e}")
            return None
    
    def load_market_values(self) -> Optional[pl.DataFrame]:
        """Load market values for fallback calculation."""
        logger.info("Loading market values...")
        
        values_file = self.features_dir / "static_2025_26.parquet"
        if not values_file.exists():
            logger.warning("Static club features not found")
            return None
        
        try:
            df = pl.read_parquet(values_file)
            if 'market_value_eur' in df.columns:
                logger.info(f"Loaded market values for {df.height} teams")
                return df.select(['team', 'market_value_eur'])
            else:
                logger.warning("No market_value_eur column found")
                return None
        
        except Exception as e:
            logger.error(f"Failed to load market values: {e}")
            return None
    
    def convert_odds_to_probabilities(self, odds_df: pl.DataFrame) -> pl.DataFrame:
        """Convert decimal odds to implied probabilities with overround correction."""
        logger.info("Converting odds to implied probabilities...")
        
        # Find odds column
        odds_col = None
        for col in odds_df.columns:
            if col != 'team' and any(term in col.lower() for term in ['odds', 'price', 'decimal']):
                odds_col = col
                break
        
        if not odds_col:
            logger.error("No odds column found")
            return odds_df
        
        logger.info(f"Using odds column: {odds_col}")
        
        # Convert to implied probabilities
        odds_df = odds_df.with_columns([
            (1.0 / pl.col(odds_col)).alias('implied_prob')
        ])
        
        # Calculate overround (sum of implied probabilities)
        total_implied = odds_df['implied_prob'].sum()
        overround = total_implied
        
        logger.info(f"Market overround: {overround:.3f} ({(overround-1)*100:.1f}%)")
        
        # Correct for overround to get true probabilities
        odds_df = odds_df.with_columns([
            (pl.col('implied_prob') / overround).alias('prior_title_prob')
        ])
        
        # Verify probabilities sum to 1.0
        corrected_sum = odds_df['prior_title_prob'].sum()
        logger.info(f"Corrected probabilities sum: {corrected_sum:.6f}")
        
        return odds_df.select(['team', 'prior_title_prob'])
    
    def calculate_heuristic_priors(self, elo_df: pl.DataFrame, values_df: Optional[pl.DataFrame]) -> pl.DataFrame:
        """Calculate heuristic priors using Elo and market value."""
        logger.info("Calculating heuristic priors...")
        
        # Start with Elo-based probabilities
        if 'elo_baseline' not in elo_df.columns:
            logger.error("No Elo baseline column found")
            return pl.DataFrame({'team': [], 'prior_title_prob': []})
        
        # Convert Elo to winning probability using scaling
        # Higher-rated teams get exponentially higher probabilities
        base_df = elo_df.with_columns([
            pl.col('elo_baseline').alias('elo')
        ])
        
        # Calculate Elo-based strength (exponential scaling)
        elo_mean = base_df['elo'].mean()
        base_df = base_df.with_columns([
            ((pl.col('elo') - elo_mean) / 100).exp().alias('elo_strength')
        ])
        
        # If market values available, incorporate them
        if values_df is not None:
            logger.info("Incorporating market values into heuristic")
            
            base_df = base_df.join(values_df, on='team', how='left')
            
            # Fill missing market values with mean
            mean_value = base_df['market_value_eur'].mean()
            base_df = base_df.with_columns([
                pl.col('market_value_eur').fill_null(mean_value)
            ])
            
            # Calculate value-based strength (also exponential scaling)
            value_mean = base_df['market_value_eur'].mean()
            base_df = base_df.with_columns([
                ((pl.col('market_value_eur') - value_mean) / value_mean).exp().alias('value_strength')
            ])
            
            # Combine Elo and value (70% Elo, 30% value)
            base_df = base_df.with_columns([
                (0.7 * pl.col('elo_strength') + 0.3 * pl.col('value_strength')).alias('combined_strength')
            ])
            
        else:
            logger.info("Using Elo-only heuristic")
            base_df = base_df.with_columns([
                pl.col('elo_strength').alias('combined_strength')
            ])
        
        # Normalize to probabilities
        total_strength = base_df['combined_strength'].sum()
        base_df = base_df.with_columns([
            (pl.col('combined_strength') / total_strength).alias('prior_title_prob')
        ])
        
        # Verify probabilities sum to 1.0
        prob_sum = base_df['prior_title_prob'].sum()
        logger.info(f"Heuristic probabilities sum: {prob_sum:.6f}")
        
        # Log top 5 favorites
        top_teams = base_df.sort('prior_title_prob', descending=True).head(5)
        logger.info("Top 5 title favorites (heuristic):")
        for i, row in enumerate(top_teams.iter_rows(named=True)):
            logger.info(f"  {i+1}. {row['team']}: {row['prior_title_prob']:.3f}")
        
        return base_df.select(['team', 'prior_title_prob'])
    
    def load_fixtures_teams(self) -> Optional[List[str]]:
        """Load canonical team list from fixtures."""
        logger.info("Loading canonical team list from fixtures...")
        
        project_root = Path(self.data_raw).parent
        fixtures_file = project_root / "data" / "processed" / "fixtures_2025_26.parquet"
        if not fixtures_file.exists():
            logger.warning("Fixtures file not found")
            return None
        
        try:
            fixtures_df = pl.read_parquet(fixtures_file)
            
            # Get unique teams from home and away
            home_teams = fixtures_df['home_team'].unique().to_list()
            away_teams = fixtures_df['away_team'].unique().to_list()
            
            all_teams = list(set(home_teams + away_teams))
            logger.info(f"Found {len(all_teams)} teams in fixtures")
            
            return sorted(all_teams)
            
        except Exception as e:
            logger.error(f"Failed to load fixtures teams: {e}")
            return None
    
    def build_bookmaker_priors(self) -> Path:
        """Build complete bookmaker priors features."""
        logger.info("Starting bookmaker priors build...")
        
        # Get canonical team list
        fixtures_teams = self.load_fixtures_teams()
        
        # Try to load outright odds first
        odds_df = self.load_outright_odds()
        
        if odds_df is not None:
            logger.info("Using outright odds for priors")
            priors_df = self.convert_odds_to_probabilities(odds_df)
        else:
            logger.info("No outright odds available - using heuristic approach")
            
            # Load Elo and market value data
            elo_df = self.load_pre_season_elo()
            values_df = self.load_market_values()
            
            if elo_df is None:
                logger.error("No Elo data available for heuristic calculation")
                # Create minimal fallback
                if fixtures_teams:
                    equal_prob = 1.0 / len(fixtures_teams)
                    priors_df = pl.DataFrame({
                        'team': fixtures_teams,
                        'prior_title_prob': [equal_prob] * len(fixtures_teams)
                    })
                else:
                    priors_df = pl.DataFrame({'team': [], 'prior_title_prob': []})
            else:
                priors_df = self.calculate_heuristic_priors(elo_df, values_df)
        
        # Ensure all fixtures teams are included
        if fixtures_teams:
            missing_teams = set(fixtures_teams) - set(priors_df['team'].to_list())
            
            if missing_teams:
                logger.info(f"Adding {len(missing_teams)} missing teams with minimal probability")
                
                # Calculate minimal probability for missing teams
                existing_prob_sum = priors_df['prior_title_prob'].sum()
                remaining_prob = max(0.01, 1.0 - existing_prob_sum)
                missing_prob = remaining_prob / len(missing_teams)
                
                # Scale down existing probabilities slightly to make room
                if existing_prob_sum > 0.99:
                    scale_factor = 0.99 / existing_prob_sum
                    priors_df = priors_df.with_columns([
                        (pl.col('prior_title_prob') * scale_factor).alias('prior_title_prob')
                    ])
                
                # Add missing teams
                missing_df = pl.DataFrame({
                    'team': list(missing_teams),
                    'prior_title_prob': [missing_prob] * len(missing_teams)
                })
                
                priors_df = pl.concat([priors_df, missing_df])
        
        # Final normalization to ensure probabilities sum to 1.0
        total_prob = priors_df['prior_title_prob'].sum()
        if abs(total_prob - 1.0) > 0.001:
            logger.info(f"Renormalizing probabilities (sum was {total_prob:.6f})")
            priors_df = priors_df.with_columns([
                (pl.col('prior_title_prob') / total_prob).alias('prior_title_prob')
            ])
        
        # Save output
        output_path = self.output_dir / "bookmaker_priors_2025_26.parquet"
        priors_df.write_parquet(output_path)
        
        logger.info(f"✅ Bookmaker priors saved: {output_path}")
        logger.info(f"📊 Final dataset: {priors_df.height} teams")
        
        # Log final probabilities sum
        final_sum = priors_df['prior_title_prob'].sum()
        logger.info(f"Final probabilities sum: {final_sum:.6f}")
        
        # Log top 5 favorites
        top_teams = priors_df.sort('prior_title_prob', descending=True).head(5)
        logger.info("Top 5 title favorites:")
        for i, row in enumerate(top_teams.iter_rows(named=True)):
            logger.info(f"  {i+1}. {row['team']}: {row['prior_title_prob']:.3f} ({row['prior_title_prob']*100:.1f}%)")
        
        return output_path


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    data_raw = project_root / "data" / "raw"
    features_dir = project_root / "data" / "processed" / "features"
    output_dir = project_root / "data" / "processed" / "features"
    
    try:
        builder = BookmakerPriorsBuilder(data_raw, features_dir, output_dir)
        output_path = builder.build_bookmaker_priors()
        
        print(f"\n📊 Bookmaker Priors Features Complete!")
        print(f"📊 Output: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to build bookmaker priors: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 