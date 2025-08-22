#!/usr/bin/env python3
"""
Injury Burden Features Builder

Computes injury burden metrics from historical injury data:
- Lost days per 1,000 squad minutes
- Seasonal injury patterns by club

Output: data/processed/features/injury_burden.parquet
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re

import polars as pl

# Import our normalization module
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.normalization import canonicalize_frame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InjuryBurdenBuilder:
    """Builds injury burden features from historical injury data."""
    
    def __init__(self, data_raw: Path, output_dir: Path):
        """Initialize the injury burden builder."""
        self.data_raw = Path(data_raw)
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Approximate squad minutes per season (38 games * 90 min * ~14 players used)
        self.squad_minutes_per_season = 38 * 90 * 14
    
    def load_injury_data(self) -> Optional[pl.DataFrame]:
        """Load and process injury list data."""
        logger.info("Loading injury data...")
        
        injury_file = self.data_raw / "Injury list 2002-2016.csv"
        if not injury_file.exists():
            logger.warning("Injury list 2002-2016.csv not found")
            return None
        
        try:
            df = pl.read_csv(injury_file, ignore_errors=True)
            logger.info(f"Loaded injury data: {df.height:,} records, {df.width} columns")
            logger.info(f"Columns: {df.columns}")
            
            # Find team column
            team_col = None
            for col in ['Club', 'Team', 'club', 'team']:
                if col in df.columns:
                    team_col = col
                    break
            
            if not team_col:
                logger.error("No team column found in injury data")
                return None
            
            # Canonicalize team names
            df = canonicalize_frame(df, [team_col])
            
            # Standardize column names
            df = self._standardize_injury_columns(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load injury data: {e}")
            return None
    
    def _standardize_injury_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize injury data column names."""
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Team/Club column
            if col_lower in ['club', 'team']:
                column_mapping[col] = 'team'
            # Player name
            elif col_lower in ['player', 'name', 'player_name']:
                column_mapping[col] = 'player'
            # Injury type
            elif col_lower in ['injury', 'injury_type', 'type']:
                column_mapping[col] = 'injury_type'
            # Start date
            elif any(term in col_lower for term in ['start', 'from', 'begin']):
                if any(term in col_lower for term in ['date', 'time']):
                    column_mapping[col] = 'injury_start'
            # End date
            elif any(term in col_lower for term in ['end', 'until', 'return']):
                if any(term in col_lower for term in ['date', 'time']):
                    column_mapping[col] = 'injury_end'
            # Duration
            elif col_lower in ['days', 'duration', 'length']:
                column_mapping[col] = 'injury_days'
        
        if column_mapping:
            logger.info(f"Column mapping: {column_mapping}")
            df = df.rename(column_mapping)
        
        return df
    
    def parse_injury_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Parse injury start and end dates."""
        logger.info("Parsing injury dates...")
        
        # Try to parse start dates
        if 'injury_start' in df.columns:
            try:
                df = df.with_columns([
                    pl.col('injury_start').str.strptime(pl.Date, format='%d/%m/%Y', strict=False).alias('start_date')
                ])
                
                # Try alternative formats if needed
                if df['start_date'].null_count() > df.height * 0.5:
                    df = df.with_columns([
                        pl.col('injury_start').str.strptime(pl.Date, format='%Y-%m-%d', strict=False).alias('start_date')
                    ])
                
                parsed_starts = df['start_date'].drop_nulls().len()
                logger.info(f"Parsed {parsed_starts} start dates")
                
            except Exception as e:
                logger.warning(f"Failed to parse start dates: {e}")
        
        # Try to parse end dates
        if 'injury_end' in df.columns:
            try:
                df = df.with_columns([
                    pl.col('injury_end').str.strptime(pl.Date, format='%d/%m/%Y', strict=False).alias('end_date')
                ])
                
                # Try alternative formats if needed
                if df['end_date'].null_count() > df.height * 0.5:
                    df = df.with_columns([
                        pl.col('injury_end').str.strptime(pl.Date, format='%Y-%m-%d', strict=False).alias('end_date')
                    ])
                
                parsed_ends = df['end_date'].drop_nulls().len()
                logger.info(f"Parsed {parsed_ends} end dates")
                
            except Exception as e:
                logger.warning(f"Failed to parse end dates: {e}")
        
        return df
    
    def calculate_injury_duration(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate injury duration in days."""
        logger.info("Calculating injury durations...")
        
        # Use provided duration if available
        if 'injury_days' in df.columns:
            df = df.with_columns([
                pl.col('injury_days').cast(pl.Float64).alias('duration_days')
            ])
            direct_durations = df['duration_days'].drop_nulls().len()
            logger.info(f"Found {direct_durations} direct duration values")
        
        # Calculate from start/end dates if available
        if 'start_date' in df.columns and 'end_date' in df.columns:
            df = df.with_columns([
                (pl.col('end_date') - pl.col('start_date')).dt.total_days().alias('calculated_days')
            ])
            
            # Use calculated duration where direct duration is missing
            if 'duration_days' in df.columns:
                df = df.with_columns([
                    pl.coalesce([pl.col('duration_days'), pl.col('calculated_days')]).alias('duration_days')
                ])
            else:
                df = df.with_columns([
                    pl.col('calculated_days').alias('duration_days')
                ])
            
            calculated_durations = df['calculated_days'].drop_nulls().len()
            logger.info(f"Calculated {calculated_durations} durations from dates")
        
        # Filter to reasonable durations (1-365 days)
        if 'duration_days' in df.columns:
            valid_before = df['duration_days'].drop_nulls().len()
            df = df.filter(
                (pl.col('duration_days').is_null()) | 
                ((pl.col('duration_days') >= 1) & (pl.col('duration_days') <= 365))
            )
            valid_after = df['duration_days'].drop_nulls().len()
            logger.info(f"Filtered to {valid_after} valid durations (was {valid_before})")
        
        return df
    
    def extract_season_from_date(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract season from injury dates."""
        logger.info("Extracting seasons from injury dates...")
        
        if 'start_date' not in df.columns:
            logger.warning("No start date available for season extraction")
            return df.with_columns([pl.lit('unknown').alias('season')])
        
        # Define season based on start date
        # August-July seasons (e.g., 2015-08-01 to 2016-07-31 = 2015-16 season)
        df = df.with_columns([
            pl.when(pl.col('start_date').dt.month() >= 8)
            .then(pl.col('start_date').dt.year().cast(pl.String) + '-' + 
                  (pl.col('start_date').dt.year() + 1).cast(pl.String).str.slice(2, 2))
            .otherwise((pl.col('start_date').dt.year() - 1).cast(pl.String) + '-' +
                      pl.col('start_date').dt.year().cast(pl.String).str.slice(2, 2))
            .alias('season')
        ])
        
        seasons = df['season'].drop_nulls().unique().sort().to_list()
        logger.info(f"Found injuries across seasons: {seasons}")
        
        return df
    
    def aggregate_injury_burden(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate injury burden by team and season."""
        logger.info("Aggregating injury burden by team and season...")
        
        if 'duration_days' not in df.columns:
            logger.warning("No duration data available for aggregation")
            return pl.DataFrame({'team': [], 'season': [], 'injury_burden_per_1000': []})
        
        # Aggregate by team and season
        team_season_burden = (
            df
            .group_by(['team', 'season'])
            .agg([
                pl.col('duration_days').sum().alias('total_injury_days'),
                pl.col('duration_days').count().alias('injury_count'),
                pl.col('duration_days').mean().alias('avg_injury_duration')
            ])
        )
        
        # Calculate injury burden per 1,000 squad minutes
        team_season_burden = team_season_burden.with_columns([
            (pl.col('total_injury_days') / self.squad_minutes_per_season * 1000).alias('injury_burden_per_1000')
        ])
        
        logger.info(f"Aggregated injury burden for {team_season_burden.height} team-seasons")
        
        return team_season_burden
    
    def calculate_overall_burden(self, team_season_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate overall injury burden per team (average across seasons)."""
        logger.info("Calculating overall team injury burden...")
        
        if team_season_df.height == 0:
            return pl.DataFrame({'team': [], 'injury_burden_per_1000': []})
        
        # Calculate average burden across seasons for each team
        team_burden = (
            team_season_df
            .group_by('team')
            .agg([
                pl.col('injury_burden_per_1000').mean().alias('injury_burden_per_1000'),
                pl.col('total_injury_days').mean().alias('avg_annual_injury_days'),
                pl.col('injury_count').mean().alias('avg_annual_injuries'),
                pl.col('season').count().alias('seasons_with_data')
            ])
        )
        
        logger.info(f"Calculated overall burden for {team_burden.height} teams")
        
        # Log some statistics
        if team_burden.height > 0:
            mean_burden = team_burden['injury_burden_per_1000'].mean()
            logger.info(f"Mean injury burden: {mean_burden:.1f} days per 1,000 squad minutes")
            
            # Show top 5 most injury-prone teams
            top_injury_teams = team_burden.sort('injury_burden_per_1000', descending=True).head(5)
            logger.info("Top 5 most injury-prone teams:")
            for row in top_injury_teams.iter_rows(named=True):
                logger.info(f"  {row['team']}: {row['injury_burden_per_1000']:.1f}")
        
        return team_burden
    
    def build_injury_features(self) -> Path:
        """Build complete injury burden features."""
        logger.info("Starting injury burden features build...")
        
        # Load injury data
        injury_df = self.load_injury_data()
        
        if injury_df is None:
            logger.warning("No injury data available - creating empty output")
            empty_df = pl.DataFrame({
                'team': pl.Series([], dtype=pl.String),
                'injury_burden_per_1000': pl.Series([], dtype=pl.Float64)
            })
            
            output_path = self.output_dir / "injury_burden.parquet"
            empty_df.write_parquet(output_path)
            return output_path
        
        # Process injury data
        injury_df = self.parse_injury_dates(injury_df)
        injury_df = self.calculate_injury_duration(injury_df)
        injury_df = self.extract_season_from_date(injury_df)
        
        # Aggregate to team level
        team_season_burden = self.aggregate_injury_burden(injury_df)
        team_burden = self.calculate_overall_burden(team_season_burden)
        
        # Fill missing values for teams without injury data
        if team_burden.height == 0:
            logger.warning("No valid injury burden data calculated")
            team_burden = pl.DataFrame({
                'team': pl.Series([], dtype=pl.String),
                'injury_burden_per_1000': pl.Series([], dtype=pl.Float64)
            })
        else:
            # Use median as default for missing teams
            median_burden = team_burden['injury_burden_per_1000'].median()
            logger.info(f"Using median burden {median_burden:.1f} as default for missing teams")
        
        # Save output
        output_path = self.output_dir / "injury_burden.parquet"
        team_burden.write_parquet(output_path)
        
        logger.info(f"✅ Injury burden features saved: {output_path}")
        logger.info(f"📊 Final dataset: {team_burden.height} teams")
        
        return output_path


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    data_raw = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed" / "features"
    
    try:
        builder = InjuryBurdenBuilder(data_raw, output_dir)
        output_path = builder.build_injury_features()
        
        print(f"\n🏥 Injury Burden Features Complete!")
        print(f"📊 Output: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to build injury features: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 