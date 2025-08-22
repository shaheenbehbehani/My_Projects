#!/usr/bin/env python3
"""
Static Club Features Builder

Merges static club attributes for 2025/26 season:
- Market value (from Club Value.csv)
- Annual wages (from Club wages.csv) 
- Attendance data (from Attendance Data.csv)
- Manager tenure (from Premier League Managers.csv)

Output: data/processed/features/static_2025_26.parquet
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, date

import polars as pl

# Import our normalization module
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.normalization import canonicalize_frame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StaticClubBuilder:
    """Builds static club attribute features."""
    
    def __init__(self, data_raw: Path, output_dir: Path):
        """Initialize the static club builder."""
        self.data_raw = Path(data_raw)
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_currency_value(self, value_str) -> Optional[float]:
        """Extract numeric value from currency strings like €1.33bn, £500M."""
        if not isinstance(value_str, str) or not value_str.strip():
            return None
        
        # Clean the string
        cleaned = re.sub(r'[£€$¥¢₹₽,\s]', '', value_str.strip())
        
        # Handle billion/million/thousand suffixes
        multiplier = 1
        if cleaned.lower().endswith('bn'):
            multiplier = 1_000_000_000
            cleaned = cleaned[:-2]
        elif cleaned.lower().endswith('b'):
            multiplier = 1_000_000_000
            cleaned = cleaned[:-1]
        elif cleaned.lower().endswith('m'):
            multiplier = 1_000_000
            cleaned = cleaned[:-1]
        elif cleaned.lower().endswith('k'):
            multiplier = 1_000
            cleaned = cleaned[:-1]
        
        try:
            return float(cleaned) * multiplier
        except ValueError:
            return None
    
    def load_club_values(self) -> Optional[pl.DataFrame]:
        """Load and process club market values."""
        logger.info("Loading club market values...")
        
        value_file = self.data_raw / "Club Value.csv"
        if not value_file.exists():
            logger.warning("Club Value.csv not found")
            return None
        
        try:
            df = pl.read_csv(value_file, ignore_errors=True)
            logger.info(f"Loaded club values: {df.height} rows, {df.width} columns")
            
            # Find team column
            team_col = None
            for col in ['Club', 'Team', 'club', 'team']:
                if col in df.columns:
                    team_col = col
                    break
            
            if not team_col:
                logger.error("No team column found in Club Value.csv")
                return None
            
            # Canonicalize team names
            df = canonicalize_frame(df, [team_col])
            
            # Find value column
            value_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['value', 'worth', 'market']):
                    value_col = col
                    break
            
            if not value_col:
                logger.error("No value column found in Club Value.csv")
                return None
            
            logger.info(f"Using team column: {team_col}, value column: {value_col}")
            
            # Extract numeric values
            df = df.with_columns([
                pl.col(value_col).map_elements(
                    self.extract_currency_value, 
                    return_dtype=pl.Float64
                ).alias('market_value_eur')
            ])
            
            # Select relevant columns
            result_df = df.select([
                pl.col(team_col).alias('team'),
                pl.col('market_value_eur')
            ])
            
            # Remove rows with null values
            result_df = result_df.filter(pl.col('market_value_eur').is_not_null())
            
            logger.info(f"Processed {result_df.height} clubs with market values")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to load club values: {e}")
            return None
    
    def load_club_wages(self) -> Optional[pl.DataFrame]:
        """Load and process club wage data."""
        logger.info("Loading club wage data...")
        
        wages_file = self.data_raw / "Club wages.csv"
        if not wages_file.exists():
            logger.warning("Club wages.csv not found")
            return None
        
        try:
            df = pl.read_csv(wages_file, ignore_errors=True)
            logger.info(f"Loaded club wages: {df.height} rows, {df.width} columns")
            
            # Find team column
            team_col = None
            for col in ['Club', 'Squad', 'Team', 'club', 'squad', 'team']:
                if col in df.columns:
                    team_col = col
                    break
            
            if not team_col:
                logger.error("No team column found in Club wages.csv")
                return None
            
            # Canonicalize team names
            df = canonicalize_frame(df, [team_col])
            
            # Find wage column (annual wages preferred)
            wage_col = None
            for col in df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['annual', 'yearly', 'total']):
                    if any(term in col_lower for term in ['wage', 'salary', 'cost']):
                        wage_col = col
                        break
            
            # Fallback to any wage column
            if not wage_col:
                for col in df.columns:
                    if any(term in col.lower() for term in ['wage', 'salary', 'cost']):
                        wage_col = col
                        break
            
            if not wage_col:
                logger.error("No wage column found in Club wages.csv")
                return None
            
            logger.info(f"Using team column: {team_col}, wage column: {wage_col}")
            
            # Extract numeric values (convert to GBP if needed)
            df = df.with_columns([
                pl.col(wage_col).map_elements(
                    self.extract_currency_value, 
                    return_dtype=pl.Float64
                ).alias('annual_wages_gbp')
            ])
            
            # Select relevant columns
            result_df = df.select([
                pl.col(team_col).alias('team'),
                pl.col('annual_wages_gbp')
            ])
            
            # Remove rows with null values
            result_df = result_df.filter(pl.col('annual_wages_gbp').is_not_null())
            
            logger.info(f"Processed {result_df.height} clubs with wage data")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to load club wages: {e}")
            return None
    
    def load_attendance_data(self) -> Optional[pl.DataFrame]:
        """Load and process attendance data."""
        logger.info("Loading attendance data...")
        
        attendance_file = self.data_raw / "Attendance Data.csv"
        if not attendance_file.exists():
            logger.warning("Attendance Data.csv not found")
            return None
        
        try:
            df = pl.read_csv(attendance_file, ignore_errors=True)
            logger.info(f"Loaded attendance data: {df.height} rows, {df.width} columns")
            
            # Find team column
            team_col = None
            for col in ['Club', 'Team', 'club', 'team']:
                if col in df.columns:
                    team_col = col
                    break
            
            if not team_col:
                logger.error("No team column found in Attendance Data.csv")
                return None
            
            # Canonicalize team names
            df = canonicalize_frame(df, [team_col])
            
            # Find attendance and capacity columns
            avg_attendance_col = None
            capacity_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['average', 'avg']) and 'attendance' in col_lower:
                    avg_attendance_col = col
                elif any(term in col_lower for term in ['capacity', 'max', 'stadium']):
                    capacity_col = col
            
            logger.info(f"Using team column: {team_col}")
            logger.info(f"Average attendance column: {avg_attendance_col}")
            logger.info(f"Capacity column: {capacity_col}")
            
            # Build result columns
            select_cols = [pl.col(team_col).alias('team')]
            
            if avg_attendance_col:
                select_cols.append(
                    pl.col(avg_attendance_col).cast(pl.Float64).alias('avg_attendance')
                )
            else:
                select_cols.append(pl.lit(None, dtype=pl.Float64).alias('avg_attendance'))
            
            if capacity_col:
                # Clean capacity values (remove commas)
                df = df.with_columns([
                    pl.col(capacity_col).str.replace_all(",", "").alias(capacity_col + "_clean")
                ])
                select_cols.append(
                    pl.col(capacity_col + "_clean").cast(pl.Float64, strict=False).alias('capacity')
                )
            else:
                select_cols.append(pl.lit(None, dtype=pl.Float64).alias('capacity'))
            
            result_df = df.select(select_cols)
            
            logger.info(f"Processed {result_df.height} clubs with attendance data")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to load attendance data: {e}")
            return None
    
    def load_manager_tenure(self) -> Optional[pl.DataFrame]:
        """Load and calculate current manager tenure."""
        logger.info("Loading manager tenure data...")
        
        managers_file = self.data_raw / "Premier League Managers.csv"
        if not managers_file.exists():
            logger.warning("Premier League Managers.csv not found")
            return None
        
        try:
            df = pl.read_csv(managers_file, ignore_errors=True)
            logger.info(f"Loaded managers data: {df.height} rows, {df.width} columns")
            
            # Find team column
            team_col = None
            for col in ['Club', 'Team', 'club', 'team']:
                if col in df.columns:
                    team_col = col
                    break
            
            if not team_col:
                logger.error("No team column found in Premier League Managers.csv")
                return None
            
            # Canonicalize team names
            df = canonicalize_frame(df, [team_col])
            
            # Find date columns
            from_col = None
            until_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'from' in col_lower:
                    from_col = col
                elif any(term in col_lower for term in ['until', 'to', 'end']):
                    until_col = col
            
            if not from_col:
                logger.error("No 'from' date column found in managers data")
                return None
            
            logger.info(f"Using team column: {team_col}, from column: {from_col}")
            if until_col:
                logger.info(f"Until column: {until_col}")
            
            # Parse from dates
            try:
                df = df.with_columns([
                    pl.col(from_col).str.strptime(pl.Date, format='%d/%m/%Y', strict=False).alias('from_date')
                ])
            except:
                try:
                    df = df.with_columns([
                        pl.col(from_col).str.strptime(pl.Date, format='%Y-%m-%d', strict=False).alias('from_date')
                    ])
                except:
                    logger.error("Could not parse manager 'from' dates")
                    return None
            
            # Filter to valid dates
            df = df.filter(pl.col('from_date').is_not_null())
            
            # Get current manager per club (most recent appointment)
            current_managers = (
                df
                .sort(['from_date'], descending=True)
                .group_by(team_col)
                .agg([
                    pl.col('from_date').first().alias('current_manager_from'),
                    pl.first('Manager').alias('current_manager_name')
                ])
            )
            
            # Calculate tenure days
            today = datetime.now().date()
            current_managers = current_managers.with_columns([
                (pl.lit(today) - pl.col('current_manager_from')).dt.total_days().alias('manager_tenure_days')
            ])
            
            # Select relevant columns
            result_df = current_managers.select([
                pl.col(team_col).alias('team'),
                pl.col('manager_tenure_days').cast(pl.Float64),
                pl.col('current_manager_name')
            ])
            
            logger.info(f"Processed {result_df.height} clubs with manager tenure data")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to load manager tenure: {e}")
            return None
    
    def build_static_features(self) -> Path:
        """Build complete static club features dataset."""
        logger.info("Starting static club features build...")
        
        # Load all datasets
        club_values = self.load_club_values()
        club_wages = self.load_club_wages()
        attendance_data = self.load_attendance_data()
        manager_tenure = self.load_manager_tenure()
        
        # Start with the dataset that has the most complete team coverage
        datasets = [
            ('values', club_values),
            ('wages', club_wages), 
            ('attendance', attendance_data),
            ('tenure', manager_tenure)
        ]
        
        # Find the base dataset
        base_df = None
        base_name = None
        
        for name, dataset in datasets:
            if dataset is not None:
                logger.info(f"Using {name} as base dataset ({dataset.height} teams)")
                base_df = dataset
                base_name = name
                break
        
        if base_df is None:
            raise ValueError("No valid datasets found")
        
        # Join all other datasets
        for name, dataset in datasets:
            if dataset is not None and name != base_name:
                logger.info(f"Joining {name} dataset...")
                
                base_df = base_df.join(
                    dataset,
                    on='team',
                    how='left'
                )
                
                # Report join statistics
                total_teams = base_df.height
                matched_teams = dataset.height
                logger.info(f"  {name}: {matched_teams} teams in source, joined to {total_teams} teams")
        
        # Fill missing values with appropriate defaults
        logger.info("Filling missing values...")
        
        columns_to_fill = {
            'market_value_eur': 50_000_000,  # 50M EUR default
            'annual_wages_gbp': 30_000_000,  # 30M GBP default
            'avg_attendance': 40_000,        # 40k attendance default
            'capacity': 50_000,              # 50k capacity default
            'manager_tenure_days': 365       # 1 year default
        }
        
        for col, default_val in columns_to_fill.items():
            if col in base_df.columns:
                null_count = base_df[col].null_count()
                if null_count > 0:
                    logger.info(f"  Filling {null_count} null values in {col} with {default_val:,}")
                    base_df = base_df.with_columns([
                        pl.col(col).fill_null(default_val)
                    ])
        
        # Add feature engineering
        logger.info("Engineering additional features...")
        
        if 'avg_attendance' in base_df.columns and 'capacity' in base_df.columns:
            base_df = base_df.with_columns([
                (pl.col('avg_attendance') / pl.col('capacity')).alias('attendance_rate')
            ])
        
        if 'market_value_eur' in base_df.columns and 'annual_wages_gbp' in base_df.columns:
            # Convert GBP to EUR (approximate rate 1.15)
            base_df = base_df.with_columns([
                (pl.col('annual_wages_gbp') * 1.15 / pl.col('market_value_eur')).alias('wage_to_value_ratio')
            ])
        
        # Save output
        output_path = self.output_dir / "static_2025_26.parquet"
        base_df.write_parquet(output_path)
        
        logger.info(f"✅ Static club features saved: {output_path}")
        logger.info(f"📊 Final dataset: {base_df.height} teams, {base_df.width} features")
        
        # Log feature summary
        logger.info("Feature summary:")
        for col in base_df.columns:
            if col != 'team' and col != 'current_manager_name':
                non_null = base_df[col].drop_nulls().len()
                mean_val = base_df[col].mean()
                logger.info(f"  {col}: {non_null}/{base_df.height} non-null, mean={mean_val:.0f}")
        
        return output_path


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    data_raw = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed" / "features"
    
    try:
        builder = StaticClubBuilder(data_raw, output_dir)
        output_path = builder.build_static_features()
        
        print(f"\n🏢 Static Club Features Complete!")
        print(f"📊 Output: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to build static features: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 