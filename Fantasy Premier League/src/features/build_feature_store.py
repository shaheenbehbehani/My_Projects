#!/usr/bin/env python3
"""
Feature Store Builder

Joins all Phase 3 feature components into a final feature store:
- Season form (Elo, rolling metrics)
- Static club attributes
- Player aggregations
- Injury burden
- Bookmaker priors

Output: data/processed/features_2025_26.parquet
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureStoreBuilder:
    """Builds the final feature store by joining all feature components."""
    
    def __init__(self, features_dir: Path, processed_dir: Path, reports_dir: Path):
        """Initialize the feature store builder."""
        self.features_dir = Path(features_dir)
        self.processed_dir = Path(processed_dir)
        self.reports_dir = Path(reports_dir)
        
        # Ensure output directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def load_fixtures_teams(self) -> List[str]:
        """Load canonical team list from fixtures."""
        logger.info("Loading canonical team list from fixtures...")
        
        fixtures_file = self.processed_dir / "fixtures_2025_26.parquet"
        if not fixtures_file.exists():
            raise FileNotFoundError("Fixtures file not found - run build_fixtures.py first")
        
        try:
            fixtures_df = pl.read_parquet(fixtures_file)
            
            # Get unique teams
            home_teams = fixtures_df['home_team'].unique().to_list()
            away_teams = fixtures_df['away_team'].unique().to_list()
            
            all_teams = sorted(list(set(home_teams + away_teams)))
            logger.info(f"Found {len(all_teams)} canonical teams in fixtures")
            
            return all_teams
            
        except Exception as e:
            logger.error(f"Failed to load fixtures teams: {e}")
            raise
    
    def load_feature_component(self, component_name: str, filename: str) -> Optional[pl.DataFrame]:
        """Load a feature component with error handling."""
        logger.info(f"Loading {component_name}...")
        
        file_path = self.features_dir / filename
        if not file_path.exists():
            logger.warning(f"{component_name} not found: {filename}")
            return None
        
        try:
            df = pl.read_parquet(file_path)
            logger.info(f"Loaded {component_name}: {df.height} teams, {df.width} features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {component_name}: {e}")
            return None
    
    def create_base_dataframe(self, teams: List[str]) -> pl.DataFrame:
        """Create base DataFrame with canonical team list."""
        logger.info(f"Creating base DataFrame with {len(teams)} teams")
        
        return pl.DataFrame({'team': teams})
    
    def join_feature_components(self, base_df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Dict]]:
        """Join all feature components to base DataFrame."""
        logger.info("Joining feature components...")
        
        join_stats = {}
        result_df = base_df.clone()
        
        # Define feature components and their expected columns
        components = [
            ('Form Features', 'form_baselines_2025_26.parquet', 
             ['elo_baseline', 'xg_diff_baseline', 'ppg_baseline']),
            ('Static Club Features', 'static_2025_26.parquet',
             ['market_value_eur', 'annual_wages_gbp', 'avg_attendance', 'capacity', 'manager_tenure_days']),
            ('Player Aggregations', 'player_agg.parquet',
             ['goals_90', 'xg_90', 'xa_90', 'avg_xg_per_shot', 'shots_per_game']),
            ('Injury Burden', 'injury_burden.parquet',
             ['injury_burden_per_1000']),
            ('Bookmaker Priors', 'bookmaker_priors_2025_26.parquet',
             ['prior_title_prob'])
        ]
        
        for component_name, filename, expected_cols in components:
            df = self.load_feature_component(component_name, filename)
            
            if df is not None:
                # Track join statistics
                before_join = result_df.height
                teams_in_component = df.height
                
                # Perform left join
                result_df = result_df.join(df, on='team', how='left')
                
                after_join = result_df.height
                
                # Calculate match statistics
                matched_teams = 0
                for col in expected_cols:
                    if col in result_df.columns:
                        matched_teams = result_df[col].drop_nulls().len()
                        break
                
                join_stats[component_name] = {
                    'source_teams': teams_in_component,
                    'matched_teams': matched_teams,
                    'match_rate': matched_teams / before_join if before_join > 0 else 0,
                    'features_added': len([col for col in expected_cols if col in result_df.columns])
                }
                
                logger.info(f"Joined {component_name}: {matched_teams}/{before_join} teams matched ({matched_teams/before_join*100:.1f}%)")
            
            else:
                join_stats[component_name] = {
                    'source_teams': 0,
                    'matched_teams': 0,
                    'match_rate': 0.0,
                    'features_added': 0
                }
                logger.warning(f"Skipped {component_name}: component not available")
        
        return result_df, join_stats
    
    def standardize_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize final column names for consistency."""
        logger.info("Standardizing column names...")
        
        # Define column name mappings
        column_mapping = {
            'elo_baseline': 'elo_pre',
            'xg_diff_baseline': 'roll5_xg_diff_pre',
            'ppg_baseline': 'roll5_ppg_pre',
            'goals_90': 'squad_goals_90',
            'xg_90': 'squad_xg_90',
            'xa_90': 'squad_xa_90'
        }
        
        # Apply mappings if columns exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename({old_name: new_name})
                logger.info(f"Renamed {old_name} -> {new_name}")
        
        return df
    
    def fill_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill missing values with appropriate defaults or league averages."""
        logger.info("Filling missing values...")
        
        # Define default values for key features
        defaults = {
            'elo_pre': 1500.0,
            'roll5_xg_diff_pre': 0.0,
            'roll5_ppg_pre': 1.0,
            'market_value_eur': 50_000_000.0,
            'annual_wages_gbp': 30_000_000.0,
            'avg_attendance': 40_000.0,
            'capacity': 50_000.0,
            'manager_tenure_days': 365.0,
            'squad_goals_90': 1.5,
            'squad_xg_90': 1.4,
            'squad_xa_90': 1.2,
            'avg_xg_per_shot': 0.1,
            'shots_per_game': 12.0,
            'injury_burden_per_1000': 50.0,
            'prior_title_prob': 1.0 / df.height if df.height > 0 else 0.05
        }
        
        fill_summary = {}
        
        for col, default_val in defaults.items():
            if col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    # For probabilities, use calculated value
                    if col == 'prior_title_prob':
                        fill_value = 1.0 / df.height
                    else:
                        fill_value = default_val
                    
                    df = df.with_columns([
                        pl.col(col).fill_null(fill_value)
                    ])
                    
                    fill_summary[col] = {
                        'null_count': null_count,
                        'fill_value': fill_value
                    }
                    
                    logger.info(f"Filled {null_count} nulls in {col} with {fill_value}")
        
        return df
    
    def add_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add derived features from existing columns."""
        logger.info("Adding derived features...")
        
        derived_features = []
        
        # Financial efficiency ratios
        if 'market_value_eur' in df.columns and 'annual_wages_gbp' in df.columns:
            derived_features.append(
                (pl.col('annual_wages_gbp') * 1.15 / pl.col('market_value_eur')).alias('wage_to_value_ratio')
            )
            logger.info("Added wage_to_value_ratio")
        
        # Stadium utilization
        if 'avg_attendance' in df.columns and 'capacity' in df.columns:
            derived_features.append(
                (pl.col('avg_attendance') / pl.col('capacity')).alias('stadium_utilization')
            )
            logger.info("Added stadium_utilization")
        
        # Expected performance vs actual (if available)
        if 'squad_goals_90' in df.columns and 'squad_xg_90' in df.columns:
            derived_features.append(
                (pl.col('squad_goals_90') - pl.col('squad_xg_90')).alias('goals_over_xg_90')
            )
            logger.info("Added goals_over_xg_90")
        
        # Quality metrics
        if 'market_value_eur' in df.columns:
            # Rank-based features
            derived_features.append(
                pl.col('market_value_eur').rank(descending=True).alias('market_value_rank')
            )
            logger.info("Added market_value_rank")
        
        if 'elo_pre' in df.columns:
            derived_features.append(
                pl.col('elo_pre').rank(descending=True).alias('elo_rank')
            )
            logger.info("Added elo_rank")
        
        # Apply derived features
        if derived_features:
            df = df.with_columns(derived_features)
        
        return df
    
    def generate_features_summary(self, df: pl.DataFrame, join_stats: Dict) -> str:
        """Generate comprehensive features summary report."""
        logger.info("Generating features summary report...")
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate completeness
        total_cells = df.height * (df.width - 1)
        null_cells = sum(df[col].null_count() for col in df.columns if col != 'team')
        completeness = ((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 100
        
        report = f"""# Features Summary Report
*Generated: {timestamp}*

## Overview

**Final Feature Store**: `data/processed/features_2025_26.parquet`  
**Teams**: {df.height}  
**Features**: {df.width - 1} (excluding team identifier)  
**Completeness**: {completeness:.1f}%

## Feature Components

"""
        
        # Join statistics
        for component, stats in join_stats.items():
            status = "✅" if stats['matched_teams'] > 0 else "❌"
            report += f"### {component} {status}\n\n"
            report += f"- **Source Teams**: {stats['source_teams']}\n"
            report += f"- **Matched Teams**: {stats['matched_teams']}\n"
            report += f"- **Match Rate**: {stats['match_rate']*100:.1f}%\n"
            report += f"- **Features Added**: {stats['features_added']}\n\n"
        
        # Feature statistics
        report += "## Feature Statistics\n\n"
        report += "| Feature | Non-Null | Mean | Min | Max | Std |\n"
        report += "|---------|----------|------|-----|-----|-----|\n"
        
        for col in df.columns:
            if col != 'team' and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                non_null = df[col].drop_nulls().len()
                mean_val = df[col].mean()
                min_val = df[col].min()
                max_val = df[col].max()
                std_val = df[col].std()
                
                report += f"| {col} | {non_null}/{df.height} | {mean_val:.2f} | {min_val:.2f} | {max_val:.2f} | {std_val:.2f} |\n"
        
        # Top/bottom teams for key features
        report += "\n## Team Rankings\n\n"
        
        key_features = ['elo_pre', 'market_value_eur', 'prior_title_prob']
        
        for feature in key_features:
            if feature in df.columns:
                report += f"### Top 5 Teams by {feature}\n\n"
                top_teams = df.sort(feature, descending=True).head(5)
                
                for i, row in enumerate(top_teams.iter_rows(named=True)):
                    report += f"{i+1}. **{row['team']}**: {row[feature]:.3f}\n"
                
                report += "\n"
        
        return report
    
    def identify_unmatched_teams(self, teams: List[str], feature_df: pl.DataFrame) -> List[str]:
        """Identify teams that couldn't be matched across feature components."""
        feature_teams = set(feature_df['team'].to_list())
        canonical_teams = set(teams)
        
        unmatched = canonical_teams - feature_teams
        return sorted(list(unmatched))
    
    def generate_unmatched_report(self, unmatched_teams: List[str]) -> Optional[Path]:
        """Generate report of unmatched teams."""
        if not unmatched_teams:
            logger.info("All teams successfully matched")
            return None
        
        logger.warning(f"Found {len(unmatched_teams)} unmatched teams")
        
        report_content = f"""# Unmatched Teams Report
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Summary

**Unmatched Teams**: {len(unmatched_teams)}  
**Issue**: These teams appear in fixtures but could not be matched to feature data

## Unmatched Teams List

"""
        
        for i, team in enumerate(unmatched_teams, 1):
            report_content += f"{i}. {team}\n"
        
        report_content += f"""

## Potential Causes

1. **Team Name Variations**: Different spellings or abbreviations in source data
2. **Missing Data**: Teams not present in historical/attribute datasets
3. **Canonicalization Issues**: Name standardization not catching all variants

## Recommended Actions

1. Review team name canonicalization mappings in `src/normalization.py`
2. Check if missing teams are newly promoted or have data gaps
3. Consider manual mapping for persistent mismatches

---

*This report indicates data quality issues that should be addressed before modeling.*
"""
        
        # Write report
        report_path = self.reports_dir / "features_unmatched.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.warning(f"Unmatched teams report saved: {report_path}")
        return report_path
    
    def build_feature_store(self) -> Path:
        """Build the complete feature store."""
        logger.info("Starting feature store build...")
        
        # Load canonical team list
        teams = self.load_fixtures_teams()
        
        # Create base DataFrame
        base_df = self.create_base_dataframe(teams)
        
        # Join all feature components
        feature_df, join_stats = self.join_feature_components(base_df)
        
        # Standardize column names
        feature_df = self.standardize_column_names(feature_df)
        
        # Fill missing values
        feature_df = self.fill_missing_values(feature_df)
        
        # Add derived features
        feature_df = self.add_derived_features(feature_df)
        
        # Save final feature store
        output_path = self.processed_dir / "features_2025_26.parquet"
        feature_df.write_parquet(output_path)
        
        logger.info(f"✅ Feature store saved: {output_path}")
        logger.info(f"📊 Final dataset: {feature_df.height} teams, {feature_df.width} features")
        
        # Generate reports
        summary_report = self.generate_features_summary(feature_df, join_stats)
        
        summary_path = self.reports_dir / "features_summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        logger.info(f"📄 Summary report saved: {summary_path}")
        
        # Check for unmatched teams
        unmatched_teams = self.identify_unmatched_teams(teams, feature_df)
        unmatched_path = self.generate_unmatched_report(unmatched_teams)
        
        if unmatched_path:
            logger.warning(f"📄 Unmatched teams report: {unmatched_path}")
        
        return output_path


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    features_dir = project_root / "data" / "processed" / "features"
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    
    try:
        builder = FeatureStoreBuilder(features_dir, processed_dir, reports_dir)
        output_path = builder.build_feature_store()
        
        print(f"\n🏪 Feature Store Complete!")
        print(f"📊 Output: {output_path}")
        print(f"📄 Summary: {reports_dir / 'features_summary.md'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to build feature store: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 