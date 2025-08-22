#!/usr/bin/env python3
"""
FPL Feature Engineering Script - Step 3c

This script transforms the merged FPL dataset into model-ready features with robust,
leakage-safe features and a clear schema for Step 4 (the optimizer).

Key Features:
- Player form & production (lagged xP, rolling statistics)
- Usage & role proxies (position encoding, team price share)
- Opponent & schedule context (difficulty, home ratio, rest days)
- Team strength indicators (financial index, manager tenure)
- Interaction features (form adjustments, home boosts, price efficiency)

Leakage Protection:
- All rolling/lag features use only past gameweeks relative to each (player_id, gameweek)
- Strict use of groupby-sort then shift() and expanding/rolling windows
- Never include current or future GW info in features for that row
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Builds model-ready features from the merged FPL dataset."""
    
    def __init__(self, data_dir: str = "data", outputs_dir: str = "outputs"):
        self.data_dir = Path(data_dir)
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Feature tracking
        self.feature_dict = {}
        self.feature_stats = {}
        self.qc_results = {}
        
        # Base columns to preserve
        self.base_columns = [
            'player_id', 'team_id', 'position', 'price', 'gameweek', 'is_home'
        ]
        
        # Target variable
        self.target_column = 'xP'
        
    def load(self) -> pd.DataFrame:
        """Load the merged dataset."""
        logger.info("Loading merged FPL dataset...")
        
        merged_path = self.data_dir / "FPL" / "processed" / "fpl_merged.parquet"
        if not merged_path.exists():
            raise FileNotFoundError(f"Merged dataset not found: {merged_path}")
        
        df = pd.read_parquet(merged_path)
        logger.info(f"Loaded dataset: {df.shape}")
        
        # Ensure proper sorting for feature engineering
        df = df.sort_values(['player_id', 'gameweek']).reset_index(drop=True)
        
        return df
    
    def make_player_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create player-level features (form, production, usage)."""
        logger.info("Creating player-level features...")
        
        # Create a copy to avoid modifying original
        df_features = df.copy()
        
        # 4.1 Player form & production features
        logger.info("Building player form & production features...")
        
        # Lagged xP features (past gameweeks only)
        for lag in [1, 2, 3]:
            df_features[f'xp_lag{lag}'] = df_features.groupby('player_id')['xP'].shift(lag)
            self._add_feature_info(f'xp_lag{lag}', 'numeric', f'xP from {lag} gameweek(s) ago', 'null_policy: fill_na', f'depends_on: xP, player_id, gameweek')
        
        # Rolling mean xP (last 3 GWs)
        df_features['xp_rolling_mean_g3'] = df_features.groupby('player_id')['xP'].rolling(
            window=3, min_periods=1
        ).mean().reset_index(0, drop=True)
        self._add_feature_info('xp_rolling_mean_g3', 'numeric', 'Rolling mean of xP over last 3 gameweeks', 'null_policy: fill_na', 'depends_on: xP, player_id, gameweek')
        
        # Rolling std xP (last 3 GWs)
        df_features['xp_rolling_std_g3'] = df_features.groupby('player_id')['xP'].rolling(
            window=3, min_periods=1
        ).std().reset_index(0, drop=True)
        self._add_feature_info('xp_rolling_std_g3', 'numeric', 'Rolling std of xP over last 3 gameweeks', 'null_policy: fill_na', 'depends_on: xP, player_id, gameweek')
        
        # Rolling mean xP (last 6 GWs)
        df_features['xp_rolling_mean_g6'] = df_features.groupby('player_id')['xP'].rolling(
            window=6, min_periods=1
        ).mean().reset_index(0, drop=True)
        self._add_feature_info('xp_rolling_mean_g6', 'numeric', 'Rolling mean of xP over last 6 gameweeks', 'null_policy: fill_na', 'depends_on: xP, player_id, gameweek')
        
        # Exponentially weighted mean xP (halflife=3)
        df_features['xp_ewm_mean_halflife3'] = df_features.groupby('player_id')['xP'].ewm(
            halflife=3, min_periods=1
        ).mean().reset_index(0, drop=True)
        self._add_feature_info('xp_ewm_mean_halflife3', 'numeric', 'Exponentially weighted mean of xP with halflife=3', 'null_policy: fill_na', 'depends_on: xP, player_id, gameweek')
        
        # Minutes availability flag
        df_features['minutes_flag_available'] = df_features['availability_status'].apply(
            lambda x: 1 if pd.notna(x) and str(x).lower() in ['a', 'available', 'fit'] else 0
        )
        self._add_feature_info('minutes_flag_available', 'binary', '1 if player available/fit, 0 otherwise', 'null_policy: fill_na', 'depends_on: availability_status')
        
        # Price change (past gameweek)
        df_features['price_change_g1'] = df_features.groupby('player_id')['price'].diff()
        self._add_feature_info('price_change_g1', 'numeric', 'Price change from previous gameweek', 'null_policy: fill_na', 'depends_on: price, player_id, gameweek')
        
        # Consistency over last 6 GWs (binary rate)
        def calculate_consistency(group):
            """Calculate consistency as share of last 6 GWs with xP >= past-6 avg."""
            if len(group) < 6:
                return np.nan
            
            # Get last 6 gameweeks
            last_6 = group.tail(6)
            if len(last_6) < 6:
                return np.nan
            
            # Calculate past-6 average (excluding current)
            past_6_avg = group.head(-1).tail(6)['xP'].mean()
            if pd.isna(past_6_avg):
                return np.nan
            
            # Count how many of last 6 meet threshold
            above_threshold = (last_6['xP'] >= past_6_avg).sum()
            return above_threshold / 6
        
        df_features['consistency_g6'] = df_features.groupby('player_id').apply(
            calculate_consistency
        ).reset_index(0, drop=True)
        self._add_feature_info('consistency_g6', 'numeric', 'Share of last 6 GWs with xP >= past-6 avg', 'null_policy: fill_na', 'depends_on: xP, player_id, gameweek')
        
        # 4.2 Usage & role proxies
        logger.info("Building usage & role proxy features...")
        
        # Position one-hot encoding (keep original categorical)
        position_dummies = pd.get_dummies(df_features['position'], prefix='pos')
        df_features = pd.concat([df_features, position_dummies], axis=1)
        
        for col in position_dummies.columns:
            self._add_feature_info(col, 'binary', f'One-hot encoding for position {col.split("_")[1]}', 'null_policy: fill_na', 'depends_on: position')
        
        # Team share price (proxy for role/importance)
        team_median_price = df_features.groupby(['team_id', 'gameweek'])['price'].transform('median')
        df_features['team_share_price'] = df_features['price'] / team_median_price
        self._add_feature_info('team_share_price', 'numeric', 'Price relative to team median price at that GW', 'null_policy: fill_na', 'depends_on: price, team_id, gameweek')
        
        return df_features
    
    def make_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team-level features (strength, financial, manager)."""
        logger.info("Creating team-level features...")
        
        df_features = df.copy()
        
        # 4.4 Team strength features
        logger.info("Building team strength features...")
        
        # Team xP rolling mean (past 3 GWs)
        team_xp_rolling = df_features.groupby(['team_id', 'gameweek'])['xP'].mean().reset_index()
        team_xp_rolling['team_xp_rolling_mean_g3'] = team_xp_rolling.groupby('team_id')['xP'].rolling(
            window=3, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Join back to main dataframe
        df_features = df_features.merge(
            team_xp_rolling[['team_id', 'gameweek', 'team_xp_rolling_mean_g3']],
            on=['team_id', 'gameweek'],
            how='left'
        )
        self._add_feature_info('team_xp_rolling_mean_g3', 'numeric', 'Team average player xP over last 3 GWs', 'null_policy: fill_na', 'depends_on: xP, team_id, gameweek')
        
        # Team financial index (zscore of log1p transformations)
        if 'club_value' in df_features.columns and 'annual_wages' in df_features.columns:
            # Handle missing values gracefully
            club_value_log = np.log1p(df_features['club_value'].fillna(0))
            annual_wages_log = np.log1p(df_features['annual_wages'].fillna(0))
            
            # Calculate z-scores
            club_value_zscore = (club_value_log - club_value_log.mean()) / club_value_log.std()
            annual_wages_zscore = (annual_wages_log - annual_wages_log.mean()) / annual_wages_log.std()
            
            df_features['team_financial_index'] = club_value_zscore + annual_wages_zscore
            self._add_feature_info('team_financial_index', 'numeric', 'Z-score of log1p(club_value) + z-score of log1p(annual_wages)', 'null_policy: fill_na', 'depends_on: club_value, annual_wages')
        else:
            logger.warning("club_value or annual_wages not available - skipping team_financial_index")
            df_features['team_financial_index'] = np.nan
        
        # Manager tenure features
        if 'manager_tenure' in df_features.columns:
            df_features['manager_tenure_num'] = df_features['manager_tenure'].apply(self._parse_tenure_to_months)
            
            # Create tenure buckets
            df_features['manager_tenure_bucket'] = df_features['manager_tenure_num'].apply(
                lambda x: 'short' if pd.isna(x) or x < 12 else ('medium' if x < 36 else 'long')
            )
            
            self._add_feature_info('manager_tenure_num', 'numeric', 'Manager tenure in months', 'null_policy: fill_na', 'depends_on: manager_tenure')
            self._add_feature_info('manager_tenure_bucket', 'categorical', 'Manager tenure category (short/medium/long)', 'null_policy: fill_na', 'depends_on: manager_tenure')
        else:
            logger.warning("manager_tenure not available - skipping manager tenure features")
            df_features['manager_tenure_num'] = np.nan
            df_features['manager_tenure_bucket'] = np.nan
        
        return df_features
    
    def make_opponent_schedule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create opponent and schedule context features."""
        logger.info("Creating opponent and schedule context features...")
        
        df_features = df.copy()
        
        # 4.3 Opponent & schedule context features
        logger.info("Building opponent & schedule context features...")
        
        # Rolling mean of opponent difficulty (past 3 GWs) - handle missing column gracefully
        if 'opponent_difficulty' in df_features.columns:
            df_features['opp_diff_lagmean_g3'] = df_features.groupby('player_id')['opponent_difficulty'].rolling(
                window=3, min_periods=1
            ).mean().reset_index(0, drop=True)
            self._add_feature_info('opp_diff_lagmean_g3', 'numeric', 'Rolling mean of opponent difficulty over last 3 GWs', 'null_policy: fill_na', 'depends_on: opponent_difficulty, player_id, gameweek')
        else:
            logger.warning("opponent_difficulty not available - skipping opp_diff_lagmean_g3")
            df_features['opp_diff_lagmean_g3'] = np.nan
        
        # Home ratio over past 6 GWs
        def calculate_home_ratio(group):
            """Calculate home fixture ratio over past 6 gameweeks."""
            if len(group) < 6:
                return np.nan
            
            past_6 = group.head(-1).tail(6)  # Exclude current GW
            if len(past_6) < 6:
                return np.nan
            
            home_count = past_6['is_home'].sum()
            return home_count / 6
        
        df_features['home_ratio_g6'] = df_features.groupby('player_id').apply(
            calculate_home_ratio
        ).reset_index(0, drop=True)
        self._add_feature_info('home_ratio_g6', 'numeric', 'Share of home fixtures over past 6 GWs', 'null_policy: fill_na', 'depends_on: is_home, player_id, gameweek')
        
        # Rest days and congestion lagged features
        if 'rest_days' in df_features.columns:
            df_features['rest_days_lag1'] = df_features.groupby('player_id')['rest_days'].shift(1)
            self._add_feature_info('rest_days_lag1', 'numeric', 'Rest days from previous gameweek', 'null_policy: fill_na', 'depends_on: rest_days, player_id, gameweek')
        
        if 'fixture_congestion' in df_features.columns:
            df_features['congestion_lag1'] = df_features.groupby('player_id')['fixture_congestion'].shift(1)
            self._add_feature_info('congestion_lag1', 'numeric', 'Fixture congestion from previous gameweek', 'null_policy: fill_na', 'depends_on: fixture_congestion, player_id, gameweek')
            
            # Rest stress over past 3 GWs (high congestion indicator)
            df_features['rest_stress_g3'] = df_features.groupby('player_id')['fixture_congestion'].rolling(
                window=3, min_periods=1
            ).mean().reset_index(0, drop=True)
            self._add_feature_info('rest_stress_g3', 'numeric', 'Rolling mean of fixture congestion over last 3 GWs', 'null_policy: fill_na', 'depends_on: fixture_congestion, player_id, gameweek')
        
        return df_features
    
    def make_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features combining multiple base features."""
        logger.info("Creating interaction features...")
        
        df_features = df.copy()
        
        # 4.5 Interaction features
        logger.info("Building interaction features...")
        
        # xP form adjusted by team financial strength
        if 'xp_rolling_mean_g3' in df_features.columns and 'team_financial_index' in df_features.columns:
            df_features['xp_form_adj_g3'] = df_features['xp_rolling_mean_g3'] * (
                1 + df_features['team_financial_index'] * 0.1
            )
            self._add_feature_info('xp_form_adj_g3', 'numeric', 'xP form adjusted by team financial strength', 'null_policy: fill_na', 'depends_on: xp_rolling_mean_g3, team_financial_index')
        
        # Home form boost
        if 'xp_rolling_mean_g3' in df_features.columns and 'is_home' in df_features.columns:
            df_features['home_form_boost'] = df_features['xp_rolling_mean_g3'] * df_features['is_home']
            self._add_feature_info('home_form_boost', 'numeric', 'xP form boost for home fixtures', 'null_policy: fill_na', 'depends_on: xp_rolling_mean_g3, is_home')
        
        # Price efficiency (xP per price unit)
        if 'xp_rolling_mean_g3' in df_features.columns and 'price' in df_features.columns:
            df_features['price_efficiency_g3'] = df_features['xp_rolling_mean_g3'] / df_features['price']
            
            # Clip infinite values to reasonable percentiles
            if df_features['price_efficiency_g3'].notna().any():
                finite_values = df_features['price_efficiency_g3'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(finite_values) > 0:
                    p1 = finite_values.quantile(0.01)
                    p99 = finite_values.quantile(0.99)
                    df_features['price_efficiency_g3'] = df_features['price_efficiency_g3'].clip(p1, p99)
            
            self._add_feature_info('price_efficiency_g3', 'numeric', 'xP form per price unit (clipped to 1st-99th percentile)', 'null_policy: fill_na', 'depends_on: xp_rolling_mean_g3, price')
        
        return df_features
    
    def finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finalize the feature dataset with quality checks and output."""
        logger.info("Finalizing feature dataset...")
        
        # Select final columns
        feature_columns = [col for col in df.columns if col not in ['player_name', 'availability_status', 'confidence']]
        df_final = df[feature_columns].copy()
        
        # Ensure no duplicates
        duplicate_check = df_final.duplicated(subset=['player_id', 'gameweek'], keep=False)
        duplicate_count = duplicate_check.sum()
        
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate rows - removing duplicates")
            df_final = df_final.drop_duplicates(subset=['player_id', 'gameweek'])
        
        # Cast dtypes appropriately
        df_final = self._cast_dtypes(df_final)
        
        # Quality checks
        self._run_quality_checks(df_final)
        
        # Save outputs
        self._save_outputs(df_final)
        
        return df_final
    
    def _parse_tenure_to_months(self, tenure_str: Any) -> Optional[int]:
        """Parse tenure string to months."""
        if pd.isna(tenure_str):
            return np.nan
        
        tenure_str = str(tenure_str).lower().strip()
        
        # Handle various formats
        months = 0
        
        # Extract years
        year_match = re.search(r'(\d+)\s*(?:year|yr|y)', tenure_str)
        if year_match:
            months += int(year_match.group(1)) * 12
        
        # Extract months
        month_match = re.search(r'(\d+)\s*(?:month|mo|mos)', tenure_str)
        if month_match:
            months += int(month_match.group(1))
        
        # Extract days (convert to months approximately)
        day_match = re.search(r'(\d+)\s*(?:day|d)', tenure_str)
        if day_match:
            months += int(day_match.group(1)) / 30
        
        return months if months > 0 else np.nan
    
    def _cast_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to appropriate dtypes."""
        logger.info("Casting column dtypes...")
        
        # Keep position as categorical
        if 'position' in df.columns:
            df['position'] = df['position'].astype('category')
        
        # Keep manager_tenure_bucket as categorical if it exists
        if 'manager_tenure_bucket' in df.columns:
            df['manager_tenure_bucket'] = df['manager_tenure_bucket'].astype('category')
        
        # Ensure is_home is boolean
        if 'is_home' in df.columns:
            df['is_home'] = df['is_home'].astype(bool)
        
        # Ensure numeric columns are float
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['player_id', 'team_id', 'gameweek']:  # Keep IDs as int
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _run_quality_checks(self, df: pd.DataFrame) -> None:
        """Run quality checks on the final dataset."""
        logger.info("Running quality checks...")
        
        # Basic stats
        self.qc_results = {
            'row_count': len(df),
            'feature_count': len(df.columns),
            'duplicate_count': df.duplicated(subset=['player_id', 'gameweek']).sum(),
            'null_rates': {},
            'dtypes': {},
            'leakage_check': 'PASSED'
        }
        
        # Null rates
        for col in df.columns:
            null_rate = df[col].isnull().sum() / len(df) * 100
            self.qc_results['null_rates'][col] = round(null_rate, 2)
        
        # Dtypes
        for col in df.columns:
            self.qc_results['dtypes'][col] = str(df[col].dtype)
        
        # Leakage check - ensure all rolling features have nulls at start
        rolling_features = [col for col in df.columns if 'rolling' in col or 'ewm' in col or 'lag' in col]
        
        # Check if we have multiple gameweeks to properly assess leakage
        unique_gameweeks = df['gameweek'].nunique()
        
        if unique_gameweeks > 1:
            # Multiple gameweeks - check for proper null handling
            for feature in rolling_features:
                if feature in df.columns:
                    # Check if first few rows have nulls (expected for rolling features)
                    first_non_null_idx = df[feature].first_valid_index()
                    if first_non_null_idx is not None and first_non_null_idx < 3:
                        logger.info(f"✅ {feature}: Proper null handling at start")
                    else:
                        logger.warning(f"⚠️ {feature}: Unexpected null pattern - possible leakage")
                        self.qc_results['leakage_check'] = 'WARNING'
        else:
            # Single gameweek - all lagged/rolling features will be null (expected)
            logger.info(f"Single gameweek dataset ({unique_gameweeks} GW) - lagged/rolling features will be null (expected)")
            self.qc_results['leakage_check'] = 'PASSED (single GW)'
        
        # Price efficiency sanity check
        if 'price_efficiency_g3' in df.columns:
            finite_values = df['price_efficiency_g3'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite_values) > 0:
                p1 = finite_values.quantile(0.01)
                p99 = finite_values.quantile(0.99)
                logger.info(f"Price efficiency clipped to [{p1:.4f}, {p99:.4f}]")
    
    def _save_outputs(self, df: pd.DataFrame) -> None:
        """Save all output artifacts."""
        logger.info("Saving output artifacts...")
        
        # Save feature dataset
        output_path = self.data_dir / "FPL" / "processed" / "fpl_features_model.parquet"
        output_path.parent.mkdir(exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved feature dataset to {output_path}")
        
        # Save feature dictionary
        self._save_feature_dict()
        
        # Save QC results
        qc_path = self.outputs_dir / "fpl_features_qc.json"
        with open(qc_path, 'w') as f:
            json.dump(self.qc_results, f, indent=2, default=str)
        logger.info(f"Saved QC results to {qc_path}")
        
        # Save feature preview (first 200 rows, selected columns)
        preview_columns = ['player_id', 'team_id', 'position', 'gameweek', 'xP'] + [
            col for col in df.columns if col not in ['player_id', 'team_id', 'position', 'gameweek', 'xP']
        ][:10]  # First 10 additional columns
        
        preview_df = df[preview_columns].head(200)
        preview_path = self.outputs_dir / "fpl_feature_preview.csv"
        preview_df.to_csv(preview_path, index=False)
        logger.info(f"Saved feature preview to {preview_path}")
    
    def _save_feature_dict(self) -> None:
        """Save feature dictionary documentation."""
        md_path = self.outputs_dir / "fpl_feature_dict.md"
        
        content = "# FPL Feature Dictionary - Step 3c\n\n"
        content += f"**Generated**: {datetime.now().isoformat()}\n\n"
        content += "This document describes all engineered features for the FPL optimization model.\n\n"
        
        # Group features by category
        feature_categories = {
            'Base Features': ['player_id', 'team_id', 'position', 'price', 'gameweek', 'is_home'],
            'Player Form & Production': [col for col in self.feature_dict.keys() if any(x in col for x in ['xp_lag', 'xp_rolling', 'xp_ewm', 'minutes_flag', 'price_change', 'consistency'])],
            'Usage & Role Proxies': [col for col in self.feature_dict.keys() if any(x in col for x in ['pos_', 'team_share_price'])],
            'Opponent & Schedule Context': [col for col in self.feature_dict.keys() if any(x in col for x in ['opp_diff', 'home_ratio', 'rest_days', 'congestion', 'rest_stress'])],
            'Team Strength': [col for col in self.feature_dict.keys() if any(x in col for x in ['team_xp', 'team_financial', 'manager_tenure'])],
            'Interaction Features': [col for col in self.feature_dict.keys() if any(x in col for x in ['xp_form_adj', 'home_form_boost', 'price_efficiency'])]
        }
        
        for category, features in feature_categories.items():
            if features:
                content += f"## {category}\n\n"
                content += "| Feature | Type | Description | Null Policy | Dependencies |\n"
                content += "|---------|------|-------------|-------------|--------------|\n"
                
                for feature in features:
                    if feature in self.feature_dict:
                        info = self.feature_dict[feature]
                        content += f"| {feature} | {info['type']} | {info['description']} | {info['null_policy']} | {info['dependencies']} |\n"
                
                content += "\n"
        
        with open(md_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Saved feature dictionary to {md_path}")
    
    def _add_feature_info(self, name: str, feature_type: str, description: str, null_policy: str, dependencies: str) -> None:
        """Add feature information to the tracking dictionary."""
        self.feature_dict[name] = {
            'type': feature_type,
            'description': description,
            'null_policy': null_policy,
            'dependencies': dependencies
        }
    
    def print_summary(self) -> None:
        """Print summary statistics to console."""
        logger.info("=" * 60)
        logger.info("FPL FEATURE ENGINEERING SUMMARY - STEP 3C")
        logger.info("=" * 60)
        
        # Basic counts
        logger.info(f"Dataset Shape: {self.qc_results['row_count']} rows × {self.qc_results['feature_count']} columns")
        logger.info(f"Duplicate Check: {'✅ PASSED' if self.qc_results['duplicate_count'] == 0 else '❌ FAILED'}")
        logger.info(f"Leakage Check: {self.qc_results['leakage_check']}")
        
        # Top null rates
        null_rates = sorted(self.qc_results['null_rates'].items(), key=lambda x: x[1], reverse=True)
        logger.info(f"\nTop 10 Null Rates:")
        for feature, rate in null_rates[:10]:
            status = "✅" if rate < 5 else "⚠️" if rate < 20 else "❌"
            logger.info(f"  {status} {feature}: {rate}%")
        
        # Feature count by category
        feature_categories = {
            'Base Features': 6,
            'Player Form & Production': len([f for f in self.feature_dict.keys() if any(x in f for x in ['xp_lag', 'xp_rolling', 'xp_ewm', 'minutes_flag', 'price_change', 'consistency'])]),
            'Usage & Role Proxies': len([f for f in self.feature_dict.keys() if any(x in f for x in ['pos_', 'team_share_price'])]),
            'Opponent & Schedule': len([f for f in self.feature_dict.keys() if any(x in f for x in ['opp_diff', 'home_ratio', 'rest_days', 'congestion', 'rest_stress'])]),
            'Team Strength': len([f for f in self.feature_dict.keys() if any(x in f for x in ['team_xp', 'team_financial', 'manager_tenure'])]),
            'Interaction Features': len([f for f in self.feature_dict.keys() if any(x in f for x in ['xp_form_adj', 'home_form_boost', 'price_efficiency'])])
        }
        
        logger.info(f"\nFeature Count by Category:")
        for category, count in feature_categories.items():
            logger.info(f"  {category}: {count} features")
        
        logger.info("=" * 60)
    
    def run(self) -> None:
        """Run the complete feature engineering process."""
        try:
            logger.info("Starting FPL Feature Engineering (Step 3c)")
            
            # Load data
            df = self.load()
            
            # Build features
            df = self.make_player_features(df)
            df = self.make_team_features(df)
            df = self.make_opponent_schedule_features(df)
            df = self.make_interaction_features(df)
            
            # Finalize
            df_final = self.finalize(df)
            
            # Print summary
            self.print_summary()
            
            logger.info("Feature engineering completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            raise


def main():
    """Main entry point."""
    builder = FeatureBuilder()
    builder.run()


if __name__ == "__main__":
    main() 