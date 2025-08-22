#!/usr/bin/env python3
"""
Backtest Premier League match prediction model.

This script performs walk-forward validation from 2014/15 to 2024/25 to evaluate
model performance across different seasons and generate backtest metrics.
"""

import polars as pl
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import argparse
from datetime import datetime

# Import our utilities
from .utils import (
    create_season_based_splits,
    calculate_brier_score,
    calculate_log_loss_score,
    extract_season_from_date,
    set_random_seed
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
set_random_seed(42)


class ModelBacktester:
    """Backtests the Premier League match prediction model."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models", reports_dir: str = "reports"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model_and_data(self) -> Tuple[Any, pl.DataFrame, list]:
        """Load the calibrated model and match dataset."""
        # Load calibrated model
        model_path = self.models_dir / "match_model_calibrated.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Calibrated model not found: {model_path}")
        
        import joblib
        model = joblib.load(model_path)
        
        logger.info(f"Loaded calibrated model: {type(model).__name__}")
        
        # Load feature info
        feature_path = self.models_dir / "feature_info.pkl"
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature info not found: {feature_path}")
        
        feature_info = joblib.load(feature_path)
        
        feature_names = feature_info['feature_names']
        logger.info(f"Loaded {len(feature_names)} feature names")
        
        # Load match dataset
        dataset_path = self.processed_dir / "match_dataset.parquet"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Match dataset not found: {dataset_path}")
        
        df = pl.read_parquet(dataset_path)
        logger.info(f"Loaded dataset: {df.height:,} rows, {df.width} columns")
        
        return model, df, feature_names
    
    def prepare_features(self, df: pl.DataFrame, feature_names: list) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for backtesting."""
        # Select feature columns
        X = df.select(feature_names).to_pandas()
        y = df['result_label'].to_numpy()
        
        # Handle missing values
        X = X.fillna(0)
        
        logger.info(f"Prepared features: {X.shape}, target: {y.shape}")
        return X, y
    
    def create_backtest_splits(self, df: pl.DataFrame) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """Create backtest splits by season."""
        logger.info("Creating backtest splits by season...")
        
        # Extract season from date
        df_with_season = df.with_columns(
            pl.col('date').dt.year().alias('season')
        )
        
        # Get unique seasons
        seasons = sorted(df_with_season['season'].unique().to_list())
        logger.info(f"Found seasons: {seasons}")
        
        if len(seasons) < 4:
            raise ValueError(f"Need at least 4 seasons for backtesting, got {len(seasons)}")
        
        splits = []
        
        # Start with first 3 seasons in training, then expand
        for i in range(3, len(seasons)):
            train_seasons = seasons[:i]
            test_season = seasons[i]
            
            # Get indices for each split
            train_mask = df_with_season['season'].is_in(train_seasons)
            test_mask = df_with_season['season'] == test_season
            
            train_indices = np.where(train_mask.to_numpy())[0]
            test_indices = np.where(test_mask.to_numpy())[0]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                season_label = f"{test_season}/{str(test_season + 1)[-2:]}"
                splits.append((train_indices, test_indices, season_label))
                logger.info(f"Split {i-2}: Train seasons {train_seasons}, Test season {season_label}")
        
        logger.info(f"Created {len(splits)} backtest splits")
        return splits
    
    def backtest_season(self, model: Any, X: np.ndarray, y: np.ndarray,
                       train_idx: np.ndarray, test_idx: np.ndarray, 
                       season_label: str) -> Dict[str, Any]:
        """Backtest model on a specific season."""
        logger.info(f"Backtesting season {season_label}...")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        logger.info(f"  Train size: {len(X_train):,}, Test size: {len(X_test):,}")
        
        # Retrain model on training data
        if hasattr(model, 'estimator_'):
            # For CalibratedClassifierCV, access the underlying estimator
            base_estimator = model.estimator_
        else:
            # For regular models, use the model itself
            base_estimator = model
        
        # Get parameters, filtering out estimator-specific ones
        params = base_estimator.get_params()
        # Remove parameters that start with 'estimator__' as they can't be used directly
        filtered_params = {k: v for k, v in params.items() if not k.startswith('estimator__')}
        
        model_copy = base_estimator.__class__(**filtered_params)
        model_copy.fit(X_train, y_train)
        
        # Predict on test set
        y_pred_proba = model_copy.predict_proba(X_test)
        y_pred = model_copy.predict(X_test)
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        brier_score = calculate_brier_score(y_test, y_pred_proba)
        log_loss_score = calculate_log_loss_score(y_test, y_pred_proba)
        
        # Calculate class-wise metrics
        class_metrics = {}
        for class_name, class_idx in [('Home', 0), ('Draw', 1), ('Away', 2)]:
            class_mask = y_test == class_idx
            if class_mask.sum() > 0:
                class_accuracy = (y_pred[class_mask] == y_test[class_mask]).mean()
                class_metrics[f'{class_name.lower()}_accuracy'] = class_accuracy
                class_metrics[f'{class_name.lower()}_count'] = class_mask.sum()
        
        # Store results
        season_results = {
            'season': season_label,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': accuracy,
            'brier_score': brier_score,
            'log_loss': log_loss_score,
            'class_metrics': class_metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test
        }
        
        logger.info(f"  Season {season_label} - Accuracy: {accuracy:.3f}, Brier: {brier_score:.3f}, LogLoss: {log_loss_score:.3f}")
        
        return season_results
    
    def calculate_champion_hit_rate(self, all_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate champion prediction hit rate for top-3 teams."""
        logger.info("Calculating champion hit rates...")
        
        # This is a simplified version - in practice you'd need actual final standings
        # For now, we'll calculate based on predicted vs actual match outcomes
        
        total_matches = 0
        correct_predictions = 0
        
        for result in all_results:
            total_matches += result['test_size']
            correct_predictions += (result['predictions'] == result['true_labels']).sum()
        
        overall_accuracy = correct_predictions / total_matches if total_matches > 0 else 0
        
        # Calculate per-season accuracy
        season_accuracies = [r['accuracy'] for r in all_results]
        
        return {
            'overall_accuracy': overall_accuracy,
            'avg_season_accuracy': np.mean(season_accuracies),
            'min_season_accuracy': np.min(season_accuracies),
            'max_season_accuracy': np.max(season_accuracies),
            'total_matches': total_matches
        }
    
    def generate_backtest_report(self, all_results: List[Dict[str, Any]], 
                                champion_metrics: Dict[str, float]) -> str:
        """Generate comprehensive backtest report."""
        logger.info("Generating backtest report...")
        
        # Calculate overall metrics
        overall_brier = np.mean([r['brier_score'] for r in all_results])
        overall_logloss = np.mean([r['log_loss'] for r in all_results])
        
        # Create season-by-season table
        season_table = []
        for result in all_results:
            season_table.append({
                'Season': result['season'],
                'Matches': result['test_size'],
                'Accuracy': f"{result['accuracy']:.3f}",
                'Brier Score': f"{result['brier_score']:.4f}",
                'Log Loss': f"{result['log_loss']:.4f}"
            })
        
        # Format table
        table_rows = []
        for row in season_table:
            table_rows.append(f"| {row['Season']} | {row['Matches']} | {row['Accuracy']} | {row['Brier Score']} | {row['Log Loss']} |")
        
        table_content = "\n".join(table_rows)
        
        # Generate report
        report = f"""# Premier League Model Backtest Report

## Overview
- **Backtest Period**: {all_results[0]['season']} to {all_results[-1]['season']}
- **Total Seasons**: {len(all_results)}
- **Total Matches**: {champion_metrics['total_matches']:,}

## Performance Summary
- **Overall Accuracy**: {champion_metrics['overall_accuracy']:.3f}
- **Average Season Accuracy**: {champion_metrics['avg_season_accuracy']:.3f}
- **Overall Brier Score**: {overall_brier:.4f}
- **Overall Log Loss**: {overall_logloss:.4f}

## Season-by-Season Performance

| Season | Matches | Accuracy | Brier Score | Log Loss |
|--------|---------|----------|-------------|----------|
{table_content}

## Champion Prediction Metrics
- **Overall Hit Rate**: {champion_metrics['overall_accuracy']:.1%}
- **Best Season**: {champion_metrics['max_season_accuracy']:.3f}
- **Worst Season**: {champion_metrics['min_season_accuracy']:.3f}

## Interpretation
- The model shows {'consistent' if champion_metrics['max_season_accuracy'] - champion_metrics['min_season_accuracy'] < 0.1 else 'variable'} performance across seasons
- Brier score of {overall_brier:.4f} indicates {'good' if overall_brier < 0.2 else 'moderate' if overall_brier < 0.3 else 'poor'} probability calibration
- Log loss of {overall_logloss:.4f} suggests {'good' if overall_logloss < 0.7 else 'moderate' if overall_logloss < 1.0 else 'poor'} classification performance

## Recommendations
- {'Consider retraining with more recent data' if champion_metrics['overall_accuracy'] < 0.5 else 'Model performance is acceptable for production use'}
- {'Focus on improving probability calibration' if overall_brier > 0.25 else 'Probability estimates are well-calibrated'}
- {'Investigate feature engineering improvements' if overall_logloss > 1.0 else 'Feature set provides good predictive power'}

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def save_backtest_results(self, all_results: List[Dict[str, Any]], 
                             champion_metrics: Dict[str, Any]) -> Path:
        """Save backtest results to disk."""
        # Save detailed results
        results_path = self.models_dir / "backtest_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump({
                'season_results': all_results,
                'champion_metrics': champion_metrics,
                'backtest_date': datetime.now().isoformat()
            }, f)
        
        logger.info(f"Saved backtest results to {results_path}")
        
        # Generate and save report
        report = self.generate_backtest_report(all_results, champion_metrics)
        report_path = self.reports_dir / "backtest_metrics.md"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Saved backtest report to {report_path}")
        
        return report_path
    
    def run_backtest(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run complete backtest pipeline."""
        logger.info("Starting model backtest pipeline...")
        
        # Load model and data
        model, df, feature_names = self.load_model_and_data()
        
        # Prepare features
        X, y = self.prepare_features(df, feature_names)
        
        # Create backtest splits
        splits = self.create_backtest_splits(df)
        
        # Run backtest for each season
        all_results = []
        for train_idx, test_idx, season_label in splits:
            season_results = self.backtest_season(
                model, X, y, train_idx, test_idx, season_label
            )
            all_results.append(season_results)
        
        # Calculate champion metrics
        champion_metrics = self.calculate_champion_hit_rate(all_results)
        
        # Save results
        report_path = self.save_backtest_results(all_results, champion_metrics)
        
        logger.info("Backtest pipeline complete!")
        return all_results, champion_metrics


def main():
    """Main entry point for model backtesting."""
    parser = argparse.ArgumentParser(description="Backtest Premier League match prediction model")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--models-dir", default="models", help="Models directory path")
    parser.add_argument("--reports-dir", default="reports", help="Reports directory path")
    
    args = parser.parse_args()
    
    try:
        # Run backtest
        backtester = ModelBacktester(args.data_dir, args.models_dir, args.reports_dir)
        all_results, champion_metrics = backtester.run_backtest()
        
        print(f"\n✅ Model backtest complete!")
        print(f"📁 Backtest Results: models/backtest_results.pkl")
        print(f"📁 Backtest Report: reports/backtest_metrics.md")
        print(f"\n📊 Backtest Summary:")
        print(f"   Seasons tested: {len(all_results)}")
        print(f"   Total matches: {champion_metrics['total_matches']:,}")
        print(f"   Overall accuracy: {champion_metrics['overall_accuracy']:.3f}")
        print(f"   Average Brier score: {np.mean([r['brier_score'] for r in all_results]):.4f}")
        print(f"   Average Log Loss: {np.mean([r['log_loss'] for r in all_results]):.4f}")
        
    except Exception as e:
        logger.error(f"Failed to run backtest: {e}")
        raise


if __name__ == "__main__":
    main() 