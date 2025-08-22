#!/usr/bin/env python3
"""
Calibrate Premier League match prediction model.

This script calibrates the trained model using the last available season
as validation data, comparing pre/post calibration performance.

Input: models/match_model.pkl, data/processed/match_dataset.parquet
Outputs: models/match_model_calibrated.pkl, reports/calibration_summary.md
"""

import polars as pl
import numpy as np
import pandas as pd
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import sklearn calibration
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    CalibratedClassifierCV = None
    IsotonicRegression = None

from .utils import (
    season_time_splits,
    compute_brier,
    compute_logloss,
    set_random_seed
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
set_random_seed(42)


class ModelCalibrator:
    """Calibrate Premier League match prediction model."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "reports").mkdir(parents=True, exist_ok=True)
        
        # Calibration parameters
        self.feature_exclusions = ['date', 'season', 'home_team', 'away_team', 'y', 'result_label']
        
    def load_trained_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Load the trained model and feature information."""
        import joblib
        
        model_path = self.models_dir / "match_model.pkl"
        feature_path = self.models_dir / "feature_info.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature info not found: {feature_path}")
        
        # Load model
        model = joblib.load(model_path)
        logger.info(f"Loaded model: {type(model).__name__}")
        
        # Load feature info
        feature_info = joblib.load(feature_path)
        logger.info(f"Loaded feature info: {feature_info['n_features']} features")
        
        return model, feature_info
    
    def load_match_dataset(self) -> pl.DataFrame:
        """Load the prepared match dataset."""
        dataset_path = self.processed_dir / "match_dataset.parquet"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Match dataset not found: {dataset_path}")
        
        logger.info(f"Loading match dataset from {dataset_path}")
        df = pl.read_parquet(dataset_path)
        logger.info(f"Loaded dataset: {df.height:,} rows, {df.width} columns")
        
        return df
    
    def prepare_calibration_data(self, df: pl.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for calibration using last available season."""
        logger.info("Preparing calibration data...")
        
        # Get unique seasons sorted
        seasons = sorted(df['season'].unique().to_list())
        logger.info(f"Available seasons: {seasons}")
        
        if len(seasons) < 2:
            raise ValueError("Need at least 2 seasons for calibration")
        
        # Use last season for validation/calibration
        train_until = seasons[-2]
        valid_season = seasons[-1]
        
        logger.info(f"Calibration split: Train until {train_until}, Calibrate on {valid_season}")
        
        # Get train/validation split
        train_idx, valid_idx = season_time_splits(df, 'season', train_until, valid_season)
        
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise ValueError("Insufficient data for calibration split")
        
        # Split data
        X_train = df.select(feature_names).to_pandas().iloc[train_idx]
        y_train = df['result_label'].to_numpy()[train_idx]
        
        X_cal = df.select(feature_names).to_pandas().iloc[valid_idx]
        y_cal = df['result_label'].to_numpy()[valid_idx]
        
        # Handle missing values
        X_train = X_train.fillna(0)
        X_cal = X_cal.fillna(0)
        
        logger.info(f"Training data: {X_train.shape}")
        logger.info(f"Calibration data: {X_cal.shape}")
        
        return X_train, y_train, X_cal, y_cal
    
    def create_calibration_model(self) -> Any:
        """Create isotonic calibration model."""
        try:
            # Create isotonic regression for each class (one-vs-rest approach)
            calibration_model = CalibratedClassifierCV(
                estimator=None,  # Will be set when fitting
                method='isotonic',
                cv='prefit'
            )
            
            logger.info("Created isotonic calibration model")
            return calibration_model
            
        except ImportError:
            logger.error("Failed to import calibration libraries")
            raise
    
    def calibrate_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                       X_cal: np.ndarray, y_cal: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Calibrate the model using validation data."""
        logger.info("Calibrating model...")
        
        # Train base model on training data
        logger.info("Training base model on training data...")
        model.fit(X_train, y_train)
        
        # Get raw predictions on calibration data
        y_pred_proba_raw = model.predict_proba(X_cal)
        y_pred_raw = model.predict(X_cal)
        
        # Calculate pre-calibration metrics
        y_cal_onehot = np.eye(3)[y_cal]
        pre_cal_metrics = {
            'accuracy': (y_pred_raw == y_cal).mean(),
            'brier_score': compute_brier(y_cal_onehot, y_pred_proba_raw),
            'log_loss': compute_logloss(y_cal, y_pred_proba_raw)
        }
        
        logger.info(f"Pre-calibration metrics:")
        logger.info(f"  Accuracy: {pre_cal_metrics['accuracy']:.3f}")
        logger.info(f"  Brier Score: {pre_cal_metrics['brier_score']:.4f}")
        logger.info(f"  Log Loss: {pre_cal_metrics['log_loss']:.4f}")
        
        # Create and fit calibration model
        calibration_model = self.create_calibration_model()
        
        # Fit calibration model using the trained base model
        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method='isotonic',
            cv='prefit'
        )
        
        # Fit calibration model
        calibrated_model.fit(X_cal, y_cal)
        
        # Get calibrated predictions
        y_pred_proba_cal = calibrated_model.predict_proba(X_cal)
        y_pred_cal = calibrated_model.predict(X_cal)
        
        # Calculate post-calibration metrics
        post_cal_metrics = {
            'accuracy': (y_pred_cal == y_cal).mean(),
            'brier_score': compute_brier(y_cal_onehot, y_pred_proba_cal),
            'log_loss': compute_logloss(y_cal, y_pred_proba_cal)
        }
        
        logger.info(f"Post-calibration metrics:")
        logger.info(f"  Accuracy: {post_cal_metrics['accuracy']:.3f}")
        logger.info(f"  Brier Score: {post_cal_metrics['brier_score']:.4f}")
        logger.info(f"  Log Loss: {post_cal_metrics['log_loss']:.4f}")
        
        # Calculate improvements
        improvements = {
            'brier_score': pre_cal_metrics['brier_score'] - post_cal_metrics['brier_score'],
            'log_loss': pre_cal_metrics['log_loss'] - post_cal_metrics['log_loss'],
            'accuracy': post_cal_metrics['accuracy'] - pre_cal_metrics['accuracy']
        }
        
        logger.info(f"Calibration improvements:")
        logger.info(f"  Brier Score: {improvements['brier_score']:.4f}")
        logger.info(f"  Log Loss: {improvements['log_loss']:.4f}")
        logger.info(f"  Accuracy: {improvements['accuracy']:.4f}")
        
        # Compile calibration results
        calibration_results = {
            'method': 'isotonic',
            'calibration_season': X_cal.shape[0],
            'pre_calibration': pre_cal_metrics,
            'post_calibration': post_cal_metrics,
            'improvements': improvements,
            'calibration_model': calibrated_model
        }
        
        return calibrated_model, calibration_results
    
    def save_calibrated_model(self, calibrated_model: Any) -> Path:
        """Save the calibrated model."""
        import joblib
        
        model_path = self.models_dir / "match_model_calibrated.pkl"
        
        logger.info(f"Saving calibrated model to {model_path}")
        joblib.dump(calibrated_model, model_path)
        
        return model_path
    
    def generate_calibration_summary(self, calibration_results: Dict[str, Any]) -> str:
        """Generate calibration summary report."""
        logger.info("Generating calibration summary...")
        
        summary_lines = []
        summary_lines.append("# Model Calibration Summary")
        summary_lines.append("")
        summary_lines.append(f"**Calibration Method:** {calibration_results['method']}")
        summary_lines.append(f"**Calibration Data Size:** {calibration_results['calibration_season']} samples")
        summary_lines.append("")
        
        # Pre-calibration metrics
        pre_metrics = calibration_results['pre_calibration']
        summary_lines.append("## Pre-Calibration Performance")
        summary_lines.append("")
        summary_lines.append(f"- **Accuracy:** {pre_metrics['accuracy']:.3f}")
        summary_lines.append(f"- **Brier Score:** {pre_metrics['brier_score']:.4f}")
        summary_lines.append(f"- **Log Loss:** {pre_metrics['log_loss']:.4f}")
        summary_lines.append("")
        
        # Post-calibration metrics
        post_metrics = calibration_results['post_calibration']
        summary_lines.append("## Post-Calibration Performance")
        summary_lines.append("")
        summary_lines.append(f"- **Accuracy:** {post_metrics['accuracy']:.3f}")
        summary_lines.append(f"- **Brier Score:** {post_metrics['brier_score']:.4f}")
        summary_lines.append(f"- **Log Loss:** {post_metrics['log_loss']:.4f}")
        summary_lines.append("")
        
        # Improvements
        improvements = calibration_results['improvements']
        summary_lines.append("## Calibration Improvements")
        summary_lines.append("")
        summary_lines.append(f"- **Brier Score Improvement:** {improvements['brier_score']:.4f}")
        summary_lines.append(f"- **Log Loss Improvement:** {improvements['log_loss']:.4f}")
        summary_lines.append(f"- **Accuracy Change:** {improvements['accuracy']:.4f}")
        summary_lines.append("")
        
        # Interpretation
        summary_lines.append("## Interpretation")
        summary_lines.append("")
        if improvements['brier_score'] > 0.01:
            summary_lines.append("✅ **Significant improvement** in probability calibration (Brier score)")
        elif improvements['brier_score'] > 0.001:
            summary_lines.append("⚠️ **Moderate improvement** in probability calibration")
        else:
            summary_lines.append("❌ **Minimal improvement** in probability calibration")
        
        if improvements['log_loss'] > 0.01:
            summary_lines.append("✅ **Significant improvement** in log loss")
        elif improvements['log_loss'] > 0.001:
            summary_lines.append("⚠️ **Moderate improvement** in log loss")
        else:
            summary_lines.append("❌ **Minimal improvement** in log loss")
        
        summary_lines.append("")
        summary_lines.append("---")
        summary_lines.append("*Report generated automatically by Model Calibrator*")
        
        return "\n".join(summary_lines)
    
    def save_calibration_summary(self, summary_content: str) -> Path:
        """Save calibration summary to markdown file."""
        summary_path = self.data_dir / "reports" / "calibration_summary.md"
        
        logger.info(f"Saving calibration summary to {summary_path}")
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        return summary_path
    
    def run_calibration(self) -> Tuple[Any, Dict[str, Any]]:
        """Run the complete calibration pipeline."""
        logger.info("🚀 Starting model calibration pipeline...")
        
        # Load trained model
        model, feature_info = self.load_trained_model()
        
        # Load dataset
        df = self.load_match_dataset()
        
        # Prepare calibration data
        X_train, y_train, X_cal, y_cal = self.prepare_calibration_data(df, feature_info['feature_names'])
        
        # Calibrate model
        calibrated_model, calibration_results = self.calibrate_model(
            model, X_train, y_train, X_cal, y_cal
        )
        
        # Save calibrated model
        model_path = self.save_calibrated_model(calibrated_model)
        
        # Generate and save summary
        summary_content = self.generate_calibration_summary(calibration_results)
        summary_path = self.save_calibration_summary(summary_content)
        
        logger.info("✅ Model calibration pipeline completed successfully!")
        return calibrated_model, calibration_results


def main():
    """Main entry point for model calibration."""
    parser = argparse.ArgumentParser(description="Calibrate Premier League match prediction model")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--models-dir", default="models", help="Models directory path")
    
    args = parser.parse_args()
    
    try:
        # Run calibration
        calibrator = ModelCalibrator(args.data_dir, args.models_dir)
        calibrated_model, results = calibrator.run_calibration()
        
        print(f"\n✅ Model calibration complete!")
        print(f"🤖 Calibrated Model: {args.models_dir}/match_model_calibrated.pkl")
        print(f"📄 Summary: {args.data_dir}/reports/calibration_summary.md")
        
        # Print key results
        improvements = results['improvements']
        print(f"\n📊 Calibration Results:")
        print(f"   Brier Score Improvement: {improvements['brier_score']:.4f}")
        print(f"   Log Loss Improvement: {improvements['log_loss']:.4f}")
        print(f"   Accuracy Change: {improvements['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to calibrate model: {e}")
        raise


if __name__ == "__main__":
    main() 