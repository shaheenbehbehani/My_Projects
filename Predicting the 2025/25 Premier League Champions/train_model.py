#!/usr/bin/env python3
"""
Train Premier League match prediction model.

This script trains an XGBoost classifier with expanding-window CV by season,
handles class imbalance, and saves the trained model and CV metrics.

Input: data/processed/match_dataset.parquet
Outputs: models/match_model.pkl, reports/cv_metrics.json
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


class MatchModelTrainer:
    """Train Premier League match prediction model with expanding-window CV."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "reports").mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.target_mapping = {'H': 0, 'D': 1, 'A': 2}
        self.feature_exclusions = ['date', 'season', 'home_team', 'away_team', 'y', 'result_label']
        
    def load_match_dataset(self) -> pl.DataFrame:
        """Load the prepared match dataset."""
        dataset_path = self.processed_dir / "match_dataset.parquet"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Match dataset not found: {dataset_path}")
        
        logger.info(f"Loading match dataset from {dataset_path}")
        df = pl.read_parquet(dataset_path)
        logger.info(f"Loaded dataset: {df.height:,} rows, {df.width} columns")
        
        return df
    
    def prepare_features_and_target(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for training."""
        logger.info("Preparing features and target...")
        
        # Get feature columns (exclude identifiers and target)
        feature_columns = [col for col in df.columns if col not in self.feature_exclusions]
        
        # Select features and convert to pandas for sklearn compatibility
        feature_df = df.select(feature_columns)
        X = feature_df.to_pandas()
        
        # Get target
        y = df['result_label'].to_numpy()
        
        # Handle missing values
        X = X.fillna(0)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {np.bincount(y)}")
        logger.info(f"Feature columns: {feature_columns}")
        
        return X, y, feature_columns
    
    def create_model(self) -> Any:
        """Create the prediction model with fallback options."""
        try:
            import xgboost as xgb
            logger.info("Using XGBoost classifier")
            
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=-1
            )
            
        except ImportError:
            logger.warning("XGBoost not available, falling back to HistGradientBoostingClassifier")
            from sklearn.ensemble import HistGradientBoostingClassifier
            
            model = HistGradientBoostingClassifier(
                max_iter=100,
                random_state=42,
                verbose=0
            )
        
        return model
    
    def compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute inverse-frequency sample weights to handle class imbalance."""
        # Count occurrences of each class
        class_counts = np.bincount(y)
        
        # Compute inverse frequency weights
        total_samples = len(y)
        sample_weights = total_samples / (len(class_counts) * class_counts)
        
        # Apply weights to each sample
        weights = sample_weights[y]
        
        logger.info(f"Class counts: {class_counts}")
        logger.info(f"Sample weights: {sample_weights}")
        
        return weights
    
    def perform_expanding_window_cv(self, X: np.ndarray, y: np.ndarray, df: pl.DataFrame) -> Dict[str, Any]:
        """Perform expanding-window cross-validation by season."""
        logger.info("Performing expanding-window CV by season...")
        
        # Get unique seasons sorted
        seasons = sorted(df['season'].unique().to_list())
        logger.info(f"Available seasons: {seasons}")
        
        cv_results = []
        fold_metrics = []
        
        # Start with first 2 seasons for training, then expand
        for i in range(2, len(seasons)):
            train_until = seasons[i-1]
            valid_season = seasons[i]
            
            logger.info(f"Fold {i-1}: Train until {train_until}, Validate {valid_season}")
            
            # Get train/validation split
            train_idx, valid_idx = season_time_splits(df, 'season', train_until, valid_season)
            
            if len(train_idx) == 0 or len(valid_idx) == 0:
                logger.warning(f"Skipping fold {i-1} - insufficient data")
                continue
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_val = y[train_idx], y[valid_idx]
            
            # Compute sample weights for training
            sample_weights = self.compute_sample_weights(y_train)
            
            # Create and train model
            model = self.create_model()
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)
            
            # Calculate metrics
            fold_metric = self._calculate_fold_metrics(y_val, y_pred, y_pred_proba)
            fold_metric['train_until'] = train_until
            fold_metric['valid_season'] = valid_season
            fold_metric['train_size'] = len(train_idx)
            fold_metric['valid_size'] = len(valid_idx)
            
            fold_metrics.append(fold_metric)
            
            # Store predictions for overall evaluation
            cv_results.append({
                'fold': i-1,
                'train_until': train_until,
                'valid_season': valid_season,
                'train_size': len(train_idx),
                'valid_size': len(valid_idx),
                'y_true': y_val.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist(),
                'metrics': fold_metric
            })
            
            logger.info(f"  Fold {i-1} - Accuracy: {fold_metric['accuracy']:.3f}, "
                       f"Brier: {fold_metric['brier_score']:.4f}, Log Loss: {fold_metric['log_loss']:.4f}")
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(cv_results)
        
        cv_summary = {
            'method': 'expanding_window_cv',
            'n_folds': len(fold_metrics),
            'seasons': seasons,
            'fold_metrics': fold_metrics,
            'overall_metrics': overall_metrics,
            'cv_results': cv_results
        }
        
        logger.info(f"CV complete: {len(fold_metrics)} folds")
        logger.info(f"Overall accuracy: {overall_metrics['accuracy']:.3f}")
        logger.info(f"Overall Brier score: {overall_metrics['brier_score']:.4f}")
        logger.info(f"Overall log loss: {overall_metrics['log_loss']:.4f}")
        
        return cv_summary
    
    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for a single CV fold."""
        from sklearn.metrics import accuracy_score
        
        # Convert to one-hot encoding for Brier score
        y_true_onehot = np.eye(3)[y_true]
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'brier_score': compute_brier(y_true_onehot, y_pred_proba),
            'log_loss': compute_logloss(y_true, y_pred_proba)
        }
        
        return metrics
    
    def _calculate_overall_metrics(self, cv_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall metrics across all CV folds."""
        all_y_true = []
        all_y_pred = []
        all_y_pred_proba = []
        
        for result in cv_results:
            all_y_true.extend(result['y_true'])
            all_y_pred.extend(result['y_pred'])
            all_y_pred_proba.extend(result['y_pred_proba'])
        
        # Convert to numpy arrays
        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)
        y_pred_proba = np.array(all_y_pred_proba)
        
        # Calculate overall metrics
        return self._calculate_fold_metrics(y_true, y_pred, y_pred_proba)
    
    def train_final_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train the final model on all available data."""
        logger.info("Training final model on all data...")
        
        # Compute sample weights
        sample_weights = self.compute_sample_weights(y)
        
        # Create and train model
        model = self.create_model()
        model.fit(X, y, sample_weight=sample_weights)
        
        logger.info("Final model training complete")
        return model
    
    def save_model_and_metrics(self, model: Any, cv_metrics: Dict[str, Any], feature_names: List[str]) -> None:
        """Save the trained model and CV metrics."""
        logger.info("Saving model and metrics...")
        
        # Save model
        import joblib
        model_path = self.models_dir / "match_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save feature info
        feature_info = {
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'target_mapping': self.target_mapping,
            'feature_exclusions': self.feature_exclusions
        }
        
        feature_path = self.models_dir / "feature_info.pkl"
        joblib.dump(feature_info, feature_path)
        logger.info(f"Feature info saved to {feature_path}")
        
        # Save CV metrics
        metrics_path = self.data_dir / "reports" / "cv_metrics.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_cv_metrics = {}
        for key, value in cv_metrics.items():
            if key == 'cv_results':
                # Clean up cv_results for JSON serialization
                serializable_results = []
                for result in value:
                    serializable_result = {k: v for k, v in result.items() 
                                         if k not in ['y_true', 'y_pred', 'y_pred_proba']}
                    serializable_results.append(serializable_result)
                serializable_cv_metrics[key] = serializable_results
            else:
                serializable_cv_metrics[key] = value
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_cv_metrics, f, indent=2, default=str)
        
        logger.info(f"CV metrics saved to {metrics_path}")
    
    def train_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Train the complete model pipeline."""
        logger.info("🚀 Starting model training pipeline...")
        
        # Load data
        df = self.load_match_dataset()
        
        # Prepare features and target
        X, y, feature_names = self.prepare_features_and_target(df)
        
        # Perform expanding-window CV
        cv_metrics = self.perform_expanding_window_cv(X, y, df)
        
        # Train final model
        final_model = self.train_final_model(X, y)
        
        # Save model and metrics
        self.save_model_and_metrics(final_model, cv_metrics, feature_names)
        
        logger.info("✅ Model training pipeline completed successfully!")
        return final_model, cv_metrics


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(description="Train Premier League match prediction model")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--models-dir", default="models", help="Models directory path")
    
    args = parser.parse_args()
    
    try:
        # Train model
        trainer = MatchModelTrainer(args.data_dir, args.models_dir)
        model, cv_metrics = trainer.train_model()
        
        print(f"\n✅ Model training complete!")
        print(f"🤖 Model: {args.models_dir}/match_model.pkl")
        print(f"📊 CV Metrics: {args.data_dir}/reports/cv_metrics.json")
        
        # Print CV summary
        if 'overall_metrics' in cv_metrics:
            metrics = cv_metrics['overall_metrics']
            print(f"\n📈 CV Performance Summary:")
            print(f"   Folds: {cv_metrics.get('n_folds', 'N/A')}")
            print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
            print(f"   Brier Score: {metrics.get('brier_score', 'N/A'):.4f}")
            print(f"   Log Loss: {metrics.get('log_loss', 'N/A'):.4f}")
        
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise


if __name__ == "__main__":
    main() 