#!/usr/bin/env python3
"""
Comprehensive model evaluation for Premier League winner prediction.

This script evaluates trained models using:
- Multiple cross-validation strategies
- Comprehensive metrics (accuracy, Brier score, log loss)
- Probability calibration
- Feature importance analysis
- Out-of-sample performance tracking
"""

import polars as pl
import numpy as np
import pandas as pd
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import argparse
from datetime import datetime

# Import our utilities
from .utils import (
    create_season_based_splits,
    create_rolling_window_splits,
    calculate_classification_metrics,
    calibrate_probabilities_with_temperature,
    create_feature_importance_plot,
    set_random_seed
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
set_random_seed(42)


class ModelEvaluator:
    """Comprehensive model evaluation for Premier League prediction models."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models", outputs_dir: str = "outputs"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.outputs_dir = Path(outputs_dir)
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure output directories exist
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "figures").mkdir(exist_ok=True)
        (self.outputs_dir / "reports").mkdir(exist_ok=True)
        
    def load_match_dataset(self) -> pl.DataFrame:
        """Load the prepared match dataset."""
        dataset_path = self.processed_dir / "match_dataset.parquet"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Match dataset not found: {dataset_path}")
        
        logger.info(f"Loading match dataset from {dataset_path}")
        df = pl.read_parquet(dataset_path)
        logger.info(f"Loaded dataset: {df.height:,} rows, {df.width} columns")
        
        return df
    
    def load_trained_model(self, model_name: str = "match_model") -> Tuple[Any, Dict[str, Any]]:
        """Load a trained model and its metadata."""
        model_path = self.models_dir / f"{model_name}.pkl"
        feature_path = self.models_dir / "feature_info.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature info not found: {feature_path}")
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load feature info
        with open(feature_path, 'rb') as f:
            feature_info = pickle.load(f)
        
        logger.info(f"Loaded model: {model_path}")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Features: {feature_info['n_features']}")
        
        return model, feature_info
    
    def prepare_features_and_target(self, df: pl.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for evaluation."""
        logger.info("Preparing features and target for evaluation...")
        
        # Select only the features used in training
        feature_df = df.select(feature_names)
        
        # Convert to pandas for sklearn compatibility
        X = feature_df.to_pandas()
        y = df['result_label'].to_numpy()
        
        # Handle missing values
        X = X.fillna(0)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {np.bincount(y)}")
        
        return X, y
    
    def evaluate_with_season_splits(self, model: Any, X: np.ndarray, y: np.ndarray, df: pl.DataFrame) -> Dict[str, Any]:
        """Evaluate model using season-based cross-validation."""
        logger.info("Evaluating with season-based splits...")
        
        # Create season-based splits
        splits = create_season_based_splits(df, min_seasons=2)
        
        if not splits:
            logger.warning("No season-based splits created, skipping this evaluation")
            return {}
        
        fold_metrics = []
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        
        for i, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Fold {i+1}: Train size {len(train_idx)}, Val size {len(val_idx)}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model on this fold
            fold_model = self._clone_model(model)
            fold_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = fold_model.predict(X_val)
            y_pred_proba = fold_model.predict_proba(X_val)
            
            # Calculate metrics
            fold_metric = calculate_classification_metrics(y_val, y_pred, y_pred_proba)
            fold_metrics.append(fold_metric)
            
            # Store predictions for overall evaluation
            all_predictions.extend(y_pred)
            all_probabilities.extend(y_pred_proba)
            all_true_labels.extend(y_val)
        
        # Calculate overall metrics
        overall_metrics = calculate_classification_metrics(
            np.array(all_true_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        season_eval = {
            'method': 'season_based_cv',
            'n_splits': len(splits),
            'fold_metrics': fold_metrics,
            'overall_metrics': overall_metrics,
            'splits_info': [{'train_size': len(train), 'val_size': len(val)} for train, val in splits]
        }
        
        logger.info(f"Season-based CV complete: {len(splits)} folds")
        logger.info(f"Overall accuracy: {overall_metrics['accuracy']:.3f}")
        
        return season_eval
    
    def evaluate_with_rolling_windows(self, model: Any, X: np.ndarray, y: np.ndarray, df: pl.DataFrame) -> Dict[str, Any]:
        """Evaluate model using rolling window cross-validation."""
        logger.info("Evaluating with rolling window splits...")
        
        # Create rolling window splits
        splits = create_rolling_window_splits(df, window_size=800, step_size=200, min_train_size=400)
        
        if not splits:
            logger.warning("No rolling window splits created, skipping this evaluation")
            return {}
        
        fold_metrics = []
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        
        for i, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Window {i+1}: Train size {len(train_idx)}, Val size {len(val_idx)}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model on this fold
            fold_model = self._clone_model(model)
            fold_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = fold_model.predict(X_val)
            y_pred_proba = fold_model.predict_proba(X_val)
            
            # Calculate metrics
            fold_metric = calculate_classification_metrics(y_val, y_pred, y_pred_proba)
            fold_metrics.append(fold_metric)
            
            # Store predictions for overall evaluation
            all_predictions.extend(y_pred)
            all_probabilities.extend(y_pred_proba)
            all_true_labels.extend(y_val)
        
        # Calculate overall metrics
        overall_metrics = calculate_classification_metrics(
            np.array(all_true_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        rolling_eval = {
            'method': 'rolling_window_cv',
            'n_splits': len(splits),
            'fold_metrics': fold_metrics,
            'overall_metrics': overall_metrics,
            'splits_info': [{'train_size': len(train), 'val_size': len(val)} for train, val in splits]
        }
        
        logger.info(f"Rolling window CV complete: {len(splits)} windows")
        logger.info(f"Overall accuracy: {overall_metrics['accuracy']:.3f}")
        
        return rolling_eval
    
    def evaluate_probability_calibration(self, model: Any, X: np.ndarray, y: np.ndarray, df: pl.DataFrame) -> Dict[str, Any]:
        """Evaluate and improve probability calibration."""
        logger.info("Evaluating probability calibration...")
        
        # Use a simple train/validation split for calibration
        n_samples = len(X)
        split_point = int(0.8 * n_samples)
        
        X_train, X_val = X[:split_point], X[split_point:]
        y_train, y_val = y[:split_point], y[split_point:]
        
        # Train model
        cal_model = self._clone_model(model)
        cal_model.fit(X_train, y_train)
        
        # Get raw predictions
        y_pred_proba_raw = cal_model.predict_proba(X_val)
        
        # Calibrate probabilities using temperature scaling
        y_pred_proba_cal, best_temp, cal_info = calibrate_probabilities_with_temperature(
            y_val, y_pred_proba_raw
        )
        
        # Calculate metrics before and after calibration
        y_pred_raw = np.argmax(y_pred_proba_raw, axis=1)
        y_pred_cal = np.argmax(y_pred_proba_cal, axis=1)
        
        metrics_raw = calculate_classification_metrics(y_val, y_pred_raw, y_pred_proba_raw)
        metrics_cal = calculate_classification_metrics(y_val, y_pred_cal, y_pred_proba_cal)
        
        calibration_eval = {
            'method': 'temperature_scaling',
            'best_temperature': best_temp,
            'pre_calibration': metrics_raw,
            'post_calibration': metrics_cal,
            'calibration_info': cal_info,
            'improvement': {
                'brier_score': metrics_raw['brier_score'] - metrics_cal['brier_score'],
                'log_loss': metrics_raw['log_loss'] - metrics_cal['log_loss']
            }
        }
        
        logger.info(f"Calibration complete - Best temperature: {best_temp:.3f}")
        logger.info(f"Brier score improvement: {calibration_eval['improvement']['brier_score']:.4f}")
        logger.info(f"Log loss improvement: {calibration_eval['improvement']['log_loss']:.4f}")
        
        return calibration_eval
    
    def analyze_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance from the trained model."""
        logger.info("Analyzing feature importance...")
        
        # Get feature importance (XGBoost specific)
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance_scores = np.abs(model.coef_[0])  # For linear models
        else:
            logger.warning("Model doesn't have feature importance attribute")
            return {}
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        importance_path = self.outputs_dir / "reports" / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        
        # Create feature importance plot
        plot_path = self.outputs_dir / "figures" / "feature_importance.png"
        create_feature_importance_plot(
            feature_names, importance_scores, top_n=20, save_path=str(plot_path)
        )
        
        feature_analysis = {
            'top_features': importance_df.head(20).to_dict('records'),
            'importance_path': str(importance_path),
            'plot_path': str(plot_path),
            'total_features': len(feature_names)
        }
        
        logger.info(f"Feature importance analysis complete")
        logger.info(f"Top feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']:.4f})")
        
        return feature_analysis
    
    def _clone_model(self, model: Any) -> Any:
        """Create a clone of the model for cross-validation."""
        try:
            from sklearn.base import clone
            return clone(model)
        except ImportError:
            # Fallback for non-sklearn models
            import copy
            return copy.deepcopy(model)
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        logger.info("Generating evaluation report...")
        
        report_lines = []
        report_lines.append("# Premier League Model Evaluation Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Overall summary
        report_lines.append("## Evaluation Summary")
        report_lines.append("")
        
        if 'season_based_cv' in evaluation_results:
            season_eval = evaluation_results['season_based_cv']
            report_lines.append(f"**Season-based Cross-validation:**")
            report_lines.append(f"- Folds: {season_eval['n_splits']}")
            report_lines.append(f"- Overall Accuracy: {season_eval['overall_metrics']['accuracy']:.3f}")
            report_lines.append(f"- Overall Brier Score: {season_eval['overall_metrics']['brier_score']:.4f}")
            report_lines.append(f"- Overall Log Loss: {season_eval['overall_metrics']['log_loss']:.4f}")
            report_lines.append("")
        
        if 'rolling_window_cv' in evaluation_results:
            rolling_eval = evaluation_results['rolling_window_cv']
            report_lines.append(f"**Rolling Window Cross-validation:**")
            report_lines.append(f"- Windows: {rolling_eval['n_splits']}")
            report_lines.append(f"- Overall Accuracy: {rolling_eval['overall_metrics']['accuracy']:.3f}")
            report_lines.append(f"- Overall Brier Score: {rolling_eval['overall_metrics']['brier_score']:.4f}")
            report_lines.append(f"- Overall Log Loss: {rolling_eval['overall_metrics']['log_loss']:.4f}")
            report_lines.append("")
        
        if 'calibration' in evaluation_results:
            cal_eval = evaluation_results['calibration']
            report_lines.append(f"**Probability Calibration:**")
            report_lines.append(f"- Method: {cal_eval['method']}")
            report_lines.append(f"- Best Temperature: {cal_eval['best_temperature']:.3f}")
            report_lines.append(f"- Brier Score Improvement: {cal_eval['improvement']['brier_score']:.4f}")
            report_lines.append(f"- Log Loss Improvement: {cal_eval['improvement']['log_loss']:.4f}")
            report_lines.append("")
        
        if 'feature_importance' in evaluation_results:
            feat_eval = evaluation_results['feature_importance']
            report_lines.append(f"**Feature Importance:**")
            report_lines.append(f"- Total Features: {feat_eval['total_features']}")
            report_lines.append(f"- Top Features: {', '.join([f['feature'] for f in feat_eval['top_features'][:5]])}")
            report_lines.append("")
        
        # Detailed fold results
        if 'season_based_cv' in evaluation_results:
            report_lines.append("## Season-based CV Detailed Results")
            report_lines.append("")
            report_lines.append("| Fold | Train Size | Val Size | Accuracy | Brier Score | Log Loss |")
            report_lines.append("|------|------------|----------|----------|-------------|----------|")
            
            for i, (fold_metric, split_info) in enumerate(zip(
                evaluation_results['season_based_cv']['fold_metrics'],
                evaluation_results['season_based_cv']['splits_info']
            )):
                report_lines.append(
                    f"| {i+1} | {split_info['train_size']} | {split_info['val_size']} | "
                    f"{fold_metric['accuracy']:.3f} | {fold_metric['brier_score']:.4f} | "
                    f"{fold_metric['log_loss']:.4f} |"
                )
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        
        if 'calibration' in evaluation_results:
            cal_eval = evaluation_results['calibration']
            if cal_eval['improvement']['brier_score'] > 0.01:
                report_lines.append("- ✅ Probability calibration significantly improved model performance")
            else:
                report_lines.append("- ⚠️ Probability calibration provided minimal improvement")
        
        if 'season_based_cv' in evaluation_results:
            season_eval = evaluation_results['season_based_cv']
            if season_eval['overall_metrics']['accuracy'] > 0.6:
                report_lines.append("- ✅ Model shows good predictive performance across seasons")
            else:
                report_lines.append("- ⚠️ Model performance could be improved")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("*Report generated automatically by Premier League Model Evaluator*")
        
        return "\n".join(report_lines)
    
    def save_evaluation_results(self, evaluation_results: Dict[str, Any]) -> None:
        """Save evaluation results to files."""
        logger.info("Saving evaluation results...")
        
        # Save detailed results as JSON
        results_path = self.outputs_dir / "reports" / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Generate and save report
        report_content = self.generate_evaluation_report(evaluation_results)
        report_path = self.outputs_dir / "reports" / "model_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Report saved to {report_path}")
    
    def run_comprehensive_evaluation(self, model_name: str = "match_model") -> Dict[str, Any]:
        """Run comprehensive model evaluation pipeline."""
        logger.info("Starting comprehensive model evaluation...")
        
        # Load data and model
        df = self.load_match_dataset()
        model, feature_info = self.load_trained_model(model_name)
        
        # Prepare features
        X, y = self.prepare_features_and_target(df, feature_info['feature_names'])
        
        # Run evaluations
        evaluation_results = {}
        
        # Season-based CV
        try:
            evaluation_results['season_based_cv'] = self.evaluate_with_season_splits(model, X, y, df)
        except Exception as e:
            logger.error(f"Season-based CV failed: {e}")
            evaluation_results['season_based_cv'] = {'error': str(e)}
        
        # Rolling window CV
        try:
            evaluation_results['rolling_window_cv'] = self.evaluate_with_rolling_windows(model, X, y, df)
        except Exception as e:
            logger.error(f"Rolling window CV failed: {e}")
            evaluation_results['rolling_window_cv'] = {'error': str(e)}
        
        # Probability calibration
        try:
            evaluation_results['calibration'] = self.evaluate_probability_calibration(model, X, y, df)
        except Exception as e:
            logger.error(f"Probability calibration failed: {e}")
            evaluation_results['calibration'] = {'error': str(e)}
        
        # Feature importance
        try:
            evaluation_results['feature_importance'] = self.analyze_feature_importance(model, feature_info['feature_names'])
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            evaluation_results['feature_importance'] = {'error': str(e)}
        
        # Save results
        self.save_evaluation_results(evaluation_results)
        
        logger.info("Comprehensive evaluation complete!")
        return evaluation_results


def main():
    """Main entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Premier League prediction model")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--models-dir", default="models", help="Models directory path")
    parser.add_argument("--outputs-dir", default="outputs", help="Outputs directory path")
    parser.add_argument("--model-name", default="match_model", help="Name of model to evaluate")
    
    args = parser.parse_args()
    
    try:
        # Run evaluation
        evaluator = ModelEvaluator(args.data_dir, args.models_dir, args.outputs_dir)
        results = evaluator.run_comprehensive_evaluation(args.model_name)
        
        print(f"\n✅ Model evaluation complete!")
        print(f"📁 Results: outputs/reports/evaluation_results.json")
        print(f"📁 Report: outputs/reports/model_evaluation_report.md")
        print(f"📊 Feature importance: outputs/figures/feature_importance.png")
        
        # Print summary
        if 'season_based_cv' in results and 'overall_metrics' in results['season_based_cv']:
            metrics = results['season_based_cv']['overall_metrics']
            print(f"\n📊 Season-based CV Performance:")
            print(f"   Accuracy: {metrics['accuracy']:.3f}")
            print(f"   Brier Score: {metrics['brier_score']:.4f}")
            print(f"   Log Loss: {metrics['log_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        raise


if __name__ == "__main__":
    main() 