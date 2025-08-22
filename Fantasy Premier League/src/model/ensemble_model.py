#!/usr/bin/env python3
"""
Model ensemble and stacking for Premier League winner prediction.

This script implements:
- Multiple base models (XGBoost, Random Forest, Logistic Regression)
- Ensemble methods (voting, averaging)
- Stacking with meta-learner
- Cross-validation for ensemble training
- Performance comparison between single and ensemble models
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
    calculate_classification_metrics,
    set_random_seed
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
set_random_seed(42)


class ModelEnsemble:
    """Ensemble of multiple models for Premier League prediction."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models", outputs_dir: str = "outputs"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.outputs_dir = Path(outputs_dir)
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure output directories exist
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "reports").mkdir(exist_ok=True)
        
        # Initialize models
        self.base_models = {}
        self.ensemble_model = None
        self.meta_learner = None
        
    def load_match_dataset(self) -> pl.DataFrame:
        """Load the prepared match dataset."""
        dataset_path = self.processed_dir / "match_dataset.parquet"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Match dataset not found: {dataset_path}")
        
        logger.info(f"Loading match dataset from {dataset_path}")
        df = pl.read_parquet(dataset_path)
        logger.info(f"Loaded dataset: {df.height:,} rows, {df.width} columns")
        
        return df
    
    def create_base_models(self) -> Dict[str, Any]:
        """Create multiple base models for ensemble."""
        logger.info("Creating base models...")
        
        try:
            import xgboost as xgb
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            
            # XGBoost
            self.base_models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            )
            
            # Random Forest
            self.base_models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Logistic Regression
            self.base_models['logistic_regression'] = LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='multinomial'
            )
            
            # Support Vector Machine (with probability)
            self.base_models['svm'] = SVC(
                probability=True,
                random_state=42,
                kernel='rbf'
            )
            
            logger.info(f"Created {len(self.base_models)} base models")
            return self.base_models
            
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            raise
    
    def prepare_features_and_target(self, df: pl.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training."""
        logger.info("Preparing features and target...")
        
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
    
    def train_base_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train all base models."""
        logger.info("Training base models...")
        
        trained_models = {}
        training_metrics = {}
        
        for name, model in self.base_models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X, y)
                trained_models[name] = model
                
                # Evaluate on training data
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)
                
                # Calculate metrics
                metrics = calculate_classification_metrics(y, y_pred, y_pred_proba)
                training_metrics[name] = metrics
                
                logger.info(f"  {name} - Accuracy: {metrics['accuracy']:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                continue
        
        self.base_models = trained_models
        
        logger.info(f"Trained {len(trained_models)} base models successfully")
        return training_metrics
    
    def create_voting_ensemble(self, voting: str = 'soft') -> Any:
        """Create a voting ensemble of base models."""
        try:
            from sklearn.ensemble import VotingClassifier
            
            # Create voting classifier
            estimators = [(name, model) for name, model in self.base_models.items()]
            
            self.ensemble_model = VotingClassifier(
                estimators=estimators,
                voting=voting,
                n_jobs=-1
            )
            
            logger.info(f"Created {voting} voting ensemble with {len(estimators)} models")
            return self.ensemble_model
            
        except ImportError as e:
            logger.error(f"Failed to create voting ensemble: {e}")
            raise
    
    def create_stacking_ensemble(self, meta_learner_type: str = 'logistic') -> Any:
        """Create a stacking ensemble with meta-learner."""
        try:
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression
            
            # Create meta-learner
            if meta_learner_type == 'logistic':
                self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
            else:
                raise ValueError(f"Unsupported meta-learner: {meta_learner_type}")
            
            # Create stacking classifier
            estimators = [(name, model) for name, model in self.base_models.items()]
            
            self.ensemble_model = StackingClassifier(
                estimators=estimators,
                final_estimator=self.meta_learner,
                cv=3,
                n_jobs=-1
            )
            
            logger.info(f"Created stacking ensemble with {meta_learner_type} meta-learner")
            return self.ensemble_model
            
        except ImportError as e:
            logger.error(f"Failed to create stacking ensemble: {e}")
            raise
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the ensemble model."""
        if self.ensemble_model is None:
            raise ValueError("No ensemble model created. Call create_voting_ensemble or create_stacking_ensemble first.")
        
        logger.info("Training ensemble model...")
        
        # Train ensemble
        self.ensemble_model.fit(X, y)
        
        # Evaluate ensemble
        y_pred = self.ensemble_model.predict(X)
        y_pred_proba = self.ensemble_model.predict_proba(X)
        
        # Calculate metrics
        ensemble_metrics = calculate_classification_metrics(y, y_pred, y_pred_proba)
        
        logger.info(f"Ensemble training complete")
        logger.info(f"  Accuracy: {ensemble_metrics['accuracy']:.3f}")
        logger.info(f"  Brier Score: {ensemble_metrics['brier_score']:.4f}")
        
        return ensemble_metrics
    
    def evaluate_ensemble_cv(self, X: np.ndarray, y: np.ndarray, df: pl.DataFrame) -> Dict[str, Any]:
        """Evaluate ensemble using cross-validation."""
        logger.info("Evaluating ensemble with cross-validation...")
        
        # Create season-based splits
        splits = create_season_based_splits(df, min_seasons=2)
        
        if not splits:
            logger.warning("No season-based splits created, skipping CV evaluation")
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
            
            # Train ensemble on this fold
            fold_ensemble = self._clone_ensemble()
            fold_ensemble.fit(X_train, y_train)
            
            # Make predictions
            y_pred = fold_ensemble.predict(X_val)
            y_pred_proba = fold_ensemble.predict_proba(X_val)
            
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
        
        cv_eval = {
            'method': 'season_based_cv',
            'n_splits': len(splits),
            'fold_metrics': fold_metrics,
            'overall_metrics': overall_metrics,
            'splits_info': [{'train_size': len(train), 'val_size': len(val)} for train, val in splits]
        }
        
        logger.info(f"Ensemble CV evaluation complete: {len(splits)} folds")
        logger.info(f"Overall accuracy: {overall_metrics['accuracy']:.3f}")
        
        return cv_eval
    
    def compare_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compare performance of individual models vs ensemble."""
        logger.info("Comparing model performances...")
        
        comparison_results = {}
        
        # Evaluate individual base models
        base_model_metrics = {}
        for name, model in self.base_models.items():
            try:
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)
                metrics = calculate_classification_metrics(y, y_pred, y_pred_proba)
                base_model_metrics[name] = metrics
                
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
                continue
        
        comparison_results['base_models'] = base_model_metrics
        
        # Evaluate ensemble if available
        if self.ensemble_model is not None:
            try:
                y_pred = self.ensemble_model.predict(X)
                y_pred_proba = self.ensemble_model.predict_proba(X)
                ensemble_metrics = calculate_classification_metrics(y, y_pred, y_pred_proba)
                comparison_results['ensemble'] = ensemble_metrics
                
            except Exception as e:
                logger.error(f"Failed to evaluate ensemble: {e}")
        
        # Create comparison summary
        comparison_summary = []
        for name, metrics in comparison_results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                comparison_summary.append({
                    'model': name,
                    'accuracy': metrics['accuracy'],
                    'brier_score': metrics['brier_score'],
                    'log_loss': metrics['log_loss']
                })
        
        # Sort by accuracy
        comparison_summary.sort(key=lambda x: x['accuracy'], reverse=True)
        
        comparison_results['summary'] = comparison_summary
        comparison_results['best_model'] = comparison_summary[0]['model'] if comparison_summary else None
        
        logger.info("Model comparison complete")
        if comparison_summary:
            logger.info(f"Best model: {comparison_summary[0]['model']} (Accuracy: {comparison_summary[0]['accuracy']:.3f})")
        
        return comparison_results
    
    def _clone_ensemble(self) -> Any:
        """Create a clone of the ensemble model for CV."""
        try:
            from sklearn.base import clone
            return clone(self.ensemble_model)
        except ImportError:
            import copy
            return copy.deepcopy(self.ensemble_model)
    
    def save_ensemble(self, output_path: Optional[str] = None) -> Path:
        """Save the trained ensemble model."""
        if self.ensemble_model is None:
            raise ValueError("No ensemble model to save")
        
        if output_path is None:
            output_path = self.models_dir / "ensemble_model.pkl"
        else:
            output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble
        with open(output_path, 'wb') as f:
            pickle.dump(self.ensemble_model, f)
        
        # Save ensemble info
        info_path = output_path.parent / "ensemble_info.pkl"
        ensemble_info = {
            'base_models': list(self.base_models.keys()),
            'ensemble_type': type(self.ensemble_model).__name__,
            'meta_learner_type': type(self.meta_learner).__name__ if self.meta_learner else None,
            'created_at': datetime.now().isoformat()
        }
        
        with open(info_path, 'wb') as f:
            pickle.dump(ensemble_info, f)
        
        logger.info(f"Ensemble saved to {output_path}")
        logger.info(f"Ensemble info saved to {info_path}")
        
        return output_path
    
    def generate_ensemble_report(self, comparison_results: Dict[str, Any], cv_results: Dict[str, Any]) -> str:
        """Generate a comprehensive ensemble report."""
        logger.info("Generating ensemble report...")
        
        report_lines = []
        report_lines.append("# Premier League Model Ensemble Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Model comparison summary
        report_lines.append("## Model Performance Comparison")
        report_lines.append("")
        report_lines.append("| Model | Accuracy | Brier Score | Log Loss |")
        report_lines.append("|-------|----------|-------------|----------|")
        
        for model_result in comparison_results.get('summary', []):
            report_lines.append(
                f"| {model_result['model']} | {model_result['accuracy']:.3f} | "
                f"{model_result['brier_score']:.4f} | {model_result['log_loss']:.4f} |"
            )
        report_lines.append("")
        
        # CV results
        if cv_results:
            report_lines.append("## Cross-Validation Results")
            report_lines.append("")
            report_lines.append(f"**Method:** {cv_results['method']}")
            report_lines.append(f"**Folds:** {cv_results['n_splits']}")
            report_lines.append(f"**Overall Accuracy:** {cv_results['overall_metrics']['accuracy']:.3f}")
            report_lines.append(f"**Overall Brier Score:** {cv_results['overall_metrics']['brier_score']:.4f}")
            report_lines.append("")
            
            # Detailed fold results
            report_lines.append("### Fold Details")
            report_lines.append("")
            report_lines.append("| Fold | Train Size | Val Size | Accuracy | Brier Score | Log Loss |")
            report_lines.append("|------|------------|----------|----------|-------------|----------|")
            
            for i, (fold_metric, split_info) in enumerate(zip(
                cv_results['fold_metrics'],
                cv_results['splits_info']
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
        
        if comparison_results.get('best_model'):
            best_model = comparison_results['best_model']
            report_lines.append(f"- 🏆 **Best performing model:** {best_model}")
            
            if best_model == 'ensemble':
                report_lines.append("- ✅ **Ensemble approach is effective** - combining multiple models improved performance")
            else:
                report_lines.append("- ⚠️ **Single model outperformed ensemble** - consider feature engineering or hyperparameter tuning")
        
        if cv_results and cv_results['overall_metrics']['accuracy'] > 0.6:
            report_lines.append("- ✅ **Good generalization** - model performs well across different time periods")
        else:
            report_lines.append("- ⚠️ **Limited generalization** - consider more robust feature engineering")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("*Report generated automatically by Premier League Model Ensemble*")
        
        return "\n".join(report_lines)
    
    def save_ensemble_results(self, comparison_results: Dict[str, Any], cv_results: Dict[str, Any]) -> None:
        """Save ensemble results to files."""
        logger.info("Saving ensemble results...")
        
        # Save detailed results as JSON
        results_path = self.outputs_dir / "reports" / "ensemble_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'comparison_results': comparison_results,
                'cv_results': cv_results
            }, f, indent=2, default=str)
        
        # Generate and save report
        report_content = self.generate_ensemble_report(comparison_results, cv_results)
        report_path = self.outputs_dir / "reports" / "ensemble_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Report saved to {report_path}")
    
    def run_ensemble_pipeline(self, ensemble_type: str = 'voting', meta_learner: str = 'logistic') -> Dict[str, Any]:
        """Run complete ensemble pipeline."""
        logger.info(f"Starting ensemble pipeline with {ensemble_type} ensemble...")
        
        # Load data
        df = self.load_match_dataset()
        
        # Create base models
        self.create_base_models()
        
        # Prepare features (use all available features for ensemble)
        feature_cols = [col for col in df.columns if col not in [
            'date', 'home_team', 'away_team', 'result', 'result_label', 
            'source_file', 'season', 'home_goals', 'away_goals'
        ]]
        
        X, y = self.prepare_features_and_target(df, feature_cols)
        
        # Train base models
        base_metrics = self.train_base_models(X, y)
        
        # Create and train ensemble
        if ensemble_type == 'voting':
            self.create_voting_ensemble(voting='soft')
        elif ensemble_type == 'stacking':
            self.create_stacking_ensemble(meta_learner)
        else:
            raise ValueError(f"Unsupported ensemble type: {ensemble_type}")
        
        # Train ensemble
        ensemble_metrics = self.train_ensemble(X, y)
        
        # Evaluate ensemble with CV
        cv_results = self.evaluate_ensemble_cv(X, y, df)
        
        # Compare models
        comparison_results = self.compare_models(X, y)
        
        # Save results
        self.save_ensemble_results(comparison_results, cv_results)
        
        # Save ensemble model
        ensemble_path = self.save_ensemble()
        
        logger.info("Ensemble pipeline complete!")
        
        return {
            'base_metrics': base_metrics,
            'ensemble_metrics': ensemble_metrics,
            'cv_results': cv_results,
            'comparison_results': comparison_results,
            'ensemble_path': str(ensemble_path)
        }


def main():
    """Main entry point for ensemble modeling."""
    parser = argparse.ArgumentParser(description="Create ensemble model for Premier League prediction")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--models-dir", default="models", help="Models directory path")
    parser.add_argument("--outputs-dir", default="outputs", help="Outputs directory path")
    parser.add_argument("--ensemble-type", choices=['voting', 'stacking'], default='voting', 
                       help="Type of ensemble to create")
    parser.add_argument("--meta-learner", default='logistic', help="Meta-learner for stacking (if applicable)")
    
    args = parser.parse_args()
    
    try:
        # Run ensemble pipeline
        ensemble = ModelEnsemble(args.data_dir, args.models_dir, args.outputs_dir)
        results = ensemble.run_ensemble_pipeline(args.ensemble_type, args.meta_learner)
        
        print(f"\n✅ Ensemble modeling complete!")
        print(f"📁 Ensemble Model: {results['ensemble_path']}")
        print(f"📁 Results: outputs/reports/ensemble_results.json")
        print(f"📁 Report: outputs/reports/ensemble_report.md")
        
        # Print summary
        if results['comparison_results'].get('best_model'):
            best_model = results['comparison_results']['best_model']
            print(f"\n🏆 Best Model: {best_model}")
            
            if best_model == 'ensemble':
                print("✅ Ensemble approach improved performance!")
            else:
                print("⚠️ Single model outperformed ensemble")
        
        if results['cv_results']:
            cv_metrics = results['cv_results']['overall_metrics']
            print(f"\n📊 Cross-validation Performance:")
            print(f"   Accuracy: {cv_metrics['accuracy']:.3f}")
            print(f"   Brier Score: {cv_metrics['brier_score']:.4f}")
            print(f"   Log Loss: {cv_metrics['log_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to create ensemble: {e}")
        raise


if __name__ == "__main__":
    main() 