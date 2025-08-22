#!/usr/bin/env python3
"""
Hyperparameter optimization for Premier League winner prediction models.

This script implements:
- Grid search optimization
- Bayesian optimization with Optuna
- Cross-validation for robust evaluation
- Multiple model types (XGBoost, Random Forest, etc.)
- Automated parameter tuning and model selection
"""

import polars as pl
import numpy as np
import pandas as pd
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import argparse
from datetime import datetime
import time

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


class HyperparameterOptimizer:
    """Hyperparameter optimization for Premier League prediction models."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models", outputs_dir: str = "outputs"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.outputs_dir = Path(outputs_dir)
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure output directories exist
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "reports").mkdir(exist_ok=True)
        (self.outputs_dir / "models").mkdir(exist_ok=True)
        
        # Optimization results
        self.optimization_results = {}
        self.best_models = {}
        
    def load_match_dataset(self) -> pl.DataFrame:
        """Load the prepared match dataset."""
        dataset_path = self.processed_dir / "match_dataset.parquet"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Match dataset not found: {dataset_path}")
        
        logger.info(f"Loading match dataset from {dataset_path}")
        df = pl.read_parquet(dataset_path)
        logger.info(f"Loaded dataset: {df.height:,} rows, {df.width} columns")
        
        return df
    
    def prepare_features_and_target(self, df: pl.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for optimization."""
        logger.info("Preparing features and target for optimization...")
        
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
    
    def get_xgboost_param_grid(self) -> Dict[str, List]:
        """Get XGBoost parameter grid for optimization."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
    
    def get_random_forest_param_grid(self) -> Dict[str, List]:
        """Get Random Forest parameter grid for optimization."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
    
    def get_logistic_regression_param_grid(self) -> Dict[str, List]:
        """Get Logistic Regression parameter grid for optimization."""
        return {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000],
            'class_weight': [None, 'balanced']
        }
    
    def optimize_xgboost_grid(self, X: np.ndarray, y: np.ndarray, df: pl.DataFrame) -> Dict[str, Any]:
        """Optimize XGBoost using grid search."""
        logger.info("Optimizing XGBoost with grid search...")
        
        try:
            import xgboost as xgb
            from sklearn.model_selection import GridSearchCV
            
            # Create base model
            base_model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=-1
            )
            
            # Get parameter grid
            param_grid = self.get_xgboost_param_grid()
            
            # Create cross-validation splits
            splits = create_season_based_splits(df, min_seasons=2)
            
            if not splits:
                logger.warning("No season-based splits created, using default CV")
                cv = 3
            else:
                cv = splits
            
            # Grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='neg_log_loss',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit grid search
            start_time = time.time()
            grid_search.fit(X, y)
            end_time = time.time()
            
            # Get results
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_  # Convert back to positive log loss
            best_model = grid_search.best_estimator_
            
            # Evaluate best model
            y_pred = best_model.predict(X)
            y_pred_proba = best_model.predict_proba(X)
            metrics = calculate_classification_metrics(y, y_pred, y_pred_proba)
            
            optimization_result = {
                'method': 'grid_search',
                'model_type': 'xgboost',
                'best_params': best_params,
                'best_score': best_score,
                'best_model': best_model,
                'metrics': metrics,
                'cv_results': grid_search.cv_results_,
                'optimization_time': end_time - start_time,
                'n_combinations': len(grid_search.cv_results_['params'])
            }
            
            # Store best model
            self.best_models['xgboost'] = best_model
            
            logger.info(f"XGBoost optimization complete in {end_time - start_time:.1f}s")
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best log loss: {best_score:.4f}")
            logger.info(f"Best accuracy: {metrics['accuracy']:.3f}")
            
            return optimization_result
            
        except ImportError as e:
            logger.error(f"Failed to import XGBoost: {e}")
            raise
    
    def optimize_random_forest_grid(self, X: np.ndarray, y: np.ndarray, df: pl.DataFrame) -> Dict[str, Any]:
        """Optimize Random Forest using grid search."""
        logger.info("Optimizing Random Forest with grid search...")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import GridSearchCV
            
            # Create base model
            base_model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
            
            # Get parameter grid
            param_grid = self.get_random_forest_param_grid()
            
            # Create cross-validation splits
            splits = create_season_based_splits(df, min_seasons=2)
            
            if not splits:
                logger.warning("No season-based splits created, using default CV")
                cv = 3
            else:
                cv = splits
            
            # Grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='neg_log_loss',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit grid search
            start_time = time.time()
            grid_search.fit(X, y)
            end_time = time.time()
            
            # Get results
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_  # Convert back to positive log loss
            best_model = grid_search.best_estimator_
            
            # Evaluate best model
            y_pred = best_model.predict(X)
            y_pred_proba = best_model.predict_proba(X)
            metrics = calculate_classification_metrics(y, y_pred, y_pred_proba)
            
            optimization_result = {
                'method': 'grid_search',
                'model_type': 'random_forest',
                'best_params': best_params,
                'best_score': best_score,
                'best_model': best_model,
                'metrics': metrics,
                'cv_results': grid_search.cv_results_,
                'optimization_time': end_time - start_time,
                'n_combinations': len(grid_search.cv_results_['params'])
            }
            
            # Store best model
            self.best_models['random_forest'] = best_model
            
            logger.info(f"Random Forest optimization complete in {end_time - start_time:.1f}s")
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best log loss: {best_score:.4f}")
            logger.info(f"Best accuracy: {metrics['accuracy']:.3f}")
            
            return optimization_result
            
        except ImportError as e:
            logger.error(f"Failed to import Random Forest: {e}")
            raise
    
    def optimize_logistic_regression_grid(self, X: np.ndarray, y: np.ndarray, df: pl.DataFrame) -> Dict[str, Any]:
        """Optimize Logistic Regression using grid search."""
        logger.info("Optimizing Logistic Regression with grid search...")
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import GridSearchCV
            
            # Create base model
            base_model = LogisticRegression(
                random_state=42,
                multi_class='multinomial'
            )
            
            # Get parameter grid
            param_grid = self.get_logistic_regression_param_grid()
            
            # Create cross-validation splits
            splits = create_season_based_splits(df, min_seasons=2)
            
            if not splits:
                logger.warning("No season-based splits created, using default CV")
                cv = 3
            else:
                cv = splits
            
            # Grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='neg_log_loss',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit grid search
            start_time = time.time()
            grid_search.fit(X, y)
            end_time = time.time()
            
            # Get results
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_  # Convert back to positive log loss
            best_model = grid_search.best_estimator_
            
            # Evaluate best model
            y_pred = best_model.predict(X)
            y_pred_proba = best_model.predict_proba(X)
            metrics = calculate_classification_metrics(y, y_pred, y_pred_proba)
            
            optimization_result = {
                'method': 'grid_search',
                'model_type': 'logistic_regression',
                'best_params': best_params,
                'best_score': best_score,
                'best_model': best_model,
                'metrics': metrics,
                'cv_results': grid_search.cv_results_,
                'optimization_time': end_time - start_time,
                'n_combinations': len(grid_search.cv_results_['params'])
            }
            
            # Store best model
            self.best_models['logistic_regression'] = best_model
            
            logger.info(f"Logistic Regression optimization complete in {end_time - start_time:.1f}s")
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best log loss: {best_score:.4f}")
            logger.info(f"Best accuracy: {metrics['accuracy']:.3f}")
            
            return optimization_result
            
        except ImportError as e:
            logger.error(f"Failed to import Logistic Regression: {e}")
            raise
    
    def optimize_xgboost_bayesian(self, X: np.ndarray, y: np.ndarray, df: pl.DataFrame, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize XGBoost using Bayesian optimization with Optuna."""
        logger.info(f"Optimizing XGBoost with Bayesian optimization ({n_trials} trials)...")
        
        try:
            import optuna
            import xgboost as xgb
            
            # Create cross-validation splits
            splits = create_season_based_splits(df, min_seasons=2)
            
            if not splits:
                logger.warning("No season-based splits created, using default CV")
                cv = 3
            else:
                cv = splits
            
            def objective(trial):
                # Suggest hyperparameters
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                }
                
                # Create model
                model = xgb.XGBClassifier(
                    **params,
                    random_state=42,
                    eval_metric='mlogloss',
                    n_jobs=-1
                )
                
                # Cross-validation
                scores = []
                for train_idx, val_idx in cv:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred_proba = model.predict_proba(X_val)
                    
                    # Calculate log loss
                    from sklearn.metrics import log_loss
                    score = log_loss(y_val, y_pred_proba)
                    scores.append(score)
                
                return np.mean(scores)
            
            # Create study
            study = optuna.create_study(direction='minimize')
            
            # Optimize
            start_time = time.time()
            study.optimize(objective, n_trials=n_trials)
            end_time = time.time()
            
            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value
            
            # Train best model
            best_model = xgb.XGBClassifier(
                **best_params,
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=-1
            )
            best_model.fit(X, y)
            
            # Evaluate best model
            y_pred = best_model.predict(X)
            y_pred_proba = best_model.predict_proba(X)
            metrics = calculate_classification_metrics(y, y_pred, y_pred_proba)
            
            optimization_result = {
                'method': 'bayesian_optimization',
                'model_type': 'xgboost',
                'best_params': best_params,
                'best_score': best_score,
                'best_model': best_model,
                'metrics': metrics,
                'optimization_time': end_time - start_time,
                'n_trials': n_trials,
                'study': study
            }
            
            # Store best model
            self.best_models['xgboost_bayesian'] = best_model
            
            logger.info(f"XGBoost Bayesian optimization complete in {end_time - start_time:.1f}s")
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best log loss: {best_score:.4f}")
            logger.info(f"Best accuracy: {metrics['accuracy']:.3f}")
            
            return optimization_result
            
        except ImportError as e:
            logger.error(f"Failed to import Optuna: {e}")
            logger.info("Falling back to grid search...")
            return self.optimize_xgboost_grid(X, y, df)
    
    def compare_optimized_models(self) -> Dict[str, Any]:
        """Compare performance of all optimized models."""
        logger.info("Comparing optimized models...")
        
        comparison_results = {}
        
        for name, model in self.best_models.items():
            try:
                # Get the dataset for evaluation
                df = self.load_match_dataset()
                feature_cols = [col for col in df.columns if col not in [
                    'date', 'home_team', 'away_team', 'result', 'result_label', 
                    'source_file', 'season', 'home_goals', 'away_goals'
                ]]
                X, y = self.prepare_features_and_target(df, feature_cols)
                
                # Evaluate model
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)
                metrics = calculate_classification_metrics(y, y_pred, y_pred_proba)
                
                comparison_results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'optimization_result': self.optimization_results.get(name, {})
                }
                
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
                continue
        
        # Create comparison summary
        comparison_summary = []
        for name, result in comparison_results.items():
            if 'metrics' in result:
                comparison_summary.append({
                    'model': name,
                    'accuracy': result['metrics']['accuracy'],
                    'brier_score': result['metrics']['brier_score'],
                    'log_loss': result['metrics']['log_loss']
                })
        
        # Sort by accuracy
        comparison_summary.sort(key=lambda x: x['accuracy'], reverse=True)
        
        comparison_results['summary'] = comparison_summary
        comparison_results['best_model'] = comparison_summary[0]['model'] if comparison_summary else None
        
        logger.info("Model comparison complete")
        if comparison_summary:
            logger.info(f"Best model: {comparison_summary[0]['model']} (Accuracy: {comparison_summary[0]['accuracy']:.3f})")
        
        return comparison_results
    
    def save_optimized_models(self) -> Dict[str, str]:
        """Save all optimized models to disk."""
        logger.info("Saving optimized models...")
        
        saved_paths = {}
        
        for name, model in self.best_models.items():
            try:
                # Create model path
                model_path = self.outputs_dir / "models" / f"{name}_optimized.pkl"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save model
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                saved_paths[name] = str(model_path)
                logger.info(f"Saved {name} to {model_path}")
                
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")
                continue
        
        return saved_paths
    
    def generate_optimization_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate a comprehensive optimization report."""
        logger.info("Generating optimization report...")
        
        report_lines = []
        report_lines.append("# Premier League Hyperparameter Optimization Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Optimization summary
        report_lines.append("## Optimization Summary")
        report_lines.append("")
        
        for name, result in comparison_results.items():
            if name == 'summary' or name == 'best_model':
                continue
                
            if 'optimization_result' in result:
                opt_result = result['optimization_result']
                report_lines.append(f"**{name.replace('_', ' ').title()}:**")
                report_lines.append(f"- Method: {opt_result.get('method', 'Unknown')}")
                report_lines.append(f"- Optimization Time: {opt_result.get('optimization_time', 0):.1f}s")
                if 'n_combinations' in opt_result:
                    report_lines.append(f"- Parameter Combinations: {opt_result['n_combinations']}")
                elif 'n_trials' in opt_result:
                    report_lines.append(f"- Optimization Trials: {opt_result['n_trials']}")
                report_lines.append("")
        
        # Model performance comparison
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
        
        # Best parameters
        report_lines.append("## Best Parameters")
        report_lines.append("")
        
        for name, result in comparison_results.items():
            if name == 'summary' or name == 'best_model':
                continue
                
            if 'optimization_result' in result and 'best_params' in result['optimization_result']:
                best_params = result['optimization_result']['best_params']
                report_lines.append(f"**{name.replace('_', ' ').title()}:**")
                for param, value in best_params.items():
                    report_lines.append(f"- {param}: {value}")
                report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        
        if comparison_results.get('best_model'):
            best_model = comparison_results['best_model']
            report_lines.append(f"- 🏆 **Best performing model:** {best_model}")
            
            # Get best model details
            best_result = comparison_results.get(best_model, {})
            if 'optimization_result' in best_result:
                opt_result = best_result['optimization_result']
                if opt_result.get('method') == 'bayesian_optimization':
                    report_lines.append("- ✅ **Bayesian optimization effective** - found better parameters than grid search")
                else:
                    report_lines.append("- ⚠️ **Grid search used** - consider Bayesian optimization for better results")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("*Report generated automatically by Premier League Hyperparameter Optimizer*")
        
        return "\n".join(report_lines)
    
    def save_optimization_results(self, comparison_results: Dict[str, Any]) -> None:
        """Save optimization results to files."""
        logger.info("Saving optimization results...")
        
        # Save detailed results as JSON
        results_path = self.outputs_dir / "reports" / "optimization_results.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for name, result in comparison_results.items():
            if name == 'summary' or name == 'best_model':
                serializable_results[name] = result
            else:
                serializable_results[name] = {
                    'metrics': result.get('metrics', {}),
                    'optimization_result': {
                        k: v for k, v in result.get('optimization_result', {}).items()
                        if k != 'best_model' and k != 'study'  # Skip non-serializable objects
                    }
                }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Generate and save report
        report_content = self.generate_optimization_report(comparison_results)
        report_path = self.outputs_dir / "reports" / "hyperparameter_optimization_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Report saved to {report_path}")
    
    def run_optimization_pipeline(self, use_bayesian: bool = True) -> Dict[str, Any]:
        """Run complete hyperparameter optimization pipeline."""
        logger.info("Starting hyperparameter optimization pipeline...")
        
        # Load data
        df = self.load_match_dataset()
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in [
            'date', 'home_team', 'away_team', 'result', 'result_label', 
            'source_file', 'season', 'home_goals', 'away_goals'
        ]]
        
        X, y = self.prepare_features_and_target(df, feature_cols)
        
        # Run optimizations
        self.optimization_results = {}
        
        # XGBoost optimization
        try:
            if use_bayesian:
                self.optimization_results['xgboost_bayesian'] = self.optimize_xgboost_bayesian(X, y, df)
            else:
                self.optimization_results['xgboost'] = self.optimize_xgboost_grid(X, y, df)
        except Exception as e:
            logger.error(f"XGBoost optimization failed: {e}")
        
        # Random Forest optimization
        try:
            self.optimization_results['random_forest'] = self.optimize_random_forest_grid(X, y, df)
        except Exception as e:
            logger.error(f"Random Forest optimization failed: {e}")
        
        # Logistic Regression optimization
        try:
            self.optimization_results['logistic_regression'] = self.optimize_logistic_regression_grid(X, y, df)
        except Exception as e:
            logger.error(f"Logistic Regression optimization failed: {e}")
        
        # Compare models
        comparison_results = self.compare_optimized_models()
        
        # Save results
        self.save_optimization_results(comparison_results)
        
        # Save models
        saved_paths = self.save_optimized_models()
        
        logger.info("Hyperparameter optimization pipeline complete!")
        
        return {
            'optimization_results': self.optimization_results,
            'comparison_results': comparison_results,
            'saved_paths': saved_paths
        }


def main():
    """Main entry point for hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for Premier League prediction models")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--models-dir", default="models", help="Models directory path")
    parser.add_argument("--outputs-dir", default="outputs", help="Outputs directory path")
    parser.add_argument("--use-bayesian", action="store_true", help="Use Bayesian optimization for XGBoost")
    
    args = parser.parse_args()
    
    try:
        # Run optimization pipeline
        optimizer = HyperparameterOptimizer(args.data_dir, args.models_dir, args.outputs_dir)
        results = optimizer.run_optimization_pipeline(args.use_bayesian)
        
        print(f"\n✅ Hyperparameter optimization complete!")
        print(f"📁 Results: outputs/reports/optimization_results.json")
        print(f"📁 Report: outputs/reports/hyperparameter_optimization_report.md")
        print(f"📁 Models: outputs/models/")
        
        # Print summary
        if results['comparison_results'].get('best_model'):
            best_model = results['comparison_results']['best_model']
            print(f"\n🏆 Best Model: {best_model}")
            
            best_result = results['comparison_results'].get(best_model, {})
            if 'metrics' in best_result:
                metrics = best_result['metrics']
                print(f"📊 Performance:")
                print(f"   Accuracy: {metrics['accuracy']:.3f}")
                print(f"   Brier Score: {metrics['brier_score']:.4f}")
                print(f"   Log Loss: {metrics['log_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to optimize hyperparameters: {e}")
        raise


if __name__ == "__main__":
    main() 