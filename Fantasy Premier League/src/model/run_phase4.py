#!/usr/bin/env python3
"""
Phase 4: Complete Modeling Pipeline for Premier League Winner Prediction

This script orchestrates the complete Phase 4 modeling pipeline:
1. Model Training
2. Hyperparameter Optimization
3. Model Evaluation
4. Ensemble Modeling
5. Final Model Selection and Deployment

Usage:
    python src/model/run_phase4.py [--data-dir DATA_DIR] [--models-dir MODELS_DIR] [--outputs-dir OUTPUTS_DIR]
"""

import polars as pl
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our Phase 4 components
from .train_model import MatchModelTrainer
from .evaluate_model import ModelEvaluator
from .ensemble_model import ModelEnsemble
from .optimize_hyperparameters import HyperparameterOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase4Orchestrator:
    """Orchestrates the complete Phase 4 modeling pipeline."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models", outputs_dir: str = "outputs"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.outputs_dir = Path(outputs_dir)
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "reports").mkdir(exist_ok=True)
        (self.outputs_dir / "figures").mkdir(exist_ok=True)
        
        # Pipeline results
        self.pipeline_results = {}
        self.start_time = None
        
    def run_phase4_pipeline(self, 
                           skip_training: bool = False,
                           skip_optimization: bool = False,
                           skip_evaluation: bool = False,
                           skip_ensemble: bool = False,
                           use_bayesian: bool = True) -> Dict[str, Any]:
        """Run the complete Phase 4 pipeline."""
        self.start_time = datetime.now()
        logger.info("🚀 Starting Phase 4: Complete Modeling Pipeline")
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"📁 Models directory: {self.models_dir}")
        logger.info(f"📁 Outputs directory: {self.outputs_dir}")
        
        try:
            # Step 1: Model Training (if not skipped)
            if not skip_training:
                self._run_model_training()
            else:
                logger.info("⏭️ Skipping model training")
            
            # Step 2: Hyperparameter Optimization (if not skipped)
            if not skip_optimization:
                self._run_hyperparameter_optimization(use_bayesian)
            else:
                logger.info("⏭️ Skipping hyperparameter optimization")
            
            # Step 3: Model Evaluation (if not skipped)
            if not skip_evaluation:
                self._run_model_evaluation()
            else:
                logger.info("⏭️ Skipping model evaluation")
            
            # Step 4: Ensemble Modeling (if not skipped)
            if not skip_ensemble:
                self._run_ensemble_modeling()
            else:
                logger.info("⏭️ Skipping ensemble modeling")
            
            # Step 5: Final Pipeline Summary
            self._generate_final_summary()
            
            # Step 6: Save Pipeline Results
            self._save_pipeline_results()
            
            logger.info("🎉 Phase 4 pipeline completed successfully!")
            return self.pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Phase 4 pipeline failed: {e}")
            raise
    
    def _run_model_training(self):
        """Run the model training step."""
        logger.info("🔧 Step 1: Training Base Models")
        
        try:
            trainer = MatchModelTrainer(str(self.data_dir), str(self.models_dir))
            model, cv_metrics = trainer.train_model()
            
            self.pipeline_results['training'] = {
                'status': 'success',
                'cv_metrics': cv_metrics,
                'model_path': str(self.models_dir / "match_model.pkl"),
                'feature_info_path': str(self.models_dir / "feature_info.pkl")
            }
            
            logger.info("✅ Model training completed successfully")
            logger.info(f"📊 CV Performance - Accuracy: {cv_metrics['accuracy']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            self.pipeline_results['training'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _run_hyperparameter_optimization(self, use_bayesian: bool = True):
        """Run the hyperparameter optimization step."""
        logger.info("🔍 Step 2: Hyperparameter Optimization")
        
        try:
            optimizer = HyperparameterOptimizer(str(self.data_dir), str(self.models_dir), str(self.outputs_dir))
            results = optimizer.run_optimization_pipeline(use_bayesian)
            
            self.pipeline_results['optimization'] = {
                'status': 'success',
                'optimization_results': results['optimization_results'],
                'comparison_results': results['comparison_results'],
                'saved_paths': results['saved_paths']
            }
            
            logger.info("✅ Hyperparameter optimization completed successfully")
            
            # Log best model
            if results['comparison_results'].get('best_model'):
                best_model = results['comparison_results']['best_model']
                logger.info(f"🏆 Best optimized model: {best_model}")
            
        except Exception as e:
            logger.error(f"❌ Hyperparameter optimization failed: {e}")
            self.pipeline_results['optimization'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _run_model_evaluation(self):
        """Run the model evaluation step."""
        logger.info("📊 Step 3: Comprehensive Model Evaluation")
        
        try:
            evaluator = ModelEvaluator(str(self.data_dir), str(self.models_dir), str(self.outputs_dir))
            results = evaluator.run_comprehensive_evaluation()
            
            self.pipeline_results['evaluation'] = {
                'status': 'success',
                'evaluation_results': results
            }
            
            logger.info("✅ Model evaluation completed successfully")
            
            # Log key metrics
            if 'season_based_cv' in results and 'overall_metrics' in results['season_based_cv']:
                metrics = results['season_based_cv']['overall_metrics']
                logger.info(f"📈 Season-based CV - Accuracy: {metrics['accuracy']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            self.pipeline_results['evaluation'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _run_ensemble_modeling(self):
        """Run the ensemble modeling step."""
        logger.info("🤝 Step 4: Ensemble Modeling")
        
        try:
            # Try voting ensemble first
            ensemble = ModelEnsemble(str(self.data_dir), str(self.models_dir), str(self.outputs_dir))
            results = ensemble.run_ensemble_pipeline('voting')
            
            self.pipeline_results['ensemble'] = {
                'status': 'success',
                'ensemble_type': 'voting',
                'ensemble_results': results
            }
            
            logger.info("✅ Ensemble modeling completed successfully")
            
            # Log ensemble performance
            if results['comparison_results'].get('best_model'):
                best_model = results['comparison_results']['best_model']
                logger.info(f"🏆 Best ensemble model: {best_model}")
            
        except Exception as e:
            logger.error(f"❌ Ensemble modeling failed: {e}")
            self.pipeline_results['ensemble'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _generate_final_summary(self):
        """Generate final pipeline summary."""
        logger.info("📋 Step 5: Generating Final Summary")
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Calculate success rate
        total_steps = len(self.pipeline_results)
        successful_steps = sum(1 for step in self.pipeline_results.values() if step.get('status') == 'success')
        success_rate = successful_steps / total_steps if total_steps > 0 else 0
        
        # Create summary
        summary = {
            'pipeline_name': 'Phase 4: Complete Modeling Pipeline',
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'success_rate': success_rate,
            'step_summary': {}
        }
        
        # Add step summaries
        for step_name, step_result in self.pipeline_results.items():
            summary['step_summary'][step_name] = {
                'status': step_result.get('status', 'unknown'),
                'duration': step_result.get('duration', 0),
                'key_metrics': step_result.get('key_metrics', {})
            }
        
        self.pipeline_results['summary'] = summary
        
        logger.info(f"📊 Pipeline Summary:")
        logger.info(f"   Duration: {duration}")
        logger.info(f"   Success Rate: {success_rate:.1%} ({successful_steps}/{total_steps})")
    
    def _save_pipeline_results(self):
        """Save all pipeline results to files."""
        logger.info("💾 Step 6: Saving Pipeline Results")
        
        try:
            # Save detailed results as JSON
            results_path = self.outputs_dir / "reports" / "phase4_pipeline_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.pipeline_results, f, indent=2, default=str)
            
            # Generate and save pipeline report
            report_content = self._generate_pipeline_report()
            report_path = self.outputs_dir / "reports" / "phase4_pipeline_report.md"
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"✅ Pipeline results saved to {results_path}")
            logger.info(f"✅ Pipeline report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save pipeline results: {e}")
    
    def _generate_pipeline_report(self) -> str:
        """Generate a comprehensive pipeline report."""
        report_lines = []
        report_lines.append("# Phase 4: Premier League Modeling Pipeline Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Pipeline overview
        summary = self.pipeline_results.get('summary', {})
        report_lines.append("## Pipeline Overview")
        report_lines.append("")
        report_lines.append(f"**Pipeline Name:** {summary.get('pipeline_name', 'Unknown')}")
        report_lines.append(f"**Start Time:** {summary.get('start_time', 'Unknown')}")
        report_lines.append(f"**End Time:** {summary.get('end_time', 'Unknown')}")
        report_lines.append(f"**Duration:** {summary.get('duration_seconds', 0):.1f} seconds")
        report_lines.append(f"**Success Rate:** {summary.get('success_rate', 0):.1%}")
        report_lines.append("")
        
        # Step results
        report_lines.append("## Step Results")
        report_lines.append("")
        
        step_names = {
            'training': 'Model Training',
            'optimization': 'Hyperparameter Optimization',
            'evaluation': 'Model Evaluation',
            'ensemble': 'Ensemble Modeling'
        }
        
        for step_key, step_name in step_names.items():
            step_result = self.pipeline_results.get(step_key, {})
            status = step_result.get('status', 'unknown')
            status_emoji = '✅' if status == 'success' else '❌' if status == 'failed' else '⚠️'
            
            report_lines.append(f"### {step_name}")
            report_lines.append(f"**Status:** {status_emoji} {status}")
            
            if status == 'success':
                if step_key == 'training' and 'cv_metrics' in step_result:
                    metrics = step_result['cv_metrics']
                    report_lines.append(f"**CV Accuracy:** {metrics.get('accuracy', 'N/A'):.3f}")
                    report_lines.append(f"**CV Brier Score:** {metrics.get('brier_score', 'N/A'):.4f}")
                
                elif step_key == 'optimization' and 'comparison_results' in step_result:
                    best_model = step_result['comparison_results'].get('best_model', 'N/A')
                    report_lines.append(f"**Best Model:** {best_model}")
                
                elif step_key == 'evaluation' and 'evaluation_results' in step_result:
                    eval_results = step_result['evaluation_results']
                    if 'season_based_cv' in eval_results:
                        cv_metrics = eval_results['season_based_cv'].get('overall_metrics', {})
                        report_lines.append(f"**Season CV Accuracy:** {cv_metrics.get('accuracy', 'N/A'):.3f}")
                
                elif step_key == 'ensemble' and 'ensemble_results' in step_result:
                    ensemble_results = step_result['ensemble_results']
                    best_model = ensemble_results.get('comparison_results', {}).get('best_model', 'N/A')
                    report_lines.append(f"**Best Ensemble Model:** {best_model}")
            
            elif status == 'failed':
                error = step_result.get('error', 'Unknown error')
                report_lines.append(f"**Error:** {error}")
            
            report_lines.append("")
        
        # Key outputs
        report_lines.append("## Key Outputs")
        report_lines.append("")
        report_lines.append("### Models")
        report_lines.append("- `models/match_model.pkl` - Base trained model")
        report_lines.append("- `models/feature_info.pkl` - Feature information")
        report_lines.append("- `outputs/models/*_optimized.pkl` - Optimized models")
        report_lines.append("- `models/ensemble_model.pkl` - Ensemble model")
        report_lines.append("")
        
        report_lines.append("### Reports")
        report_lines.append("- `outputs/reports/evaluation_results.json` - Model evaluation results")
        report_lines.append("- `outputs/reports/optimization_results.json` - Hyperparameter optimization results")
        report_lines.append("- `outputs/reports/ensemble_results.json` - Ensemble modeling results")
        report_lines.append("- `outputs/reports/phase4_pipeline_results.json` - Complete pipeline results")
        report_lines.append("")
        
        report_lines.append("### Visualizations")
        report_lines.append("- `outputs/figures/feature_importance.png` - Feature importance plot")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        
        success_rate = summary.get('success_rate', 0)
        if success_rate >= 0.75:
            report_lines.append("✅ **Pipeline successful** - All major components completed successfully")
        elif success_rate >= 0.5:
            report_lines.append("⚠️ **Partial success** - Some components failed, review errors")
        else:
            report_lines.append("❌ **Pipeline failed** - Multiple components failed, investigate issues")
        
        # Model performance recommendations
        if 'evaluation' in self.pipeline_results and self.pipeline_results['evaluation']['status'] == 'success':
            eval_results = self.pipeline_results['evaluation']['evaluation_results']
            if 'season_based_cv' in eval_results:
                cv_metrics = eval_results['season_based_cv'].get('overall_metrics', {})
                accuracy = cv_metrics.get('accuracy', 0)
                if accuracy > 0.6:
                    report_lines.append("✅ **Good model performance** - Model shows predictive capability")
                else:
                    report_lines.append("⚠️ **Limited model performance** - Consider feature engineering improvements")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("*Report generated automatically by Phase 4 Pipeline Orchestrator*")
        
        return "\n".join(report_lines)


def main():
    """Main entry point for Phase 4 pipeline."""
    parser = argparse.ArgumentParser(description="Run Phase 4: Complete Modeling Pipeline")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--models-dir", default="models", help="Models directory path")
    parser.add_argument("--outputs-dir", default="outputs", help="Outputs directory path")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip hyperparameter optimization step")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip model evaluation step")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip ensemble modeling step")
    parser.add_argument("--no-bayesian", action="store_true", help="Use grid search instead of Bayesian optimization")
    
    args = parser.parse_args()
    
    try:
        # Create orchestrator
        orchestrator = Phase4Orchestrator(args.data_dir, args.models_dir, args.outputs_dir)
        
        # Run pipeline
        results = orchestrator.run_phase4_pipeline(
            skip_training=args.skip_training,
            skip_optimization=args.skip_optimization,
            skip_evaluation=args.skip_evaluation,
            skip_ensemble=args.skip_ensemble,
            use_bayesian=not args.no_bayesian
        )
        
        # Print summary
        summary = results.get('summary', {})
        print(f"\n🎉 Phase 4 Pipeline Complete!")
        print(f"⏱️ Duration: {summary.get('duration_seconds', 0):.1f} seconds")
        print(f"✅ Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"📁 Results: {args.outputs_dir}/reports/phase4_pipeline_results.json")
        print(f"📄 Report: {args.outputs_dir}/reports/phase4_pipeline_report.md")
        
        # Check for any failures
        failed_steps = [name for name, result in results.items() 
                       if name != 'summary' and result.get('status') == 'failed']
        
        if failed_steps:
            print(f"\n⚠️ Failed steps: {', '.join(failed_steps)}")
            print("Check the logs and reports for details")
        else:
            print(f"\n✅ All pipeline steps completed successfully!")
        
    except Exception as e:
        logger.error(f"Phase 4 pipeline failed: {e}")
        print(f"\n❌ Phase 4 pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 