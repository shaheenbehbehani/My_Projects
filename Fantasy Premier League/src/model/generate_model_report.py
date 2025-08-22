#!/usr/bin/env python3
"""
Generate comprehensive Premier League model report.

This script compiles all Phase 4 results into a final comprehensive report
including data lineage, model performance, calibration results, backtest metrics,
and final season predictions.

Inputs: All Phase 4 outputs and reports
Output: reports/model_report.md
"""

import polars as pl
import pandas as pd
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelReportGenerator:
    """Generate comprehensive Premier League model report."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models", outputs_dir: str = "outputs"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.outputs_dir = Path(outputs_dir)
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure reports directory exists
        (self.data_dir / "reports").mkdir(parents=True, exist_ok=True)
        
    def load_cv_metrics(self) -> Optional[Dict[str, Any]]:
        """Load cross-validation metrics."""
        cv_path = self.data_dir / "reports" / "cv_metrics.json"
        
        if not cv_path.exists():
            logger.warning("CV metrics not found")
            return None
        
        try:
            with open(cv_path, 'r') as f:
                cv_metrics = json.load(f)
            logger.info("Loaded CV metrics")
            return cv_metrics
        except Exception as e:
            logger.warning(f"Failed to load CV metrics: {e}")
            return None
    
    def load_calibration_summary(self) -> Optional[str]:
        """Load calibration summary."""
        cal_path = self.data_dir / "reports" / "calibration_summary.md"
        
        if not cal_path.exists():
            logger.warning("Calibration summary not found")
            return None
        
        try:
            with open(cal_path, 'r') as f:
                content = f.read()
            logger.info("Loaded calibration summary")
            return content
        except Exception as e:
            logger.warning(f"Failed to load calibration summary: {e}")
            return None
    
    def load_backtest_metrics(self) -> Optional[Dict[str, Any]]:
        """Load backtest metrics."""
        backtest_path = self.data_dir / "reports" / "backtest_metrics.csv"
        
        if not backtest_path.exists():
            logger.warning("Backtest metrics not found")
            return None
        
        try:
            backtest_df = pd.read_csv(backtest_path)
            logger.info("Loaded backtest metrics")
            return {'dataframe': backtest_df, 'path': backtest_path}
        except Exception as e:
            logger.warning(f"Failed to load backtest metrics: {e}")
            return None
    
    def load_simulation_summary(self) -> Optional[pd.DataFrame]:
        """Load simulation summary."""
        sim_path = self.outputs_dir / "sim_summary_2025_26.parquet"
        
        if not sim_path.exists():
            logger.warning("Simulation summary not found")
            return None
        
        try:
            sim_df = pd.read_parquet(sim_path)
            logger.info("Loaded simulation summary")
            return sim_df
        except Exception as e:
            logger.warning(f"Failed to load simulation summary: {e}")
            return None
    
    def load_match_dataset_info(self) -> Dict[str, Any]:
        """Load information about the match dataset."""
        dataset_path = self.processed_dir / "match_dataset.parquet"
        
        if not dataset_path.exists():
            return {'status': 'not_found'}
        
        try:
            df = pl.read_parquet(dataset_path)
            
            # Get basic info
            info = {
                'status': 'found',
                'rows': df.height,
                'columns': df.width,
                'date_range': {
                    'start': df['date'].min().strftime('%Y-%m-%d'),
                    'end': df['date'].max().strftime('%Y-%m-%d')
                },
                'seasons': sorted(df['season'].unique().to_list()),
                'teams': df['home_team'].n_unique(),
                'target_distribution': df['y'].value_counts().to_dict()
            }
            
            # Get feature info
            feature_cols = [col for col in df.columns if col not in 
                          ['date', 'season', 'home_team', 'away_team', 'y', 'result_label']]
            info['features'] = {
                'count': len(feature_cols),
                'names': feature_cols
            }
            
            logger.info("Loaded match dataset info")
            return info
            
        except Exception as e:
            logger.warning(f"Failed to load match dataset info: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def generate_data_lineage_section(self, dataset_info: Dict[str, Any]) -> str:
        """Generate data lineage section."""
        lines = []
        lines.append("## Data Lineage")
        lines.append("")
        
        if dataset_info['status'] == 'found':
            lines.append("### Training Dataset")
            lines.append(f"- **Source:** Historical Premier League matches (2014/15 - 2024/25)")
            lines.append(f"- **Size:** {dataset_info['rows']:,} matches, {dataset_info['columns']} columns")
            lines.append(f"- **Date Range:** {dataset_info['date_range']['start']} to {dataset_info['date_range']['end']}")
            lines.append(f"- **Seasons:** {', '.join(dataset_info['seasons'])}")
            lines.append(f"- **Teams:** {dataset_info['teams']} unique teams")
            lines.append(f"- **Features:** {dataset_info['features']['count']} engineered features")
            
            # Target distribution
            target_dist = dataset_info['target_distribution']
            lines.append(f"- **Target Distribution:**")
            lines.append(f"  - Home Win (H): {target_dist.get('H', 0):,} ({target_dist.get('H', 0)/dataset_info['rows']*100:.1f}%)")
            lines.append(f"  - Draw (D): {target_dist.get('D', 0):,} ({target_dist.get('D', 0)/dataset_info['rows']*100:.1f}%)")
            lines.append(f"  - Away Win (A): {target_dist.get('A', 0):,} ({target_dist.get('A', 0)/dataset_info['rows']*100:.1f}%)")
            
            lines.append("")
            lines.append("### Feature Engineering")
            lines.append("- **Elo Ratings:** Computed chronologically to avoid data leakage")
            lines.append("- **Rolling Form:** Last 5 matches points, last 10 matches goals")
            lines.append("- **Team Standardization:** Applied Phase 1 normalizer for consistent team names")
            lines.append("- **Temporal Features:** All features computed using only past information")
            
        else:
            lines.append("*Dataset information not available*")
        
        lines.append("")
        return "\n".join(lines)
    
    def generate_model_performance_section(self, cv_metrics: Optional[Dict[str, Any]]) -> str:
        """Generate model performance section."""
        lines = []
        lines.append("## Model Performance")
        lines.append("")
        
        if cv_metrics:
            lines.append("### Cross-Validation Results")
            lines.append(f"- **Method:** {cv_metrics.get('method', 'Unknown')}")
            lines.append(f"- **Folds:** {cv_metrics.get('n_folds', 'Unknown')}")
            lines.append(f"- **Seasons:** {', '.join(cv_metrics.get('seasons', []))}")
            
            if 'overall_metrics' in cv_metrics:
                metrics = cv_metrics['overall_metrics']
                lines.append("")
                lines.append("#### Overall Performance")
                lines.append(f"- **Accuracy:** {metrics.get('accuracy', 'N/A'):.3f}")
                lines.append(f"- **Brier Score:** {metrics.get('brier_score', 'N/A'):.4f}")
                lines.append(f"- **Log Loss:** {metrics.get('log_loss', 'N/A'):.4f}")
            
            if 'fold_metrics' in cv_metrics:
                lines.append("")
                lines.append("#### Fold-by-Fold Performance")
                lines.append("| Fold | Train Until | Validate | Accuracy | Brier Score | Log Loss |")
                lines.append("|------|-------------|----------|----------|-------------|----------|")
                
                for fold_metric in cv_metrics['fold_metrics']:
                    lines.append(
                        f"| {fold_metric.get('fold', 'N/A')} | {fold_metric.get('train_until', 'N/A')} | "
                        f"{fold_metric.get('valid_season', 'N/A')} | {fold_metric.get('accuracy', 'N/A'):.3f} | "
                        f"{fold_metric.get('brier_score', 'N/A'):.4f} | {fold_metric.get('log_loss', 'N/A'):.4f} |"
                    )
        else:
            lines.append("*Cross-validation metrics not available*")
        
        lines.append("")
        return "\n".join(lines)
    
    def generate_calibration_section(self, cal_summary: Optional[str]) -> str:
        """Generate calibration section."""
        lines = []
        lines.append("## Model Calibration")
        lines.append("")
        
        if cal_summary:
            # Extract key information from calibration summary
            lines.append("### Calibration Results")
            
            # Parse key metrics from the summary
            if "Brier Score Improvement:" in cal_summary:
                lines.append("- **Method:** Isotonic regression (one-vs-rest)")
                lines.append("- **Validation Data:** Last available season")
                
                # Extract improvements
                for line in cal_summary.split('\n'):
                    if "Brier Score Improvement:" in line:
                        lines.append(f"- **Brier Score Improvement:** {line.split(':')[1].strip()}")
                    elif "Log Loss Improvement:" in line:
                        lines.append(f"- **Log Loss Improvement:** {line.split(':')[1].strip()}")
                    elif "Accuracy Change:" in line:
                        lines.append(f"- **Accuracy Change:** {line.split(':')[1].strip()}")
                        break
            else:
                lines.append(cal_summary)
        else:
            lines.append("*Calibration results not available*")
        
        lines.append("")
        return "\n".join(lines)
    
    def generate_backtest_section(self, backtest_data: Optional[Dict[str, Any]]) -> str:
        """Generate backtest section."""
        lines = []
        lines.append("## Backtest Performance")
        lines.append("")
        
        if backtest_data:
            df = backtest_data['dataframe']
            
            lines.append("### Walk-Forward Testing (2014/15 - 2024/25)")
            lines.append(f"- **Method:** Train ≤ t-1, Test t")
            lines.append(f"- **Seasons Tested:** {len(df)}")
            
            # Calculate averages
            avg_accuracy = df['accuracy'].mean()
            avg_brier = df['brier_score'].mean()
            avg_logloss = df['log_loss'].mean()
            
            lines.append("")
            lines.append("#### Average Performance")
            lines.append(f"- **Accuracy:** {avg_accuracy:.3f}")
            lines.append(f"- **Brier Score:** {avg_brier:.4f}")
            lines.append(f"- **Log Loss:** {avg_logloss:.4f}")
            
            # Best and worst seasons
            best_season = df.loc[df['accuracy'].idxmax()]
            worst_season = df.loc[df['accuracy'].idxmin()]
            
            lines.append("")
            lines.append("#### Performance Range")
            lines.append(f"- **Best Season:** {best_season['season']} (Accuracy: {best_season['accuracy']:.3f})")
            lines.append(f"- **Worst Season:** {worst_season['season']} (Accuracy: {worst_season['accuracy']:.3f})")
            
        else:
            lines.append("*Backtest results not available*")
        
        lines.append("")
        return "\n".join(lines)
    
    def generate_season_predictions_section(self, sim_summary: Optional[pd.DataFrame]) -> str:
        """Generate season predictions section."""
        lines = []
        lines.append("## 2025/26 Season Predictions")
        lines.append("")
        
        if sim_summary is not None:
            lines.append("### Monte Carlo Simulation Results")
            lines.append(f"- **Simulations:** {sim_summary['simulations'].iloc[0]:,} seasons")
            lines.append(f"- **Teams:** {len(sim_summary)}")
            
            lines.append("")
            lines.append("#### Final League Table (Expected Points)")
            lines.append("| Rank | Team | Expected Points | Title % | Top 4 % | Most Common Position |")
            lines.append("|------|------|-----------------|---------|---------|----------------------|")
            
            for _, row in sim_summary.head(10).iterrows():
                lines.append(
                    f"| {row['rank']} | {row['team']} | {row['expected_points']:.1f} | "
                    f"{row['title_probability']:.1%} | {row['top4_probability']:.1%} | "
                    f"{row['most_common_position']} |"
                )
            
            lines.append("")
            lines.append("#### Title Contenders")
            title_contenders = sim_summary[sim_summary['title_probability'] > 0.05].head(5)
            for _, row in title_contenders.iterrows():
                lines.append(f"- **{row['team']}:** {row['title_probability']:.1%} chance")
            
            lines.append("")
            lines.append("#### Top 4 Qualification")
            top4_contenders = sim_summary[sim_summary['top4_probability'] > 0.3].head(8)
            for _, row in top4_contenders.iterrows():
                lines.append(f"- **{row['team']}:** {row['top4_probability']:.1%} chance")
            
        else:
            lines.append("*Season predictions not available*")
        
        lines.append("")
        return "\n".join(lines)
    
    def generate_technical_details_section(self) -> str:
        """Generate technical details section."""
        lines = []
        lines.append("## Technical Details")
        lines.append("")
        
        lines.append("### Model Architecture")
        lines.append("- **Base Model:** XGBoost Classifier (objective: multi:softprob)")
        lines.append("- **Fallback:** HistGradientBoostingClassifier (if XGBoost unavailable)")
        lines.append("- **Classes:** 3 (Home Win, Draw, Away Win)")
        lines.append("- **Calibration:** Isotonic regression (one-vs-rest)")
        
        lines.append("")
        lines.append("### Feature Engineering")
        lines.append("- **Elo Ratings:** K-factor 32, initial rating 1500")
        lines.append("- **Form Features:** Rolling windows (5 matches for points, 10 for goals)")
        lines.append("- **Data Leakage Prevention:** Strict temporal feature engineering")
        lines.append("- **Team Standardization:** Phase 1 normalizer for consistent naming")
        
        lines.append("")
        lines.append("### Training & Validation")
        lines.append("- **Cross-Validation:** Expanding window by season")
        lines.append("- **Class Imbalance:** Inverse-frequency sample weights")
        lines.append("- **Random Seed:** Fixed at 42 for reproducibility")
        lines.append("- **Data Split:** Chronological (no random shuffling)")
        
        lines.append("")
        lines.append("### Evaluation Metrics")
        lines.append("- **Accuracy:** Overall prediction correctness")
        lines.append("- **Brier Score:** Probability calibration quality (lower is better)")
        lines.append("- **Log Loss:** Probabilistic prediction quality (lower is better)")
        lines.append("- **Top-3 Hit Rate:** Champion prediction accuracy")
        
        lines.append("")
        return "\n".join(lines)
    
    def generate_usage_instructions_section(self) -> str:
        """Generate usage instructions section."""
        lines = []
        lines.append("## Usage Instructions")
        lines.append("")
        
        lines.append("### Running the Complete Pipeline")
        lines.append("```bash")
        lines.append("# Run complete Phase 4 pipeline")
        lines.append("make phase4")
        lines.append("")
        lines.append("# Or run individual components")
        lines.append("make build-match-dataset")
        lines.append("make train-model")
        lines.append("make calibrate-model")
        lines.append("make backtest-model")
        lines.append("make simulate-season")
        lines.append("make model-report")
        lines.append("```")
        
        lines.append("")
        lines.append("### Input Requirements")
        lines.append("- `data/raw/historical_matches.parquet` - Historical match data")
        lines.append("- `data/processed/fixtures_2025_26.parquet` - 2025/26 season fixtures")
        lines.append("- `src/normalization.py` - Team name standardization")
        
        lines.append("")
        lines.append("### Output Files")
        lines.append("- `data/processed/match_dataset.parquet` - Training dataset")
        lines.append("- `models/match_model.pkl` - Trained base model")
        lines.append("- `models/match_model_calibrated.pkl` - Calibrated model")
        lines.append("- `reports/cv_metrics.json` - Cross-validation results")
        lines.append("- `reports/calibration_summary.md` - Calibration report")
        lines.append("- `reports/backtest_metrics.md` - Backtest results")
        lines.append("- `outputs/sim_summary_2025_26.parquet` - Season predictions")
        lines.append("- `reports/model_report.md` - This comprehensive report")
        
        lines.append("")
        lines.append("### Model Deployment")
        lines.append("The calibrated model (`match_model_calibrated.pkl`) is ready for:")
        lines.append("- **Real-time predictions:** Match outcome probabilities")
        lines.append("- **Season simulations:** Monte Carlo season projections")
        lines.append("- **Risk assessment:** Team performance expectations")
        lines.append("- **Betting analysis:** Probability vs. bookmaker odds")
        
        lines.append("")
        return "\n".join(lines)
    
    def generate_comprehensive_report(self) -> str:
        """Generate the complete comprehensive report."""
        logger.info("Generating comprehensive model report...")
        
        # Load all data sources
        cv_metrics = self.load_cv_metrics()
        cal_summary = self.load_calibration_summary()
        backtest_data = self.load_backtest_metrics()
        sim_summary = self.load_simulation_summary()
        dataset_info = self.load_match_dataset_info()
        
        # Generate report sections
        report_lines = []
        report_lines.append("# Premier League Winner Prediction - Comprehensive Model Report")
        report_lines.append("")
        report_lines.append("## Executive Summary")
        report_lines.append("")
        report_lines.append("This report presents the complete Phase 4 implementation of the Premier League")
        report_lines.append("winner prediction system. The model has been trained on historical data from")
        report_lines.append("2014/15 to 2024/25, calibrated for probability accuracy, and validated through")
        report_lines.append("comprehensive backtesting and season simulation.")
        report_lines.append("")
        
        # Add all sections
        report_lines.append(self.generate_data_lineage_section(dataset_info))
        report_lines.append(self.generate_model_performance_section(cv_metrics))
        report_lines.append(self.generate_calibration_section(cal_summary))
        report_lines.append(self.generate_backtest_section(backtest_data))
        report_lines.append(self.generate_season_predictions_section(sim_summary))
        report_lines.append(self.generate_technical_details_section())
        report_lines.append(self.generate_usage_instructions_section())
        
        # Add footer
        report_lines.append("---")
        report_lines.append("*Report generated automatically by Premier League Model Report Generator*")
        report_lines.append(f"*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(report_lines)
    
    def save_report(self, report_content: str) -> Path:
        """Save the comprehensive report to file."""
        report_path = self.data_dir / "reports" / "model_report.md"
        
        logger.info(f"Saving comprehensive report to {report_path}")
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return report_path
    
    def run_report_generation(self) -> str:
        """Run the complete report generation pipeline."""
        logger.info("🚀 Starting comprehensive report generation...")
        
        # Generate report
        report_content = self.generate_comprehensive_report()
        
        # Save report
        report_path = self.save_report(report_content)
        
        logger.info("✅ Comprehensive report generation completed successfully!")
        return report_content


def main():
    """Main entry point for report generation."""
    parser = argparse.ArgumentParser(description="Generate comprehensive Premier League model report")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--models-dir", default="models", help="Models directory path")
    parser.add_argument("--outputs-dir", default="outputs", help="Outputs directory path")
    
    args = parser.parse_args()
    
    try:
        # Generate report
        generator = ModelReportGenerator(args.data_dir, args.models_dir, args.outputs_dir)
        report_content = generator.run_report_generation()
        
        print(f"\n✅ Comprehensive model report generated!")
        print(f"📄 Report: {args.data_dir}/reports/model_report.md")
        
        # Print summary
        print(f"\n📊 Report Summary:")
        print(f"   Data lineage and feature engineering details")
        print(f"   Model performance and cross-validation results")
        print(f"   Calibration improvements and backtest metrics")
        print(f"   2025/26 season predictions and probabilities")
        print(f"   Technical details and usage instructions")
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise


if __name__ == "__main__":
    main() 