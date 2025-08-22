#!/usr/bin/env python3
"""
Quick EDA Views Generator

Generates key visualizations from Premier League data and creates a summary report.
Focuses on rapid insights without requiring Jupyter notebooks.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Import our normalization module
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.normalization import canonicalize_frame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuickViewsGenerator:
    """Generates quick EDA visualizations and reports."""
    
    def __init__(self, data_raw: Path, figures_dir: Path, reports_dir: Path):
        """Initialize the quick views generator."""
        self.data_raw = Path(data_raw)
        self.figures_dir = Path(figures_dir)
        self.reports_dir = Path(reports_dir)
        
        # Ensure directories exist
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        
        # Track generated figures and insights
        self.generated_figures = []
        self.insights = []
        
    def load_historical_matches(self) -> Optional[pl.DataFrame]:
        """Load a sample of historical match data."""
        logger.info("Loading historical match data...")
        
        historical_files = list(self.data_raw.glob("E0*.csv"))
        if not historical_files:
            logger.warning("No historical match files (E0*.csv) found")
            return None
        
        # Use the first file for quick analysis
        match_file = historical_files[0]
        logger.info(f"Using match file: {match_file.name}")
        
        try:
            # Load with sampling for large files
            df = pl.scan_csv(match_file, ignore_errors=True).limit(5000).collect()
            logger.info(f"Loaded {df.height:,} rows, {df.width} columns")
            
            # Apply team canonicalization if team columns exist
            team_cols = []
            for col in ['HomeTeam', 'AwayTeam', 'Home', 'Away']:
                if col in df.columns:
                    team_cols.append(col)
            
            if team_cols:
                df = canonicalize_frame(df, team_cols)
                logger.info(f"Canonicalized team columns: {team_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {match_file.name}: {e}")
            return None
    
    def load_club_financial_data(self) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """Load club value and wage data."""
        logger.info("Loading club financial data...")
        
        club_values = None
        club_wages = None
        
        # Load club values
        value_file = self.data_raw / "Club Value.csv"
        if value_file.exists():
            try:
                club_values = pl.read_csv(value_file, ignore_errors=True)
                
                # Canonicalize team names
                team_cols = []
                for col in ['Club', 'Team', 'club', 'team']:
                    if col in club_values.columns:
                        team_cols.append(col)
                
                if team_cols:
                    club_values = canonicalize_frame(club_values, team_cols)
                
                logger.info(f"Loaded club values: {club_values.height} rows")
                
            except Exception as e:
                logger.error(f"Failed to load club values: {e}")
        
        # Load club wages
        wages_file = self.data_raw / "Club wages.csv"
        if wages_file.exists():
            try:
                club_wages = pl.read_csv(wages_file, ignore_errors=True)
                
                # Canonicalize team names
                team_cols = []
                for col in ['Club', 'Squad', 'Team', 'club', 'squad', 'team']:
                    if col in club_wages.columns:
                        team_cols.append(col)
                
                if team_cols:
                    club_wages = canonicalize_frame(club_wages, team_cols)
                
                logger.info(f"Loaded club wages: {club_wages.height} rows")
                
            except Exception as e:
                logger.error(f"Failed to load club wages: {e}")
        
        return club_values, club_wages
    
    def load_possession_data(self) -> Optional[pl.DataFrame]:
        """Load possession/xG data if available."""
        logger.info("Looking for possession/xG data...")
        
        possession_file = self.data_raw / "Possession data 24-25.csv"
        if possession_file.exists():
            try:
                df = pl.read_csv(possession_file, ignore_errors=True)
                logger.info(f"Loaded possession data: {df.height} rows")
                return df
            except Exception as e:
                logger.error(f"Failed to load possession data: {e}")
        else:
            logger.info("Possession data file not found")
        
        return None
    
    def generate_outcomes_distribution(self, matches_df: pl.DataFrame) -> bool:
        """Generate outcomes distribution chart."""
        logger.info("Generating outcomes distribution chart...")
        
        try:
            # Look for Full Time Result column
            ftr_col = None
            for col in ['FTR', 'Result', 'Res']:
                if col in matches_df.columns:
                    ftr_col = col
                    break
            
            if not ftr_col:
                logger.warning("No Full Time Result column found")
                return False
            
            # Count outcomes
            outcomes = matches_df[ftr_col].value_counts().sort('count', descending=True)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Extract data for plotting
            labels = outcomes[ftr_col].to_list()
            counts = outcomes['count'].to_list()
            
            # Map labels to meaningful names
            label_map = {'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'}
            display_labels = [label_map.get(label, label) for label in labels]
            
            # Create bar chart
            bars = ax.bar(display_labels, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            
            # Customize plot
            ax.set_title('Premier League Match Outcomes Distribution', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Matches')
            ax.set_xlabel('Match Result')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{count:,}', ha='center', va='bottom')
            
            # Add percentage labels
            total = sum(counts)
            for i, (bar, count) in enumerate(zip(bars, counts)):
                pct = (count / total) * 100
                ax.text(bar.get_x() + bar.get_width()/2., count/2,
                       f'{pct:.1f}%', ha='center', va='center', 
                       color='white', fontweight='bold')
            
            plt.tight_layout()
            
            # Save figure
            output_path = self.figures_dir / "outcomes_distribution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.generated_figures.append(output_path)
            
            # Record insights
            home_wins = counts[labels.index('H')] if 'H' in labels else 0
            away_wins = counts[labels.index('A')] if 'A' in labels else 0
            draws = counts[labels.index('D')] if 'D' in labels else 0
            
            home_pct = (home_wins / total) * 100
            away_pct = (away_wins / total) * 100
            draw_pct = (draws / total) * 100
            
            insight = f"**Home Advantage**: {home_pct:.1f}% home wins vs {away_pct:.1f}% away wins ({draw_pct:.1f}% draws)"
            self.insights.append(("Match Outcomes", insight))
            
            logger.info(f"✅ Outcomes distribution saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate outcomes distribution: {e}")
            return False
    
    def generate_odds_calibration(self, matches_df: pl.DataFrame) -> bool:
        """Generate odds calibration plot."""
        logger.info("Generating odds calibration chart...")
        
        try:
            # Look for Bet365 home win odds
            odds_col = None
            for col in ['B365H', 'BbAvH', 'BWH']:
                if col in matches_df.columns:
                    odds_col = col
                    break
            
            ftr_col = None
            for col in ['FTR', 'Result', 'Res']:
                if col in matches_df.columns:
                    ftr_col = col
                    break
            
            if not odds_col or not ftr_col:
                logger.warning("Required columns for odds calibration not found")
                return False
            
            # Filter valid data
            valid_data = matches_df.filter(
                (pl.col(odds_col).is_not_null()) & 
                (pl.col(odds_col) > 1.0) &
                (pl.col(ftr_col).is_not_null())
            )
            
            if valid_data.height < 50:
                logger.warning("Insufficient data for odds calibration")
                return False
            
            # Convert odds to implied probabilities
            implied_probs = (1.0 / valid_data[odds_col]).to_numpy()
            actual_outcomes = (valid_data[ftr_col] == 'H').to_numpy().astype(int)
            
            # Create probability bins
            bins = np.linspace(0, 1, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            binned_implied = []
            binned_actual = []
            
            for i in range(len(bins)-1):
                mask = (implied_probs >= bins[i]) & (implied_probs < bins[i+1])
                if np.sum(mask) > 0:
                    binned_implied.append(np.mean(implied_probs[mask]))
                    binned_actual.append(np.mean(actual_outcomes[mask]))
            
            if not binned_implied:
                logger.warning("No valid bins for calibration plot")
                return False
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot calibration curve
            ax.scatter(binned_implied, binned_actual, s=100, alpha=0.7, color='blue', label='Observed')
            ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
            
            # Customize plot
            ax.set_title('Betting Odds Calibration - Home Win Probability', fontsize=14, fontweight='bold')
            ax.set_xlabel('Implied Probability (from odds)')
            ax.set_ylabel('Empirical Probability (actual outcomes)')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add calibration statistics
            if len(binned_implied) > 1:
                mse = np.mean([(imp - act)**2 for imp, act in zip(binned_implied, binned_actual)])
                ax.text(0.05, 0.95, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save figure
            output_path = self.figures_dir / "odds_calibration.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.generated_figures.append(output_path)
            
            # Record insights
            if len(binned_implied) > 1:
                mse = np.mean([(imp - act)**2 for imp, act in zip(binned_implied, binned_actual)])
                insight = f"**Market Efficiency**: MSE of {mse:.4f} between implied and empirical probabilities"
            else:
                insight = "**Market Efficiency**: Limited data for full calibration assessment"
            
            self.insights.append(("Odds Calibration", insight))
            
            logger.info(f"✅ Odds calibration saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate odds calibration: {e}")
            return False
    
    def generate_value_vs_wage_scatter(self, club_values: pl.DataFrame, club_wages: pl.DataFrame) -> bool:
        """Generate squad value vs wage bill scatter plot."""
        logger.info("Generating value vs wage scatter plot...")
        
        try:
            # Find team columns
            value_team_col = None
            for col in ['Club', 'Team', 'club', 'team']:
                if col in club_values.columns:
                    value_team_col = col
                    break
            
            wage_team_col = None
            for col in ['Club', 'Squad', 'Team', 'club', 'squad', 'team']:
                if col in club_wages.columns:
                    wage_team_col = col
                    break
            
            if not value_team_col or not wage_team_col:
                logger.warning("Team columns not found in financial data")
                return False
            
            # Find value and wage columns
            value_col = None
            for col in club_values.columns:
                if any(term in col.lower() for term in ['value', 'worth']):
                    value_col = col
                    break
            
            wage_col = None
            for col in club_wages.columns:
                if any(term in col.lower() for term in ['wage', 'salary', 'annual']):
                    wage_col = col
                    break
            
            if not value_col or not wage_col:
                logger.warning("Value or wage columns not found")
                return False
            
            # Join datasets on team names
            merged = club_values.join(
                club_wages, 
                left_on=value_team_col, 
                right_on=wage_team_col, 
                how='inner'
            )
            
            if merged.height == 0:
                logger.warning("No matching teams found between value and wage data")
                return False
            
            # Extract numeric values (remove currency symbols and convert)
            def extract_numeric(series):
                return series.map_elements(lambda x: self._extract_currency_value(x), return_dtype=pl.Float64)
            
            values = extract_numeric(merged[value_col]).to_numpy()
            wages = extract_numeric(merged[wage_col]).to_numpy()
            teams = merged[value_team_col].to_list()
            
            # Filter valid data
            valid_mask = ~(np.isnan(values) | np.isnan(wages))
            values = values[valid_mask]
            wages = wages[valid_mask]
            teams = [team for i, team in enumerate(teams) if valid_mask[i]]
            
            if len(values) == 0:
                logger.warning("No valid financial data for plotting")
                return False
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Scatter plot
            scatter = ax.scatter(wages, values, s=100, alpha=0.7, color='blue')
            
            # Annotate top 5 by value
            top_indices = np.argsort(values)[-5:]
            for idx in top_indices:
                ax.annotate(teams[idx], (wages[idx], values[idx]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, alpha=0.8)
            
            # Customize plot
            ax.set_title('Club Squad Value vs Annual Wage Bill', fontsize=14, fontweight='bold')
            ax.set_xlabel('Annual Wage Bill (€M)')
            ax.set_ylabel('Squad Value (€M)')
            ax.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            if len(values) > 2:
                correlation = np.corrcoef(wages, values)[0, 1]
                ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save figure
            output_path = self.figures_dir / "value_vs_wage.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.generated_figures.append(output_path)
            
            # Record insights
            if len(values) > 2:
                correlation = np.corrcoef(wages, values)[0, 1]
                insight = f"**Financial Correlation**: {correlation:.3f} correlation between squad value and wage bill"
            else:
                insight = "**Financial Correlation**: Limited data for correlation analysis"
            
            self.insights.append(("Financial Analytics", insight))
            
            logger.info(f"✅ Value vs wage scatter saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate value vs wage scatter: {e}")
            return False
    
    def _extract_currency_value(self, value_str) -> float:
        """Extract numeric value from currency string."""
        if not isinstance(value_str, str):
            return float('nan')
        
        # Remove currency symbols and whitespace
        cleaned = re.sub(r'[£€$¥¢₹₽,\s]', '', value_str.strip())
        
        # Handle billion/million/thousand suffixes
        multiplier = 1
        if cleaned.lower().endswith('bn'):
            multiplier = 1000
            cleaned = cleaned[:-2]
        elif cleaned.lower().endswith('b'):
            multiplier = 1000
            cleaned = cleaned[:-1]
        elif cleaned.lower().endswith('m'):
            multiplier = 1
            cleaned = cleaned[:-1]
        elif cleaned.lower().endswith('k'):
            multiplier = 0.001
            cleaned = cleaned[:-1]
        
        try:
            return float(cleaned) * multiplier
        except ValueError:
            return float('nan')
    
    def generate_summary_report(self) -> Path:
        """Generate markdown summary report of quick views."""
        logger.info("Generating summary report...")
        
        report_content = f"""# EDA Quick Views Summary
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Overview
This report summarizes the key visualizations generated for Phase 2 EDA analysis.

## Generated Figures

"""
        
        # Add figures
        for fig_path in self.generated_figures:
            fig_name = fig_path.stem.replace('_', ' ').title()
            report_content += f"### {fig_name}\n"
            report_content += f"![{fig_name}](../outputs/figures/{fig_path.name})\n\n"
        
        # Add insights
        if self.insights:
            report_content += "## Key Insights\n\n"
            for category, insight in self.insights:
                report_content += f"**{category}**: {insight}\n\n"
        
        # Add recommendations
        report_content += """## Recommendations for Further Analysis

### Priority Areas
1. **Match Outcomes**: Investigate home advantage trends over time
2. **Market Efficiency**: Analyze systematic biases in betting markets
3. **Financial Impact**: Correlate spending with league performance
4. **Squad Management**: Examine rotation strategies and injury patterns

### Next Steps
- Run full EDA notebook for detailed analysis
- Validate findings with statistical significance tests
- Identify features for predictive modeling
- Generate insights for tactical analysis

---

*This report was generated automatically by the Premier League Data Pipeline Phase 2.*
"""
        
        # Write report
        report_path = self.reports_dir / "eda_quick_views.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"✅ Summary report saved: {report_path}")
        return report_path
    
    def run_analysis(self) -> Dict[str, any]:
        """Run the complete quick views analysis."""
        logger.info("Starting quick views analysis...")
        
        results = {
            'figures_generated': 0,
            'insights_count': 0,
            'report_path': None,
            'errors': []
        }
        
        try:
            # Load data
            matches_df = self.load_historical_matches()
            club_values, club_wages = self.load_club_financial_data()
            possession_df = self.load_possession_data()
            
            # Generate visualizations
            if matches_df is not None:
                if self.generate_outcomes_distribution(matches_df):
                    results['figures_generated'] += 1
                
                if self.generate_odds_calibration(matches_df):
                    results['figures_generated'] += 1
            
            if club_values is not None and club_wages is not None:
                if self.generate_value_vs_wage_scatter(club_values, club_wages):
                    results['figures_generated'] += 1
            
            # Generate summary report
            results['report_path'] = self.generate_summary_report()
            results['insights_count'] = len(self.insights)
            
            logger.info(f"✅ Quick views analysis complete: {results['figures_generated']} figures, {results['insights_count']} insights")
            
        except Exception as e:
            error_msg = f"Analysis failed: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    data_raw = project_root / "data" / "raw"
    figures_dir = project_root / "outputs" / "figures"
    reports_dir = project_root / "reports"
    
    try:
        generator = QuickViewsGenerator(data_raw, figures_dir, reports_dir)
        results = generator.run_analysis()
        
        print(f"\n🎨 Quick Views Analysis Complete!")
        print(f"📊 Figures Generated: {results['figures_generated']}")
        print(f"💡 Insights: {results['insights_count']}")
        
        if results['report_path']:
            print(f"📄 Summary Report: {results['report_path']}")
        
        if results['errors']:
            print(f"⚠️ Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  • {error}")
        
        if results['figures_generated'] > 0:
            print(f"\n📁 View figures in: {figures_dir}")
        
        return 0 if not results['errors'] else 1
        
    except Exception as e:
        logger.error(f"Failed to run quick views analysis: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 