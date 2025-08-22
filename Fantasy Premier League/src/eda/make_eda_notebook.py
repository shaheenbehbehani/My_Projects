#!/usr/bin/env python3
"""
EDA Notebook Template Generator

Creates a structured Jupyter notebook template for Phase 2 EDA analysis
with predefined sections and empty code cells for exploration.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NotebookGenerator:
    """Generates structured Jupyter notebook templates for EDA."""
    
    def __init__(self, output_path: Path):
        """Initialize the notebook generator."""
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def create_markdown_cell(self, content: str) -> Dict[str, Any]:
        """Create a markdown cell with given content."""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": content.split('\n')
        }
    
    def create_code_cell(self, content: str = "") -> Dict[str, Any]:
        """Create a code cell with optional content."""
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": content.split('\n') if content else [""]
        }
    
    def create_notebook_structure(self) -> Dict[str, Any]:
        """Create the complete notebook structure."""
        cells = []
        
        # 1. Overview & Objectives
        cells.append(self.create_markdown_cell("""# Premier League EDA - Phase 2 Analysis

## Overview & Objectives

This notebook contains exploratory data analysis for the Premier League dataset, focusing on:

- **Match Outcomes**: Distribution of wins, draws, losses across historical data
- **Betting Market Analysis**: Calibration of bookmaker odds vs actual outcomes  
- **Financial Analytics**: Relationship between squad values and wage bills
- **Injury Impact**: Time-series analysis of injury burden patterns
- **Data Quality Assessment**: Coverage and completeness validation

## Data Sources
- Historical match results (E0*.csv)
- Club financial data (Club Value.csv, Club wages.csv)
- Injury records (Injury list 2002-2016.csv)
- Possession/xG statistics (Possession data 24-25.csv)
- Manager tenure data (Premier League Managers.csv)

---"""))
        
        cells.append(self.create_code_cell("""# Import required libraries
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Define data paths
data_raw = Path("../data/raw")
data_processed = Path("../data/processed")
figures_dir = Path("../outputs/figures")
figures_dir.mkdir(exist_ok=True)

print("🏈 Premier League EDA - Setup Complete")
print(f"📁 Raw data directory: {data_raw}")
print(f"📁 Processed data directory: {data_processed}")
print(f"📁 Figures output: {figures_dir}")"""))
        
        # 2. Outcomes Distribution
        cells.append(self.create_markdown_cell("""---

## 2. Match Outcomes Distribution

Analyzing the distribution of match results (wins, draws, losses) across historical Premier League data.

**Questions to explore:**
- What is the overall distribution of home wins vs away wins vs draws?
- Has this distribution changed over time?
- Are there seasonal patterns in outcome frequencies?"""))
        
        cells.append(self.create_code_cell("""# Load historical match data
historical_files = list(data_raw.glob("E0*.csv"))
print(f"Found {len(historical_files)} historical match files")

# TODO: Load and combine historical match data
# TODO: Analyze FTR (Full Time Result) distribution
# TODO: Create visualization of outcomes distribution
# TODO: Examine trends over time"""))
        
        cells.append(self.create_code_cell())
        
        # 3. Odds Calibration
        cells.append(self.create_markdown_cell("""---

## 3. Betting Odds Calibration Analysis

Examining how well bookmaker odds predict actual match outcomes.

**Key Metrics:**
- Implied probability vs empirical probability
- Calibration curves for different bet types
- Market efficiency indicators

**Bookmaker columns to analyze:**
- B365H, B365D, B365A (Bet365 odds)
- Other available bookmakers for comparison"""))
        
        cells.append(self.create_code_cell("""# TODO: Extract bookmaker odds columns
# TODO: Convert odds to implied probabilities
# TODO: Calculate empirical probabilities by binned predictions
# TODO: Create calibration plots
# TODO: Assess market efficiency"""))
        
        cells.append(self.create_code_cell())
        
        # 4. Squad Value vs Wage Bill
        cells.append(self.create_markdown_cell("""---

## 4. Financial Analytics: Squad Value vs Wage Bill

Exploring the relationship between club valuations and wage expenditure.

**Research Questions:**
- Is there a correlation between squad value and wage bill?
- Which clubs are over/under-spending relative to their squad value?
- How does financial power translate to on-field performance?"""))
        
        cells.append(self.create_code_cell("""# Load financial data
if (data_raw / "Club Value.csv").exists():
    club_values = pl.read_csv(data_raw / "Club Value.csv")
    print("✅ Loaded club values data")
    print(club_values.head())
else:
    print("⚠️ Club Value.csv not found")

# TODO: Load Club wages.csv
# TODO: Merge datasets on team names (apply canonicalization)
# TODO: Create scatter plot with annotations
# TODO: Identify outliers and interesting patterns"""))
        
        cells.append(self.create_code_cell())
        
        # 5. Injury Burden Analysis
        cells.append(self.create_markdown_cell("""---

## 5. Injury Burden Time-Series Analysis

Analyzing injury patterns and their potential impact on team performance.

**Analysis Goals:**
- Identify seasonal injury patterns
- Examine injury burden by team/position
- Correlate injury levels with performance metrics"""))
        
        cells.append(self.create_code_cell("""# Load injury data if available
injury_file = data_raw / "Injury list 2002-2016.csv"
if injury_file.exists():
    injuries = pl.read_csv(injury_file)
    print("✅ Loaded injury data")
    print(f"Injury records: {injuries.height:,} rows")
    print("Columns:", injuries.columns)
else:
    print("⚠️ Injury list 2002-2016.csv not found")

# TODO: Parse injury dates and durations
# TODO: Aggregate by time periods (monthly/seasonal)
# TODO: Create time-series visualizations
# TODO: Analyze patterns by position/severity"""))
        
        cells.append(self.create_code_cell())
        
        # 6. Questions & Hypotheses
        cells.append(self.create_markdown_cell("""---

## 6. Research Questions & Hypotheses

### Key Questions to Investigate

**Match Outcomes & Patterns:**
- [ ] Is there a significant home advantage in the Premier League?
- [ ] How has the competitive balance changed over different seasons?
- [ ] Are there specific matchweeks with unusual outcome patterns?

**Financial Impact:**
- [ ] Do higher wage bills correlate with better league positions?
- [ ] Which clubs show the best value-for-money in terms of points per £ spent?
- [ ] Has financial fair play affected spending patterns?

**Betting Market Efficiency:**
- [ ] Are bookmaker odds well-calibrated across different probability ranges?
- [ ] Do certain types of matches show systematic mispricing?
- [ ] How quickly do odds adjust to team news and form?

**Performance Analytics:**
- [ ] How well do xG metrics predict actual goals scored?
- [ ] Which teams consistently over/under-perform their xG?
- [ ] What factors drive the largest xG vs actual goal differences?

**Injury & Squad Management:**
- [ ] Do injury-prone teams have different performance patterns?
- [ ] Are there optimal squad rotation strategies visible in the data?
- [ ] How do manager changes correlate with performance shifts?

### Working Hypotheses

1. **Home Advantage Hypothesis**: Home teams should win ~45-50% of matches
2. **Financial Power Hypothesis**: Wage bill correlates more strongly with performance than squad value
3. **Market Efficiency Hypothesis**: Betting odds should be well-calibrated for high-volume markets
4. **Squad Depth Hypothesis**: Teams with more balanced squads perform better over full seasons

---

## Next Steps

After completing this EDA:
1. Validate key hypotheses with statistical tests
2. Identify features for predictive modeling
3. Document data quality issues for Phase 3
4. Generate insights for tactical/strategic analysis

---

*Generated by Premier League Data Pipeline - Phase 2*"""))
        
        # Create the complete notebook structure
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.11.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        return notebook
    
    def generate_notebook(self) -> Path:
        """Generate the complete EDA notebook."""
        logger.info(f"Generating EDA notebook template: {self.output_path}")
        
        notebook = self.create_notebook_structure()
        
        # Write notebook to file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Notebook template created: {self.output_path}")
        return self.output_path


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "notebooks" / "phase2_eda_template.ipynb"
    
    try:
        generator = NotebookGenerator(output_path)
        created_path = generator.generate_notebook()
        
        print(f"\n🎉 EDA Notebook Template Created!")
        print(f"📓 Path: {created_path}")
        print(f"📁 Size: {created_path.stat().st_size:,} bytes")
        print(f"\n💡 Open with: jupyter notebook {created_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to generate notebook: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 