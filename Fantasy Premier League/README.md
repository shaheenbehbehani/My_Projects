# CSV/TSV Data Profiler

A Python script that analyzes CSV and TSV files using Polars and generates comprehensive data quality reports.

## Features

- **Fast Analysis**: Uses Polars lazy evaluation for efficient processing
- **Comprehensive Profiling**: Analyzes data types, null percentages, and distinct values
- **Data Quality Detection**: Identifies common issues like:
  - High null value percentages (>20%)
  - Currency symbols in text columns
  - Date-like patterns not parsed as dates
- **Large File Support**: Sample rows for huge datasets
- **Clean Reports**: Generates human-readable Markdown reports

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
# Analyze all CSV/TSV files in data/raw/
python src/profile_csvs.py
```

### Sample Large Files
```bash
# Process only first 250,000 rows of each file
python src/profile_csvs.py --sample 250000
```

## Directory Structure

```
Premier League/
├── src/
│   └── profile_csvs.py     # Main profiling script
├── data/
│   └── raw/                # Place CSV/TSV files here
├── reports/
│   └── data_profile.md     # Generated report (after running)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Output

The script generates a Markdown report at `reports/data_profile.md` containing:

- **File Overview**: Row counts, column counts for each file
- **Column Details**: Data types, null percentages, distinct value counts
- **Data Quality Issues**: Automated detection of common problems
- **Clean Formatting**: Tables and sections for easy reading

## Example Report Section

```markdown
## players.csv

**File Path**: `data/raw/players.csv`  
**Rows**: 15,432  
**Columns**: 8  

| Column Name | Data Type | % Null | # Distinct | Notes |
|-------------|-----------|--------|------------|-------|
| player_id   | Int64     | 0.0%   | 15432     | -     |
| name        | Utf8      | 0.2%   | 14891     | -     |
| salary      | Utf8      | 5.1%   | 8234      | Currency symbols |
| birth_date  | Utf8      | 1.3%   | 4521      | Date-like |
```

## Requirements

- Python 3.11+
- Polars >= 0.20.0

## Error Handling

The script is designed to be robust:
- Continues processing other files if one fails
- Logs errors and warnings
- Handles empty files gracefully
- Optimizes performance for wide datasets (>200 columns) 