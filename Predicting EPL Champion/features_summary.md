# Features Summary Report
*Generated: 2025-08-09 10:26:57*

## Overview

**Final Feature Store**: `data/processed/features_2025_26.parquet`  
**Teams**: 10  
**Features**: 21 (excluding team identifier)  
**Completeness**: 100.0%

## Feature Components

### Form Features ✅

- **Source Teams**: 24
- **Matched Teams**: 10
- **Match Rate**: 100.0%
- **Features Added**: 3

### Static Club Features ✅

- **Source Teams**: 20
- **Matched Teams**: 10
- **Match Rate**: 100.0%
- **Features Added**: 4

### Player Aggregations ❌

- **Source Teams**: 0
- **Matched Teams**: 0
- **Match Rate**: 0.0%
- **Features Added**: 2

### Injury Burden ❌

- **Source Teams**: 0
- **Matched Teams**: 0
- **Match Rate**: 0.0%
- **Features Added**: 1

### Bookmaker Priors ✅

- **Source Teams**: 24
- **Matched Teams**: 10
- **Match Rate**: 100.0%
- **Features Added**: 1

## Feature Statistics

| Feature | Non-Null | Mean | Min | Max | Std |
|---------|----------|------|-----|-----|-----|
| elo_pre | 10/10 | 1578.88 | 1423.23 | 1764.62 | 136.21 |
| roll5_xg_diff_pre | 10/10 | 0.06 | -1.80 | 1.20 | 0.88 |
| roll5_ppg_pre | 10/10 | 1.40 | 0.20 | 2.60 | 0.81 |
| market_value_eur | 10/10 | 811890000.00 | 264200000.00 | 1330000000.00 | 392396351.35 |
| annual_wages_gbp | 10/10 | 30000000.00 | 30000000.00 | 30000000.00 | 0.00 |
| avg_attendance | 10/10 | 40000.00 | 40000.00 | 40000.00 | 0.00 |
| capacity | 10/10 | 52197.70 | 31750.00 | 74879.00 | 14919.74 |
| attendance_rate | 10/10 | 0.83 | 0.53 | 1.26 | 0.27 |
| wage_to_value_ratio | 10/10 | 0.06 | 0.03 | 0.13 | 0.04 |
| squad_goals_90 | 10/10 | 1.50 | 1.50 | 1.50 | 0.00 |
| squad_xg_90 | 10/10 | 1.40 | 1.40 | 1.40 | 0.00 |
| squad_xa_90 | 10/10 | 1.20 | 1.20 | 1.20 | 0.00 |
| avg_xg_per_shot | 10/10 | 0.10 | 0.10 | 0.10 | 0.00 |
| shots_per_game | 10/10 | 12.00 | 12.00 | 12.00 | 0.00 |
| injury_burden_per_1000 | 10/10 | 50.00 | 50.00 | 50.00 | 0.00 |
| prior_title_prob | 10/10 | 0.07 | 0.01 | 0.19 | 0.07 |
| stadium_utilization | 10/10 | 0.83 | 0.53 | 1.26 | 0.27 |
| goals_over_xg_90 | 10/10 | 0.10 | 0.10 | 0.10 | 0.00 |
| market_value_rank | 10/10 | 5.50 | 1.00 | 10.00 | 3.03 |
| elo_rank | 10/10 | 5.50 | 1.00 | 10.00 | 3.03 |

## Team Rankings

### Top 5 Teams by elo_pre

1. **Liverpool**: 1764.625
2. **Arsenal**: 1750.170
3. **Manchester City**: 1735.160
4. **Chelsea**: 1639.740
5. **Brighton & Hove Albion**: 1583.763

### Top 5 Teams by market_value_eur

1. **Manchester City**: 1330000000.000
2. **Chelsea**: 1220000000.000
3. **Liverpool**: 1140000000.000
4. **Arsenal**: 1120000000.000
5. **Manchester United**: 818000000.000

### Top 5 Teams by prior_title_prob

1. **Liverpool**: 0.190
2. **Arsenal**: 0.166
3. **Manchester City**: 0.149
4. **Chelsea**: 0.065
5. **Brighton & Hove Albion**: 0.036


---

## Validation Results
*Validated: 2025-08-09 10:26:57*

### Summary

**Overall Status**: ⚠️ WARNINGS  
**Total Checks**: 19  
**Passed**: 18 ✅  
**Warnings**: 1 ⚠️  
**Errors**: 0 ❌

### Detailed Results

#### WARNING ⚠️

**Probability Sum (prior_title_prob)**: Sum deviates from 1.0: 0.678020
  - Title probabilities should sum to 1.0

#### PASS ✅

**Row Count**: Correct number of teams: 10
  - Matches expected count from fixtures

**Team Uniqueness**: All teams are unique
  - 10 unique teams

**Probability Range (prior_title_prob)**: Valid range: [0.010975, 0.189962]

**Range Check (elo_pre)**: Values in expected range: [1423.2, 1764.6]

**Range Check (market_value_eur)**: Values in expected range: [264200000.0, 1330000000.0]

**Range Check (annual_wages_gbp)**: Values in expected range: [30000000.0, 30000000.0]

**Range Check (avg_attendance)**: Values in expected range: [40000.0, 40000.0]

**Range Check (capacity)**: Values in expected range: [31750.0, 74879.0]

**Range Check (squad_goals_90)**: Values in expected range: [1.5, 1.5]

**Range Check (squad_xg_90)**: Values in expected range: [1.4, 1.4]

**Range Check (avg_xg_per_shot)**: Values in expected range: [0.1, 0.1]

**Range Check (shots_per_game)**: Values in expected range: [12.0, 12.0]

**Range Check (injury_burden_per_1000)**: Values in expected range: [50.0, 50.0]

**Data Completeness**: Good completeness: 100.0%
  - 0 missing values out of 210 cells

**Wage-to-Value Ratio**: Calculation correct

**Stadium Utilization**: Calculation correct

**Ranking Range (market_value_rank)**: Correct range: [1, 10]

**Ranking Range (elo_rank)**: Correct range: [1, 10]

### Recommendations

**Warnings Detected** ⚠️
- Review warnings for potential data quality issues
- Consider if values are reasonable for domain
- Document any expected deviations

---

*Validation completed by Premier League Data Pipeline Feature Validator*
