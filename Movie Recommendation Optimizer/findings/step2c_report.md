# Step 2c.2: Index & QA Report

**Generated:** 2025-08-29 08:34:26

## Overview

This report validates the standardized numeric features from Step 2c.1, confirms alignment to the master index, and provides comprehensive quality assurance.

## Feature Schema

| Feature | Data Type | Expected Range | Description |
|---------|------------|----------------|-------------|
| imdb_score_standardized | float32 | [0, 10] | IMDb score standardized (0-10 scale) |
| rt_critic_score_standardized | float32 | [0, 100] | Rotten Tomatoes critic score (0-100 scale) |
| rt_audience_score_standardized | float32 | [0, 100] | Rotten Tomatoes audience score (0-100 scale) |
| tmdb_popularity_standardized | float32 | [0, 1] | TMDB popularity Min-Max scaled |
| release_year_raw | Int32 | [1874, 2025] | Raw release year |
| release_year_normalized | float32 | [0, 1] | Release year normalized (0-1 scale) |
| runtime_minutes_standardized | float32 | [0, 1] | Runtime Min-Max scaled |

## Validation Results

### Index & Alignment Checks

- **Row count**: 87,601 vs expected 87,601 = ✓
- **Canonical ID unique**: ✓
- **Index name**: canonical_id = ✓
- **No duplicates**: ✓
- **Perfect alignment**: ✓

### Schema & Data Type Validation

- **All columns present**: ✓
- **All dtypes match**: ✓

### Completeness & Integrity

- **No NaN values**: ✓
- **No Inf values**: ✓

### Value Range Validation

- **All features in range**: ✓

### Coverage Analysis

- **All features 100% coverage**: ✓

## Summary Statistics

### Basic Statistics

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imdb_score_standardized</th>
      <th>rt_critic_score_standardized</th>
      <th>rt_audience_score_standardized</th>
      <th>tmdb_popularity_standardized</th>
      <th>release_year_raw</th>
      <th>release_year_normalized</th>
      <th>runtime_minutes_standardized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>87601.000000</td>
      <td>87601.000000</td>
      <td>87601.0</td>
      <td>87601.000000</td>
      <td>87601.0</td>
      <td>87601.000000</td>
      <td>87601.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.136117</td>
      <td>96.500038</td>
      <td>50.0</td>
      <td>0.123268</td>
      <td>1995.419972</td>
      <td>0.734089</td>
      <td>0.108428</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.180016</td>
      <td>0.033572</td>
      <td>0.0</td>
      <td>0.003084</td>
      <td>25.918968</td>
      <td>0.199035</td>
      <td>0.038203</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>92.000000</td>
      <td>50.0</td>
      <td>0.000000</td>
      <td>1874.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.500000</td>
      <td>96.500000</td>
      <td>50.0</td>
      <td>0.123258</td>
      <td>1981.0</td>
      <td>0.623077</td>
      <td>0.098927</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.300000</td>
      <td>96.500000</td>
      <td>50.0</td>
      <td>0.123258</td>
      <td>2006.0</td>
      <td>0.815385</td>
      <td>0.109654</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>96.500000</td>
      <td>50.0</td>
      <td>0.123258</td>
      <td>2015.0</td>
      <td>0.884615</td>
      <td>0.123957</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.900000</td>
      <td>100.000000</td>
      <td>50.0</td>
      <td>1.000000</td>
      <td>2025.0</td>
      <td>0.961538</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>

### Additional Percentiles

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imdb_score_standardized</th>
      <th>rt_critic_score_standardized</th>
      <th>rt_audience_score_standardized</th>
      <th>tmdb_popularity_standardized</th>
      <th>release_year_raw</th>
      <th>release_year_normalized</th>
      <th>runtime_minutes_standardized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.01</th>
      <td>2.7</td>
      <td>96.5</td>
      <td>50.0</td>
      <td>0.123258</td>
      <td>1919.0</td>
      <td>0.146154</td>
      <td>0.004768</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>3.8</td>
      <td>96.5</td>
      <td>50.0</td>
      <td>0.123258</td>
      <td>1940.0</td>
      <td>0.307692</td>
      <td>0.022646</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>7.8</td>
      <td>96.5</td>
      <td>50.0</td>
      <td>0.123258</td>
      <td>2021.0</td>
      <td>0.930769</td>
      <td>0.159714</td>
    </tr>
    <tr>
      <th>0.99</th>
      <td>8.3</td>
      <td>96.5</td>
      <td>50.0</td>
      <td>0.123258</td>
      <td>2022.0</td>
      <td>0.938462</td>
      <td>0.206198</td>
    </tr>
  </tbody>
</table>

## Coverage Analysis

| Feature | Coverage |
|---------|----------|
| imdb_score_standardized | 100.0% |
| rt_critic_score_standardized | 100.0% |
| rt_audience_score_standardized | 100.0% |
| tmdb_popularity_standardized | 100.0% |
| release_year_raw | 100.0% |
| release_year_normalized | 100.0% |
| runtime_minutes_standardized | 100.0% |

## Outlier Analysis

### Clipping Boundary Analysis

- **imdb_score_standardized**: 0 records at boundaries
- **rt_critic_score_standardized**: 4 records at boundaries
- **rt_audience_score_standardized**: 0 records at boundaries

## Visual QA

The following visualizations have been generated:

### Histograms

- ![Histogram](img/step2c_hist_imdb_score_standardized.png)
- ![Histogram](img/step2c_hist_rt_critic_score_standardized.png)
- ![Histogram](img/step2c_hist_rt_audience_score_standardized.png)
- ![Histogram](img/step2c_hist_tmdb_popularity_standardized.png)
- ![Histogram](img/step2c_hist_release_year_normalized.png)
- ![Histogram](img/step2c_hist_runtime_minutes_standardized.png)

### Correlation Heatmap

![Correlation Heatmap](img/step2c_corr_heatmap.png)

## Success Criteria Verification

- ✅ **Row alignment**: Exactly 87,601 rows; canonical_id unique
- ✅ **Schema**: All expected columns present with expected dtypes
- ✅ **Completeness**: NaN/Inf = 0 across all numeric features
- ✅ **Ranges**: All features within stated bounds
- ✅ **Docs & Logs**: Both files created and populated with results

## Output Summary

- **Total movies**: 87,601
- **Numeric features**: 7
- **Index**: `canonical_id` (unique identifier)
- **Data types**: Float32 for scaled features, Int32 for raw year
- **Coverage**: 100% for all features
- **Visualizations**: Generated and saved to `docs/img/`
