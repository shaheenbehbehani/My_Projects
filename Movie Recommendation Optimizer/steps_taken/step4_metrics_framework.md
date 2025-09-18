# Step 4.1.1: Metrics Framework Setup

## Overview

This document describes the comprehensive metrics framework implemented for evaluating movie recommendation systems in the Movie Recommendation Optimizer project. The framework supports evaluation of content-based, collaborative filtering, and hybrid recommendation systems.

## Framework Architecture

The metrics framework is implemented in `scripts/eval/metrics.py` and provides:

- **Ranking Metrics**: For evaluating recommendation quality
- **Prediction Metrics**: For evaluating rating prediction accuracy
- **Coverage Metrics**: For evaluating recommendation diversity and coverage
- **Evaluation Strategies**: Multiple approaches for splitting data

## Implemented Metrics

### 1. Ranking Metrics

#### Recall@K
- **Description**: Fraction of relevant items that are recommended
- **Formula**: |relevant ∩ recommended| / |relevant|
- **Range**: [0, 1] (higher is better)
- **Use Case**: Measures how well the system finds relevant items

#### Precision@K
- **Description**: Fraction of recommended items that are relevant
- **Formula**: |relevant ∩ recommended| / K
- **Range**: [0, 1] (higher is better)
- **Use Case**: Measures recommendation accuracy

#### MAP@K (Mean Average Precision@K)
- **Description**: Average precision across all relevant items
- **Formula**: Σ(precision@i × rel_i) / |relevant|
- **Range**: [0, 1] (higher is better)
- **Use Case**: Considers both precision and recall, penalizes late relevant items

#### NDCG@K (Normalized Discounted Cumulative Gain@K)
- **Description**: Ranking quality with position discount
- **Formula**: DCG@K / IDCG@K
- **Range**: [0, 1] (higher is better)
- **Use Case**: Considers ranking order and relevance scores

### 2. Prediction Metrics

#### RMSE (Root Mean Square Error)
- **Description**: Square root of mean squared differences
- **Formula**: √(Σ(rating_true - rating_pred)² / n)
- **Range**: [0, ∞) (lower is better)
- **Use Case**: Penalizes large prediction errors more heavily

#### MAE (Mean Absolute Error)
- **Description**: Mean of absolute differences
- **Formula**: Σ|rating_true - rating_pred| / n
- **Range**: [0, ∞) (lower is better)
- **Use Case**: Linear penalty for prediction errors

### 3. Coverage Metrics

#### User Coverage
- **Description**: Percentage of users with at least one recommendation
- **Formula**: |users_with_recommendations| / |all_users|
- **Range**: [0, 1] (higher is better)
- **Use Case**: Measures system accessibility

#### Item Coverage
- **Description**: Percentage of items that appear in at least one recommendation
- **Formula**: |recommended_items| / |all_items|
- **Range**: [0, 1] (higher is better)
- **Use Case**: Measures recommendation diversity

## K Values

The framework supports configurable K values for ranking metrics:
- **Default K values**: [5, 10, 20, 50]
- **Parameterized**: Easy to extend to other K values
- **Consistent**: Same K values used across all ranking metrics

## Evaluation Strategies

### 1. Holdout Split
- **Description**: Split ratings into train/test sets globally
- **Use Case**: General model evaluation
- **Implementation**: `evaluate_holdout_split()`
- **Data Requirements**: Separate train and test datasets

### 2. User-Sampled Split
- **Description**: For each user, hold out a fraction of their history for testing
- **Use Case**: Personalized recommendation evaluation
- **Implementation**: `evaluate_user_sampled_split()`
- **Parameters**: `sample_ratio` (default: 0.2)

## Ground Truth Definition

The framework uses MovieLens/IMDb ratings as ground truth:

- **Data Format**: Parquet files with columns ['user_index', 'canonical_id', 'rating']
- **Rating Scale**: 1.0 to 5.0 (MovieLens standard)
- **Relevance Threshold**: Ratings ≥ 4.0 considered relevant (configurable)
- **User Indexing**: Consistent user indexing across train/test splits

## Usage Examples

### Basic Usage

```python
from scripts.eval.metrics import MetricsFramework, MetricConfig

# Initialize framework
config = MetricConfig(k_values=[5, 10, 20])
framework = MetricsFramework(config)

# Evaluate ranking metrics
ground_truth = {'user1': ['movie_A', 'movie_B'], 'user2': ['movie_C']}
recommendations = {'user1': ['movie_A', 'movie_X', 'movie_B'], 'user2': ['movie_C', 'movie_Y']}
results = framework.evaluate_ranking_metrics(ground_truth, recommendations)
```

### Holdout Split Evaluation

```python
import pandas as pd

# Load data
train_data = pd.read_parquet('data/eval/checkpoints/ratings_split_train.parquet')
test_data = pd.read_parquet('data/eval/checkpoints/ratings_split_test.parquet')

# Generate recommendations (example)
recommendations = generate_recommendations(train_data)

# Evaluate
results = framework.evaluate_holdout_split(train_data, test_data, recommendations)
```

### User-Sampled Split Evaluation

```python
# Prepare user data
user_data = {
    'user1': [('movie_A', 4.0), ('movie_B', 5.0), ('movie_C', 3.0)],
    'user2': [('movie_D', 4.0), ('movie_E', 2.0)]
}

# Evaluate
results = framework.evaluate_user_sampled_split(user_data, recommendations, sample_ratio=0.3)
```

## Configuration

The framework configuration is stored in `data/eval/metrics_config.json`:

```json
{
  "framework_version": "1.0.0",
  "k_values": [5, 10, 20, 50],
  "ranking_metrics": ["recall", "precision", "map", "ndcg"],
  "prediction_metrics": ["rmse", "mae"],
  "coverage_metrics": ["user_coverage", "item_coverage"]
}
```

## Unit Tests

Comprehensive unit tests are provided in `scripts/eval/test_metrics.py`:

- **Synthetic Examples**: Manually verifiable test cases
- **Edge Cases**: Empty lists, perfect predictions, no relevant items
- **Error Handling**: Mismatched lengths, invalid inputs
- **Coverage**: All metrics and evaluation strategies tested

Run tests with:
```bash
python scripts/eval/test_metrics.py
```

## Integration with Recommendation Systems

The framework is designed to work with all three recommendation systems:

### Content-Based System
- **Input**: Item features, user preferences
- **Output**: Content-based recommendations
- **Evaluation**: Ranking metrics, coverage metrics

### Collaborative Filtering System
- **Input**: User-item rating matrix
- **Output**: CF recommendations + rating predictions
- **Evaluation**: All metrics (ranking, prediction, coverage)

### Hybrid System
- **Input**: Combined content and collaborative features
- **Output**: Hybrid recommendations + rating predictions
- **Evaluation**: All metrics with comparative analysis

## Performance Considerations

- **Efficient Implementation**: Vectorized operations using NumPy
- **Memory Management**: Handles large datasets with chunked processing
- **Caching**: Results can be cached for repeated evaluations
- **Parallelization**: Framework supports parallel evaluation across users

## Future Extensions

The framework is designed for easy extension:

1. **Additional Metrics**: Easy to add new ranking or prediction metrics
2. **Custom K Values**: Configurable K values for different use cases
3. **Evaluation Strategies**: New splitting strategies can be added
4. **Relevance Thresholds**: Configurable relevance definitions
5. **Statistical Testing**: Significance testing for metric comparisons

## Dependencies

- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Python 3.7+**: Core language requirements
- **Pathlib**: File path handling
- **JSON**: Configuration and results serialization

## File Structure

```
scripts/eval/
├── metrics.py              # Main metrics framework
├── test_metrics.py         # Unit tests
└── __init__.py

data/eval/
├── metrics_config.json     # Configuration manifest
└── checkpoints/            # Train/test data
    ├── ratings_split_train.parquet
    └── ratings_split_test.parquet

docs/
└── step4_metrics_framework.md  # This documentation
```

## Next Steps

This metrics framework is ready for use in:
- **Step 4.1.2**: Content-based system evaluation
- **Step 4.1.3**: Collaborative filtering system evaluation  
- **Step 4.1.4**: Hybrid system evaluation

The framework provides a solid foundation for comprehensive evaluation of all recommendation systems in the Movie Recommendation Optimizer project.



