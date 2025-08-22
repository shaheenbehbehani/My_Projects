#!/usr/bin/env python3
"""
Utility functions for Premier League modeling pipeline.

This module provides helper functions for:
- Time-based data splitting
- Evaluation metrics (Brier score, log loss)
- Model calibration wrappers
- Reproducible random seeds
- Softmax and additional metrics for Phase 4
"""

import numpy as np
import polars as pl
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, classification_report
import logging

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def create_time_based_splits(
    df: pl.DataFrame, 
    date_col: str = 'date',
    min_train_size: int = 100,
    test_size: int = 50,
    gap: int = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time-based train/validation splits for expanding window CV.
    
    Args:
        df: DataFrame with match data
        date_col: Name of the date column
        min_train_size: Minimum training set size
        test_size: Size of each validation fold
        gap: Gap between train and validation sets
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Sort by date
    df_sorted = df.sort(date_col)
    
    # Convert to pandas for sklearn compatibility
    df_pd = df_sorted.to_pandas()
    
    # Create time series split
    tscv = TimeSeriesSplit(
        n_splits=max(1, (len(df_pd) - min_train_size) // test_size),
        test_size=test_size,
        gap=gap
    )
    
    splits = []
    for train_idx, val_idx in tscv.split(df_pd):
        splits.append((train_idx, val_idx))
    
    logger.info(f"Created {len(splits)} time-based splits")
    logger.info(f"Training sizes: {[len(train) for train, _ in splits]}")
    logger.info(f"Validation sizes: {[len(val) for _, val in splits]}")
    
    return splits


def create_season_based_splits(
    df: pl.DataFrame,
    date_col: str = 'date',
    min_seasons: int = 3
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create season-based train/validation splits.
    
    Args:
        df: DataFrame with match data
        date_col: Name of the date column
        min_seasons: Minimum number of seasons in training set
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Extract season from date
    df_with_season = df.with_columns(
        pl.col(date_col).dt.year().alias('season')
    )
    
    # Get unique seasons
    seasons = sorted(df_with_season['season'].unique().to_list())
    logger.info(f"Found seasons: {seasons}")
    
    if len(seasons) < min_seasons + 1:
        raise ValueError(f"Need at least {min_seasons + 1} seasons, got {len(seasons)}")
    
    splits = []
    
    # Start with min_seasons in training, then expand
    for i in range(min_seasons, len(seasons)):
        train_seasons = seasons[:i]
        val_season = seasons[i]
        
        # Get indices for each split
        train_mask = df_with_season['season'].is_in(train_seasons)
        val_mask = df_with_season['season'] == val_season
        
        train_indices = np.where(train_mask.to_numpy())[0]
        val_indices = np.where(val_mask.to_numpy())[0]
        
        if len(train_indices) > 0 and len(val_indices) > 0:
            splits.append((train_indices, val_indices))
            logger.info(f"Split {i-min_seasons+1}: Train seasons {train_seasons}, Val season {val_season}")
    
    logger.info(f"Created {len(splits)} season-based splits")
    return splits


def create_rolling_window_splits(
    df: pl.DataFrame,
    date_col: str = 'date',
    window_size: int = 1000,
    step_size: int = 200,
    min_train_size: int = 500
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create rolling window train/validation splits for time series data.
    
    Args:
        df: DataFrame with match data
        date_col: Name of the date column
        window_size: Size of training window
        step_size: Step size between windows
        min_train_size: Minimum training set size
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Sort by date
    df_sorted = df.sort(date_col)
    
    splits = []
    n_samples = len(df_sorted)
    
    # Start position
    start = 0
    
    while start + window_size < n_samples:
        # Training window
        train_end = start + window_size
        train_indices = np.arange(start, train_end)
        
        # Validation window (next step_size samples)
        val_start = train_end
        val_end = min(val_start + step_size, n_samples)
        val_indices = np.arange(val_start, val_end)
        
        if len(train_indices) >= min_train_size and len(val_indices) > 0:
            splits.append((train_indices, val_indices))
        
        # Move to next window
        start += step_size
    
    logger.info(f"Created {len(splits)} rolling window splits")
    logger.info(f"Window size: {window_size}, Step size: {step_size}")
    return splits


def calculate_brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate Brier score for multi-class classification.
    
    Args:
        y_true: True labels (0, 1, 2 for H, D, A)
        y_pred_proba: Predicted probabilities [n_samples, n_classes]
        
    Returns:
        Brier score (lower is better)
    """
    # Convert to one-hot encoding for Brier score calculation
    n_classes = y_pred_proba.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true]
    
    # Calculate Brier score for each class and average
    brier_scores = []
    for i in range(n_classes):
        brier_scores.append(brier_score_loss(y_true_onehot[:, i], y_pred_proba[:, i]))
    
    return np.mean(brier_scores)


def calculate_log_loss_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate log loss for multi-class classification.
    
    Args:
        y_true: True labels (0, 1, 2 for H, D, A)
        y_pred_proba: Predicted probabilities [n_samples, n_classes]
        
    Returns:
        Log loss score (lower is better)
    """
    return log_loss(y_true, y_pred_proba)


def calculate_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score for classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score (higher is better)
    """
    return accuracy_score(y_true, y_pred)


def calculate_classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_pred_proba: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        class_names: Names for classes (e.g., ['Home', 'Draw', 'Away'])
        
    Returns:
        Dictionary of metrics
    """
    if class_names is None:
        class_names = ['Home', 'Draw', 'Away']
    
    # Basic metrics
    accuracy = calculate_accuracy_score(y_true, y_pred)
    brier_score = calculate_brier_score(y_true, y_pred_proba)
    log_loss_score = calculate_log_loss_score(y_true, y_pred_proba)
    
    # Detailed classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    metrics = {
        'accuracy': accuracy,
        'brier_score': brier_score,
        'log_loss': log_loss_score,
        'classification_report': report,
        'class_names': class_names
    }
    
    return metrics


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Apply softmax function with temperature scaling.
    
    Args:
        x: Input array of logits
        temperature: Temperature parameter (lower = more confident)
        
    Returns:
        Softmax probabilities
    """
    # Apply temperature scaling
    x_scaled = x / temperature
    
    # Subtract max for numerical stability
    x_shifted = x_scaled - np.max(x_scaled, axis=-1, keepdims=True)
    
    # Apply softmax
    exp_x = np.exp(x_shifted)
    softmax_probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    return softmax_probs


def calibrate_probabilities_with_temperature(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    temperature_range: Tuple[float, float] = (0.1, 10.0),
    n_trials: int = 100
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Calibrate probabilities using temperature scaling.
    
    Args:
        y_true: True labels
        y_pred_proba: Raw predicted probabilities
        temperature_range: Range of temperatures to try
        n_trials: Number of temperature values to test
        
    Returns:
        Tuple of (calibrated_probabilities, best_temperature, calibration_info)
    """
    temperatures = np.logspace(
        np.log10(temperature_range[0]), 
        np.log10(temperature_range[1]), 
        n_trials
    )
    
    best_temperature = 1.0
    best_score = float('inf')
    
    # Convert probabilities to logits for temperature scaling
    logits = np.log(np.clip(y_pred_proba, 1e-10, 1.0))
    
    # Find best temperature
    for temp in temperatures:
        calibrated_proba = softmax(logits, temperature=temp)
        score = calculate_log_loss_score(y_true, calibrated_proba)
        
        if score < best_score:
            best_score = score
            best_temperature = temp
    
    # Apply best temperature
    best_calibrated_proba = softmax(logits, temperature=best_temperature)
    
    calibration_info = {
        'method': 'temperature_scaling',
        'best_temperature': best_temperature,
        'best_log_loss': best_score,
        'original_log_loss': calculate_log_loss_score(y_true, y_pred_proba),
        'temperature_range': temperature_range,
        'n_trials': n_trials
    }
    
    return best_calibrated_proba, best_temperature, calibration_info


def create_calibration_wrapper(
    method: str = 'isotonic',
    cv: int = 5
) -> Any:
    """
    Create a calibration wrapper for multi-class probabilities.
    
    Args:
        method: Calibration method ('isotonic' or 'sigmoid')
        cv: Cross-validation folds for calibration
        
    Returns:
        Calibrated classifier
    """
    if method == 'isotonic':
        return IsotonicRegression(out_of_bounds='clip')
    else:
        raise ValueError(f"Unsupported calibration method: {method}")


def calibrate_probabilities(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    method: str = 'isotonic'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Calibrate predicted probabilities using isotonic regression.
    
    Args:
        y_true: True labels
        y_pred_proba: Raw predicted probabilities
        method: Calibration method
        
    Returns:
        Tuple of (calibrated_probabilities, calibration_info)
    """
    n_classes = y_pred_proba.shape[1]
    calibrated_proba = np.zeros_like(y_pred_proba)
    calibration_models = {}
    
    # Calibrate each class separately (one-vs-rest approach)
    for i in range(n_classes):
        # Create binary labels for this class
        y_binary = (y_true == i).astype(int)
        
        # Fit calibration model
        if method == 'isotonic':
            cal_model = IsotonicRegression(out_of_bounds='clip')
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Fit on validation data
        cal_model.fit(y_pred_proba[:, i], y_binary)
        
        # Transform probabilities
        calibrated_proba[:, i] = cal_model.transform(y_pred_proba[:, i])
        
        calibration_models[f'class_{i}'] = cal_model
    
    # Renormalize to ensure probabilities sum to 1
    row_sums = calibrated_proba.sum(axis=1)
    calibrated_proba = calibrated_proba / row_sums[:, np.newaxis]
    
    calibration_info = {
        'method': method,
        'models': calibration_models,
        'pre_calibration_brier': calculate_brier_score(y_true, y_pred_proba),
        'post_calibration_brier': calculate_brier_score(y_true, calibrated_proba),
        'pre_calibration_logloss': calculate_log_loss_score(y_true, y_pred_proba),
        'post_calibration_logloss': calculate_log_loss_score(y_true, calibrated_proba)
    }
    
    return calibrated_proba, calibration_info


def extract_season_from_date(date_str: str) -> str:
    """
    Extract season string from date (e.g., '2024-08-17' -> '2024/25').
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        
    Returns:
        Season string in YYYY/YY format
    """
    try:
        year = int(date_str[:4])
        return f"{year}/{str(year + 1)[-2:]}"
    except (ValueError, IndexError):
        return "Unknown"


def calculate_team_rankings(points: Dict[str, int]) -> List[Tuple[str, int, int]]:
    """
    Calculate team rankings from points dictionary.
    
    Args:
        points: Dictionary mapping team names to points
        
    Returns:
        List of (team, points, rank) tuples, sorted by rank
    """
    sorted_teams = sorted(points.items(), key=lambda x: x[1], reverse=True)
    rankings = []
    
    for rank, (team, pts) in enumerate(sorted_teams, 1):
        rankings.append((team, pts, rank))
    
    return rankings


def format_probability(prob: float) -> str:
    """
    Format probability as percentage string.
    
    Args:
        prob: Probability between 0 and 1
        
    Returns:
        Formatted percentage string
    """
    return f"{prob * 100:.1f}%"


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    global RANDOM_SEED
    RANDOM_SEED = seed
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def create_feature_importance_plot(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Create and optionally save a feature importance plot.
    
    Args:
        feature_names: List of feature names
        importance_scores: Array of importance scores
        top_n: Number of top features to show
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        top_indices = sorted_indices[:top_n]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_indices)), importance_scores[top_indices])
        plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available, skipping feature importance plot")


def season_time_splits(
    df: pl.DataFrame, 
    season_col: str, 
    train_until: str, 
    valid_season: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create time-based train/validation split using season information.
    
    Args:
        df: DataFrame with match data
        season_col: Name of the season column
        train_until: Last season to include in training (e.g., "2023/24")
        valid_season: Season to use for validation (e.g., "2024/25")
        
    Returns:
        Tuple of (train_indices, valid_indices)
    """
    # Get training data (all seasons up to and including train_until)
    train_mask = df[season_col] <= train_until
    train_indices = np.where(train_mask.to_numpy())[0]
    
    # Get validation data (specific validation season)
    valid_mask = df[season_col] == valid_season
    valid_indices = np.where(valid_mask.to_numpy())[0]
    
    logger.info(f"Season split: Train until {train_until} ({len(train_indices)} samples), "
                f"Validation {valid_season} ({len(valid_indices)} samples)")
    
    return train_indices, valid_indices


def compute_brier(y_true_onehot: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """
    Compute Brier score for multi-class classification.
    
    Args:
        y_true_onehot: True probability distribution [n_samples, n_classes]
        y_pred_probs: Predicted probability distribution [n_samples, n_classes]
        
    Returns:
        Brier score (lower is better)
    """
    # Brier score is mean squared error between true and predicted probabilities
    brier_score = np.mean((y_true_onehot - y_pred_probs) ** 2)
    return brier_score


def compute_logloss(y_true_int: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """
    Compute log loss for multi-class classification.
    
    Args:
        y_true_int: True labels as integers [n_samples]
        y_pred_probs: Predicted probability matrix [n_samples, n_classes]
        
    Returns:
        Log loss score (lower is better)
    """
    # Convert true labels to one-hot encoding
    n_classes = y_pred_probs.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true_int]
    
    # Compute log loss
    log_loss_score = log_loss(y_true_onehot, y_pred_probs)
    return log_loss_score


def softmax3(logits: np.ndarray) -> np.ndarray:
    """
    Apply softmax function to 3-class logits.
    
    Args:
        logits: Input logits [n_samples, 3] for Home/Draw/Away
        
    Returns:
        Softmax probabilities [n_samples, 3]
    """
    # Subtract max for numerical stability
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    
    # Apply softmax
    exp_logits = np.exp(logits_shifted)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    return softmax_probs


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test season extraction
    test_dates = ['2024-08-17', '2025-05-19']
    for date in test_dates:
        season = extract_season_from_date(date)
        print(f"Date: {date} -> Season: {season}")
    
    # Test rankings
    test_points = {'Arsenal': 89, 'Man City': 91, 'Liverpool': 88}
    rankings = calculate_team_rankings(test_points)
    print(f"Rankings: {rankings}")
    
    # Test softmax
    test_logits = np.array([[1.0, 0.5, 0.1]])
    softmax_probs = softmax(test_logits, temperature=1.0)
    print(f"Softmax probabilities: {softmax_probs}")
    
    print("Utility functions test complete!") 