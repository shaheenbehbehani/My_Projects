#!/usr/bin/env python3
"""
Tests for Phase 4: Modeling Pipeline

This test suite validates:
- Model training functionality
- Model evaluation capabilities
- Ensemble modeling
- Hyperparameter optimization
- Pipeline orchestration
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import Phase 4 components
from src.model.utils import (
    create_season_based_splits,
    calculate_classification_metrics,
    softmax,
    calibrate_probabilities_with_temperature
)


class TestPhase4Utils:
    """Test Phase 4 utility functions."""
    
    def test_softmax(self):
        """Test softmax function with temperature scaling."""
        # Test basic softmax
        logits = np.array([[1.0, 0.5, 0.1]])
        probs = softmax(logits, temperature=1.0)
        
        assert probs.shape == (1, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0) and np.all(probs <= 1)
        
        # Test temperature scaling
        probs_cold = softmax(logits, temperature=0.5)
        probs_hot = softmax(logits, temperature=2.0)
        
        # Lower temperature should make probabilities more extreme
        assert np.std(probs_cold) > np.std(probs_hot)
    
    def test_calculate_classification_metrics(self):
        """Test classification metrics calculation."""
        # Create mock data
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])
        y_pred_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1]
        ])
        
        metrics = calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'accuracy' in metrics
        assert 'brier_score' in metrics
        assert 'log_loss' in metrics
        assert 'classification_report' in metrics
        
        # Perfect predictions should give accuracy = 1.0
        assert metrics['accuracy'] == 1.0
    
    def test_create_season_based_splits(self):
        """Test season-based cross-validation splits."""
        # Create mock data with dates
        dates = pl.date_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2023, 12, 31),
            interval="1d"
        )
        
        # Create mock DataFrame
        df = pl.DataFrame({
            'date': dates,
            'feature1': np.random.randn(len(dates)),
            'feature2': np.random.randn(len(dates))
        })
        
        splits = create_season_based_splits(df, min_seasons=2)
        
        # Should create splits for 2022 and 2023 as validation
        assert len(splits) >= 1
        
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            assert len(set(train_idx).intersection(set(val_idx))) == 0


class TestPhase4Components:
    """Test Phase 4 component classes."""
    
    @patch('src.model.train_model.MatchModelTrainer.load_match_dataset')
    def test_model_trainer_initialization(self, mock_load):
        """Test model trainer initialization."""
        from src.model.train_model import MatchModelTrainer
        
        # Mock the dataset loading
        mock_df = pl.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'result_label': [0, 1, 0]
        })
        mock_load.return_value = mock_df
        
        trainer = MatchModelTrainer("data", "models")
        
        assert trainer.data_dir == Path("data")
        assert trainer.models_dir == Path("models")
        assert trainer.processed_dir == Path("data") / "processed"
    
    @patch('src.model.evaluate_model.ModelEvaluator.load_match_dataset')
    def test_model_evaluator_initialization(self, mock_load):
        """Test model evaluator initialization."""
        from src.model.evaluate_model import ModelEvaluator
        
        # Mock the dataset loading
        mock_df = pl.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'result_label': [0, 1, 0]
        })
        mock_load.return_value = mock_df
        
        evaluator = ModelEvaluator("data", "models", "outputs")
        
        assert evaluator.data_dir == Path("data")
        assert evaluator.models_dir == Path("models")
        assert evaluator.outputs_dir == Path("outputs")
        assert (evaluator.outputs_dir / "figures").exists()
        assert (evaluator.outputs_dir / "reports").exists()
    
    @patch('src.model.ensemble_model.ModelEnsemble.load_match_dataset')
    def test_ensemble_model_initialization(self, mock_load):
        """Test ensemble model initialization."""
        from src.model.ensemble_model import ModelEnsemble
        
        # Mock the dataset loading
        mock_df = pl.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'result_label': [0, 1, 0]
        })
        mock_load.return_value = mock_df
        
        ensemble = ModelEnsemble("data", "models", "outputs")
        
        assert ensemble.data_dir == Path("data")
        assert ensemble.models_dir == Path("models")
        assert ensemble.outputs_dir == Path("outputs")
        assert ensemble.base_models == {}
        assert ensemble.ensemble_model is None
    
    @patch('src.model.optimize_hyperparameters.HyperparameterOptimizer.load_match_dataset')
    def test_hyperparameter_optimizer_initialization(self, mock_load):
        """Test hyperparameter optimizer initialization."""
        from src.model.optimize_hyperparameters import HyperparameterOptimizer
        
        # Mock the dataset loading
        mock_df = pl.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'result_label': [0, 1, 0]
        })
        mock_load.return_value = mock_df
        
        optimizer = HyperparameterOptimizer("data", "models", "outputs")
        
        assert optimizer.data_dir == Path("data")
        assert optimizer.models_dir == Path("models")
        assert optimizer.outputs_dir == Path("outputs")
        assert optimizer.optimization_results == {}
        assert optimizer.best_models == {}


class TestPhase4Pipeline:
    """Test Phase 4 pipeline orchestration."""
    
    @patch('src.model.run_phase4.Phase4Orchestrator._run_model_training')
    @patch('src.model.run_phase4.Phase4Orchestrator._run_hyperparameter_optimization')
    @patch('src.model.run_phase4.Phase4Orchestrator._run_model_evaluation')
    @patch('src.model.run_phase4.Phase4Orchestrator._run_ensemble_modeling')
    def test_pipeline_orchestration(self, mock_ensemble, mock_eval, mock_opt, mock_train):
        """Test pipeline orchestration."""
        from src.model.run_phase4 import Phase4Orchestrator
        
        # Mock all pipeline steps
        mock_train.return_value = None
        mock_opt.return_value = None
        mock_eval.return_value = None
        mock_ensemble.return_value = None
        
        orchestrator = Phase4Orchestrator("data", "models", "outputs")
        
        # Run pipeline
        results = orchestrator.run_phase4_pipeline()
        
        # Verify all steps were called
        mock_train.assert_called_once()
        mock_opt.assert_called_once()
        mock_eval.assert_called_once()
        mock_ensemble.assert_called_once()
        
        # Verify results structure
        assert 'training' in results
        assert 'optimization' in results
        assert 'evaluation' in results
        assert 'ensemble' in results
        assert 'summary' in results


def test_phase4_file_structure():
    """Test that all Phase 4 files exist and are properly structured."""
    phase4_files = [
        'src/model/utils.py',
        'src/model/evaluate_model.py',
        'src/model/ensemble_model.py',
        'src/model/optimize_hyperparameters.py',
        'src/model/run_phase4.py'
    ]
    
    for file_path in phase4_files:
        assert Path(file_path).exists(), f"Phase 4 file missing: {file_path}"
        
        # Check that files have proper Python structure
        with open(file_path, 'r') as f:
            content = f.read()
            assert '#!/usr/bin/env python3' in content, f"Missing shebang in {file_path}"
            assert '"""' in content, f"Missing docstring in {file_path}"
            assert 'import' in content, f"Missing imports in {file_path}"


if __name__ == "__main__":
    """Run tests directly with pytest."""
    pytest.main([__file__, "-v", "--tb=short"]) 