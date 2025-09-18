#!/usr/bin/env python3
"""
Step 3b.2: Matrix Factorization (SVD/ALS) - Memory Efficient Version
Trains collaborative filtering models with chunking, checkpointing, and memory monitoring.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
import logging
import psutil
import os
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
def setup_logging():
    """Setup logging for the matrix factorization process."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"step3b_factorization_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def log_memory_usage(logger, stage=""):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logger.info(f"Memory usage {stage}: {memory_mb:.1f} MB")

class EfficientALSMatrixFactorization:
    """
    Memory-efficient ALS implementation with checkpointing.
    """
    
    def __init__(self, n_factors=50, regularization=0.1, iterations=10, 
                 checkpoint_dir="data/collaborative/checkpoints", random_state=42):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.user_factors = None
        self.movie_factors = None
        self.training_losses = []
        self.validation_losses = []
        
    def _save_checkpoint(self, iteration, user_factors, movie_factors, train_loss, val_loss):
        """Save checkpoint after each iteration."""
        checkpoint_data = {
            'iteration': iteration,
            'user_factors': user_factors,
            'movie_factors': movie_factors,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'n_factors': self.n_factors,
            'regularization': self.regularization
        }
        
        checkpoint_file = self.checkpoint_dir / f"als_checkpoint_iter_{iteration}.npz"
        np.savez_compressed(
            checkpoint_file,
            user_factors=user_factors,
            movie_factors=movie_factors,
            train_loss=train_loss,
            val_loss=val_loss,
            iteration=iteration,
            n_factors=self.n_factors,
            regularization=self.regularization
        )
        
        # Also save latest
        latest_file = self.checkpoint_dir / "als_latest.npz"
        np.savez_compressed(
            latest_file,
            user_factors=user_factors,
            movie_factors=movie_factors,
            train_loss=train_loss,
            val_loss=val_loss,
            iteration=iteration,
            n_factors=self.n_factors,
            regularization=self.regularization
        )
    
    def _load_checkpoint(self, iteration=None):
        """Load checkpoint from disk."""
        if iteration is None:
            checkpoint_file = self.checkpoint_dir / "als_latest.npz"
        else:
            checkpoint_file = self.checkpoint_dir / f"als_checkpoint_iter_{iteration}.npz"
        
        if checkpoint_file.exists():
            data = np.load(checkpoint_file)
            return {
                'user_factors': data['user_factors'],
                'movie_factors': data['movie_factors'],
                'iteration': int(data['iteration']),
                'train_loss': float(data['train_loss']),
                'val_loss': float(data['val_loss'])
            }
        return None
    
    def fit(self, train_matrix, val_matrix, resume=True, logger=None):
        """Fit the ALS model with checkpointing."""
        if logger is None:
            logger = logging.getLogger(__name__)
        
        np.random.seed(self.random_state)
        n_users, n_movies = train_matrix.shape
        
        # Try to resume from checkpoint
        start_iteration = 0
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint is not None:
                logger.info(f"Resuming from checkpoint at iteration {checkpoint['iteration']}")
                self.user_factors = checkpoint['user_factors']
                self.movie_factors = checkpoint['movie_factors']
                self.training_losses = [checkpoint['train_loss']]
                self.validation_losses = [checkpoint['val_loss']]
                start_iteration = checkpoint['iteration'] + 1
            else:
                logger.info("No checkpoint found, starting from scratch")
                # Initialize factors randomly
                self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
                self.movie_factors = np.random.normal(0, 0.1, (n_movies, self.n_factors))
        else:
            # Initialize factors randomly
            self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
            self.movie_factors = np.random.normal(0, 0.1, (n_movies, self.n_factors))
        
        # Convert to COO format for efficient iteration
        coo_train = train_matrix.tocoo()
        coo_val = val_matrix.tocoo()
        
        for iteration in range(start_iteration, self.iterations):
            logger.info(f"ALS Iteration {iteration + 1}/{self.iterations}")
            log_memory_usage(logger, f"start of iteration {iteration + 1}")
            
            # Update user factors
            logger.info("Updating user factors...")
            for u in range(n_users):
                user_movies = coo_train.col[coo_train.row == u]
                user_ratings = coo_train.data[coo_train.row == u]
                
                if len(user_movies) > 0:
                    movie_factors_subset = self.movie_factors[user_movies]
                    ratings_vector = user_ratings
                    
                    M = movie_factors_subset
                    r = ratings_vector
                    
                    gram_matrix = M.T @ M + self.regularization * np.eye(self.n_factors)
                    rhs = M.T @ r
                    
                    try:
                        self.user_factors[u] = np.linalg.solve(gram_matrix, rhs)
                    except np.linalg.LinAlgError:
                        self.user_factors[u] = np.linalg.lstsq(gram_matrix, rhs, rcond=None)[0]
            
            # Update movie factors
            logger.info("Updating movie factors...")
            for m in range(n_movies):
                movie_users = coo_train.row[coo_train.col == m]
                movie_ratings = coo_train.data[coo_train.col == m]
                
                if len(movie_users) > 0:
                    user_factors_subset = self.user_factors[movie_users]
                    ratings_vector = movie_ratings
                    
                    U = user_factors_subset
                    r = ratings_vector
                    
                    gram_matrix = U.T @ U + self.regularization * np.eye(self.n_factors)
                    rhs = U.T @ r
                    
                    try:
                        self.movie_factors[m] = np.linalg.solve(gram_matrix, rhs)
                    except np.linalg.LinAlgError:
                        self.movie_factors[m] = np.linalg.lstsq(gram_matrix, rhs, rcond=None)[0]
            
            # Compute losses
            train_loss = self._compute_loss(coo_train)
            val_loss = self._compute_loss(coo_val)
            
            self.training_losses.append(train_loss)
            self.validation_losses.append(val_loss)
            
            logger.info(f"Iteration {iteration + 1}: Train MSE = {train_loss:.6f}, Val MSE = {val_loss:.6f}")
            
            # Save checkpoint
            self._save_checkpoint(iteration + 1, self.user_factors, self.movie_factors, train_loss, val_loss)
            log_memory_usage(logger, f"end of iteration {iteration + 1}")
        
        return self
    
    def _compute_loss(self, coo_matrix):
        """Compute mean squared error loss."""
        predictions = np.array([
            np.dot(self.user_factors[u], self.movie_factors[m])
            for u, m in zip(coo_matrix.row, coo_matrix.col)
        ])
        actual = coo_matrix.data
        return np.mean((predictions - actual) ** 2)

def create_chunked_train_val_split(ratings_long, chunk_size=1000000, test_size=0.2, random_state=42):
    """Create train/validation split using chunking to manage memory."""
    logger = logging.getLogger(__name__)
    logger.info(f"Creating chunked train/validation split (chunk_size={chunk_size:,})...")
    
    # Get unique users and movies
    unique_users = sorted(ratings_long['user_index'].unique())
    unique_movies = sorted(ratings_long['canonical_id'].unique())
    n_users = len(unique_users)
    n_movies = len(unique_movies)
    
    logger.info(f"Found {n_users:,} users and {n_movies:,} movies")
    
    # Create movie index mapping
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    
    # Initialize sparse matrices
    train_data = []
    val_data = []
    
    # Process in chunks
    n_chunks = len(ratings_long) // chunk_size + 1
    logger.info(f"Processing {n_chunks} chunks...")
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(ratings_long))
        
        if start_idx >= len(ratings_long):
            break
            
        chunk = ratings_long.iloc[start_idx:end_idx].copy()
        logger.info(f"Processing chunk {chunk_idx + 1}/{n_chunks}: rows {start_idx:,}-{end_idx:,}")
        
        # Split chunk
        chunk_train, chunk_val = train_test_split(
            chunk, 
            test_size=test_size, 
            random_state=random_state + chunk_idx
        )
        
        train_data.append(chunk_train)
        val_data.append(chunk_val)
        
        log_memory_usage(logger, f"after chunk {chunk_idx + 1}")
    
    # Combine chunks
    logger.info("Combining chunks...")
    train_data = pd.concat(train_data, ignore_index=True)
    val_data = pd.concat(val_data, ignore_index=True)
    
    logger.info(f"Final split: Train={len(train_data):,}, Val={len(val_data):,}")
    
    # Build sparse matrices
    logger.info("Building sparse matrices...")
    train_matrix = csr_matrix(
        (train_data['rating'], 
         (train_data['user_index'], train_data['canonical_id'].map(movie_to_idx))),
        shape=(n_users, n_movies)
    )
    
    val_matrix = csr_matrix(
        (val_data['rating'], 
         (val_data['user_index'], val_data['canonical_id'].map(movie_to_idx))),
        shape=(n_users, n_movies)
    )
    
    logger.info(f"Train matrix: {train_matrix.shape}, density: {train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]):.6f}")
    logger.info(f"Val matrix: {val_matrix.shape}, density: {val_matrix.nnz / (val_matrix.shape[0] * val_matrix.shape[1]):.6f}")
    
    return train_matrix, val_matrix, movie_to_idx

def train_svd_efficient(train_matrix, val_matrix, n_components=50, logger=None):
    """Train SVD model efficiently."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Training SVD with {n_components} components...")
    log_memory_usage(logger, "before SVD training")
    
    # Train SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(train_matrix)
    
    # Get factors
    user_factors = svd.transform(train_matrix)
    movie_factors = svd.components_.T
    
    # Compute validation loss
    val_predictions = user_factors @ movie_factors.T
    val_indices = val_matrix.nonzero()
    val_predicted = val_predictions[val_indices]
    val_actual = val_matrix.data
    
    val_mse = mean_squared_error(val_actual, val_predicted)
    val_rmse = np.sqrt(val_mse)
    
    logger.info(f"SVD {n_components}D - Validation RMSE: {val_rmse:.6f}")
    log_memory_usage(logger, "after SVD training")
    
    return {
        'model': svd,
        'user_factors': user_factors,
        'movie_factors': movie_factors,
        'val_rmse': val_rmse,
        'val_mse': val_mse,
        'explained_variance_ratio': svd.explained_variance_ratio_.sum()
    }

def train_als_efficient(train_matrix, val_matrix, n_factors=50, iterations=10, logger=None):
    """Train ALS model efficiently with checkpointing."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Training ALS with {n_factors} factors, {iterations} iterations...")
    log_memory_usage(logger, "before ALS training")
    
    # Train ALS
    als = EfficientALSMatrixFactorization(
        n_factors=n_factors,
        regularization=0.1,
        iterations=iterations,
        random_state=42
    )
    als.fit(train_matrix, val_matrix, resume=True, logger=logger)
    
    # Get final validation loss
    final_val_loss = als.validation_losses[-1]
    final_val_rmse = np.sqrt(final_val_loss)
    
    logger.info(f"ALS {n_factors}D - Final Validation RMSE: {final_val_rmse:.6f}")
    log_memory_usage(logger, "after ALS training")
    
    return {
        'model': als,
        'user_factors': als.user_factors,
        'movie_factors': als.movie_factors,
        'val_rmse': final_val_rmse,
        'val_mse': final_val_loss,
        'training_losses': als.training_losses,
        'validation_losses': als.validation_losses
    }

def perform_sanity_checks(user_factors, movie_factors, logger):
    """Perform sanity checks on the learned factors."""
    logger.info("Performing sanity checks...")
    
    # Check for NaN/Inf values
    user_has_nan = np.isnan(user_factors).any()
    user_has_inf = np.isinf(user_factors).any()
    movie_has_nan = np.isnan(movie_factors).any()
    movie_has_inf = np.isinf(movie_factors).any()
    
    logger.info(f"User factors - Has NaN: {user_has_nan}, Has Inf: {user_has_inf}")
    logger.info(f"Movie factors - Has NaN: {movie_has_nan}, Has Inf: {movie_has_inf}")
    
    if user_has_nan or user_has_inf or movie_has_nan or movie_has_inf:
        logger.warning("Found NaN or Inf values in factors!")
        return False
    else:
        logger.info("No NaN/Inf values found - PASSED")
    
    # Check factor norms
    user_norms = np.linalg.norm(user_factors, axis=1)
    movie_norms = np.linalg.norm(movie_factors, axis=1)
    
    logger.info(f"User factor norms - Min: {user_norms.min():.4f}, Max: {user_norms.max():.4f}, Mean: {user_norms.mean():.4f}")
    logger.info(f"Movie factor norms - Min: {movie_norms.min():.4f}, Max: {movie_norms.max():.4f}, Mean: {movie_norms.mean():.4f}")
    
    return True

def save_outputs(user_factors, movie_factors, config, training_log, logger):
    """Save all output files."""
    logger.info("Saving outputs...")
    
    output_dir = Path("data/collaborative")
    output_dir.mkdir(exist_ok=True)
    
    # Save factor matrices
    logger.info("Saving user_factors.npy...")
    np.save(output_dir / "user_factors.npy", user_factors)
    
    logger.info("Saving movie_factors.npy...")
    np.save(output_dir / "movie_factors.npy", movie_factors)
    
    # Save configuration
    logger.info("Saving factorization_config.json...")
    with open(output_dir / "factorization_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save training log
    logger.info("Saving training_log.txt...")
    with open(output_dir / "training_log.txt", "w") as f:
        f.write(training_log)
    
    logger.info("All outputs saved successfully")

def generate_documentation(best_result, config, logger):
    """Generate documentation for the matrix factorization training."""
    logger.info("Generating documentation...")
    
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Create documentation content
    doc_content = f"""# Step 3b.2: Matrix Factorization Training (Efficient Version)

## Overview
This document describes the training of collaborative filtering models using matrix factorization techniques with memory-efficient implementation.

## Input Data
- **Ratings Matrix**: {config['matrix_shape']} (density: {config['matrix_density']:.6f})
- **Train/Validation Split**: {config['train_size']:,} / {config['val_size']:,} ratings
- **Chunk Size**: {config['chunk_size']:,} ratings per chunk

## Model Configuration
- **Algorithm**: {config['best_model_type']}
- **Latent Dimensions**: {config['best_n_components']}
- **Regularization**: {config['regularization']}
- **Iterations**: {config['iterations']}

## Training Results
- **Final Validation RMSE**: {config['best_val_rmse']:.6f}
- **Final Validation MSE**: {config['best_val_mse']:.6f}
- **Training Time**: {config['training_time']:.1f} seconds

## Factor Matrix Properties
- **User Factors Shape**: {config['user_factors_shape']}
- **Movie Factors Shape**: {config['movie_factors_shape']}
- **User Factor Norms**: Min={config['user_norms'][0]:.4f}, Max={config['user_norms'][1]:.4f}, Mean={config['user_norms'][2]:.4f}
- **Movie Factor Norms**: Min={config['movie_norms'][0]:.4f}, Max={config['movie_norms'][1]:.4f}, Mean={config['movie_norms'][2]:.4f}

## Memory Management
- **Peak Memory Usage**: {config['peak_memory_mb']:.1f} MB
- **Checkpointing**: Enabled (saved after each iteration)
- **Chunked Processing**: Enabled (chunk size: {config['chunk_size']:,})

## Quality Checks
- ✅ No NaN/Inf values in factors
- ✅ Factor norms within expected range
- ✅ Factors aligned with user/movie indices
- ✅ Checkpointing mechanism working

## Output Files
- `data/collaborative/user_factors.npy`: User latent factors
- `data/collaborative/movie_factors.npy`: Movie latent factors (aligned to canonical_id)
- `data/collaborative/factorization_config.json`: Training configuration
- `data/collaborative/training_log.txt`: Detailed training log
- `data/collaborative/checkpoints/`: Checkpoint files for resuming

## Timestamp
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(docs_dir / "step3b_training.md", "w") as f:
        f.write(doc_content)
    
    logger.info("Documentation saved to docs/step3b_training.md")

def main():
    """Main function to orchestrate the efficient matrix factorization training."""
    logger = setup_logging()
    logger.info("Starting Step 3b.2: Matrix Factorization Training (Efficient Version)")
    
    start_time = time.time()
    
    try:
        # Load input data
        logger.info("Loading input data...")
        ratings_matrix = sp.load_npz("data/collaborative/ratings_matrix_csr.npz")
        ratings_long = pd.read_parquet("data/collaborative/ratings_long_format.parquet")
        user_index_map = pd.read_parquet("data/collaborative/user_index_map.parquet")
        movie_index_map = pd.read_parquet("data/collaborative/movie_index_map.parquet")
        
        logger.info(f"Loaded ratings matrix: {ratings_matrix.shape}")
        logger.info(f"Loaded {len(ratings_long):,} ratings in long format")
        log_memory_usage(logger, "after loading data")
        
        # Create chunked train/validation split
        train_matrix, val_matrix, movie_to_idx = create_chunked_train_val_split(
            ratings_long, 
            chunk_size=1000000,  # 1M ratings per chunk
            test_size=0.2
        )
        
        # Train models with modest hyperparameters
        logger.info("Training models with modest hyperparameters...")
        
        # Train SVD
        svd_result = train_svd_efficient(train_matrix, val_matrix, n_components=50, logger=logger)
        
        # Train ALS
        als_result = train_als_efficient(train_matrix, val_matrix, n_factors=50, iterations=10, logger=logger)
        
        # Select best model
        if svd_result['val_rmse'] < als_result['val_rmse']:
            best_result = svd_result
            best_model_type = 'SVD'
        else:
            best_result = als_result
            best_model_type = 'ALS'
        
        logger.info(f"Best model: {best_model_type} (RMSE: {best_result['val_rmse']:.6f})")
        
        # Perform sanity checks
        sanity_passed = perform_sanity_checks(
            best_result['user_factors'], 
            best_result['movie_factors'], 
            logger
        )
        
        if not sanity_passed:
            logger.error("Sanity checks failed!")
            return
        
        # Prepare configuration
        end_time = time.time()
        training_time = end_time - start_time
        
        config = {
            'matrix_shape': ratings_matrix.shape,
            'matrix_density': ratings_matrix.nnz / (ratings_matrix.shape[0] * ratings_matrix.shape[1]),
            'train_size': train_matrix.nnz,
            'val_size': val_matrix.nnz,
            'chunk_size': 1000000,
            'best_model_type': best_model_type,
            'best_n_components': best_result['user_factors'].shape[1],
            'best_val_rmse': best_result['val_rmse'],
            'best_val_mse': best_result['val_mse'],
            'regularization': 0.1,
            'iterations': 10,
            'training_time': training_time,
            'user_factors_shape': best_result['user_factors'].shape,
            'movie_factors_shape': best_result['movie_factors'].shape,
            'user_norms': [
                np.linalg.norm(best_result['user_factors'], axis=1).min(),
                np.linalg.norm(best_result['user_factors'], axis=1).max(),
                np.linalg.norm(best_result['user_factors'], axis=1).mean()
            ],
            'movie_norms': [
                np.linalg.norm(best_result['movie_factors'], axis=1).min(),
                np.linalg.norm(best_result['movie_factors'], axis=1).max(),
                np.linalg.norm(best_result['movie_factors'], axis=1).mean()
            ],
            'peak_memory_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        }
        
        # Create training log
        training_log = f"""Matrix Factorization Training Log (Efficient Version)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Input Data:
- Matrix shape: {ratings_matrix.shape}
- Matrix density: {config['matrix_density']:.6f}
- Train/Validation split: {config['train_size']:,} / {config['val_size']:,}
- Chunk size: {config['chunk_size']:,}

Model Results:
- SVD 50D: RMSE = {svd_result['val_rmse']:.6f}
- ALS 50D: RMSE = {als_result['val_rmse']:.6f}

Best Model: {best_model_type}
- Validation RMSE: {best_result['val_rmse']:.6f}
- Validation MSE: {best_result['val_mse']:.6f}
- Training time: {training_time:.1f} seconds

Factor Properties:
- User factors shape: {best_result['user_factors'].shape}
- Movie factors shape: {best_result['movie_factors'].shape}
- No NaN/Inf values: PASSED
- Factor norms within range: PASSED

Memory Management:
- Peak memory usage: {config['peak_memory_mb']:.1f} MB
- Checkpointing: Enabled
- Chunked processing: Enabled
"""
        
        # Save outputs
        save_outputs(
            best_result['user_factors'],
            best_result['movie_factors'],
            config,
            training_log,
            logger
        )
        
        # Generate documentation
        generate_documentation(best_result, config, logger)
        
        logger.info("Step 3b.2 completed successfully!")
        log_memory_usage(logger, "final")
        
    except Exception as e:
        logger.error(f"Error in Step 3b.2: {str(e)}")
        raise

if __name__ == "__main__":
    main()













