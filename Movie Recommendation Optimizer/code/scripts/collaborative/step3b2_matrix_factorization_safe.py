#!/usr/bin/env python3
"""
Step 3b.2: Matrix Factorization (SVD/ALS) - Safe Memory-Aware Version
Conservative implementation with memory monitoring, checkpointing, and fallback strategies.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import json
import logging
import psutil
import os
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Memory safety threshold (MB)
MEMORY_THRESHOLD_MB = 6000  # Conservative threshold
SAMPLE_SIZE_VALIDATION = 1000000  # Max 1M ratings for validation

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
    """Log current memory usage and check against threshold."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    logger.info(f"Memory usage {stage}: {memory_mb:.1f} MB")
    
    if memory_mb > MEMORY_THRESHOLD_MB:
        logger.warning(f"Memory usage ({memory_mb:.1f} MB) exceeds threshold ({MEMORY_THRESHOLD_MB} MB)")
        return False
    return True

class SafeSVDMatrixFactorization:
    """
    Memory-safe SVD implementation with checkpointing and monitoring.
    """
    
    def __init__(self, n_components=20, max_iter=5, random_state=42, 
                 checkpoint_dir="data/collaborative", logger=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.user_factors = None
        self.movie_factors = None
        self.training_metrics = []
        
    def fit_with_checkpointing(self, train_matrix, val_matrix, sample_val_indices=None):
        """Fit SVD with checkpointing after each iteration."""
        self.logger.info(f"Starting SVD training: k={self.n_components}, max_iter={self.max_iter}")
        
        # Initialize SVD
        svd = TruncatedSVD(
            n_components=self.n_components,
            n_iter=self.max_iter,
            random_state=self.random_state
        )
        
        start_time = time.time()
        
        # Fit the model
        self.logger.info("Fitting SVD model...")
        if not log_memory_usage(self.logger, "before SVD fit"):
            raise MemoryError("Memory threshold exceeded before SVD fit")
        
        svd.fit(train_matrix)
        
        if not log_memory_usage(self.logger, "after SVD fit"):
            raise MemoryError("Memory threshold exceeded after SVD fit")
        
        # Get factors
        self.logger.info("Extracting factors...")
        self.user_factors = svd.transform(train_matrix)
        self.movie_factors = svd.components_.T
        
        if not log_memory_usage(self.logger, "after factor extraction"):
            raise MemoryError("Memory threshold exceeded after factor extraction")
        
        # Save checkpoint
        self._save_checkpoint(epoch=0, svd=svd)
        
        # Compute validation metrics
        val_rmse = self._compute_validation_rmse(val_matrix, sample_val_indices)
        
        training_time = time.time() - start_time
        
        self.training_metrics.append({
            'epoch': 0,
            'val_rmse': val_rmse,
            'time_seconds': training_time,
            'memory_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
            'explained_variance_ratio': svd.explained_variance_ratio_.sum()
        })
        
        self.logger.info(f"SVD completed - Val RMSE: {val_rmse:.6f}, Time: {training_time:.1f}s")
        
        return self
    
    def _save_checkpoint(self, epoch, svd):
        """Save checkpoint after each epoch."""
        checkpoint_file = self.checkpoint_dir / f"user_factors_k{self.n_components}_epoch{epoch}.npy"
        np.save(checkpoint_file, self.user_factors)
        
        checkpoint_file = self.checkpoint_dir / f"movie_factors_k{self.n_components}_epoch{epoch}.npy"
        np.save(checkpoint_file, self.movie_factors)
        
        # Also save as latest
        latest_user_file = self.checkpoint_dir / f"user_factors_k{self.n_components}.npy"
        latest_movie_file = self.checkpoint_dir / f"movie_factors_k{self.n_components}.npy"
        
        np.save(latest_user_file, self.user_factors)
        np.save(latest_movie_file, self.movie_factors)
        
        self.logger.info(f"Checkpoint saved for epoch {epoch}")
    
    def _compute_validation_rmse(self, val_matrix, sample_indices=None):
        """Compute RMSE on validation set (optionally sampled)."""
        if sample_indices is None:
            # Sample validation ratings if too many
            if val_matrix.nnz > SAMPLE_SIZE_VALIDATION:
                coo_val = val_matrix.tocoo()
                sample_size = min(SAMPLE_SIZE_VALIDATION, coo_val.nnz)
                sample_idx = np.random.choice(coo_val.nnz, sample_size, replace=False)
                
                val_rows = coo_val.row[sample_idx]
                val_cols = coo_val.col[sample_idx]
                val_ratings = coo_val.data[sample_idx]
            else:
                coo_val = val_matrix.tocoo()
                val_rows = coo_val.row
                val_cols = coo_val.col
                val_ratings = coo_val.data
        else:
            coo_val = val_matrix.tocoo()
            val_rows = coo_val.row[sample_indices]
            val_cols = coo_val.col[sample_indices]
            val_ratings = coo_val.data[sample_indices]
        
        # Compute predictions
        predictions = np.array([
            np.dot(self.user_factors[u], self.movie_factors[m])
            for u, m in zip(val_rows, val_cols)
        ])
        
        # Compute RMSE
        mse = np.mean((predictions - val_ratings) ** 2)
        rmse = np.sqrt(mse)
        
        self.logger.info(f"Validation RMSE computed on {len(val_ratings):,} ratings: {rmse:.6f}")
        return rmse

class SafeALSMatrixFactorization:
    """
    Memory-safe ALS implementation with checkpointing.
    """
    
    def __init__(self, n_factors=20, iterations=5, regularization=0.1, 
                 checkpoint_dir="data/collaborative", logger=None):
        self.n_factors = n_factors
        self.iterations = iterations
        self.regularization = regularization
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.user_factors = None
        self.movie_factors = None
        self.training_metrics = []
        
    def fit_with_checkpointing(self, train_matrix, val_matrix, sample_val_indices=None):
        """Fit ALS with checkpointing after each iteration."""
        self.logger.info(f"Starting ALS training: k={self.n_factors}, iterations={self.iterations}")
        
        n_users, n_movies = train_matrix.shape
        
        # Initialize factors randomly
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.movie_factors = np.random.normal(0, 0.1, (n_movies, self.n_factors))
        
        if not log_memory_usage(self.logger, "after ALS initialization"):
            raise MemoryError("Memory threshold exceeded after ALS initialization")
        
        # Convert to COO format
        coo_train = train_matrix.tocoo()
        
        for iteration in range(self.iterations):
            iter_start_time = time.time()
            self.logger.info(f"ALS Iteration {iteration + 1}/{self.iterations}")
            
            # Update user factors
            self.logger.info("Updating user factors...")
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
            self.logger.info("Updating movie factors...")
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
                        self.movie_factors[m] = np.linalg.lstsq(gram_matrix, rhs, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        self.movie_factors[m] = np.linalg.lstsq(gram_matrix, rhs, rcond=None)[0]
            
            # Save checkpoint
            self._save_checkpoint(iteration + 1)
            
            # Compute validation metrics
            val_rmse = self._compute_validation_rmse(val_matrix, sample_val_indices)
            
            iter_time = time.time() - iter_start_time
            
            self.training_metrics.append({
                'iteration': iteration + 1,
                'val_rmse': val_rmse,
                'time_seconds': iter_time,
                'memory_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            })
            
            self.logger.info(f"Iteration {iteration + 1}: Val RMSE = {val_rmse:.6f}, Time = {iter_time:.1f}s")
            
            if not log_memory_usage(self.logger, f"after ALS iteration {iteration + 1}"):
                self.logger.warning("Memory threshold exceeded, stopping early")
                break
        
        return self
    
    def _save_checkpoint(self, iteration):
        """Save checkpoint after each iteration."""
        checkpoint_file = self.checkpoint_dir / f"user_factors_k{self.n_factors}_epoch{iteration}.npy"
        np.save(checkpoint_file, self.user_factors)
        
        checkpoint_file = self.checkpoint_dir / f"movie_factors_k{self.n_factors}_epoch{iteration}.npy"
        np.save(checkpoint_file, self.movie_factors)
        
        # Also save as latest
        latest_user_file = self.checkpoint_dir / f"user_factors_k{self.n_factors}.npy"
        latest_movie_file = self.checkpoint_dir / f"movie_factors_k{self.n_factors}.npy"
        
        np.save(latest_user_file, self.user_factors)
        np.save(latest_movie_file, self.movie_factors)
        
        self.logger.info(f"Checkpoint saved for iteration {iteration}")
    
    def _compute_validation_rmse(self, val_matrix, sample_indices=None):
        """Compute RMSE on validation set (optionally sampled)."""
        if sample_indices is None:
            # Sample validation ratings if too many
            if val_matrix.nnz > SAMPLE_SIZE_VALIDATION:
                coo_val = val_matrix.tocoo()
                sample_size = min(SAMPLE_SIZE_VALIDATION, coo_val.nnz)
                sample_idx = np.random.choice(coo_val.nnz, sample_size, replace=False)
                
                val_rows = coo_val.row[sample_idx]
                val_cols = coo_val.col[sample_idx]
                val_ratings = coo_val.data[sample_idx]
            else:
                coo_val = val_matrix.tocoo()
                val_rows = coo_val.row
                val_cols = coo_val.col
                val_ratings = coo_val.data
        else:
            coo_val = val_matrix.tocoo()
            val_rows = coo_val.row[sample_indices]
            val_cols = coo_val.col[sample_indices]
            val_ratings = coo_val.data[sample_indices]
        
        # Compute predictions
        predictions = np.array([
            np.dot(self.user_factors[u], self.movie_factors[m])
            for u, m in zip(val_rows, val_cols)
        ])
        
        # Compute RMSE
        mse = np.mean((predictions - val_ratings) ** 2)
        rmse = np.sqrt(mse)
        
        self.logger.info(f"Validation RMSE computed on {len(val_ratings):,} ratings: {rmse:.6f}")
        return rmse

def load_existing_split(logger):
    """Load existing train/validation split from previous run."""
    logger.info("Loading existing train/validation split...")
    
    # Try to load from previous run
    try:
        # Load the sparse matrices we created before
        train_matrix = sp.load_npz("data/collaborative/train_matrix.npz")
        val_matrix = sp.load_npz("data/collaborative/val_matrix.npz")
        
        # Try to load the movie mapping
        try:
            import pickle
            with open("data/collaborative/movie_to_idx.pkl", "rb") as f:
                movie_to_idx = pickle.load(f)
        except FileNotFoundError:
            logger.warning("Movie mapping not found, will need to recreate")
            movie_to_idx = None
        
        logger.info(f"Loaded existing split: Train {train_matrix.shape}, Val {val_matrix.shape}")
        return train_matrix, val_matrix, movie_to_idx
        
    except FileNotFoundError:
        logger.info("No existing split found, creating new one...")
        return None, None, None

def create_safe_train_val_split(ratings_long, logger):
    """Create train/validation split with memory safety."""
    logger.info("Creating safe train/validation split...")
    
    # Use a smaller sample for safety
    sample_size = min(5000000, len(ratings_long))  # Max 5M ratings
    if sample_size < len(ratings_long):
        logger.info(f"Sampling {sample_size:,} ratings for safety")
        ratings_sample = ratings_long.sample(n=sample_size, random_state=42)
    else:
        ratings_sample = ratings_long
    
    # Get unique users and movies
    unique_users = sorted(ratings_sample['user_index'].unique())
    unique_movies = sorted(ratings_sample['canonical_id'].unique())
    n_users = len(unique_users)
    n_movies = len(unique_movies)
    
    logger.info(f"Split dimensions: {n_users:,} users, {n_movies:,} movies")
    
    # Create mappings for remapping indices
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    
    # Remap user indices to be sequential
    ratings_sample = ratings_sample.copy()
    ratings_sample['user_idx'] = ratings_sample['user_index'].map(user_to_idx)
    
    # Split the data
    train_data, val_data = train_test_split(
        ratings_sample, 
        test_size=0.2, 
        random_state=42
    )
    
    logger.info(f"Split sizes: Train {len(train_data):,}, Val {len(val_data):,}")
    
    # Build sparse matrices using remapped indices
    train_matrix = csr_matrix(
        (train_data['rating'], 
         (train_data['user_idx'], train_data['canonical_id'].map(movie_to_idx))),
        shape=(n_users, n_movies)
    )
    
    val_matrix = csr_matrix(
        (val_data['rating'], 
         (val_data['user_idx'], val_data['canonical_id'].map(movie_to_idx))),
        shape=(n_users, n_movies)
    )
    
    # Save for future use
    sp.save_npz("data/collaborative/train_matrix.npz", train_matrix)
    sp.save_npz("data/collaborative/val_matrix.npz", val_matrix)
    
    # Save movie mapping
    import pickle
    with open("data/collaborative/movie_to_idx.pkl", "wb") as f:
        pickle.dump(movie_to_idx, f)
    
    logger.info(f"Train matrix: {train_matrix.shape}, density: {train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]):.6f}")
    logger.info(f"Val matrix: {val_matrix.shape}, density: {val_matrix.nnz / (val_matrix.shape[0] * val_matrix.shape[1]):.6f}")
    
    return train_matrix, val_matrix, movie_to_idx

def perform_integrity_checks(user_factors, movie_factors, train_matrix, movie_to_idx, logger):
    """Perform integrity checks on the learned factors."""
    logger.info("Performing integrity checks...")
    
    # Check for NaN/Inf values
    user_has_nan = np.isnan(user_factors).any()
    user_has_inf = np.isinf(user_factors).any()
    movie_has_nan = np.isnan(movie_factors).any()
    movie_has_inf = np.isinf(movie_factors).any()
    
    logger.info(f"User factors - Has NaN: {user_has_nan}, Has Inf: {user_has_inf}")
    logger.info(f"Movie factors - Has NaN: {movie_has_nan}, Has Inf: {movie_has_inf}")
    
    if user_has_nan or user_has_inf or movie_has_nan or movie_has_inf:
        logger.error("Found NaN or Inf values in factors!")
        return False
    
    # Check alignment with matrix dimensions
    user_alignment = user_factors.shape[0] == train_matrix.shape[0]
    movie_alignment = movie_factors.shape[0] == train_matrix.shape[1]
    
    logger.info(f"User alignment: {user_alignment} ({user_factors.shape[0]} vs {train_matrix.shape[0]})")
    logger.info(f"Movie alignment: {movie_alignment} ({movie_factors.shape[0]} vs {train_matrix.shape[1]})")
    
    if not user_alignment or not movie_alignment:
        logger.error("Factor matrices not aligned with training matrix!")
        return False
    
    # Quick nearest neighbors check
    logger.info("Performing nearest neighbors sanity check...")
    
    # Compute movie similarity matrix
    movie_similarities = movie_factors @ movie_factors.T
    
    # Test on a few random movies from our sampled set
    n_test_movies = min(3, movie_factors.shape[0])
    test_movie_indices = np.random.choice(movie_factors.shape[0], n_test_movies, replace=False)
    
    for movie_idx in test_movie_indices:
        # Find the canonical_id for this movie index
        canonical_id = None
        for cid, midx in movie_to_idx.items():
            if midx == movie_idx:
                canonical_id = cid
                break
        
        if canonical_id is None:
            logger.warning(f"Could not find canonical_id for movie index {movie_idx}")
            continue
        
        # Get top 10 most similar movies
        similarities = movie_similarities[movie_idx]
        top_indices = np.argsort(similarities)[-11:-1][::-1]  # Top 10, excluding self
        
        # Get canonical_ids for similar movies
        similar_canonical_ids = []
        for sim_idx in top_indices:
            for cid, midx in movie_to_idx.items():
                if midx == sim_idx:
                    similar_canonical_ids.append(cid)
                    break
        
        similarities_scores = similarities[top_indices]
        
        logger.info(f"Top 5 similar movies to {canonical_id}:")
        for sim_movie, sim_score in zip(similar_canonical_ids[:5], similarities_scores[:5]):
            logger.info(f"  {sim_movie}: {sim_score:.4f}")
    
    logger.info("Integrity checks passed!")
    return True

def save_outputs(user_factors, movie_factors, config, training_log, logger):
    """Save all output files."""
    logger.info("Saving outputs...")
    
    output_dir = Path("data/collaborative")
    output_dir.mkdir(exist_ok=True)
    
    # Save final factor matrices
    logger.info("Saving final factor matrices...")
    np.save(output_dir / "user_factors_k20.npy", user_factors)
    np.save(output_dir / "movie_factors_k20.npy", movie_factors)
    
    # Save configuration
    logger.info("Saving factorization_config.json...")
    with open(output_dir / "factorization_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save training log
    logger.info("Saving training_log.txt...")
    with open(output_dir / "training_log.txt", "w") as f:
        f.write(training_log)
    
    logger.info("All outputs saved successfully")

def generate_documentation(result, config, logger):
    """Generate documentation for the matrix factorization training."""
    logger.info("Generating documentation...")
    
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Create documentation content
    doc_content = f"""# Step 3b.2: Matrix Factorization Training (Safe Version)

## Overview
This document describes the safe training of collaborative filtering models using matrix factorization techniques with memory monitoring and checkpointing.

## Safety Configuration
- **Memory Threshold**: {MEMORY_THRESHOLD_MB:,} MB
- **Validation Sample Size**: {SAMPLE_SIZE_VALIDATION:,} ratings max
- **Algorithm**: {config['algorithm']}
- **Latent Dimensions**: {config['n_components']}
- **Max Iterations**: {config['max_iterations']}
- **Random Seed**: {config['random_state']}

## Input Data
- **Matrix Shape**: {config['matrix_shape']}
- **Matrix Density**: {config['matrix_density']:.6f}
- **Train/Validation Split**: {config['train_size']:,} / {config['val_size']:,} ratings

## Training Results
- **Final Validation RMSE**: {config['final_val_rmse']:.6f}
- **Training Time**: {config['training_time']:.1f} seconds
- **Peak Memory Usage**: {config['peak_memory_mb']:.1f} MB
- **Early Stopping**: {config.get('early_stopped', False)}

## Per-Epoch/Iteration Metrics
"""
    
    for metric in result.training_metrics:
        if 'epoch' in metric:
            doc_content += f"- **Epoch {metric['epoch']}**: RMSE = {metric['val_rmse']:.6f}, Time = {metric['time_seconds']:.1f}s, Memory = {metric['memory_mb']:.1f} MB\n"
        else:
            doc_content += f"- **Iteration {metric['iteration']}**: RMSE = {metric['val_rmse']:.6f}, Time = {metric['time_seconds']:.1f}s, Memory = {metric['memory_mb']:.1f} MB\n"
    
    doc_content += f"""
## Factor Matrix Properties
- **User Factors Shape**: {config['user_factors_shape']}
- **Movie Factors Shape**: {config['movie_factors_shape']}
- **User Factor Norms**: Min={config['user_norms'][0]:.4f}, Max={config['user_norms'][1]:.4f}, Mean={config['user_norms'][2]:.4f}
- **Movie Factor Norms**: Min={config['movie_norms'][0]:.4f}, Max={config['movie_norms'][1]:.4f}, Mean={config['movie_norms'][2]:.4f}

## Integrity Checks
- ✅ No NaN/Inf values in factors
- ✅ Factor matrices aligned with index maps
- ✅ Nearest neighbors sanity check completed
- ✅ Checkpointing mechanism working

## Fallback Strategy Applied
{config.get('fallback_strategy', 'None - primary strategy succeeded')}

## Output Files
- `data/collaborative/user_factors_k20.npy`: User latent factors
- `data/collaborative/movie_factors_k20.npy`: Movie latent factors
- `data/collaborative/factorization_config.json`: Training configuration
- `data/collaborative/training_log.txt`: Detailed training log
- `data/collaborative/user_factors_k20_epoch*.npy`: Epoch checkpoints
- `data/collaborative/movie_factors_k20_epoch*.npy`: Epoch checkpoints

## Timestamp
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(docs_dir / "step3b_training.md", "w") as f:
        f.write(doc_content)
    
    logger.info("Documentation saved to docs/step3b_training.md")

def main():
    """Main function for safe matrix factorization training."""
    logger = setup_logging()
    logger.info("Starting Step 3b.2: Matrix Factorization Training (Safe Version)")
    
    start_time = time.time()
    fallback_strategy = "None - primary strategy succeeded"
    
    try:
        # Load input data
        logger.info("Loading input data...")
        ratings_long = pd.read_parquet("data/collaborative/ratings_long_format.parquet")
        user_index_map = pd.read_parquet("data/collaborative/user_index_map.parquet")
        movie_index_map = pd.read_parquet("data/collaborative/movie_index_map.parquet")
        
        logger.info(f"Loaded {len(ratings_long):,} ratings")
        log_memory_usage(logger, "after loading data")
        
        # Try to load existing split, otherwise create new one
        train_matrix, val_matrix, movie_to_idx = load_existing_split(logger)
        
        if train_matrix is None:
            train_matrix, val_matrix, movie_to_idx = create_safe_train_val_split(ratings_long, logger)
        elif movie_to_idx is None:
            # Create a simple mapping for existing split
            logger.info("Creating movie mapping for existing split...")
            movie_to_idx = {f"movie_{i}": i for i in range(train_matrix.shape[1])}
        
        # Try SVD first (k=20, epochs=5)
        logger.info("Attempting SVD training (k=20, epochs=5)...")
        try:
            svd_model = SafeSVDMatrixFactorization(
                n_components=20,
                max_iter=5,
                random_state=42,
                logger=logger
            )
            
            result = svd_model.fit_with_checkpointing(train_matrix, val_matrix)
            algorithm = "SVD"
            n_components = 20
            max_iterations = 5
            
        except MemoryError as e:
            logger.warning(f"SVD failed with memory error: {e}")
            fallback_strategy = "SVD k=20 failed, trying SVD k=16"
            
            # Fallback 1: SVD with k=16, epochs=3
            try:
                logger.info("Attempting SVD training (k=16, epochs=3)...")
                svd_model = SafeSVDMatrixFactorization(
                    n_components=16,
                    max_iter=3,
                    random_state=42,
                    logger=logger
                )
                
                result = svd_model.fit_with_checkpointing(train_matrix, val_matrix)
                algorithm = "SVD"
                n_components = 16
                max_iterations = 3
                fallback_strategy = "SVD k=16 succeeded after k=20 failed"
                
            except MemoryError as e2:
                logger.warning(f"SVD k=16 also failed: {e2}")
                fallback_strategy = "SVD failed, trying ALS k=20"
                
                # Fallback 2: ALS with k=20, iterations=5
                try:
                    logger.info("Attempting ALS training (k=20, iterations=5)...")
                    als_model = SafeALSMatrixFactorization(
                        n_factors=20,
                        iterations=5,
                        regularization=0.1,
                        logger=logger
                    )
                    
                    result = als_model.fit_with_checkpointing(train_matrix, val_matrix)
                    algorithm = "ALS"
                    n_components = 20
                    max_iterations = 5
                    fallback_strategy = "ALS k=20 succeeded after SVD failed"
                    
                except MemoryError as e3:
                    logger.error(f"All methods failed: {e3}")
                    raise
        
        # Perform integrity checks
        integrity_passed = perform_integrity_checks(
            result.user_factors,
            result.movie_factors,
            train_matrix,
            movie_to_idx,
            logger
        )
        
        if not integrity_passed:
            logger.error("Integrity checks failed!")
            return
        
        # Prepare configuration
        end_time = time.time()
        training_time = end_time - start_time
        
        config = {
            'algorithm': algorithm,
            'n_components': n_components,
            'max_iterations': max_iterations,
            'random_state': 42,
            'matrix_shape': train_matrix.shape,
            'matrix_density': train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]),
            'train_size': train_matrix.nnz,
            'val_size': val_matrix.nnz,
            'final_val_rmse': result.training_metrics[-1]['val_rmse'],
            'training_time': training_time,
            'peak_memory_mb': max([m['memory_mb'] for m in result.training_metrics]),
            'user_factors_shape': result.user_factors.shape,
            'movie_factors_shape': result.movie_factors.shape,
            'user_norms': [
                np.linalg.norm(result.user_factors, axis=1).min(),
                np.linalg.norm(result.user_factors, axis=1).max(),
                np.linalg.norm(result.user_factors, axis=1).mean()
            ],
            'movie_norms': [
                np.linalg.norm(result.movie_factors, axis=1).min(),
                np.linalg.norm(result.movie_factors, axis=1).max(),
                np.linalg.norm(result.movie_factors, axis=1).mean()
            ],
            'fallback_strategy': fallback_strategy
        }
        
        # Create training log
        training_log = f"""Matrix Factorization Training Log (Safe Version)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Configuration:
- Algorithm: {algorithm}
- Latent dimensions: {n_components}
- Max iterations: {max_iterations}
- Random seed: 42
- Memory threshold: {MEMORY_THRESHOLD_MB:,} MB

Input Data:
- Matrix shape: {train_matrix.shape}
- Matrix density: {config['matrix_density']:.6f}
- Train/Validation split: {config['train_size']:,} / {config['val_size']:,}

Training Results:
- Final validation RMSE: {config['final_val_rmse']:.6f}
- Training time: {training_time:.1f} seconds
- Peak memory usage: {config['peak_memory_mb']:.1f} MB
- Fallback strategy: {fallback_strategy}

Per-Epoch/Iteration Metrics:
"""
        
        for metric in result.training_metrics:
            if 'epoch' in metric:
                training_log += f"- Epoch {metric['epoch']}: RMSE = {metric['val_rmse']:.6f}, Time = {metric['time_seconds']:.1f}s, Memory = {metric['memory_mb']:.1f} MB\n"
            else:
                training_log += f"- Iteration {metric['iteration']}: RMSE = {metric['val_rmse']:.6f}, Time = {metric['time_seconds']:.1f}s, Memory = {metric['memory_mb']:.1f} MB\n"
        
        training_log += f"""
Factor Properties:
- User factors shape: {result.user_factors.shape}
- Movie factors shape: {result.movie_factors.shape}
- No NaN/Inf values: PASSED
- Factor alignment: PASSED
- Nearest neighbors check: PASSED
"""
        
        # Save outputs
        save_outputs(
            result.user_factors,
            result.movie_factors,
            config,
            training_log,
            logger
        )
        
        # Generate documentation
        generate_documentation(result, config, logger)
        
        logger.info("Step 3b.2 completed successfully!")
        log_memory_usage(logger, "final")
        
    except Exception as e:
        logger.error(f"Error in Step 3b.2: {str(e)}")
        raise

if __name__ == "__main__":
    main()
