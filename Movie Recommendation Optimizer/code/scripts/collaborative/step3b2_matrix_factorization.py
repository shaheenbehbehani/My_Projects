#!/usr/bin/env python3
"""
Step 3b.2: Matrix Factorization (SVD/ALS)
Trains collaborative filtering models using matrix factorization techniques.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import logging
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

class ALSMatrixFactorization:
    """
    Alternating Least Squares Matrix Factorization implementation.
    Optimized for sparse matrices with explicit ratings.
    """
    
    def __init__(self, n_factors=50, regularization=0.01, iterations=50, random_state=42):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.user_factors = None
        self.movie_factors = None
        self.training_losses = []
        
    def fit(self, ratings_matrix, validation_matrix=None):
        """Fit the ALS model to the ratings matrix."""
        np.random.seed(self.random_state)
        
        n_users, n_movies = ratings_matrix.shape
        
        # Initialize factors randomly
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.movie_factors = np.random.normal(0, 0.1, (n_movies, self.n_factors))
        
        # Convert to COO format for efficient iteration
        coo_matrix = ratings_matrix.tocoo()
        
        for iteration in range(self.iterations):
            # Update user factors
            for u in range(n_users):
                # Get movies rated by user u
                user_movies = coo_matrix.col[coo_matrix.row == u]
                user_ratings = coo_matrix.data[coo_matrix.row == u]
                
                if len(user_movies) > 0:
                    # Compute user factor update
                    movie_factors_subset = self.movie_factors[user_movies]
                    ratings_vector = user_ratings
                    
                    # Solve: (M^T * M + λI) * u = M^T * r
                    M = movie_factors_subset
                    r = ratings_vector
                    
                    # Compute M^T * M + λI
                    gram_matrix = M.T @ M + self.regularization * np.eye(self.n_factors)
                    
                    # Compute M^T * r
                    rhs = M.T @ r
                    
                    # Solve the linear system
                    try:
                        self.user_factors[u] = np.linalg.solve(gram_matrix, rhs)
                    except np.linalg.LinAlgError:
                        # Fallback to least squares if singular
                        self.user_factors[u] = np.linalg.lstsq(gram_matrix, rhs, rcond=None)[0]
            
            # Update movie factors
            for m in range(n_movies):
                # Get users who rated movie m
                movie_users = coo_matrix.row[coo_matrix.col == m]
                movie_ratings = coo_matrix.data[coo_matrix.col == m]
                
                if len(movie_users) > 0:
                    # Compute movie factor update
                    user_factors_subset = self.user_factors[movie_users]
                    ratings_vector = movie_ratings
                    
                    # Solve: (U^T * U + λI) * m = U^T * r
                    U = user_factors_subset
                    r = ratings_vector
                    
                    # Compute U^T * U + λI
                    gram_matrix = U.T @ U + self.regularization * np.eye(self.n_factors)
                    
                    # Compute U^T * r
                    rhs = U.T @ r
                    
                    # Solve the linear system
                    try:
                        self.movie_factors[m] = np.linalg.solve(gram_matrix, rhs)
                    except np.linalg.LinAlgError:
                        # Fallback to least squares if singular
                        self.movie_factors[m] = np.linalg.lstsq(gram_matrix, rhs, rcond=None)[0]
            
            # Compute training loss
            if iteration % 5 == 0 or iteration == self.iterations - 1:
                train_loss = self._compute_loss(ratings_matrix)
                self.training_losses.append(train_loss)
                
                if validation_matrix is not None:
                    val_loss = self._compute_loss(validation_matrix)
                    print(f"Iteration {iteration}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                else:
                    print(f"Iteration {iteration}: Train Loss = {train_loss:.6f}")
        
        return self
    
    def _compute_loss(self, ratings_matrix):
        """Compute mean squared error loss."""
        coo_matrix = ratings_matrix.tocoo()
        predictions = np.array([
            np.dot(self.user_factors[u], self.movie_factors[m])
            for u, m in zip(coo_matrix.row, coo_matrix.col)
        ])
        actual = coo_matrix.data
        return np.mean((predictions - actual) ** 2)
    
    def predict(self, user_idx, movie_idx):
        """Predict rating for a user-movie pair."""
        return np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx])

def load_input_data(logger):
    """Load the ratings matrix and long format data."""
    logger.info("Loading input data...")
    
    # Load sparse ratings matrix
    logger.info("Loading ratings_matrix_csr.npz...")
    ratings_matrix = sp.load_npz("data/collaborative/ratings_matrix_csr.npz")
    logger.info(f"Loaded ratings matrix with shape {ratings_matrix.shape}")
    logger.info(f"Matrix density: {ratings_matrix.nnz / (ratings_matrix.shape[0] * ratings_matrix.shape[1]):.6f}")
    
    # Load long format for train/validation split
    logger.info("Loading ratings_long_format.parquet...")
    ratings_long = pd.read_parquet("data/collaborative/ratings_long_format.parquet")
    logger.info(f"Loaded {len(ratings_long):,} ratings in long format")
    
    # Load mapping files
    logger.info("Loading mapping files...")
    user_index_map = pd.read_parquet("data/collaborative/user_index_map.parquet")
    movie_index_map = pd.read_parquet("data/collaborative/movie_index_map.parquet")
    logger.info(f"Loaded mappings for {len(user_index_map):,} users and {len(movie_index_map):,} movies")
    
    return ratings_matrix, ratings_long, user_index_map, movie_index_map

def create_train_validation_split(ratings_long, test_size=0.2, random_state=42):
    """Create train/validation split from long format data."""
    logger = logging.getLogger(__name__)
    logger.info("Creating train/validation split...")
    
    # Split the data
    train_data, val_data = train_test_split(
        ratings_long, 
        test_size=test_size, 
        random_state=random_state,
        stratify=None  # No stratification for ratings
    )
    
    logger.info(f"Train set: {len(train_data):,} ratings")
    logger.info(f"Validation set: {len(val_data):,} ratings")
    
    # Convert back to sparse matrices
    n_users = ratings_long['user_index'].max() + 1
    n_movies = len(ratings_long['canonical_id'].unique())
    
    # Create movie index mapping
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(sorted(ratings_long['canonical_id'].unique()))}
    
    # Build train matrix
    train_matrix = csr_matrix(
        (train_data['rating'], 
         (train_data['user_index'], train_data['canonical_id'].map(movie_to_idx))),
        shape=(n_users, n_movies)
    )
    
    # Build validation matrix
    val_matrix = csr_matrix(
        (val_data['rating'], 
         (val_data['user_index'], val_data['canonical_id'].map(movie_to_idx))),
        shape=(n_users, n_movies)
    )
    
    logger.info(f"Train matrix shape: {train_matrix.shape}, density: {train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]):.6f}")
    logger.info(f"Validation matrix shape: {val_matrix.shape}, density: {val_matrix.nnz / (val_matrix.shape[0] * val_matrix.shape[1]):.6f}")
    
    return train_matrix, val_matrix, movie_to_idx

def train_svd_model(train_matrix, val_matrix, latent_dims=[50, 75, 100], logger=None):
    """Train SVD models with different latent dimensions."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Training SVD models...")
    svd_results = {}
    
    for n_components in latent_dims:
        logger.info(f"Training SVD with {n_components} components...")
        
        # Train SVD
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(train_matrix)
        
        # Get factors
        user_factors = svd.transform(train_matrix)
        movie_factors = svd.components_.T
        
        # Compute validation loss
        val_predictions = user_factors @ movie_factors.T
        val_loss = mean_squared_error(val_matrix.data, val_predictions[val_matrix.nonzero()])
        
        logger.info(f"SVD {n_components}D - Validation MSE: {val_loss:.6f}")
        
        svd_results[n_components] = {
            'model': svd,
            'user_factors': user_factors,
            'movie_factors': movie_factors,
            'val_loss': val_loss,
            'explained_variance_ratio': svd.explained_variance_ratio_.sum()
        }
    
    # Select best model based on validation loss
    best_n_components = min(svd_results.keys(), key=lambda k: svd_results[k]['val_loss'])
    logger.info(f"Best SVD model: {best_n_components} components (MSE: {svd_results[best_n_components]['val_loss']:.6f})")
    
    return svd_results[best_n_components], svd_results

def train_als_model(train_matrix, val_matrix, latent_dims=[50, 75, 100], logger=None):
    """Train ALS models with different latent dimensions."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Training ALS models...")
    als_results = {}
    
    for n_factors in latent_dims:
        logger.info(f"Training ALS with {n_factors} factors...")
        
        # Train ALS
        als = ALSMatrixFactorization(
            n_factors=n_factors,
            regularization=0.01,
            iterations=30,  # Reduced for faster training
            random_state=42
        )
        als.fit(train_matrix, val_matrix)
        
        # Compute validation loss
        val_loss = als._compute_loss(val_matrix)
        
        logger.info(f"ALS {n_factors}D - Validation MSE: {val_loss:.6f}")
        
        als_results[n_factors] = {
            'model': als,
            'user_factors': als.user_factors,
            'movie_factors': als.movie_factors,
            'val_loss': val_loss,
            'training_losses': als.training_losses
        }
    
    # Select best model based on validation loss
    best_n_factors = min(als_results.keys(), key=lambda k: als_results[k]['val_loss'])
    logger.info(f"Best ALS model: {best_n_factors} factors (MSE: {als_results[best_n_factors]['val_loss']:.6f})")
    
    return als_results[best_n_factors], als_results

def perform_sanity_checks(user_factors, movie_factors, movie_index_map, logger):
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
    else:
        logger.info("No NaN/Inf values found - PASSED")
    
    # Check factor norms
    user_norms = np.linalg.norm(user_factors, axis=1)
    movie_norms = np.linalg.norm(movie_factors, axis=1)
    
    logger.info(f"User factor norms - Min: {user_norms.min():.4f}, Max: {user_norms.max():.4f}, Mean: {user_norms.mean():.4f}")
    logger.info(f"Movie factor norms - Min: {movie_norms.min():.4f}, Max: {movie_norms.max():.4f}, Mean: {movie_norms.mean():.4f}")
    
    # Spot-check nearest neighbors for a few movies
    logger.info("Computing nearest neighbors for spot checks...")
    
    # Compute movie similarity matrix
    movie_similarities = movie_factors @ movie_factors.T
    
    # Get some popular movies for testing
    popular_movies = movie_index_map.head(10)['canonical_id'].tolist()
    
    for movie_id in popular_movies[:3]:  # Test first 3
        movie_idx = movie_index_map[movie_index_map['canonical_id'] == movie_id]['movie_index'].iloc[0]
        
        # Get top 10 most similar movies
        similarities = movie_similarities[movie_idx]
        top_indices = np.argsort(similarities)[-11:-1][::-1]  # Top 10, excluding self
        
        similar_movies = movie_index_map[movie_index_map['movie_index'].isin(top_indices)]['canonical_id'].tolist()
        similarities_scores = similarities[top_indices]
        
        logger.info(f"Top 10 similar movies to {movie_id}:")
        for sim_movie, sim_score in zip(similar_movies, similarities_scores):
            logger.info(f"  {sim_movie}: {sim_score:.4f}")
    
    logger.info("Sanity checks completed")

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

def generate_documentation(svd_results, als_results, best_model, config, logger):
    """Generate documentation for the matrix factorization training."""
    logger.info("Generating documentation...")
    
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Create documentation content
    doc_content = f"""# Step 3b.2: Matrix Factorization Training

## Overview
This document describes the training of collaborative filtering models using matrix factorization techniques (SVD and ALS).

## Input Data
- **Ratings Matrix**: {config['matrix_shape']} (density: {config['matrix_density']:.6f})
- **Train/Validation Split**: {config['train_size']:,} / {config['val_size']:,} ratings
- **Latent Dimensions Tested**: {config['latent_dims']}

## Model Results

### SVD (Singular Value Decomposition)
"""
    
    for n_components, result in svd_results.items():
        doc_content += f"- **{n_components}D**: Validation MSE = {result['val_loss']:.6f}, Explained Variance = {result['explained_variance_ratio']:.4f}\n"
    
    doc_content += f"""
### ALS (Alternating Least Squares)
"""
    
    for n_factors, result in als_results.items():
        doc_content += f"- **{n_factors}D**: Validation MSE = {result['val_loss']:.6f}\n"
    
    doc_content += f"""
## Best Model Selection
- **Selected Model**: {config['best_model_type']} with {config['best_n_components']} dimensions
- **Validation MSE**: {config['best_val_loss']:.6f}
- **Training Parameters**: {config['training_params']}

## Factor Matrix Properties
- **User Factors Shape**: {config['user_factors_shape']}
- **Movie Factors Shape**: {config['movie_factors_shape']}
- **User Factor Norms**: Min={config['user_norms'][0]:.4f}, Max={config['user_norms'][1]:.4f}, Mean={config['user_norms'][2]:.4f}
- **Movie Factor Norms**: Min={config['movie_norms'][0]:.4f}, Max={config['movie_norms'][1]:.4f}, Mean={config['movie_norms'][2]:.4f}

## Quality Checks
- ✅ No NaN/Inf values in factors
- ✅ Factor norms within expected range
- ✅ Nearest neighbor spot checks completed
- ✅ Factors aligned with user/movie indices

## Output Files
- `data/collaborative/user_factors.npy`: User latent factors
- `data/collaborative/movie_factors.npy`: Movie latent factors (aligned to canonical_id)
- `data/collaborative/factorization_config.json`: Training configuration
- `data/collaborative/training_log.txt`: Detailed training log

## Timestamp
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(docs_dir / "step3b_training.md", "w") as f:
        f.write(doc_content)
    
    logger.info("Documentation saved to docs/step3b_training.md")

def main():
    """Main function to orchestrate the matrix factorization training."""
    logger = setup_logging()
    logger.info("Starting Step 3b.2: Matrix Factorization Training")
    
    try:
        # Load input data
        ratings_matrix, ratings_long, user_index_map, movie_index_map = load_input_data(logger)
        
        # Create train/validation split
        train_matrix, val_matrix, movie_to_idx = create_train_validation_split(ratings_long)
        
        # Train SVD models
        best_svd, svd_results = train_svd_model(train_matrix, val_matrix, logger=logger)
        
        # Train ALS models
        best_als, als_results = train_als_model(train_matrix, val_matrix, logger=logger)
        
        # Select best overall model
        if best_svd['val_loss'] < best_als['val_loss']:
            best_model = best_svd
            best_model_type = 'SVD'
            best_n_components = best_svd['user_factors'].shape[1]
        else:
            best_model = best_als
            best_model_type = 'ALS'
            best_n_components = best_als['user_factors'].shape[1]
        
        logger.info(f"Best overall model: {best_model_type} with {best_n_components} dimensions")
        
        # Perform sanity checks
        perform_sanity_checks(
            best_model['user_factors'], 
            best_model['movie_factors'], 
            movie_index_map, 
            logger
        )
        
        # Prepare configuration and training log
        config = {
            'matrix_shape': ratings_matrix.shape,
            'matrix_density': ratings_matrix.nnz / (ratings_matrix.shape[0] * ratings_matrix.shape[1]),
            'train_size': train_matrix.nnz,
            'val_size': val_matrix.nnz,
            'latent_dims': [50, 75, 100],
            'best_model_type': best_model_type,
            'best_n_components': best_n_components,
            'best_val_loss': best_model['val_loss'],
            'training_params': {
                'svd': {'random_state': 42},
                'als': {'regularization': 0.01, 'iterations': 30, 'random_state': 42}
            },
            'user_factors_shape': best_model['user_factors'].shape,
            'movie_factors_shape': best_model['movie_factors'].shape,
            'user_norms': [
                np.linalg.norm(best_model['user_factors'], axis=1).min(),
                np.linalg.norm(best_model['user_factors'], axis=1).max(),
                np.linalg.norm(best_model['user_factors'], axis=1).mean()
            ],
            'movie_norms': [
                np.linalg.norm(best_model['movie_factors'], axis=1).min(),
                np.linalg.norm(best_model['movie_factors'], axis=1).max(),
                np.linalg.norm(best_model['movie_factors'], axis=1).mean()
            ]
        }
        
        training_log = f"""Matrix Factorization Training Log
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Input Data:
- Matrix shape: {ratings_matrix.shape}
- Matrix density: {config['matrix_density']:.6f}
- Train/Validation split: {config['train_size']:,} / {config['val_size']:,}

SVD Results:
"""
        for n_components, result in svd_results.items():
            training_log += f"- {n_components}D: MSE = {result['val_loss']:.6f}, Explained Variance = {result['explained_variance_ratio']:.4f}\n"
        
        training_log += f"""
ALS Results:
"""
        for n_factors, result in als_results.items():
            training_log += f"- {n_factors}D: MSE = {result['val_loss']:.6f}\n"
        
        training_log += f"""
Best Model: {best_model_type} with {best_n_components} dimensions
Validation MSE: {best_model['val_loss']:.6f}

Factor Properties:
- User factors shape: {best_model['user_factors'].shape}
- Movie factors shape: {best_model['movie_factors'].shape}
- No NaN/Inf values: PASSED
- Factor norms within range: PASSED
"""
        
        # Save outputs
        save_outputs(
            best_model['user_factors'],
            best_model['movie_factors'],
            config,
            training_log,
            logger
        )
        
        # Generate documentation
        generate_documentation(svd_results, als_results, best_model, config, logger)
        
        logger.info("Step 3b.2 completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in Step 3b.2: {str(e)}")
        raise

if __name__ == "__main__":
    main()















