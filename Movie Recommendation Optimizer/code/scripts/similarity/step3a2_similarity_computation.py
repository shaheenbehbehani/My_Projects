#!/usr/bin/env python3
"""
Step 3a.2: Similarity Computation (Cosine + kNN)
Compute cosine nearest neighbors for each movie using the composite embedding.
Produces kNN indices, scores, and a long-format neighbor table.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/step3a_similarity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimilarityComputer:
    """Computes cosine similarity and kNN for movie embeddings."""
    
    def __init__(self, k=50, batch_size=2000):
        self.k = k
        self.batch_size = batch_size
        self.output_dir = Path("data/similarity")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checks").mkdir(exist_ok=True)
        
        # Expected dimensions from 3a.1
        self.expected_rows = 87601
        self.expected_dim = 384
        
        # Performance tracking
        self.timings = {}
        self.memory_peaks = {}
        
    def load_embedding(self):
        """Load the composite embedding with memory mapping."""
        start_time = time.time()
        logger.info("Loading composite embedding...")
        
        embedding_path = "data/features/composite/movies_embedding_v1.npy"
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        # Load with memory mapping for low RAM overhead
        embedding = np.load(embedding_path, mmap_mode='r')
        logger.info(f"Loaded embedding: {embedding.shape}, dtype: {embedding.dtype}")
        
        # Validate dimensions
        if embedding.shape != (self.expected_rows, self.expected_dim):
            raise ValueError(f"Expected shape ({self.expected_rows}, {self.expected_dim}), got {embedding.shape}")
        
        # Validate L2 normalization
        norms = np.linalg.norm(embedding, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"Embedding not L2-normalized. Norm range: {norms.min():.6f} - {norms.max():.6f}")
        
        logger.info(f"Embedding validation passed. Norm range: {norms.min():.6f} - {norms.max():.6f}")
        
        self.timings['load'] = time.time() - start_time
        return embedding
    
    def load_metadata(self):
        """Load metadata for sanity checks."""
        logger.info("Loading metadata...")
        
        metadata_path = "data/features/composite/movies_features_v1.parquet"
        metadata = pd.read_parquet(metadata_path)
        logger.info(f"Loaded metadata: {metadata.shape}")
        
        # Validate canonical IDs
        if len(metadata) != self.expected_rows:
            raise ValueError(f"Metadata has {len(metadata)} rows, expected {self.expected_rows}")
        
        return metadata
    
    def compute_similarities_batched(self, embedding):
        """Compute cosine similarities in batches to manage memory."""
        start_time = time.time()
        logger.info(f"Computing similarities in batches of {self.batch_size}...")
        
        n_movies = embedding.shape[0]
        n_batches = (n_movies + self.batch_size - 1) // self.batch_size
        
        # Initialize output arrays
        knn_indices = np.zeros((n_movies, self.k), dtype=np.int32)
        knn_scores = np.zeros((n_movies, self.k), dtype=np.float32)
        
        batch_times = []
        memory_usage = []
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, n_movies)
            batch_size_actual = batch_end - batch_start
            
            logger.info(f"Processing batch {batch_idx + 1}/{n_batches} (movies {batch_start}-{batch_end-1})")
            
            batch_start_time = time.time()
            
            # Get batch of query vectors
            query_vectors = embedding[batch_start:batch_end]
            
            # Compute dot product with all vectors (cosine similarity since vectors are L2-normalized)
            # This gives us similarity scores for each query vector against all movies
            similarities = query_vectors @ embedding.T  # Shape: (batch_size, n_movies)
            
            # For each query vector, find top-k neighbors (excluding self)
            for i in range(batch_size_actual):
                movie_idx = batch_start + i
                
                # Get similarities for this movie
                movie_similarities = similarities[i]
                
                # Set self-similarity to -1 to exclude it from top-k
                movie_similarities[movie_idx] = -1
                
                # Find top-k neighbors
                top_k_indices = np.argpartition(movie_similarities, -self.k)[-self.k:]
                top_k_scores = movie_similarities[top_k_indices]
                
                # Sort by score (descending)
                sort_order = np.argsort(top_k_scores)[::-1]
                top_k_indices = top_k_indices[sort_order]
                top_k_scores = top_k_scores[sort_order]
                
                # Store results
                knn_indices[movie_idx] = top_k_indices
                knn_scores[movie_idx] = top_k_scores
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Log memory usage (approximate)
            batch_memory = similarities.nbytes / (1024**3)  # GB
            memory_usage.append(batch_memory)
            
            logger.info(f"Batch {batch_idx + 1} completed in {batch_time:.2f}s, memory: {batch_memory:.2f}GB")
        
        # Performance statistics
        avg_batch_time = np.mean(batch_times)
        max_memory = np.max(memory_usage)
        
        logger.info(f"All batches completed. Avg batch time: {avg_batch_time:.2f}s, Max memory: {max_memory:.2f}GB")
        
        self.timings['compute'] = time.time() - start_time
        self.memory_peaks['max_batch_memory'] = max_memory
        self.memory_peaks['avg_batch_time'] = avg_batch_time
        
        return knn_indices, knn_scores
    
    def create_neighbor_table(self, knn_indices, knn_scores, metadata):
        """Create long-format neighbor table."""
        logger.info("Creating neighbor table...")
        
        rows = []
        for movie_idx in range(len(knn_indices)):
            movie_id = metadata.iloc[movie_idx]['canonical_id']
            
            for rank in range(self.k):
                neighbor_idx = knn_indices[movie_idx, rank]
                neighbor_id = metadata.iloc[neighbor_idx]['canonical_id']
                score = knn_scores[movie_idx, rank]
                
                rows.append({
                    'movie_id': movie_id,
                    'neighbor_id': neighbor_id,
                    'score': float(score),
                    'rank': rank + 1  # 1-based ranking
                })
        
        neighbor_df = pd.DataFrame(rows)
        logger.info(f"Created neighbor table: {neighbor_df.shape}")
        
        return neighbor_df
    
    def run_qa_checks(self, knn_indices, knn_scores, neighbor_df, embedding):
        """Run acceptance gate checks."""
        logger.info("Running QA checks...")
        
        checks = {}
        
        # Check 1: Shapes match expected dimensions
        expected_shape = (self.expected_rows, self.k)
        checks['shapes_match'] = (knn_indices.shape == expected_shape and knn_scores.shape == expected_shape)
        logger.info(f"Shape check: indices {knn_indices.shape}, scores {knn_scores.shape} -> {checks['shapes_match']}")
        
        # Check 2: No self-neighbors
        diagonal_indices = np.arange(self.expected_rows)
        self_neighbor_count = np.sum(knn_indices == diagonal_indices[:, np.newaxis])
        checks['no_self_neighbors'] = self_neighbor_count == 0
        logger.info(f"No self-neighbors check: {self_neighbor_count} self-neighbors found -> {checks['no_self_neighbors']}")
        
        # Check 3: Scores in [0, 1] range
        score_range_check = np.all((knn_scores >= 0) & (knn_scores <= 1))
        checks['score_range'] = score_range_check
        logger.info(f"Score range check: [{knn_scores.min():.6f}, {knn_scores.max():.6f}] -> {score_range_check}")
        
        # Check 4: Monotonicity (scores non-increasing by rank)
        monotonicity_violations = 0
        for i in range(len(knn_scores)):
            if not np.all(np.diff(knn_scores[i]) <= 0):
                monotonicity_violations += 1
        
        checks['monotonicity'] = monotonicity_violations == 0
        logger.info(f"Monotonicity check: {monotonicity_violations} violations -> {checks['monotonicity']}")
        
        # Check 5: Parquet row count matches expected
        expected_parquet_rows = self.expected_rows * self.k
        checks['parquet_row_count'] = len(neighbor_df) == expected_parquet_rows
        logger.info(f"Parquet row count: {len(neighbor_df)} == {expected_parquet_rows} -> {checks['parquet_row_count']}")
        
        # Check 6: Diagonal similarity ≈ 1.0 (sample test)
        sample_size = min(200, self.expected_rows)
        sample_indices = np.random.choice(self.expected_rows, sample_size, replace=False)
        sample_diagonal_sims = np.array([embedding[i] @ embedding[i] for i in sample_indices])
        diagonal_check = np.allclose(sample_diagonal_sims, 1.0, atol=1e-3)
        checks['diagonal_similarity'] = diagonal_check
        logger.info(f"Diagonal similarity check: range [{sample_diagonal_sims.min():.6f}, {sample_diagonal_sims.max():.6f}] -> {diagonal_check}")
        
        # Log summary
        all_passed = all(checks.values())
        logger.info(f"QA checks summary: {sum(checks.values())}/{len(checks)} passed")
        
        if not all_passed:
            failed_checks = [k for k, v in checks.items() if not v]
            logger.error(f"Failed checks: {failed_checks}")
            raise ValueError(f"QA checks failed: {failed_checks}")
        
        return checks
    
    def create_qa_artifacts(self, knn_indices, knn_scores, neighbor_df, embedding):
        """Create QA artifacts for inspection."""
        logger.info("Creating QA artifacts...")
        
        # 1. Sample of diagonal similarities
        sample_size = min(200, self.expected_rows)
        np.random.seed(42)  # Deterministic sampling
        sample_indices = np.random.choice(self.expected_rows, sample_size, replace=False)
        sample_diagonal_sims = np.array([embedding[i] @ embedding[i] for i in sample_indices])
        
        diag_sample_df = pd.DataFrame({
            'movie_index': sample_indices,
            'diagonal_similarity': sample_diagonal_sims
        })
        diag_sample_path = self.output_dir / "checks" / "sim_diag_sample.csv"
        diag_sample_df.to_csv(diag_sample_path, index=False)
        logger.info(f"Saved diagonal similarity sample to {diag_sample_path}")
        
        # 2. Monotonicity violations
        monotonicity_violations = []
        for i in range(len(knn_scores)):
            if not np.all(np.diff(knn_scores[i]) <= 0):
                violations = np.where(np.diff(knn_scores[i]) > 0)[0]
                for v in violations:
                    monotonicity_violations.append({
                        'movie_index': i,
                        'rank_1': v + 1,
                        'rank_2': v + 2,
                        'score_1': knn_scores[i, v],
                        'score_2': knn_scores[i, v + 1],
                        'violation': knn_scores[i, v + 1] - knn_scores[i, v]
                    })
        
        if monotonicity_violations:
            violations_df = pd.DataFrame(monotonicity_violations)
            violations_path = self.output_dir / "checks" / "monotonicity_violations.csv"
            violations_df.to_csv(violations_path, index=False)
            logger.info(f"Found {len(monotonicity_violations)} monotonicity violations, saved to {violations_path}")
        else:
            # Create empty file
            violations_path = self.output_dir / "checks" / "monotonicity_violations.csv"
            pd.DataFrame(columns=['movie_index', 'rank_1', 'rank_2', 'score_1', 'score_2', 'violation']).to_csv(violations_path, index=False)
            logger.info(f"No monotonicity violations found, created empty file at {violations_path}")
        
        # 3. Percentage of unique neighbors
        unique_neighbor_counts = []
        for i in range(len(knn_indices)):
            unique_count = len(np.unique(knn_indices[i]))
            unique_neighbor_counts.append(unique_count)
        
        pct_unique = np.mean([count == self.k for count in unique_neighbor_counts]) * 100
        
        pct_unique_path = self.output_dir / "checks" / "pct_unique_neighbors.txt"
        with open(pct_unique_path, 'w') as f:
            f.write(f"Percentage of movies with all unique neighbors: {pct_unique:.2f}%\n")
            f.write(f"Movies with all unique neighbors: {sum([count == self.k for count in unique_neighbor_counts])}\n")
            f.write(f"Total movies: {len(unique_neighbor_counts)}\n")
        
        logger.info(f"Saved unique neighbors analysis to {pct_unique_path}")
        logger.info(f"Percentage of movies with all unique neighbors: {pct_unique:.2f}%")
    
    def save_artifacts(self, knn_indices, knn_scores, neighbor_df):
        """Save all artifacts."""
        start_time = time.time()
        logger.info("Saving artifacts...")
        
        # Save kNN indices
        indices_path = self.output_dir / "knn_indices_k50.npz"
        np.savez_compressed(indices_path, indices=knn_indices)
        indices_size = indices_path.stat().st_size / (1024**2)  # MB
        logger.info(f"Saved kNN indices to {indices_path} ({indices_size:.2f} MB)")
        
        # Save kNN scores
        scores_path = self.output_dir / "knn_scores_k50.npz"
        np.savez_compressed(scores_path, scores=knn_scores)
        scores_size = scores_path.stat().st_size / (1024**2)  # MB
        logger.info(f"Saved kNN scores to {scores_path} ({scores_size:.2f} MB)")
        
        # Save neighbor table
        neighbor_path = self.output_dir / "movies_neighbors_k50.parquet"
        neighbor_df.to_parquet(neighbor_path, index=False)
        neighbor_size = neighbor_path.stat().st_size / (1024**2)  # MB
        logger.info(f"Saved neighbor table to {neighbor_path} ({neighbor_size:.2f} MB)")
        
        self.timings['save'] = time.time() - start_time
        self.memory_peaks['artifact_sizes'] = {
            'indices_mb': indices_size,
            'scores_mb': scores_size,
            'neighbor_table_mb': neighbor_size
        }
    
    def create_readme(self):
        """Create README for the similarity directory."""
        readme_content = """# Movie Similarity Data

## Overview
This directory contains cosine similarity and k-nearest neighbor (kNN) data computed from the composite movie embeddings.

## Artifacts

### 1. knn_indices_k50.npz
Compressed NumPy array containing neighbor indices for each movie:
- **Shape**: (87,601, 50)
- **Dtype**: int32
- **Content**: For each movie (row), contains indices of its 50 most similar movies
- **Usage**: `np.load('knn_indices_k50.npz')['indices']`

### 2. knn_scores_k50.npz
Compressed NumPy array containing similarity scores for each movie:
- **Shape**: (87,601, 50)
- **Dtype**: float32
- **Content**: For each movie (row), contains cosine similarity scores with its 50 most similar movies
- **Range**: [0, 1] where 1.0 = identical, 0.0 = completely different
- **Usage**: `np.load('knn_scores_k50.npz')['scores']`

### 3. movies_neighbors_k50.parquet
Long-format table with all neighbor relationships:
- **Columns**: movie_id, neighbor_id, score, rank
- **Rows**: 87,601 × 50 = 4,380,050
- **Usage**: `pd.read_parquet('movies_neighbors_k50.parquet')`

## Key Properties

### Cosine Similarity = Dot Product
Since the input embeddings are L2-normalized, cosine similarity is computed using simple dot products:
```python
# No need for cosine calculation - just use dot product
similarity = embedding[movie1] @ embedding[movie2]
```

### Self-Exclusion
Each movie's neighbor list excludes itself (no self-similarity).

### Score Monotonicity
Within each movie's neighbor list, scores are non-increasing by rank (rank 1 has highest score).

### Memory Efficiency
- Embeddings loaded with memory mapping
- Computations done in batches to control peak memory usage
- Output arrays compressed for storage efficiency

## Performance Notes
- **Recommended K**: 50 neighbors per movie
- **Batch size**: 2,000 movies per batch (configurable)
- **Memory usage**: Peak ~2-5 GB per batch
- **Computation**: CPU-based matrix multiplication using optimized BLAS

## Future Optimizations
For larger scale or faster computation, consider:
- FAISS IndexFlatIP for GPU acceleration
- Annoy for approximate nearest neighbor search
- Parallel processing across multiple cores
"""
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        logger.info(f"Created README at {readme_path}")
    
    def log_final_summary(self):
        """Log final summary with all timings and statistics."""
        total_time = sum(self.timings.values())
        
        summary = f"""
=== Step 3a.2 Similarity Computation Summary ===
Total execution time: {total_time:.2f}s
Load time: {self.timings.get('load', 0):.2f}s
Compute time: {self.timings.get('compute', 0):.2f}s
Save time: {self.timings.get('save', 0):.2f}s

Performance metrics:
- Batch size: {self.batch_size}
- Average batch time: {self.memory_peaks.get('avg_batch_time', 0):.2f}s
- Peak memory per batch: {self.memory_peaks.get('max_batch_memory', 0):.2f}GB

Output artifacts:
- kNN indices: {self.memory_peaks.get('artifact_sizes', {}).get('indices_mb', 0):.2f} MB
- kNN scores: {self.memory_peaks.get('artifact_sizes', {}).get('scores_mb', 0):.2f} MB
- Neighbor table: {self.memory_peaks.get('artifact_sizes', {}).get('neighbor_table_mb', 0):.2f} MB

Ready for Step 3a.3: QA & Spot Checks
"""
        
        logger.info(summary)
        
        # Also write to log file
        with open('logs/step3a_similarity.log', 'a') as f:
            f.write(summary)
    
    def compute_similarities(self):
        """Main computation pipeline."""
        start_time = time.time()
        logger.info("Starting Step 3a.2: Similarity Computation")
        
        try:
            # Step 1: Load embedding and metadata
            embedding = self.load_embedding()
            metadata = self.load_metadata()
            
            # Step 2: Compute similarities in batches
            knn_indices, knn_scores = self.compute_similarities_batched(embedding)
            
            # Step 3: Create neighbor table
            neighbor_df = self.create_neighbor_table(knn_indices, knn_scores, metadata)
            
            # Step 4: Run QA checks
            checks = self.run_qa_checks(knn_indices, knn_scores, neighbor_df, embedding)
            
            # Step 5: Create QA artifacts
            self.create_qa_artifacts(knn_indices, knn_scores, neighbor_df, embedding)
            
            # Step 6: Save artifacts
            self.save_artifacts(knn_indices, knn_scores, neighbor_df)
            
            # Step 7: Create README
            self.create_readme()
            
            # Step 8: Log final summary
            self.log_final_summary()
            
            total_time = time.time() - start_time
            logger.info(f"Step 3a.2 completed successfully in {total_time:.2f} seconds")
            logger.info(f"Output: {self.expected_rows} movies × {self.k} neighbors")
            logger.info(f"Ready for Step 3a.3: QA & Spot Checks")
            
            return True
            
        except Exception as e:
            logger.error(f"Step 3a.2 failed: {str(e)}")
            raise

def main():
    """Main entry point."""
    computer = SimilarityComputer()
    computer.compute_similarities()

if __name__ == "__main__":
    main()
















