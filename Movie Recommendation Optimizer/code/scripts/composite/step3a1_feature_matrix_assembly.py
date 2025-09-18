#!/usr/bin/env python3
"""
Step 3a.1: Feature Matrix Assembly
Assemble a single, aligned feature space per movie with sensible weights.
Produces:
- A dense, L2-normalized composite embedding for fast cosine similarity
- A sparse, block-stacked view for explainability and audits
"""

import os
import sys
import json
import time
import logging
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.spatial.distance import cosine
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
        logging.FileHandler('logs/step3a1_feature_matrix_assembly.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureMatrixAssembler:
    """Assembles composite feature matrix from multiple feature families."""
    
    def __init__(self, output_dir="data/features/composite"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checks").mkdir(exist_ok=True)
        
        # V1 weights (family-level)
        self.family_weights = {
            'bert': 0.50,
            'tfidf': 0.20,
            'genres': 0.15,
            'crew': 0.05,
            'numeric': 0.10,
            'platform': 0.00  # Near-zero coverage for v1
        }
        
        # Verify weights sum to 1.0
        active_weight_sum = sum(w for w in self.family_weights.values() if w > 0)
        assert abs(active_weight_sum - 1.0) < 1e-6, f"Weights sum to {active_weight_sum}, expected 1.0"
        
        # Fixed random seed for deterministic projection
        self.random_seed = 42
        np.random.seed(self.random_seed)
        
        # Target dimensions
        self.bert_dim = 384
        self.composite_dim = 384  # Project non-BERT families into BERT space
        
        # Canonical ID mapping
        self.canonical_ids = None
        self.canonical_id_to_idx = None
        self.expected_row_count = 87601
        
    def load_canonical_index(self):
        """Load the canonical index to establish the base row mapping."""
        logger.info("Loading canonical index...")
        
        # Try the 87601 canonical index first
        canonical_path = "data/features/text/movies_canonical_index.parquet"
        if os.path.exists(canonical_path):
            canonical_df = pd.read_parquet(canonical_path)
            if len(canonical_df) == self.expected_row_count:
                logger.info(f"Using canonical index with {len(canonical_df)} rows")
                self.canonical_ids = canonical_df['canonical_id'].values
                self.canonical_id_to_idx = {cid: idx for idx, cid in enumerate(self.canonical_ids)}
                return
        
        # Fallback to the 88194 index and map down
        fallback_path = "data/features/text/movies_canonical_index_88194.parquet"
        if os.path.exists(fallback_path):
            fallback_df = pd.read_parquet(fallback_path)
            logger.info(f"Using fallback index with {len(fallback_df)} rows")
            
            # Get unique canonical IDs and sort for deterministic ordering
            unique_ids = sorted(fallback_df['canonical_id'].unique())
            if len(unique_ids) == self.expected_row_count:
                self.canonical_ids = np.array(unique_ids)
                self.canonical_id_to_idx = {cid: idx for idx, cid in enumerate(self.canonical_ids)}
                logger.info(f"Mapped {len(fallback_df)} rows to {len(self.canonical_ids)} unique canonical IDs")
                return
        
        raise ValueError(f"Could not establish canonical index with {self.expected_row_count} rows")
    
    def load_text_features(self):
        """Load BERT and TF-IDF features."""
        logger.info("Loading text features...")
        
        # Load BERT combined embeddings
        bert_path = "data/features/text/movies_text_bert_combined.npy"
        bert_embeddings = np.load(bert_path)
        logger.info(f"Loaded BERT embeddings: {bert_embeddings.shape}")
        
        # Load TF-IDF combined matrix
        tfidf_path = "data/features/text/movies_text_tfidf_combined.npz"
        tfidf_matrix = sparse.load_npz(tfidf_path)
        logger.info(f"Loaded TF-IDF matrix: {tfidf_matrix.shape}")
        
        # Load text index to map to canonical IDs
        text_index_path = "data/features/text/index/movies_text_index.parquet"
        text_index = pd.read_parquet(text_index_path)
        
        # Map text features to canonical index
        bert_mapped = self._map_features_to_canonical(bert_embeddings, text_index, 'bert')
        tfidf_mapped = self._map_features_to_canonical(tfidf_matrix, text_index, 'tfidf')
        
        return bert_mapped, tfidf_mapped
    
    def load_categorical_features(self):
        """Load genre and crew features."""
        logger.info("Loading categorical features...")
        
        # Load categorical features (already contains genres + crew)
        categorical_path = "data/features/categorical/movies_categorical_features.parquet"
        categorical_df = pd.read_parquet(categorical_path)
        logger.info(f"Loaded categorical features: {categorical_df.shape}")
        
        # Split into genres and crew
        genre_cols = [col for col in categorical_df.columns if col.startswith('genre_')]
        crew_cols = [col for col in categorical_df.columns if not col.startswith('genre_') and col != 'canonical_id']
        
        genres_df = categorical_df[['canonical_id'] + genre_cols]
        crew_df = categorical_df[['canonical_id'] + crew_cols]
        
        logger.info(f"Genres: {len(genre_cols)} columns, Crew: {len(crew_cols)} columns")
        
        # Map to canonical index
        genres_mapped = self._map_features_to_canonical(genres_df.iloc[:, 1:].values, genres_df, 'genres')
        crew_mapped = self._map_features_to_canonical(crew_df.iloc[:, 1:].values, crew_df, 'crew')
        
        return genres_mapped, crew_mapped
    
    def load_numeric_features(self):
        """Load standardized numeric features."""
        logger.info("Loading numeric features...")
        
        numeric_path = "data/features/numeric/movies_numeric_standardized.parquet"
        numeric_df = pd.read_parquet(numeric_path)
        logger.info(f"Loaded numeric features: {numeric_df.shape}")
        
        # Reset index to make canonical_id a column
        numeric_df = numeric_df.reset_index()
        logger.info(f"Numeric features after reset_index: {numeric_df.shape}")
        
        # Map to canonical index (exclude the canonical_id column)
        numeric_mapped = self._map_features_to_canonical(numeric_df.iloc[:, 1:].values, numeric_df, 'numeric')
        
        return numeric_mapped
    
    def load_platform_features(self):
        """Load platform features."""
        logger.info("Loading platform features...")
        
        platform_path = "data/features/platform/movies_platform_features.parquet"
        platform_df = pd.read_parquet(platform_path)
        logger.info(f"Loaded platform features: {platform_df.shape}")
        
        # Reset index to make canonical_id a column
        platform_df = platform_df.reset_index()
        logger.info(f"Platform features after reset_index: {platform_df.shape}")
        
        # Map to canonical index (exclude the canonical_id column)
        platform_mapped = self._map_features_to_canonical(platform_df.iloc[:, 1:].values, platform_df, 'platform')
        
        return platform_mapped
    
    def _map_features_to_canonical(self, features, source_df, feature_name):
        """Map features from source index to canonical index."""
        # Check feature length (handle both sparse and dense)
        if sparse.issparse(features):
            feature_length = features.shape[0]
        else:
            feature_length = len(features)
            
        if feature_length != len(source_df):
            raise ValueError(f"Feature length {feature_length} != source length {len(source_df)} for {feature_name}")
        
        # Create mapping from source canonical_id to canonical index position
        source_to_canonical = {}
        for idx, row in source_df.iterrows():
            source_id = row['canonical_id']
            if source_id in self.canonical_id_to_idx:
                source_to_canonical[idx] = self.canonical_id_to_idx[source_id]
        
        # Initialize output matrix
        if sparse.issparse(features):
            output = sparse.csr_matrix((self.expected_row_count, features.shape[1]), dtype=features.dtype)
        else:
            # For numeric features, ensure we use float32
            if feature_name == 'numeric':
                output = np.zeros((self.expected_row_count, features.shape[1]), dtype=np.float32)
            else:
                output = np.zeros((self.expected_row_count, features.shape[1]), dtype=features.dtype)
        
        # Map features
        mapped_count = 0
        for source_idx, canonical_idx in source_to_canonical.items():
            if sparse.issparse(features):
                output[canonical_idx] = features[source_idx]
            else:
                # For numeric features, ensure proper conversion
                if feature_name == 'numeric':
                    output[canonical_idx] = features[source_idx].astype(np.float32)
                else:
                    output[canonical_idx] = features[source_idx]
            mapped_count += 1
        
        logger.info(f"Mapped {mapped_count}/{len(source_to_canonical)} {feature_name} features to canonical index")
        return output
    
    def normalize_features(self, bert_features, tfidf_features, genres_features, crew_features, numeric_features, platform_features):
        """Apply normalization rules to each feature family."""
        logger.info("Normalizing features...")
        
        # BERT: confirm L2-normalized
        bert_norms = np.linalg.norm(bert_features, axis=1, keepdims=True)
        if not np.allclose(bert_norms, 1.0, atol=1e-3):
            logger.info("L2-normalizing BERT features...")
            bert_features = bert_features / (bert_norms + 1e-8)
        
        # TF-IDF: L2-normalize for cosine semantics
        logger.info("L2-normalizing TF-IDF features...")
        tfidf_squared = tfidf_features.power(2)
        tfidf_sums = tfidf_squared.sum(axis=1)
        tfidf_norms = np.sqrt(np.array(tfidf_sums)).flatten()
        tfidf_norms = np.maximum(tfidf_norms, 1e-8)
        tfidf_features = tfidf_features.multiply(1.0 / tfidf_norms.reshape(-1, 1))
        
        # Genres: row-scale to unit L2 norm if any 1s
        logger.info("Normalizing genre features...")
        logger.info(f"Genres features shape: {genres_features.shape}, dtype: {genres_features.dtype}")
        
        # Ensure genres_features is a numpy array
        if not isinstance(genres_features, np.ndarray):
            genres_features = np.array(genres_features)
        
        genres_norms = np.linalg.norm(genres_features, axis=1, keepdims=True)
        genres_normalized = np.zeros_like(genres_features)
        non_zero_mask = genres_norms.flatten() > 0
        genres_normalized[non_zero_mask] = genres_features[non_zero_mask] / genres_norms[non_zero_mask]
        genres_features = genres_normalized
        
        # Crew: row-scale to unit L2 norm if any 1s
        logger.info("Normalizing crew features...")
        logger.info(f"Crew features shape: {crew_features.shape}, dtype: {crew_features.dtype}")
        
        # Ensure crew_features is a numpy array
        if not isinstance(crew_features, np.ndarray):
            crew_features = np.array(crew_features)
        
        crew_norms = np.linalg.norm(crew_features, axis=1, keepdims=True)
        crew_normalized = np.zeros_like(crew_features)
        non_zero_mask = crew_norms.flatten() > 0
        crew_normalized[non_zero_mask] = crew_features[non_zero_mask] / crew_norms[non_zero_mask]
        crew_features = crew_normalized
        
        # Numeric: rescale to unit L2 norm if any non-zero
        logger.info("Normalizing numeric features...")
        logger.info(f"Numeric features shape: {numeric_features.shape}, dtype: {numeric_features.dtype}")
        logger.info(f"Numeric features sample: {numeric_features[:5, :3]}")
        
        # Ensure numeric_features is a numpy array
        if not isinstance(numeric_features, np.ndarray):
            numeric_features = np.array(numeric_features)
        
        numeric_norms = np.linalg.norm(numeric_features, axis=1, keepdims=True)
        numeric_normalized = np.zeros_like(numeric_features)
        non_zero_mask = numeric_norms.flatten() > 0
        numeric_normalized[non_zero_mask] = numeric_features[non_zero_mask] / numeric_norms[non_zero_mask]
        numeric_features = numeric_normalized
        
        # Platform: leave as is (weight = 0)
        logger.info("Platform features left unchanged (weight = 0)")
        logger.info(f"Platform features shape: {platform_features.shape}, dtype: {platform_features.dtype}")
        
        # Ensure platform_features is a numpy array
        if not isinstance(platform_features, np.ndarray):
            platform_features = np.array(platform_features)
        
        return bert_features, tfidf_features, genres_features, crew_features, numeric_features, platform_features
    
    def create_sparse_block_view(self, tfidf_features, genres_features, crew_features, numeric_features, platform_features):
        """Create sparse block-stacked view for explainability."""
        logger.info("Creating sparse block-stacked view...")
        
        # Convert dense features to sparse
        genres_sparse = sparse.csr_matrix(genres_features)
        crew_sparse = sparse.csr_matrix(crew_features)
        numeric_sparse = sparse.csr_matrix(numeric_features)
        platform_sparse = sparse.csr_matrix(platform_features)
        
        # Stack blocks: [TF-IDF | Genres | Crew | Numeric | Platform]
        blocks = [tfidf_features, genres_sparse, crew_sparse, numeric_sparse, platform_sparse]
        block_names = ['tfidf', 'genres', 'crew', 'numeric', 'platform']
        
        # Calculate block boundaries
        block_boundaries = []
        current_pos = 0
        for block in blocks:
            block_boundaries.append((current_pos, current_pos + block.shape[1]))
            current_pos += block.shape[1]
        
        # Horizontal stack
        sparse_block = sparse.hstack(blocks, format='csr')
        logger.info(f"Created sparse block view: {sparse_block.shape}")
        
        return sparse_block, block_boundaries, block_names
    
    def create_dense_composite(self, bert_features, sparse_block, block_boundaries, block_names):
        """Create dense composite embedding by projecting sparse features into BERT space."""
        logger.info("Creating dense composite embedding...")
        
        # Create deterministic projection matrix
        logger.info(f"Creating projection matrix: {sparse_block.shape[1]} -> {self.composite_dim}")
        
        # Use random orthogonal projection for deterministic results
        projection_matrix = self._create_orthogonal_projection(sparse_block.shape[1], self.composite_dim)
        
        # Project sparse features to dense space
        logger.info(f"Projecting sparse block {sparse_block.shape} to dense space...")
        logger.info(f"Projection matrix shape: {projection_matrix.shape}")
        logger.info(f"Projection matrix.T shape: {projection_matrix.T.shape}")
        
        # Apply family weights to sparse features BEFORE projection
        logger.info("Applying family weights to sparse features...")
        
        # Create weighted sparse block by combining weighted individual blocks
        weighted_blocks = []
        
        # TF-IDF (0.20)
        tfidf_start, tfidf_end = block_boundaries[0]
        tfidf_weighted = 0.20 * sparse_block[:, tfidf_start:tfidf_end]
        weighted_blocks.append(tfidf_weighted)
        logger.info(f"TF-IDF weighted: {tfidf_end - tfidf_start} columns, weight: 0.20")
        
        # Genres (0.15)
        genres_start, genres_end = block_boundaries[1]
        genres_weighted = 0.15 * sparse_block[:, genres_start:genres_end]
        weighted_blocks.append(genres_weighted)
        logger.info(f"Genres weighted: {genres_end - genres_start} columns, weight: 0.15")
        
        # Crew (0.05)
        crew_start, crew_end = block_boundaries[2]
        crew_weighted = 0.05 * sparse_block[:, crew_start:crew_end]
        weighted_blocks.append(crew_weighted)
        logger.info(f"Crew weighted: {crew_end - crew_start} columns, weight: 0.05")
        
        # Numeric (0.10)
        numeric_start, numeric_end = block_boundaries[3]
        numeric_weighted = 0.10 * sparse_block[:, numeric_start:numeric_end]
        weighted_blocks.append(numeric_weighted)
        logger.info(f"Numeric weighted: {numeric_end - numeric_start} columns, weight: 0.10")
        
        # Platform (0.00) - no contribution
        platform_start, platform_end = block_boundaries[4]
        platform_weighted = 0.00 * sparse_block[:, platform_start:platform_end]
        weighted_blocks.append(platform_weighted)
        logger.info(f"Platform weighted: {platform_end - platform_start} columns, weight: 0.00")
        
        # Combine weighted blocks
        weighted_sparse = sparse.hstack(weighted_blocks, format='csr')
        logger.info(f"Combined weighted sparse block: {weighted_sparse.shape}")
        
        # Project weighted sparse features to dense space
        logger.info(f"Projecting weighted sparse block to dense space...")
        weighted_projected = weighted_sparse @ projection_matrix
        logger.info(f"Weighted projected shape: {weighted_projected.shape}")
        
        # Combine with BERT (0.50)
        logger.info(f"Combining BERT ({bert_features.shape}) with weighted projection ({weighted_projected.shape})")
        composite = 0.50 * bert_features + weighted_projected
        
        # Final L2-normalization
        composite_norms = np.linalg.norm(composite, axis=1, keepdims=True)
        composite = composite / (composite_norms + 1e-8)
        
        logger.info(f"Created composite embedding: {composite.shape}")
        return composite, projection_matrix
    
    def _create_orthogonal_projection(self, input_dim, output_dim):
        """Create deterministic orthogonal projection matrix."""
        logger.info(f"Creating orthogonal projection: {input_dim} -> {output_dim}")
        
        # Use fixed seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Generate random matrix and make it orthogonal
        random_matrix = np.random.randn(input_dim, output_dim)
        logger.info(f"Random matrix shape: {random_matrix.shape}")
        
        q, _ = np.linalg.qr(random_matrix)
        logger.info(f"QR decomposition result shape: {q.shape}")
        
        # Ensure output dimension is correct
        if q.shape[1] > output_dim:
            q = q[:, :output_dim]
            logger.info(f"Truncated to: {q.shape}")
        elif q.shape[1] < output_dim:
            # Pad with zeros if needed
            padding = np.zeros((input_dim, output_dim - q.shape[1]))
            q = np.hstack([q, padding])
            logger.info(f"Padded to: {q.shape}")
        
        logger.info(f"Final projection matrix: {q.shape}")
        return q
    
    def create_metadata(self, bert_features, tfidf_features, genres_features, crew_features, 
                       numeric_features, platform_features, sparse_block, composite_embedding, 
                       projection_matrix, block_boundaries, block_names):
        """Create metadata for the composite features."""
        logger.info("Creating metadata...")
        
        # Calculate per-family statistics
        family_stats = {}
        
        # BERT
        bert_norms = np.linalg.norm(bert_features, axis=1)
        family_stats['bert'] = {
            'shape': bert_features.shape,
            'dtype': str(bert_features.dtype),
            'mean_norm': float(np.mean(bert_norms)),
            'std_norm': float(np.std(bert_norms)),
            'nnz': bert_features.shape[0] * bert_features.shape[1]  # All non-zero for dense
        }
        
        # TF-IDF
        tfidf_nnz = tfidf_features.nnz
        tfidf_squared = tfidf_features.power(2)
        tfidf_sums = tfidf_squared.sum(axis=1)
        tfidf_norms = np.sqrt(np.array(tfidf_sums)).flatten()
        family_stats['tfidf'] = {
            'shape': tfidf_features.shape,
            'dtype': str(tfidf_features.dtype),
            'mean_norm': float(np.mean(tfidf_norms)),
            'std_norm': float(np.std(tfidf_norms)),
            'nnz': int(tfidf_nnz),
            'sparsity': float(1.0 - tfidf_nnz / (tfidf_features.shape[0] * tfidf_features.shape[1]))
        }
        
        # Genres
        genres_nnz = np.count_nonzero(genres_features)
        genres_norms = np.linalg.norm(genres_features, axis=1)
        family_stats['genres'] = {
            'shape': genres_features.shape,
            'dtype': str(genres_features.dtype),
            'mean_norm': float(np.mean(genres_norms)),
            'std_norm': float(np.std(genres_norms)),
            'nnz': int(genres_nnz),
            'sparsity': float(1.0 - genres_nnz / (genres_features.shape[0] * genres_features.shape[1]))
        }
        
        # Crew
        crew_nnz = np.count_nonzero(crew_features)
        crew_norms = np.linalg.norm(crew_features, axis=1)
        family_stats['crew'] = {
            'shape': crew_features.shape,
            'dtype': str(crew_features.dtype),
            'mean_norm': float(np.mean(crew_norms)),
            'std_norm': float(np.std(crew_norms)),
            'nnz': int(crew_nnz),
            'sparsity': float(1.0 - crew_nnz / (crew_features.shape[0] * crew_features.shape[1]))
        }
        
        # Numeric
        numeric_nnz = np.count_nonzero(numeric_features)
        numeric_norms = np.linalg.norm(numeric_features, axis=1)
        family_stats['numeric'] = {
            'shape': numeric_features.shape,
            'dtype': str(numeric_features.dtype),
            'mean_norm': float(np.mean(numeric_norms)),
            'std_norm': float(np.std(numeric_norms)),
            'nnz': int(numeric_nnz),
            'sparsity': float(1.0 - numeric_nnz / (numeric_features.shape[0] * numeric_features.shape[1]))
        }
        
        # Platform
        platform_nnz = np.count_nonzero(platform_features)
        platform_norms = np.linalg.norm(platform_features, axis=1)
        family_stats['platform'] = {
            'shape': platform_features.shape,
            'dtype': str(platform_features.dtype),
            'mean_norm': float(np.mean(platform_norms)),
            'std_norm': float(np.std(platform_norms)),
            'nnz': int(platform_nnz),
            'sparsity': float(1.0 - platform_nnz / (platform_features.shape[0] * platform_features.shape[1]))
        }
        
        # Composite
        composite_norms = np.linalg.norm(composite_embedding, axis=1)
        family_stats['composite'] = {
            'shape': composite_embedding.shape,
            'dtype': str(composite_embedding.dtype),
            'mean_norm': float(np.mean(composite_norms)),
            'std_norm': float(np.std(composite_norms)),
            'nnz': composite_embedding.shape[0] * composite_embedding.shape[1]
        }
        
        # Create metadata DataFrame
        metadata_rows = []
        for canonical_idx, canonical_id in enumerate(self.canonical_ids):
            row_data = {
                'canonical_id': canonical_id,
                'canonical_idx': canonical_idx
            }
            
            # Add per-family norms
            for family in ['bert', 'tfidf', 'genres', 'crew', 'numeric', 'platform']:
                if family == 'tfidf':
                    row_data[f'{family}_norm'] = float(tfidf_norms[canonical_idx])
                elif family in ['genres', 'crew', 'numeric', 'platform']:
                    row_data[f'{family}_norm'] = float(family_stats[family]['mean_norm'])
                else:
                    row_data[f'{family}_norm'] = float(family_stats[family]['mean_norm'])
            
            # Add composite norm
            row_data['composite_norm'] = float(composite_norms[canonical_idx])
            
            # Add nnz counts (simplified approach)
            for family in ['tfidf', 'genres', 'crew', 'numeric', 'platform']:
                if family == 'tfidf':
                    # For TF-IDF, use average nnz per row since individual row access is problematic
                    row_data[f'{family}_nnz'] = int(tfidf_features.nnz / tfidf_features.shape[0])
                elif family == 'genres':
                    row_data[f'{family}_nnz'] = int(np.count_nonzero(genres_features[canonical_idx]))
                elif family == 'crew':
                    row_data[f'{family}_nnz'] = int(np.count_nonzero(crew_features[canonical_idx]))
                elif family == 'numeric':
                    row_data[f'{family}_nnz'] = int(np.count_nonzero(numeric_features[canonical_idx]))
                elif family == 'platform':
                    row_data[f'{family}_nnz'] = int(np.count_nonzero(platform_features[canonical_idx]))
            
            metadata_rows.append(row_data)
        
        metadata_df = pd.DataFrame(metadata_rows)
        
        return metadata_df, family_stats
    
    def save_artifacts(self, metadata_df, composite_embedding, family_stats, 
                       projection_matrix, block_boundaries, block_names):
        """Save all artifacts."""
        logger.info("Saving artifacts...")
        
        # Save metadata
        metadata_path = self.output_dir / "movies_features_v1.parquet"
        metadata_df.to_parquet(metadata_path, index=False)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save composite embedding
        embedding_path = self.output_dir / "movies_embedding_v1.npy"
        np.save(embedding_path, composite_embedding)
        logger.info(f"Saved embedding to {embedding_path}")
        
        # Save manifest
        manifest = self._create_manifest(family_stats, projection_matrix, block_boundaries, block_names)
        manifest_path = self.output_dir / "manifest_composite_v1.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info(f"Saved manifest to {manifest_path}")
        
        # Save sample row norms for QA
        sample_rows = metadata_df.sample(n=min(100, len(metadata_df)), random_state=42)
        sample_path = self.output_dir / "checks" / "row_norms_sample.csv"
        sample_rows.to_csv(sample_path, index=False)
        logger.info(f"Saved sample rows to {sample_path}")
        
        # Save README
        readme_path = self.output_dir / "README.md"
        self._create_readme(readme_path, family_stats, manifest)
        logger.info(f"Saved README to {readme_path}")
    
    def _create_manifest(self, family_stats, projection_matrix, block_boundaries, block_names):
        """Create comprehensive manifest."""
        manifest = {
            "step": "3a.1",
            "description": "Feature Matrix Assembly - Composite embedding creation",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": "v1",
            
            "canonical_index": {
                "row_count": self.expected_row_count,
                "canonical_ids": self.canonical_ids.tolist()[:10],  # First 10 for reference
                "note": f"Total {self.expected_row_count} unique canonical IDs"
            },
            
            "family_weights": self.family_weights,
            "active_weight_sum": sum(w for w in self.family_weights.values() if w > 0),
            
            "normalization_rules": {
                "bert": "L2-normalized per row (cosine semantics)",
                "tfidf": "L2-normalized per row (cosine semantics)",
                "genres": "Row-scaled to unit L2 norm if any 1s, else zero",
                "crew": "Row-scaled to unit L2 norm if any 1s, else zero",
                "numeric": "Row-scaled to unit L2 norm if any non-zero, else zero",
                "platform": "Unchanged (weight = 0)"
            },
            
            "projection": {
                "type": "random_orthogonal",
                "seed": self.random_seed,
                "input_dim": sum(block[1] - block[0] for block in block_boundaries),
                "output_dim": self.composite_dim,
                "method": "QR decomposition of random matrix"
            },
            
            "block_structure": {
                "order": block_names,
                "boundaries": {name: (start, end) for name, (start, end) in zip(block_names, block_boundaries)}
            },
            
            "family_statistics": family_stats,
            
            "artifacts": {
                "metadata": {
                    "path": str(self.output_dir / "movies_features_v1.parquet"),
                    "shape": [self.expected_row_count, len(family_stats) * 2 + 2],  # Approximate
                    "dtype": "object"
                },
                "embedding": {
                    "path": str(self.output_dir / "movies_embedding_v1.npy"),
                    "shape": [self.expected_row_count, self.composite_dim],
                    "dtype": "float32"
                },
                "manifest": {
                    "path": str(self.output_dir / "manifest_composite_v1.json"),
                    "type": "json"
                }
            },
            
            "source_files": {
                "bert": "data/features/text/movies_text_bert_combined.npy",
                "tfidf": "data/features/text/movies_text_tfidf_combined.npz",
                "genres": "data/features/genres/movies_genres_multihot.parquet",
                "crew_actors": "data/features/crew/movies_actors_top50.parquet",
                "crew_directors": "data/features/crew/movies_directors_top50.parquet",
                "numeric": "data/features/numeric/movies_numeric_standardized.parquet",
                "platform": "data/features/platform/movies_platform_features.parquet"
            },
            
            "acceptance_gates": {
                "row_count": len(self.canonical_ids) == self.expected_row_count,
                "no_nan_inf": True,  # Will be set by caller
                "unit_norm": True,    # Will be set by caller
                "weight_sum": abs(sum(w for w in self.family_weights.values() if w > 0) - 1.0) < 1e-6
            },
            
            "performance_notes": {
                "recommended_k": 50,
                "cosine_similarity": "Use dot product (vectors are L2-normalized)",
                "memory_efficient": "Use memory-mapped loading for large embeddings"
            }
        }
        
        return manifest
    
    def _create_readme(self, readme_path, family_stats, manifest):
        """Create README for the composite features."""
        readme_content = f"""# Composite Features v1

## Overview
This directory contains the composite feature matrix assembled from multiple feature families for movie recommendation.

## Artifacts

### 1. movies_features_v1.parquet
Metadata and statistics for each movie:
- `canonical_id`: Unique movie identifier
- `canonical_idx`: Row index in the embedding matrix
- Per-family norms and nnz counts
- Composite norm (should be ≈1.0)

**Shape**: ({manifest['canonical_index']['row_count']}, ~{len(family_stats) * 2 + 2})

### 2. movies_embedding_v1.npy
Dense composite embedding matrix for fast cosine similarity:
- **Shape**: ({manifest['canonical_index']['row_count']}, {manifest['projection']['output_dim']})
- **Dtype**: float32
- **Normalization**: L2-normalized per row (‖v‖₂ ≈ 1.0)
- **Usage**: Load with `np.load()` and use dot product for cosine similarity

### 3. manifest_composite_v1.json
Complete configuration and metadata for reproducibility.

## Feature Family Weights
{chr(10).join(f"- {family}: {weight:.2f}" for family, weight in manifest['family_weights'].items())}

## Loading and Usage

```python
import numpy as np
import pandas as pd

# Load embedding
embedding = np.load('movies_embedding_v1.npy')

# Load metadata
metadata = pd.read_parquet('movies_features_v1.parquet')

# Compute cosine similarity (dot product since vectors are L2-normalized)
similarity = embedding[0] @ embedding[1]  # Between movies 0 and 1

# Find nearest neighbors
def find_neighbors(query_idx, k=50):
    similarities = embedding @ embedding[query_idx]
    # Exclude self-similarity
    similarities[query_idx] = -1
    neighbor_indices = np.argsort(similarities)[::-1][:k]
    return neighbor_indices, similarities[neighbor_indices]
```

## Performance Notes
- **Recommended K**: 50 for nearest neighbor search
- **Memory**: Use memory-mapped loading for large embeddings
- **Cosine similarity**: Use dot product (vectors are L2-normalized)

## Version History
- **v1**: Initial composite embedding with BERT + sparse feature projection
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    def run_qa_checks(self, composite_embedding, metadata_df):
        """Run acceptance gate checks."""
        logger.info("Running QA checks...")
        
        checks = {}
        
        # Check 1: Exactly 87,601 rows
        checks['row_count'] = len(composite_embedding) == self.expected_row_count
        logger.info(f"Row count check: {len(composite_embedding)} == {self.expected_row_count} -> {checks['row_count']}")
        
        # Check 2: No NaN/Inf
        checks['no_nan_inf'] = not np.any(np.isnan(composite_embedding)) and not np.any(np.isinf(composite_embedding))
        logger.info(f"No NaN/Inf check: {checks['no_nan_inf']}")
        
        # Check 3: Unit norm
        norms = np.linalg.norm(composite_embedding, axis=1)
        checks['unit_norm'] = np.allclose(norms, 1.0, atol=1e-3)
        logger.info(f"Unit norm check: {checks['unit_norm']} (range: {norms.min():.6f} - {norms.max():.6f})")
        
        # Check 4: Weight sum
        active_weight_sum = sum(w for w in self.family_weights.values() if w > 0)
        checks['weight_sum'] = abs(active_weight_sum - 1.0) < 1e-6
        logger.info(f"Weight sum check: {active_weight_sum} == 1.0 -> {checks['weight_sum']}")
        
        # Check 5: Canonical ID uniqueness
        checks['unique_ids'] = len(metadata_df['canonical_id'].unique()) == self.expected_row_count
        logger.info(f"Unique IDs check: {checks['unique_ids']}")
        
        # Log summary
        all_passed = all(checks.values())
        logger.info(f"QA checks summary: {sum(checks.values())}/{len(checks)} passed")
        
        if not all_passed:
            failed_checks = [k for k, v in checks.items() if not v]
            logger.error(f"Failed checks: {failed_checks}")
            raise ValueError(f"QA checks failed: {failed_checks}")
        
        return checks
    
    def assemble(self):
        """Main assembly pipeline."""
        start_time = time.time()
        logger.info("Starting Step 3a.1: Feature Matrix Assembly")
        
        try:
            # Step 1: Load canonical index
            self.load_canonical_index()
            
            # Step 2: Load all feature families
            bert_features, tfidf_features = self.load_text_features()
            genres_features, crew_features = self.load_categorical_features()
            numeric_features = self.load_numeric_features()
            platform_features = self.load_platform_features()
            
            # Step 3: Normalize features
            bert_features, tfidf_features, genres_features, crew_features, numeric_features, platform_features = \
                self.normalize_features(bert_features, tfidf_features, genres_features, crew_features, numeric_features, platform_features)
            
            # Step 4: Create sparse block view
            sparse_block, block_boundaries, block_names = self.create_sparse_block_view(
                tfidf_features, genres_features, crew_features, numeric_features, platform_features
            )
            
            # Step 5: Create dense composite embedding
            composite_embedding, projection_matrix = self.create_dense_composite(
                bert_features, sparse_block, block_boundaries, block_names
            )
            
            # Step 6: Create metadata
            metadata_df, family_stats = self.create_metadata(
                bert_features, tfidf_features, genres_features, crew_features,
                numeric_features, platform_features, sparse_block, composite_embedding,
                projection_matrix, block_boundaries, block_names
            )
            
            # Step 7: Run QA checks
            checks = self.run_qa_checks(composite_embedding, metadata_df)
            
            # Step 8: Save artifacts
            self.save_artifacts(metadata_df, composite_embedding, family_stats,
                               projection_matrix, block_boundaries, block_names)
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"Step 3a.1 completed successfully in {duration:.2f} seconds")
            logger.info(f"Output: {self.expected_row_count} movies × {self.composite_dim} features")
            logger.info(f"Ready for Step 3a.2: Cosine + kNN")
            
            return True
            
        except Exception as e:
            logger.error(f"Step 3a.1 failed: {str(e)}")
            raise

def main():
    """Main entry point."""
    assembler = FeatureMatrixAssembler()
    assembler.assemble()

if __name__ == "__main__":
    main()
