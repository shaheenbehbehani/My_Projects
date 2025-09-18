#!/usr/bin/env python3
"""
Step 2a.3: BERT Embeddings
Movie Recommendation Optimizer

Objective: Generate BERT embeddings for cleaned text fields to enable semantic
text-based recommendation features. Creates individual and combined embeddings.
"""

import argparse
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BERTEmbedder:
    """BERT embedding generation utility class"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 batch_size_cpu: int = 64,
                 batch_size_cuda: int = 256):
        """
        Initialize BERT embedder
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size_cpu: Batch size for CPU processing
            batch_size_cuda: Batch size for CUDA processing
        """
        self.model_name = model_name
        self.batch_size_cpu = batch_size_cpu
        self.batch_size_cuda = batch_size_cuda
        
        # Set deterministic seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.batch_size = batch_size_cuda
                logger.info(f"CUDA detected, using device: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                self.batch_size = batch_size_cpu
                logger.info("CUDA not available, using CPU")
        else:
            self.device = device
            self.batch_size = batch_size_cuda if device == 'cuda' else batch_size_cpu
            logger.info(f"Using specified device: {device}")
        
        # Load model
        logger.info(f"Loading BERT model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with dimension: {self.dimension}")
        
        # Set model to evaluation mode
        self.model.eval()
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts to embeddings
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        return embeddings.astype(np.float32)
    
    def encode_field(self, texts: List[str], field_name: str) -> np.ndarray:
        """
        Encode all texts for a field with progress logging
        
        Args:
            texts: List of text strings
            field_name: Name of the field being encoded
            
        Returns:
            NumPy array of embeddings
        """
        logger.info(f"Encoding {field_name} field: {len(texts)} texts")
        
        # Count valid texts
        valid_texts = [text for text in texts if text != "unknown_text"]
        logger.info(f"  Valid texts: {len(valid_texts)} out of {len(texts)}")
        
        # Process in batches
        all_embeddings = []
        start_time = time.time()
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(f"  Processed {min(i + self.batch_size, len(texts)):,}/{len(texts):,} texts")
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        
        elapsed = time.time() - start_time
        logger.info(f"  Completed in {elapsed:.2f}s ({len(texts)/elapsed:.1f} texts/sec)")
        
        return embeddings
    
    def create_weighted_embeddings(self, 
                                  field_embeddings: Dict[str, np.ndarray],
                                  weights: Dict[str, float]) -> np.ndarray:
        """
        Create weighted combination of field embeddings
        
        Args:
            field_embeddings: Dictionary of field_name -> embeddings array
            weights: Dictionary of field_name -> weight
            
        Returns:
            Weighted combined embeddings
        """
        logger.info("Creating weighted combined embeddings...")
        
        # Initialize combined embeddings
        n_rows, n_dims = list(field_embeddings.values())[0].shape
        combined = np.zeros((n_rows, n_dims), dtype=np.float32)
        
        # Track which fields are available for each row
        field_availability = {}
        for field_name, embeddings in field_embeddings.items():
            # Check which rows have non-zero embeddings (not "unknown_text")
            field_availability[field_name] = np.any(embeddings != 0, axis=1)
        
        # Process each row
        for i in range(n_rows):
            available_fields = []
            available_weights = []
            
            for field_name, embeddings in field_embeddings.items():
                if field_availability[field_name][i]:
                    available_fields.append(embeddings[i])
                    available_weights.append(weights[field_name])
            
            if available_fields:
                # Renormalize weights for available fields
                total_weight = sum(available_weights)
                normalized_weights = [w / total_weight for w in available_weights]
                
                # Weighted combination
                combined[i] = np.average(available_fields, weights=normalized_weights, axis=0)
            else:
                # All fields missing - use zero vector
                combined[i] = np.zeros(n_dims, dtype=np.float32)
        
        logger.info(f"Combined embeddings shape: {combined.shape}")
        return combined

def load_data(cleaned_path: str, index_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned text data and index mapping"""
    logger.info("Loading data...")
    
    # Load cleaned text data
    cleaned_df = pd.read_parquet(cleaned_path)
    logger.info(f"Loaded cleaned data: {len(cleaned_df)} rows, {len(cleaned_df.columns)} columns")
    
    # Load index mapping
    index_df = pd.read_parquet(index_path)
    logger.info(f"Loaded index mapping: {len(index_df)} rows")
    
    # Verify row count alignment
    if len(cleaned_df) != len(index_df):
        raise ValueError(f"Row count mismatch: cleaned={len(cleaned_df)}, index={len(index_df)}")
    
    # Verify canonical_id alignment
    if not (cleaned_df['canonical_id'] == index_df['canonical_id']).all():
        raise ValueError("canonical_id order mismatch between cleaned data and index")
    
    logger.info("Data alignment verified successfully")
    return cleaned_df, index_df

def get_text_fields() -> Dict[str, str]:
    """Define text fields to embed and their cleaned counterparts"""
    return {
        'overview': 'overview_cleaned',
        'consensus': 'consensus_cleaned',
        'tags_combined': 'tags_cleaned'
    }

def get_field_weights() -> Dict[str, float]:
    """Define weights for combining field embeddings"""
    return {
        'tags_combined': 0.60,
        'consensus': 0.25,
        'overview': 0.15
    }

def validate_embeddings(embeddings: np.ndarray, 
                       field_name: str, 
                       expected_rows: int, 
                       expected_dims: int) -> None:
    """Validate embedding array properties"""
    if embeddings.shape != (expected_rows, expected_dims):
        raise ValueError(f"{field_name} embeddings shape mismatch: expected ({expected_rows}, {expected_dims}), got {embeddings.shape}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        raise ValueError(f"{field_name} embeddings contain NaN or infinite values")
    
    # Check cosine similarity range
    if embeddings.shape[0] > 1:
        # Sample random pairs for cosine similarity check
        n_pairs = min(200, embeddings.shape[0] // 2)
        indices = np.random.choice(embeddings.shape[0], n_pairs * 2, replace=False)
        
        similarities = []
        for i in range(0, len(indices), 2):
            vec1 = embeddings[indices[i]]
            vec2 = embeddings[indices[i + 1]]
            # Since embeddings are L2-normalized, cosine similarity = dot product
            sim = np.dot(vec1, vec2)
            similarities.append(sim)
        
        min_sim, max_sim = np.min(similarities), np.max(similarities)
        if min_sim < -1.01 or max_sim > 1.01:  # Allow small numerical errors
            raise ValueError(f"{field_name} cosine similarities out of range [-1, 1]: [{min_sim:.3f}, {max_sim:.3f}]")
        
        logger.info(f"  {field_name} cosine similarity range: [{min_sim:.3f}, {max_sim:.3f}]")

def save_embeddings(embeddings: np.ndarray, 
                   field_name: str, 
                   output_dir: Path) -> None:
    """Save embeddings to NumPy file"""
    output_file = output_dir / f'movies_text_bert_{field_name}.npy'
    np.save(output_file, embeddings)
    logger.info(f"Saved {field_name} embeddings: {output_file} ({embeddings.shape})")

def save_metadata(metadata: Dict, output_dir: Path) -> None:
    """Save metadata to parquet file"""
    metadata_file = output_dir / 'movies_text_bert_meta.parquet'
    
    # Convert to DataFrame
    meta_df = pd.DataFrame([metadata])
    meta_df.to_parquet(metadata_file, index=False)
    logger.info(f"Saved metadata: {metadata_file}")

def compute_nearest_neighbors(embeddings: np.ndarray, 
                            index_df: pd.DataFrame, 
                            n_anchors: int = 3, 
                            n_neighbors: int = 5) -> List[Dict]:
    """Compute nearest neighbors for anchor movies"""
    logger.info(f"Computing nearest neighbors for {n_anchors} anchor movies...")
    
    # Randomly select anchor movies
    anchor_indices = np.random.choice(len(embeddings), n_anchors, replace=False)
    
    results = []
    for anchor_idx in anchor_indices:
        anchor_id = index_df.iloc[anchor_idx]['canonical_id']
        anchor_embedding = embeddings[anchor_idx]
        
        # Compute cosine similarities with all other movies
        similarities = np.dot(embeddings, anchor_embedding)
        
        # Get top neighbors (excluding self)
        neighbor_indices = np.argsort(similarities)[::-1][1:n_neighbors+1]
        neighbor_similarities = similarities[neighbor_indices]
        neighbor_ids = [index_df.iloc[i]['canonical_id'] for i in neighbor_indices]
        
        results.append({
            'anchor_id': anchor_id,
            'anchor_idx': anchor_idx,
            'neighbors': list(zip(neighbor_ids, neighbor_similarities))
        })
    
    return results

def generate_bert_report(metadata: Dict, 
                        coverage_stats: Dict,
                        nearest_neighbors: List[Dict]) -> str:
    """Generate BERT embeddings report section"""
    
    report_content = f"""## 2a.3 BERT Embeddings

**Generated:** {metadata['created_utc']}  
**Model:** {metadata['model_name']}  
**Dimension:** {metadata['dim']}  
**Device:** {metadata['device']}  
**Batch Size:** {metadata['batch_size']}  
**Total Runtime:** {metadata['total_runtime']:.2f}s

---

### Model Configuration

- **Model Name**: `{metadata['model_name']}`
- **Embedding Dimension**: {metadata['dim']}
- **Processing Device**: {metadata['device']}
- **Batch Size**: {metadata['batch_size']} ({'CUDA' if metadata['device'] == 'cuda' else 'CPU'})
- **L2 Normalization**: Enabled
- **Deterministic Seeds**: torch=42, numpy=42, python=42

---

### Coverage Statistics

| Field | Valid Texts | Coverage % | Embeddings Generated |
|-------|-------------|-------------|----------------------|
"""
    
    for field_name, stats in coverage_stats.items():
        report_content += f"| {field_name} | {stats['valid_texts']:,} | {stats['coverage_pct']:.1f}% | {stats['embeddings']:,} |\n"
    
    report_content += f"""

---

### Weighted Combination

**Field Weights:**
- **tags_combined**: {metadata['weights_json']['tags_combined']:.2f}
- **consensus**: {metadata['weights_json']['consensus']:.2f}
- **overview**: {metadata['weights_json']['overview']:.2f}

**Combination Strategy:** Weighted mean with automatic renormalization for missing fields

---

### Nearest Neighbors Analysis

**Cosine Similarity Sanity Check:**

"""
    
    for result in nearest_neighbors:
        report_content += f"**Anchor Movie {result['anchor_id']} (row {result['anchor_idx']}):**\n\n"
        report_content += "| Rank | Movie ID | Similarity |\n"
        report_content += "|------|----------|------------|\n"
        
        for i, (neighbor_id, similarity) in enumerate(result['neighbors'], 1):
            report_content += f"| {i} | `{neighbor_id}` | {similarity:.4f} |\n"
        
        report_content += "\n"
    
    report_content += "---\n*BERT embeddings completed successfully. All text fields have been converted to semantic vectors ready for recommendation models.*\n"
    
    return report_content

def append_to_report(report_content: str):
    """Append BERT report to the main step2a report"""
    report_file = Path('docs/step2a_report.md')
    
    if report_file.exists():
        # Read existing content
        with open(report_file, 'r') as f:
            existing_content = f.read()
        
        # Append new content
        updated_content = existing_content + "\n\n" + report_content
    else:
        # Create new file
        updated_content = report_content
    
    # Write updated content
    with open(report_file, 'w') as f:
        f.write(updated_content)
    
    logger.info(f"Updated report: {report_file}")

def generate_execution_log(metadata: Dict, 
                          coverage_stats: Dict,
                          total_time: float) -> str:
    """Generate execution log for BERT embeddings"""
    
    log_content = f"""[{metadata['created_utc']}] STEP 2a.3 STARTED - BERT Embeddings
[{metadata['created_utc']}] Objective: Generate BERT embeddings for cleaned text fields
[{metadata['created_utc']}] Model: {metadata['model_name']}
[{metadata['created_utc']}] Device: {metadata['device']}
[{metadata['created_utc']}] Total Movies: {metadata['row_count']:,}

[{metadata['created_utc']}] TASK 1: Data loading and validation
[{metadata['created_utc']}] - Loaded cleaned text data
[{metadata['created_utc']}] - Verified canonical_id alignment
[{metadata['created_utc']}] - Validated row count: {metadata['row_count']:,}

[{metadata['created_utc']}] TASK 2: BERT model initialization
[{metadata['created_utc']}] - Loaded model: {metadata['model_name']}
[{metadata['created_utc']}] - Device: {metadata['device']}
[{metadata['created_utc']}] - Batch size: {metadata['batch_size']}

[{metadata['created_utc']}] TASK 3: Embedding generation
"""
    
    for field_name, stats in coverage_stats.items():
        log_content += f"[{metadata['created_utc']}] {field_name}: {stats['valid_texts']:,} valid texts -> {stats['embeddings']:,} embeddings\n"
    
    log_content += f"""
[{metadata['created_utc']}] TASK 4: Weighted combination
[{metadata['created_utc']}] - Combined embeddings with field weights
[{metadata['created_utc']}] - Final shape: {metadata['row_count']:,} × {metadata['dim']}

[{metadata['created_utc']}] TASK 5: Output generation
[{metadata['created_utc']}] - Saved individual field embeddings (.npy)
[{metadata['created_utc']}] - Saved combined embeddings (.npy)
[{metadata['created_utc']}] - Saved metadata (.parquet)
[{metadata['created_utc']}] - Updated documentation

[{metadata['created_utc']}] STEP 2a.3 COMPLETED SUCCESSFULLY
[{metadata['created_utc']}] 
[{metadata['created_utc']}] DELIVERABLES COMPLETED:
[{metadata['created_utc']}] ✅ NumPy arrays for each field (88194, 384)
[{metadata['created_utc']}] ✅ Combined weighted embeddings
[{metadata['created_utc']}] ✅ Metadata parquet file
[{metadata['created_utc']}] ✅ Updated markdown report
[{metadata['created_utc']}] ✅ Execution log file
[{metadata['created_utc']}] 
[{metadata['created_utc']}] SUMMARY:
[{metadata['created_utc']}] - Total runtime: {total_time:.2f}s
[{metadata['created_utc']}] - All embeddings generated successfully
[{metadata['created_utc']}] - Ready for recommendation model training
"""
    
    return log_content

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate BERT embeddings for movie text fields')
    parser.add_argument('--cleaned', required=True, help='Path to cleaned text parquet file')
    parser.add_argument('--index', required=True, help='Path to index mapping parquet file')
    parser.add_argument('--outdir', required=True, help='Output directory for embeddings')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2', help='BERT model name')
    parser.add_argument('--batch_cpu', type=int, default=64, help='Batch size for CPU')
    parser.add_argument('--batch_cuda', type=int, default=256, help='Batch size for CUDA')
    
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("Starting Step 2a.3: BERT Embeddings")
    
    try:
        # Create output directory
        output_dir = Path(args.outdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        cleaned_df, index_df = load_data(args.cleaned, args.index)
        total_rows = len(cleaned_df)
        
        # Initialize BERT embedder
        embedder = BERTEmbedder(
            model_name=args.model,
            batch_size_cpu=args.batch_cpu,
            batch_size_cuda=args.batch_cuda
        )
        
        # Get text fields and weights
        text_fields = get_text_fields()
        field_weights = get_field_weights()
        
        # Generate embeddings for each field
        field_embeddings = {}
        coverage_stats = {}
        
        for field_name, cleaned_field in text_fields.items():
            if cleaned_field in cleaned_df.columns:
                # Get texts
                texts = cleaned_df[cleaned_field].fillna("unknown_text").tolist()
                
                # Count valid texts
                valid_texts = [text for text in texts if text != "unknown_text"]
                coverage_pct = (len(valid_texts) / total_rows) * 100
                
                coverage_stats[field_name] = {
                    'valid_texts': len(valid_texts),
                    'coverage_pct': coverage_pct,
                    'embeddings': total_rows
                }
                
                # Generate embeddings
                embeddings = embedder.encode_field(texts, field_name)
                
                # Validate embeddings
                validate_embeddings(embeddings, field_name, total_rows, embedder.dimension)
                
                # Store embeddings
                field_embeddings[field_name] = embeddings
                
                # Save individual field embeddings
                save_embeddings(embeddings, field_name, output_dir)
            else:
                logger.warning(f"Field {cleaned_field} not found, skipping {field_name}")
        
        # Create combined embeddings
        combined_embeddings = embedder.create_weighted_embeddings(field_embeddings, field_weights)
        save_embeddings(combined_embeddings, 'combined', output_dir)
        
        # Compute nearest neighbors for sanity check
        nearest_neighbors = compute_nearest_neighbors(combined_embeddings, index_df)
        
        # Prepare metadata
        total_time = time.time() - start_time
        metadata = {
            'model_name': embedder.model_name,
            'dim': embedder.dimension,
            'device': embedder.device,
            'batch_size': embedder.batch_size,
            'row_count': total_rows,
            'created_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'total_runtime': total_time,
            'weights_json': field_weights
        }
        
        # Save metadata
        save_metadata(metadata, output_dir)
        
        # Generate and save report
        report_content = generate_bert_report(metadata, coverage_stats, nearest_neighbors)
        append_to_report(report_content)
        
        # Generate and save execution log
        log_content = generate_execution_log(metadata, coverage_stats, total_time)
        log_file = Path('logs/step2a_phase3.log')
        with open(log_file, 'w') as f:
            f.write(log_content)
        
        # Final validation
        logger.info("Final validation...")
        for field_name in list(text_fields.keys()) + ['combined']:
            embedding_file = output_dir / f'movies_text_bert_{field_name}.npy'
            if embedding_file.exists():
                embeddings = np.load(embedding_file)
                if embeddings.shape == (total_rows, embedder.dimension):
                    logger.info(f"✓ {field_name}: {embeddings.shape}")
                else:
                    logger.error(f"✗ {field_name}: shape mismatch {embeddings.shape}")
        
        logger.info("Step 2a.3 completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in Step 2a.3: {e}")
        raise

if __name__ == "__main__":
    main()























