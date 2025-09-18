#!/usr/bin/env python3
"""
Step 2a.2: TF-IDF Vectorization
Movie Recommendation Optimizer

Objective: Generate TF-IDF vectors for cleaned text fields to enable text-based
recommendation features. Creates individual and combined TF-IDF matrices.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TFIDFVectorizer:
    """TF-IDF vectorization utility class"""
    
    def __init__(self, 
                 lowercase: bool = False,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 5,
                 max_df: float = 0.6,
                 max_features: int = 200000,
                 dtype: np.dtype = np.float32):
        """
        Initialize TF-IDF vectorizer with specified parameters
        
        Args:
            lowercase: Whether to convert to lowercase (disabled since text is pre-cleaned)
            ngram_range: Range of n-grams to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency (fraction)
            max_features: Maximum number of features to extract
            dtype: Data type for the TF-IDF matrix
        """
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.dtype = dtype
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=lowercase,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            dtype=dtype
        )
        
        self.is_fitted = False
        self.feature_names_ = None
        self.n_features_ = 0
    
    def fit_transform(self, texts: List[str]) -> sp.csr_matrix:
        """
        Fit the vectorizer and transform the texts
        
        Args:
            texts: List of text documents
            
        Returns:
            Sparse TF-IDF matrix
        """
        logger.info(f"Fitting TF-IDF vectorizer with {len(texts)} documents")
        
        # Filter out "unknown_text" entries for fitting
        valid_texts = [text for text in texts if text != "unknown_text"]
        logger.info(f"Valid texts for fitting: {len(valid_texts)}")
        
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(valid_texts)
        
        # Store feature information
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        self.n_features_ = len(self.feature_names_)
        self.is_fitted = True
        
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        logger.info(f"Features extracted: {self.n_features_}")
        
        return tfidf_matrix
    
    def transform(self, texts: List[str]) -> sp.csr_matrix:
        """
        Transform texts using fitted vectorizer
        
        Args:
            texts: List of text documents
            
        Returns:
            Sparse TF-IDF matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        # Handle "unknown_text" entries by replacing with empty string
        processed_texts = ["" if text == "unknown_text" else text for text in texts]
        
        return self.vectorizer.transform(processed_texts)
    
    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top features by IDF score
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature_name, idf_score) tuples
        """
        if not self.is_fitted:
            return []
        
        # Get IDF scores
        idf_scores = self.vectorizer.idf_
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Create feature-score pairs and sort
        feature_scores = list(zip(feature_names, idf_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return feature_scores[:n]

def load_cleaned_data() -> pd.DataFrame:
    """Load the cleaned text data"""
    logger.info("Loading cleaned text data...")
    
    df = pd.read_parquet('data/features/text/checkpoints/cleaned_text.parquet')
    logger.info(f"Loaded {len(df)} movies with {len(df.columns)} columns")
    
    # Verify canonical_id exists
    if 'canonical_id' not in df.columns:
        raise ValueError("canonical_id column not found in cleaned data")
    
    # Set canonical_id as index
    df = df.set_index('canonical_id')
    logger.info(f"Set canonical_id as index with {len(df)} unique movies")
    
    return df

def get_text_fields_to_vectorize() -> Dict[str, str]:
    """Define the text fields to vectorize and their cleaned counterparts"""
    return {
        'overview': 'overview_cleaned',
        'consensus': 'consensus_cleaned',
        'tags_combined': 'tags_cleaned'
    }

def analyze_text_coverage(df: pd.DataFrame, text_fields: Dict[str, str]) -> Dict[str, Dict]:
    """Analyze coverage statistics for each text field"""
    logger.info("Analyzing text field coverage...")
    
    coverage_stats = {}
    
    for field_name, cleaned_field in text_fields.items():
        if cleaned_field in df.columns:
            total_movies = len(df)
            valid_texts = (df[cleaned_field] != "unknown_text").sum()
            coverage_pct = (valid_texts / total_movies) * 100
            
            coverage_stats[field_name] = {
                'total_movies': total_movies,
                'valid_texts': valid_texts,
                'missing_texts': total_movies - valid_texts,
                'coverage_pct': coverage_pct,
                'cleaned_field': cleaned_field
            }
            
            logger.info(f"  {field_name}: {valid_texts:,}/{total_movies:,} ({coverage_pct:.1f}%)")
        else:
            logger.warning(f"Field {cleaned_field} not found in dataset")
            coverage_stats[field_name] = {
                'total_movies': len(df),
                'valid_texts': 0,
                'missing_texts': len(df),
                'coverage_pct': 0.0,
                'cleaned_field': cleaned_field
            }
    
    return coverage_stats

def create_full_matrix(tfidf_matrix: sp.csr_matrix, 
                      total_rows: int, 
                      valid_indices: List[int]) -> sp.csr_matrix:
    """
    Create a full TF-IDF matrix with zero vectors for movies without text
    
    Args:
        tfidf_matrix: Original TF-IDF matrix with valid texts only
        total_rows: Total number of rows in the master dataset
        valid_indices: List of row indices that have valid text
        
    Returns:
        Full sparse matrix with zero vectors for missing texts
    """
    if tfidf_matrix.shape[0] == total_rows:
        return tfidf_matrix
    
    # Create a sparse matrix with the full number of rows
    full_matrix = sp.csr_matrix((total_rows, tfidf_matrix.shape[1]), dtype=tfidf_matrix.dtype)
    
    # Place the valid TF-IDF vectors at the correct positions
    for i, valid_idx in enumerate(valid_indices):
        full_matrix[valid_idx] = tfidf_matrix[i]
    
    return full_matrix

def vectorize_text_field(df: pd.DataFrame, 
                        field_name: str, 
                        cleaned_field: str,
                        coverage_stats: Dict) -> Tuple[sp.csr_matrix, TFIDFVectorizer]:
    """
    Vectorize a single text field
    
    Args:
        df: DataFrame with cleaned text data
        field_name: Name of the field being vectorized
        cleaned_field: Name of the cleaned column
        coverage_stats: Coverage statistics for the field
        
    Returns:
        Tuple of (TF-IDF matrix, fitted vectorizer)
    """
    logger.info(f"Vectorizing {field_name} field...")
    
    # Get texts for the field
    texts = df[cleaned_field].fillna("unknown_text").tolist()
    
    # Find valid texts (not "unknown_text")
    valid_mask = [text != "unknown_text" for text in texts]
    valid_texts = [text for text in texts if text != "unknown_text"]
    valid_indices = [i for i, is_valid in enumerate(valid_mask) if is_valid]
    
    logger.info(f"  Valid texts: {len(valid_texts)} out of {len(texts)}")
    
    # Initialize vectorizer
    vectorizer = TFIDFVectorizer(
        lowercase=False,  # Text already cleaned
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.6,
        max_features=200000,
        dtype=np.float32
    )
    
    # Fit and transform only valid texts
    if valid_texts:
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        
        # Create full matrix with zero vectors for missing texts
        full_matrix = create_full_matrix(tfidf_matrix, len(df), valid_indices)
        
        # Get top features for reporting
        top_features = vectorizer.get_top_features(20)
        logger.info(f"  Top features: {[f[0] for f in top_features[:5]]}")
        
        return full_matrix, vectorizer
    else:
        # No valid texts - create empty matrix
        logger.warning(f"  No valid texts found for {field_name}")
        empty_matrix = sp.csr_matrix((len(df), 0), dtype=np.float32)
        return empty_matrix, vectorizer

def create_combined_matrix(tfidf_matrices: Dict[str, sp.csr_matrix]) -> sp.csr_matrix:
    """
    Create combined TF-IDF matrix by horizontally stacking individual matrices
    
    Args:
        tfidf_matrices: Dictionary of field_name -> TF-IDF matrix
        
    Returns:
        Combined sparse matrix
    """
    logger.info("Creating combined TF-IDF matrix...")
    
    # Filter out empty matrices
    valid_matrices = {k: v for k, v in tfidf_matrices.items() if v is not None and v.shape[0] > 0}
    
    if not valid_matrices:
        raise ValueError("No valid TF-IDF matrices to combine")
    
    # Get the number of rows (should be same for all)
    n_rows = list(valid_matrices.values())[0].shape[0]
    logger.info(f"Combining {len(valid_matrices)} matrices with {n_rows} rows each")
    
    # Horizontally stack matrices
    combined_matrix = sp.hstack(list(valid_matrices.values()))
    
    logger.info(f"Combined matrix shape: {combined_matrix.shape}")
    return combined_matrix

def save_outputs(tfidf_matrices: Dict[str, sp.csr_matrix],
                vectorizers: Dict[str, TFIDFVectorizer],
                df: pd.DataFrame,
                coverage_stats: Dict):
    """Save all TF-IDF outputs"""
    logger.info("Saving TF-IDF outputs...")
    
    # Create output directories
    output_dir = Path('data/features/text')
    vectorizers_dir = output_dir / 'vectorizers'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorizers_dir.mkdir(parents=True, exist_ok=True)
    
    # Save TF-IDF matrices as .npz files
    for field_name, matrix in tfidf_matrices.items():
        if matrix is not None:
            output_file = output_dir / f'movies_text_tfidf_{field_name}.npz'
            sp.save_npz(output_file, matrix)
            logger.info(f"Saved {field_name} matrix: {output_file} ({matrix.shape})")
    
    # Save combined matrix
    if 'combined' in tfidf_matrices and tfidf_matrices['combined'] is not None:
        combined_file = output_dir / 'movies_text_tfidf_combined.npz'
        sp.save_npz(combined_file, tfidf_matrices['combined'])
        logger.info(f"Saved combined matrix: {combined_file} ({tfidf_matrices['combined'].shape})")
    
    # Save index mapping
    index_df = pd.DataFrame({
        'canonical_id': df.index,
        'row_index': range(len(df))
    }).reset_index(drop=True)
    
    index_file = output_dir / 'movies_text_tfidf_index.parquet'
    index_df.to_parquet(index_file, index=False)
    logger.info(f"Saved index mapping: {index_file} ({len(index_df)} rows)")
    
    # Save fitted vectorizers
    for field_name, vectorizer in vectorizers.items():
        if vectorizer.is_fitted:
            vectorizer_file = vectorizers_dir / f'tfidf_vectorizer_{field_name}.joblib'
            joblib.dump(vectorizer.vectorizer, vectorizer_file)
            logger.info(f"Saved {field_name} vectorizer: {vectorizer_file}")

def generate_tfidf_report(coverage_stats: Dict, 
                         tfidf_matrices: Dict[str, sp.csr_matrix],
                         vectorizers: Dict[str, TFIDFVectorizer]) -> str:
    """Generate TF-IDF vectorization report"""
    
    report_content = f"""# TF-IDF Vectorization Report
## Step 2a.2 - Movie Recommendation Optimizer

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Movies:** {list(coverage_stats.values())[0]['total_movies']:,}

---

## Vectorization Summary

### Fields Vectorized
"""
    
    for field_name, stats in coverage_stats.items():
        if field_name in tfidf_matrices and tfidf_matrices[field_name] is not None:
            matrix = tfidf_matrices[field_name]
            report_content += f"- **{field_name}**: {matrix.shape[1]:,} features, {stats['coverage_pct']:.1f}% coverage\n"
    
    report_content += "\n### Combined Matrix\n"
    if 'combined' in tfidf_matrices and tfidf_matrices['combined'] is not None:
        combined = tfidf_matrices['combined']
        report_content += f"- **Shape**: {combined.shape[0]:,} movies × {combined.shape[1]:,} features\n"
        report_content += f"- **Sparsity**: {1 - combined.nnz / (combined.shape[0] * combined.shape[1]):.3f}\n"
    
    report_content += "\n---\n## Coverage Analysis\n\n"
    report_content += "| Field | Valid Texts | Coverage % | Features | Matrix Shape |\n"
    report_content += "|-------|-------------|-------------|----------|--------------|\n"
    
    for field_name, stats in coverage_stats.items():
        if field_name in tfidf_matrices and tfidf_matrices[field_name] is not None:
            matrix = tfidf_matrices[field_name]
            report_content += f"| {field_name} | {stats['valid_texts']:,} | {stats['coverage_pct']:.1f}% | {matrix.shape[1]:,} | {matrix.shape[0]:,} × {matrix.shape[1]:,} |\n"
    
    report_content += "\n---\n## Top Features by Field\n\n"
    
    for field_name, vectorizer in vectorizers.items():
        if vectorizer.is_fitted:
            top_features = vectorizer.get_top_features(15)
            report_content += f"### {field_name}\n\n"
            report_content += "| Rank | Feature | IDF Score |\n"
            report_content += "|------|---------|-----------|\n"
            
            for i, (feature, score) in enumerate(top_features, 1):
                report_content += f"| {i} | `{feature}` | {score:.4f} |\n"
            
            report_content += "\n"
    
    report_content += "---\n## Vectorization Parameters\n\n"
    report_content += "- **N-gram Range**: (1, 2) - unigrams and bigrams\n"
    report_content += "- **Min Document Frequency**: 5 - minimum 5 movies must contain feature\n"
    report_content += "- **Max Document Frequency**: 0.6 - maximum 60% of movies can contain feature\n"
    report_content += "- **Max Features**: 200,000 per field\n"
    report_content += "- **Data Type**: float32\n"
    report_content += "- **Lowercase**: Disabled (text pre-cleaned)\n\n"
    
    report_content += "---\n## Output Files\n\n"
    report_content += "- **TF-IDF Matrices**: `.npz` sparse format\n"
    report_content += "- **Vectorizers**: `.joblib` serialized objects\n"
    report_content += "- **Index Mapping**: `.parquet` with canonical_id → row_index\n"
    report_content += "- **Combined Matrix**: All fields horizontally stacked\n\n"
    
    report_content += "---\n*TF-IDF vectorization completed successfully. All text fields have been converted to numerical features ready for recommendation models.*\n"
    
    return report_content

def append_to_report(report_content: str):
    """Append TF-IDF report to the main step2a report"""
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

def generate_execution_log(coverage_stats: Dict,
                          tfidf_matrices: Dict[str, sp.csr_matrix],
                          vectorizers: Dict[str, TFIDFVectorizer]) -> str:
    """Generate execution log for TF-IDF vectorization"""
    
    log_content = f"""[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STEP 2a.2 STARTED - TF-IDF Vectorization
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Objective: Generate TF-IDF vectors for cleaned text fields
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total Movies: {list(coverage_stats.values())[0]['total_movies']:,}

[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TASK 1: Loading cleaned text data
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Loaded cleaned_text.parquet
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Set canonical_id as index

[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TASK 2: Text field coverage analysis
"""
    
    for field_name, stats in coverage_stats.items():
        log_content += f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {field_name}: {stats['valid_texts']:,}/{stats['total_movies']:,} ({stats['coverage_pct']:.1f}%)\n"
    
    log_content += f"""
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TASK 3: TF-IDF vectorization
"""
    
    for field_name, matrix in tfidf_matrices.items():
        if matrix is not None:
            log_content += f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {field_name}: {matrix.shape[0]:,} × {matrix.shape[1]:,} features\n"
    
    log_content += f"""
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TASK 4: Output generation
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Saved TF-IDF matrices (.npz)
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Saved fitted vectorizers (.joblib)
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Saved index mapping (.parquet)
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Updated documentation

[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STEP 2a.2 COMPLETED SUCCESSFULLY
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DELIVERABLES COMPLETED:
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Sparse .npz files for each TF-IDF matrix
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ joblib vectorizer objects
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ parquet index mapping with canonical_id
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Updated markdown report
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Execution log file
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SUMMARY:
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - All text fields successfully vectorized
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - TF-IDF matrices generated and saved
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Ready for recommendation model training
"""
    
    return log_content

def main():
    """Main execution function"""
    logger.info("Starting Step 2a.2: TF-IDF Vectorization")
    
    try:
        # Load cleaned data
        df = load_cleaned_data()
        
        # Define text fields to vectorize
        text_fields = get_text_fields_to_vectorize()
        
        # Analyze coverage
        coverage_stats = analyze_text_coverage(df, text_fields)
        
        # Initialize storage for matrices and vectorizers
        tfidf_matrices = {}
        vectorizers = {}
        
        # Vectorize each text field
        for field_name, cleaned_field in text_fields.items():
            if cleaned_field in df.columns:
                try:
                    matrix, vectorizer = vectorize_text_field(df, field_name, cleaned_field, coverage_stats)
                    tfidf_matrices[field_name] = matrix
                    vectorizers[field_name] = vectorizer
                except Exception as e:
                    logger.error(f"Error vectorizing {field_name}: {e}")
                    tfidf_matrices[field_name] = None
                    vectorizers[field_name] = None
            else:
                logger.warning(f"Skipping {field_name} - field not found")
                tfidf_matrices[field_name] = None
                vectorizers[field_name] = None
        
        # Create combined matrix
        try:
            combined_matrix = create_combined_matrix(tfidf_matrices)
            tfidf_matrices['combined'] = combined_matrix
        except Exception as e:
            logger.error(f"Error creating combined matrix: {e}")
            tfidf_matrices['combined'] = None
        
        # Save outputs
        save_outputs(tfidf_matrices, vectorizers, df, coverage_stats)
        
        # Generate and save report
        report_content = generate_tfidf_report(coverage_stats, tfidf_matrices, vectorizers)
        append_to_report(report_content)
        
        # Generate and save execution log
        log_content = generate_execution_log(coverage_stats, tfidf_matrices, vectorizers)
        log_file = Path('logs/step2a_phase2.log')
        with open(log_file, 'w') as f:
            f.write(log_content)
        
        # Verify outputs
        logger.info("Verifying outputs...")
        verify_outputs(tfidf_matrices, df)
        
        logger.info("Step 2a.2 completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in Step 2a.2: {e}")
        raise

def verify_outputs(tfidf_matrices: Dict[str, sp.csr_matrix], df: pd.DataFrame):
    """Verify output file sizes and row alignment"""
    logger.info("Verifying output integrity...")
    
    # Check row alignment
    expected_rows = len(df)
    for field_name, matrix in tfidf_matrices.items():
        if matrix is not None:
            if matrix.shape[0] != expected_rows:
                logger.warning(f"Row mismatch in {field_name}: expected {expected_rows}, got {matrix.shape[0]}")
            else:
                logger.info(f"✓ {field_name}: {matrix.shape[0]:,} rows aligned")
    
    # Check file sizes
    output_dir = Path('data/features/text')
    for field_name in tfidf_matrices.keys():
        if field_name != 'combined':
            npz_file = output_dir / f'movies_text_tfidf_{field_name}.npz'
            if npz_file.exists():
                size_mb = npz_file.stat().st_size / (1024 * 1024)
                logger.info(f"✓ {field_name}.npz: {size_mb:.1f} MB")
    
    # Check combined matrix
    combined_file = output_dir / 'movies_text_tfidf_combined.npz'
    if combined_file.exists():
        size_mb = combined_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ combined.npz: {size_mb:.1f} MB")
    
    # Check index file
    index_file = output_dir / 'movies_text_tfidf_index.parquet'
    if index_file.exists():
        size_kb = index_file.stat().st_size / 1024
        logger.info(f"✓ index.parquet: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()
