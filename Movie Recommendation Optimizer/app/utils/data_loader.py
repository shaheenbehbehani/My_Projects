"""
Data loader utilities for runtime artifacts.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Optional, Tuple, Dict, Any
import streamlit as st

# Data directory relative to app root
APP_ROOT = Path(__file__).parent.parent
DATA_DIR = APP_ROOT / "data"

@st.cache_data
def load_id_map() -> Optional[pd.DataFrame]:
    """Load movie ID mapping."""
    path = DATA_DIR / "id_map.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)

@st.cache_resource
def load_tfidf_vectorizer():
    """Load TF-IDF vectorizer."""
    path = DATA_DIR / "tfidf_title.pkl"
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_data
def load_movie_factors() -> Optional[Tuple[np.ndarray, pd.DataFrame]]:
    """Load movie factors and index mapping."""
    factors_path = DATA_DIR / "movie_factors.npy"
    idx_path = DATA_DIR / "movie_factor_index.parquet"
    
    if not factors_path.exists() or not idx_path.exists():
        return None
    
    factors = np.load(factors_path)
    idx_map = pd.read_parquet(idx_path)
    return factors, idx_map

def check_artifacts_available() -> bool:
    """Check if required artifacts are available."""
    required = [
        DATA_DIR / "id_map.parquet",
        DATA_DIR / "tfidf_title.pkl"
    ]
    return all(p.exists() for p in required)

def search_movies(query: str, id_map: pd.DataFrame, vectorizer, top_k: int = 10) -> pd.DataFrame:
    """Search movies using TF-IDF similarity."""
    if vectorizer is None or id_map is None:
        return pd.DataFrame()
    
    # Vectorize query
    query_vec = vectorizer.transform([query.lower()])
    
    # Load pre-computed TF-IDF matrix (we'll compute on-the-fly for now)
    # For production, we'd cache the full matrix, but for now compute similarity
    search_text = id_map.apply(
        lambda row: f"{row['title_norm']} {row['genres'].replace('|', ' ').lower() if pd.notna(row['genres']) else ''}",
        axis=1
    ).tolist()
    
    doc_vecs = vectorizer.transform(search_text)
    
    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_vec, doc_vecs).flatten()
    
    # Get top K
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = id_map.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    
    return results

