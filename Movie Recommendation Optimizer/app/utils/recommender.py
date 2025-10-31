"""
Recommendation engine using runtime artifacts.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from .data_loader import load_id_map, load_tfidf_vectorizer, load_movie_factors, search_movies

def get_recommendations(
    title_query: str = "",
    genres: Optional[List[str]] = None,
    year_range: Optional[Tuple[int, int]] = None,
    top_k: int = 10
) -> List[Dict]:
    """Get movie recommendations based on query."""
    # Load artifacts
    id_map = load_id_map()
    vectorizer = load_tfidf_vectorizer()
    factors_result = load_movie_factors()
    
    if id_map is None or vectorizer is None:
        return []
    
    # Build search query from title + genres
    query_parts = []
    if title_query:
        query_parts.append(title_query.lower())
    if genres:
        query_parts.extend([g.lower() for g in genres])
    
    query = " ".join(query_parts) if query_parts else ""
    
    if not query:
        # No query - return popular movies (top by year if available)
        results = id_map.copy()
        if 'year' in results.columns:
            results = results[results['year'].notna()]
            results = results.sort_values('year', ascending=False)
        else:
            results = results.head(100)
        results = results.head(top_k)
    else:
        # Search using TF-IDF
        results = search_movies(query, id_map, vectorizer, top_k=top_k * 2)
        
        # Apply filters
        if year_range:
            results = results[
                (results['year'] >= year_range[0]) & 
                (results['year'] <= year_range[1])
            ]
        
        if genres:
            # Filter by genres (check if any genre matches)
            genre_filter = results['genres'].apply(
                lambda g: any(genre.lower() in str(g).lower() for genre in genres)
                if pd.notna(g) else False
            )
            results = results[genre_filter]
        
        results = results.head(top_k)
    
    # Format results
    recommendations = []
    for _, row in results.iterrows():
        rec = {
            'title': row['title'],
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'genres': row['genres'] if pd.notna(row['genres']) else '',
        }
        if 'similarity' in row:
            rec['score'] = float(row['similarity'])
        recommendations.append(rec)
    
    return recommendations

