#!/usr/bin/env python3
"""
Step 1b Phase 4: Genres & Taxonomy
Standardizes and enriches genre information across all movies.
Delivers normalized lists, multi-hot encodings, and taxonomy reports.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import re
from collections import Counter

# Setup logging
log_file = 'logs/step1b_phase4.log'
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)

def setup_directories():
    """Ensure output directories exist"""
    os.makedirs('data/normalized', exist_ok=True)
    os.makedirs('data/features/genres', exist_ok=True)
    os.makedirs('docs', exist_ok=True)
    logging.info("Directories setup complete")

def load_master_data():
    """Load the master movies table"""
    logging.info("Loading master movies table...")
    master_path = "data/normalized/movies_master.parquet"
    
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Master table not found: {master_path}")
    
    df = pd.read_parquet(master_path)
    logging.info(f"Loaded master table: {df.shape}")
    
    return df

def create_genre_mapping():
    """Create canonical genre mapping dictionary"""
    logging.info("Creating canonical genre mapping...")
    
    # Define canonical genres and their variants
    genre_mapping = {
        # Core genres
        'drama': ['drama', 'dramas', 'dramatic'],
        'comedy': ['comedy', 'comedies', 'comic', 'humor', 'humour', 'funny'],
        'action': ['action', 'action-adventure'],
        'romance': ['romance', 'romantic', 'rom-com', 'romcom'],
        'thriller': ['thriller', 'thrilling', 'suspense'],
        'horror': ['horror', 'horror film', 'scary'],
        'adventure': ['adventure', 'adventurous'],
        'crime': ['crime', 'criminal', 'gangster'],
        'mystery': ['mystery', 'mysterious', 'detective'],
        'documentary': ['documentary', 'doc', 'documentaries'],
        'sci-fi': ['sci-fi', 'scifi', 'science fiction', 'sf'],
        'fantasy': ['fantasy', 'fantastical'],
        'animation': ['animation', 'animated', 'cartoon'],
        'family': ['family', 'family-friendly'],
        'war': ['war', 'war film', 'military'],
        'western': ['western', 'cowboy'],
        'musical': ['musical', 'music'],
        'biography': ['biography', 'biographical', 'bio'],
        'history': ['history', 'historical'],
        'sport': ['sport', 'sports', 'athletic'],
        'news': ['news', 'current events'],
        'reality-tv': ['reality-tv', 'reality', 'reality show'],
        'game-show': ['game-show', 'game show', 'quiz'],
        'talk-show': ['talk-show', 'talk show'],
        'variety': ['variety', 'variety show'],
        'short': ['short', 'short film'],
        'film-noir': ['film-noir', 'noir', 'black and white'],
        'adult': ['adult', 'mature'],
        'unknown': ['unknown', 'none', 'n/a', 'nan', '']
    }
    
    # Create reverse mapping for quick lookups
    reverse_mapping = {}
    for canonical, variants in genre_mapping.items():
        for variant in variants:
            reverse_mapping[variant.lower()] = canonical
    
    logging.info(f"Created genre mapping with {len(genre_mapping)} canonical genres")
    return genre_mapping, reverse_mapping

def normalize_genre(genre, reverse_mapping):
    """Normalize a single genre to canonical form"""
    if pd.isna(genre) or genre == '':
        return 'unknown'
    
    # Convert to lowercase and strip whitespace
    genre_clean = str(genre).lower().strip()
    
    # Look up in reverse mapping
    if genre_clean in reverse_mapping:
        return reverse_mapping[genre_clean]
    
    # If not found, return as-is (will be handled later)
    return genre_clean

def process_genres(df, reverse_mapping):
    """Process and normalize all genres"""
    logging.info("Processing and normalizing genres...")
    
    # Convert numpy arrays to lists and normalize
    genres_processed = []
    genres_str_list = []
    
    for idx, genres_array in enumerate(df['genres_norm']):
        # Handle numpy arrays properly
        if genres_array is None or (hasattr(genres_array, '__len__') and len(genres_array) == 0):
            genres_processed.append(['unknown'])
            genres_str_list.append('unknown')
            continue
        
        # Convert numpy array to list and normalize each genre
        if hasattr(genres_array, '__iter__') and not isinstance(genres_array, str):
            genres_list = [normalize_genre(g, reverse_mapping) for g in genres_array]
        else:
            # Single genre case
            genres_list = [normalize_genre(genres_array, reverse_mapping)]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_genres = []
        for g in genres_list:
            if g not in seen:
                seen.add(g)
                unique_genres.append(g)
        
        # Ensure at least one genre
        if not unique_genres or (len(unique_genres) == 1 and unique_genres[0] == 'unknown'):
            unique_genres = ['unknown']
        
        genres_processed.append(unique_genres)
        genres_str_list.append('|'.join(unique_genres))
    
    logging.info(f"Processed genres for {len(genres_processed)} movies")
    return genres_processed, genres_str_list

def create_multihot_encoding(genres_processed, top_genres):
    """Create multi-hot encoding for top genres"""
    logging.info("Creating multi-hot encoding...")
    
    # Initialize multi-hot dataframe
    multihot_data = {}
    
    # Create binary columns for each top genre
    for genre in top_genres:
        multihot_data[f'genre_{genre}'] = [
            1 if genre in movie_genres else 0 
            for movie_genres in genres_processed
        ]
    
    # Create dataframe
    multihot_df = pd.DataFrame(multihot_data, dtype='int8')
    
    logging.info(f"Created multi-hot encoding with {len(top_genres)} genre columns")
    return multihot_df

def analyze_genre_coverage(genres_processed):
    """Analyze genre coverage and statistics"""
    logging.info("Analyzing genre coverage...")
    
    # Count all genres
    all_genres = []
    genre_counts = []
    
    for movie_genres in genres_processed:
        if movie_genres and movie_genres != ['unknown']:
            all_genres.extend(movie_genres)
            genre_counts.append(len(movie_genres))
        else:
            genre_counts.append(0)
    
    # Genre frequency
    genre_freq = Counter(all_genres)
    top_genres = [genre for genre, count in genre_freq.most_common(20)]
    
    # Coverage statistics
    movies_with_genres = sum(1 for count in genre_counts if count > 0)
    total_movies = len(genre_counts)
    coverage_pct = (movies_with_genres / total_movies) * 100
    
    # Genre count distribution
    genre_counts_series = pd.Series(genre_counts)
    
    analysis = {
        'total_movies': total_movies,
        'movies_with_genres': movies_with_genres,
        'coverage_pct': coverage_pct,
        'top_genres': top_genres,
        'genre_frequencies': dict(genre_freq.most_common(20)),
        'genre_count_stats': {
            'min': int(genre_counts_series.min()),
            'median': float(genre_counts_series.median()),
            'max': int(genre_counts_series.max()),
            'mean': float(genre_counts_series.mean())
        }
    }
    
    logging.info(f"Genre coverage: {coverage_pct:.1f}% ({movies_with_genres:,}/{total_movies:,})")
    logging.info(f"Top 5 genres: {top_genres[:5]}")
    
    return analysis, top_genres

def save_outputs(genres_df, multihot_df, genre_mapping, analysis):
    """Save all output files"""
    logging.info("=== SAVING OUTPUTS ===")
    
    # 1. Save normalized genres parquet
    output_path = "data/normalized/movies_genres.parquet"
    genres_df.to_parquet(output_path, index=True)
    logging.info(f"Saved normalized genres: {output_path}")
    
    # 2. Save multi-hot encoding
    multihot_path = "data/features/genres/movies_genres_multihot.parquet"
    multihot_df.to_parquet(multihot_path, index=True)
    logging.info(f"Saved multi-hot encoding: {multihot_path}")
    
    # 3. Save genre taxonomy JSON
    taxonomy = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "phase": "Step 1b Phase 4: Genres & Taxonomy",
            "description": "Canonical genre mapping and taxonomy rules"
        },
        "genre_mapping": genre_mapping,
        "analysis": analysis,
        "notes": [
            "Genres are normalized to lowercase, singular forms",
            "Duplicates are removed while preserving order",
            "Movies without genres are marked as 'unknown'",
            "Multi-hot encoding covers top 20 most frequent genres"
        ]
    }
    
    taxonomy_path = "docs/genre_taxonomy.json"
    with open(taxonomy_path, 'w') as f:
        json.dump(taxonomy, f, indent=2)
    logging.info(f"Saved genre taxonomy: {taxonomy_path}")
    
    # 4. Update step1b report
    update_step1b_report(analysis)
    
    return {
        'genres': output_path,
        'multihot': multihot_path,
        'taxonomy': taxonomy_path
    }

def update_step1b_report(analysis):
    """Append Phase 4 results to step1b report"""
    logging.info("Updating step1b report...")
    
    report_path = "docs/step1b_report.md"
    
    # Read existing report or create new
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            content = f.read()
    else:
        content = "# Step 1b Report\n\n"
    
    # Add Phase 4 section
    phase4_section = f"""
## Phase 4 — Genres & Taxonomy

**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Overview
Standardized and enriched genre information across all movies, delivering normalized lists, multi-hot encodings, and comprehensive taxonomy.

### Coverage Summary
- **Total Movies:** {analysis['total_movies']:,}
- **Movies with Genres:** {analysis['movies_with_genres']:,}
- **Coverage:** {analysis['coverage_pct']:.1f}%

### Top 20 Genres
| Rank | Genre | Count |
|------|-------|-------|
"""
    
    for i, (genre, count) in enumerate(analysis['genre_frequencies'].items(), 1):
        phase4_section += f"| {i} | {genre} | {count:,} |\n"
    
    phase4_section += f"""
### Genre Count Distribution
- **Min:** {analysis['genre_count_stats']['min']}
- **Median:** {analysis['genre_count_stats']['median']:.1f}
- **Max:** {analysis['genre_count_stats']['max']}
- **Mean:** {analysis['genre_count_stats']['mean']:.1f}

### Outputs
- `data/normalized/movies_genres.parquet`: Normalized genres dataset
- `data/features/genres/movies_genres_multihot.parquet`: Multi-hot encoding
- `docs/genre_taxonomy.json`: Genre mapping and taxonomy
"""
    
    # Append to existing content
    if "## Phase 4 — Genres & Taxonomy" not in content:
        content += phase4_section
    else:
        # Replace existing Phase 4 section
        import re
        pattern = r"## Phase 4 — Genres & Taxonomy.*?(?=##|\Z)"
        content = re.sub(pattern, phase4_section.strip(), content, flags=re.DOTALL)
    
    # Write updated report
    with open(report_path, 'w') as f:
        f.write(content)
    
    logging.info(f"Updated step1b report: {report_path}")

def print_final_summary(analysis, top_genres):
    """Print final summary block"""
    logging.info("=== FINAL SUMMARY ===")
    
    print("\n" + "="*80)
    print("STEP 1B PHASE 4: GENRES & TAXONOMY COMPLETE")
    print("="*80)
    
    print(f"\nGenre Coverage: {analysis['coverage_pct']:.1f}%")
    print(f"Movies with genres: {analysis['movies_with_genres']:,}/{analysis['total_movies']:,}")
    
    print(f"\nTop 10 Genres:")
    for i, (genre, count) in enumerate(list(analysis['genre_frequencies'].items())[:10], 1):
        print(f"  {i:2d}. {genre:15s}: {count:6,}")
    
    print(f"\nGenre Count Distribution:")
    print(f"  Min: {analysis['genre_count_stats']['min']}")
    print(f"  Median: {analysis['genre_count_stats']['median']:.1f}")
    print(f"  Max: {analysis['genre_count_stats']['max']}")
    print(f"  Mean: {analysis['genre_count_stats']['mean']:.1f}")
    
    print("\n" + "="*80)

def main():
    """Main function for Phase 4: Genres & Taxonomy"""
    start_time = datetime.now()
    logging.info("=== STARTING STEP 1B PHASE 4: GENRES & TAXONOMY ===")
    
    try:
        # Setup
        setup_directories()
        
        # Load data
        master_df = load_master_data()
        
        # Create genre mapping
        genre_mapping, reverse_mapping = create_genre_mapping()
        
        # Process genres
        genres_processed, genres_str_list = process_genres(master_df, reverse_mapping)
        
        # Analyze coverage
        analysis, top_genres = analyze_genre_coverage(genres_processed)
        
        # Create multi-hot encoding
        multihot_df = create_multihot_encoding(genres_processed, top_genres)
        
        # Create genres dataframe
        genres_df = pd.DataFrame({
            'genres_list': genres_processed,
            'genres_str': genres_str_list
        }, index=master_df.index)
        
        # Save outputs
        output_paths = save_outputs(genres_df, multihot_df, genre_mapping, analysis)
        
        # Print final summary
        print_final_summary(analysis, top_genres)
        
        # Log completion
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"=== PHASE 4 COMPLETE in {duration} ===")
        logging.info(f"Outputs saved to: {list(output_paths.values())}")
        
    except Exception as e:
        logging.error(f"Phase 4 failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
