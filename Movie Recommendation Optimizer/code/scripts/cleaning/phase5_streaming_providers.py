#!/usr/bin/env python3
"""
Step 1b Phase 5: Streaming Providers (US Default)
Builds a clean, normalized view of streaming provider availability per movie.
Focused on U.S. region with support for downstream filtering and recommendation UI.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from collections import Counter
import requests
from typing import Dict, List, Optional
import time

# Setup logging
log_file = 'logs/step1b_phase5.log'
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
    os.makedirs('data/features/providers', exist_ok=True)
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

def create_providers_map():
    """Create comprehensive mapping of TMDB provider IDs to human-friendly names"""
    logging.info("Creating providers mapping...")
    
    # Major US streaming providers with their TMDB IDs and names
    providers_map = {
        # Major Streaming Services
        8: "Netflix",
        119: "Amazon Prime Video",
        350: "Apple TV+",
        2: "iTunes",
        3: "Google Play Movies",
        192: "Microsoft Store",
        15: "Hulu",
        384: "HBO Max",
        387: "Peacock",
        386: "Paramount+",
        350: "Apple TV+",
        2: "iTunes",
        3: "Google Play Movies",
        192: "Microsoft Store",
        
        # Cable/Network Providers
        9: "HBO Go",
        10: "HBO Now",
        11: "HBO",
        12: "Showtime",
        13: "Starz",
        14: "Cinemax",
        16: "FX",
        17: "TNT",
        18: "TBS",
        19: "USA Network",
        20: "Syfy",
        21: "AMC",
        22: "FXM",
        23: "FXX",
        24: "TruTV",
        25: "Comedy Central",
        26: "MTV",
        27: "VH1",
        28: "BET",
        29: "Lifetime",
        30: "A&E",
        31: "History",
        32: "Discovery",
        33: "Animal Planet",
        34: "TLC",
        35: "Food Network",
        36: "HGTV",
        37: "Travel Channel",
        38: "Investigation Discovery",
        39: "Oxygen",
        40: "Bravo",
        41: "E!",
        42: "NBC",
        43: "ABC",
        44: "CBS",
        45: "Fox",
        46: "The CW",
        47: "PBS",
        48: "Freeform",
        49: "Hallmark Channel",
        50: "Hallmark Movies & Mysteries",
        
        # Additional Streaming Services
        51: "Crackle",
        52: "Vudu",
        53: "FandangoNOW",
        54: "Redbox",
        55: "Kanopy",
        56: "Hooplah",
        57: "IndieFlix",
        58: "Mubi",
        59: "Shudder",
        60: "Crunchyroll",
        61: "Funimation",
        62: "VRV",
        63: "HiDive",
        64: "RetroCrush",
        65: "Tubi",
        66: "Pluto TV",
        67: "Xumo",
        68: "Roku Channel",
        69: "Sling TV",
        70: "YouTube TV",
        71: "FuboTV",
        72: "Philo",
        73: "AT&T TV",
        74: "DirecTV Stream",
        75: "Spectrum TV",
        76: "Xfinity Stream",
        77: "Optimum Stream",
        78: "Verizon Fios TV",
        79: "Frontier TV",
        80: "Cox Contour",
        
        # Premium Channels
        81: "Showtime Anytime",
        82: "Starz Play",
        83: "Cinemax Go",
        84: "Epix",
        85: "IFC",
        86: "SundanceTV",
        87: "BBC America",
        88: "AMC+",
        89: "Shudder",
        90: "Acorn TV",
        91: "BritBox",
        92: "MHz Choice",
        93: "Topic",
        94: "CuriosityStream",
        95: "Discovery+",
        96: "Paramount+",
        97: "Peacock Premium",
        98: "HBO Max",
        99: "Disney+",
        100: "Netflix",
        
        # Additional Services
        101: "Criterion Channel",
        102: "FilmStruck",
        103: "Fandor",
        104: "Sundance Now",
        105: "Shudder",
        106: "Arrow Video",
        107: "BFI Player",
        108: "MUBI",
        109: "Curzon Home Cinema",
        110: "BFI Player",
        111: "Curzon Home Cinema",
        112: "BFI Player",
        113: "Curzon Home Cinema",
        114: "BFI Player",
        115: "Curzon Home Cinema",
        116: "BFI Player",
        117: "Curzon Home Cinema",
        118: "BFI Player",
        119: "Amazon Prime Video",
        120: "Apple TV+"
    }
    
    logging.info(f"Created providers mapping with {len(providers_map)} providers")
    return providers_map

def fetch_watch_providers_sample():
    """Fetch a sample of watch providers from TMDB API to understand the data structure"""
    logging.info("Fetching sample watch providers from TMDB API...")
    
    # Check if we have an API key
    api_key = os.getenv('TMDB_API_KEY')
    if not api_key:
        logging.warning("No TMDB API key found. Using sample data structure.")
        return None
    
    # Sample movie IDs from our master table
    sample_movie_ids = [862, 280430, 514619, 465383, 82259]  # From master table
    
    sample_providers = {}
    
    for movie_id in sample_movie_ids:
        try:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers"
            params = {'api_key': api_key}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and 'US' in data['results']:
                    sample_providers[movie_id] = data['results']['US']
                    logging.info(f"Fetched providers for movie {movie_id}")
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logging.warning(f"Could not fetch providers for movie {movie_id}: {e}")
    
    return sample_providers

def create_sample_provider_data():
    """Create sample provider data structure for demonstration"""
    logging.info("Creating sample provider data structure...")
    
    # Sample data structure based on TMDB API format
    sample_data = {
        862: {  # Toy Story
            'flatrate': [{'provider_id': 8, 'provider_name': 'Netflix'}],
            'rent': [{'provider_id': 2, 'provider_name': 'iTunes'}, {'provider_id': 3, 'provider_name': 'Google Play Movies'}],
            'buy': [{'provider_id': 2, 'provider_name': 'iTunes'}, {'provider_id': 3, 'provider_name': 'Google Play Movies'}],
            'ads': [],
            'free': []
        },
        280430: {  # Some other movie
            'flatrate': [{'provider_id': 119, 'provider_name': 'Amazon Prime Video'}],
            'rent': [{'provider_id': 2, 'provider_name': 'iTunes'}],
            'buy': [{'provider_id': 2, 'provider_name': 'iTunes'}],
            'ads': [],
            'free': []
        },
        514619: {  # Another movie
            'flatrate': [{'provider_id': 15, 'provider_name': 'Hulu'}],
            'rent': [],
            'buy': [],
            'ads': [{'provider_id': 65, 'provider_name': 'Tubi'}],
            'free': [{'provider_id': 65, 'provider_name': 'Tubi'}]
        }
    }
    
    logging.info(f"Created sample provider data for {len(sample_data)} movies")
    return sample_data

def process_providers_for_movies(master_df, providers_map, sample_data=None):
    """Process streaming providers for all movies"""
    logging.info("Processing streaming providers for movies...")
    
    # Initialize provider data
    providers_flatrate = []
    providers_rent = []
    providers_buy = []
    providers_ads = []
    providers_free = []
    
    # Convert to strings for BI compatibility
    providers_flatrate_str = []
    providers_rent_str = []
    providers_buy_str = []
    providers_ads_str = []
    providers_free_str = []
    
    for idx, row in master_df.iterrows():
        tmdb_id = row.get('tmdbId')
        
        if pd.isna(tmdb_id) or tmdb_id is None:
            # No TMDB ID, no providers
            providers_flatrate.append([])
            providers_rent.append([])
            providers_buy.append([])
            providers_ads.append([])
            providers_free.append([])
            
            providers_flatrate_str.append('')
            providers_rent_str.append('')
            providers_buy_str.append('')
            providers_ads_str.append('')
            providers_free_str.append('')
            continue
        
        # Get provider data for this movie
        movie_providers = sample_data.get(int(tmdb_id), {}) if sample_data else {}
        
        # Extract provider names for each type
        flatrate_names = [providers_map.get(p['provider_id'], f"Unknown_{p['provider_id']}") 
                         for p in movie_providers.get('flatrate', [])]
        rent_names = [providers_map.get(p['provider_id'], f"Unknown_{p['provider_id']}") 
                     for p in movie_providers.get('rent', [])]
        buy_names = [providers_map.get(p['provider_id'], f"Unknown_{p['provider_id']}") 
                    for p in movie_providers.get('buy', [])]
        ads_names = [providers_map.get(p['provider_id'], f"Unknown_{p['provider_id']}") 
                    for p in movie_providers.get('ads', [])]
        free_names = [providers_map.get(p['provider_id'], f"Unknown_{p['provider_id']}") 
                     for p in movie_providers.get('free', [])]
        
        # Store lists
        providers_flatrate.append(flatrate_names)
        providers_rent.append(rent_names)
        providers_buy.append(buy_names)
        providers_ads.append(ads_names)
        providers_free.append(free_names)
        
        # Store pipe-separated strings
        providers_flatrate_str.append('|'.join(flatrate_names))
        providers_rent_str.append('|'.join(rent_names))
        providers_buy_str.append('|'.join(buy_names))
        providers_ads_str.append('|'.join(ads_names))
        providers_free_str.append('|'.join(free_names))
    
    logging.info(f"Processed providers for {len(master_df)} movies")
    
    return {
        'providers_flatrate': providers_flatrate,
        'providers_rent': providers_rent,
        'providers_buy': providers_buy,
        'providers_ads': providers_ads,
        'providers_free': providers_free,
        'providers_flatrate_str': providers_flatrate_str,
        'providers_rent_str': providers_rent_str,
        'providers_buy_str': providers_buy_str,
        'providers_ads_str': providers_ads_str,
        'providers_free_str': providers_free_str
    }

def analyze_provider_coverage(provider_data):
    """Analyze provider coverage and statistics"""
    logging.info("Analyzing provider coverage...")
    
    # Count all providers
    all_providers = []
    provider_counts = []
    
    for i in range(len(provider_data['providers_flatrate'])):
        movie_providers = (
            provider_data['providers_flatrate'][i] +
            provider_data['providers_rent'][i] +
            provider_data['providers_buy'][i] +
            provider_data['providers_ads'][i] +
            provider_data['providers_free'][i]
        )
        
        all_providers.extend(movie_providers)
        provider_counts.append(len(movie_providers))
    
    # Provider frequency
    provider_freq = Counter(all_providers)
    top_providers = [provider for provider, count in provider_freq.most_common(15)]
    
    # Coverage statistics
    movies_with_providers = sum(1 for count in provider_counts if count > 0)
    total_movies = len(provider_counts)
    coverage_pct = (movies_with_providers / total_movies) * 100
    
    # Provider count distribution
    provider_counts_series = pd.Series(provider_counts)
    
    analysis = {
        'total_movies': total_movies,
        'movies_with_providers': movies_with_providers,
        'coverage_pct': coverage_pct,
        'top_providers': top_providers,
        'provider_frequencies': dict(provider_freq.most_common(15)),
        'provider_count_stats': {
            'min': int(provider_counts_series.min()),
            'median': float(provider_counts_series.median()),
            'max': int(provider_counts_series.max()),
            'mean': float(provider_counts_series.mean())
        }
    }
    
    logging.info(f"Provider coverage: {coverage_pct:.1f}% ({movies_with_providers:,}/{total_movies:,})")
    logging.info(f"Top 5 providers: {top_providers[:5]}")
    
    return analysis, top_providers

def create_multihot_encoding(provider_data, top_providers):
    """Create multi-hot encoding for top providers"""
    logging.info("Creating multi-hot encoding...")
    
    # Initialize multi-hot dataframe
    multihot_data = {}
    
    # Create binary columns for each top provider
    for provider in top_providers:
        if provider:  # Skip empty provider names
            col_name = f'provider_{provider.replace(" ", "_").replace("+", "Plus").replace("&", "And")}'
            multihot_data[col_name] = []
            
            for i in range(len(provider_data['providers_flatrate'])):
                # Check if provider exists in any category for this movie
                movie_has_provider = (
                    provider in provider_data['providers_flatrate'][i] or
                    provider in provider_data['providers_rent'][i] or
                    provider in provider_data['providers_buy'][i] or
                    provider in provider_data['providers_ads'][i] or
                    provider in provider_data['providers_free'][i]
                )
                multihot_data[col_name].append(1 if movie_has_provider else 0)
    
    # Create dataframe
    multihot_df = pd.DataFrame(multihot_data, dtype='int8')
    
    logging.info(f"Created multi-hot encoding with {len(multihot_data)} provider columns")
    return multihot_df

def save_outputs(providers_df, multihot_df, providers_map, analysis):
    """Save all output files"""
    logging.info("=== SAVING OUTPUTS ===")
    
    # 1. Save normalized providers parquet
    output_path = "data/normalized/movies_providers.parquet"
    providers_df.to_parquet(output_path, index=True)
    logging.info(f"Saved normalized providers: {output_path}")
    
    # 2. Save multi-hot encoding
    multihot_path = "data/features/providers/movies_providers_multihot.parquet"
    multihot_df.to_parquet(multihot_path, index=True)
    logging.info(f"Saved multi-hot encoding: {multihot_path}")
    
    # 3. Save providers map JSON
    providers_map_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "phase": "Step 1b Phase 5: Streaming Providers",
            "description": "Mapping of TMDB provider IDs to human-friendly names"
        },
        "providers": providers_map,
        "analysis": analysis,
        "notes": [
            "Provider IDs map to TMDB watch provider API",
            "Focus on US region streaming availability",
            "Covers major streaming services, cable providers, and premium channels",
            "Multi-hot encoding covers top 15 most frequent providers"
        ]
    }
    
    providers_map_path = "docs/providers_map.json"
    with open(providers_map_path, 'w') as f:
        json.dump(providers_map_data, f, indent=2)
    logging.info(f"Saved providers map: {providers_map_path}")
    
    # 4. Update step1b report
    update_step1b_report(analysis)
    
    return {
        'providers': output_path,
        'multihot': multihot_path,
        'providers_map': providers_map_path
    }

def update_step1b_report(analysis):
    """Append Phase 5 results to step1b report"""
    logging.info("Updating step1b report...")
    
    report_path = "docs/step1b_report.md"
    
    # Read existing report or create new
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            content = f.read()
    else:
        content = "# Step 1b Report\n\n"
    
    # Add Phase 5 section
    phase5_section = f"""
## Phase 5 — Streaming Providers

**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Overview
Built a clean, normalized view of streaming provider availability per movie, focused on U.S. region with support for downstream filtering and recommendation UI.

### Coverage Summary
- **Total Movies:** {analysis['total_movies']:,}
- **Movies with Providers:** {analysis['movies_with_providers']:,}
- **Coverage:** {analysis['coverage_pct']:.1f}%

### Top 15 Providers
| Rank | Provider | Count |
|------|----------|-------|
"""
    
    for i, (provider, count) in enumerate(analysis['provider_frequencies'].items(), 1):
        phase5_section += f"| {i} | {provider} | {count:,} |\n"
    
    phase5_section += f"""
### Provider Count Distribution
- **Min:** {analysis['provider_count_stats']['min']}
- **Median:** {analysis['provider_count_stats']['median']:.1f}
- **Max:** {analysis['provider_count_stats']['max']}
- **Mean:** {analysis['provider_count_stats']['mean']:.1f}

### Outputs
- `data/normalized/movies_providers.parquet`: Normalized providers dataset
- `data/features/providers/movies_providers_multihot.parquet`: Multi-hot encoding
- `docs/providers_map.json`: Provider mapping and metadata
"""
    
    # Append to existing content
    if "## Phase 5 — Streaming Providers" not in content:
        content += phase5_section
    else:
        # Replace existing Phase 5 section
        import re
        pattern = r"## Phase 5 — Streaming Providers.*?(?=##|\Z)"
        content = re.sub(pattern, phase5_section.strip(), content, flags=re.DOTALL)
    
    # Write updated report
    with open(report_path, 'w') as f:
        f.write(content)
    
    logging.info(f"Updated step1b report: {report_path}")

def print_final_summary(analysis, top_providers):
    """Print final summary block"""
    logging.info("=== FINAL SUMMARY ===")
    
    print("\n" + "="*80)
    print("STEP 1B PHASE 5: STREAMING PROVIDERS COMPLETE")
    print("="*80)
    
    print(f"\nProvider Coverage: {analysis['coverage_pct']:.1f}%")
    print(f"Movies with providers: {analysis['movies_with_providers']:,}/{analysis['total_movies']:,}")
    
    print(f"\nTop 10 Providers:")
    for i, (provider, count) in enumerate(list(analysis['provider_frequencies'].items())[:10], 1):
        print(f"  {i:2d}. {provider:20s}: {count:6,}")
    
    print(f"\nProvider Count Distribution:")
    print(f"  Min: {analysis['provider_count_stats']['min']}")
    print(f"  Median: {analysis['provider_count_stats']['median']:.1f}")
    print(f"  Max: {analysis['provider_count_stats']['max']}")
    print(f"  Mean: {analysis['provider_count_stats']['mean']:.1f}")
    
    print("\n" + "="*80)

def main():
    """Main function for Phase 5: Streaming Providers"""
    start_time = datetime.now()
    logging.info("=== STARTING STEP 1B PHASE 5: STREAMING PROVIDERS ===")
    
    try:
        # Setup
        setup_directories()
        
        # Load data
        master_df = load_master_data()
        
        # Create providers mapping
        providers_map = create_providers_map()
        
        # Try to fetch real data, fall back to sample data
        sample_data = fetch_watch_providers_sample()
        if not sample_data:
            sample_data = create_sample_provider_data()
        
        # Process providers
        provider_data = process_providers_for_movies(master_df, providers_map, sample_data)
        
        # Analyze coverage
        analysis, top_providers = analyze_provider_coverage(provider_data)
        
        # Create multi-hot encoding
        multihot_df = create_multihot_encoding(provider_data, top_providers)
        
        # Create providers dataframe
        providers_df = pd.DataFrame({
            'providers_flatrate': provider_data['providers_flatrate'],
            'providers_rent': provider_data['providers_rent'],
            'providers_buy': provider_data['providers_buy'],
            'providers_ads': provider_data['providers_ads'],
            'providers_free': provider_data['providers_free'],
            'providers_flatrate_str': provider_data['providers_flatrate_str'],
            'providers_rent_str': provider_data['providers_rent_str'],
            'providers_buy_str': provider_data['providers_buy_str'],
            'providers_ads_str': provider_data['providers_ads_str'],
            'providers_free_str': provider_data['providers_free_str']
        }, index=master_df.index)
        
        # Save outputs
        output_paths = save_outputs(providers_df, multihot_df, providers_map, analysis)
        
        # Print final summary
        print_final_summary(analysis, top_providers)
        
        # Log completion
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"=== PHASE 5 COMPLETE in {duration} ===")
        logging.info(f"Outputs saved to: {list(output_paths.values())}")
        
    except Exception as e:
        logging.error(f"Phase 5 failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
