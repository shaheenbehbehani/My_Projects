#!/usr/bin/env python3
"""
Team Name Normalization Module

This module provides utilities to standardize Premier League team names
across different datasets that may use various abbreviations or formats.
"""

import re
from typing import List, Union
import polars as pl


# Standard mapping of team name variants to canonical names
STANDARD_TEAM_MAP = {
    # Manchester United variants
    "man utd": "Manchester United",
    "manchester utd": "Manchester United", 
    "manchester united": "Manchester United",
    "man u": "Manchester United",
    
    # Manchester City variants
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "city": "Manchester City",
    
    # Tottenham variants
    "spurs": "Tottenham Hotspur",
    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",
    
    # Wolverhampton variants
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "wolverhampton wanderers": "Wolverhampton Wanderers",
    
    # West Ham variants
    "west ham": "West Ham United",
    "west ham united": "West Ham United",
    
    # Nottingham Forest variants
    "nottingham forest": "Nottingham Forest",
    "nottm forest": "Nottingham Forest",
    "notts forest": "Nottingham Forest",
    "forest": "Nottingham Forest",
    
    # Brighton variants
    "brighton": "Brighton & Hove Albion",
    "brighton & hove albion": "Brighton & Hove Albion",
    "brighton and hove albion": "Brighton & Hove Albion",
    
    # Newcastle variants
    "newcastle": "Newcastle United",
    "newcastle united": "Newcastle United",
    
    # Ipswich variants
    "ipswich": "Ipswich Town",
    "ipswich town": "Ipswich Town",
    
    # Leicester variants
    "leicester": "Leicester City",
    "leicester city": "Leicester City",
    
    # Leeds variants
    "leeds": "Leeds United",
    "leeds united": "Leeds United",
    
    # AFC Bournemouth variants
    "bournemouth": "AFC Bournemouth",
    "afc bournemouth": "AFC Bournemouth",
    
    # Sheffield United variants
    "sheff utd": "Sheffield United",
    "sheffield united": "Sheffield United",
    "sheffield utd": "Sheffield United",
    
    # Crystal Palace variants
    "crystal palace": "Crystal Palace",
    "palace": "Crystal Palace",
    
    # Other Premier League teams
    "brentford": "Brentford",
    "brentford fc": "Brentford",
    "fulham": "Fulham",
    "fulham fc": "Fulham",
    "everton": "Everton",
    "everton fc": "Everton",
    "aston villa": "Aston Villa",
    "villa": "Aston Villa",
    "arsenal": "Arsenal",
    "arsenal fc": "Arsenal",
    "chelsea": "Chelsea",
    "chelsea fc": "Chelsea",
    "liverpool": "Liverpool",
    "liverpool fc": "Liverpool",
    "southampton": "Southampton",
    "burnley": "Burnley",
    "burnley fc": "Burnley",
    "sunderland": "Sunderland",
    "sunderland afc": "Sunderland",
    
    # Additional variants found in data
    "man united": "Manchester United",
    "nott'm forest": "Nottingham Forest",
    "leicester city": "Leicester City",
    "southampton": "Southampton",
}


def canonicalize_team(name: str) -> str:
    """
    Canonicalize a team name to its standard form.
    
    Args:
        name: Team name to canonicalize
        
    Returns:
        Canonical team name
    """
    if not isinstance(name, str) or not name.strip():
        return name
    
    # Normalize input: lowercase, strip, collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', name.strip().lower())
    
    # Check if we have a direct mapping
    if normalized in STANDARD_TEAM_MAP:
        return STANDARD_TEAM_MAP[normalized]
    
    # If no mapping found, apply intelligent title casing
    return _intelligent_title_case(name.strip())


def _intelligent_title_case(name: str) -> str:
    """Apply intelligent title casing that preserves special football terms."""
    # Words that should maintain specific capitalization
    special_words = {
        'united': 'United',
        'city': 'City', 
        'town': 'Town',
        'hotspur': 'Hotspur',
        'wanderers': 'Wanderers',
        'albion': 'Albion',
        'forest': 'Forest',
        'palace': 'Palace',
        'villa': 'Villa',
        'rovers': 'Rovers',
        'athletic': 'Athletic',
        'county': 'County',
        'and': 'and',
        'of': 'of',
        'the': 'the',
        'fc': 'FC',
        'afc': 'AFC',
        'rfc': 'RFC',
        'cf': 'CF',
    }
    
    # Handle & symbol specifically
    name = name.replace(' and ', ' & ').replace(' AND ', ' & ')
    
    # Split and process each word
    words = name.split()
    result_words = []
    
    for word in words:
        if word == '&':
            result_words.append('&')
            continue
            
        clean_word = word.lower().rstrip('.,!?')  # Remove trailing punctuation for lookup
        if clean_word in special_words:
            result_words.append(special_words[clean_word])
        else:
            result_words.append(word.capitalize())
    
    return ' '.join(result_words)


def canonicalize_frame(df: pl.DataFrame, cols: Union[str, List[str]]) -> pl.DataFrame:
    """
    Canonicalize team name columns in a Polars DataFrame.
    
    Args:
        df: Input DataFrame
        cols: Column name(s) to canonicalize
        
    Returns:
        New DataFrame with canonicalized team names
    """
    if isinstance(cols, str):
        cols = [cols]
    
    # Validate columns exist
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Create a copy and apply canonicalization
    result_df = df.clone()
    
    for col in cols:
        result_df = result_df.with_columns(
            pl.col(col).map_elements(canonicalize_team, return_dtype=pl.String).alias(col)
        )
    
    return result_df


if __name__ == "__main__":
    """Demo script showing team name canonicalization."""
    print("Premier League Team Name Canonicalization Demo")
    print("=" * 50)
    
    # Test individual canonicalization
    test_names = ["man utd", "SPURS", "man city", "wolves", "west ham", "arsenal", "chelsea fc", "liverpool fc"]
    
    print("\nIndividual Name Canonicalization:")
    for name in test_names:
        canonical = canonicalize_team(name)
        print(f"'{name}' → '{canonical}'")
    
    # Create sample DataFrame
    sample_data = {
        "home_team": ["man utd", "spurs", "wolves", "nottm forest"],
        "away_team": ["arsenal", "man city", "west ham", "newcastle"],
        "home_score": [2, 1, 0, 3],
        "away_score": [1, 2, 1, 1]
    }
    
    df = pl.DataFrame(sample_data)
    
    print(f"\nDataFrame Canonicalization:")
    print("BEFORE:")
    print(df)
    
    # Canonicalize team name columns
    df_canonical = canonicalize_frame(df, ["home_team", "away_team"])
    
    print("\nAFTER:")
    print(df_canonical) 