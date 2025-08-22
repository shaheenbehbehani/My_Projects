#!/usr/bin/env python3
"""
FPL Gameweek Live Data Fetcher

This script fetches live data for a specific gameweek from the FPL public API and saves it
as a normalized Parquet snapshot. It handles the complex nested structure of live gameweek
data and flattens it into a clean player-level DataFrame.

Inputs: FPL public API endpoint (/event/{gw}/live)
Outputs: Parquet file in data/FPL/raw/ with normalized player stats
Usage: python -m scripts.fpl.fetch_gw --gw <gameweek_number>

The script handles both list and dict formats of the 'elements' field from the API,
flattens nested JSON structures, and preserves all original data while ensuring
core columns are properly typed for downstream processing.

Design notes:
- Retry policy: Exponential backoff up to 5 attempts, starting at 0.5s, multiplying by 1.8
- Data normalization: Handles both list and dict formats of elements field
- Flattening: Recursively flattens nested JSON structures into clean columns
- Core columns: Ensures minimum required columns are present and properly typed
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Union

import pandas as pd
import requests

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fpl.fpl_client import FPLClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FPL API base URL
FPL_BASE_URL = "https://fantasy.premierleague.com/api"

# Core columns that should always be present
CORE_COLUMNS = [
    'id', 'minutes', 'total_points', 'goals_scored', 'assists', 
    'clean_sheets', 'goals_conceded', 'saves', 'yellow_cards', 
    'red_cards', 'bps', 'bonus'
]


def setup_data_directory() -> Path:
    """Ensure the data directory exists and return the path."""
    data_dir = Path("data/FPL/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def fetch_with_retries(url: str, max_retries: int = 5, start_delay: float = 0.5) -> Dict[str, Any]:
    """
    Fetch data from URL with exponential backoff retry logic.
    
    Args:
        url: URL to fetch
        max_retries: Maximum number of retry attempts
        start_delay: Initial delay in seconds
        
    Returns:
        JSON response as dictionary
        
    Raises:
        RuntimeError: If all retries fail
    """
    delay = start_delay
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching {url} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 1.8
                continue
            else:
                raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts. Final error: {e}")
    
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


def flatten_dict(data: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """
    Recursively flatten a nested dictionary.
    
    Args:
        data: Nested dictionary to flatten
        prefix: Prefix for nested keys
        
    Returns:
        Flattened dictionary with dot-separated keys
    """
    flattened = {}
    
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key))
        elif isinstance(value, list):
            # For lists, join with comma if they contain strings/numbers
            if all(isinstance(item, (str, int, float, bool)) for item in value):
                flattened[new_key] = ','.join(str(item) for item in value)
            else:
                # For complex lists, convert to string representation
                flattened[new_key] = str(value)
        else:
            flattened[new_key] = value
    
    return flattened


def normalize_elements_data(elements_data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize elements data whether it's a list or dict.
    
    Args:
        elements_data: Elements data from API (list or dict)
        
    Returns:
        List of normalized player dictionaries
    """
    if isinstance(elements_data, list):
        # Already a list, just flatten each element
        normalized = []
        for element in elements_data:
            flattened = flatten_dict(element)
            normalized.append(flattened)
        return normalized
    
    elif isinstance(elements_data, dict):
        # Convert dict to list of flattened elements
        normalized = []
        for player_id, element_data in elements_data.items():
            # Add the player ID if it's not already present
            if 'id' not in element_data:
                element_data['id'] = int(player_id)
            
            flattened = flatten_dict(element_data)
            normalized.append(flattened)
        return normalized
    
    else:
        raise ValueError(f"Unexpected elements data type: {type(elements_data)}")


def fetch_gameweek_data(gw: int) -> pd.DataFrame:
    """
    Fetch live gameweek data from FPL API.
    
    Args:
        gw: Gameweek number
        
    Returns:
        DataFrame with normalized player stats
        
    Raises:
        RuntimeError: If no data is returned or API call fails
    """
    try:
        client = FPLClient()
        data = client.event_live(gw)
        
        if not data or 'elements' not in data:
            raise RuntimeError(f"No data returned for gameweek {gw}")
        
        elements_data = data['elements']
        if not elements_data:
            raise RuntimeError(f"No elements data returned for gameweek {gw}")
        
        # Normalize the elements data
        normalized_elements = normalize_elements_data(elements_data)
        
        if not normalized_elements:
            raise RuntimeError(f"No normalized elements data for gameweek {gw}")
        
        # Convert to DataFrame
        df = pd.DataFrame(normalized_elements)
        
        if df.empty:
            raise RuntimeError(f"Empty DataFrame created for gameweek {gw}")
        
        logger.info(f"Fetched {len(df)} players for gameweek {gw}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch gameweek {gw} via client: {e}")
        # Fallback to direct API call
        url = f"{FPL_BASE_URL}/event/{gw}/live/"
        data = fetch_with_retries(url)
        
        if not data or 'elements' not in data:
            raise RuntimeError(f"No data returned for gameweek {gw}")
        
        elements_data = data['elements']
        if not elements_data:
            raise RuntimeError(f"No elements data returned for gameweek {gw}")
        
        # Normalize the elements data
        normalized_elements = normalize_elements_data(elements_data)
        
        if not normalized_elements:
            raise RuntimeError(f"No normalized elements data for gameweek {gw}")
        
        # Convert to DataFrame
        df = pd.DataFrame(normalized_elements)
        
        if df.empty:
            raise RuntimeError(f"Empty DataFrame created for gameweek {gw}")
        
        logger.info(f"Fetched {len(df)} players for gameweek {gw} (direct API)")
        return df


def ensure_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure core columns exist and are properly typed.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with core columns ensured
    """
    # Check which core columns are missing
    missing_columns = [col for col in CORE_COLUMNS if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Missing core columns: {missing_columns}")
        
        # Add missing columns with default values
        for col in missing_columns:
            if col in ['id', 'minutes', 'total_points', 'goals_scored', 'assists', 
                      'clean_sheets', 'goals_conceded', 'saves', 'yellow_cards', 
                      'red_cards', 'bps', 'bonus']:
                df[col] = 0
            else:
                df[col] = None
    
    # Ensure proper dtypes for core columns
    dtype_mapping = {
        'id': 'int64',
        'minutes': 'Int32',
        'total_points': 'Int32',
        'goals_scored': 'Int32',
        'assists': 'Int32',
        'clean_sheets': 'Int32',
        'goals_conceded': 'Int32',
        'saves': 'Int32',
        'yellow_cards': 'Int32',
        'red_cards': 'Int32',
        'bps': 'Int32',
        'bonus': 'Int32'
    }
    
    for col, dtype in dtype_mapping.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Could not cast column {col} to {dtype}: {e}")
    
    return df


def save_gameweek_data(df: pd.DataFrame, gw: int, data_dir: Path) -> None:
    """
    Save gameweek DataFrame to Parquet file.
    
    Args:
        df: DataFrame to save
        gw: Gameweek number
        data_dir: Directory to save the file
    """
    # Ensure core columns exist
    df = ensure_core_columns(df)
    
    # Save to Parquet
    output_path = data_dir / f"gw{gw}_live.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} players to {output_path}")


def main():
    """Main function to fetch and save gameweek data."""
    parser = argparse.ArgumentParser(description="Fetch FPL gameweek live data and save as Parquet")
    parser.add_argument(
        '--gw',
        type=int,
        required=True,
        help='Gameweek number to fetch'
    )
    
    args = parser.parse_args()
    gw = args.gw
    
    if gw < 1:
        logger.error("Gameweek must be a positive integer")
        sys.exit(1)
    
    start_time = time.time()
    logger.info(f"Starting FPL gameweek {gw} data fetch")
    
    try:
        # Setup data directory
        data_dir = setup_data_directory()
        
        # Fetch gameweek data
        df = fetch_gameweek_data(gw)
        
        # Save to Parquet
        save_gameweek_data(df, gw, data_dir)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Gameweek {gw} data fetch completed successfully in {elapsed_time:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Gameweek {gw} data fetch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 