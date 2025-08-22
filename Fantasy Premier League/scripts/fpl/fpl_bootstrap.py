#!/usr/bin/env python3
"""
FPL Bootstrap Data Fetcher

This script fetches the latest FPL data from the public API and persists it as Parquet snapshots
for downstream processing. It handles three core reference tables: players, teams, and fixtures.

Inputs: FPL public API endpoints (bootstrap-static, fixtures)
Outputs: Parquet files in data/FPL/raw/ with standardized schemas
Usage: python -m scripts.fpl.fpl_bootstrap [--only {players,teams,fixtures}]

The script creates data/FPL/raw/ if it doesn't exist and always overwrites existing files
to ensure the latest snapshot is available for subsequent processing steps.

Design notes:
- Retry policy: Exponential backoff up to 5 attempts, starting at 0.5s, multiplying by 1.8
- Dtype casting: Applied only where columns exist; missing columns don't cause failures
- Overwrite policy: Always overwrite to maintain latest snapshot
- Extensibility: Add new endpoints by extending the fetch_functions dict and schema_mappings
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

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

# Schema mappings for each table
SCHEMA_MAPPINGS = {
    'players': {
        'id': 'int64',
        'web_name': 'string',
        'first_name': 'string',
        'second_name': 'string',
        'team': 'Int16',
        'team_code': 'Int32',
        'status': 'string',
        'now_cost': 'Int16',
        'element_type': 'Int8',
        'selected_by_percent': 'float64',
        'total_points': 'Int32',
        'minutes': 'Int32',
        'goals_scored': 'Int32',
        'assists': 'Int32',
        'clean_sheets': 'Int32',
        'goals_conceded': 'Int32',
        'yellow_cards': 'Int32',
        'red_cards': 'Int32',
        'saves': 'Int32',
        'bonus': 'Int32',
        'bps': 'Int32',
        'influence': 'float64',
        'creativity': 'float64',
        'threat': 'float64',
        'ict_index': 'float64',
        'ep_this': 'string',
        'ep_next': 'string',
        'form': 'float64',
        'news': 'string',
        'news_added': 'datetime64',
        'chance_of_playing_this_round': 'float64',
        'chance_of_playing_next_round': 'float64',
        'cost_change_event': 'Int16',
        'cost_change_event_fall': 'Int16',
        'cost_change_start': 'Int16',
        'cost_change_start_fall': 'Int16'
    },
    'teams': {
        'id': 'Int16',
        'code': 'Int32',
        'name': 'string',
        'short_name': 'string',
        'strength': 'Int16',
        'strength_attack_home': 'Int16',
        'strength_attack_away': 'Int16',
        'strength_defence_home': 'Int16',
        'strength_defence_away': 'Int16',
        'pulse_id': 'Int32'
    },
    'fixtures': {
        'id': 'Int32',
        'event': 'Int64',
        'kickoff_time': 'datetime64',
        'finished': 'bool',
        'minutes': 'Int16',
        'provisional_start_time': 'bool',
        'team_h': 'Int16',
        'team_h_score': 'Int64',
        'team_a': 'Int16',
        'team_a_score': 'Int64',
        'team_h_difficulty': 'Int16',
        'team_a_difficulty': 'Int16'
    }
}


def setup_data_directory() -> Path:
    """Ensure the data directory exists and return the path."""
    data_dir = Path("data/FPL/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {data_dir.absolute()}")
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
            else:
                raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts. Final error: {e}")
    
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


def cast_dataframe_schema(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Cast DataFrame columns to expected dtypes where possible.
    
    Args:
        df: Input DataFrame
        table_name: Name of the table for schema mapping
        
    Returns:
        DataFrame with casted columns where possible
    """
    schema = SCHEMA_MAPPINGS.get(table_name, {})
    missing_columns = []
    casting_issues = []
    
    for column, expected_dtype in schema.items():
        if column in df.columns:
            try:
                if expected_dtype == 'datetime64':
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif expected_dtype == 'bool':
                    df[column] = df[column].astype('bool')
                elif expected_dtype == 'Int64':
                    # Handle nullable integers
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                else:
                    df[column] = df[column].astype(expected_dtype)
            except Exception as e:
                casting_issues.append(f"{column}: {e}")
                logger.warning(f"Could not cast column {column} to {expected_dtype}: {e}")
        else:
            missing_columns.append(column)
    
    if missing_columns:
        logger.info(f"Missing columns in {table_name}: {missing_columns}")
    
    if casting_issues:
        logger.warning(f"Casting issues in {table_name}: {casting_issues}")
    
    return df


def fetch_players_data() -> pd.DataFrame:
    """Fetch players data from bootstrap endpoint."""
    try:
        client = FPLClient()
        data = client.bootstrap()
        players_df = pd.DataFrame(data.get('elements', []))
        
        if players_df.empty:
            raise RuntimeError("No players data received from bootstrap endpoint")
        
        logger.info(f"Fetched {len(players_df)} players")
        return players_df
        
    except Exception as e:
        logger.error(f"Failed to fetch players via client: {e}")
        # Fallback to direct API call
        url = f"{FPL_BASE_URL}/bootstrap-static/"
        data = fetch_with_retries(url)
        players_df = pd.DataFrame(data.get('elements', []))
        
        if players_df.empty:
            raise RuntimeError("No players data received from bootstrap endpoint")
        
        logger.info(f"Fetched {len(players_df)} players (direct API)")
        return players_df


def fetch_teams_data() -> pd.DataFrame:
    """Fetch teams data from bootstrap endpoint."""
    try:
        client = FPLClient()
        data = client.bootstrap()
        teams_df = pd.DataFrame(data.get('teams', []))
        
        if teams_df.empty:
            raise RuntimeError("No teams data received from bootstrap endpoint")
        
        logger.info(f"Fetched {len(teams_df)} teams")
        return teams_df
        
    except Exception as e:
        logger.error(f"Failed to fetch teams via client: {e}")
        # Fallback to direct API call
        url = f"{FPL_BASE_URL}/bootstrap-static/"
        data = fetch_with_retries(url)
        teams_df = pd.DataFrame(data.get('teams', []))
        
        if teams_df.empty:
            raise RuntimeError("No teams data received from bootstrap endpoint")
        
        logger.info(f"Fetched {len(teams_df)} teams (direct API)")
        return teams_df


def fetch_fixtures_data() -> pd.DataFrame:
    """Fetch fixtures data from fixtures endpoint."""
    try:
        client = FPLClient()
        data = client.fixtures()
        fixtures_df = pd.DataFrame(data)
        
        if fixtures_df.empty:
            raise RuntimeError("No fixtures data received from fixtures endpoint")
        
        logger.info(f"Fetched {len(fixtures_df)} fixtures")
        return fixtures_df
        
    except Exception as e:
        logger.error(f"Failed to fetch fixtures via client: {e}")
        # Fallback to direct API call
        url = f"{FPL_BASE_URL}/fixtures/"
        data = fetch_with_retries(url)
        fixtures_df = pd.DataFrame(data)
        
        if fixtures_df.empty:
            raise RuntimeError("No fixtures data received from fixtures endpoint")
        
        logger.info(f"Fetched {len(fixtures_df)} fixtures (direct API)")
        return fixtures_df


def save_table(df: pd.DataFrame, table_name: str, data_dir: Path) -> None:
    """
    Save DataFrame to Parquet file with schema casting.
    
    Args:
        df: DataFrame to save
        table_name: Name of the table
        data_dir: Directory to save the file
    """
    # Cast schema before saving
    df = cast_dataframe_schema(df, table_name)
    
    # Save to Parquet
    output_path = data_dir / f"{table_name}.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} {table_name} to {output_path}")


def main():
    """Main function to fetch and save FPL data."""
    parser = argparse.ArgumentParser(description="Fetch FPL data and save as Parquet files")
    parser.add_argument(
        '--only',
        choices=['players', 'teams', 'fixtures'],
        help='Fetch only a specific table (default: all)'
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("Starting FPL bootstrap data fetch")
    
    try:
        # Setup data directory
        data_dir = setup_data_directory()
        
        # Define fetch functions
        fetch_functions = {
            'players': fetch_players_data,
            'teams': fetch_teams_data,
            'fixtures': fetch_fixtures_data
        }
        
        # Determine which tables to fetch
        if args.only:
            tables_to_fetch = [args.only]
            logger.info(f"Fetching only: {args.only}")
        else:
            tables_to_fetch = list(fetch_functions.keys())
            logger.info("Fetching all tables: players, teams, fixtures")
        
        # Fetch and save each table
        for table_name in tables_to_fetch:
            logger.info(f"Processing {table_name}...")
            df = fetch_functions[table_name]()
            save_table(df, table_name, data_dir)
        
        elapsed_time = time.time() - start_time
        logger.info(f"FPL bootstrap completed successfully in {elapsed_time:.1f} seconds")
        
    except Exception as e:
        logger.error(f"FPL bootstrap failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 