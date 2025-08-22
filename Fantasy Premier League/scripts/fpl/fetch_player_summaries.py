#!/usr/bin/env python3
"""
FPL Player Historical Data Fetcher

This script fetches historical summary data for each player from the FPL public API
and saves individual Parquet files containing their season-level statistics.

Inputs: Player IDs from data/FPL/raw/players.parquet
Outputs: Individual Parquet files in data/FPL/raw/player_hist/{player_id}.parquet
Usage: python -m scripts.fpl.fetch_player_summaries

The script reads all player IDs from the bootstrap data, fetches each player's
historical summary via the /element-summary/{id} endpoint, and saves the data
as individual Parquet files for downstream processing.

Design notes:
- Retry policy: Exponential backoff up to 5 attempts, starting at 0.5s, multiplying by 1.8
- Polite delays: 0.5-1s between API calls to avoid overwhelming the server
- Individual files: Each player gets their own Parquet file for easy access
- Graceful failures: Continues processing other players if one fails after retries
- Progress tracking: Shows current progress and summary statistics
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

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

# Delay between API calls (in seconds)
API_DELAY = 0.75


def setup_data_directory() -> Path:
    """Ensure the player history data directory exists and return the path."""
    data_dir = Path("data/FPL/raw/player_hist")
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Player history directory: {data_dir.absolute()}")
    return data_dir


def load_player_ids() -> List[int]:
    """
    Load player IDs from the bootstrap players.parquet file.
    
    Returns:
        List of player IDs
        
    Raises:
        RuntimeError: If players file doesn't exist or is empty
    """
    players_file = Path("data/FPL/raw/players.parquet")
    
    if not players_file.exists():
        raise RuntimeError(f"Players file not found: {players_file}")
    
    try:
        players_df = pd.read_parquet(players_file)
        
        if players_df.empty:
            raise RuntimeError("Players file is empty")
        
        if 'id' not in players_df.columns:
            raise RuntimeError("Players file missing 'id' column")
        
        player_ids = players_df['id'].tolist()
        logger.info(f"Loaded {len(player_ids)} player IDs from {players_file}")
        return player_ids
        
    except Exception as e:
        raise RuntimeError(f"Failed to load player IDs: {e}")


def fetch_with_retries(url: str, max_retries: int = 5, start_delay: float = 0.5) -> Dict[str, Any]:
    """
    Fetch data from URL with exponential backoff retry logic.
    
    Args:
        url: URL to fetch
        max_retries: Maximum number of retries
        start_delay: Initial delay in seconds
        
    Returns:
        JSON response as dictionary
        
    Raises:
        RuntimeError: If all retries fail
    """
    delay = start_delay
    
    for attempt in range(max_retries):
        try:
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


def fetch_player_summary(player_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch player summary data from FPL API.
    
    Args:
        player_id: FPL player ID
        
    Returns:
        Player summary data dictionary or None if failed
    """
    try:
        client = FPLClient()
        data = client.element_summary(player_id)
        
        if not data:
            logger.warning(f"No data returned for player {player_id}")
            return None
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to fetch player {player_id} via client: {e}")
        # Fallback to direct API call
        try:
            url = f"{FPL_BASE_URL}/element-summary/{player_id}/"
            data = fetch_with_retries(url)
            
            if not data:
                logger.warning(f"No data returned for player {player_id} (direct API)")
                return None
            
            return data
            
        except Exception as fallback_error:
            logger.error(f"Failed to fetch player {player_id} via direct API: {fallback_error}")
            return None


def save_player_history(player_id: int, data: Dict[str, Any], output_dir: Path) -> bool:
    """
    Save player history data to Parquet file.
    
    Args:
        player_id: FPL player ID
        data: Player summary data
        output_dir: Directory to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # The player summary data has three main sections:
        # - fixtures: upcoming fixtures for the player
        # - history: current season performance by gameweek
        # - history_past: past seasons performance
        
        # Convert each section to DataFrame and save
        sections_saved = 0
        
        for section_name, section_data in data.items():
            if section_data and isinstance(section_data, list):
                try:
                    section_df = pd.DataFrame(section_data)
                    
                    if not section_df.empty:
                        # Save each section as a separate file
                        section_path = output_dir / f"{player_id}_{section_name}.parquet"
                        section_df.to_parquet(section_path, index=False)
                        sections_saved += 1
                        logger.debug(f"Saved {section_name} data for player {player_id}")
                        
                except Exception as section_error:
                    logger.warning(f"Failed to save {section_name} for player {player_id}: {section_error}")
        
        if sections_saved == 0:
            logger.warning(f"No valid sections found for player {player_id}")
            return False
        
        # Count seasons if available
        history_past_count = len(data.get('history_past', [])) if data.get('history_past') else 0
        logger.info(f"Saved player {player_id} history ({history_past_count} past seasons, {sections_saved} sections) to {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save player {player_id} history: {e}")
        return False


def process_players(player_ids: List[int], output_dir: Path) -> Dict[str, int]:
    """
    Process all players and fetch their historical data.
    
    Args:
        player_ids: List of player IDs to process
        output_dir: Directory to save player history files
        
    Returns:
        Dictionary with success/failure counts
    """
    total_players = len(player_ids)
    successful = 0
    failed = 0
    
    logger.info(f"Starting to process {total_players} players...")
    
    for i, player_id in enumerate(player_ids, 1):
        logger.info(f"Processing player {i}/{total_players}: ID {player_id}")
        
        # Fetch player summary
        data = fetch_player_summary(player_id)
        
        if data is None:
            failed += 1
            logger.warning(f"Failed to fetch data for player {player_id}")
            continue
        
        # Save player history
        if save_player_history(player_id, data, output_dir):
            successful += 1
        else:
            failed += 1
        
        # Polite delay between API calls (except for the last one)
        if i < total_players:
            time.sleep(API_DELAY)
    
    return {
        'successful': successful,
        'failed': failed,
        'total': total_players
    }


def main():
    """Main function to fetch and save all player historical data."""
    parser = argparse.ArgumentParser(description="Fetch FPL player historical data and save as individual Parquet files")
    
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("Starting FPL player historical data fetch")
    
    try:
        # Setup data directory
        output_dir = setup_data_directory()
        
        # Load player IDs
        player_ids = load_player_ids()
        
        # Process all players
        results = process_players(player_ids, output_dir)
        
        # Summary
        elapsed_time = time.time() - start_time
        logger.info(f"Player historical data fetch completed in {elapsed_time:.1f} seconds")
        logger.info(f"Summary: {results['successful']} successful, {results['failed']} failed out of {results['total']} total players")
        
        if results['failed'] > 0:
            logger.warning(f"{results['failed']} players failed to process - check logs for details")
        
    except Exception as e:
        logger.error(f"Player historical data fetch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 