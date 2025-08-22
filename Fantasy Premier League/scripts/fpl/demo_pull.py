"""
FPL Demo Data Pull Script

This script demonstrates how to use the FPLClient to fetch data and save it as parquet files.
Run with: make fpl-bootstrap

Outputs will be written to data/FPL/raw/:
- players.parquet
- teams.parquet  
- fixtures.parquet
- gw1_live.parquet
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path to import FPLClient
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fpl.fpl_client import FPLClient


def main():
    """Main function to fetch FPL data and save as parquet files."""
    # Initialize FPL client
    client = FPLClient()
    
    # Ensure output directory exists
    output_dir = Path("data/FPL/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Fetching FPL bootstrap data...")
    bootstrap_data = client.bootstrap()
    
    # Extract players and teams
    players = bootstrap_data.get("elements", [])
    teams = bootstrap_data.get("teams", [])
    
    print(f"Found {len(players)} players and {len(teams)} teams")
    
    # Save players data
    players_df = pd.json_normalize(players)
    players_output = output_dir / "players.parquet"
    players_df.to_parquet(players_output, index=False)
    print(f"Saved players data to {players_output}")
    
    # Save teams data
    teams_df = pd.json_normalize(teams)
    teams_output = output_dir / "teams.parquet"
    teams_df.to_parquet(teams_output, index=False)
    print(f"Saved teams data to {teams_output}")
    
    # Fetch fixtures
    print("Fetching fixtures data...")
    fixtures_data = client.fixtures()
    fixtures = fixtures_data if isinstance(fixtures_data, list) else fixtures_data.get("fixtures", [])
    
    print(f"Found {len(fixtures)} fixtures")
    
    # Save fixtures data
    fixtures_df = pd.json_normalize(fixtures)
    fixtures_output = output_dir / "fixtures.parquet"
    fixtures_df.to_parquet(fixtures_output, index=False)
    print(f"Saved fixtures data to {fixtures_output}")
    
    # Fetch gameweek 1 live data
    print("Fetching gameweek 1 live data...")
    try:
        gw1_live_data = client.event_live(gw=1)
        gw1_live = gw1_live_data.get("elements", [])
        
        # Handle different data formats for elements
        if isinstance(gw1_live, list):
            # Elements is already a list of dicts
            gw1_live_list = gw1_live
            print(f"Found live data for {len(gw1_live_list)} players in GW1 (list format)")
        elif isinstance(gw1_live, dict):
            # Elements is a dict, convert to list format
            gw1_live_list = [{"player_id": k, **v} for k, v in gw1_live.items()]
            print(f"Found live data for {len(gw1_live_list)} players in GW1 (dict format)")
        else:
            # Neither list nor dict, use empty list
            gw1_live_list = []
            print("No live data available for GW1 (unexpected format)")
        
        # Save GW1 live data
        if gw1_live_list:
            gw1_live_df = pd.json_normalize(gw1_live_list)
            gw1_live_output = output_dir / "gw1_live.parquet"
            gw1_live_df.to_parquet(gw1_live_output, index=False)
            print(f"Saved GW1 live data to {gw1_live_output}")
        else:
            print("No live data available for GW1")
            
    except Exception as e:
        print(f"Could not fetch GW1 live data: {e}")
    
    print("Data pull complete!")


if __name__ == "__main__":
    main() 