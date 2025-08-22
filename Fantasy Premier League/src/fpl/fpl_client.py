"""
FPL API Client

A minimal client for the Fantasy Premier League API with caching and retry logic.
Supports both public endpoints and private endpoints (with session cookie).
"""

import os
import time
from typing import Dict, Any, Optional
import requests
import requests_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Install cache for HTTP requests
requests_cache.install_cache(cache_name=".http_cache/fpl", expire_after=300)


class FPLClient:
    """Client for the Fantasy Premier League API."""
    
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    def __init__(self):
        """Initialize the FPL client with session and optional authentication."""
        self.session = requests.Session()
        
        # Set session cookie if available
        session_cookie = os.getenv("FPL_SESSION")
        if session_cookie:
            self.session.cookies.set("session", session_cookie, domain="fantasy.premierleague.com")
    
    def _get(self, url: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Make a GET request with retry logic for rate limiting and server errors.
        
        Args:
            url: The URL to request
            max_retries: Maximum number of retry attempts
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.RequestException: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                response = self.session.get(url)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limited
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + 1  # Exponential backoff
                        time.sleep(wait_time)
                        continue
                elif e.response.status_code >= 500:  # Server error
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + 1
                        time.sleep(wait_time)
                        continue
                raise
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1
                    time.sleep(wait_time)
                    continue
                raise
        
        raise requests.RequestException(f"Failed after {max_retries} retries")
    
    def bootstrap(self) -> Dict[str, Any]:
        """
        Get bootstrap data including players, teams, and gameweek information.
        
        Returns:
            Bootstrap data dictionary
        """
        url = f"{self.BASE_URL}/bootstrap-static/"
        return self._get(url)
    
    def fixtures(self) -> Dict[str, Any]:
        """
        Get all fixtures for the season.
        
        Returns:
            Fixtures data dictionary
        """
        url = f"{self.BASE_URL}/fixtures/"
        return self._get(url)
    
    def event_live(self, gw: int) -> Dict[str, Any]:
        """
        Get live data for a specific gameweek.
        
        Args:
            gw: Gameweek number
            
        Returns:
            Live gameweek data dictionary
        """
        url = f"{self.BASE_URL}/event/{gw}/live/"
        return self._get(url)
    
    def event_fixtures(self, gw: int) -> Dict[str, Any]:
        """
        Get fixtures for a specific gameweek.
        
        Args:
            gw: Gameweek number
            
        Returns:
            Gameweek fixtures data dictionary
        """
        url = f"{self.BASE_URL}/event/{gw}/fixtures/"
        return self._get(url)
    
    def element_summary(self, player_id: int) -> Dict[str, Any]:
        """
        Get detailed summary for a specific player.
        
        Args:
            player_id: FPL player ID
            
        Returns:
            Player summary data dictionary
        """
        url = f"{self.BASE_URL}/element-summary/{player_id}/"
        return self._get(url)
    
    def entry(self, team_id: int) -> Dict[str, Any]:
        """
        Get public information for a specific team.
        
        Args:
            team_id: FPL team ID
            
        Returns:
            Team data dictionary
        """
        url = f"{self.BASE_URL}/entry/{team_id}/"
        return self._get(url)
    
    def league_classic_standings(self, league_id: int) -> Dict[str, Any]:
        """
        Get standings for a classic league.
        
        Args:
            league_id: FPL league ID
            
        Returns:
            League standings data dictionary
        """
        url = f"{self.BASE_URL}/leagues-classic/{league_id}/standings/"
        return self._get(url)
    
    def league_h2h_matches(self, league_id: int, page: int = 1) -> Dict[str, Any]:
        """
        Get H2H league matches with pagination.
        
        Args:
            league_id: FPL league ID
            page: Page number for pagination
            
        Returns:
            H2H matches data dictionary
        """
        url = f"{self.BASE_URL}/leagues-h2h-matches/league/{league_id}/?page={page}"
        return self._get(url)
    
    def entry_picks(self, team_id: int, gw: int) -> Dict[str, Any]:
        """
        Get team picks for a specific gameweek (requires authentication).
        
        Args:
            team_id: FPL team ID
            gw: Gameweek number
            
        Returns:
            Team picks data dictionary
            
        Raises:
            requests.RequestException: If not authenticated
        """
        url = f"{self.BASE_URL}/entry/{team_id}/event/{gw}/picks/"
        return self._get(url)
    
    def entry_transfers(self, team_id: int) -> Dict[str, Any]:
        """
        Get team transfer history (requires authentication).
        
        Args:
            team_id: FPL team ID
            
        Returns:
            Team transfers data dictionary
            
        Raises:
            requests.RequestException: If not authenticated
        """
        url = f"{self.BASE_URL}/entry/{team_id}/transfers/"
        return self._get(url) 