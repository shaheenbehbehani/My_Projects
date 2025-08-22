"""
Tests for FPL Client

Smoke tests using pytest and requests-mock to avoid real network calls.
"""

import pytest
import requests
import requests_mock
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path to import FPLClient
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fpl.fpl_client import FPLClient


class TestFPLClient:
    """Test cases for FPLClient."""
    
    def test_client_initialization(self):
        """Test that FPLClient initializes correctly."""
        client = FPLClient()
        assert client.BASE_URL == "https://fantasy.premierleague.com/api"
        assert hasattr(client, 'session')
        assert isinstance(client.session, requests.Session)
    
    @patch.dict('os.environ', {'FPL_SESSION': 'test_session_cookie'})
    def test_client_with_session_cookie(self):
        """Test that FPLClient sets session cookie when environment variable is present."""
        client = FPLClient()
        cookie = client.session.cookies.get("session", domain="fantasy.premierleague.com")
        assert cookie is not None
        assert cookie == "test_session_cookie"
    
    def test_client_without_session_cookie(self):
        """Test that FPLClient works without session cookie."""
        with patch.dict('os.environ', {}, clear=True):
            client = FPLClient()
            cookie = client.session.cookies.get("session", domain="fantasy.premierleague.com")
            assert cookie is None
    
    def test_bootstrap_endpoint(self):
        """Test that bootstrap() hits the expected URL and returns data."""
        with requests_mock.Mocker() as m:
            # Mock the bootstrap endpoint
            mock_data = {
                "events": [],
                "game_settings": {},
                "phases": [],
                "teams": [],
                "total_players": 0,
                "elements": []
            }
            m.get(
                "https://fantasy.premierleague.com/api/bootstrap-static/",
                json=mock_data
            )
            
            client = FPLClient()
            result = client.bootstrap()
            
            assert result == mock_data
            assert m.call_count == 1
    
    def test_event_live_endpoint(self):
        """Test that event_live(gw=1) hits the expected URL."""
        with requests_mock.Mocker() as m:
            # Mock the event live endpoint
            mock_data = {"elements": {}}
            m.get(
                "https://fantasy.premierleague.com/api/event/1/live/",
                json=mock_data
            )
            
            client = FPLClient()
            result = client.event_live(gw=1)
            
            assert result == mock_data
            assert m.call_count == 1
    
    def test_fixtures_endpoint(self):
        """Test that fixtures() hits the expected URL."""
        with requests_mock.Mocker() as m:
            # Mock the fixtures endpoint
            mock_data = []
            m.get(
                "https://fantasy.premierleague.com/api/fixtures/",
                json=mock_data
            )
            
            client = FPLClient()
            result = client.fixtures()
            
            assert result == mock_data
            assert m.call_count == 1
    
    def test_event_fixtures_endpoint(self):
        """Test that event_fixtures(gw=2) hits the expected URL."""
        with requests_mock.Mocker() as m:
            # Mock the event fixtures endpoint
            mock_data = []
            m.get(
                "https://fantasy.premierleague.com/api/event/2/fixtures/",
                json=mock_data
            )
            
            client = FPLClient()
            result = client.event_fixtures(gw=2)
            
            assert result == mock_data
            assert m.call_count == 1
    
    def test_element_summary_endpoint(self):
        """Test that element_summary(player_id=1) hits the expected URL."""
        with requests_mock.Mocker() as m:
            # Mock the element summary endpoint
            mock_data = {"history": [], "fixtures": []}
            m.get(
                "https://fantasy.premierleague.com/api/element-summary/1/",
                json=mock_data
            )
            
            client = FPLClient()
            result = client.element_summary(player_id=1)
            
            assert result == mock_data
            assert m.call_count == 1
    
    def test_entry_endpoint(self):
        """Test that entry(team_id=1) hits the expected URL."""
        with requests_mock.Mocker() as m:
            # Mock the entry endpoint
            mock_data = {"id": 1, "player_first_name": "Test", "player_last_name": "User"}
            m.get(
                "https://fantasy.premierleague.com/api/entry/1/",
                json=mock_data
            )
            
            client = FPLClient()
            result = client.entry(team_id=1)
            
            assert result == mock_data
            assert m.call_count == 1
    
    def test_league_classic_standings_endpoint(self):
        """Test that league_classic_standings(league_id=1) hits the expected URL."""
        with requests_mock.Mocker() as m:
            # Mock the league classic standings endpoint
            mock_data = {"league": {}, "standings": {"results": []}}
            m.get(
                "https://fantasy.premierleague.com/api/leagues-classic/1/standings/",
                json=mock_data
            )
            
            client = FPLClient()
            result = client.league_classic_standings(league_id=1)
            
            assert result == mock_data
            assert m.call_count == 1
    
    def test_league_h2h_matches_endpoint(self):
        """Test that league_h2h_matches(league_id=1, page=1) hits the expected URL."""
        with requests_mock.Mocker() as m:
            # Mock the league H2H matches endpoint
            mock_data = {"matches": []}
            m.get(
                "https://fantasy.premierleague.com/api/leagues-h2h-matches/league/1/?page=1",
                json=mock_data
            )
            
            client = FPLClient()
            result = client.league_h2h_matches(league_id=1, page=1)
            
            assert result == mock_data
            assert m.call_count == 1
    
    def test_entry_picks_endpoint(self):
        """Test that entry_picks(team_id=1, gw=1) hits the expected URL."""
        with requests_mock.Mocker() as m:
            # Mock the entry picks endpoint
            mock_data = {"picks": [], "entry_history": {}}
            m.get(
                "https://fantasy.premierleague.com/api/entry/1/event/1/picks/",
                json=mock_data
            )
            
            client = FPLClient()
            result = client.entry_picks(team_id=1, gw=1)
            
            assert result == mock_data
            assert m.call_count == 1
    
    def test_entry_transfers_endpoint(self):
        """Test that entry_transfers(team_id=1) hits the expected URL."""
        with requests_mock.Mocker() as m:
            # Mock the entry transfers endpoint
            mock_data = []
            m.get(
                "https://fantasy.premierleague.com/api/entry/1/transfers/",
                json=mock_data
            )
            
            client = FPLClient()
            result = client.entry_transfers(team_id=1)
            
            assert result == mock_data
            assert m.call_count == 1 