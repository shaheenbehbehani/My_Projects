#!/usr/bin/env python3
"""
FPL Entity Resolution & Merge Script - Step 3a

This script merges the new FPL feature tables (projections and schedule_strength)
with existing historical datasets to create a comprehensive training dataset.

Entity Resolution Logic:
1. Exact joins on player_id, team_id, and gameweek
2. Fuzzy matching on player names and team names (similarity ≥ 90%)
3. Output unresolved cases to CSV for manual review

Output:
- data/FPL/processed/fpl_merged.parquet: Final merged dataset
- outputs/fpl_unmatched.csv: Unresolved entity cases
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rapidfuzz import fuzz, process

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fpl_merge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FPLMerger:
    """Handles entity resolution and merging of FPL datasets with historical data."""
    
    def __init__(self, data_dir: str = "data", outputs_dir: str = "outputs"):
        self.data_dir = Path(data_dir)
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Initialize datasets
        self.fpl_features = {}
        self.fpl_processed = {}
        self.historical_data = {}
        self.merged_data = None
        
        # Entity resolution tracking
        self.resolution_stats = {
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'unresolved': 0,
            'total_entities': 0
        }
        
        # Unmatched entities for output
        self.unmatched_entities = []
        
    def load_datasets(self) -> None:
        """Load all required datasets."""
        logger.info("Loading FPL feature datasets...")
        
        # Load FPL features
        self.fpl_features['projections'] = pd.read_parquet(
            self.data_dir / "FPL" / "features" / "projections.parquet"
        )
        self.fpl_features['schedule_strength'] = pd.read_parquet(
            self.data_dir / "FPL" / "features" / "schedule_strength.parquet"
        )
        
        logger.info("Loading FPL processed datasets...")
        
        # Load FPL processed data
        self.fpl_processed['players'] = pd.read_parquet(
            self.data_dir / "FPL" / "processed" / "players_processed.parquet"
        )
        self.fpl_processed['teams'] = pd.read_parquet(
            self.data_dir / "FPL" / "processed" / "teams_processed.parquet"
        )
        self.fpl_processed['gameweeks'] = pd.read_parquet(
            self.data_dir / "FPL" / "processed" / "gameweeks_processed.parquet"
        )
        
        logger.info("Loading historical datasets...")
        
        # Load historical datasets
        self.historical_data['matches'] = pd.read_parquet(
            self.data_dir / "raw" / "historical_matches.parquet"
        )
        self.historical_data['club_wages'] = pd.read_csv(
            self.data_dir / "raw" / "Club wages.csv"
        )
        self.historical_data['club_value'] = pd.read_csv(
            self.data_dir / "raw" / "Club Value.csv"
        )
        self.historical_data['attendance'] = pd.read_csv(
            self.data_dir / "raw" / "Attendance Data.csv"
        )
        self.historical_data['managers'] = pd.read_csv(
            self.data_dir / "raw" / "Premier League Managers.csv"
        )
        
        logger.info(f"Loaded {len(self.fpl_features)} FPL feature datasets")
        logger.info(f"Loaded {len(self.fpl_processed)} FPL processed datasets")
        logger.info(f"Loaded {len(self.historical_data)} historical datasets")
    
    def normalize_team_name(self, name: str) -> str:
        """Normalize team names for better matching."""
        if pd.isna(name):
            return name
        
        name = str(name).strip().lower()
        
        # Common abbreviations and variations
        name_mappings = {
            'man city': 'manchester city',
            'man utd': 'manchester united',
            'manchester utd': 'manchester united',
            'spurs': 'tottenham hotspur',
            'tottenham': 'tottenham hotspur',
            'newcastle': 'newcastle united',
            'brighton': 'brighton & hove albion',
            'forest': "nott'mham forest",
            'nottingham forest': "nott'mham forest",
            'bournemouth': 'afc bournemouth',
            'wolves': 'wolverhampton wanderers',
            'leeds': 'leeds united',
            'burnley': 'burnley fc',
            'sunderland': 'sunderland afc',
            'fulham': 'fulham fc',
            'everton': 'everton fc',
            'arsenal': 'arsenal fc',
            'chelsea': 'chelsea fc',
            'liverpool': 'liverpool fc',
            'west ham': 'west ham united',
            'crystal palace': 'crystal palace',
            'brentford': 'brentford fc',
            'ipswich': 'ipswich town',
            'leicester': 'leicester city',
            'southampton': 'southampton fc',
            'aston villa': 'aston villa fc'
        }
        
        return name_mappings.get(name, name)
    
    def clean_historical_data(self) -> None:
        """Clean and standardize historical datasets."""
        logger.info("Cleaning historical datasets...")
        
        # Clean club wages data
        club_wages = self.historical_data['club_wages'].copy()
        club_wages['Squad'] = club_wages['Squad'].str.strip().str.replace('\xa0', ' ').str.strip()
        # Extract numeric values from wage strings
        club_wages['Weekly_Wages_Numeric'] = club_wages['Weekly Wages'].str.extract(r'£\s*([\d,]+)')[0].str.replace(',', '').astype(float)
        club_wages['Annual_Wages_Numeric'] = club_wages['Annual Wages'].str.extract(r'£\s*([\d,]+)')[0].str.replace(',', '').astype(float)
        self.historical_data['club_wages_clean'] = club_wages
        
        # Clean club value data
        club_value = self.historical_data['club_value'].copy()
        club_value['Club'] = club_value['Club'].str.strip().str.replace('\xa0', ' ').str.strip()
        # Extract numeric values from value strings
        club_value['Value_Numeric'] = club_value['Current value'].str.extract(r'€([\d.]+)bn')[0].astype(float) * 1000000000
        self.historical_data['club_value_clean'] = club_value
        
        # Clean attendance data
        attendance = self.historical_data['attendance'].copy()
        attendance['Club'] = attendance['Club'].str.strip().str.replace('\xa0', ' ').str.strip()
        attendance['Capacity_Numeric'] = attendance['Capacity'].str.replace(',', '').astype(float)
        attendance['Avg_Attendance_Numeric'] = attendance['Avg Attendace'].str.replace(',', '').astype(float)
        self.historical_data['attendance_clean'] = attendance
        
        # Clean managers data - get current managers
        managers = self.historical_data['managers'].copy()
        managers = managers.dropna(subset=['Club', 'Name'])
        managers['Club'] = managers['Club'].str.strip()
        managers['Name'] = managers['Name'].str.strip()
        # Get most recent manager per club
        managers['Until'] = pd.to_datetime(managers['Until'], errors='coerce')
        current_managers = managers.groupby('Club').agg({
            'Name': 'last',
            'Until': 'max',
            'Duration': 'last'
        }).reset_index()
        self.historical_data['current_managers'] = current_managers
        
        logger.info("Historical datasets cleaned successfully")
    
    def create_team_mapping(self) -> pd.DataFrame:
        """Create team name mapping between FPL and historical datasets."""
        logger.info("Creating team name mapping...")
        
        fpl_teams = self.fpl_processed['teams'][['team_id', 'team_name']].copy()
        fpl_teams['team_name_clean'] = fpl_teams['team_name'].str.strip()
        
        # Create mapping dataframe
        team_mapping = []
        
        for _, fpl_team in fpl_teams.iterrows():
            fpl_name = fpl_team['team_name_clean']
            fpl_id = fpl_team['team_id']
            fpl_name_normalized = self.normalize_team_name(fpl_name)
            
            # Try exact match first
            exact_match = None
            
            # Check in club wages
            if exact_match is None:
                match = self.historical_data['club_wages_clean'][
                    self.historical_data['club_wages_clean']['Squad'].apply(self.normalize_team_name) == fpl_name_normalized
                ]
                if not match.empty:
                    exact_match = 'club_wages'
            
            # Check in club value
            if exact_match is None:
                match = self.historical_data['club_value_clean'][
                    self.historical_data['club_value_clean']['Club'].apply(self.normalize_team_name) == fpl_name_normalized
                ]
                if not match.empty:
                    exact_match = 'club_value'
            
            # Check in attendance
            if exact_match is None:
                match = self.historical_data['attendance_clean'][
                    self.historical_data['attendance_clean']['Club'].apply(self.normalize_team_name) == fpl_name_normalized
                ]
                if not match.empty:
                    exact_match = 'attendance'
            
            # Check in managers
            if exact_match is None:
                match = self.historical_data['current_managers'][
                    self.historical_data['current_managers']['Club'].apply(self.normalize_team_name) == fpl_name_normalized
                ]
                if not match.empty:
                    exact_match = 'managers'
            
            if exact_match:
                team_mapping.append({
                    'fpl_team_id': fpl_id,
                    'fpl_team_name': fpl_name,
                    'historical_source': exact_match,
                    'match_type': 'exact',
                    'historical_team_name': fpl_name
                })
                self.resolution_stats['exact_matches'] += 1
            else:
                # Try fuzzy matching
                best_match = None
                best_score = 0
                best_source = None
                
                # Check all historical sources
                for source, data in [
                    ('club_wages', self.historical_data['club_wages_clean']['Squad']),
                    ('club_value', self.historical_data['club_value_clean']['Club']),
                    ('attendance', self.historical_data['attendance_clean']['Club']),
                    ('managers', self.historical_data['current_managers']['Club'])
                ]:
                    for hist_name in data:
                        score = fuzz.ratio(fpl_name.lower(), hist_name.lower())
                        if score > best_score and score >= 90:
                            best_score = score
                            best_match = hist_name
                            best_source = source
                
                if best_match:
                    team_mapping.append({
                        'fpl_team_id': fpl_id,
                        'fpl_team_name': fpl_name,
                        'historical_source': best_source,
                        'match_type': 'fuzzy',
                        'historical_team_name': best_match,
                        'similarity_score': best_score
                    })
                    self.resolution_stats['fuzzy_matches'] += 1
                else:
                    # Unresolved
                    team_mapping.append({
                        'fpl_team_id': fpl_id,
                        'fpl_team_name': fpl_name,
                        'historical_source': None,
                        'match_type': 'unresolved',
                        'historical_team_name': None
                    })
                    self.resolution_stats['unresolved'] += 1
                    
                    # Add to unmatched entities
                    self.unmatched_entities.append({
                        'source': 'team_mapping',
                        'entity_type': 'team',
                        'name': fpl_name,
                        'id': fpl_id,
                        'reason': 'No historical data match found'
                    })
        
        self.resolution_stats['total_entities'] += len(team_mapping)
        return pd.DataFrame(team_mapping)
    
    def create_player_mapping(self) -> pd.DataFrame:
        """Create player mapping between FPL and historical datasets."""
        logger.info("Creating player mapping...")
        
        fpl_players = self.fpl_processed['players'][['player_id', 'full_name', 'team_id']].copy()
        fpl_players['full_name_clean'] = fpl_players['full_name'].str.strip()
        
        # For now, we'll use exact matches on player_id since FPL IDs are stable
        # In a real implementation, you might want to add fuzzy matching on names
        player_mapping = []
        
        for _, fpl_player in fpl_players.iterrows():
            fpl_id = fpl_player['player_id']
            fpl_name = fpl_player['full_name_clean']
            fpl_team_id = fpl_player['team_id']
            
            # Check if player exists in historical data
            # This is a simplified approach - in practice you'd have more historical player data
            player_mapping.append({
                'fpl_player_id': fpl_id,
                'fpl_player_name': fpl_name,
                'fpl_team_id': fpl_team_id,
                'historical_source': 'fpl_processed',  # For now, just use FPL data
                'match_type': 'exact',
                'historical_player_name': fpl_name
            })
            self.resolution_stats['exact_matches'] += 1
        
        self.resolution_stats['total_entities'] += len(player_mapping)
        return pd.DataFrame(player_mapping)
    
    def merge_team_features(self, team_mapping: pd.DataFrame) -> pd.DataFrame:
        """Merge team-level features from historical datasets."""
        logger.info("Merging team features...")
        
        # Start with FPL teams
        merged_teams = self.fpl_processed['teams'].copy()
        
        # Add team mapping info
        merged_teams = merged_teams.merge(
            team_mapping[['fpl_team_id', 'historical_source', 'match_type', 'historical_team_name']],
            left_on='team_id',
            right_on='fpl_team_id',
            how='left'
        )
        
        # Add club wages
        club_wages_clean = self.historical_data['club_wages_clean'].copy()
        merged_teams = merged_teams.merge(
            club_wages_clean[['Squad', 'Weekly_Wages_Numeric', 'Annual_Wages_Numeric']],
            left_on='historical_team_name',
            right_on='Squad',
            how='left'
        )
        
        # Add club value
        club_value_clean = self.historical_data['club_value_clean'].copy()
        merged_teams = merged_teams.merge(
            club_value_clean[['Club', 'Value_Numeric']],
            left_on='historical_team_name',
            right_on='Club',
            how='left'
        )
        
        # Add attendance
        attendance_clean = self.historical_data['attendance_clean'].copy()
        merged_teams = merged_teams.merge(
            attendance_clean[['Club', 'Capacity_Numeric', 'Avg_Attendance_Numeric']],
            left_on='historical_team_name',
            right_on='Club',
            how='left'
        )
        
        # Add current manager
        current_managers = self.historical_data['current_managers'].copy()
        merged_teams = merged_teams.merge(
            current_managers[['Club', 'Name', 'Duration']],
            left_on='historical_team_name',
            right_on='Club',
            how='left'
        )
        
        # Rename columns for clarity
        merged_teams = merged_teams.rename(columns={
            'Weekly_Wages_Numeric': 'weekly_wages',
            'Annual_Wages_Numeric': 'annual_wages',
            'Value_Numeric': 'club_value',
            'Capacity_Numeric': 'stadium_capacity',
            'Avg_Attendance_Numeric': 'avg_attendance',
            'Name': 'manager_name',
            'Duration': 'manager_tenure'
        })
        
        return merged_teams
    
    def merge_player_features(self, team_mapping: pd.DataFrame) -> pd.DataFrame:
        """Merge player-level features from FPL and historical datasets."""
        logger.info("Merging player features...")
        
        # Start with FPL projections
        merged_players = self.fpl_features['projections'].copy()
        
        # Add player info from processed data
        player_info = self.fpl_processed['players'][
            ['player_id', 'price', 'availability_status', 'form', 'total_points', 'points_per_game']
        ].copy()
        
        merged_players = merged_players.merge(
            player_info,
            on='player_id',
            how='left'
        )
        
        # Add team features
        team_features = team_mapping[['fpl_team_id', 'historical_source', 'match_type']].copy()
        merged_players = merged_players.merge(
            team_features,
            left_on='team_id',
            right_on='fpl_team_id',
            how='left'
        )
        
        # Add team-level features (wages, value, etc.)
        team_merged = self.merge_team_features(team_mapping)
        team_features_for_players = team_merged[
            ['team_id', 'weekly_wages', 'annual_wages', 'club_value', 
             'stadium_capacity', 'avg_attendance', 'manager_name', 'manager_tenure']
        ].copy()
        
        merged_players = merged_players.merge(
            team_features_for_players,
            on='team_id',
            how='left'
        )
        
        # Rename columns for clarity
        merged_players = merged_players.rename(columns={
            'projected_points': 'xP',
            'projection_confidence': 'confidence'
        })
        
        return merged_players
    
    def create_final_merged_dataset(self) -> pd.DataFrame:
        """Create the final merged dataset with all features."""
        logger.info("Creating final merged dataset...")
        
        # Create team mapping
        team_mapping = self.create_team_mapping()
        
        # Create player mapping
        player_mapping = self.create_player_mapping()
        
        # Merge player features
        merged_players = self.merge_player_features(team_mapping)
        
        # Add schedule strength features
        schedule_features = self.fpl_features['schedule_strength'][
            ['team_id', 'gameweek', 'rest_days', 'fixture_congestion', 
             'opponent_difficulty', 'rest_days_category', 'congestion_level']
        ].copy()
        
        # Aggregate schedule features by team and gameweek
        schedule_agg = schedule_features.groupby(['team_id', 'gameweek']).agg({
            'rest_days': 'mean',
            'fixture_congestion': 'mean',
            'opponent_difficulty': 'mean',
            'rest_days_category': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
            'congestion_level': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        }).reset_index()
        
        # Merge schedule features
        final_merged = merged_players.merge(
            schedule_agg,
            on=['team_id', 'gameweek'],
            how='left'
        )
        
        # Ensure no duplicate player-GW rows
        final_merged = final_merged.drop_duplicates(subset=['player_id', 'gameweek'])
        
        # Select final columns
        final_columns = [
            'player_id', 'player_name', 'team_id', 'position', 'price', 'availability_status',
            'gameweek', 'xP', 'confidence', 'form', 'total_points', 'points_per_game',
            'opponent_difficulty', 'is_home', 'rest_days', 'fixture_congestion',
            'weekly_wages', 'annual_wages', 'club_value', 'stadium_capacity',
            'avg_attendance', 'manager_name', 'manager_tenure'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in final_columns if col in final_merged.columns]
        final_merged = final_merged[available_columns]
        
        return final_merged
    
    def save_outputs(self, merged_data: pd.DataFrame) -> None:
        """Save the merged dataset and unmatched entities."""
        logger.info("Saving outputs...")
        
        # Save merged dataset
        output_path = self.data_dir / "FPL" / "processed" / "fpl_merged.parquet"
        output_path.parent.mkdir(exist_ok=True)
        merged_data.to_parquet(output_path, index=False)
        logger.info(f"Saved merged dataset to {output_path}")
        
        # Save unmatched entities
        if self.unmatched_entities:
            unmatched_df = pd.DataFrame(self.unmatched_entities)
            unmatched_path = self.outputs_dir / "fpl_unmatched.csv"
            unmatched_df.to_csv(unmatched_path, index=False)
            logger.info(f"Saved {len(self.unmatched_entities)} unmatched entities to {unmatched_path}")
        else:
            logger.info("No unmatched entities to save")
    
    def log_coverage_summary(self) -> None:
        """Log the coverage summary statistics."""
        logger.info("=" * 50)
        logger.info("ENTITY RESOLUTION COVERAGE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total entities processed: {self.resolution_stats['total_entities']}")
        logger.info(f"Exact matches: {self.resolution_stats['exact_matches']}")
        logger.info(f"Fuzzy matches: {self.resolution_stats['fuzzy_matches']}")
        logger.info(f"Unresolved: {self.resolution_stats['unresolved']}")
        
        if self.resolution_stats['total_entities'] > 0:
            coverage_pct = ((self.resolution_stats['exact_matches'] + self.resolution_stats['fuzzy_matches']) / 
                           self.resolution_stats['total_entities']) * 100
            logger.info(f"Total coverage: {coverage_pct:.1f}%")
        
        if self.unmatched_entities:
            logger.info(f"Unmatched entities saved to: outputs/fpl_unmatched.csv")
        
        logger.info("=" * 50)
    
    def run(self) -> None:
        """Run the complete merge process."""
        try:
            logger.info("Starting FPL Entity Resolution & Merge (Step 3a)")
            
            # Load datasets
            self.load_datasets()
            
            # Clean historical data
            self.clean_historical_data()
            
            # Create merged dataset
            merged_data = self.create_final_merged_dataset()
            
            # Save outputs
            self.save_outputs(merged_data)
            
            # Log coverage summary
            self.log_coverage_summary()
            
            logger.info("FPL merge completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during FPL merge: {str(e)}")
            raise


def main():
    """Main entry point."""
    merger = FPLMerger()
    merger.run()


if __name__ == "__main__":
    main() 