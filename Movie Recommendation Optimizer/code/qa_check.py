#!/usr/bin/env python3
"""
Quick QA Script for Step 1a Data
Checks data quality, non-null keys, row counts, and score ranges
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import glob

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/qa_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QAChecker:
    def __init__(self):
        self.results = {}
        
    def check_data_availability(self):
        """Check what data files are available"""
        logger.info("🔍 Checking data availability...")
        
        data_sources = {
            'TMDB': 'data/normalized/tmdb_movies_*.parquet',
            'IMDb': 'data/normalized/imdb_movies_*.parquet',
            'MovieLens': 'data/normalized/movielens_movies_*.parquet',
            'Rotten Tomatoes': 'data/normalized/rottentomatoes_movies_*.parquet',
            'ID Bridge': 'data/normalized/id_bridge*.parquet'
        }
        
        availability = {}
        for source, pattern in data_sources.items():
            files = glob.glob(pattern)
            if files:
                latest_file = max(files)
                availability[source] = {
                    'available': True,
                    'latest_file': latest_file,
                    'file_count': len(files)
                }
                logger.info(f"✅ {source}: {len(files)} files, latest: {Path(latest_file).name}")
            else:
                availability[source] = {
                    'available': False,
                    'latest_file': None,
                    'file_count': 0
                }
                logger.warning(f"⚠️  {source}: No files found")
        
        self.results['availability'] = availability
        return availability
    
    def load_latest_data(self, source_name):
        """Load the latest data file for a source"""
        if not self.results.get('availability', {}).get(source_name, {}).get('available', False):
            return None
        
        pattern = f'data/normalized/{source_name.lower().replace(" ", "").replace("+", "_")}_movies_*.parquet'
        files = glob.glob(pattern)
        
        if not files:
            # Try ID bridge pattern
            if source_name == 'ID Bridge':
                files = glob.glob('data/normalized/id_bridge*.parquet')
        
        if files:
            latest_file = max(files)
            try:
                df = pd.read_parquet(latest_file)
                logger.info(f"📊 Loaded {source_name}: {len(df)} records, {len(df.columns)} columns")
                return df
            except Exception as e:
                logger.error(f"❌ Error loading {source_name}: {e}")
                return None
        
        return None
    
    def check_row_counts(self):
        """Check row counts for each data source"""
        logger.info("\n📊 Checking row counts...")
        
        row_counts = {}
        for source in ['TMDB', 'IMDb', 'MovieLens', 'Rotten Tomatoes', 'ID Bridge']:
            df = self.load_latest_data(source)
            if df is not None:
                row_counts[source] = len(df)
                logger.info(f"📈 {source}: {len(df):,} records")
            else:
                row_counts[source] = 0
                logger.warning(f"⚠️  {source}: No data available")
        
        self.results['row_counts'] = row_counts
        return row_counts
    
    def check_non_null_keys(self):
        """Check non-null values for key fields"""
        logger.info("\n🔑 Checking non-null keys...")
        
        key_analysis = {}
        
        # Check each source
        for source in ['TMDB', 'IMDb', 'MovieLens', 'Rotten Tomatoes']:
            df = self.load_latest_data(source)
            if df is None:
                continue
            
            # Define key fields for each source
            if source == 'TMDB':
                key_fields = ['tmdb_id', 'title', 'vote_average', 'genres']
            elif source == 'IMDb':
                key_fields = ['tconst', 'title', 'averageRating', 'genres']
            elif source == 'MovieLens':
                key_fields = ['movieId', 'title', 'avg_rating', 'genres']
            elif source == 'Rotten Tomatoes':
                key_fields = ['rt_id', 'title', 'tomatometer', 'genres']
            
            # Check non-null counts
            non_null_counts = {}
            for field in key_fields:
                if field in df.columns:
                    non_null_count = df[field].notna().sum()
                    total_count = len(df)
                    percentage = (non_null_count / total_count) * 100
                    non_null_counts[field] = {
                        'non_null': non_null_count,
                        'total': total_count,
                        'percentage': percentage
                    }
                    logger.info(f"  {source}.{field}: {non_null_count:,}/{total_count:,} ({percentage:.1f}%)")
                else:
                    non_null_counts[field] = {'non_null': 0, 'total': 0, 'percentage': 0}
                    logger.warning(f"  {source}.{field}: Field not found")
            
            key_analysis[source] = non_null_counts
        
        self.results['key_analysis'] = key_analysis
        return key_analysis
    
    def check_score_ranges(self):
        """Check score ranges for rating fields"""
        logger.info("\n⭐ Checking score ranges...")
        
        score_analysis = {}
        
        # Check each source
        for source in ['TMDB', 'IMDb', 'MovieLens', 'Rotten Tomatoes']:
            df = self.load_latest_data(source)
            if df is None:
                continue
            
            source_scores = {}
            
            if source == 'TMDB':
                if 'vote_average' in df.columns:
                    scores = df['vote_average'].dropna()
                    if len(scores) > 0:
                        source_scores['vote_average'] = {
                            'min': scores.min(),
                            'max': scores.max(),
                            'mean': scores.mean(),
                            'count': len(scores)
                        }
                        logger.info(f"  {source}.vote_average: {scores.min():.1f} - {scores.max():.1f} (mean: {scores.mean():.2f})")
            
            elif source == 'IMDb':
                if 'averageRating' in df.columns:
                    scores = df['averageRating'].dropna()
                    if len(scores) > 0:
                        source_scores['averageRating'] = {
                            'min': scores.min(),
                            'max': scores.max(),
                            'mean': scores.mean(),
                            'count': len(scores)
                        }
                        logger.info(f"  {source}.averageRating: {scores.min():.1f} - {scores.max():.1f} (mean: {scores.mean():.2f})")
            
            elif source == 'MovieLens':
                if 'avg_rating' in df.columns:
                    scores = df['avg_rating'].dropna()
                    if len(scores) > 0:
                        source_scores['avg_rating'] = {
                            'min': scores.min(),
                            'max': scores.max(),
                            'mean': scores.mean(),
                            'count': len(scores)
                        }
                        logger.info(f"  {source}.avg_rating: {scores.min():.1f} - {scores.max():.1f} (mean: {scores.mean():.2f})")
            
            elif source == 'Rotten Tomatoes':
                if 'tomatometer' in df.columns:
                    scores = df['tomatometer'].dropna()
                    if len(scores) > 0:
                        source_scores['tomatometer'] = {
                            'min': scores.min(),
                            'max': scores.max(),
                            'mean': scores.mean(),
                            'count': len(scores)
                        }
                        logger.info(f"  {source}.tomatometer: {scores.min():.1f} - {scores.max():.1f} (mean: {scores.mean():.2f})")
                
                if 'audience_score' in df.columns:
                    scores = df['audience_score'].dropna()
                    if len(scores) > 0:
                        source_scores['audience_score'] = {
                            'min': scores.min(),
                            'max': scores.max(),
                            'mean': scores.mean(),
                            'count': len(scores)
                        }
                        logger.info(f"  {source}.audience_score: {scores.min():.1f} - {scores.max():.1f} (mean: {scores.mean():.2f})")
            
            score_analysis[source] = source_scores
        
        self.results['score_analysis'] = score_analysis
        return score_analysis
    
    def check_id_bridge(self):
        """Check ID bridge table quality"""
        logger.info("\n🌉 Checking ID bridge table...")
        
        bridge_df = self.load_latest_data('ID Bridge')
        if bridge_df is None:
            logger.warning("⚠️  ID bridge table not available")
            return None
        
        bridge_analysis = {
            'total_records': len(bridge_df),
            'id_coverage': {}
        }
        
        # Check coverage for each ID type
        id_fields = ['movieId', 'imdbId', 'tconst', 'tmdbId', 'rt_id']
        for field in id_fields:
            if field in bridge_df.columns:
                non_null_count = bridge_df[field].notna().sum()
                percentage = (non_null_count / len(bridge_df)) * 100
                bridge_analysis['id_coverage'][field] = {
                    'non_null': non_null_count,
                    'total': len(bridge_df),
                    'percentage': percentage
                }
                logger.info(f"  {field}: {non_null_count:,}/{len(bridge_df):,} ({percentage:.1f}%)")
        
        # Check movies with multiple IDs
        multi_id_count = 0
        for _, row in bridge_df.iterrows():
            id_count = sum(1 for field in id_fields if field in bridge_df.columns and pd.notna(row.get(field)))
            if id_count >= 2:
                multi_id_count += 1
        
        bridge_analysis['multi_id_movies'] = multi_id_count
        logger.info(f"  Movies with ≥2 IDs: {multi_id_count:,}/{len(bridge_df):,}")
        
        self.results['bridge_analysis'] = bridge_analysis
        return bridge_analysis
    
    def generate_summary_report(self):
        """Generate a comprehensive QA summary report"""
        logger.info("\n" + "="*60)
        logger.info("📋 QA SUMMARY REPORT")
        logger.info("="*60)
        
        # Data availability
        logger.info("\n📁 DATA AVAILABILITY:")
        for source, info in self.results.get('availability', {}).items():
            status = "✅ Available" if info['available'] else "❌ Not Available"
            logger.info(f"  {source}: {status}")
        
        # Row counts
        logger.info("\n📊 ROW COUNTS:")
        for source, count in self.results.get('row_counts', {}).items():
            if count > 0:
                logger.info(f"  {source}: {count:,} records")
            else:
                logger.warning(f"  {source}: No data")
        
        # Key field quality
        logger.info("\n🔑 KEY FIELD QUALITY:")
        for source, fields in self.results.get('key_analysis', {}).items():
            logger.info(f"  {source}:")
            for field, stats in fields.items():
                if stats['total'] > 0:
                    quality = "✅" if stats['percentage'] >= 90 else "⚠️" if stats['percentage'] >= 70 else "❌"
                    logger.info(f"    {quality} {field}: {stats['percentage']:.1f}% complete")
        
        # Score ranges
        logger.info("\n⭐ SCORE RANGES:")
        for source, scores in self.results.get('score_analysis', {}).items():
            logger.info(f"  {source}:")
            for field, stats in scores.items():
                logger.info(f"    {field}: {stats['min']:.1f} - {stats['max']:.1f} (mean: {stats['mean']:.2f})")
        
        # ID bridge summary
        if 'bridge_analysis' in self.results:
            bridge = self.results['bridge_analysis']
            logger.info(f"\n🌉 ID BRIDGE SUMMARY:")
            logger.info(f"  Total records: {bridge['total_records']:,}")
            logger.info(f"  Movies with ≥2 IDs: {bridge['multi_id_movies']:,}")
            
            for field, coverage in bridge['id_coverage'].items():
                quality = "✅" if coverage['percentage'] >= 80 else "⚠️" if coverage['percentage'] >= 50 else "❌"
                logger.info(f"    {quality} {field}: {coverage['percentage']:.1f}% coverage")
        
        # Overall assessment
        logger.info(f"\n🎯 OVERALL ASSESSMENT:")
        total_sources = len(self.results.get('availability', {}))
        available_sources = sum(1 for info in self.results.get('availability', {}).values() if info['available'])
        
        if available_sources == total_sources:
            logger.info("  ✅ All data sources available and processed")
        elif available_sources > 0:
            logger.info(f"  ⚠️  {available_sources}/{total_sources} data sources available")
        else:
            logger.error("  ❌ No data sources available")
        
        if 'bridge_analysis' in self.results:
            bridge = self.results['bridge_analysis']
            if bridge['total_records'] > 0:
                logger.info(f"  ✅ ID bridge table created with {bridge['total_records']:,} records")
            else:
                logger.warning("  ⚠️  ID bridge table is empty")
        
        logger.info(f"\n📁 Check logs/qa_check.log for detailed information")

def main():
    """Main execution function"""
    logger.info("🚀 Starting QA check for Step 1a data")
    
    qa = QAChecker()
    
    # Run all checks
    qa.check_data_availability()
    qa.check_row_counts()
    qa.check_non_null_keys()
    qa.check_score_ranges()
    qa.check_id_bridge()
    
    # Generate summary report
    qa.generate_summary_report()
    
    logger.info("\n✅ QA check completed")

if __name__ == "__main__":
    main()






