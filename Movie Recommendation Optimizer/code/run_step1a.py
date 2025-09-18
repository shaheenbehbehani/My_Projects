#!/usr/bin/env python3
"""
Main Execution Script for Step 1a: Data Collection
Runs all ingestion processes in sequence
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/step1a_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if environment is properly set up"""
    logger.info("Checking environment setup...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        logger.error("❌ .env file not found. Please create it with your TMDB_API_KEY")
        logger.info("Example .env content:")
        logger.info("TMDB_API_KEY=your_api_key_here")
        return False
    
    # Check if required directories exist
    required_dirs = [
        'data/raw/tmdb',
        'data/raw/imdb', 
        'data/raw/movielens',
        'data/raw/rottentomatoes',
        'data/normalized',
        'logs',
        'docs'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.warning(f"⚠️  Directory {dir_path} does not exist, creating...")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Environment setup complete")
    return True

def run_ingestion_step(step_name, script_path, description):
    """Run a single ingestion step"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting {step_name}")
    logger.info(f"Description: {description}")
    logger.info(f"{'='*60}")
    
    try:
        # Import and run the script
        if step_name == "TMDB Ingestion":
            from tmdb_ingestion import main as tmdb_main
            tmdb_main()
        elif step_name == "IMDb + MovieLens Ingestion":
            from imdb_movielens_ingestion import main as imdb_ml_main
            imdb_ml_main()
        elif step_name == "Rotten Tomatoes Ingestion":
            from rottentomatoes_ingestion import main as rt_main
            rt_main()
        elif step_name == "ID Bridge Creation":
            from create_id_bridge import main as bridge_main
            bridge_main()
        else:
            logger.error(f"Unknown step: {step_name}")
            return False
        
        logger.info(f"✅ {step_name} completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ {step_name} failed: {e}")
        return False

def main():
    """Main execution function"""
    start_time = datetime.now()
    logger.info("🚀 Starting Step 1a: Data Collection for Movie Recommendation Optimizer")
    logger.info(f"Start time: {start_time}")
    
    # Check environment
    if not check_environment():
        logger.error("Environment setup failed. Exiting.")
        sys.exit(1)
    
    # Define ingestion steps
    ingestion_steps = [
        {
            "name": "TMDB Ingestion",
            "script": "tmdb_ingestion.py",
            "description": "Fetch movie data from TMDB API including titles, genres, cast, crew, ratings, and streaming providers"
        },
        {
            "name": "IMDb + MovieLens Ingestion", 
            "script": "imdb_movielens_ingestion.py",
            "description": "Load and process IMDb TSV and MovieLens CSV datasets with cross-linking"
        },
        {
            "name": "Rotten Tomatoes Ingestion",
            "script": "rottentomatoes_ingestion.py", 
            "description": "Load and process Rotten Tomatoes CSV datasets with score normalization"
        },
        {
            "name": "ID Bridge Creation",
            "script": "create_id_bridge.py",
            "description": "Build comprehensive ID bridge table connecting all data sources"
        }
    ]
    
    # Track execution results
    results = {}
    failed_steps = []
    
    # Execute each step
    for step in ingestion_steps:
        step_name = step["name"]
        script_path = step["script"]
        description = step["description"]
        
        logger.info(f"\n📋 Executing: {step_name}")
        logger.info(f"📁 Script: {script_path}")
        
        success = run_ingestion_step(step_name, script_path, description)
        results[step_name] = success
        
        if not success:
            failed_steps.append(step_name)
        
        logger.info(f"📊 Step result: {'✅ PASS' if success else '❌ FAIL'}")
    
    # Generate execution summary
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info("📊 STEP 1A EXECUTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Start time: {start_time}")
    logger.info(f"End time: {end_time}")
    logger.info(f"Total execution time: {execution_time}")
    logger.info(f"Total steps: {len(ingestion_steps)}")
    logger.info(f"Successful steps: {sum(results.values())}")
    logger.info(f"Failed steps: {len(failed_steps)}")
    
    if failed_steps:
        logger.error(f"❌ Failed steps: {', '.join(failed_steps)}")
    else:
        logger.info("🎉 All steps completed successfully!")
    
    # Check deliverables
    logger.info(f"\n📦 DELIVERABLES CHECK")
    logger.info(f"{'='*60}")
    
    deliverables = [
        ("data/raw/tmdb/*.json", "TMDB raw data"),
        ("data/raw/imdb/*.tsv", "IMDb raw data"),
        ("data/raw/movielens/*.csv", "MovieLens raw data"),
        ("data/raw/rottentomatoes/*.csv", "Rotten Tomatoes raw data"),
        ("data/normalized/*.parquet", "Normalized data files"),
        ("data/normalized/id_bridge.parquet", "ID bridge table"),
        ("logs/*.log", "Log files"),
        ("docs/*.md", "Documentation files")
    ]
    
    for pattern, description in deliverables:
        files = list(Path('.').glob(pattern))
        if files:
            logger.info(f"✅ {description}: {len(files)} files found")
        else:
            logger.warning(f"⚠️  {description}: No files found")
    
    # Final status
    if not failed_steps:
        logger.info(f"\n🎯 STEP 1A COMPLETED SUCCESSFULLY!")
        logger.info("All data sources have been ingested and linked.")
        logger.info("The ID bridge table is ready for use in subsequent phases.")
        logger.info("\nNext steps:")
        logger.info("1. Review log files for any warnings or issues")
        logger.info("2. Validate data quality and completeness")
        logger.info("3. Proceed to Step 1b: Data Exploration and Profiling")
    else:
        logger.error(f"\n❌ STEP 1A COMPLETED WITH ERRORS!")
        logger.error("Some steps failed. Please review the logs and fix issues before proceeding.")
        sys.exit(1)
    
    logger.info(f"\n📁 Check the following directories for outputs:")
    logger.info(f"   - Raw data: data/raw/")
    logger.info(f"   - Normalized data: data/normalized/")
    logger.info(f"   - Logs: logs/")
    logger.info(f"   - Documentation: docs/")

if __name__ == "__main__":
    main()






