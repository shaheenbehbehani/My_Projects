#!/usr/bin/env python3
"""
Smoke test for runtime artifacts.
Verifies artifacts can be loaded and recommendations work.
"""

import sys
from pathlib import Path

# Add app directory to path
APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))

from utils.data_loader import check_artifacts_available, load_id_map, load_tfidf_vectorizer
from utils.recommender import get_recommendations

def main():
    print("=" * 60)
    print("Runtime Artifacts Smoke Test")
    print("=" * 60)
    
    # Check artifacts
    print("\n1. Checking artifacts availability...")
    available = check_artifacts_available()
    if not available:
        print("❌ Artifacts not available!")
        return 1
    print("✓ Artifacts available")
    
    # Load artifacts
    print("\n2. Loading artifacts...")
    try:
        id_map = load_id_map()
        vectorizer = load_tfidf_vectorizer()
        print(f"✓ Loaded {len(id_map)} movies")
        print(f"✓ Loaded TF-IDF vectorizer (vocab size: {len(vectorizer.vocabulary_)})")
    except Exception as e:
        print(f"❌ Failed to load artifacts: {e}")
        return 1
    
    # Test recommendation
    print("\n3. Testing recommendation for 'The Matrix'...")
    try:
        recommendations = get_recommendations(
            title_query="The Matrix",
            top_k=5
        )
        
        if recommendations:
            print(f"✓ Generated {len(recommendations)} recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                year_str = f" ({rec['year']})" if rec.get('year') else ""
                print(f"  {i}. {rec['title']}{year_str}")
                if rec.get('genres'):
                    print(f"     Genres: {rec['genres']}")
        else:
            print("⚠ No recommendations returned")
    except Exception as e:
        print(f"❌ Recommendation test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("✓ All smoke tests passed!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())

