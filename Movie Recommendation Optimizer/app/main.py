import streamlit as st
from utils.data_loader import check_artifacts_available
from utils.recommender import get_recommendations

# App constants
APP_NAME = "Movie Recommendation Optimizer"

# Check if artifacts are available
ARTIFACTS_AVAILABLE = check_artifacts_available()

# Check for MODE/DEMO secrets (tolerant handling)
def get_mode():
    """Get mode from secrets, defaults to full if artifacts exist."""
    if not ARTIFACTS_AVAILABLE:
        return "demo"
    
    # Check MODE secret
    mode = st.secrets.get("MODE", "").lower()
    if mode in ["prod", "full", "production"]:
        return "full"
    
    # Check DEMO_MODE secret
    demo_mode = st.secrets.get("DEMO_MODE", "").lower()
    if demo_mode in ["false", "0", "no", "off"]:
        return "full"
    
    # Default: use artifacts if available
    return "full" if ARTIFACTS_AVAILABLE else "demo"

def render_header():
    """Render the app header with title and logo"""
    st.title(f"🎬 {APP_NAME}")
    st.markdown("---")

def render_sidebar():
    """Render the sidebar navigation"""
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    st.sidebar.button("🏠 Home")
    st.sidebar.button("📚 Case Studies")
    st.sidebar.button("📊 Evaluation")
    st.sidebar.button("ℹ️ About")

def render_footer():
    """Render the global footer"""
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666; font-size: 0.8em;'>v0.1 — scaffold | <a href='https://github.com/shaheenbehbehani/My_Projects' target='_blank'>GitHub</a></div>", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎬",
    layout="wide"
)

# Render layout components
render_header()
render_sidebar()

# Main content - Home Page Controls
st.header("🔍 Search Controls")

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # Movie title search input
    movie_title = st.text_input(
        "🎬 Movie Title Search",
        placeholder="Enter a movie title to find similar recommendations...",
        help="Search for movies by title to get personalized recommendations"
    )

with col2:
    # Year slider
    year_range = st.slider(
        "📅 Release Year",
        min_value=1950,
        max_value=2025,
        value=(1990, 2020),
        help="Select the range of release years for recommendations"
    )

# Genre multi-select
genres = st.multiselect(
    "🎭 Genres",
    options=["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Documentary", "Animation", "Crime"],
    default=["Drama", "Comedy"],
    help="Select one or more genres to filter recommendations"
)

# Provider dropdown
providers = st.selectbox(
    "📺 Streaming Provider",
    options=["All Providers", "Netflix", "Hulu", "Prime Video", "Disney+", "HBO Max", "Apple TV+"],
    index=0,
    help="Filter recommendations by streaming provider availability"
)

# Get Recommendations button
get_recommendations = st.button("🚀 Get Recommendations", type="primary", use_container_width=True)

# Results section
st.header("📋 Results")

# Mock variation engine for demo purposes
def get_mock_recommendations(title_input, selected_genres, selected_provider):
    """Return different movie sets based on input for demo purposes"""
    title_lower = title_input.lower() if title_input else ""
    genres_lower = [g.lower() for g in selected_genres] if selected_genres else []
    
    # Sci-Fi Set
    if (any(keyword in title_lower for keyword in ["inception", "matrix", "blade runner", "interstellar"]) or 
        any(genre in genres_lower for genre in ["sci-fi", "science fiction"])):
        return [
            {"title": "The Matrix", "year": 1999, "genre": "Sci-Fi", "rating": 8.7},
            {"title": "Interstellar", "year": 2014, "genre": "Sci-Fi", "rating": 8.6},
            {"title": "Blade Runner 2049", "year": 2017, "genre": "Sci-Fi", "rating": 8.0},
            {"title": "Tenet", "year": 2020, "genre": "Sci-Fi", "rating": 7.5},
            {"title": "Arrival", "year": 2016, "genre": "Sci-Fi", "rating": 7.9}
        ]
    
    # Crime Drama Set
    elif (any(keyword in title_lower for keyword in ["godfather", "goodfellas", "scarface", "casino"]) or 
          any(genre in genres_lower for genre in ["drama", "crime"])):
        return [
            {"title": "The Godfather", "year": 1972, "genre": "Crime", "rating": 9.2},
            {"title": "Goodfellas", "year": 1990, "genre": "Crime", "rating": 8.7},
            {"title": "Scarface", "year": 1983, "genre": "Crime", "rating": 8.3},
            {"title": "Casino", "year": 1995, "genre": "Crime", "rating": 8.2},
            {"title": "The Departed", "year": 2006, "genre": "Crime", "rating": 8.5}
        ]
    
    # Animation Set
    elif (any(keyword in title_lower for keyword in ["toy story", "frozen", "finding nemo", "monsters"]) or 
          any(genre in genres_lower for genre in ["animation", "animated"])):
        return [
            {"title": "Toy Story", "year": 1995, "genre": "Animation", "rating": 8.3},
            {"title": "Finding Nemo", "year": 2003, "genre": "Animation", "rating": 8.1},
            {"title": "Frozen", "year": 2013, "genre": "Animation", "rating": 7.4},
            {"title": "Monsters, Inc.", "year": 2001, "genre": "Animation", "rating": 8.1},
            {"title": "Up", "year": 2009, "genre": "Animation", "rating": 8.2}
        ]
    
    # Default Set (fallback)
    else:
        return [
            {"title": "The Shawshank Redemption", "year": 1994, "genre": "Drama", "rating": 9.3},
            {"title": "Pulp Fiction", "year": 1994, "genre": "Crime", "rating": 8.9},
            {"title": "Forrest Gump", "year": 1994, "genre": "Drama", "rating": 8.8},
            {"title": "Gladiator", "year": 2000, "genre": "Action", "rating": 8.5},
            {"title": "The Dark Knight", "year": 2008, "genre": "Action", "rating": 9.0}
        ]

# Display results based on button press
if get_recommendations:
    mode = get_mode()
    
    if mode == "full" and ARTIFACTS_AVAILABLE:
        # Use real recommendations
        st.success("🎬 Finding recommendations...")
        
        # Get real recommendations
        recommendations = get_recommendations(
            title_query=movie_title,
            genres=genres if genres else None,
            year_range=year_range,
            top_k=10
        )
        
        if recommendations:
            st.success(f"Found {len(recommendations)} recommendations")
            
            # Display movies
            for i, movie in enumerate(recommendations, 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        year_str = f" ({movie['year']})" if movie.get('year') else ""
                        st.write(f"**{i}. {movie['title']}{year_str}**")
                    with col2:
                        if movie.get('genres'):
                            genres_str = movie['genres'].split('|')[0] if '|' in movie['genres'] else movie['genres']
                            st.write(f"🎭 {genres_str}")
                    if movie.get('score'):
                        st.caption(f"Similarity: {movie['score']:.3f}")
                    st.divider()
        else:
            st.warning("No recommendations found. Try adjusting your search criteria.")
    else:
        # Fallback to demo mode
        st.info("⚠️ Demo mode: Using sample recommendations (artifacts not available)")
        
        # Get mock recommendations based on user input
        mock_movies = get_mock_recommendations(movie_title, genres, providers)
        
        # Display movies in a nice format
        for i, movie in enumerate(mock_movies, 1):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{i}. {movie['title']}** ({movie['year']})")
                with col2:
                    st.write(f"🎭 {movie['genre']}")
                with col3:
                    st.write(f"⭐ {movie['rating']}")
                st.divider()
        
        # Add demo mode notice
        st.caption("(Demo mode: results are sample variations)")
else:
    if ARTIFACTS_AVAILABLE:
        st.info("Recommended Movies will appear here")
        st.write("Use the controls above to search for movie recommendations from the full catalog.")
    else:
        st.info("Recommended Movies will appear here")
        st.write("Use the controls above to search for movie recommendations.")
        st.warning("⚠️ Artifacts not available - running in demo mode")

# Render footer
render_footer()


