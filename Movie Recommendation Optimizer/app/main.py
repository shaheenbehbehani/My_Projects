import streamlit as st

# App constants
APP_NAME = "Movie Recommendation Optimizer"

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

# Sample movie data
sample_movies = [
    {"title": "The Shawshank Redemption", "year": 1994, "genre": "Drama", "rating": 9.3},
    {"title": "The Godfather", "year": 1972, "genre": "Crime", "rating": 9.2},
    {"title": "Pulp Fiction", "year": 1994, "genre": "Crime", "rating": 8.9},
    {"title": "Forrest Gump", "year": 1994, "genre": "Drama", "rating": 8.8},
    {"title": "The Matrix", "year": 1999, "genre": "Sci-Fi", "rating": 8.7}
]

# Display results based on button press
if get_recommendations:
    st.success("Showing sample recommendations...")
    
    # Display sample movies in a nice format
    for i, movie in enumerate(sample_movies, 1):
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{i}. {movie['title']}** ({movie['year']})")
            with col2:
                st.write(f"🎭 {movie['genre']}")
            with col3:
                st.write(f"⭐ {movie['rating']}")
            st.divider()
else:
    st.info("Recommended Movies will appear here")
    st.write("Use the controls above to search for movie recommendations.")

# Render footer
render_footer()


