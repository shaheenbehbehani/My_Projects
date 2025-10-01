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

# Main content
st.write("Page scaffold OK")

st.markdown("### Next: Search Box UI")
st.write("Step 6.4 will bring interactive search functionality and data placeholders.")

# Render footer
render_footer()


