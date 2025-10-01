import streamlit as st
from .constants import APP_NAME, APP_VERSION, NAV_ITEMS

def render_header():
    """Render the app header with title and logo"""
    st.title(f"🎬 {APP_NAME}")
    st.markdown("---")

def render_sidebar():
    """Render the sidebar navigation"""
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    for item in NAV_ITEMS:
        if st.sidebar.button(f"{item['icon']} {item['name']}", key=f"nav_{item['name']}"):
            if item['page'] == 'main':
                st.switch_page("main.py")
            else:
                st.switch_page(f"pages/{item['page']}.py")

def render_footer():
    """Render the global footer"""
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: #666; font-size: 0.8em;'>{APP_VERSION} — scaffold | <a href='https://github.com/shaheenbehbehani/My_Projects' target='_blank'>GitHub</a></div>", unsafe_allow_html=True)
