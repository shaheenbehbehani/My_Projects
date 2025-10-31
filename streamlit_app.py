# streamlit_app.py — tolerant launcher for Streamlit Cloud
import os, sys, importlib, runpy

APP_DIR = os.path.join(os.path.dirname(__file__), "Movie Recommendation Optimizer", "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    mod = importlib.import_module("main")  # Movie Recommendation Optimizer/app/main.py
    # If the module exposes a main() function, call it; otherwise execute as a script
    if hasattr(mod, "main") and callable(mod.main):
        mod.main()
    else:
        runpy.run_module("main", run_name="__main__")
except ModuleNotFoundError as e:
    import streamlit as st
    st.error(f"Entrypoint not found: {e}. "
             "Expected file: 'Movie Recommendation Optimizer/app/main.py'.")
