# streamlit_app.py
# Root-level launcher for Streamlit Cloud deployment
import os, sys

# Handle the space in the folder name safely
sys.path.insert(0, os.path.join(os.getcwd(), "Movie Recommendation Optimizer", "app"))

from main import main  # imports the actual app entrypoint

if __name__ == "__main__":
    main()
