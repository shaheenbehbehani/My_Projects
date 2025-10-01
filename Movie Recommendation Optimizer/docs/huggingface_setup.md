# Hugging Face Spaces Setup Guide

## Quick Setup for Contingency Hosting

### 1. Create New Space
1. Go to [Hugging Face Spaces](https://huggingface.co/new-space)
2. Choose **Streamlit** as the SDK
3. Set Space name: `movie-recommendation-optimizer`
4. Set visibility to **Public**

### 2. Connect Repository
1. In Space settings, connect to GitHub repository
2. Repository: `shaheenbehbehani/My_Projects`
3. Branch: `main`
4. App file path: `Movie Recommendation Optimizer/app/main.py`

### 3. Verify Configuration
- Requirements.txt is automatically detected from root
- Entrypoint is set to `app/main.py`
- All dependencies should install automatically

### 4. Deploy
- Space will automatically build and deploy
- URL will be: `https://huggingface.co/spaces/shaheenbehbehani/movie-recommendation-optimizer`

## Troubleshooting

If the app file path is not recognized:
- Try setting the app file to just `app/main.py`
- Ensure the repository structure is correct
- Check that requirements.txt is in the root directory

## Benefits of Hugging Face Spaces
- Free hosting with good performance
- Automatic deployments from GitHub
- Built-in support for Streamlit
- Reliable uptime and global CDN
