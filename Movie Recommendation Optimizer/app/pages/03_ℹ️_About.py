import streamlit as st

st.title("ℹ️ About")
st.markdown("Comprehensive overview of the Movie Recommendation Optimizer project and methodology.")

# Project Overview Section
st.header("🎯 Project Overview")

st.write("""
The **Movie Recommendation Optimizer** is a production-ready hybrid recommendation system designed to deliver personalized movie recommendations at scale. This application addresses the critical challenge of recommendation system optimization by combining multiple approaches to achieve superior performance across diverse user scenarios.
""")

st.subheader("Problem Statement")
st.write("""
Traditional recommendation systems often struggle with:
- **Cold Start Problem**: Difficulty recommending to new users or new items
- **Sparsity Issues**: Limited user-item interaction data
- **Coverage vs. Precision Trade-offs**: Balancing recommendation diversity with accuracy
- **Scalability Challenges**: Maintaining performance with growing user bases and catalogs

Our hybrid approach solves these challenges through intelligent combination of content-based and collaborative filtering techniques.
""")

st.divider()

# Methodology Section
st.header("🔬 Methodology")

st.subheader("Hybrid Recommendation Architecture")
st.write("""
Our system employs a sophisticated hybrid approach that combines:

• **Content-Based Filtering**: Uses TF-IDF vectors and BERT embeddings to analyze movie content, genres, crew, and textual features
• **Collaborative Filtering**: Leverages SVD matrix factorization to identify user preference patterns from rating data
• **Adaptive α Blending**: Dynamically adjusts the weight between content and collaborative signals based on user cohorts
• **Bucket-Gate Policy**: Implements intelligent routing to optimize recommendations for different user types
""")

st.subheader("Key Technical Innovations")
st.write("""
• **Multi-Modal Feature Engineering**: Combines text, categorical, and numerical features for comprehensive movie representation
• **Cohort-Based α Tuning**: Cold users (α=0.15), Light users (α=0.3), Medium users (α=0.7), Heavy users (α=0.9)
• **Long-Tail Quota System**: Ensures 30% of recommendations come from less popular items
• **Diversity Enforcement**: Uses Maximal Marginal Relevance (MMR) with λ=0.7 for recommendation diversity
• **Temporal Alignment**: Incorporates recency boost (0.1) to prioritize newer content
""")

st.divider()

# App Structure Section
st.header("🏗️ App Structure")

st.subheader("Navigation Overview")
st.write("""
The application is organized into four main sections, each serving a specific purpose in the recommendation workflow:

• **🏠 Home**: Interactive recommendation interface with search controls, genre filters, provider selection, and real-time recommendation display
• **📚 Case Studies**: Curated examples demonstrating how the system adapts to different user scenarios (Classic Movie Fan, Sci-Fi Enthusiast, Family Movie Night)
• **📊 Evaluation**: Performance metrics dashboard showing Recall@K, MAP@K, Coverage, and interactive visualizations of system performance
• **ℹ️ About**: This comprehensive overview of the project, methodology, and technical architecture
""")

st.subheader("User Journey Flow")
st.write("""
1. **Input**: Users specify movie preferences, genres, streaming providers, and year ranges
2. **Processing**: System applies hybrid recommendation logic with cohort-specific α tuning
3. **Output**: Personalized recommendations with match scores and explanatory insights
4. **Analysis**: Users can explore case studies and performance metrics to understand system behavior
""")

st.divider()

# Credits & Links Section
st.header("👥 Credits & Links")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔗 Project Links")
    st.write("""
    • **GitHub Repository**: [shaheenbehbehani/My_Projects](https://github.com/shaheenbehbehani/My_Projects)
    • **Live Application**: [Streamlit Cloud Deployment](https://myprojects-7bm7pnq5hyjqyln9gpax6t.streamlit.app)
    • **LinkedIn Profile**: [Shaheen Behbehani](https://linkedin.com/in/shaheenbehbehani)
    """)

with col2:
    st.subheader("📊 Data Sources")
    st.write("""
    • **MovieLens Dataset**: User ratings and movie metadata
    • **IMDB Dataset**: Movie titles, genres, crew information
    • **Rotten Tomatoes**: Movie reviews and ratings
    • **Streaming Provider APIs**: Availability data (placeholder)
    """)

st.subheader("🛠️ Technology Stack")
st.write("""
• **Backend**: Python, Pandas, NumPy, Scikit-learn
• **ML Libraries**: Surprise (collaborative filtering), PyTorch (BERT embeddings)
• **Visualization**: Streamlit, Plotly, Matplotlib, Seaborn
• **Deployment**: Streamlit Cloud, GitHub Actions
• **Data Processing**: PyArrow, FastParquet for efficient data handling
""")

st.subheader("🎓 Acknowledgments")
st.write("""
• **Loyola Marymount University**: Academic foundation and research support
• **Plug and Play Tech Center**: Industry mentorship and networking opportunities
• **Netflix Research**: Inspiration from state-of-the-art recommendation systems
• **Open Source Community**: Libraries and frameworks that made this project possible
""")

st.divider()

# Performance Summary
st.header("📈 Performance Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="MAP@10",
        value="0.0066",
        help="Mean Average Precision at rank 10"
    )

with col2:
    st.metric(
        label="Recall@10", 
        value="0.0111",
        help="Proportion of relevant items found"
    )

with col3:
    st.metric(
        label="Coverage",
        value="24.7%",
        help="Item catalog coverage"
    )

st.info("""
**Key Achievement**: Our hybrid bucket-gate approach achieves a **214.3% performance lift** over traditional content-based methods while maintaining robust coverage across diverse user cohorts.
""")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
<strong>Version 0.1 — For demonstration purposes only</strong><br>
Built with ❤️ by Shaheen Behbehani | Portfolio Project 2025
</div>
""", unsafe_allow_html=True)
