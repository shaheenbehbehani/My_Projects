import streamlit as st

st.title("📚 Case Studies")
st.markdown("Explore curated examples of how our recommendation system works across different user scenarios.")

# Case Study 1: Classic Movie Fan
with st.expander("🎭 Case Study 1: Classic Movie Fan", expanded=True):
    st.subheader("User Profile: Classic Movie Fan")
    st.write("**Input Parameters:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("🎬 **Movie:** The Godfather")
    with col2:
        st.write("🎭 **Genre:** Drama")
    with col3:
        st.write("📺 **Provider:** Netflix")
    
    st.write("**Year Range:** 1970-2000")
    
    st.subheader("📋 Recommendations:")
    recommendations_1 = [
        {"title": "The Shawshank Redemption", "year": 1994, "genre": "Drama", "rating": 9.3, "match": "95%"},
        {"title": "Goodfellas", "year": 1990, "genre": "Crime", "rating": 8.7, "match": "92%"},
        {"title": "Scarface", "year": 1983, "genre": "Crime", "rating": 8.3, "match": "89%"},
        {"title": "Casino", "year": 1995, "genre": "Crime", "rating": 8.2, "match": "87%"},
        {"title": "The Departed", "year": 2006, "genre": "Crime", "rating": 8.5, "match": "85%"}
    ]
    
    for i, movie in enumerate(recommendations_1, 1):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.write(f"**{i}. {movie['title']}** ({movie['year']})")
        with col2:
            st.write(f"🎭 {movie['genre']}")
        with col3:
            st.write(f"⭐ {movie['rating']}")
        with col4:
            st.write(f"🎯 {movie['match']}")
    
    st.info("💡 **Why these recommendations?** The system identified your preference for crime dramas with strong character development and complex narratives, similar to The Godfather's themes of family, power, and morality.")

# Case Study 2: Sci-Fi Enthusiast
with st.expander("🚀 Case Study 2: Sci-Fi Enthusiast", expanded=False):
    st.subheader("User Profile: Sci-Fi Enthusiast")
    st.write("**Input Parameters:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("🎬 **Movie:** Inception")
    with col2:
        st.write("🎭 **Genre:** Sci-Fi")
    with col3:
        st.write("📺 **Provider:** Prime Video")
    
    st.write("**Year Range:** 2000-2025")
    
    st.subheader("📋 Recommendations:")
    recommendations_2 = [
        {"title": "The Matrix", "year": 1999, "genre": "Sci-Fi", "rating": 8.7, "match": "96%"},
        {"title": "Interstellar", "year": 2014, "genre": "Sci-Fi", "rating": 8.6, "match": "94%"},
        {"title": "Blade Runner 2049", "year": 2017, "genre": "Sci-Fi", "rating": 8.0, "match": "91%"},
        {"title": "Tenet", "year": 2020, "genre": "Sci-Fi", "rating": 7.5, "match": "88%"},
        {"title": "Arrival", "year": 2016, "genre": "Sci-Fi", "rating": 7.9, "match": "86%"}
    ]
    
    for i, movie in enumerate(recommendations_2, 1):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.write(f"**{i}. {movie['title']}** ({movie['year']})")
        with col2:
            st.write(f"🎭 {movie['genre']}")
        with col3:
            st.write(f"⭐ {movie['rating']}")
        with col4:
            st.write(f"🎯 {movie['match']}")
    
    st.info("💡 **Why these recommendations?** The system recognized your interest in mind-bending sci-fi concepts and complex narratives, recommending films with similar themes of reality, time, and consciousness.")

# Case Study 3: Family Movie Night
with st.expander("👨‍👩‍👧‍👦 Case Study 3: Family Movie Night", expanded=False):
    st.subheader("User Profile: Family Movie Night")
    st.write("**Input Parameters:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("🎬 **Movie:** Toy Story")
    with col2:
        st.write("🎭 **Genre:** Animation")
    with col3:
        st.write("📺 **Provider:** Disney+")
    
    st.write("**Year Range:** 1990-2025")
    
    st.subheader("📋 Recommendations:")
    recommendations_3 = [
        {"title": "Finding Nemo", "year": 2003, "genre": "Animation", "rating": 8.1, "match": "94%"},
        {"title": "Frozen", "year": 2013, "genre": "Animation", "rating": 7.4, "match": "92%"},
        {"title": "The Lion King", "year": 1994, "genre": "Animation", "rating": 8.5, "match": "90%"},
        {"title": "Monsters, Inc.", "year": 2001, "genre": "Animation", "rating": 8.1, "match": "88%"},
        {"title": "Up", "year": 2009, "genre": "Animation", "rating": 8.2, "match": "86%"}
    ]
    
    for i, movie in enumerate(recommendations_3, 1):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.write(f"**{i}. {movie['title']}** ({movie['year']})")
        with col2:
            st.write(f"🎭 {movie['genre']}")
        with col3:
            st.write(f"⭐ {movie['rating']}")
        with col4:
            st.write(f"🎯 {movie['match']}")
    
    st.info("💡 **Why these recommendations?** The system identified your preference for family-friendly animated films with heartwarming stories and universal appeal, perfect for all ages.")

# Summary section
st.markdown("---")
st.subheader("📊 Case Studies Summary")
st.write("These examples demonstrate how our hybrid recommendation system adapts to different user preferences and contexts:")
st.write("• **Classic Movie Fan**: Focuses on thematic similarity and narrative complexity")
st.write("• **Sci-Fi Enthusiast**: Emphasizes genre-specific elements and conceptual depth")
st.write("• **Family Movie Night**: Prioritizes age-appropriate content and universal appeal")

st.success("🎯 Each case study shows how the system combines content-based filtering with collaborative filtering to deliver personalized recommendations that match user intent and context.")
