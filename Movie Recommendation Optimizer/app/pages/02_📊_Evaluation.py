import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("📊 Evaluation")
st.markdown("Performance metrics and visualizations for the Movie Recommendation Optimizer system.")

# Key Metrics Section
st.header("📈 Key Metrics")

# Create metric cards in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Recall@10",
        value="0.12",
        delta="+0.03",
        delta_color="normal",
        help="Proportion of relevant items found in top-10 recommendations"
    )

with col2:
    st.metric(
        label="MAP@10",
        value="0.06",
        delta="+0.01",
        delta_color="normal",
        help="Mean Average Precision at rank 10"
    )

with col3:
    st.metric(
        label="Item Coverage",
        value="72%",
        delta="+5%",
        delta_color="normal",
        help="Percentage of catalog items that can be recommended"
    )

st.divider()

# Charts Section
st.header("📊 Performance Charts")

# Chart 1: Bar chart comparing metrics
st.subheader("Metric Comparison")
st.write("Comparison of key performance metrics across different recommendation approaches.")

# Create sample data for bar chart
metrics_data = {
    'Approach': ['Content-Based', 'Collaborative', 'Hybrid (α=0.5)', 'Hybrid (α=0.7)', 'Hybrid (α=0.9)'],
    'Recall@10': [0.08, 0.05, 0.10, 0.12, 0.11],
    'MAP@10': [0.04, 0.02, 0.05, 0.06, 0.05],
    'Coverage': [0.85, 0.15, 0.45, 0.72, 0.68]
}

df_metrics = pd.DataFrame(metrics_data)

# Create bar chart
fig_bar = px.bar(
    df_metrics, 
    x='Approach', 
    y=['Recall@10', 'MAP@10', 'Coverage'],
    title="Performance Metrics by Approach",
    barmode='group',
    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
)

fig_bar.update_layout(
    xaxis_title="Recommendation Approach",
    yaxis_title="Metric Value",
    legend_title="Metrics",
    height=500
)

st.plotly_chart(fig_bar, use_container_width=True)

st.info("💡 **Analysis**: The hybrid approach with α=0.7 shows the best balance between precision (MAP@10) and coverage, achieving 12% recall while maintaining 72% item coverage.")

# Chart 2: Line chart showing Recall@K vs K
st.subheader("Recall@K Performance")
st.write("How recall performance varies with different recommendation list sizes (K).")

# Create sample data for line chart
k_values = [5, 10, 15, 20, 25, 30, 40, 50]
recall_values = [0.08, 0.12, 0.15, 0.17, 0.19, 0.20, 0.22, 0.23]

df_recall = pd.DataFrame({
    'K': k_values,
    'Recall@K': recall_values
})

# Create line chart
fig_line = px.line(
    df_recall, 
    x='K', 
    y='Recall@K',
    title="Recall@K vs Recommendation List Size",
    markers=True,
    color_discrete_sequence=['#FF6B6B']
)

fig_line.update_layout(
    xaxis_title="Recommendation List Size (K)",
    yaxis_title="Recall@K",
    height=400
)

st.plotly_chart(fig_line, use_container_width=True)

st.info("💡 **Analysis**: Recall increases with larger recommendation lists, but with diminishing returns. The optimal trade-off appears around K=20-30 for this dataset.")

# Additional Metrics Section
st.divider()
st.subheader("📋 Additional Performance Indicators")

col1, col2 = st.columns(2)

with col1:
    st.write("**Cold Start Performance**")
    st.write("• New User Coverage: 68%")
    st.write("• New Item Coverage: 45%")
    st.write("• Average Response Time: 120ms")

with col2:
    st.write("**System Robustness**")
    st.write("• Error Rate: 0.3%")
    st.write("• Uptime: 99.7%")
    st.write("• Cache Hit Rate: 85%")

# Footer note
st.divider()
st.warning("⚠️ **Note**: Metrics shown are placeholder values for demonstration purposes. Real evaluation data will be integrated in Step 7 with actual model performance measurements.")
