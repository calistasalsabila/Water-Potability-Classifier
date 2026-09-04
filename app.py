import streamlit as st
from utils import apply_theme

st.set_page_config(
    page_title="Water Potability Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

st.markdown("""
<div class="hero">
    <h1>Water Potability Predictor</h1>
    <p>An interactive tool that uses machine learning to estimate whether a water sample is safe to drink, based on nine chemical and physical properties.</p>
    <div class="badge-row">
        <span class="badge">Random Forest</span>
        <span class="badge">K-Nearest Neighbors</span>
        <span class="badge">Naive Bayes</span>
        <span class="badge">9 chemical features</span>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>What is this?</h3>
        <p>This app estimates water potability — whether a sample is safe for human consumption — using a model trained on a labeled dataset of water quality measurements.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>How it works</h3>
        <p>You enter nine chemical readings (pH, hardness, solids, and more), the values are scaled the same way as during training, and a trained classifier predicts potable or not potable.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3>Models available</h3>
        <p>Three classifiers were trained on the same dataset: Random Forest, K-Nearest Neighbors, and Naive Bayes. You can run one, or compare all three side by side on the Prediction page.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="card" style="margin-top: 0.5rem;">
    <h3>Getting started</h3>
    <p>Head to the <b>Feature Guide</b> page in the sidebar to learn what each measurement means and its typical safe range, or jump straight into the <b>Prediction</b> page to test a water sample.</p>
</div>
""", unsafe_allow_html=True)