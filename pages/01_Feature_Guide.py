import streamlit as st
from utils import apply_theme, FEATURE_INFO

st.set_page_config(page_title="Feature Guide", layout="wide")
apply_theme()

st.markdown("""
<div class="hero">
    <h1>Feature Guide</h1>
    <p>Each water sample is described by nine measurements. Here's what they mean and the ranges generally considered safe.</p>
</div>
""", unsafe_allow_html=True)

cols = st.columns(3)

for i, (name, info) in enumerate(FEATURE_INFO.items()):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="feature-card">
            <h4>{name}</h4>
            <span class="unit">{info['unit']}</span>
            <p>{info['description']}</p>
            <div class="range">Typical safe range: {info['safe_range']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="card" style="margin-top: 0.5rem;">
    <h3>Note</h3>
    <p>These ranges are general guidelines (WHO / EPA) meant to give context to each value. The model's prediction is based on learned patterns in the training data, not on these thresholds directly — a sample outside one "safe range" isn't automatically classified as unsafe.</p>
</div>
""", unsafe_allow_html=True)