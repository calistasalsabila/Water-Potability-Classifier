import streamlit as st
import numpy as np
import joblib

# ------------------------------------------------------------------
# 1. Basic Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Water Potability Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# 2. Custom CSS — modern "tech" look
# ------------------------------------------------------------------
st.markdown("""
<style>
    /* Import a clean, modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* App background */
    .stApp {
        background: radial-gradient(circle at 10% 0%, #0f172a 0%, #0b1220 45%, #060a14 100%);
        color: #e2e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }

    /* Top toolbar / header bar (was showing as a white strip) */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    div[data-testid="stToolbar"] {
        background: transparent;
    }
    div[data-testid="stDecoration"] {
        background: transparent;
    }
    div[data-testid="stAppViewContainer"] {
        background: transparent;
    }
    .stApp > header {
        background: transparent;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2.2rem 1rem 1.4rem 1rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #22d3ee, #0ea5e9, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 620px;
        margin: 0 auto;
    }

    /* Section card wrapper */
    .card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 18px;
        padding: 1.5rem 1.6rem;
        backdrop-filter: blur(6px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        margin-bottom: 1.2rem;
    }
    .card h3 {
        color: #38bdf8;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Metric badges */
    .badge-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 0.6rem;
    }
    .badge {
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #7dd3fc;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Sliders label color */
    .stSlider label, .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* Primary button */
    div.stButton > button {
        background: linear-gradient(90deg, #0ea5e9, #22d3ee);
        color: #04101d;
        font-weight: 800;
        font-size: 1.05rem;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 6px 22px rgba(14, 165, 233, 0.35);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(14, 165, 233, 0.5);
        color: #04101d;
    }

    /* Result panels */
    .result-safe {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.03));
        border: 1px solid rgba(16,185,129,0.4);
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-unsafe {
        background: linear-gradient(135deg, rgba(244,63,94,0.15), rgba(244,63,94,0.03));
        border: 1px solid rgba(244,63,94,0.4);
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-safe h2 { color: #34d399; margin-bottom: 0.3rem; }
    .result-unsafe h2 { color: #fb7185; margin-bottom: 0.3rem; }
    .result-safe p, .result-unsafe p { color: #cbd5e1; }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. Load Model & Scaler (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('models/rf_model.pkl'), joblib.load('models/scaler.pkl')

model, scaler = load_model()

FEATURE_NAMES = [
    "pH", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic Carbon", "Trihalomethanes", "Turbidity"
]

# ------------------------------------------------------------------
# 4. Sidebar — info / about panel
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About this tool")
    st.markdown(
        "This app uses a **Random Forest** classifier trained on water "
        "chemistry data to estimate whether a sample is potable."
    )
    st.markdown("---")
    st.markdown("### Features analyzed")
    for f in FEATURE_NAMES:
        st.markdown(f"- {f}")
    st.markdown("---")
    st.caption("This tool is for educational purposes and does not replace certified lab testing.")

# ------------------------------------------------------------------
# 5. Hero header
# ------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>Water Potability Predictor</h1>
    <p>Adjust the chemical parameters below and let the model estimate whether the water sample is safe to drink.</p>
    <div class="badge-row">
        <span class="badge">Random Forest</span>
        <span class="badge">Real-time inference</span>
        <span class="badge">9 chemical features</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 6. Input Layout — grouped into two "cards"
# ------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card"><h3>Core Chemistry</h3>', unsafe_allow_html=True)
    ph = st.slider("pH Level", 0.0, 14.0, 7.04, step=0.01)
    hardness = st.slider("Hardness (mg/L)", 47.43, 323.12, 196.97)
    solids = st.number_input("Solids — Total Dissolved Solids (ppm)", 320.94, 61227.20, 20927.83)
    chloramines = st.slider("Chloramines (ppm)", 0.35, 13.13, 7.13)
    sulfate = st.slider("Sulfate (mg/L)", 129.00, 481.03, 333.07)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><h3>Physical & Organic Indicators</h3>', unsafe_allow_html=True)
    conductivity = st.slider("Conductivity (μS/cm)", 181.48, 753.34, 421.88)
    organic_carbon = st.slider("Organic Carbon (ppm)", 2.20, 28.30, 14.22)
    trihalomethanes = st.slider("Trihalomethanes (μg/L)", 8.73, 124.00, 66.62)
    turbidity = st.slider("Turbidity (NTU)", 1.45, 6.74, 3.96)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ------------------------------------------------------------------
# 7. Prediction
# ------------------------------------------------------------------
predict_clicked = st.button("Analyze Water Quality", use_container_width=True)

if predict_clicked:
    with st.spinner("Running the model on your sample..."):

        user_data = np.array([[
            ph,
            hardness,
            solids,
            chloramines,
            sulfate,
            conductivity,
            organic_carbon,
            trihalomethanes,
            turbidity
        ]])

        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)

        # Try to get a confidence score if the model supports it
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_data_scaled)[0]
            confidence = float(np.max(proba)) * 100

    if prediction[0] == 1:
        st.markdown(f"""
        <div class="result-safe">
            <h2>Water is Safe to Drink</h2>
            <p>The chemical properties fall within the range typically considered potable.</p>
        </div>
        """, unsafe_allow_html=True)
        if confidence is not None:
            st.progress(int(confidence), text=f"Model confidence: {confidence:.1f}%")
        st.balloons()
    else:
        st.markdown(f"""
        <div class="result-unsafe">
            <h2>Water is Not Safe to Drink</h2>
            <p>The sample's properties suggest it may be unsafe for consumption.</p>
        </div>
        """, unsafe_allow_html=True)
        if confidence is not None:
            st.progress(int(confidence), text=f"Model confidence: {confidence:.1f}%")

    with st.expander("View submitted values"):
        for name, val in zip(FEATURE_NAMES, user_data[0]):
            st.write(f"**{name}:** {val:.2f}")