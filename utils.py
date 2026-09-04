import streamlit as st
import joblib

# ------------------------------------------------------------------
# Feature order — MUST match the order used during model training
# ------------------------------------------------------------------
FEATURE_NAMES = [
    "pH", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic Carbon", "Trihalomethanes", "Turbidity"
]

# ------------------------------------------------------------------
# Reference info shown on the Feature Guide page
# ------------------------------------------------------------------
FEATURE_INFO = {
    "pH": {
        "unit": "pH scale (0-14)",
        "description": "Measures how acidic or alkaline the water is. Values far from neutral can indicate contamination or corrosive water.",
        "safe_range": "6.5 - 8.5 (WHO guideline)",
    },
    "Hardness": {
        "unit": "mg/L",
        "description": "Reflects the calcium and magnesium content. Hard water isn't necessarily unsafe, but it affects taste and can cause scale buildup.",
        "safe_range": "Below ~300 mg/L is considered moderate",
    },
    "Solids": {
        "unit": "ppm (TDS)",
        "description": "Total Dissolved Solids — the amount of minerals, salts and metals dissolved in the water.",
        "safe_range": "Below 1000 ppm (WHO guideline)",
    },
    "Chloramines": {
        "unit": "ppm",
        "description": "A disinfectant used to treat drinking water. Safe in small amounts, but harmful at high concentrations.",
        "safe_range": "Up to 4 ppm (EPA limit)",
    },
    "Sulfate": {
        "unit": "mg/L",
        "description": "A naturally occurring substance found in minerals, soil and rock. High levels can affect taste and have a laxative effect.",
        "safe_range": "Below 250 mg/L (WHO guideline)",
    },
    "Conductivity": {
        "unit": "μS/cm",
        "description": "Indicates how well water conducts electricity, which correlates with the concentration of dissolved ions.",
        "safe_range": "Below 400 μS/cm (WHO guideline)",
    },
    "Organic Carbon": {
        "unit": "ppm",
        "description": "Total Organic Carbon — a measure of organic compounds in the water, often from decaying matter or pollutants.",
        "safe_range": "Below 4 ppm in source water (EPA)",
    },
    "Trihalomethanes": {
        "unit": "μg/L",
        "description": "Byproducts formed when chlorine used for disinfection reacts with organic matter. Linked to health risks at high levels.",
        "safe_range": "Below 80 μg/L (EPA limit)",
    },
    "Turbidity": {
        "unit": "NTU",
        "description": "Measures water clarity. Cloudy water can indicate the presence of suspended particles or microorganisms.",
        "safe_range": "Below 5 NTU (WHO guideline)",
    },
}

# ------------------------------------------------------------------
# Model registry — all three trained models
# ------------------------------------------------------------------
MODEL_FILES = {
    "Random Forest": "models/rf_model.pkl",
    "K-Nearest Neighbors": "models/knn_model.pkl",
    "Naive Bayes": "models/nb_model.pkl",
}


@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")


@st.cache_resource
def load_model(model_name: str):
    return joblib.load(MODEL_FILES[model_name])


# ------------------------------------------------------------------
# Shared dark "tech" theme — call once at the top of every page
# ------------------------------------------------------------------
def apply_theme():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 10% 0%, #0f172a 0%, #0b1220 45%, #060a14 100%);
            color: #e2e8f0;
        }

        section[data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid rgba(148, 163, 184, 0.15);
        }

        /* Native multipage navigation (Home / Feature Guide / Prediction links) */
        [data-testid="stSidebarNav"] {
            background: transparent;
            padding-top: 0.5rem;
            position: relative;
            z-index: 999;
            pointer-events: auto !important;
        }
        [data-testid="stSidebarNav"] * {
            pointer-events: auto !important;
        }
        [data-testid="stSidebarNav"] ul { padding-left: 0; }
        [data-testid="stSidebarNav"] a,
        [data-testid="stSidebarNavLink"],
        [data-testid="stSidebarNav"] span,
        [data-testid="stSidebarNav"] p {
            color: #ffffff !important;
            font-weight: 600 !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebarNav"] li {
            border-radius: 10px;
        }
        [data-testid="stSidebarNav"] a {
            border-radius: 10px;
            padding: 0.5rem 0.8rem !important;
            cursor: pointer !important;
        }
        [data-testid="stSidebarNav"] a:hover {
            background: rgba(56, 189, 248, 0.1) !important;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: rgba(56, 189, 248, 0.15) !important;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] span {
            color: #38bdf8 !important;
        }
        [data-testid="stSidebarNavSeparator"] {
            border-color: rgba(148, 163, 184, 0.15) !important;
        }

        /* Manual "Pages" links added with st.page_link() */
        div[data-testid="stPageLink"] {
            border-radius: 10px;
        }
        div[data-testid="stPageLink"] a,
        div[data-testid="stPageLink"] p,
        div[data-testid="stPageLink"] span {
            color: #ffffff !important;
            font-weight: 600 !important;
            opacity: 1 !important;
        }
        div[data-testid="stPageLink"] a {
            border-radius: 10px;
            padding: 0.4rem 0.6rem !important;
        }
        div[data-testid="stPageLink"] a:hover {
            background: rgba(56, 189, 248, 0.12) !important;
            color: #38bdf8 !important;
        }
        div[data-testid="stPageLink"] a:hover span {
            color: #38bdf8 !important;
        }


        header[data-testid="stHeader"] { background: transparent; }
        div[data-testid="stToolbar"] { background: transparent; }
        div[data-testid="stDecoration"] { background: transparent; }
        div[data-testid="stAppViewContainer"] { background: transparent; }
        .stApp > header { background: transparent; }

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
            max-width: 680px;
            margin: 0 auto;
        }

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
        }
        .card p, .card li {
            color: #cbd5e1;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .feature-card {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(56, 189, 248, 0.15);
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }
        .feature-card h4 {
            color: #38bdf8;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .feature-card .unit {
            display: inline-block;
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.25);
            color: #7dd3fc;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }
        .feature-card p { color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0.4rem; }
        .feature-card .range { color: #34d399; font-size: 0.85rem; font-weight: 600; }

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

        .stSlider label, .stNumberInput label, .stSelectbox label, .stRadio label {
            color: #cbd5e1 !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }

        /* ---- Slider: recolor red -> blue regardless of internal DOM/class names ---- */
        div[data-testid="stSlider"], .stSlider {
            filter: hue-rotate(198deg) saturate(1.3) brightness(1.05) !important;
        }

        /* ---- Number input: field + stepper buttons (class + testid, both as fallback) ---- */
        .stNumberInput input, div[data-testid="stNumberInput"] input {
            background-color: #0f172a !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(148, 163, 184, 0.25) !important;
        }
        .stNumberInput button, div[data-testid="stNumberInput"] button {
            background-color: #0f172a !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(148, 163, 184, 0.25) !important;
        }
        .stNumberInput button:hover, div[data-testid="stNumberInput"] button:hover {
            background-color: rgba(56, 189, 248, 0.18) !important;
            color: #38bdf8 !important;
            border-color: #38bdf8 !important;
        }
        .stNumberInput button:active, div[data-testid="stNumberInput"] button:active,
        .stNumberInput button:focus, div[data-testid="stNumberInput"] button:focus {
            background-color: rgba(56, 189, 248, 0.25) !important;
            color: #38bdf8 !important;
            box-shadow: none !important;
        }
        .stNumberInput svg, div[data-testid="stNumberInput"] svg { fill: currentColor !important; }

        /* ---- Selectbox: closed field + dropdown menu (class + testid, both as fallback) ---- */
        .stSelectbox div[data-baseweb="select"] > div,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background-color: #0f172a !important;
            border-color: rgba(148, 163, 184, 0.25) !important;
            color: #e2e8f0 !important;
        }
        .stSelectbox svg, div[data-testid="stSelectbox"] svg {
            fill: #94a3b8 !important;
        }
        ul[data-baseweb="menu"] {
            background-color: #0f172a !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
        }
        ul[data-baseweb="menu"] li {
            background-color: transparent !important;
            color: #e2e8f0 !important;
        }
        ul[data-baseweb="menu"] li:hover {
            background-color: rgba(56, 189, 248, 0.15) !important;
        }
       
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

        .model-result-card {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            text-align: center;
        }
        .model-result-card h4 { color: #e2e8f0; margin-bottom: 0.5rem; }
        .model-result-card .verdict-safe { color: #34d399; font-weight: 700; font-size: 1.1rem; }
        .model-result-card .verdict-unsafe { color: #fb7185; font-weight: 700; font-size: 1.1rem; }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)