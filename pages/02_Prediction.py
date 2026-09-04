import streamlit as st
import numpy as np
from utils import apply_theme, FEATURE_NAMES, MODEL_FILES, load_model, load_scaler

st.set_page_config(page_title="Prediction", layout="wide")
apply_theme()

st.markdown("""
<div class="hero">
    <h1>Water Potability Predictor</h1>
    <p>Adjust the chemical parameters below and let the model estimate whether the water sample is safe to drink.</p>
    
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Model choice
# ------------------------------------------------------------------
model_choice = st.selectbox(
    "Choose a model",
    list(MODEL_FILES.keys()) + ["Compare All Models"],
    index=0
)

st.write("")

# ------------------------------------------------------------------
# Input layout
# ------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card"><h3>Core Chemistry</h3>', unsafe_allow_html=True)
    ph = st.slider("pH Level", 0.0, 14.0, 7.04, step=0.01)
    hardness = st.slider("Hardness (mg/L)", 47.43, 323.12, 196.97)
    solids = st.slider("Solids — Total Dissolved Solids (ppm)", 320.94, 61227.20, 20927.83, step=10.0)
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

predict_clicked = st.button("Analyze Water Quality", use_container_width=True)

if predict_clicked:
    user_data = np.array([[
        ph, hardness, solids, chloramines, sulfate,
        conductivity, organic_carbon, trihalomethanes, turbidity
    ]])

    try:
        scaler = load_scaler()
        user_data_scaled = scaler.transform(user_data)
    except FileNotFoundError:
        st.error("Could not find models/scaler.pkl. Make sure the models folder is next to app.py.")
        st.stop()

    def run_model(name):
        try:
            model = load_model(name)
        except FileNotFoundError:
            return None, None
        pred = model.predict(user_data_scaled)[0]
        conf = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_data_scaled)[0]
            conf = float(np.max(proba)) * 100
        return pred, conf

    with st.spinner("Running the model on your sample..."):

        if model_choice == "Compare All Models":
            results = {name: run_model(name) for name in MODEL_FILES}

            missing = [name for name, (pred, _) in results.items() if pred is None]
            if missing:
                st.warning(f"Could not load: {', '.join(missing)}. Check that these files exist in the models folder.")

            cols = st.columns(len(MODEL_FILES))
            for c, (name, (pred, conf)) in zip(cols, results.items()):
                with c:
                    if pred is None:
                        st.markdown(f"""
                        <div class="model-result-card">
                            <h4>{name}</h4>
                            <p style="color:#94a3b8;">Model file not found</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif pred == 1:
                        conf_text = f"<p style='color:#94a3b8;font-size:0.85rem;'>Confidence: {conf:.1f}%</p>" if conf is not None else ""
                        st.markdown(f"""
                        <div class="model-result-card">
                            <h4>{name}</h4>
                            <div class="verdict-safe">Safe</div>
                            {conf_text}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        conf_text = f"<p style='color:#94a3b8;font-size:0.85rem;'>Confidence: {conf:.1f}%</p>" if conf is not None else ""
                        st.markdown(f"""
                        <div class="model-result-card">
                            <h4>{name}</h4>
                            <div class="verdict-unsafe">Not Safe</div>
                            {conf_text}
                        </div>
                        """, unsafe_allow_html=True)

            valid_preds = [pred for pred, _ in results.values() if pred is not None]
            if valid_preds:
                safe_votes = sum(1 for p in valid_preds if p == 1)
                majority_safe = safe_votes > len(valid_preds) / 2
                st.markdown(f"""
                <div class="{'result-safe' if majority_safe else 'result-unsafe'}">
                    <h2>{'Majority verdict: Safe to Drink' if majority_safe else 'Majority verdict: Not Safe to Drink'}</h2>
                    <p>{safe_votes} of {len(valid_preds)} models classified this sample as potable.</p>
                </div>
                """, unsafe_allow_html=True)
                if majority_safe:
                    st.warning("⚠️ This prediction is an estimate and may not always be accurate. Please verify the water quality through laboratory testing before drinking.")
        else:
            pred, conf = run_model(model_choice)

            if pred is None:
                st.error(f"Could not find the model file for {model_choice}. Check that it exists in the models folder.")
            elif pred == 1:
                st.markdown("""
                <div class="result-safe">
                    <h2>Water is Safe to Drink</h2>
                    <p>The chemical properties fall within the range typically considered potable.</p>
                </div>
                """, unsafe_allow_html=True)
                if conf is not None:
                    st.progress(int(conf), text=f"Model confidence: {conf:.1f}%")
                st.balloons()
            else:
                st.markdown("""
                <div class="result-unsafe">
                    <h2>Water is Not Safe to Drink</h2>
                    <p>The sample's properties suggest it may be unsafe for consumption.</p>
                </div>
                """, unsafe_allow_html=True)
                if conf is not None:
                    st.progress(int(conf), text=f"Model confidence: {conf:.1f}%")

    with st.expander("View submitted values"):
        for name, val in zip(FEATURE_NAMES, user_data[0]):
            st.write(f"**{name}:** {val:.2f}")