import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Set Page layout configurations
st.set_page_config(page_title="Breast Cancer Diagnostic System", layout="wide")

st.title("🩺 Breast Cancer Diagnostic Support System")
st.markdown("---")

MODEL_PATH = "models/champion_breast_cancer_model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ Model artifact missing at `{MODEL_PATH}`. Run the training script (`src/train_model.py`) to initialize the model pipeline.")
else:
    pipeline = joblib.load(MODEL_PATH)
    features = pipeline.steps[0][1].feature_names_in_

    st.sidebar.header("🔬 Patient Fine-Needle Aspirate (FNA) Inputs")
    st.sidebar.markdown("Adjust parameters below to update predictions:")
    
    # Organize fields into main clusters
    input_data = {}
    
    # We will provide default realistic midpoint configurations for key parameters
    for f in features:
        if "mean" in f:
            input_data[f] = st.sidebar.slider(f"{f.replace('_', ' ').title()}", 0.0, 40.0, 15.0)
        elif "worst" in f:
            input_data[f] = st.sidebar.slider(f"{f.replace('_', ' ').title()}", 0.0, 50.0, 20.0)
        else:
            input_data[f] = st.sidebar.slider(f"{f.replace('_', ' ').title()}", 0.0, 5.0, 0.5)

    # Main dashboard analysis panel
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Patient Raw Metrics Profile")
        st.dataframe(pd.DataFrame([input_data]), use_container_width=True)
        
    with col2:
        st.subheader("🔮 Diagnostic Inference Output")
        
        input_df = pd.DataFrame([input_data])[features]
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]
        
        if prediction == 1:
            st.error(f"**Classification: Malignant (Positive)**")
            st.metric(label="Malignancy Confidence Score", value=f"{probability * 100:.2f}%")
            st.progress(float(probability))
        else:
            st.success(f"**Classification: Benign (Negative)**")
            st.metric(label="Malignancy Confidence Score", value=f"{probability * 100:.2f}%")
            st.progress(float(probability))
            
    st.markdown("---")
    st.caption("Disclaimer: This tool is designed for educational evaluation purposes and does not replace medical professional assessment.")