import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Configure minimalist dark UI
st.set_page_config(page_title="SECOM Yield Predictor", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #E0E0E0; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏭 Semiconductor Yield & Root Cause Analysis")
st.markdown("Select a processed chip from the test batch to predict yield and generate a LIME local explanation.")

@st.cache_resource
def load_assets():
    # Load native XGBoost model safely
    model = xgb.XGBClassifier()
    model.load_model('models/xgb_model.json')
    
    threshold = joblib.load('models/best_threshold.joblib')
    X_test = pd.read_csv('models/X_test_preprocessed.csv')
    y_test = pd.read_csv('models/y_test.csv')
    X_train = pd.read_csv('models/X_train_resampled.csv')
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Pass', 'Fail'],
        mode='classification',
        random_state=42
    )
    return model, threshold, X_test, y_test, explainer

model, threshold, X_test, y_test, explainer = load_assets()

# Sidebar Selection
st.sidebar.header("Batch Inspection")
chip_idx = st.sidebar.slider("Select Chip ID (Test Set)", 0, len(X_test) - 1, 0)

sample_data = X_test.iloc[chip_idx]
actual_label = "Fail" if y_test.iloc[chip_idx].values[0] == 1 else "Pass"

# Prediction
prob = model.predict_proba(sample_data.values.reshape(1, -1))[0, 1]
predicted_label = "Fail 🚨" if prob >= threshold else "Pass ✅"

col1, col2, col3 = st.columns(3)
col1.metric("Selected Chip ID", f"#{chip_idx}")
col2.metric("Model Prediction", predicted_label, f"{prob:.1%} Failure Risk", delta_color="inverse")
col3.metric("Actual Ground Truth", actual_label)

# LIME Explanation
st.subheader("Root Cause Analysis (LIME)")
with st.spinner("Generating local explanation..."):
    exp = explainer.explain_instance(
        data_row=sample_data.values,
        predict_fn=model.predict_proba,
        num_features=10
    )
    
    fig = exp.as_pyplot_figure()
    fig.patch.set_facecolor('#121212')
    for text in fig.texts:
        text.set_color('#E0E0E0')
    ax = fig.gca()
    ax.tick_params(colors='#E0E0E0')
    ax.xaxis.label.set_color('#E0E0E0')
    
    st.pyplot(fig)