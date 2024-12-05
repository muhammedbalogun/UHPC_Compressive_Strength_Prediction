# app.py
import streamlit as st
import joblib
import numpy as np
import os

# Load models
model_path_rf = os.path.join(os.path.dirname(__file__), 'models', 'rf_gcv_corr.pkl')
model_path_xgb = os.path.join(os.path.dirname(__file__), 'models', 'xgb_gcv_corr.pkl')
rf_model = joblib.load(model_path_rf)
xgb_model = joblib.load(model_path_xgb)

# Streamlit App
models = {
    "Xgboost Regression": xgb_model.best_estimator_,
    "Random Forest": rf_model.best_estimator_,
}

# Streamlit App
st.title("UHPC Compression Strength Predictions")
#st.write("Select a model and provide input features to get predictions.")

# Dropdown to select the model
model_name = st.selectbox("Select a Model", list(models.keys()))
selected_model = models[model_name]

# Define unique names for the features
feature_names = [
    "Cement (kg/m^3)", 
    "Silica Fume (kg/m^3)", 
    "Quartz Powder (kg/m^3)", 
    "Nano Silica (kg/m^3)", 
    "Fine Aggregate (kg/m^3)",
    "Steel Fiber (kg/m^3)",
    "Superplasticizer (kg/m^3)", 
    "Age (days)"
]

# Input fields for the features
features = []
for feature_name in feature_names:
    feature_value = st.number_input(f"Enter {feature_name}:", key=feature_name)
    features.append(feature_value)

# Predict button
if st.button("Predict"):
    input_data = np.array([features]).reshape(1, -1)  # Ensure proper shape
    prediction = selected_model.predict(input_data)
    st.write(f"Selected Model: {model_name}")
    st.write(f"Predicted Target: {prediction[0]:.2f}")
