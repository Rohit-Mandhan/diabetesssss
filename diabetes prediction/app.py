import os
# os.system("pip install joblib scikit-learn")  # Force install joblib & scikit-learn

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler



ann_model = None  # Initialize as None
try:
    ann_model = tf.keras.models.load_model(r"C:\Users\HP\Desktop\diabetes prediction\models\ann_model.keras")
    print("✅ ANN Model Loaded Successfully!")
except Exception as e:
    print("❌ Error loading ANN model:", e)

# Load ONNX Models
try:
    rf_sess = rt.InferenceSession(r"C:\Users\HP\Desktop\diabetes prediction\models\random_forest_model.onnx")
    xgb_sess = rt.InferenceSession(r"C:\Users\HP\Desktop\diabetes prediction\models\xgb_meta_model.onnx")
    print("✅ ONNX Models Loaded Successfully!")
except Exception as e:
    print("❌ Error loading ONNX models:", e)


# Function to make predictions
import xgboost as xgb
import numpy as np

def predict_diabetes(features):
    """
    Predicts diabetes based on user input features.

    :param features: A list or numpy array of 8 numerical values.
    :return: (Prediction label, Probability score)
    """

    features = np.array(features).reshape(1, -1)

    # Step 1: Get probability outputs from Random Forest & ANN
    rf_prob = rf_model.predict_proba(features)[:, 1]  # Probability of class 1
    ann_prob = ann_model.predict(features, verbose=0).flatten()  # ANN output

    # Step 2: Create Hybrid Feature Vector
    hybrid_features = np.hstack((rf_prob, ann_prob)).reshape(1, -1)

    # ✅ Convert to DMatrix for XGBoost
    dmatrix = xgb.DMatrix(hybrid_features)

    # Step 3: Make Final Prediction using XGBoost
    final_pred = xgb_model.predict(dmatrix)[0]
    final_prob = final_pred  # Since XGBoost outputs probabilities directly

    return int(final_pred > 0.5), final_prob  # Convert probability to binary class

# Example usage in Streamlit
st.title("Diabetes Prediction Web App")

# User input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input("Age", min_value=1, max_value=120, step=1)

# Predict button
if st.button("Predict Diabetes"):
    input_values = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    prediction, probability = predict_diabetes(input_values)
    
    if prediction == 1:
        st.error(f"Prediction: Diabetic (Probability: {probability:.2f})")
    else:
        st.success(f"Prediction: Not Diabetic (Probability: {1 - probability:.2f})")
