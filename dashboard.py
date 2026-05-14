import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction Dashboard")
st.write("Enter the patient's health details below to predict the likelihood of heart disease.")

@st.cache_resource
def load_models():
    try:
        dt = joblib.load('dt_model.pkl')
        rf = joblib.load('rf_model.pkl')
        knn = joblib.load('knn_model.pkl')
        return {'Decision Tree': dt, 'Random Forest': rf, 'KNN': knn}
    except Exception as e:
        st.error(f"Could not load models. Please ensure you have run the notebook to train models. Error: {e}")
        return None

models = load_models()

if models:
    st.sidebar.header("Prediction Settings")
    model_choice = st.sidebar.selectbox("Select ML Model", list(models.keys()))
    
    st.header("Patient Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", options=["TA", "ATA", "NAP", "ASY"], 
                                  help="TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic")
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0, max_value=600, value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], help="1 = True, 0 = False")
        
    with col2:
        resting_ecg = st.selectbox("Resting ECG", options=["Normal", "ST", "LVH"])
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=220, value=150)
        exercise_angina = st.selectbox("Exercise Induced Angina", options=["Y", "N"])
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
        st_slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])
        
    if st.button("Predict", type="primary"):
        input_data = {
            'Age': age,
            'Sex': sex,
            'ChestPainType': chest_pain,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'RestingECG': resting_ecg,
            'MaxHR': max_hr,
            'ExerciseAngina': exercise_angina,
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope
        }
        
        input_df = pd.DataFrame([input_data])
        
        model = models[model_choice]
        
        with st.spinner("Analyzing..."):
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None
            
        st.subheader("Prediction Results")
        if prediction == 1:
            st.error("⚠️ High risk of Heart Disease detected.")
        else:
            st.success("✅ Low risk of Heart Disease.")
            
        if probability is not None:
            st.write(f"Confidence (Probability of Heart Disease): **{probability:.2%}**")
        st.write(f"Model Used: **{model_choice}**")
