import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS (THEMING) ---
def load_css():
    theme = st.session_state.get('theme', 'Light')
    
    sidebar_css = """<style>
[data-testid="stSidebar"] * { color: #FAFAFA !important; }
</style>"""
    
    if theme == "Dark":
        css = """<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background-color: #0E1117; color: #FAFAFA; }
.card { background-color: #1E1E1E; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(255,255,255,0.05); margin-bottom: 20px; border: 1px solid #333;}
.card h4 { color: #64B5F6; margin-top: 0; font-weight: 600;}
.stat-card { text-align: center; background-color: #262730; padding: 20px; border-radius: 8px; border-left: 6px solid #64B5F6; box-shadow: 0 2px 4px rgba(0,0,0,0.2);}
</style>"""
    else:
        css = """<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background-color: #FFFFFF; color: #212529; }
.stMarkdown p, .stText p, label, h1, h2, h3, h4, h5, h6 { color: #212529 !important; }
.card { background-color: #FFFFFF; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #E0E0E0;}
.card h4 { color: #1565C0 !important; margin-top: 0; font-weight: 600;}
.stat-card { text-align: center; background-color: #F8F9FA; padding: 20px; border-radius: 8px; border-left: 6px solid #1565C0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
</style>"""
        
    st.markdown(sidebar_css + css, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'
if 'page' not in st.session_state:
    st.session_state.page = 'Prediction'

# --- LOAD MODELS ---
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

# --- HELPER: GAUGE CHART ---
def create_gauge(value, title, max_val, color):
    theme = st.session_state.get('theme', 'Light')
    font_color = "#FAFAFA" if theme == "Dark" else "#212529"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 16, 'color': font_color}},
        number = {'font': {'color': font_color}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "gray", 'tickfont': {'color': font_color}},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0.05)",
            'borderwidth': 0,
        }
    ))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# --- SIDEBAR NAVIGATION ---
def render_sidebar():
    st.sidebar.title("❤️ Menu Navigation")
    
    # Navigation options
    st.session_state.page = st.sidebar.radio("Go to", ["Home", "Prediction", "Model Information", "Dataset Insights", "About Project"])
    
    st.sidebar.markdown("---")
    
    # Theme toggle
    st.sidebar.subheader("🎨 Settings")
    theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1, horizontal=True)
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()
        
    st.sidebar.markdown("---")
    st.sidebar.info("**Heart Disease Prediction**\n\nProfessional healthcare analytics and prediction dashboard.")

# --- PAGES ---
def render_home():
    st.title("🩺 Heart Disease Prediction Platform")
    st.markdown('<div class="card"><h3>Welcome to the AI-Powered Cardiology Dashboard</h3><p>Leverage the power of Machine Learning to predict the likelihood of heart disease instantly. The platform integrates seamlessly with your inputs to provide real-time, actionable insights.</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h4>🌟 Key Features</h4><ul><li>Modern UI/UX with responsive design.</li><li>Real-time dynamic health metrics visualization.</li><li>Utilizes robust ML Models (Random Forest, Decision Tree, KNN).</li></ul></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h4>🚀 Getting Started</h4><p>Navigate to the <b>Prediction</b> tab via the menu sidebar. Enter the required medical attributes, and the intelligent models will assess and display the potential cardiovascular risk.</p></div>', unsafe_allow_html=True)

def render_model_info():
    st.title("🧠 Model Information")
    st.write("Understand the algorithms powering your predictions.")
    
    st.markdown('<div class="card"><h4>🌲 Random Forest</h4><p>An ensemble learning method that builds a multitude of decision trees and merges them together to get a more accurate and stable prediction. It helps to prevent overfitting and is highly robust.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><h4>🌿 Decision Tree</h4><p>A supervised learning algorithm used for classification. It splits the data into subsets based on feature values, forming a tree-like model of decisions. It is highly interpretable.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><h4>📐 K-Nearest Neighbors (KNN)</h4><p>A simple, instance-based learning algorithm that classifies new cases based on a similarity measure (e.g., distance functions) to the instances in the training data.</p></div>', unsafe_allow_html=True)

def render_dataset_insights():
    st.title("📊 Dataset Insights")
    st.markdown('<div class="card"><p>The application is built on a comprehensive heart disease dataset containing 11 key features.</p><ul><li><b>Demographics:</b> Age, Sex</li><li><b>Symptoms:</b> Chest Pain Type, Exercise Induced Angina</li><li><b>Vitals & Tests:</b> Resting BP, Serum Cholesterol, Fasting Blood Sugar, Resting ECG, Max HR, Oldpeak, ST Slope</li></ul><p>These features were carefully selected to optimize the models for accurate classification of heart disease status.</p></div>', unsafe_allow_html=True)

def render_about():
    st.title("ℹ️ About Project")
    st.markdown('<div class="card"><p>This professional healthcare analytics dashboard was redesigned to feel like a modern AI-powered medical prediction platform, prioritizing usability, dynamic insights, and aesthetic appeal.</p></div>', unsafe_allow_html=True)

def render_prediction(models):
    st.title("🔬 Patient Risk Assessment")
    st.write("Enter patient medical information below to generate a real-time prediction and statistical analysis.")
    
    st.markdown('<div class="card"><h4>⚙️ Prediction Model Selection</h4>', unsafe_allow_html=True)
    model_choice = st.selectbox("Select ML Model", list(models.keys()), help="Choose the machine learning algorithm to perform the prediction.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col_input, col_stats = st.columns([2, 1])
    
    with col_input:
        st.markdown('<div class="card"><h4>Demographics</h4>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", min_value=1, max_value=120, value=35, help="Age of the patient in years")
        sex = c2.selectbox("Sex", options=["M", "F"], index=1, help="Sex of the patient (M: Male, F: Female)")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card"><h4>Vitals & Symptoms</h4>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        resting_bp = c1.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=110, help="Resting blood pressure on admission to the hospital")
        cholesterol = c2.number_input("Serum Cholesterol (mm/dl)", min_value=0, max_value=600, value=150, help="Serum cholesterol level")
        max_hr = c1.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=220, value=170, help="Maximum heart rate achieved during exercise (beats per minute)")
        fasting_bs = c2.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], index=0, help="Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)")
        chest_pain = c1.selectbox("Chest Pain Type", options=["TA", "ATA", "NAP", "ASY"], index=1, help="TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic")
        exercise_angina = c2.selectbox("Exercise Induced Angina", options=["Y", "N"], index=1, help="Exercise-induced angina (Y: Yes, N: No)")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card"><h4>ECG & Tests</h4>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        resting_ecg = c1.selectbox("Resting ECG", options=["Normal", "ST", "LVH"], index=0, help="Resting electrocardiogram results (Normal: Normal, ST: ST-T wave abnormality, LVH: left ventricular hypertrophy)")
        oldpeak = c2.number_input("Oldpeak (ST depression)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1, help="ST depression induced by exercise relative to rest")
        st_slope = c1.selectbox("ST Slope", options=["Up", "Flat", "Down"], index=0, help="The slope of the peak exercise ST segment (Up: upsloping, Flat: flat, Down: downsloping)")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_stats:
        st.markdown('<div class="card"><h4>Live Patient Vitals</h4>', unsafe_allow_html=True)
        
        # Age Category
        age_cat = "Young" if age < 40 else ("Middle-aged" if age < 60 else "Senior")
        st.markdown(f"**Age Category:** {age_cat}")
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        
        # Dynamic Gauges based on live inputs
        bp_color = "#4CAF50" if resting_bp < 130 else ("#FF9800" if resting_bp < 140 else "#F44336")
        st.plotly_chart(create_gauge(resting_bp, "Blood Pressure", 250, bp_color), use_container_width=True)
        
        chol_color = "#4CAF50" if cholesterol < 200 else ("#FF9800" if cholesterol < 240 else "#F44336")
        st.plotly_chart(create_gauge(cholesterol, "Cholesterol", 600, chol_color), use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_btn = st.button("🔮 Predict Risk", type="primary", use_container_width=True)
        
    if predict_btn:
        # Input Validation
        valid = True
        if age < 1 or age > 120:
            st.error("⚠️ Invalid input: Age must be between 1 and 120.")
            valid = False
        if resting_bp < 50 or resting_bp > 250:
            st.error("⚠️ Invalid input: Resting Blood Pressure must be between 50 and 250 mm Hg.")
            valid = False
        if max_hr < 50 or max_hr > 220:
            st.error("⚠️ Invalid input: Maximum Heart Rate must be between 50 and 220 bpm.")
            valid = False
        if cholesterol < 0 or cholesterol > 600:
            st.error("⚠️ Invalid input: Serum Cholesterol must be between 0 and 600 mm/dl.")
            valid = False
            
        if valid:
            input_data = {
                'Age': age, 'Sex': sex, 'ChestPainType': chest_pain,
                'RestingBP': resting_bp, 'Cholesterol': cholesterol,
                'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
                'MaxHR': max_hr, 'ExerciseAngina': exercise_angina,
                'Oldpeak': oldpeak, 'ST_Slope': st_slope
            }
            input_df = pd.DataFrame([input_data])
            model = models[model_choice]
            
            with st.spinner("Analyzing patient data..."):
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None
                
            render_results(prediction, probability, model_choice)

def render_results(prediction, probability, model_choice):
    st.markdown("---")
    st.subheader("📊 Prediction Results Dashboard")
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if prediction == 1:
            st.markdown('<div class="stat-card" style="border-left-color: #F44336;"><h2>⚠️ High Risk Detected</h2><p>The model indicates a high likelihood of heart disease. Immediate consultation with a healthcare professional is recommended.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="stat-card" style="border-left-color: #4CAF50;"><h2>✅ Low Risk</h2><p>The model indicates a low likelihood of heart disease. Maintain a healthy lifestyle!</p></div>', unsafe_allow_html=True)
            
        st.info(f"**Model Evaluated:** {model_choice}")
        
    with res_col2:
        if probability is not None:
            theme = st.session_state.get('theme', 'Light')
            font_color = "#FAFAFA" if theme == "Dark" else "#212529"
            risk_color = "#4CAF50" if probability < 0.3 else ("#FF9800" if probability < 0.7 else "#F44336")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                number = {'suffix': "%", 'font': {'color': font_color}},
                title = {'text': "Prediction Confidence Score", 'font': {'size': 18, 'color': font_color}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray", 'tickfont': {'color': font_color}},
                    'bar': {'color': risk_color},
                    'bgcolor': "rgba(0,0,0,0.05)",
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(76, 175, 80, 0.2)"},
                        {'range': [30, 70], 'color': "rgba(255, 152, 0, 0.2)"},
                        {'range': [70, 100], 'color': "rgba(244, 67, 54, 0.2)"}
                    ]
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Confidence score (probability) is not available for this model.")

# --- MAIN EXECUTION ---
def main():
    load_css()
    render_sidebar()
    
    models = load_models()
    if not models:
        st.stop()
        
    if st.session_state.page == "Home":
        render_home()
    elif st.session_state.page == "Prediction":
        render_prediction(models)
    elif st.session_state.page == "Model Information":
        render_model_info()
    elif st.session_state.page == "Dataset Insights":
        render_dataset_insights()
    elif st.session_state.page == "About Project":
        render_about()

if __name__ == "__main__":
    main()
