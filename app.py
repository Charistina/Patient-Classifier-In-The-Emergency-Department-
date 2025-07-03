import streamlit as st
import joblib
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
import os
from sklearn.base import BaseEstimator, TransformerMixin

# ===== CUSTOM TRANSFORMER =====
class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Convert text column to sentence embeddings using a pretrained model."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # Will be lazy-loaded

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def fit(self, X, y=None):
        # No fitting needed for embeddings
        return self
    
    def transform(self, X):
        """Return embeddings with shape (n_samples, embedding_dim)."""
        import numpy as np
        self._load_model()  # Ensure model is loaded
        # Ensure X is iterable of strings
        texts = ["" if x is None else str(x) for x in X]
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)
    
    # Add for proper unpickling
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the model
        if '_model' in state:
            del state['_model']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None  # Reset model to be lazy-loaded

# ===== APP CONFIGURATION =====
st.set_page_config(
    page_title="ED Patient Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

custom_css = """
<style>
    /* Main content styling */
    .main {
        background-color: #f9f9f9;
    }
    
    /* Section headers */
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Form elements */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, 
    .stSelectbox>div>div>select {
        border: 1px solid #dfe6e9;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Cards for results */
    .triage-card {
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Selected level highlight */
    .selected-level {
        background-color: #3498db;
        color: white !important;
        transform: scale(1.05);
        transition: all 0.3s ease;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ===== MODEL LOADING =====
@st.cache_resource
def load_models():
    try:
        pipeline_path = "outputs/preprocessing_pipeline.pkl"
        model_path = "outputs/models/xgboost_enhanced_1751447170.pkl"
        
        if os.path.exists(pipeline_path) and os.path.exists(model_path):
            # Mock the original module structure
            import sys, types
            sys.modules['src'] = types.ModuleType('src')
            sys.modules['src.Processing_data'] = types.ModuleType('src.Processing_data')
            
            # Use our actual class definition for unpickling
            sys.modules['src.Processing_data'].SentenceEmbeddingTransformer = SentenceEmbeddingTransformer
            
            pipeline = joblib.load(pipeline_path)
            model = joblib.load(model_path)
            
            # Initialize models in embedding transformers
            for name, transformer, cols in pipeline.transformers:
                if isinstance(transformer, SentenceEmbeddingTransformer):
                    # Trigger lazy loading
                    transformer._load_model()
            
            # Get expected feature count from model
            model_feature_count = None
            try:
                import xgboost as xgb
                if isinstance(model, xgb.Booster):
                    model_feature_count = model.num_features()
                elif hasattr(model, 'n_features_in_'):
                    model_feature_count = model.n_features_in_
            except:
                pass
            
            return pipeline, model, model_feature_count
        else:
            st.error("Model files not found. Please check the paths.")
            return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

pipeline, model, model_feature_count = load_models()

# ===== THEME TOGGLE =====
def theme_toggle():
    st.session_state.dark_mode = not st.session_state.get("dark_mode", False)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 2rem; background-color: {'#e3f2fd' if not st.session_state.get('dark_mode', False) else '#23272e'}; border-radius: 10px; padding: 1rem;'>
        <h1 style='color: #3498db;'>ED Patient Classifier</h1>
        <h3 style='color: {'#2c3e50' if not st.session_state.get('dark_mode', False) else '#ecf0f1'};'>Machine Learning Based</h3>
        <div style='font-size: 3rem;'>🏥</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme toggle
    col1, col2 = st.columns([1, 2])
    with col1:
        st.button("🌓", on_click=theme_toggle, key="theme_toggle")
    with col2:
        st.write("Toggle Theme")
    
    # Navigation
    selected = option_menu(
        menu_title=None,
        options=["Triage Predictor", "About"],
        icons=["clipboard2-pulse", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "#3498db", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": "#3498db"},
        }
    )

# Apply theme
if st.session_state.get("dark_mode", False):
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #2c3e50 !important;
            color: #ecf0f1 !important;
        }
        .stApp, .main, .block-container {
            background-color: #2c3e50 !important;
            color: #ecf0f1 !important;
        }
        .stButton>button, .stSelectbox>div>div>select, .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #34495e !important;
            color: #ecf0f1 !important;
            border: 1px solid #7f8c8d !important;
        }
        .section-header, h1, h2, h3, h4, h5, h6 {
            color: #ecf0f1 !important;
        }
        /* Fix field labels in dark mode */
        label, .stNumberInput label, .stTextInput label, .stSelectbox label {
            color: #ecf0f1 !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    st.session_state["theme"] = "dark"
else:
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #f9f9f9 !important;
            color: #2c3e50 !important;
        }
        .stApp, .main, .block-container {
            background-color: #f9f9f9 !important;
            color: #2c3e50 !important;
        }
        .stButton>button, .stSelectbox>div>div>select, .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #fff !important;
            color: #2c3e50 !important;
            border: 1px solid #dfe6e9 !important;
        }
        .section-header, h1, h2, h3, h4, h5, h6 {
            color: #2c3e50 !important;
        }
        /* Fix field labels in light mode */
        label, .stNumberInput label, .stTextInput label, .stSelectbox label {
            color: #2c3e50 !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    st.session_state["theme"] = "light"

# ===== MAIN CONTENT =====
if selected == "Triage Predictor":
    st.markdown("""
    <div class='section-header'>
        <h1 style='color: #2c3e50;'>Patient Classifier</h1>
        <p style='color: #7f8c8d;'>Predict ESI/KTAS levels for incoming patients</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ About this tool", expanded=False):
        st.markdown(
            f"""
            <div style='background-color: {'#23272e' if st.session_state.get('dark_mode', False) else '#f1f9fe'}; padding: 1.5rem; border-radius: 10px; color: {'#ecf0f1' if st.session_state.get('dark_mode', False) else '#2c3e50'};'>
                This tool uses machine learning to predict the appropriate triage level (ESI/KTAS) 
                for emergency department patients based on their clinical presentation and vital signs.<br><br>
                <b>Triage Levels:</b><br>
                <ul>
                    <li><b>Level 1</b>: Resuscitation - Immediate life-saving intervention required</li>
                    <li><b>Level 2</b>: Emergency - High risk of deterioration, severe pain, or distress</li>
                    <li><b>Level 3</b>: Urgent - Stable but may require urgent investigation/treatment</li>
                    <li><b>Level 4</b>: Semi-urgent - Could be delayed, non-urgent conditions</li>
                    <li><b>Level 5</b>: Non-urgent - Chronic or minor problems that could be handled in primary care</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # ===== PATIENT INPUT FORM =====
    with st.form("patient_form"):
        st.markdown("""
        <div class='section-header'>
            <h3>Patient Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Demographics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Basic Information**")
            group = st.selectbox("Type of ED (Group)", [
                "Local ED 3rd Degree",
                "Regional ED 4th Degree"
            ])
            group_code = 1 if group == "Local ED 3rd Degree" else 2
            
        with col2:
            st.markdown("**Demographics**")
            sex = st.selectbox("Gender (Sex)", ["Female", "Male"])
            sex_code = 1 if sex == "Female" else 2
            age = st.number_input("Age", 0, 120, 30, help="Patient age in years")
            
        with col3:
            st.markdown("**Arrival Details**")
            patients_per_hour = st.number_input("Patients per hour", 0, 50, 5)
            arrival_mode = st.selectbox("Mode of Arrival", [
                "Walking",
                "Public Ambulance",
                "Private Vehicle",
                "Private Ambulance",
                "Other"
            ])
            arrival_map = {
                "Walking": 1, "Public Ambulance":2, "Private Vehicle":3,
                "Private Ambulance":4, "Other":5
            }
            arrival_code = arrival_map[arrival_mode]
        
        # Clinical Presentation
        st.markdown("""
        <div class='section-header'>
            <h3>Clinical Presentation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            chief_complain = st.text_input("Chief Complaint", "chest pain", 
                                        help="Primary reason for ED visit")
            diagnosis = st.text_input("Diagnosis in ED", "pneumonia", 
                                    help="Initial diagnosis in emergency department")
            
        with col2:
            pain_present = st.selectbox("Pain Present (Pain)", ["No", "Yes"], key="pain_present")
            pain_code = 0 if pain_present == "No" else 1
            nrs_pain = st.slider(
                label="Pain Score (NRS)",
                min_value=0, max_value=10, value=st.session_state.get("nrs_pain_slider", 0),
                help="Numeric Rating Scale for pain (0-10)",
                key="nrs_pain_slider"
            )
            if pain_code == 0:
                st.markdown("""
                <style>
                div[data-testid='stSlider'] {opacity: 0.5; pointer-events: none;}
                </style>
                """, unsafe_allow_html=True)
            st.markdown(f"""
                <style>
                label[for^='nrs_pain_slider'] {{
                    color: {'#ecf0f1' if st.session_state.get('dark_mode', False) else '#2c3e50'} !important;
                    font-weight: bold;
                }}
                </style>
            """, unsafe_allow_html=True)
        
        # Status and Vital Signs
        st.markdown("""
        <div class='section-header'>
            <h3>Status & Vital Signs</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Mental Status**")
            mental = st.selectbox("Mental Status", [
                "Alert",
                "Verbal Response",
                "Pain Response",
                "Unresponsive"
            ], label_visibility="collapsed")
            mental_map = {"Alert":1, "Verbal Response":2, "Pain Response":3, "Unresponsive":4}
            mental_code = mental_map[mental]
            
            st.markdown("**Injury**")
            injury = st.selectbox("Injury", ["No", "Yes"], label_visibility="collapsed")
            injury_code = 0 if injury == "No" else 1
            
        with col2:
            st.markdown("**Cardiovascular**")
            sbp = st.number_input("Systolic BP (mmHg)", 0, 300, 120)
            dbp = st.number_input("Diastolic BP (mmHg)", 0, 200, 80)
            hr = st.number_input("Heart Rate (bpm)", 0, 300, 75)
            
        with col3:
            st.markdown("**Respiratory & Other**")
            rr = st.number_input("Respiratory Rate (/min)", 0, 60, 18)
            bt = st.number_input("Body Temperature (°C)", 25.0, 45.0, 37.0)
            saturation = st.number_input("Oxygen Saturation (%)", 0, 100, 98)
        
        submitted = st.form_submit_button("Predict Triage Level", use_container_width=True)

    if submitted:
        if pipeline is None or model is None:
            st.error("Model not loaded. Please check the model files.")
        else:
            try:
                # Prepare input data with all required columns
                new_data = pd.DataFrame({
                    "Group": [group_code],
                    "Sex": [sex_code],
                    "Age": [age],
                    "Patients number per hour": [patients_per_hour],
                    "Arrival mode": [arrival_code],
                    "Injury": [injury_code],
                    "Chief_complain": [chief_complain],
                    "Mental": [mental_code],
                    "Pain": [pain_code],
                    "NRS_pain": [nrs_pain if pain_code == 1 else 0],
                    "SBP": [sbp],
                    "DBP": [dbp],
                    "HR": [hr],
                    "RR": [rr],
                    "BT": [bt],
                    "Saturation": [saturation],
                    "KTAS_RN": [3],  # Default triage nurse assessment
                    "Diagnosis in ED": [diagnosis],
                    "Disposition": [1],  # 1=Discharged, 2=Admitted, etc.
                    "KTAS_expert": [3],  # Default expert assessment
                    "Error_group": [0],  # 0=No error
                    "Length of stay_min": [60],  # Default 60 minutes
                    "KTAS duration_min": [30],  # Default 30 minutes
                    "mistriage": [0],  # 0=No mistriage
                    "Chief_complain_clean": [chief_complain],
                    "Diagnosis_clean": [diagnosis]
                })

                # Debug: Show the prepared data
                st.write("Prepared input data:")
                # Convert to string to avoid Arrow serialization issues
                st.dataframe(new_data.astype(str))

                # Convert numeric columns
                numeric_cols = [
                    'Group', 'Sex', 'Age', 'Patients number per hour', 'Arrival mode',
                    'Injury', 'Mental', 'Pain', 'NRS_pain', 'SBP', 'DBP', 'HR', 'RR',
                    'BT', 'Saturation', 'KTAS_RN', 'Disposition', 'KTAS_expert',
                    'Error_group', 'Length of stay_min', 'KTAS duration_min', 'mistriage'
                ]
                for col in numeric_cols:
                    new_data[col] = pd.to_numeric(new_data[col], errors='coerce')

                # Convert text columns
                text_cols = ['Chief_complain', 'Chief_complain_clean', 'Diagnosis in ED', 'Diagnosis_clean']
                for col in text_cols:
                    new_data[col] = new_data[col].astype(str)

                # Verify all required columns
                required_columns = [
                    'Group', 'Sex', 'Age', 'Patients number per hour', 'Arrival mode',
                    'Injury', 'Chief_complain', 'Mental', 'Pain', 'NRS_pain', 'SBP', 'DBP',
                    'HR', 'RR', 'BT', 'Saturation', 'KTAS_RN', 'Diagnosis in ED', 'Disposition',
                    'KTAS_expert', 'Error_group', 'Length of stay_min', 'KTAS duration_min',
                    'mistriage', 'Chief_complain_clean', 'Diagnosis_clean'
                ]
                
                # Ensure correct column order
                new_data = new_data[required_columns]

                # Debug: Show processed data
                st.write("Processed data with correct types:")
                st.write(new_data.dtypes)

                # Transform the data using pipeline
                try:
                    X_new = pipeline.transform(new_data)
                    st.write("Data after pipeline transformation:")
                    st.write(f"Shape: {X_new.shape}")
                    
                    # Handle feature dimension mismatch
                    if model_feature_count and X_new.shape[1] != model_feature_count:
                        if X_new.shape[1] > model_feature_count:
                            X_new = X_new[:, :model_feature_count]
                            if st.session_state.get('dark_mode', False):
                                st.markdown(f"<div style='background-color:#e67e22;color:#ecf0f1;padding:0.5rem;border-radius:5px;'>⚠️ Reduced features from {X_new.shape[1]} to {model_feature_count}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div style='background-color:#fff4e6;color:#2c3e50;padding:0.5rem;border-radius:5px;'>⚠️ Reduced features from {X_new.shape[1]} to {model_feature_count}</div>", unsafe_allow_html=True)
                        else:
                            st.error(f"Feature mismatch! Model expects {model_feature_count} but got {X_new.shape[1]}")
                            st.stop()

                    # Make prediction
                    try:
                        import xgboost as xgb
                        if isinstance(model, xgb.Booster):
                            dmatrix = xgb.DMatrix(X_new)
                            prediction = model.predict(dmatrix)
                        else:
                            prediction = model.predict(X_new)
                        
                        # Process prediction
                        import numpy as _np
                        if isinstance(prediction, (list, _np.ndarray)):
                            prediction = _np.asarray(prediction)
                            # If prediction is probability matrix (n_samples, n_classes)
                            if prediction.ndim == 2:
                                prediction_value = int(_np.argmax(prediction, axis=1)[0]) + 1  # shift 0-based to 1-based
                            else:
                                prediction_value = int(prediction.flatten()[0])
                                # If model trained with 0-based labels, shift
                                if prediction_value == 0:
                                    prediction_value += 1
                        else:
                            prediction_value = int(prediction)
                            if prediction_value == 0:
                                prediction_value += 1
                        
                        # Clamp to 1-5 range
                        prediction_value = max(1, min(5, prediction_value))
                        
                        # Theme-aware success banner
                        if st.session_state.get('dark_mode', False):
                            st.markdown(f"<div style='background-color:#27ae60;color:#ecf0f1;padding:0.75rem;border-radius:6px;'>✅ Prediction successful: {prediction_value}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='background-color:#d4edda;color:#155724;padding:0.75rem;border-radius:6px;'>✅ Prediction successful: {prediction_value}</div>", unsafe_allow_html=True)
                        
                    except ImportError:
                        # Fallback without xgboost bindings
                        prediction = model.predict(X_new)
                        prediction_value = int(prediction[0]) if isinstance(prediction, (np.ndarray, list)) else int(prediction)
                        prediction_value = max(1, min(5, prediction_value))
                    
                except Exception as e:
                    st.error(f"Error during preprocessing/prediction: {e}")
                    st.stop()


                st.markdown("""
                <div class='section-header'>
                    <h2>Triage Classification Result</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Dynamic card background based on theme
                if st.session_state.get('dark_mode', False):
                    card_bg = '#1abc9c'  # brighter for dark mode
                    text_color = '#ecf0f1'
                else:
                    card_bg = '#f1f9fe'
                    text_color = '#2c3e50'
                st.markdown(f"""
                <div class='triage-card' style='background-color: {card_bg}; color: {text_color};'>
                    The patient has been classified with the following urgency level:
                </div>
                """, unsafe_allow_html=True)
                
                # Create metric cards for each level
                cols = st.columns(5)
                levels = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
                colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
                
                for i, col in enumerate(cols):
                    with col:
                        is_selected = prediction_value == i+1
                        card_style = f"""
                            background-color: {colors[i] if is_selected else '#ecf0f1'};
                            color: {'white' if is_selected else '#2c3e50'};
                            padding: 1rem;
                            border-radius: 10px;
                            text-align: center;
                            transition: all 0.3s ease;
                            {'box-shadow: 0 4px 8px rgba(0,0,0,0.2); transform: scale(1.05);' if is_selected else ''}
                        """
                        with stylable_container(
                            key=f"level_{i+1}_container",
                            css_styles=f"""
                                {{
                                    {card_style}
                                }}
                            """
                        ):
                            st.markdown(f"**{levels[i]}**")
                            st.markdown("✔️" if is_selected else "")
                
                # Explanation
                with st.expander("Clinical Interpretation", expanded=True):
                    explanations = {
                        1: "**Resuscitation** - Patient requires immediate life-saving intervention. Critical condition with compromised vital functions.",
                        2: "**Emergency** - High-risk situation where delay could result in serious complications. Severe pain or distress present.",
                        3: "**Urgent** - Stable condition but requires urgent investigation or treatment. May need ED resources.",
                        4: "**Semi-urgent** - Condition that could be delayed without significant risk. Non-urgent problems.",
                        5: "**Non-urgent** - Could be managed in primary care. Chronic or minor problems."
                    }
                    # Theme-aware explanation card
                    bg_exp = '#2c3e50' if st.session_state.get('dark_mode', False) else '#f8f9fa'
                    txt_exp = '#ecf0f1' if st.session_state.get('dark_mode', False) else '#2c3e50'
                    exp_title, exp_body = explanations[prediction_value].split(' - ', 1)
                    st.markdown(f"""
                    <div style='background-color:{bg_exp};padding:1.5rem;border-radius:10px;'>
                        <h4 style='color:{txt_exp};'>{exp_title}</h4>
                        <p style='color:{txt_exp};'>{exp_body}</p>
                    </div>""", unsafe_allow_html=True)
                    
                    # Additional clinical guidance
                    if prediction_value in [1, 2]:
                        st.warning("**❗ Immediate Action Required** - Physician assessment needed immediately")
                    elif prediction_value == 3:
                        st.info("**ℹ️ Priority Assessment** - Nurse evaluation within 20 minutes recommended")
                    else:
                        if st.session_state.get('dark_mode', False):
                            st.markdown("<div style='background-color:#27ae60;color:#ecf0f1;padding:0.75rem;border-radius:6px;'>✅ Routine Assessment - Standard evaluation process appropriate</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='background-color:#d4edda;color:#155724;padding:0.75rem;border-radius:6px;'>✅ Routine Assessment - Standard evaluation process appropriate</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.write("Debug info - Data being sent to pipeline:")
                # Convert to string to avoid Arrow serialization issues
                st.write(new_data.astype(str))
                st.write("Data types:")
                st.write(new_data.dtypes.astype(str))

elif selected == "About":
    bg_color = '#1a2026' if st.session_state.get('dark_mode', False) else '#e8f4fc'
    st.markdown("""
    <div class='section-header'>
        <h1>About This Project</h1>
        <p>Emergency Department Patient Classification System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background-color: {'#23272e' if st.session_state.get('dark_mode', False) else '#f8f9fa'}; padding: 2rem; border-radius: 10px; color: {'#ecf0f1' if st.session_state.get('dark_mode', False) else '#2c3e50'};'>
        <h2 style='color: #3498db;'>Machine Learning Based Patient Classification</h2>
        <h3 style='color: {'#ecf0f1' if st.session_state.get('dark_mode', False) else '#2c3e50'};'>in the Emergency Department</h3>
        <div style='margin-top: 2rem;'>
            <p>This application uses machine learning to assist emergency department staff in classifying 
            patients according to the Emergency Severity Index (ESI) or Korean Triage and Acuity 
            Scale (KTAS) standards.</p>
            <h4 style='color: #3498db; margin-top: 1.5rem;'>How It Works</h4>
            <ul>
                <li>The model analyzes patient demographics, vital signs, and clinical presentation</li>
                <li>Predicts the appropriate triage level (1-5) based on clinical urgency</li>
                <li>Helps streamline ED workflow and prioritize patient care</li>
            </ul>
            <h4 style='color: #3498db; margin-top: 1.5rem;'>Model Details</h4>
            <ul>
                <li><strong>Algorithm:</strong> XGBoost Classifier</li>
                <li><strong>Training Data:</strong> Kaggle</li>
            </ul>
            <div style='margin-top: 2rem; padding: 1.5rem; background-color: {bg_color}; border-radius: 8px;'>
                <h5 style='color: {'#ecf0f1' if st.session_state.get('dark_mode', False) else '#2c3e50'};'>Batch - 17</h5>
                <p>Project created by:</p>
                <ul>
                    <li>A. Sandhya [22RH1A6611]</li>
                    <li>A. Silvia Jasmine [22RH1A6612]</li>
                    <li>Christina Charis [22RH1A6637]</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)