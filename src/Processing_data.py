import os
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import random

# Download all required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Add this specific resource

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# =====================================================================
# CLINICAL FEATURE ENGINEERING (UPDATED WITH DEFAULT VALUES)
# =====================================================================
def add_clinical_features(df):
    """Add clinically important feature combinations with safe defaults"""
    # Set default values for missing vitals
    vital_cols = ['SBP', 'Saturation', 'HR', 'RR']
    for col in vital_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Critical vital signs flag
    if all(col in df.columns for col in ['SBP', 'Saturation', 'HR', 'RR']):
        df['critical_vitals'] = ((df['SBP'] < 90) | 
                                (df['Saturation'] < 90) | 
                                (df['HR'] > 130) | 
                                (df['RR'] > 30)).astype(int)
    else:
        df['critical_vitals'] = 0
    
    # Mental status severity
    mental_map = {
        'Alert': 0,
        'Agitated': 1,
        'Verbal Response': 1,
        'Pain Response': 2,
        'Unresponsive': 3
    }
    if 'Mental' in df.columns:
        df['mental_severity'] = df['Mental'].map(mental_map).fillna(0)
    else:
        df['mental_severity'] = 0
    
    # Pediatric respiratory distress
    if 'Age' in df.columns and 'RR' in df.columns and 'Saturation' in df.columns:
        df['peds_resp_distress'] = ((df['Age'] < 5) & 
                                   ((df['RR'] > 40) | 
                                    (df['Saturation'] < 92))).astype(int)
    else:
        df['peds_resp_distress'] = 0
    
    # Cardiac risk (case-insensitive check)
    if 'Age' in df.columns and 'Chief_complain_clean' in df.columns:
        df['cardiac_risk'] = ((df['Age'] > 50) & 
                             (df['Chief_complain_clean'].str.contains('chest pain', case=False))).astype(int)
    else:
        df['cardiac_risk'] = 0
    
    # Pain severity groups
    if 'NRS_pain' in df.columns:
        df['severe_pain'] = (df['NRS_pain'] >= 8).astype(int)
        df['moderate_pain'] = ((df['NRS_pain'] >= 4) & (df['NRS_pain'] < 8)).astype(int)
    else:
        df['severe_pain'] = 0
        df['moderate_pain'] = 0
    
    return df
# =====================================================================
# TEXT PROCESSING UTILITIES
# =====================================================================
def get_synonyms(word, pos=None):
    """Get synonyms for a word with optional POS filtering."""
    synonyms = set()
    for syn in wn.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def augment_text(text, p=0.3):
    """Custom text augmentation using WordNet synonyms with error handling."""
    if not text or pd.isna(text):
        return text
    
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        tokens = word_tokenize(text)
    
    augmented_tokens = []
    
    for token in tokens:
        if random.random() > p:
            augmented_tokens.append(token)
            continue
            
        try:
            pos_tagged = pos_tag([token])
            pos = pos_tagged[0][1][0].upper() if pos_tagged else None
            pos = pos if pos in ['N', 'V', 'R', 'J'] else None
            
            synonyms = get_synonyms(token, pos=pos)
            if synonyms:
                augmented_tokens.append(random.choice(synonyms))
            else:
                augmented_tokens.append(token)
        except:
            augmented_tokens.append(token)
    
    return ' '.join(augmented_tokens)

def clean_text_column(text_series):
    return text_series.fillna("").apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x.lower().strip()))

# =====================================================================
# EMBEDDING TRANSFORMER
# =====================================================================
class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", column=None):
        self.model_name = model_name
        self.column = column
        self.model = None  # Lazy loading

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self._load_model()
        texts = X[self.column].fillna("").astype(str).tolist()
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'model' in state:
            del state['model']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = None

# =====================================================================
# DATA LOADING AND CLEANING WITH TYPE CONVERSION
# =====================================================================
def load_and_clean_data(filepath):
    try:
        df = pd.read_csv(filepath, sep=";", encoding="utf-8", engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, sep=";", encoding="ISO-8859-1", engine="python", on_bad_lines="skip")

    # Ensure KTAS_expert exists and has valid values
    if 'KTAS_expert' not in df.columns:
        raise ValueError("KTAS_expert column missing in dataset")
    
    # Convert numeric columns - handle comma decimals and empty strings
    numeric_cols = ['Age', 'Patients number per hour', 'NRS_pain', 
                   'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation',
                   'Length of stay_min', 'KTAS duration_min']
    
    for col in numeric_cols:
        if col in df.columns:
            # Replace commas with dots and convert to float
            df[col] = df[col].astype(str).str.replace(',', '.')
            # Convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert coded categorical columns to proper categories
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({1: 'Female', 2: 'Male'}).fillna('Unknown')
    
    if 'Injury' in df.columns:
        df['Injury'] = df['Injury'].map({1: 'No', 2: 'Yes'}).fillna('Unknown')
    
    if 'Pain' in df.columns:
        df['Pain'] = df['Pain'].map({0: 'No', 1: 'Yes'}).fillna('Unknown')
    
    if 'Mental' in df.columns:
        df['Mental'] = df['Mental'].map({
            1: 'Alert',
            2: 'Verbal Response',
            3: 'Pain Response',
            4: 'Unresponsive'
        }).fillna('Alert')
    
    if 'Group' in df.columns:
        df['Group'] = df['Group'].map({
            1: 'Local ED 3rd Degree',
            2: 'Regional ED 4th Degree'
        }).fillna('Unknown')
    
    if 'Arrival mode' in df.columns:
        df['Arrival mode'] = df['Arrival mode'].map({
            1: 'Walking',
            2: 'Public Ambulance',
            3: 'Private Vehicle',
            4: 'Private Ambulance'
        }).fillna('Unknown')
    
    if 'Disposition' in df.columns:
        df['Disposition'] = df['Disposition'].map({
            1: 'Discharge',
            2: 'Admission to ward',
            3: 'Admission to ICU',
            4: 'Discharge',
            5: 'Transfer',
            6: 'Death',
            7: 'Surgery'
        }).fillna('Unknown')

    # Filter valid KTAS_expert values
    df = df.dropna(subset=["KTAS_expert"])
    df = df[df["KTAS_expert"].isin([1, 2, 3, 4, 5])]

    # Clean text columns
    df["Chief_complain_clean"] = clean_text_column(df["Chief_complain"])
    df["Diagnosis_clean"] = clean_text_column(df["Diagnosis in ED"])

    # Add clinical features before augmentation
    df = add_clinical_features(df)

    # Skip augmentation if it fails
    try:
        # Augment only KTAS 4-5 to balance classes
        for ktas in [4, 5]:
            mask = df["KTAS_expert"] == ktas
            df.loc[mask, "Chief_complain_clean"] = df.loc[mask, "Chief_complain_clean"].apply(
                lambda x: augment_text(str(x)) if pd.notna(x) else x
            )
    except Exception as e:
        print(f"⚠️ Text augmentation skipped due to error: {e}")

    return df

# =====================================================================
# PIPELINE CONSTRUCTION
# =====================================================================
def build_preprocessing_pipeline(df):
    # Get all clinical features we added
    clinical_features = ['critical_vitals', 'mental_severity', 
                         'peds_resp_distress', 'cardiac_risk',
                         'severe_pain', 'moderate_pain']
    
    # Columns to drop
    drop_cols = ["KTAS_expert", "KTAS_RN", "Error_group", "mistriage",
                 "Chief_complain", "Diagnosis in ED"]
    
    # Structured columns include original features + clinical features
    structured_cols = df.drop(columns=drop_cols, errors='ignore').columns.tolist()
    
    # Separate feature types
    numeric_cols = df[structured_cols].select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_cols = df[structured_cols].select_dtypes(include=["object"]).columns.tolist()
    
    # Ensure clinical features are in numeric cols
    for feat in clinical_features:
        if feat in structured_cols and feat not in numeric_cols:
            numeric_cols.append(feat)

    # Create embedding transformers
    chief_embed = SentenceEmbeddingTransformer(column="Chief_complain_clean")
    diag_embed = SentenceEmbeddingTransformer(column="Diagnosis_clean")

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # More robust for clinical data
            ('scaler', StandardScaler())
        ]), numeric_cols),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols),
        ('chief_embed', chief_embed, ["Chief_complain_clean"]),
        ('diag_embed', diag_embed, ["Diagnosis_clean"])
    ])

    y = df["KTAS_expert"]
    return preprocessor, df, y

# =====================================================================
# MAIN PROCESSING FUNCTION
# =====================================================================
def preprocess_and_save(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    # Ensure NLTK resources are available
    try:
        word_tokenize("test")  # This will trigger any missing resource errors
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger')

    df = load_and_clean_data(filepath)
    if len(df) == 0:
        raise ValueError("No valid data remaining after cleaning")

    os.makedirs("outputs", exist_ok=True)

    try:
        pipeline, df, y = build_preprocessing_pipeline(df)
        
        # Apply preprocessing
        X = pipeline.fit_transform(df)
        
        # Balance classes using SMOTE
        smote = SMOTE(sampling_strategy={
            1: min(300, sum(y == 1) * 3),  # Oversample severe cases
            2: min(600, sum(y == 2) * 2),
            3: sum(y == 3),
            4: sum(y == 4),
            5: max(sum(y == 5), 500)  # Cap non-urgent cases
        }, random_state=42, k_neighbors=2)
        
        X_res, y_res = smote.fit_resample(X, y)
        
        # Save pipeline and data
        joblib.dump(pipeline, "outputs/preprocessing_pipeline.pkl")
        joblib.dump((X_res, y_res), "outputs/processed_data.pkl")
        
        # Save class distribution info
        class_dist = y_res.value_counts().to_dict()
        joblib.dump(class_dist, "outputs/class_distribution.pkl")

        print("✅ Saved processed data and pipeline.")
        print(f"Final shape: {X_res.shape}")
        print("Class distribution after balancing:")
        print(class_dist)
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        raise

# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    # Download all required resources first
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    preprocess_and_save("data/data.csv")