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
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('punkt_tab')  # Add this specific resource

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sentence_transformers import SentenceTransformer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

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

class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", column=None):
        self.model_name = model_name
        self.column = column
        self.model = SentenceTransformer(self.model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = X[self.column].fillna("").astype(str).tolist()
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings

def load_and_clean_data(filepath):
    try:
        df = pd.read_csv(filepath, sep=";", encoding="utf-8", engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, sep=";", encoding="ISO-8859-1", engine="python", on_bad_lines="skip")

    df = df.dropna(subset=["KTAS_expert"])
    df = df[df["KTAS_expert"].isin([1, 2, 3, 4, 5])]

    # Skip augmentation if it fails
    try:
        df.loc[df["KTAS_expert"] == 5, "Chief_complain"] = df.loc[df["KTAS_expert"] == 5, "Chief_complain"].apply(
            lambda x: augment_text(str(x)) if pd.notna(x) else x
        )
    except Exception as e:
        print(f"⚠️ Text augmentation skipped due to error: {e}")

    return df

def build_preprocessing_pipeline(df):
    df["Chief_complain_clean"] = clean_text_column(df["Chief_complain"])
    df["Diagnosis_clean"] = clean_text_column(df["Diagnosis in ED"])

    drop_cols = ["KTAS_expert", "KTAS_RN", "Error_group", "mistriage",
                 "Chief_complain", "Diagnosis in ED", "Chief_complain_clean", "Diagnosis_clean"]
    structured_cols = df.drop(columns=drop_cols, errors='ignore').columns.tolist()

    numeric_cols = df[structured_cols].select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_cols = df[structured_cols].select_dtypes(include=["object"]).columns.tolist()

    chief_embed = SentenceEmbeddingTransformer(column="Chief_complain_clean")
    diag_embed = SentenceEmbeddingTransformer(column="Diagnosis_clean")

    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_cols),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols),
        ('chief_embed', chief_embed, ["Chief_complain_clean"]),
        ('diag_embed', diag_embed, ["Diagnosis_clean"])
    ])

    y = df["KTAS_expert"]
    return preprocessor, df, y

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
        X = pipeline.fit_transform(df)

        joblib.dump(pipeline, "outputs/preprocessing_pipeline.pkl")
        joblib.dump((X, y), "outputs/processed_data.pkl")

        print("✅ Saved processed data and pipeline.")
        print(f"Final shape: {X.shape}")
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Download all required resources first
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')
    
    preprocess_and_save("data/data.csv")