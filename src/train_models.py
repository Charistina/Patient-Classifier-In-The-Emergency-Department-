# train_additional_models.py
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for saving models and results
os.makedirs("trained_models", exist_ok=True)
os.makedirs("model_results", exist_ok=True)

def load_data():
    """Load preprocessed data and class distribution"""
    X_res, y_res = joblib.load("outputs/processed_data.pkl")
    class_dist = joblib.load("outputs/class_distribution.pkl")
    print("\nLoaded preprocessed data:")
    print(f"- Features shape: {X_res.shape}")
    print(f"- Target shape: {y_res.shape}")
    print(f"- Class distribution: {class_dist}")
    
    return X_res, y_res

def split_data(X, y):
    """Split data into train and validation sets with stratification"""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    print("\nData split results:")
    print(f"- Training set: {X_train.shape[0]} samples")
    print(f"- Validation set: {X_val.shape[0]} samples")
    return X_train, X_val, y_train, y_val

def train_model(model, model_name, X_train, y_train, X_val, y_val):
    """Train and evaluate a single model"""
    print(f"\n=== Training {model_name} ===")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"\n{model_name} results:")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Weighted F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))
    
    # Save confusion matrix visualization
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_val, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=[1, 2, 3, 4, 5], 
                yticklabels=[1, 2, 3, 4, 5])
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted KTAS")
    plt.ylabel("True KTAS")
    plt.tight_layout()
    plt.savefig(f"model_results/{model_name}_confusion_matrix.png")
    plt.close()
    
    # Save model
    joblib.dump(model, f"trained_models/{model_name}.pkl")
    print(f"\nSaved {model_name} model to trained_models/{model_name}.pkl")
    
    return model

def main():
    print("Starting training of additional models...")
    
    # Load preprocessed data
    X, y = load_data()
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(X, y)
    
    # Initialize models with appropriate parameters for clinical data
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver='saga',
            penalty='elasticnet',
            l1_ratio=0.5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "SupportVectorMachine": SVC(
            C=1.0,
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }
    
    # Train and evaluate each model
    trained_models = {}
    for model_name, model in models.items():
        trained_model = train_model(
            model, model_name,
            X_train, y_train,
            X_val, y_val
        )
        trained_models[model_name] = trained_model
    
    print("\nTraining completed for all models!")
    print("Models saved in 'trained_models' directory")
    print("Evaluation metrics and visualizations saved in 'model_results' directory")

if __name__ == "__main__":
    main()