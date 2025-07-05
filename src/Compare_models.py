# evaluate_all_models.py
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    f1_score,
    cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import sys
import subprocess
from sklearn.calibration import calibration_curve
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import traceback
import xgboost as xgb  # Import XGBoost explicitly

# ===================================================================
# ADD MISSING TRANSFORMER CLASS DEFINITION
# ===================================================================
class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", column=None):
        self.model_name = model_name
        self.column = column
        self.model = None

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

# ===================================================================
# DIRECTORY SETUP
# ===================================================================
os.makedirs("outputs/plots/confusion_matrix", exist_ok=True)
os.makedirs("outputs/plots/calibration", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)

# ===================================================================
# MODEL LOADING WITH ERROR HANDLING
# ===================================================================
def load_model_safely(file_path, model_name):
    """Load model with comprehensive error handling"""
    try:
        model = joblib.load(file_path)
        
        # Handle LightGBM dictionary format
        if model_name == "LightGBM" and isinstance(model, dict):
            print("⚠️ LightGBM model is a dictionary. Attempting to extract model...")
            if 'model' in model:
                return model['model']
            elif 'classifier' in model:
                return model['classifier']
            elif 'lgbm' in model:
                return model['lgbm']
            else:
                # Try to find the first model-like object
                for key, value in model.items():
                    if hasattr(value, 'predict'):
                        print(f"⚠️ Using {key} as LightGBM model")
                        return value
                print("⚠️ Could not find model in LightGBM dictionary")
                return None
                
        return model
    except Exception as e:
        print(f"❌ Error loading {model_name}: {str(e)}")
        traceback.print_exc()
        return None

# ===================================================================
# MAIN FUNCTIONS
# ===================================================================
def load_data_and_models():
    """Load preprocessed data and all trained models"""
    try:
        # Load processed data
        X_res, y_res = joblib.load("outputs/processed_data.pkl")
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_res, y_res, 
            test_size=0.2, 
            random_state=42,
            stratify=y_res
        )
        print("✅ Data loaded and split successfully")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        traceback.print_exc()
        return {}, None, None, None, None
    
    # Model paths relative to outputs directory
    model_files = {
        "XGBoost": "models/xgboost_enhanced_1751566425.pkl",
        "LightGBM": "models/lgbm_model_1751569133.pkl",
        "RandomForest": "models/RandomForest.pkl",
        "LogisticRegression": "models/LogisticRegression.pkl",
        "SupportVectorMachine": "models/SupportVectorMachine.pkl"
    }
    
    models = {}
    for name, file in model_files.items():
        file_path = os.path.join("outputs", file)
        if os.path.exists(file_path):
            model = load_model_safely(file_path, name)
            if model is not None:
                models[name] = model
                print(f"✅ Loaded {name} model from {file_path}")
            else:
                print(f"⚠️ Failed to load {name} model")
        else:
            print(f"⚠️ File not found: {file_path}")
    
    return models, X_val, y_val, X_train, y_train

def evaluate_model(model, model_name, X_val, y_val):
    """Evaluate a single model and return metrics"""
    if model is None:
        print(f"⚠️ Skipping evaluation for {model_name} - model not loaded")
        return None
        
    start_time = time.time()
    
    # Handle XGBoost DMatrix requirement
    if model_name == "XGBoost":
        try:
            dval = xgb.DMatrix(X_val)
            preds = model.predict(dval)
            y_pred = np.argmax(preds, axis=1) + 1  # Convert probabilities to class labels
            y_proba = preds
        except Exception as e:
            print(f"❌ XGBoost prediction error: {str(e)}")
            return None
    # Handle LightGBM specifically
    elif model_name == "LightGBM":
        try:
            # LightGBM may return probabilities instead of class labels
            preds = model.predict(X_val)
            
            # Check if we got probabilities (2D array) or class labels (1D array)
            if preds.ndim == 2 and preds.shape[1] == 5:
                y_pred = np.argmax(preds, axis=1) + 1  # Convert to class labels
                y_proba = preds
            else:
                # If it's already class labels, use directly
                y_pred = preds
                y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        except Exception as e:
            print(f"❌ LightGBM prediction error: {str(e)}")
            return None
    else:
        # Predictions for other models
        try:
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        except Exception as e:
            print(f"❌ Prediction error for {model_name}: {str(e)}")
            return None
    
    # Save predictions
    try:
        pred_df = pd.DataFrame({
            'True_KTAS': y_val,
            'Predicted_KTAS': y_pred
        })
        pred_df.to_csv(f"outputs/predictions/{model_name}_predictions.csv", index=False)
        
        # Save probabilities if available
        if y_proba is not None:
            proba_df = pd.DataFrame(y_proba, columns=[f"Prob_KTAS_{i}" for i in range(1, 6)])
            proba_df['True_KTAS'] = y_val
            proba_df.to_csv(f"outputs/predictions/{model_name}_probabilities.csv", index=False)
    except Exception as e:
        print(f"⚠️ Error saving predictions for {model_name}: {str(e)}")
    
    # Calculate metrics
    try:
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1_weighted": f1_score(y_val, y_pred, average='weighted'),
            "kappa": cohen_kappa_score(y_val, y_pred),
            "report": classification_report(y_val, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_val, y_pred),
            "prediction_time": time.time() - start_time
        }
    except Exception as e:
        print(f"❌ Metric calculation error for {model_name}: {str(e)}")
        return None
    
    # Save classification report
    try:
        report_df = pd.DataFrame(metrics['report']).transpose()
        report_df.to_csv(f"outputs/metrics/{model_name}_classification_report.csv")
    except Exception as e:
        print(f"⚠️ Error saving report for {model_name}: {str(e)}")
    
    # Save confusion matrix visualization
    try:
        plt.figure(figsize=(10, 8))
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=[1, 2, 3, 4, 5], 
                    yticklabels=[1, 2, 3, 4, 5])
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted KTAS")
        plt.ylabel("True KTAS")
        plt.tight_layout()
        plt.savefig(f"outputs/plots/confusion_matrix/{model_name}_confusion_matrix.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Error creating confusion matrix for {model_name}: {str(e)}")
    
    # Create calibration plot if probabilities available
    if y_proba is not None:
        try:
            plt.figure(figsize=(10, 8))
            for ktas in range(5):
                true_binary = (y_val == ktas+1).astype(int)
                prob_pos = y_proba[:, ktas]
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    true_binary, prob_pos, n_bins=10
                )
                plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                         label=f"KTAS {ktas+1}")
            
            plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Fraction of positives")
            plt.ylim([-0.05, 1.05])
            plt.title(f"Calibration Plot: {model_name}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f"outputs/plots/calibration/{model_name}_calibration.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Error creating calibration plot for {model_name}: {str(e)}")
    
    return metrics

def generate_comparison_report(metrics):
    """Generate comparison report across all models"""
    comparison = []
    for model_name, model_metrics in metrics.items():
        if model_metrics is not None:
            comparison.append({
                "Model": model_name,
                "Accuracy": model_metrics["accuracy"],
                "Weighted F1": model_metrics["f1_weighted"],
                "Cohen's Kappa": model_metrics["kappa"],
                "Prediction Time (s)": model_metrics["prediction_time"]
            })
    
    if not comparison:
        return None
    
    df = pd.DataFrame(comparison)
    df.sort_values(by="Weighted F1", ascending=False, inplace=True)
    
    # Save to CSV
    try:
        df.to_csv("outputs/metrics/model_comparison.csv", index=False)
    except Exception as e:
        print(f"⚠️ Error saving comparison report: {str(e)}")
    
    # Create visual comparison
    try:
        plt.figure(figsize=(12, 8))
        df.set_index("Model")[["Accuracy", "Weighted F1"]].plot(kind="bar")
        plt.title("Model Performance Comparison")
        plt.ylabel("Score")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig("outputs/plots/model_comparison.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Error creating comparison plot: {str(e)}")
    
    return df

def main():
    print("===== Evaluating All Models =====")
    
    # Load data and models
    models, X_val, y_val, X_train, y_train = load_data_and_models()
    
    if not models:
        print("❌ No models available for evaluation")
        return
    
    print(f"\nLoaded {len(models)} models for evaluation")
    
    # Evaluate each model
    all_metrics = {}
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_model(model, model_name, X_val, y_val)
        all_metrics[model_name] = metrics
        
        if metrics is not None:
            print(f"✅ Evaluation complete for {model_name}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Weighted F1: {metrics['f1_weighted']:.4f}")
            print(f"  Prediction time: {metrics['prediction_time']:.4f}s")
        else:
            print(f"⚠️ Evaluation failed for {model_name}")
    
    # Generate comparison report
    comparison_df = generate_comparison_report(all_metrics)
    
    if comparison_df is not None:
        print("\n" + "="*50)
        print("Model Comparison Summary:")
        print(comparison_df.to_string(index=False))
        print("\n✅ Full evaluation completed!")
    else:
        print("\n⚠️ No valid evaluation results to compare")
    
    print("\nOutputs saved in:")
    print("- outputs/metrics/")
    print("- outputs/plots/")
    print("- outputs/predictions/")

if __name__ == "__main__":
    # Ensure dependencies are installed
    try:
        import xgboost
    except ImportError:
        print("Installing XGBoost...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==1.7.6"])
        
    try:
        import lightgbm
    except ImportError:
        print("Installing LightGBM...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    
    main()