# File: src/train_lgbm.py (Fixed Version)
import joblib
import os
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                           classification_report, balanced_accuracy_score)
from scipy.stats import randint, uniform
from collections import Counter

# =====================================================================
# ADD THE MISSING TRANSFORMER CLASS DEFINITION
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
# MAIN TRAINING CODE
# =====================================================================
# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODELS_DIR = "outputs/models"
METRICS_DIR = "outputs/metrics"
PREDICTIONS_DIR = "outputs/predictions"
TUNING_ITER = 30
EARLY_STOPPING_ROUNDS = 50

def log_message(message):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {message}")

def load_processed_data():
    """Load and adjust class labels for LightGBM compatibility"""
    try:
        X, y = joblib.load("outputs/processed_data.pkl")
        # Convert KTAS labels from [1-5] to [0-4] for LightGBM
        y_adjusted = y - 1
        log_message(f"📊 Loaded data shape: {X.shape}, Class distribution: {Counter(y)}")
        return X, y_adjusted
    except Exception as e:
        log_message(f"❌ Error loading processed data: {str(e)}")
        raise

def train_lightgbm(X, y):
    """LightGBM training with proper label handling and feature_pre_filter fix"""
    start_time = time.time()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Parameter space with FIXED feature_pre_filter
    param_space = {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': ['multi_logloss', 'multi_error'],
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': RANDOM_STATE,
        'learning_rate': uniform(0.01, 0.3),
        'num_leaves': randint(20, 150),
        'max_depth': randint(3, 12),
        'min_child_samples': randint(10, 200),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0, 2),
        'reg_lambda': uniform(0, 2),
        'class_weight': 'balanced',
        'feature_pre_filter': False
    }
    
    best_score = 0
    best_model = None
    best_params = None
    
    for i in range(TUNING_ITER):
        iter_start = time.time()
        
        # Sample parameters
        params = {
            k: (v.rvs() if hasattr(v, 'rvs') else v)
            for k, v in param_space.items()
        }
        
        try:
            # Train with early stopping
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[test_data],
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(0),
                ]
            )
            
            # Evaluate (convert predictions back to original labels)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            y_pred_class = np.argmax(y_pred, axis=1) + 1  # Convert back to [1-5]
            y_test_original = y_test + 1  # Convert back to original labels
            
            metrics = {
                'accuracy': accuracy_score(y_test_original, y_pred_class),
                'balanced_accuracy': balanced_accuracy_score(y_test_original, y_pred_class),
                'f1_weighted': f1_score(y_test_original, y_pred_class, average='weighted'),
                'f1_macro': f1_score(y_test_original, y_pred_class, average='macro')
            }
            
            if metrics['f1_macro'] > best_score:
                best_score = metrics['f1_macro']
                best_model = model
                best_params = params
                log_message(f"🔥 New best (F1 Macro: {best_score:.4f}, Acc: {metrics['accuracy']:.4f})")
            
            iter_time = time.time() - iter_start
            log_message(f"⏳ Trial {i+1}/{TUNING_ITER}: F1 Macro={metrics['f1_macro']:.4f}, Time={iter_time:.1f}s")
            
        except Exception as e:
            log_message(f"⚠️ Trial {i+1} failed (skipping): {str(e)}")
            continue
    
    if best_model is None:
        raise RuntimeError("All trials failed - check your parameters")
    
    # Final evaluation with original labels
    y_pred = best_model.predict(X_test, num_iteration=best_model.best_iteration)
    y_pred_class = np.argmax(y_pred, axis=1) + 1
    y_test_original = y_test + 1

    metrics = {
        'accuracy': accuracy_score(y_test_original, y_pred_class),
        'balanced_accuracy': balanced_accuracy_score(y_test_original, y_pred_class),
        'f1_weighted': f1_score(y_test_original, y_pred_class, average='weighted'),
        'f1_macro': f1_score(y_test_original, y_pred_class, average='macro'),
        'f1_per_class': {cls: f1_score(y_test_original, y_pred_class, labels=[cls], average=None)[0] 
                        for cls in range(1, 6)},
        'confusion_matrix': confusion_matrix(y_test_original, y_pred_class, labels=[1, 2, 3, 4, 5]).tolist(),
        'classification_report': classification_report(y_test_original, y_pred_class, output_dict=True),
        'best_params': {k: float(v) if isinstance(v, np.floating) else v 
                       for k, v in best_params.items()},
        'training_time': time.time() - start_time,
        'best_iteration': best_model.best_iteration
    }
    
    # Save outputs
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    timestamp = int(time.time())
    model_path = f"{MODELS_DIR}/lgbm_model_{timestamp}.pkl"
    
    # Save model with preprocessing pipeline
    joblib.dump({
        'model': best_model,
        'preprocessor': joblib.load("outputs/preprocessing_pipeline.pkl"),
        'metadata': {
            'class_labels': [1, 2, 3, 4, 5],
            'timestamp': timestamp,
            'metrics': metrics
        }
    }, model_path)
    
    # Save metrics
    with open(f"{METRICS_DIR}/lgbm_metrics_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    pd.DataFrame({
        'Actual': y_test_original,
        'Predicted': y_pred_class
    }).to_csv(f"{PREDICTIONS_DIR}/lgbm_predictions_{timestamp}.csv", index=False)
    
    log_message("\n🎯 Final Model Performance:")
    log_message(f"🏆 Accuracy: {metrics['accuracy']:.4f}")
    log_message(f"⚖️ Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    log_message(f"📊 F1 Macro: {metrics['f1_macro']:.4f}")
    log_message(f"📈 F1 Weighted: {metrics['f1_weighted']:.4f}")
    log_message("\n📝 Class-wise F1 Scores:")
    for cls, score in metrics['f1_per_class'].items():
        log_message(f"Class {cls}: {score:.4f}")
    
    log_message(f"\n💾 Model saved to: {model_path}")
    
    return metrics, model_path

if __name__ == "__main__":
    log_message("🚀 Starting LightGBM Training Pipeline")
    try:
        X, y = load_processed_data()
        results, model_path = train_lightgbm(X, y)
        log_message(f"✅ Training completed! Final F1 Macro: {results['f1_macro']:.4f}")
    except Exception as e:
        log_message(f"❌ Training failed: {str(e)}")
        raise