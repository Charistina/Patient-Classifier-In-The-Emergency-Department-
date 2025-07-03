# File: src/train_lgbm.py (Streamlit-Compatible Version)
import joblib
import os
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, balanced_accuracy_score
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE
from collections import Counter

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODELS_DIR = "outputs/models"
METRICS_DIR = "outputs/metrics"
PREDICTIONS_DIR = "outputs/predictions"
TUNING_ITER = 30
EARLY_STOPPING_ROUNDS = 50
CLASS_LABEL_OFFSET = 1

def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_and_preprocess_data():
    """Load and balance data"""
    data_path = "outputs/processed_data.pkl"
    X, y = joblib.load(data_path)
    y = y - CLASS_LABEL_OFFSET
    
    # Balance classes using SMOTE
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X, y = smote.fit_resample(X, y)
    
    log_message(f"📊 Balanced class distribution: {Counter(y)}")
    return X, y

def train_lightgbm(X, y):
    """Streamlit-compatible version with .pkl saving"""
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    
    # Create dataset with fixed feature_pre_filter
    train_data = lgb.Dataset(X_train, label=y_train, params={'feature_pre_filter': False})
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, 
                          params={'feature_pre_filter': False})
    
    # Parameter space
    param_space = {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': ['multi_logloss', 'multi_error'],
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'verbosity': -1,
        'seed': RANDOM_STATE,
        'learning_rate': uniform(0.05, 0.2),
        'num_leaves': randint(31, 128),
        'max_depth': randint(-1, 12),
        'min_child_samples': randint(20, 100),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1),
        'class_weight': 'balanced',
        'feature_pre_filter': False
    }
    
    best_score = 0
    best_model = None
    best_params = None
    
    for i in range(TUNING_ITER):
        iter_start = time.time()
        
        # Sample parameters with constraints
        params = {
            k: (v.rvs() if hasattr(v, 'rvs') else v)
            for k, v in param_space.items()
        }
        
        # Ensure safe parameter combinations
        params['min_child_samples'] = max(20, params['min_child_samples'])
        if params['max_depth'] > 0:
            params['num_leaves'] = min(params['num_leaves'], 2**params['max_depth'])
        
        try:
            # Train with early stopping
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[test_data],
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(0),
                ]
            )
            
            # Evaluate
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            y_pred_class = np.argmax(y_pred, axis=1)
            
            metrics = {
                'accuracy': accuracy_score(y_test + CLASS_LABEL_OFFSET, y_pred_class + CLASS_LABEL_OFFSET),
                'balanced_accuracy': balanced_accuracy_score(y_test + CLASS_LABEL_OFFSET, y_pred_class + CLASS_LABEL_OFFSET),
                'f1_weighted': f1_score(y_test + CLASS_LABEL_OFFSET, y_pred_class + CLASS_LABEL_OFFSET, average='weighted'),
                'f1_macro': f1_score(y_test + CLASS_LABEL_OFFSET, y_pred_class + CLASS_LABEL_OFFSET, average='macro')
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
    
    # Final evaluation
    y_pred = best_model.predict(X_test, num_iteration=best_model.best_iteration)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true = y_test + CLASS_LABEL_OFFSET
    y_pred_class = y_pred_class + CLASS_LABEL_OFFSET
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_class),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred_class),
        'f1_weighted': f1_score(y_true, y_pred_class, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred_class, average='macro'),
        'f1_per_class': {cls: f1_score(y_true, y_pred_class, labels=[cls], average=None)[0] 
                        for cls in range(1, 6)},
        'confusion_matrix': confusion_matrix(y_true, y_pred_class, labels=[1, 2, 3, 4, 5]).tolist(),
        'classification_report': classification_report(y_true, y_pred_class, output_dict=True),
        'best_params': {k: float(v) if isinstance(v, np.floating) else v 
                       for k, v in best_params.items()},
        'training_time': time.time() - start_time,
        'best_iteration': best_model.best_iteration
    }
    
    # Save outputs - Modified for Streamlit compatibility
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    timestamp = int(time.time())
    
    # Save in both formats for flexibility
    model_path_txt = f"{MODELS_DIR}/lgbm_model_{timestamp}.txt"
    model_path_pkl = f"{MODELS_DIR}/lgbm_model_{timestamp}.pkl"
    
    # Save native LGBM format (for potential future training)
    best_model.save_model(model_path_txt)
    
    # Save as pickle for Streamlit
    joblib.dump({
        'model': best_model,
        'metadata': {
            'class_labels': [1, 2, 3, 4, 5],
            'feature_names': [f"feature_{i}" for i in range(X.shape[1])],  # Update with real feature names
            'timestamp': timestamp,
            'metrics': metrics
        }
    }, model_path_pkl)
    
    # Save metrics
    with open(f"{METRICS_DIR}/lgbm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    pd.DataFrame({"Actual": y_true, "Predicted": y_pred_class}).to_csv(
        f"{PREDICTIONS_DIR}/lgbm_predictions.csv", index=False)
    
    log_message("\n🎯 Final Model Performance:")
    log_message(f"🏆 Accuracy: {metrics['accuracy']:.4f}")
    log_message(f"⚖️ Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    log_message(f"📊 F1 Macro: {metrics['f1_macro']:.4f}")
    log_message(f"📈 F1 Weighted: {metrics['f1_weighted']:.4f}")
    log_message("\n📝 Class-wise F1 Scores:")
    for cls, score in metrics['f1_per_class'].items():
        log_message(f"Class {cls}: {score:.4f}")
    
    log_message(f"\n💾 Model saved in Streamlit-compatible format to: {model_path_pkl}")
    
    return metrics, model_path_pkl  # Return path to .pkl file

if __name__ == "__main__":
    log_message("🚀 Starting LightGBM Training Pipeline")
    try:
        X, y = load_and_preprocess_data()
        results, model_path = train_lightgbm(X, y)
        log_message(f"✅ Training completed! Final F1 Macro: {results['f1_macro']:.4f}")
        log_message(f"📦 Model ready for Streamlit at: {model_path}")
    except Exception as e:
        log_message(f"❌ Training failed: {str(e)}")
        raise