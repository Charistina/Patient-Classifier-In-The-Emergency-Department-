# File: src/train_xgboost_enhanced.py
import joblib
import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from collections import Counter

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODELS_DIR = "outputs/models"
METRICS_DIR = "outputs/metrics"
PREDICTIONS_DIR = "outputs/predictions"
TUNING_ITER = 30  # Balanced tuning
EARLY_STOPPING_ROUNDS = 50
CLASS_LABEL_OFFSET = 1
CV_FOLDS = 5

def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_and_preprocess_data():
    """Enhanced data loading with feature selection"""
    data_path = "outputs/processed_data.pkl"
    X, y = joblib.load(data_path)
    y = y - CLASS_LABEL_OFFSET
    
    # Feature selection - two stage process
    log_message("🔍 Performing feature selection...")
    initial_model = xgb.XGBClassifier(
        objective='multi:softprob',
        tree_method='hist',
        device='cuda:0',
        n_estimators=100,
        random_state=RANDOM_STATE
    )
    
    # Stage 1: Coarse selection
    selector = SelectFromModel(initial_model, threshold='median', max_features=800)
    X_reduced = selector.fit_transform(X, y)
    kept_indices = selector.get_support(indices=True)
    log_message(f"📉 Reduced features from {X.shape[1]} to {X_reduced.shape[1]}")
    
    # Stage 2: Handle imbalance
    log_message("🔄 Applying SMOTENC for categorical-aware balancing")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_balanced, y_balanced = smote.fit_resample(X_reduced, y)
    
    log_message(f"📊 New class distribution: {Counter(y_balanced)}")
    return X_balanced, y_balanced, kept_indices

def train_xgboost(X, y):
    """Advanced training with all enhancements"""
    start_time = time.time()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    
    # Dynamic sample weights
    sample_weights = compute_sample_weight('balanced', y_train)
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Enhanced parameter space
    param_space = {
        'objective': 'multi:softprob',  # Better for imbalanced classes
        'num_class': 5,
        'eval_metric': ['mlogloss', 'merror'],
        'seed': RANDOM_STATE,
        'verbosity': 0,
        'tree_method': 'hist',
        'device': 'cuda:0',
        'learning_rate': uniform(0.05, 0.1),  # Higher learning rate
        'max_depth': randint(6, 12),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'gamma': uniform(0.1, 0.4),
        'reg_alpha': uniform(0.5, 1.5),
        'reg_lambda': uniform(0.5, 1.5),
        'min_child_weight': randint(3, 8),
        'grow_policy': 'depthwise'
    }
    
    # Bayesian optimization setup
    best_score = 0
    best_model = None
    best_params = None
    
    for i in range(TUNING_ITER):
        iter_start = time.time()
        
        # Sample parameters with adaptive strategy
        params = {k: v.rvs() if hasattr(v, 'rvs') else v 
                 for k, v in param_space.items()}
        
        # Dynamic early stopping
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=500,
            nfold=CV_FOLDS,
            stratified=True,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            metrics=['mlogloss', 'merror'],
            seed=RANDOM_STATE,
            verbose_eval=False
        )
        
        best_round = cv_results.shape[0]
        mean_test_score = 1 - cv_results['test-merror-mean'].iloc[-1]
        
        # Train final model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=best_round,
            evals=[(dtest, 'test')],
            verbose_eval=False
        )
        
        # Comprehensive evaluation
        y_prob = model.predict(dtest)
        y_pred = np.argmax(y_prob, axis=1)
        
        metrics = {
            'accuracy': accuracy_score(y_test + CLASS_LABEL_OFFSET, y_pred + CLASS_LABEL_OFFSET),
            'balanced_accuracy': balanced_accuracy_score(y_test + CLASS_LABEL_OFFSET, y_pred + CLASS_LABEL_OFFSET),
            'f1_weighted': f1_score(y_test + CLASS_LABEL_OFFSET, y_pred + CLASS_LABEL_OFFSET, average='weighted'),
            'f1_macro': f1_score(y_test + CLASS_LABEL_OFFSET, y_pred + CLASS_LABEL_OFFSET, average='macro')
        }
        
        if metrics['f1_macro'] > best_score:  # Focus on macro F1 for imbalance
            best_score = metrics['f1_macro']
            best_model = model
            best_params = params
            best_round = best_round
            log_message(f"🔥 New best (F1 Macro: {best_score:.4f}, Acc: {metrics['accuracy']:.4f})")
        
        iter_time = time.time() - iter_start
        log_message(f"⏳ Trial {i+1}/{TUNING_ITER}: F1 Macro={metrics['f1_macro']:.4f}, Time={iter_time:.1f}s")
    
    # Final evaluation
    y_prob = best_model.predict(dtest)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = y_test + CLASS_LABEL_OFFSET
    y_pred = y_pred + CLASS_LABEL_OFFSET
    
    # Enhanced metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_per_class': {cls: f1_score(y_true, y_pred, labels=[cls], average=None)[0] 
                        for cls in range(1, 6)},
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5]).tolist(),
        'classification_report': classification_report(y_true, y_pred, output_dict=True),
        'best_params': {k: float(v) if isinstance(v, np.floating) else v 
                       for k, v in best_params.items()},
        'training_time': time.time() - start_time,
        'best_rounds': best_round
    }
    
    # Save outputs
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    timestamp = int(time.time())
    model_path = f"{MODELS_DIR}/xgboost_enhanced_{timestamp}.pkl"
    joblib.dump(best_model, model_path)
    
    with open(f"{METRICS_DIR}/xgboost_enhanced_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    pd.DataFrame({"Actual": y_true, "Predicted": y_pred}).to_csv(
        f"{PREDICTIONS_DIR}/xgboost_enhanced_predictions.csv", index=False)
    
    log_message("\n🎯 Final Model Performance:")
    log_message(f"🏆 Accuracy: {metrics['accuracy']:.4f}")
    log_message(f"⚖️ Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    log_message(f"📊 F1 Macro: {metrics['f1_macro']:.4f}")
    log_message(f"📈 F1 Weighted: {metrics['f1_weighted']:.4f}")
    log_message("\n📝 Class-wise F1 Scores:")
    for cls, score in metrics['f1_per_class'].items():
        log_message(f"Class {cls}: {score:.4f}")
    
    return metrics

if __name__ == "__main__":
    log_message("🚀 Starting Enhanced XGBoost Training Pipeline")
    try:
        X, y, _ = load_and_preprocess_data()
        results = train_xgboost(X, y)
        log_message(f"✅ Training completed! Final F1 Macro: {results['f1_macro']:.4f}")
    except Exception as e:
        log_message(f"❌ Training failed: {str(e)}")
        raise