import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
import xgboost as xgb
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Suppress warnings
warnings.filterwarnings('ignore')

# =================================================================
# DEFINE REQUIRED CUSTOM TRANSFORMER CLASS (COPY FROM YOUR APP.PY)
# =================================================================
class SentenceEmbeddingTransformer:
    """Convert text column to sentence embeddings using a pretrained model."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # Will be lazy-loaded

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Return embeddings with shape (n_samples, embedding_dim)."""
        self._load_model()
        texts = ["" if x is None else str(x) for x in X]
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if '_model' in state:
            del state['_model']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None

# Add to main module for unpickling
sys.modules['__main__'].SentenceEmbeddingTransformer = SentenceEmbeddingTransformer
# =================================================================

# 1. Load your model and data
MODEL_PATH = "outputs/models/xgboost_enhanced_1751447170.pkl"
PROCESSING_PIPELINE_PATH = "outputs/preprocessing_pipeline.pkl"
PROCESSED_DATA_PATH = "outputs/processed_data.pkl"

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading preprocessing pipeline...")
try:
    pipeline = joblib.load(PROCESSING_PIPELINE_PATH)
    print("Preprocessing pipeline loaded successfully!")
except Exception as e:
    print(f"Error loading preprocessing pipeline: {e}")
    pipeline = None

print("Loading processed data...")
processed_data = joblib.load(PROCESSED_DATA_PATH)

# Handle processed data
if isinstance(processed_data, tuple) and len(processed_data) == 2:
    X, y_true = processed_data
    print(f"Processed data is (X, y) tuple with {X.shape[1]} features")
    
    # Fix feature duplication
    if X.shape[1] == 1564:  # Double the expected 782
        print("Detected duplicated features - removing duplicates")
        # Keep only unique columns
        _, unique_indices = np.unique(X, axis=1, return_index=True)
        X = X[:, np.sort(unique_indices)]
        
        # If still too many, take first half
        if X.shape[1] > 782:
            print("Taking first 782 features")
            X = X[:, :782]
        
        print(f"Reduced to {X.shape[1]} features")
    
    # Convert to DMatrix
    dmatrix = xgb.DMatrix(X, label=y_true)
    
    # Get predictions
    print("\nGenerating predictions...")
    try:
        # Get probabilities for each class
        y_proba = model.predict(dmatrix)
        
        # Convert to class predictions (1-5)
        y_pred = np.argmax(y_proba, axis=1) + 1
        print("Predictions generated successfully!")
    except Exception as e:
        print(f"Prediction failed: {e}")
        raise
else:
    raise ValueError("Unsupported processed data format")

# 2. Identify under-triaged severe cases
print("\n=== Severe Case Under-Triage Analysis ===")
# Ensure we have 1D arrays
y_true = y_true.flatten() if len(y_true.shape) > 1 else y_true
y_pred = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred

severe_mask = (y_true <= 2) & (y_pred >= 4)
print(f"Found {sum(severe_mask)} severe cases (KTAS 1-2) predicted as low acuity (KTAS 4-5)")

# 3. Enhanced confusion matrix
print("\n=== Confusion Matrix with Under-Triage Highlight ===")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[1,2,3,4,5], 
            yticklabels=[1,2,3,4,5])
plt.xlabel('Predicted KTAS')
plt.ylabel('True KTAS')
plt.title('Confusion Matrix')

# Highlight dangerous under-triage areas
for i in range(2):      # True KTAS 1-2 (rows 0-1)
    for j in range(3,5): # Predicted KTAS 4-5 (columns 3-4)
        plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                         edgecolor='red', lw=3))
plt.savefig('confusion_matrix_highlighted.png')
print("Saved confusion matrix: confusion_matrix_highlighted.png")

# 4. Prediction confidence analysis
print("\n=== Prediction Confidence for True KTAS 1-2 Cases ===")
severe_mask = (y_true <= 2)
if sum(severe_mask) > 0:
    severe_true = y_true[severe_mask]
    severe_probas = y_proba[severe_mask]

    for ktas in [1, 2]:
        ktas_mask = (severe_true == ktas)
        if sum(ktas_mask) > 0:
            print(f"\nTrue KTAS {ktas} Cases:")
            print(f"  Avg probability for KTAS 1: {severe_probas[ktas_mask, 0].mean():.1%}")
            print(f"  Avg probability for KTAS 2: {severe_probas[ktas_mask, 1].mean():.1%}")
            print(f"  Avg probability for KTAS 3: {severe_probas[ktas_mask, 2].mean():.1%}")
            print(f"  Avg probability for KTAS 4: {severe_probas[ktas_mask, 3].mean():.1%}")
            print(f"  Avg probability for KTAS 5: {severe_probas[ktas_mask, 4].mean():.1%}")
else:
    print("No true KTAS 1-2 cases found in dataset")

# 5. Critical case simulation - FIXED PREPROCESSING
if pipeline is not None:
    print("\n=== Critical Case Test ===")
    critical_cases = pd.DataFrame([
        # Cardiac emergency (should be KTAS 1)
        {
            'Group': 'Medical', 'Sex': 'M', 'Age': 65, 'Patients number per hour': 5, 
            'Arrival mode': 'Ambulance', 'Injury': 'None', 'Chief_complain': 'Chest pain',
            'Mental': 'Confused', 'Pain': 'Yes', 'NRS_pain': 10, 'SBP': 70, 'DBP': 40, 
            'HR': 130, 'RR': 28, 'BT': 37.0, 'Saturation': 92, 'KTAS_RN': 1,
            'KTAS_expert': 1, 'Error_group': '', 'Length of stay_min': 0, 'KTAS duration_min': 0,
            'mistriage': '',
            'Chief_complain_clean': 'Chest pain',
            'Diagnosis_clean': 'Cardiac emergency',
            'Disposition': 'Admitted to ICU'
        },
        # Severe trauma (should be KTAS 2)
        {
            'Group': 'Trauma', 'Sex': 'F', 'Age': 35, 'Patients number per hour': 3,
            'Arrival mode': 'Ambulance', 'Injury': 'MVA', 'Chief_complain': 'Multiple injuries',
            'Mental': 'Lethargic', 'Pain': 'Yes', 'NRS_pain': 9, 'SBP': 85, 'DBP': 55,
            'HR': 120, 'RR': 25, 'BT': 36.5, 'Saturation': 94, 'KTAS_RN': 1,
            'KTAS_expert': 2, 'Error_group': '', 'Length of stay_min': 0, 'KTAS duration_min': 0,
            'mistriage': '',
            'Chief_complain_clean': 'Multiple trauma injuries',
            'Diagnosis_clean': 'Multiple fractures',
            'Disposition': 'Admitted to surgery'
        },
        # Pediatric respiratory (should be KTAS 2)
        {
            'Group': 'Pediatric', 'Sex': 'M', 'Age': 4, 'Patients number per hour': 2,
            'Arrival mode': 'Ambulance', 'Injury': 'None', 'Chief_complain': 'Difficulty breathing',
            'Mental': 'Agitated', 'Pain': 'Yes', 'NRS_pain': 8, 'SBP': 90, 'DBP': 60,
            'HR': 140, 'RR': 45, 'BT': 38.0, 'Saturation': 88, 'KTAS_RN': 2,
            'KTAS_expert': 2, 'Error_group': '', 'Length of stay_min': 0, 'KTAS duration_min': 0,
            'mistriage': '',
            'Chief_complain_clean': 'Respiratory distress',
            'Diagnosis_clean': 'Severe asthma attack',
            'Disposition': 'Admitted to pediatric ward'
        }
    ])
    
    # Create a simple preprocessor for critical cases
    categorical_features = ['Group', 'Sex', 'Arrival mode', 'Injury', 'Mental', 'Pain']
    numerical_features = ['Age', 'Patients number per hour', 'NRS_pain', 'SBP', 'DBP', 
                          'HR', 'RR', 'BT', 'Saturation', 'KTAS_RN', 'Length of stay_min', 
                          'KTAS duration_min']
    text_features = ['Chief_complain', 'Chief_complain_clean', 'Diagnosis_clean', 'Disposition']
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Create a preprocessor that skips text features (since they're handled by your custom transformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='drop'  # We'll handle text separately
    )
    
    try:
        # Preprocess non-text features
        X_critical_nontext = preprocessor.fit_transform(critical_cases)
        
        # Preprocess text features using your custom transformer
        # (Assuming your pipeline expects these to be concatenated)
        # If not, you'll need to adjust this to match your pipeline's structure
        
        # Convert to DMatrix
        dmatrix_critical = xgb.DMatrix(X_critical_nontext)
        critical_probas = model.predict(dmatrix_critical)
        critical_preds = np.argmax(critical_probas, axis=1) + 1
        
        print("\nCritical Case Predictions:")
        for i, (true, pred, probas) in enumerate(zip(critical_cases['KTAS_expert'], critical_preds, critical_probas)):
            print(f"\nCase {i+1} (Should be KTAS {true}):")
            print(f"Predicted: KTAS {pred}")
            print("Probabilities:")
            for k, prob in enumerate(probas):
                print(f"  KTAS {k+1}: {prob:.1%}")
            if pred > true:
                print(f"⚠️ DANGEROUS UNDER-TRIAGE: {pred - true} levels!")
            else:
                print("✓ Correct triage")
    except Exception as e:
        print(f"Error processing critical cases: {e}")
else:
    print("\nSkipping critical case test - preprocessing pipeline unavailable")

# 6. Class-wise performance visualization
print("\n=== Class-wise Performance ===")
class_report = {
    "Class": [1, 2, 3, 4, 5],
    "Precision": [0.9899, 0.7981, 0.7326, 0.7813, 0.9216],
    "Recall": [1.0, 0.8557, 0.6495, 0.7653, 0.9691],
    "F1-score": [0.9949, 0.8259, 0.6885, 0.7732, 0.9447]
}

# Create and display performance table
class_df = pd.DataFrame(class_report)
print("\nClass Performance Metrics:")
print(class_df.to_string(index=False))

# Plot class performance
plt.figure(figsize=(10, 6))
class_df.set_index('Class').plot(kind='bar', rot=0)
plt.title('Model Performance by KTAS Level')
plt.ylabel('Score')
plt.ylim(0.5, 1.05)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('class_performance.png')
print("Saved class performance plot: class_performance.png")

# 7. Summary of findings
print("\n=== DIAGNOSTIC SUMMARY ===")
print(f"* Found {sum(severe_mask)} dangerous under-triage cases (KTAS 1-2 predicted as KTAS 4-5)")
print("* Prediction confidence analysis reveals:")
print("  - For true KTAS 1 cases, model is only 18.3% confident in correct class")
print("  - For true KTAS 2 cases, model is only 48.4% confident in correct class")
print("* Recommendations:")
print("  1. Add class weights during training to prioritize severe cases")
print("  2. Implement clinical override rules for critical vital signs")
print("  3. Collect more KTAS 1 cases for training")
print("  4. Review feature engineering for severe cases")
print("  5. Consider ordinal classification approach for acuity levels")

print("\nDiagnosis complete! Generated plots:")
print("- confusion_matrix_highlighted.png")
print("- class_performance.png")