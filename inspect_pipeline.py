# inspect_pipeline.py
import joblib
import types
import sys
import pandas as pd
import numpy as np

# Define dummy transformer class
class SentenceEmbeddingTransformer:
    def fit(self, X, y=None): return self
    def transform(self, X): return X
    def fit_transform(self, X, y=None): return X

# Mock the original module structure
sys.modules['src'] = types.ModuleType('src')
sys.modules['src.Processing_data'] = types.ModuleType('src.Processing_data')
sys.modules['src.Processing_data'].SentenceEmbeddingTransformer = SentenceEmbeddingTransformer

def inspect_pipeline():
    try:
        # Load the pipeline (which is actually a ColumnTransformer)
        preprocessor = joblib.load("outputs/preprocessing_pipeline.pkl")
        print("✅ Pipeline (ColumnTransformer) loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load pipeline: {str(e)}")
        return

    print("\n" + "="*50)
    print("ColumnTransformer Configuration")
    print("="*50)

    # Check if this is a ColumnTransformer or full Pipeline
    if hasattr(preprocessor, 'transformers'):
        transformers = preprocessor.transformers
    else:
        print("❌ Loaded object is not a ColumnTransformer")
        return

    for name, transformer, cols in transformers:
        print(f"\nTransformer: {name}")
        print(f"Columns ({len(cols)}): {cols}")
        print(f"Transformer type: {type(transformer).__name__}")
        
        if hasattr(transformer, 'steps'):
            print("\nSub-steps:")
            for step_name, step_trans in transformer.steps:
                print(f"  - {step_name}: {type(step_trans).__name__}")
        
        # Special warnings
        if name == 'num':
            print("\n⚠️ NUMERIC TRANSFORMER WARNING:")
            print("These columns MUST contain only numeric values")
            print("Check for any string values like 'unknown'")

    print("\n" + "="*50)
    print("Embedding Columns Check")
    print("="*50)

    embed_cols = ['Chief_complain_clean', 'Diagnosis_clean']
    found_in = []
    for name, _, cols in transformers:
        if set(embed_cols) & set(cols):
            found_in.append(name)
    
    if found_in:
        print(f"Found embedding columns in: {', '.join(found_in)}")
        if 'num' in found_in:
            print("\n❌ CRITICAL ISSUE: Embedding columns in numeric transformer!")
            print("This explains the 'unknown' string conversion error")
    else:
        print("❌ Embedding columns not found in any transformer")

    print("\n" + "="*50)
    print("Recommended Actions")
    print("="*50)
    if 'num' in found_in:
        print("1. TEMPORARY FIX (app.py):")
        print("   - Manually exclude embedding columns from numeric processing")
        print("   - Add: numeric_columns = [col for col in numeric_columns if col not in embed_cols]")
        print("\n2. PERMANENT FIX:")
        print("   - Retrain model with proper column specification")
        print("   - Ensure embedding columns are routed to text transformer")
    else:
        print("The error might be coming from another column")
        print("Check all columns in the numeric transformer for string values")

if __name__ == "__main__":
    inspect_pipeline()