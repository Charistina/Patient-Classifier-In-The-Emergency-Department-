# File: src/Compare_models.py

import joblib
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    precision_recall_curve, auc
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import warnings
warnings.filterwarnings("ignore")

# --- Load processed data ---
data_path = "outputs/processed_data.pkl"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Preprocessed data not found at {data_path}. Please run Processing_data.py first.")

X, y = joblib.load(data_path)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Model Evaluation ---
model_names = ["DecisionTree", "RandomForest", "LogisticRegression", "KNeighbors", "SVC"]
metrics_summary = []
os.makedirs("outputs", exist_ok=True)

for name in model_names:
    model_path = f"outputs/{name}_model.pkl"
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found: {model_path}")
        continue

    print(f"\n🔍 Evaluating {name}")
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')

    print(classification_report(y_test, y_pred))

    metrics_summary.append({
        "Model": name,
        "Accuracy": acc,
        "F1_macro": f1_macro,
        "F1_weighted": f1_weighted,
        "Precision": precision_weighted,
        "Recall": recall_weighted
    })

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/{name}_confusion_matrix.png")
    plt.close()

    # Precision-Recall Curve
    y_test_bin = label_binarize(y_test, classes=[1, 2, 3, 4, 5])
    n_classes = y_test_bin.shape[1]

    if hasattr(model, "predict_proba"):
        classifier = OneVsRestClassifier(model)
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    else:
        continue  # Skip PR curve

    precision = dict()
    recall = dict()
    pr_auc = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    plt.figure(figsize=(8, 6))
    colors = cycle(['blue', 'green', 'red', 'purple', 'orange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color,
                 label=f"Class {i+1} (AUC={pr_auc[i]:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {name}")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"outputs/{name}_precision_recall_curve.png")
    plt.close()

# --- Save metrics ---
df_metrics = pd.DataFrame(metrics_summary)
df_metrics.to_csv("outputs/model_metrics_summary.csv", index=False)

# --- Bar Plot ---
plt.figure(figsize=(10, 6))
df_metrics.set_index("Model")[["Accuracy", "F1_macro", "F1_weighted"]].plot.bar(rot=0)
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("outputs/model_metrics_comparison.png")
plt.show()

# --- Cross-validation ---
print("\n📊 5-Fold Cross-Validation Scores (on training set):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name in model_names:
    model_path = f"outputs/{name}_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"🔁 {name}: Mean CV Accuracy = {scores.mean():.4f} ± {scores.std():.4f}")

# --- Save best model ---
best_row = df_metrics.sort_values("F1_weighted", ascending=False).iloc[0]
best_model_name = best_row["Model"]
print(f"\n🏆 Best model based on F1_weighted: {best_model_name}")

shutil.copy(f"outputs/{best_model_name}_model.pkl", "outputs/best_model.pkl")
print("💾 Saved best model to outputs/best_model.pkl")
