import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# ============================================================
# Load Dataset
# ============================================================

DATA_PATH = "data/creditcard.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]


# ============================================================
# Recreate Test Split
# ============================================================

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# Load Scaler and Models
# ============================================================

scaler = joblib.load(
    "models/scaler.joblib"
)

baseline_model = joblib.load(
    "models/baseline_random_forest.joblib"
)

smote_model = joblib.load(
    "models/smote_random_forest.joblib"
)


# ============================================================
# Scale Test Data
# ============================================================

X_test_scaled = scaler.transform(X_test)


# ============================================================
# Generate Fraud Probabilities
# ============================================================

baseline_prob = baseline_model.predict_proba(
    X_test_scaled
)[:, 1]

smote_prob = smote_model.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# ROC CURVE
# ============================================================

baseline_fpr, baseline_tpr, _ = roc_curve(
    y_test,
    baseline_prob
)

smote_fpr, smote_tpr, _ = roc_curve(
    y_test,
    smote_prob
)

baseline_auc = auc(
    baseline_fpr,
    baseline_tpr
)

smote_auc = auc(
    smote_fpr,
    smote_tpr
)


plt.figure(figsize=(9, 6))

plt.plot(
    baseline_fpr,
    baseline_tpr,
    label=f"Baseline RF (AUC = {baseline_auc:.4f})"
)

plt.plot(
    smote_fpr,
    smote_tpr,
    label=f"SMOTE RF (AUC = {smote_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve: Baseline vs SMOTE Random Forest"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "images/roc_curve_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

baseline_precision, baseline_recall, _ = precision_recall_curve(
    y_test,
    baseline_prob
)

smote_precision, smote_recall, _ = precision_recall_curve(
    y_test,
    smote_prob
)

baseline_ap = average_precision_score(
    y_test,
    baseline_prob
)

smote_ap = average_precision_score(
    y_test,
    smote_prob
)


plt.figure(figsize=(9, 6))

plt.plot(
    baseline_recall,
    baseline_precision,
    label=f"Baseline RF (AP = {baseline_ap:.4f})"
)

plt.plot(
    smote_recall,
    smote_precision,
    label=f"SMOTE RF (AP = {smote_ap:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve: Baseline vs SMOTE Random Forest"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "images/precision_recall_curve_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# Evaluation Summary
# ============================================================

print("\n===== CURVE EVALUATION =====")

print(f"Baseline ROC-AUC: {baseline_auc:.4f}")
print(f"SMOTE ROC-AUC   : {smote_auc:.4f}")

print(f"Baseline Average Precision: {baseline_ap:.4f}")
print(f"SMOTE Average Precision   : {smote_ap:.4f}")

print("\nGraphs saved successfully:")
print("images/roc_curve_comparison.png")
print("images/precision_recall_curve_comparison.png")