import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# Load Dataset
# ============================================================

DATA_PATH = "data/creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ============================================================
# Separate Features and Target
# ============================================================

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===== TRAIN / TEST SPLIT =====")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# ============================================================
# Feature Scaling
# ============================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n===== FEATURE SCALING =====")
print("Scaled training shape:", X_train_scaled.shape)
print("Scaled testing shape :", X_test_scaled.shape)

# ============================================================
# Baseline Random Forest
# ============================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("\n===== TRAINING BASELINE RANDOM FOREST =====")

baseline_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight=None
)

baseline_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_baseline = baseline_model.predict(X_test_scaled)

# Probability of fraud
y_prob_baseline = baseline_model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# Baseline Evaluation
# ============================================================

print("\n===== BASELINE CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred_baseline,
        target_names=["Legitimate", "Fraud"],
        digits=4
    )
)

print("\n===== BASELINE CONFUSION MATRIX =====")

print(
    confusion_matrix(
        y_test,
        y_pred_baseline
    )
)

print("\n===== BASELINE ROC-AUC =====")

baseline_roc_auc = roc_auc_score(
    y_test,
    y_prob_baseline
)

print(f"ROC-AUC: {baseline_roc_auc:.4f}")

# ============================================================
# SMOTE - Synthetic Minority Oversampling
# ============================================================

from imblearn.over_sampling import SMOTE

print("\n===== APPLYING SMOTE =====")

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nClass distribution BEFORE SMOTE:")
print(y_train.value_counts())

print("\nClass distribution AFTER SMOTE:")
print(y_train_smote.value_counts())

print("\nTraining data BEFORE SMOTE:", X_train_scaled.shape)
print("Training data AFTER SMOTE :", X_train_smote.shape)

# ============================================================
# Random Forest with SMOTE
# ============================================================

print("\n===== TRAINING SMOTE RANDOM FOREST =====")

smote_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight=None
)

smote_model.fit(
    X_train_smote,
    y_train_smote
)

# Predictions
y_pred_smote = smote_model.predict(X_test_scaled)

# Fraud probabilities
y_prob_smote = smote_model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# SMOTE Model Evaluation
# ============================================================

print("\n===== SMOTE CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred_smote,
        target_names=["Legitimate", "Fraud"],
        digits=4
    )
)


print("\n===== SMOTE CONFUSION MATRIX =====")

print(
    confusion_matrix(
        y_test,
        y_pred_smote
    )
)


print("\n===== SMOTE ROC-AUC =====")

smote_roc_auc = roc_auc_score(
    y_test,
    y_prob_smote
)

print(f"ROC-AUC: {smote_roc_auc:.4f}")

# ============================================================
# Model Comparison
# ============================================================

from sklearn.metrics import precision_score, recall_score, f1_score

baseline_precision = precision_score(y_test, y_pred_baseline)
baseline_recall = recall_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline)

smote_precision = precision_score(y_test, y_pred_smote)
smote_recall = recall_score(y_test, y_pred_smote)
smote_f1 = f1_score(y_test, y_pred_smote)


comparison = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ],
    "Baseline Random Forest": [
        baseline_precision,
        baseline_recall,
        baseline_f1,
        baseline_roc_auc
    ],
    "SMOTE Random Forest": [
        smote_precision,
        smote_recall,
        smote_f1,
        smote_roc_auc
    ]
})

print("\n===== MODEL COMPARISON =====")
print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nComparison saved to reports/model_comparison.csv")

# ============================================================
# Save Trained Models
# ============================================================

import joblib

joblib.dump(
    baseline_model,
    "models/baseline_random_forest.joblib"
)

joblib.dump(
    smote_model,
    "models/smote_random_forest.joblib"
)

joblib.dump(
    scaler,
    "models/scaler.joblib"
)

print("\n===== MODELS SAVED =====")
print("Baseline model saved.")
print("SMOTE model saved.")
print("Scaler saved.")