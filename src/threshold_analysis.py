import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
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
# Load Saved Models
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
# Fraud Probabilities
# ============================================================

baseline_prob = baseline_model.predict_proba(
    X_test_scaled
)[:, 1]

smote_prob = smote_model.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# Threshold Analysis
# ============================================================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]


results = []


for threshold in thresholds:

    baseline_pred = (
        baseline_prob >= threshold
    ).astype(int)

    smote_pred = (
        smote_prob >= threshold
    ).astype(int)


    # Baseline metrics
    baseline_precision = precision_score(
        y_test,
        baseline_pred,
        zero_division=0
    )

    baseline_recall = recall_score(
        y_test,
        baseline_pred,
        zero_division=0
    )

    baseline_f1 = f1_score(
        y_test,
        baseline_pred,
        zero_division=0
    )


    # SMOTE metrics
    smote_precision = precision_score(
        y_test,
        smote_pred,
        zero_division=0
    )

    smote_recall = recall_score(
        y_test,
        smote_pred,
        zero_division=0
    )

    smote_f1 = f1_score(
        y_test,
        smote_pred,
        zero_division=0
    )


    results.append({
        "Threshold": threshold,

        "Baseline Precision": baseline_precision,
        "Baseline Recall": baseline_recall,
        "Baseline F1": baseline_f1,

        "SMOTE Precision": smote_precision,
        "SMOTE Recall": smote_recall,
        "SMOTE F1": smote_f1
    })


# ============================================================
# Create Results DataFrame
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# Display Results
# ============================================================

print("\n===== THRESHOLD ANALYSIS =====")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# Find Best SMOTE F1 Threshold
# ============================================================

best_row = results_df.loc[
    results_df["SMOTE F1"].idxmax()
]

print("\n===== BEST SMOTE THRESHOLD =====")

print(
    f"Best Threshold: {best_row['Threshold']:.2f}"
)

print(
    f"Precision: {best_row['SMOTE Precision']:.4f}"
)

print(
    f"Recall: {best_row['SMOTE Recall']:.4f}"
)

print(
    f"F1-Score: {best_row['SMOTE F1']:.4f}"
)


# ============================================================
# Save Results
# ============================================================

results_df.to_csv(
    "reports/threshold_analysis.csv",
    index=False
)


# ============================================================
# Plot SMOTE Threshold Performance
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["Threshold"],
    results_df["SMOTE Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["SMOTE Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["SMOTE F1"],
    marker="o",
    label="F1-Score"
)

plt.xlabel("Classification Threshold")

plt.ylabel("Score")

plt.title(
    "SMOTE Random Forest: Threshold Performance"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "images/threshold_analysis.png",
    dpi=300
)

plt.show()


print("\nThreshold analysis saved successfully.")

print(
    "reports/threshold_analysis.csv"
)

print(
    "images/threshold_analysis.png"
)