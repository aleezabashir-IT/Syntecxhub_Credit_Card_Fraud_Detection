import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/creditcard.csv"

FINAL_THRESHOLD = 0.80


# ============================================================
# Load Dataset
# ============================================================

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
# Load Final Model and Scaler
# ============================================================

model = joblib.load(
    "models/final_fraud_detector.joblib"
)

scaler = joblib.load(
    "models/final_scaler.joblib"
)


# ============================================================
# Generate Predictions
# ============================================================

X_test_scaled = scaler.transform(X_test)

fraud_probability = model.predict_proba(
    X_test_scaled
)[:, 1]

y_pred = (
    fraud_probability >= FINAL_THRESHOLD
).astype(int)


# ============================================================
# Calculate Metrics
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    fraud_probability
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# Load Existing Analysis Files
# ============================================================

comparison = pd.read_csv(
    "reports/model_comparison.csv"
)

threshold_analysis = pd.read_csv(
    "reports/threshold_analysis.csv"
)

feature_importance = pd.read_csv(
    "reports/feature_importance.csv"
)


# ============================================================
# Create Final Report
# ============================================================

report_path = "reports/final_evaluation_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "CREDIT CARD FRAUD DETECTION\n"
    )

    report.write(
        "FINAL MODEL EVALUATION REPORT\n"
    )

    report.write(
        "=" * 60 + "\n\n"
    )


    # Dataset
    report.write(
        "1. DATASET SUMMARY\n"
    )

    report.write(
        "-" * 60 + "\n"
    )

    report.write(
        f"Total transactions: {len(df):,}\n"
    )

    report.write(
        f"Total features: {X.shape[1]}\n"
    )

    report.write(
        f"Legitimate transactions: "
        f"{(y == 0).sum():,}\n"
    )

    report.write(
        f"Fraudulent transactions: "
        f"{(y == 1).sum():,}\n"
    )

    report.write(
        f"Fraud percentage: "
        f"{(y == 1).mean() * 100:.4f}%\n\n"
    )


    # Model comparison
    report.write(
        "2. MODEL COMPARISON\n"
    )

    report.write(
        "-" * 60 + "\n"
    )

    report.write(
        comparison.to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )


    # Threshold
    report.write(
        "3. THRESHOLD ANALYSIS\n"
    )

    report.write(
        "-" * 60 + "\n"
    )

    report.write(
        f"Selected threshold: "
        f"{FINAL_THRESHOLD:.2f}\n\n"
    )

    report.write(
        threshold_analysis.to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )


    # Final metrics
    report.write(
        "4. FINAL MODEL PERFORMANCE\n"
    )

    report.write(
        "-" * 60 + "\n"
    )

    report.write(
        f"Precision: {precision:.4f}\n"
    )

    report.write(
        f"Recall:    {recall:.4f}\n"
    )

    report.write(
        f"F1-Score:  {f1:.4f}\n"
    )

    report.write(
        f"ROC-AUC:   {roc_auc:.4f}\n\n"
    )


    # Confusion matrix
    report.write(
        "5. CONFUSION MATRIX\n"
    )

    report.write(
        "-" * 60 + "\n"
    )

    report.write(
        str(cm)
    )

    report.write(
        "\n\n"
    )


    # Feature importance
    report.write(
        "6. TOP 15 FEATURE IMPORTANCE\n"
    )

    report.write(
        "-" * 60 + "\n"
    )

    report.write(
        feature_importance.head(15).to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )


    # Classification report
    report.write(
        "7. CLASSIFICATION REPORT\n"
    )

    report.write(
        "-" * 60 + "\n"
    )

    report.write(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Legitimate",
                "Fraud"
            ],
            digits=4
        )
    )


print(
    "Final evaluation report created successfully."
)

print(
    report_path
)