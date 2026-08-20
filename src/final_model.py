import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/creditcard.csv"

FINAL_THRESHOLD = 0.80

RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

print("===== LOADING DATASET =====")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop("Class", axis=1)
y = df["Class"]

print("Features:", X.shape)
print("Target:", y.shape)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n===== TRAIN / TEST SPLIT =====")

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# FEATURE SCALING
# ============================================================

print("\n===== FEATURE SCALING =====")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# APPLY SMOTE
# ============================================================

print("\n===== APPLYING SMOTE =====")

smote = SMOTE(
    random_state=RANDOM_STATE
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("Before SMOTE:", y_train.value_counts().to_dict())

print("After SMOTE :", y_train_smote.value_counts().to_dict())


# ============================================================
# TRAIN FINAL RANDOM FOREST
# ============================================================

print("\n===== TRAINING FINAL RANDOM FOREST =====")

final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight=None
)

final_model.fit(
    X_train_smote,
    y_train_smote
)

print("Final model training completed.")


# ============================================================
# FRAUD PROBABILITIES
# ============================================================

print("\n===== GENERATING PREDICTIONS =====")

fraud_probability = final_model.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# APPLY OPTIMIZED THRESHOLD
# ============================================================

y_pred_final = (
    fraud_probability >= FINAL_THRESHOLD
).astype(int)


# ============================================================
# FINAL CLASSIFICATION REPORT
# ============================================================

print("\n===== FINAL CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred_final,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        digits=4
    )
)


# ============================================================
# FINAL CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred_final
)

print("\n===== FINAL CONFUSION MATRIX =====")

print(cm)


# ============================================================
# FINAL METRICS
# ============================================================

precision = precision_score(
    y_test,
    y_pred_final,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred_final,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred_final,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    fraud_probability
)


print("\n===== FINAL MODEL METRICS =====")

print(f"Threshold : {FINAL_THRESHOLD:.2f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# SAVE FINAL MODEL
# ============================================================

joblib.dump(
    final_model,
    "models/final_fraud_detector.joblib"
)

joblib.dump(
    scaler,
    "models/final_scaler.joblib"
)


print("\n===== MODEL SAVED =====")

print(
    "models/final_fraud_detector.joblib"
)

print(
    "models/final_scaler.joblib"
)


# ============================================================
# SAVE FINAL CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "Final Fraud Detection Confusion Matrix"
)

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

plt.yticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)


for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "images/final_confusion_matrix.png",
    dpi=300
)

plt.show()


print(
    "\nFinal confusion matrix saved:"
)

print(
    "images/final_confusion_matrix.png"
)