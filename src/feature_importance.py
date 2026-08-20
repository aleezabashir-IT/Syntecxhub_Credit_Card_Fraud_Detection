import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ============================================================
# Load Dataset
# ============================================================

DATA_PATH = "data/creditcard.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# Prepare Features
# ============================================================

X = df.drop("Class", axis=1)


# ============================================================
# Load Final Model
# ============================================================

model = joblib.load(
    "models/final_fraud_detector.joblib"
)


# ============================================================
# Extract Feature Importance
# ============================================================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})


# ============================================================
# Sort Features
# ============================================================

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


# ============================================================
# Display Top Features
# ============================================================

print("\n===== TOP 15 IMPORTANT FEATURES =====")

print(
    feature_importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# Save Feature Importance
# ============================================================

feature_importance.to_csv(
    "reports/feature_importance.csv",
    index=False
)


# ============================================================
# Plot Top 15 Features
# ============================================================

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance Score")

plt.ylabel("Feature")

plt.title(
    "Top 15 Features Used by Random Forest"
)

plt.tight_layout()

plt.savefig(
    "images/feature_importance.png",
    dpi=300
)

plt.show()


print("\nFeature importance analysis completed.")

print(
    "reports/feature_importance.csv"
)

print(
    "images/feature_importance.png"
)