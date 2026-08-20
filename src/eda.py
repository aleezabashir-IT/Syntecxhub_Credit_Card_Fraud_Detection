import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset path
DATA_PATH = "data/creditcard.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Create class labels for visualization
class_labels = {
    0: "Legitimate",
    1: "Fraud"
}

# Count transactions by class
class_counts = df["Class"].value_counts().sort_index()

# Create visualization
plt.figure(figsize=(8, 5))

sns.barplot(
    x=class_counts.index.map(class_labels),
    y=class_counts.values
)

plt.title("Credit Card Transaction Class Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

# Add values above bars
for index, value in enumerate(class_counts.values):
    plt.text(
        index,
        value + 3000,
        f"{value:,}",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()

# Save visualization
plt.savefig(
    "images/class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Print class statistics
print("\n===== CLASS DISTRIBUTION =====")
print(class_counts)

print("\n===== CLASS PERCENTAGES =====")
print(
    df["Class"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)
# ============================================================
# 1. Transaction Amount Distribution
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Amount",
    hue="Class",
    bins=50,
    kde=True,
    element="step",
    stat="density",
    common_norm=False
)

plt.title("Transaction Amount Distribution by Class")
plt.xlabel("Transaction Amount")
plt.ylabel("Density")

plt.tight_layout()

plt.savefig(
    "images/amount_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 2. Transaction Amount: Legitimate vs Fraud
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

plt.title("Transaction Amount Comparison")
plt.xlabel("Transaction Type")
plt.ylabel("Transaction Amount")

plt.tight_layout()

plt.savefig(
    "images/amount_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 3. Transaction Time Distribution
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Time",
    hue="Class",
    bins=50,
    element="step",
    stat="density",
    common_norm=False
)

plt.title("Transaction Time Distribution by Class")
plt.xlabel("Time")
plt.ylabel("Density")

plt.tight_layout()

plt.savefig(
    "images/time_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 4. Statistical Summary
# ============================================================

print("\n===== TRANSACTION AMOUNT STATISTICS =====")

print(
    df.groupby("Class")["Amount"]
    .describe()
)

print("\n===== AVERAGE TRANSACTION AMOUNT =====")

print(
    df.groupby("Class")["Amount"]
    .mean()
)

print("\n===== MEDIAN TRANSACTION AMOUNT =====")

print(
    df.groupby("Class")["Amount"]
    .median()
)
# ============================================================
# 5. Duplicate Check
# ============================================================

duplicate_count = df.duplicated().sum()

print("\n===== DUPLICATE ROWS =====")
print(f"Number of duplicate rows: {duplicate_count}")