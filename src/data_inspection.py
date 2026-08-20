import pandas as pd

# Dataset path
DATA_PATH = "data/creditcard.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic information
print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== CLASS DISTRIBUTION =====")
print(df["Class"].value_counts())

print("\n===== CLASS PERCENTAGE =====")
print(df["Class"].value_counts(normalize=True) * 100)