<div align="center">

# 🛡️ Credit Card Fraud Detection System
### Imbalance-Aware Machine Learning Pipeline for Fraud Risk Analytics

An end-to-end production-ready machine learning framework designed to detect financial fraud in severely imbalanced transaction data.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-red?style=for-the-badge)](https://imbalanced-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br />

| 💳 Total Transactions | 🚨 Fraud Cases | 🎯 Precision (t=0.80) | 📈 ROC-AUC Score |
| :---: | :---: | :---: | :---: |
| **284,807** | **492 (0.17%)** | **97.40%** | **0.9685** |

</div>

---

# 📖 Project Overview

Credit card fraud detection is a challenging binary classification problem due to extreme class imbalance, where fraudulent transactions represent a fraction of total activity. A model predicting all transactions as legitimate achieves standard accuracy over 99% while completely failing its core task.

This project addresses the challenge through an end-to-end machine learning pipeline:

**Data Inspection → Exploratory Analysis → Preprocessing → Baseline Model → SMOTE → Model Comparison → Threshold Optimization → Final Evaluation**

The system provides:

- **End-to-End ML Pipeline:** Complete lifecycle handling from data loading to threshold tuning.
- **Handling Extreme Imbalance:** SMOTE integration to prevent model bias towards non-fraud cases.
- **Threshold Optimization:** Moving beyond default 0.50 thresholds to maximize operational utility.
- **Comprehensive Evaluation:** Metric emphasis on Precision, Recall, F1-score, ROC-AUC, and PR-AUC.

---

# ✨ Key Features

- 🎯 **Imbalance-Aware Classification:** Handles extreme class skew (0.1727% minority class).
- ⚖️ **Synthetic Sampling:** Integrates SMOTE strictly on training splits to prevent data leakage.
- 🎚️ **Probability Threshold Tuning:** Custom threshold selection optimizing operational precision/recall.
- 📊 **Visual Analytics:** ROC curves, PR curves, feature importance, and confusion matrices.
- 💾 **Model Persistence:** Serialized model pipelines and scalers via `joblib`.
- 📋 **Automated Reporting:** Programmatic generation of evaluation metrics and summary reports.

---

# 🛠 Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **Python** | Core Programming Language |
| **Pandas & NumPy** | Data Manipulation & Processing |
| **Scikit-learn** | Model Training, Scaling & Evaluation Metrics |
| **Imbalanced-learn** | SMOTE Implementation |
| **Matplotlib & Seaborn** | Statistical Visualizations & Charts |
| **Joblib** | Serialization & Model Storage |

---

# 📊 Dataset Overview

| Property | Value |
| :--- | ---: |
| **Total Transactions** | 284,807 |
| **Features** | 30 |
| **Legitimate Transactions** | 284,315 |
| **Fraudulent Transactions** | 492 |
| **Fraud Rate** | 0.1727% |
| **Missing Values** | 0 |

> **Note:** The original dataset is excluded from the repository via `.gitignore`.

---

# 📂 Project Structure

```text
Syntecxhub_Credit_Card_Fraud_Detection/
│
├── data/
│   └── creditcard.csv
│
├── images/
│   ├── amount_boxplot.png
│   ├── amount_distribution.png
│   ├── class_distribution.png
│   ├── feature_importance.png
│   ├── final_confusion_matrix.png
│   ├── precision_recall_curve_comparison.png
│   ├── roc_curve_comparison.png
│   ├── threshold_analysis.png
│   └── time_distribution.png
│
├── models/
│   ├── baseline_random_forest.joblib
│   ├── smote_random_forest.joblib
│   ├── scaler.joblib
│   ├── final_fraud_detector.joblib
│   └── final_scaler.joblib
│
├── reports/
│   ├── feature_importance.csv
│   ├── final_evaluation_report.txt
│   ├── model_comparison.csv
│   └── threshold_analysis.csv
│
├── src/
│   ├── data_inspection.py
│   ├── eda.py
│   ├── model_preparation.py
│   ├── evaluation.py
│   ├── threshold_analysis.py
│   ├── feature_importance.py
│   ├── final_model.py
│   └── final_report.py
│
├── .gitignore
├── requirements.txt
└── README.md
```
---

# 🚀 Installation & Setup

Clone the repository

git clone https://github.com/aleezabashir-IT/Syntecxhub_Credit_Card_Fraud_Detection.git

Go to the project folder

cd Syntecxhub_Credit_Card_Fraud_Detection

Create a virtual environment

python -m venv venv

Activate the virtual environment

*   **Windows:**
    .\venv\Scripts\Activate.ps1
*   **Linux/macOS:**
    source venv/bin/activate

Install dependencies

pip install -r requirements.txt

Place the dataset

> Download `creditcard.csv` and save it to the `data/` directory.

Run the execution pipeline

python src/data_inspection.py
python src/eda.py
python src/model_preparation.py
python src/evaluation.py
python src/threshold_analysis.py
python src/feature_importance.py
python src/final_model.py
python src/final_report.py

---

# 🔄 ML Pipeline Workflow

```text
Raw Credit Card Transactions
       │
       ▼
Data Inspection & EDA
       │
       ▼
Feature / Target Separation
       │
       ▼
Stratified Train-Test Split (80% / 20%)
       │
       ▼
Feature Scaling (StandardScaler)
       │
       ▼
Class Balancing via SMOTE (Training Set Only)
       │
       ▼
Model Training (Baseline RF vs SMOTE RF)
       │
       ▼
Model Evaluation (Precision, Recall, F1, ROC-AUC, PR-AUC)
       │
       ▼
Probability Threshold Optimization (0.80)
       │
       ▼
Final Fraud Detector Saved (Models, Reports & Visualizations)
```

---

# 📈 Performance & Results

### Model Comparison

| Metric | Baseline Random Forest | SMOTE Random Forest (t=0.50) | SMOTE Random Forest (t=0.80) |
| :--- | ---: | ---: | ---: |
| **Precision** | 0.9412 | 0.8617 | **0.9740** |
| **Recall** | 0.8163 | **0.8265** | 0.7653 |
| **F1-Score** | 0.8743 | 0.8438 | **0.8571** |
| **ROC-AUC** | 0.9630 | **0.9685** | **0.9685** |

### Confusion Matrix Comparison

*   **Baseline (Default Threshold):**
    [[56859     5]
     [   18    80]]

*   **SMOTE Model (Threshold = 0.80 Optimized):**
    [[56862     2]
     [   23    75]]

---

# 📸 Visual Analytics

### Threshold Analysis & Trade-offs
Threshold optimization identified **0.80** as the optimal cutoff to maximize precision (97.40%) while controlling false alerts.

---

# 🔮 Future Enhancements

- ⚙️ **Hyperparameter Optimization:** Automated tuning via Optuna or GridSearchCV.
- ⚡ **Model Scaling:** Benchmark performance against XGBoost and LightGBM models.
- 💰 **Cost-Sensitive Learning:** Integrate business loss functions direct into objective metrics.
- 🌐 **Deployment:** Serve real-time predictions via a FastAPI REST endpoint and Streamlit UI.
- 📊 **Drift Detection:** Monitor model degradation and feature distribution shifts over time.

---

# 👩‍💻 Developer

**Aleeza Bashir**

BS Information Technology (BSIT) | Machine Learning Enthusiast

Developed as part of the **Syntecxhub Machine Learning Internship**.

---

# 🌐 Connect With Me

*   **GitHub:** [github.com/aleezabashir-IT](https://github.com/aleezabashir-IT)
*   **LinkedIn:** [linkedin.com/in/aleeza-bashir-33469a419](https://www.linkedin.com/in/aleeza-bashir-33469a419)
*   **Email:** aleezabashir301@gmail.com

---

# 🙏 Acknowledgements

Special thanks to the team at **Syntecxhub** for providing the platform and structure for this machine learning internship project.

---

<div align="center">

## ⭐ Thank you for visiting this repository!

</div>
