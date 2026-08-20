# 🛡️ SentinelFraud ML

<div align="center">

## Credit Card Fraud Detection & Risk Analysis System

### An Imbalance-Aware Machine Learning Pipeline for Detecting Suspicious Financial Transactions

<p>
A complete machine learning workflow that combines exploratory data analysis,
Random Forest classification, SMOTE-based imbalance handling,
probability threshold optimization, and multi-metric evaluation
to identify potentially fraudulent credit card transactions.
</p>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)
![SMOTE](https://img.shields.io/badge/SMOTE-Imbalance%20Handling-red)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-2E8B57)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Persistence-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

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

| Technology               | Purpose                                      |
| :----------------------- | :------------------------------------------- |
| **Python**               | Core Programming Language                    |
| **Pandas & NumPy**       | Data Manipulation & Processing               |
| **Scikit-learn**         | Model Training, Scaling & Evaluation Metrics |
| **Imbalanced-learn**     | SMOTE Implementation                         |
| **Matplotlib & Seaborn** | Statistical Visualizations & Charts          |
| **Joblib**               | Serialization & Model Storage                |

---

# 📊 Dataset Overview

| Property                    |   Value |
| :-------------------------- | ------: |
| **Total Transactions**      | 284,807 |
| **Features**                |      30 |
| **Legitimate Transactions** | 284,315 |
| **Fraudulent Transactions** |     492 |
| **Fraud Rate**              | 0.1727% |
| **Missing Values**          |       0 |

> **Note:** The original dataset is excluded from the repository via `.gitignore`.

---

# 📂 Project Structure

Syntecxhub_Credit_Card_Fraud_Detection
│
├── data/
│ └── creditcard.csv
│
├── images/
│ ├── amount_boxplot.png
│ ├── amount_distribution.png
│ ├── class_distribution.png
│ ├── feature_importance.png
│ ├── final_confusion_matrix.png
│ ├── precision_recall_curve_comparison.png
│ ├── roc_curve_comparison.png
│ ├── threshold_analysis.png
│ └── time_distribution.png
│
├── models/
│ ├── baseline_random_forest.joblib
│ ├── smote_random_forest.joblib
│ ├── scaler.joblib
│ ├── final_fraud_detector.joblib
│ └── final_scaler.joblib
│
├── reports/
│ ├── feature_importance.csv
│ ├── final_evaluation_report.txt
│ ├── model_comparison.csv
│ └── threshold_analysis.csv
│
├── src/
│ ├── data_inspection.py
│ ├── eda.py
│ ├── model_preparation.py
│ ├── evaluation.py
│ ├── threshold_analysis.py
│ ├── feature_importance.py
│ ├── final_model.py
│ └── final_report.py
│
├── .gitignore
├── requirements.txt
└── README.md

---

# 🚀 Installation & Setup

Clone the repository

git clone https://github.com/aleezabashir-IT/Syntecxhub_Credit_Card_Fraud_Detection.git

Go to the project folder

cd Syntecxhub_Credit_Card_Fraud_Detection

Create a virtual environment

python -m venv venv

Activate the virtual environment

- **Windows:**
  .\venv\Scripts\Activate.ps1
- **Linux/macOS:**
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

# 🔄 ML PIPELINE FLOW

Raw Credit Card Transactions
│
▼
Data Inspection
│
▼
Exploratory Data Analysis
│
▼
Feature / Target Separation
│
▼
Stratified Train-Test Split
80% / 20%
│
▼
Feature Scaling
StandardScaler
│
▼
┌──────┴──────┐
│ │
▼ ▼
Baseline RF SMOTE
│ │
│ ▼
│ Balanced Training
│ │
│ ▼
│ SMOTE Random Forest
│ │
└──────┬──────┘
▼
Model Evaluation
│
▼
Precision • Recall • F1
ROC-AUC • PR-AUC
│
▼
Threshold Optimization
│
▼
Final Fraud Detector
│
▼
Models • Reports • Visualizations

# 📈 Performance & Results

### Model Comparison

| Metric        | Baseline Random Forest | SMOTE Random Forest (t=0.50) | SMOTE Random Forest (t=0.80) |
| :------------ | ---------------------: | ---------------------------: | ---------------------------: |
| **Precision** |                 0.9412 |                       0.8617 |                   **0.9740** |
| **Recall**    |                 0.8163 |                   **0.8265** |                       0.7653 |
| **F1-Score**  |                 0.8743 |                       0.8438 |                   **0.8571** |
| **ROC-AUC**   |                 0.9630 |                   **0.9685** |                   **0.9685** |

### Confusion Matrix Comparison

- **Baseline (Default Threshold):**
  [[56859     5]
[   18    80]]

- **SMOTE Model (Threshold = 0.80 Optimized):**
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

## 👩‍💻 Developer

**Aleeza Bashir**  
_BS Information Technology | Lahore Garrison University_

Machine Learning enthusiast focused on building practical, data-driven solutions with Python and modern ML techniques.

Developed as part of the **Syntecxhub Machine Learning Internship**.

---

## 🌐 Connect

- 💻 **GitHub:** [aleezabashir-IT](https://github.com/aleezabashir-IT)
- 🔗 **LinkedIn:** [Aleeza Bashir](https://www.linkedin.com/in/aleeza-bashir-33469a419)
- 📧 **Email:** aleezabashir301@gmail.com

---

## 🙏 Acknowledgements

Built during the **Syntecxhub Machine Learning Internship** as a practical implementation of machine learning, imbalanced-data handling, model evaluation, and fraud-risk analysis.

---

<div align="center">

### 🛡️ SentinelFraud ML

**Detecting the unusual. Measuring the risk. Learning from the data.**

⭐ _Explore the project, review the results, and feel free to connect._

</div>
