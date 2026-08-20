# 🛡️ SentinelFraud ML

<div align="center">

## Credit Card Fraud Detection & Risk Analysis System

**An imbalance-aware machine learning pipeline for detecting suspicious financial transactions.**

SentinelFraud ML applies exploratory analysis, Random Forest classification, SMOTE-based
imbalance handling, probability threshold optimization, and multi-metric evaluation
to build a practical fraud detection workflow.

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

Credit card fraud detection is an imbalanced binary classification problem in which
fraudulent transactions represent only a very small portion of the dataset.

This project develops an end-to-end machine learning workflow designed to evaluate
fraud detection performance beyond simple accuracy.

The pipeline covers:

**Data Inspection → Exploratory Analysis → Preprocessing → Baseline Model → SMOTE → Model Evaluation → Threshold Optimization → Final Model**

The project focuses on:

- Extreme class imbalance handling
- Training-only SMOTE resampling
- Random Forest classification
- Probability threshold optimization
- Precision, Recall and F1-score analysis
- ROC-AUC and Average Precision evaluation
- Confusion matrix analysis
- Feature importance analysis
- Model and report persistence

---

# ✨ Key Features

- 🎯 **Imbalance-Aware Classification**
- ⚖️ **SMOTE-Based Training Data Balancing**
- 🌲 **Random Forest Fraud Classification**
- 🎚️ **Probability Threshold Optimization**
- 📊 **ROC & Precision-Recall Analysis**
- 🔍 **Feature Importance Analysis**
- 🧮 **Confusion Matrix Evaluation**
- 💾 **Persistent Models with Joblib**
- 📋 **Automated Evaluation Reports**

---

# 🛠️ Technologies Used

| Technology               | Purpose                               |
| ------------------------ | ------------------------------------- |
| **Python**               | Core programming language             |
| **Pandas & NumPy**       | Data processing and analysis          |
| **Scikit-learn**         | Scaling, Random Forest and evaluation |
| **Imbalanced-learn**     | SMOTE implementation                  |
| **Matplotlib & Seaborn** | Data visualization                    |
| **Joblib**               | Model and scaler persistence          |

---

# 📊 Dataset Overview

The project uses a credit card transaction dataset containing **284,807 transactions**
and **30 input features**.

| Property                |   Value |
| ----------------------- | ------: |
| Total Transactions      | 284,807 |
| Features                |      30 |
| Legitimate Transactions | 284,315 |
| Fraudulent Transactions |     492 |
| Fraud Rate              | 0.1727% |
| Missing Values          |       0 |

The dataset is highly imbalanced, making precision, recall, F1-score and
ranking-based metrics more informative than accuracy alone.

> **Dataset Note:** The original `creditcard.csv` dataset is not intended to be
> committed to the repository and is excluded through `.gitignore`.

---

# 📂 Project Structure

````text
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

---

# 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aleezabashir-IT/Syntecxhub_Credit_Card_Fraud_Detection.git
````

### 2. Navigate to the Project

```bash
cd Syntecxhub_Credit_Card_Fraud_Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Add the Dataset

Place the dataset at:

```text
data/creditcard.csv
```

> The original dataset is intentionally excluded from version control through `.gitignore`.

---

# ▶️ Run the Project

Execute the scripts in the following order:

```bash
python src/data_inspection.py
python src/eda.py
python src/model_preparation.py
python src/evaluation.py
python src/threshold_analysis.py
python src/feature_importance.py
python src/final_model.py
python src/final_report.py
```

Each stage builds on the output of the previous stage, creating a reproducible end-to-end fraud detection workflow.

---

# 🔄 ML Pipeline Workflow

Raw Transaction Dataset
       │
       ▼
Stratified Train-Test Split (80/20)
       │
       ▼
StandardScaler Transformation
       │
       ▼
SMOTE Resampling (Training Set Only)
       │
       ▼
Random Forest Classifier Training
       │
       ▼
Probability Threshold Optimization (0.80)
       │
       ▼
Final Model & Metrics Generation

---

# 📈 Performance & Results

The models were evaluated using metrics that are more informative for highly imbalanced fraud detection than accuracy alone.

### Model Comparison

| Metric                | Baseline Random Forest | SMOTE Random Forest<br>Threshold 0.50 | SMOTE Random Forest<br>Threshold 0.80 |
| --------------------- | ---------------------: | ------------------------------------: | ------------------------------------: |
| **Precision**         |                 0.9412 |                                0.8617 |                            **0.9740** |
| **Recall**            |             **0.8163** |                            **0.8265** |                                0.7653 |
| **F1-Score**          |             **0.8743** |                                0.8438 |                                0.8571 |
| **ROC-AUC**           |                 0.9630 |                            **0.9685** |                            **0.9685** |
| **Average Precision** |                 0.8734 |                                0.8723 |                                0.8723 |

### Key Evaluation Result

The SMOTE model achieved a slightly higher ROC-AUC than the baseline model:

- **Baseline ROC-AUC:** 0.9630
- **SMOTE ROC-AUC:** 0.9685

Average Precision remained very similar:

- **Baseline AP:** 0.8734
- **SMOTE AP:** 0.8723

This demonstrates that SMOTE improved ranking performance while threshold selection was used to adjust the precision-recall trade-off.

---

# 🎚️ Threshold Optimization

Instead of relying only on the conventional `0.50` probability threshold, multiple thresholds were evaluated to identify a more suitable operating point.

### Selected Threshold

**Optimized Threshold: `0.80`**

| Metric        |      Score |
| ------------- | ---------: |
| **Precision** | **0.9740** |
| **Recall**    |     0.7653 |
| **F1-Score**  | **0.8571** |

At the selected threshold, the model prioritizes **higher precision**, reducing false-positive alerts while retaining a substantial proportion of fraudulent transactions.

> Threshold selection is application-dependent. A different operational environment may prefer a lower threshold to prioritize recall over precision.

---

# 🧮 Confusion Matrix

### Baseline Random Forest

```text
[[56859     5]
 [   18    80]]
```

### SMOTE Random Forest at Threshold 0.80

```text
[[56862     2]
 [   23    75]]
```

The optimized SMOTE model reduced false positives from **5 to 2**, while the number of correctly identified fraudulent transactions changed from **80 to 75**.

This illustrates the practical precision-recall trade-off introduced by threshold optimization.

---

# 📊 Visual Analytics

The project generates several visualizations to support model and data analysis.

### Exploratory Analysis

- `images/class_distribution.png`
- `images/amount_distribution.png`
- `images/amount_boxplot.png`
- `images/time_distribution.png`

### Model Analysis

- `images/roc_curve_comparison.png`
- `images/precision_recall_curve_comparison.png`
- `images/threshold_analysis.png`
- `images/feature_importance.png`
- `images/final_confusion_matrix.png`

These visualizations provide insight into class imbalance, transaction behavior, model discrimination, threshold trade-offs, and feature contribution.

---

# 📁 Generated Reports

The pipeline produces structured outputs inside the `reports/` directory:

| Report                        | Purpose                                    |
| ----------------------------- | ------------------------------------------ |
| `feature_importance.csv`      | Ranked model features                      |
| `model_comparison.csv`        | Baseline and SMOTE performance             |
| `threshold_analysis.csv`      | Precision, recall and F1 across thresholds |
| `final_evaluation_report.txt` | Final model evaluation summary             |

---

# 💾 Model Artifacts

Trained models and preprocessing objects are stored in the `models/` directory:

- `baseline_random_forest.joblib`
- `smote_random_forest.joblib`
- `scaler.joblib`
- `final_fraud_detector.joblib`
- `final_scaler.joblib`

These artifacts allow the trained pipeline components to be reused without retraining from scratch.

---

# 🔮 Future Enhancements

- ⚙️ **Hyperparameter Optimization** using GridSearchCV or Optuna
- ⚡ **Advanced Model Benchmarking** with XGBoost and LightGBM
- 💰 **Cost-Sensitive Learning** based on fraud detection business costs
- 🌐 **API Deployment** using FastAPI
- 🖥️ **Interactive Interface** using Streamlit
- 📊 **Model Drift Monitoring** for changing transaction patterns
- 🔐 **Real-Time Fraud Risk Scoring**
- 📈 **Business-Oriented Evaluation** using transaction-level cost analysis

---

# 👩‍💻 Developer

**Aleeza Bashir**

_BS Information Technology | Lahore Garrison University_

Machine Learning enthusiast focused on developing practical, data-driven solutions using Python and modern machine learning techniques.

Developed as part of the **Syntecxhub Machine Learning Internship**.

---

# 🌐 Connect

- 💻 **GitHub:** [aleezabashir-IT](https://github.com/aleezabashir-IT)
- 🔗 **LinkedIn:** [Aleeza Bashir](https://www.linkedin.com/in/aleeza-bashir-33469a419)
- 📧 **Email:** aleezabashir301@gmail.com

---

# 🙏 Acknowledgements

This project was developed during the **Syntecxhub Machine Learning Internship**.

Special thanks to **Syntecxhub** for providing the learning environment and project structure that supported the practical application of machine learning, imbalanced-data handling, model evaluation, and fraud detection techniques.

---

<div align="center">

### 🛡️ SentinelFraud ML

**Detecting the unusual. Measuring the risk.**

⭐ If you find this project useful, consider giving the repository a star.

</div>
