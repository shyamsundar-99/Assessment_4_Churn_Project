# Customer Churn Prediction for a Retail Loyalty Program

## 📌 Project Overview

This project develops a **Customer Churn Prediction System** for a retail loyalty program using machine learning.

The objective is to identify customers who are likely to stop making purchases in the future, allowing businesses to take proactive retention actions.

The project follows a complete machine learning workflow:

* Data cleaning and exploration
* Customer-level feature engineering
* Temporal and leakage-aware validation
* Machine learning model development
* Model comparison
* Explainability using SHAP
* Customer segmentation and cohort analysis
* Ablation analysis
* Retention scenario simulation
* Final churn prediction generation

---

## 🎯 Objective

The primary objective is to predict whether an eligible customer will make **zero purchases during the April–June 2024 period**.

The model is designed as an **early-warning retention system**, with particular emphasis on customers who are still active at the prediction date but show signs of potential future churn.

---

## 🗂️ Project Structure

```text
churn_project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   └── evaluation.py
│
├── outputs/
│   ├── charts/
│   ├── tables/
│   └── predictions/
│
├── requirements.txt
├── README.md
└── .gitignore
```

> Folder names may vary slightly depending on the final repository structure.

---

## 📊 Dataset

The project uses three major data sources:

### 1. Customer Profile

Contains customer-level demographic and profile information.

Examples include:

* Customer ID
* City
* Tier
* Demographic attributes
* Customer cohort information

### 2. Monthly Activity

Contains customer purchasing and engagement activity over time.

Features include information related to:

* Purchase frequency
* Purchase value
* Recency
* Spending trends
* Engagement
* Customer support activity

### 3. Churn Labels

The target variable identifies whether a customer makes **zero purchases during the future prediction window**.

---

# ⚙️ Methodology

## 1. Data Loading

The project loads the raw datasets using reusable Python modules.

```python
from src import config as C
from src import data as D
from src import features as F

prof_raw, act_raw, lab_raw = D.load_raw()

print({
    'profile': prof_raw.shape,
    'activity': act_raw.shape,
    'label': lab_raw.shape
})
```

---

## 2. Data Cleaning

The data preparation stage includes:

* Missing-value analysis
* Duplicate detection
* Data-type validation
* Customer ID validation
* Date validation
* Activity consistency checks
* Handling of missing customer-month records

A key assumption is that missing monthly activity represents zero activity. This assumption is documented as a data-quality limitation and should be validated against the source system.

---

# 🔧 Feature Engineering

Customer-level features are created from historical activity without using future information.

Important feature groups include:

### Purchase Behavior

* Purchase frequency
* Total purchases
* Average purchase value
* Total spending
* Average spending

### Recency

* Days since last purchase
* Recent purchase indicators
* Purchase inactivity periods

### Spending Trends

* Recent spending
* Historical spending
* Spending change
* Purchase trend

### Engagement

* Customer engagement indicators
* Activity frequency
* Recent engagement

### Support

* Customer support activity
* Support frequency

### Demographics

* Customer tier
* City
* Cohort
* Other available profile attributes

---

# 🤖 Machine Learning Models

The project compares multiple approaches.

## Rule-Based Baseline

A simple business-rule model is used as a baseline.

This provides a reference point for evaluating whether machine learning provides meaningful improvement.

## Logistic Regression

A Logistic Regression model provides a strong and interpretable linear baseline.

## LightGBM

LightGBM is used as the primary tree-based machine learning model.

It is particularly useful for this problem because it can capture nonlinear relationships between customer behavior and churn.

## Balanced LightGBM

A class-balanced LightGBM model is also evaluated to understand the effect of class balancing on churn prediction.

---

# 📈 Model Evaluation

The models are evaluated using metrics suitable for an imbalanced churn prediction problem.

### Metrics

* Precision
* Recall
* F1-score
* PR-AUC

PR-AUC is especially important because the objective is to identify potential churners within the customer population.

---

# 🕒 Temporal Validation

The project uses a time-ordered validation strategy instead of a random train-test split.

Historical snapshots are used for training, while a later snapshot is used for testing.

Example:

```text
Training Snapshots
        ↓
June 2023
September 2023
December 2023
        ↓
March 2024
        ↓
Future Churn
April–June 2024
```

This better represents how the model would operate in a real business environment.

---

# 👥 Active Customer Evaluation

A major focus of the project is distinguishing between:

### All Eligible Customers

This population can include customers who had already stopped purchasing before the prediction date.

### Active Customers

These are customers who were still purchasing near the prediction date.

The active-customer evaluation is more representative of an **early-warning retention system**, because the objective is to identify customers before they fully lapse.

Therefore, active-customer performance is emphasized when interpreting the model's business value.

---

# 🏆 Model Results

### Overall Evaluation

| Model               |    PR-AUC |    Recall |  F1-Score |
| ------------------- | --------: | --------: | --------: |
| Rule-Based Baseline |     0.945 |     0.798 |     0.885 |
| Logistic Regression |     0.988 |     0.927 |     0.945 |
| LightGBM            | **0.988** | **0.929** | **0.951** |
| Balanced LightGBM   |     0.988 |     0.926 |     0.948 |

LightGBM achieves the highest F1-score among the evaluated models.

### Active Customer Evaluation

For the more business-relevant active-customer population:

| Metric   |  LightGBM |
| -------- | --------: |
| PR-AUC   | **0.806** |
| Recall   | **0.720** |
| F1-Score | **0.735** |

The active-customer results are emphasized because they provide a more realistic estimate of the model's ability to identify customers who may churn before they completely disengage.

---

# 🔍 Explainability

SHAP (SHapley Additive exPlanations) is used to understand how different features influence churn predictions.

The analysis helps answer questions such as:

* Which customer behaviors are strongest indicators of churn?
* Does purchase recency influence churn probability?
* How does spending behavior affect predictions?
* Which engagement signals are important?
* Which customer characteristics contribute to churn risk?

This improves the interpretability of the machine learning model and supports business decision-making.

---

# 🧪 Ablation Study

An ablation analysis is performed to understand the contribution of different feature groups.

Feature groups can be removed or modified to determine how much they contribute to predictive performance.

This helps identify the most valuable sources of information for churn prediction.

---

# 👤 Customer Segmentation

The project also analyzes churn behavior across different customer groups.

Analysis includes:

* Customer tier
* City
* Cohort
* Customer activity levels
* Spending behavior

This allows the business to identify segments with higher churn risk and potentially design targeted retention strategies.

---

# 💼 Retention Scenario Analysis

A business retention scenario is included to estimate the potential impact of intervention.

For example:

```text
Identify high-risk customers
            ↓
Apply retention intervention
            ↓
Assume a reduction in churn probability
            ↓
Estimate potential retained customers
```

The retention analysis is a **scenario simulation**, not a causal experiment.

Therefore, simulated reductions in churn should not be interpreted as proven real-world intervention effects.

---

# 🚨 Important Data & Modeling Considerations

## Missing Activity Records

Missing customer-month records are treated as zero activity.

This assumption may be appropriate if the source system removes inactive customers from monthly records.

However, if missing records represent missing data rather than inactivity, this could introduce bias.

---

## Customer Overlap Across Snapshots

Customers can appear in multiple temporal snapshots.

The project therefore includes additional robustness analysis using customer-disjoint evaluation where appropriate.

---

## Leakage Prevention

Features are constructed using information available before the prediction period.

Future purchasing behavior is not used as an input feature.

This is essential because using future information would artificially inflate model performance.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* LightGBM
* SHAP
* Jupyter Notebook
* Git & GitHub

---

# 📦 Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd churn_project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the EDA/modeling notebook and run the cells sequentially.

The project follows this workflow:

```text
Raw Data
   ↓
Data Validation
   ↓
Data Cleaning
   ↓
EDA
   ↓
Feature Engineering
   ↓
Temporal Validation
   ↓
Model Training
   ↓
Model Evaluation
   ↓
SHAP Explainability
   ↓
Ablation Analysis
   ↓
Customer Segmentation
   ↓
Retention Scenario
   ↓
Final Predictions
```

---

# 📁 Outputs

The project generates:

* Exploratory data analysis charts
* Feature-engineered datasets
* Model evaluation tables
* Prediction files
* SHAP visualizations
* Feature importance plots
* Customer segment analysis
* Ablation results
* Retention scenario results

---

# 💡 Business Impact

The system can help a retail loyalty program:

* Identify customers at high risk of churn
* Prioritize retention campaigns
* Understand customer behavior associated with churn
* Segment customers based on risk
* Allocate retention resources more efficiently
* Provide interpretable reasons behind churn predictions

The model is intended to support **proactive customer retention rather than simply reporting historical churn**.

---

# 🔮 Future Improvements

Potential improvements include:

* Validate missing-month assumptions using source-system documentation
* Incorporate additional customer behavioral signals
* Tune LightGBM hyperparameters
* Calibrate predicted probabilities
* Test additional gradient-boosting models
* Perform customer-disjoint temporal validation
* Develop a production prediction pipeline
* Build a dashboard for business users
* A/B test retention interventions using real customer outcomes

---

# 👨‍💻 Author

**Shyam Sundar**

B.Tech — Data Science and Business Systems

SRM Institute of Science and Technology

---

## ⭐ Project Summary

This project demonstrates an end-to-end machine learning solution for customer churn prediction, combining **feature engineering, temporal validation, LightGBM modeling, explainable AI, customer segmentation, robustness analysis, and business-oriented retention analysis**.

The primary focus is not only achieving high predictive performance, but also ensuring that the model's evaluation is **leakage-aware, interpretable, reproducible, and relevant to real-world customer retention.**
