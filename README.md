# Predict Employee Attrition

## Description

This project builds a machine learning model to predict whether an employee will leave the company (Attrition) or stay.

The objective is to help HR identify employees at risk of leaving and understand the key drivers behind attrition.

---

## Model Overview

- Algorithm: Random Forest Classifier
- Imbalance Handling: SMOTE (Synthetic Minority Oversampling)
- Threshold Tuning: Optimized to improve precision / recall tradeoff
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score

---

## Key Improvements

The dataset was highly imbalanced (fewer employees leave than stay).

To improve model performance:

- Applied SMOTE to balance the training dataset
- Used stratified train/test split
- Tuned prediction threshold to improve minority class performance
- Evaluated using precision and recall instead of accuracy only

---

## Notebook Version

A Jupyter Notebook version of the project is included:

`employee_attrition_model.ipynb`

The notebook contains:

- Exploratory Data Analysis (EDA)
- Class imbalance visualization
- SMOTE application
- Threshold tuning
- Confusion matrix
- Feature importance analysis
- Final model evaluation

---

# How to Run (just run the Jupyter file)

1. Open the notebook in Google Colab or Jupyter.
2. Upload the dataset file:
   `WA_Fn-UseC_-HR-Employee-Attrition.csv`
3. Run all cells.

---

## Business Insight

The model identifies important factors influencing employee attrition, including:

- StockOptionLevel
- MonthlyIncome
- JobSatisfaction
- JobInvolvement
- Age
- YearsAtCompany

These insights can help HR teams design better retention strategies.


