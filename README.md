# Predict Employee Attrition

## Description

This project builds a machine learning model to predict whether an employee will stay or leave the company.

The objective is to help HR understand the key factors that drive employee attrition.

---

## Model

- Algorithm: Random Forest Classifier
- Task: Binary Classification (Stay / Leave)
- Accuracy: ~87%

---

## Key Features Influencing Attrition

- MonthlyIncome
- OverTime
- Age
- TotalWorkingYears
- DistanceFromHome
- YearsAtCompany

These features were identified using feature importance analysis.

---

## How to Run

1. Install dependencies:

pip install -r requirements.txt


2. Place the dataset file in the project folder:

WA_Fn-UseC_-HR-Employee-Attrition.csv


3. Run:

python model.py


The script will train the model and generate a feature importance plot.
