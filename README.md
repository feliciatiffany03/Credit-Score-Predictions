# Credit Score Prediction

An end-to-end Machine Learning project for predicting customer Credit Score
using customer financial and payment behavior data.

This project was developed as an academic project and covers the complete
machine learning workflow, from data exploration and preprocessing to model
training, evaluation, experiment tracking, and cloud deployment.

---

## Project Overview

Credit Score classification is a multiclass classification problem with three
target categories:

- Good
- Standard
- Poor

The project aims to build a machine learning model that can classify a
customer's credit score based on financial information, credit history,
payment behavior, and other customer attributes.

---

## Machine Learning Workflow

The project follows the following workflow:

1. Data Understanding
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
4. Train-Validation-Test Split
5. Data Preprocessing
6. Model Training
7. Model Comparison
8. Model Evaluation
9. MLflow Experiment Tracking
10. Model Deployment
11. Streamlit Prediction Interface

---

## Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Handling invalid data
- Numerical feature imputation
- Categorical feature imputation
- Categorical encoding using One-Hot Encoding
- Numerical feature scaling using StandardScaler
- Label encoding for the target variable

The data was split into training, validation, and testing sets before
preprocessing to reduce the risk of data leakage.

---

## Models

Four machine learning algorithms were evaluated in the notebook:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### Model Comparison

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Random Forest | 72.12% | 70.41% | 70.16% | 70.24% |
| Gradient Boosting | 69.84% | 68.34% | 67.33% | 67.65% |
| Logistic Regression | 68.76% | 67.19% | 64.61% | 65.67% |
| Decision Tree | 66.56% | 64.12% | 63.86% | 63.98% |

The model comparison was performed using Accuracy, Macro Precision,
Macro Recall, and Macro F1 Score.

---

## MLflow

MLflow was used for experiment tracking and model evaluation.

The training pipeline records model performance and supports the selection
of the final model based on validation performance.

---

## AWS Deployment

The project was also developed with a cloud deployment workflow using:

- AWS SageMaker
- AWS SageMaker Endpoint
- Streamlit
- Boto3

The Streamlit application sends customer information to an AWS SageMaker
endpoint and displays the predicted Credit Score.

The deployment application was configured to use the SageMaker endpoint:

`credit-score-endpoint`

Region:

`us-east-1`

---

## Deployment Status

> **Note:** The live AWS demo is currently unavailable.

The project was previously deployed using the AWS Academy Learner Lab
environment. The Learner Lab currently shows usage exceeding its displayed
cloud budget (`$53.2 of $50`), and the lab environment is currently inactive.
As a result, the previously deployed AWS resources cannot currently be
accessed for a live demonstration.

The AWS deployment source code is still included in this repository to
document the deployment workflow and implementation.

---

## Repository Structure

```text
credit-score-prediction/
│
├── README.md
│
├── notebook/
│   └── modeluas_3-2_REVISI.ipynb
│
├── local_pipeline/
│   ├── data_ingestion.py
│   ├── train.py
│   ├── evaluation.py
│   ├── pipeline.py
│   └── inferencing.py
│
├── aws_deployment/
│   ├── train.py
│   ├── inference.py
│   ├── pipeline.py
│   ├── app_streamlit.py
│   └── deploy_endpoint.ipynb
│
├── screenshots/
│   ├── streamlit_demo.png
│   ├── aws_sagemaker.png
│   └── model_evaluation.png
│
└── requirements.txt
