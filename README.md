# Fraud Detection using Machine Learning

An end-to-end fraud detection project that compares classical statistical methods, supervised machine learning models and neural-network approaches on a synthetic financial transactions dataset.

---

## Overview

Fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a small fraction of all observations. This project develops a complete analytical pipeline, from data preprocessing and exploratory analysis to predictive modelling and performance evaluation.

Both classical statistical models and machine learning techniques are implemented and compared to assess their effectiveness in identifying fraudulent transactions.

---

## Objectives

- Build a complete fraud detection pipeline using Python.
- Compare multiple modelling approaches for fraud detection.
- Evaluate model performance using metrics appropriate for imbalanced datasets.
- Interpret the main fraud indicators and compare modelling strategies.

---

## Dataset

The analysis uses a synthetic financial transactions dataset containing:

- **12,000 transactions**
- **24 variables**
- **Fraud rate:** 2.13%

The dataset includes:

- Customer characteristics
- Transaction attributes
- Behavioural indicators
- Binary fraud label

---

## Methodology

The project follows the workflow below:

1. Data preprocessing
2. Missing value imputation
3. Feature engineering
4. Exploratory Data Analysis (EDA)
5. Linear Regression
6. Logistic Regression
7. Neural Network (MLP)
8. Autoencoder-based anomaly detection
9. Model evaluation and comparison

---

## Models

The following models were implemented:

- Linear Regression
- Logistic Regression
- Multilayer Perceptron (MLP)
- Autoencoder

---

## Evaluation Metrics

Model performance was assessed using:

- Precision
- Recall
- F1-score
- ROC-AUC

Special emphasis was placed on metrics suitable for highly imbalanced classification problems.

---

## Results

The analysis identified several important fraud indicators, including:

- Risk Score
- Amount Ratio
- Foreign Transactions
- Night Transactions
- High-Risk Merchants

The supervised models achieved excellent predictive performance on the synthetic dataset, while the autoencoder demonstrated the use of unsupervised anomaly detection techniques for fraud identification.

---

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

---

## Repository Structure

```
fraud-detection-machine-learning/
│
├── data/
├── notebooks/
├── src/
├── results/
├── report/
└── README.md
```

---

## Repository Contents

| Folder | Description |
|----------|-------------|
| data | Synthetic transaction dataset |
| notebooks | Jupyter notebooks used during development |
| src | Python implementation |
| results | Figures and model outputs |
| report | Final project report |

---

## Author

**Konstantinos Papavrontos**

MSc Business Economics with Analytics  
Athens University of Economics and Business
