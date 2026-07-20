# 🔬 Clinical-Grade Breast Cancer Detection Framework
### *An End-to-End, Leak-Free Machine Learning Diagnostics Engine*

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ML Engine](https://img.shields.io/badge/ml--engine-scikit--learn%20%7C%20xgboost-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Project Overview
This repository contains a production-ready, modular machine learning architecture designed to classify Fine-Needle Aspirate (FNA) breast tissue biopsy profiles into **Malignant** or **Benign** categories. 

Most introductory ML implementations suffer from structural **Data Leakage** by scaling datasets globally before partitioning validation folds. This framework implements a rigid MLOps boundary by wrapping feature transformers (`StandardScaler`) and estimators natively within scikit-learn `Pipeline` objects. This guarantees complete mathematical isolation across all cross-validation blocks.

### Key Highlights
* **Leak-Free Transformation Pipeline:** Zero data leakage achieved via custom encapsulation pipelines.
* **Exhaustive Hyperparameter Tuning:** Automated optimization across 8 distinct algorithms via Stratified 5-Fold Grid Search (`GridSearchCV`).
* **Clinical Optimization Matrix:** Evaluation metrics are prioritized for **Recall (Sensitivity)** and **F1-Score** to deliberately minimize false negatives in diagnostic inferences.
* **Production Deployment:** Includes a lightweight, interactive doctor-facing Streamlit dashboard for real-time risk assessment.

---

## 📂 System Architecture & Folder Layout

The project follows a decoupled, clean-code directory structure separating core computation pipelines from deployment layers:

```text
breast_cancer_detection/
│
├── data/
│   └── data.csv                # Wisconsin Breast Cancer Dataset (FNA Metrics)
│
├── models/                     # Auto-generated target destination for artifacts
│   ├── champion_breast_cancer_model.joblib   # Serialized peak inference pipeline
│   ├── model_comparison.csv                 # Detailed algorithm metric index
│   └── model_comparison_chart.png           # Comparative performance visuals
│
├── src/                        # Encapsulated application core modules
│   ├── __init__.py
│   ├── preprocessing.py        # Isolated data loading and train/test partition splits
│   ├── model.py                # Hyperparameter tuning search spaces & grids
│   ├── train_model.py          # Training loop execution, grid tuning, and scoring
│   └── predict.py              # Low-latency inference wrapper class
│
├── app.py                      # Interactive front-end Streamlit web panel
├── notebook.ipynb              # Visual Exploratory Data Analysis (EDA) dashboard
├── requirements.txt            # Explicitly pinned third-party dependencies
└── README.md                   # Technical operation documentation