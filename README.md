# FraudScope: Transaction Fraud Detection & Analysis
FraudScope is a complete machine learning pipeline for detecting fraudulent financial transactions using only classical ML techniques and core concepts. Designed as a hands-on, transparent system for fraud analytics, it guides users through every stage—from raw data to model evaluation—using only scikit-learn and core Python libraries.

# Dataset & Problem Statement
Dataset Source: MoMTSim_20240722202413_1000_dataset.csv (synthetic transactional data)

Objective: Build a classifier to detect fraudulent transactions (isFraud)

Features: Includes transaction metadata such as:

amount, oldBalInitiator, newBalInitiator

transactionType, origin, destination, etc.

# Architecture Overview
A fully modular pipeline:


Raw CSV Data
   ↓
Exploratory Data Analysis (EDA)
   ↓
Handling Missing Values & Duplicates
   ↓
Outlier Detection (IQR & Z-score methods)
   ↓
Encoding & Feature Engineering
   ↓
Feature Scaling (MinMaxScaler, StandardScaler)
   ↓
Feature Selection (Correlation Matrix Filtering)
   ↓
Train/Test Split (80/20, stratified on 'isFraud')
   ↓
Model Training & Hyperparameter Tuning:
   - K-Nearest Neighbors (varied k, distance metrics)
   - Support Vector Machine (kernel, C tuning)
   - Decision Tree (depth, min_samples_split)
   - Logistic Regression (L1 & L2 regularization)
   ↓
Evaluation:
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix & Classification Report
   - Training & Inference Time Comparison
     
# Setup & Requirements
Install dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
Required libraries:

pandas, numpy

matplotlib, seaborn

scikit-learn

 # Notebook Walkthrough
fraudscope.ipynb:

EDA: Value counts, missing data, class imbalance

Visualizations: Boxplots, heatmaps, pairplots

Outlier Handling: IQR and Z-score thresholds

Feature Engineering: Encoding transaction types, calculating balance deltas

Scaling: Compared MinMax and Z-score scaling effects

Model Training: Manual loop for each model with grid-tuned hyperparameters

Evaluation: Comparative analysis of model metrics and runtime

# Results & Model Comparisons
Model	F1-Score	Accuracy	Inference Time
Logistic Regression	0.92	0.94	⚡ Fast
KNN (k=5, Euclidean)	0.88	0.91	🐢 Slow
Decision Tree	0.93	0.95	⚡ Fast
SVM (RBF)	0.94	0.96	⚖️ Moderate

Best performer: SVM (RBF kernel) in terms of F1 and overall accuracy

Most efficient: Logistic Regression (fastest inference)

# Usage & Reproducibility
To run the notebook:

Place the dataset (MoMTSim_20240722202413_1000_dataset.csv) in the root directory.

Open and run fraudscope_analysis.ipynb in Jupyter or Google Colab.

Follow the cell-by-cell analysis and observe output metrics.

# Contributing
Feel free to fork this repo and submit a pull request if you'd like to:

Add new models (e.g., ensemble methods)

Improve feature engineering

Visualize confusion matrices interactively

# License & Academic Integrity
This project is for educational and demonstrative purposes only. Do not reuse directly in academic submissions unless explicitly permitted.

# Developed by
Absar Raashid
GitHub Repository
Open for feedback and collaboration.

