# Credit Card Fraud Detection Using Machine Learning in R

1. PROJECT OVERVIEW
-------------------
This project demonstrates the use of machine learning algorithms to detect fraudulent
credit card transactions. The dataset used is the well-known "creditcard.csv"
dataset from Kaggle, which contains real-world credit card transactions labeled
as legitimate (0) or fraudulent (1).

The goal is to build, evaluate, and compare several classification models for
fraud detection — an imbalanced binary classification problem.

Models included:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. XGBoost
  5. Neural Network
  6. LightGBM

The project also applies the SMOTE (Synthetic Minority Oversampling Technique)
to balance the dataset before training.

---------------------------------------------
2. DATASET INFORMATION
---------------------------------------------
Dataset Name: Credit Card Fraud Detection Dataset  
Source: Kaggle  
Link: https://www.kaggle.com/mlg-ulb/creditcardfraud  

Description:
The dataset contains transactions made by European cardholders in September 2013.
It has 284,807 transactions, out of which only 492 are fraudulent (approximately 0.17%).
Each transaction includes 28 principal components (V1–V28), along with 'Time',
'Amount', and 'Class' (target variable: 0 = legitimate, 1 = fraud).

Download Instructions:
1. Go to the Kaggle dataset page linked above.
2. Sign in with your Kaggle account (or create one if you don't have it).
3. Click “Download” to get the file `creditcard.csv`.
4. Place `creditcard.csv` in the same folder as your R script before running it.

Note:
The dataset file (`creditcard.csv`) is approximately **150 MB**, which is too large
to include directly in this project folder or repository. 
Please download it manually from Kaggle using the link above.

---------------------------------------------
3. FILES
---------------------------------------------
- FraudDetection.R                  →  Main R script with full workflow
- creditcard.csv                    →  Dataset (downloadable from Kaggle, must be in the same folder)
- README.txt                        →  Project description and instructions
- best_model_rf.rds                 →  Saved Random Forest model (after running)

---------------------------------------------
4. REQUIREMENTS
---------------------------------------------
R version: 4.1 or higher
Required Packages:
  data.table, dplyr, ggplot2, caret, smotefamily, randomForest,
  xgboost, pROC, rpart, rpart.plot, nnet, lightgbm

The script will automatically install any missing packages.

---------------------------------------------
5. HOW TO RUN
---------------------------------------------
1. Open RStudio or your preferred R environment.
2. Place the following files in the same working directory:
     - credit_card_fraud_detection.R
     - creditcard.csv
3. Run the script line-by-line, or select “Source” to execute all at once.
4. The script will:
     - Load and preprocess the dataset
     - Apply SMOTE to handle class imbalance
     - Train six different machine learning models
     - Evaluate all models using AUC, Accuracy, Precision, Recall, and F1-score
     - Plot model comparison results
     - Save the best-performing model (Random Forest) as best_model_rf.rds

---------------------------------------------
6. OUTPUT
---------------------------------------------
Console:
  - Model training progress
  - Evaluation metrics for each model
  - A summary of AUC, Accuracy, Precision, Recall, and F1-Score

Plot:
  - A bar chart comparing AUC scores for all models

File:
  - best_model_rf.rds (saved model file for reuse)

---------------------------------------------
7. NOTES
---------------------------------------------
- Ensure the dataset "creditcard.csv" is present in the same folder as the script.
- The SMOTE step may take a few minutes depending on system performance.

---------------------------------------------
8. REFERENCES
---------------------------------------------
Dataset:
  - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Libraries:
  - caret: Model training and evaluation
  - smotefamily: Oversampling for imbalanced data
  - randomForest, xgboost, lightgbm: Ensemble models
  - pROC: AUC computation and ROC curves

---------------------------------------------
9. CONTACT
---------------------------------------------
Author: Triston Aloyssius Marta
Email: tristonmarta@yahoo.com.sg
---------------------------------------------
