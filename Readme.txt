House Price Prediction using CRISP-DM (Ames Housing Dataset)

#--------------------------------------------------------------------------------------
Project Overview

This project predicts residential housing prices using the Ames Housing dataset from Kaggle.
The workflow follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology,
 including data understanding, preparation, feature engineering, modeling, and evaluation.
The project demonstrates best practices for modularity, reproducibility, and transparent documentation.

#--------------------------------------------------------------------------------------
Dataset Information

Dataset: House Prices – Advanced Regression Techniques
Source: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

Rows: 1460
Features: 80 (79 predictors + SalePrice)

Main variables include structural, quality, size, and neighborhood features.

#--------------------------------------------------------------------------------------
CRISP-DM PROCESS
1. Business Understanding

Goal: Build an accurate regression model to predict SalePrice and identify important predictors influencing housing values.

2. Data Understanding

Performed EDA, correlation analysis, missing value assessment, and visualization.

3. Data Preparation

• Missing-value imputation
• Scaling, normalization
• Feature encoding
• PCA
• Feature creation (TotalArea, HouseAge, interaction terms)

4. Modeling

Algorithms used:
• Linear Regression
• LASSO/Ridge
• Random Forest Regression
• Gradient Boosting Models (XGBoost/LightGBM)

5. Evaluation

Metrics used: RMSE, MAE, R²
Feature importance visualizations included.

#--------------------------------------------------------------------------------------
Repository Structure

data/
 ├── raw/             train.csv                    # Original Kaggle file
 └── processed/       Housingdata_cleaned.csv      # Cleaned and transformed dataset

scripts/
 ├── Housingdata_analysis & pre-processing.R
 ├── Model_Training_Comparison.R
 

notebooks/
 ├── housingdata_analysis.Rmd
 ├── housingdata_modeling.Rmd


outputs/
 ├── housing_prices_outputs.html              # Trained models, Plots and visualization

README.md
requirements.txt


#--------------------------------------------------------------------------------------
Dependencies

Option A: Install from requirements.txt

install.packages(scan("requirements.txt", what="", sep="\n"))

Option B: Install manually

install.packages(c(
  "tidyverse",
  "corrplot",
  "caret",
  "glmnet",
  "randomForest",
  "fastDummies",
  "ggplot2"
))