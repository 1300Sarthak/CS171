# Gun Violence Prediction Using Machine Learning Models

## Overview

This project explores the use of machine learning to predict trends and patterns in gun violence incidents across the United States. It integrates data from multiple sources and employs models like Logistic Regression, Decision Tree, Random Forest, and Support Vector Machines (SVM). An experimental implementation of the BERT model was also attempted for deeper textual analysis, though limited by computational constraints.

## Motivation

Gun violence is a significant issue in the U.S., with over 385 mass shootings recorded in 2024 (up to September). This project aims to analyze patterns in historical data to identify high-risk factors, enabling proactive interventions.

## Dataset

- **Sources**: Multiple Kaggle datasets
- **Attributes**:
  - Incident details: ID, date, location, fatalities, injuries, victims, description
  - Gun laws: Handgun/long gun purchase age, open carry status
  - Demographic data: Population, latitude, longitude
  - Temporal data: Year, month, day for seasonal analysis

## Methodology

### Data Preprocessing
- Standardized column names and cleaned inconsistencies
- Integrated additional features like gun laws and demographic data
- Split the dataset into training and testing subsets
- Hyperparameter tuning using GridSearchCV

### Models Used
- **Logistic Regression**: Binary classification
- **Decision Tree**: Interpretability and feature importance
- **Random Forest**: Robust performance
- **SVM**: Relationship analysis
- **BERT**: Textual feature analysis (limited due to GPU constraints)

## Results

- **Random Forest**: Achieved the highest accuracy of 90.6% with balanced precision, recall, and F1 scores.
- **Decision Tree**: Comparable to Random Forest with high interpretability.
- **Logistic Regression**: Lower accuracy (40.13%) but strong recall for high-risk cases.
- **SVM**: Performed inconsistently with 39.81% accuracy.
- **BERT**: Provided promising insights despite computational limitations.

## Challenges
- Data imbalance affecting model precision
- Computational constraints for advanced models like BERT
- Data cleaning issues with inconsistent formats
- Failed heatmap visualizations, resolved partially

## Visualizations
- Heatmaps: Comparative model performance
- Bar plots: Model accuracies
- Distribution plots: Prediction ranges and trends for BERT

## Future Work
- Enhance data preprocessing and feature engineering
- Investigate ensemble methods for better generalizability
- Expand datasets to improve model robustness

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn transformers torch
2. Run the preprocessing script to integrate and clean datasets.
3. Train models by executing the CS171_Final.ipynb notebook.
4. Visualize results using the plotting functions provided in the notebook.

## Acknowledgments
- Datasets sourced from Kaggle
- Guidance from Prof. Sengupta (CS171)
- Tools like Colab and PyTorch for model development
