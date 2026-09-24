# sleep-dataset-ML-project

## Overview

This project uses machine learning to predict sleep disorder risk, sleep quality, restability based on sleep, lifestyle, health, and daily performance features.

utilizing the following algorithms are compared:

- Decision Tree
- Random Forest
- Linear Regression

## Dataset

The project uses the Sleep Health and Daily Performance Dataset from Kaggle.

Dataset:
https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset

## Machine Learning Workflow

1. Load the dataset
2. Exploratory Data Analysis
3. Data preprocessing
6. Train/test split
7. Train Models
9. Evaluate models
10. Compare models performance

## Target Variables

The target variable is:

`sleep_disorder_risk`, `sleep_quality_score`, `felt_rested`

## Models Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

## Project Structure

```text
sleep-dataset-ML-project/
├── README.md
├── requirements.txt
├── Sleep_Health_ML.ipynb
├── LICENSE
├── app.py
├── resources/
│   └── sleep_health_dataset.csv
```
## Streamlit App Link

https://sleep-dataset-ml-project.streamlit.app/
