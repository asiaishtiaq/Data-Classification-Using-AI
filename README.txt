# AI Project 2 - Data Classification Using AI

## Project Overview
This project implements a supervised machine-learning classification model using the Iris dataset.

### Dataset
- 150 samples
- 3 classes
- 4 features
- Classes: setosa, versicolor, virginica

### Features
- sepal_length
- sepal_width
- petal_length
- petal_width

## Machine Learning Pipeline

Iris Dataset
-> 80/20 Train-Test Split
-> StandardScaler
-> K-Nearest Neighbors (KNN), k=5
-> Predictions
-> Confusion Matrix and F1 Score

## Results

- Accuracy: 93.33%
- Weighted F1 Score: 0.9327

## Project Structure

```text
AI_Project_2_GitHub_UPLOAD/
├── data/
│   └── iris.csv
├── src/
│   └── main.py
├── outputs/
│   ├── confusion_matrix.png
│   ├── f1_score.png
│   └── metrics.txt
├── report/
│   └── AI_Project_2_Report.pdf
├── presentation/
│   └── AI_Project_2_Presentation.pptx
├── requirements.txt
├── run.py
└── README.md