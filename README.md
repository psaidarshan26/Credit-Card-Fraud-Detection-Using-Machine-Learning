# Credit Card Fraud Detection

A binary classification project to identify potentially fraudulent credit-card transactions. The original dataset is highly imbalanced, so random under-sampling is applied to the training set only.

## Dataset
Download the Kaggle **Credit Card Fraud Detection** dataset and place `creditcard.csv` in `data/`.

The CSV should contain transaction features and a target column named `Class` (`0` = legitimate, `1` = fraud).

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Example output format
```text
Dataset shape: (number_of_rows, number_of_columns)

Class distribution (%):
0    [legitimate transaction percentage]
1    [fraud transaction percentage]

Confusion matrix:
[[true_negatives false_positives]
 [false_negatives true_positives]]

Classification report:
              precision    recall  f1-score   support
           0       ...
           1       ...

ROC-AUC: [score]
Fraud detection training and evaluation completed.
```
Metrics vary with the dataset and train/test split. Fraud detection should be judged using precision, recall, F1-score and ROC-AUC, not accuracy alone.

## Repository structure
```text
credit-card-fraud-detection-ml/
├── data/
│   └── creditcard.csv
├── main.py
├── requirements.txt
└── README.md
```
