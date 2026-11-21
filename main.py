import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

DATA_FILE = Path("data/creditcard.csv")
if not DATA_FILE.exists():
    raise FileNotFoundError("Download the Kaggle Credit Card Fraud Detection dataset and place creditcard.csv in data/.")

df = pd.read_csv(DATA_FILE)
if "Class" not in df.columns:
    raise ValueError("Dataset must contain the target column named 'Class'.")

print("Dataset shape:", df.shape)
print("\nClass distribution (%):")
print((df["Class"].value_counts(normalize=True) * 100).round(3))

X = df.drop(columns=["Class"]).copy()
y = df["Class"].astype(int)

# Robustly scale transaction amount and time when those columns exist.
for col in ["Amount", "Time"]:
    if col in X.columns:
        X[col] = RobustScaler().fit_transform(X[[col]])

# Split before resampling to avoid information leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Under-sampling is applied only to the training data inside the pipeline.
model = Pipeline(steps=[
    ("under_sample", RandomUnderSampler(random_state=42)),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification report:")
print(classification_report(y_test, predictions, digits=4, zero_division=0))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("\nFraud detection training and evaluation completed.")
