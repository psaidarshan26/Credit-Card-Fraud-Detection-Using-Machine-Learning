# Credit-Card-Fraud-Detection-Using-Machine-Learning
Machine Learning-based credit card fraud detection system that identifies fraudulent transactions using classification algorithms, class imbalance handling techniques, SMOTE, and neural networks.

What is this project about?
This project focuses on identifying fraudulent credit card transactions using Machine Learning algorithms.
The main problem is that most credit card transactions are legitimate, while only a very small percentage are fraudulent.
This creates an imbalanced dataset, making it difficult for machine learning models to identify fraud correctly.
The project applies different techniques to handle this problem and improve fraud detection.

What does the project do?
- Analyzes credit card transaction data.
- Identifies the imbalance between fraudulent and legitimate transactions.
- Uses Random Under-Sampling to balance the dataset.
- Uses SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic fraud samples.
- Identifies outliers using the IQR method.
- Applies dimensionality reduction techniques such as PCA and t-SNE.
- Trains different classification algorithms.
- Compares model performance using confusion matrices, precision, recall, F1-score, and ROC curves.
- Compares neural network performance on under-sampled and over-sampled datasets.

Machine Learning algorithms and techniques:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Neural Networks
- SMOTE
- PCA
- t-SNE

Technologies used:
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Imbalanced-learn
