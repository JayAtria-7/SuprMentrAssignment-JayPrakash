'''
ask 1: Threshold Experiment

Using same dataset:

Try thresholds:

0.3

0.5

0.7

Calculate:

Accuracy

Precision

Recall

Compare results in table.

Question:
Which threshold is best for cancer detection and why?
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_probs = model.predict_proba(X_test)[:, 1]

y_pred_03 = (y_probs >= 0.3).astype(int)
y_pred_05 = (y_probs >= 0.5).astype(int)
y_pred_07 = (y_probs >= 0.7).astype(int)

results = {
    "Threshold": [0.3, 0.5, 0.7],
    "Accuracy" : [round(accuracy_score(y_test, y_pred_03), 4),
                  round(accuracy_score(y_test, y_pred_05), 4),
                  round(accuracy_score(y_test, y_pred_07), 4)],
    "Precision": [round(precision_score(y_test, y_pred_03), 4),
                  round(precision_score(y_test, y_pred_05), 4),
                  round(precision_score(y_test, y_pred_07), 4)],
    "Recall"   : [round(recall_score(y_test, y_pred_03), 4),
                  round(recall_score(y_test, y_pred_05), 4),
                  round(recall_score(y_test, y_pred_07), 4)]
}

df = pd.DataFrame(results)
print(df.to_string(index=False))

x = np.arange(3)
labels = ["Threshold 0.3", "Threshold 0.5", "Threshold 0.7"]

plt.figure(figsize=(8, 5))
plt.bar(x - 0.25, results["Accuracy"],  0.25, label="Accuracy",  color="#4e79a7")
plt.bar(x,        results["Precision"], 0.25, label="Precision", color="#f28e2b")
plt.bar(x + 0.25, results["Recall"],    0.25, label="Recall",    color="#e15759")

plt.xticks(x, labels)
plt.ylim(0.85, 1.05)
plt.ylabel("Score")
plt.title("Threshold Comparison")
plt.legend()
plt.tight_layout()
plt.show()