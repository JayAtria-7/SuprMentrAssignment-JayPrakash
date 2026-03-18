'''


Create dataset manually with:

Hours studied

Sleep hours

Previous score

Final score

Tasks:

Identify feature & label

Split into train-test (70-30)

Train Linear Regression

Print MAE, MSE, R²

Comment whether model is good or bad
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = {
    "Hours_Studied": [2, 4, 6, 8, 10, 3, 7, 9],
    "Sleep_Hours": [8, 7, 6, 6, 5, 9, 6, 5],
    "Previous_Score": [50, 60, 65, 75, 85, 55, 70, 80],
    "Final_Score": [55, 65, 70, 80, 90, 58, 78, 88]
}

df = pd.DataFrame(data)


X = df[["Hours_Studied", "Sleep_Hours", "Previous_Score"]]
y = df["Final_Score"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)



model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

if r2 > 0.8:
    print("Model is good. It explains most of the variance.")
else:
    print("Model is not very strong and may need improvement.")