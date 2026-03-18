"""
Assignment Name: Build Your First Dataset
Description:
1) Create a dataset (study hours vs marks)
2) Identify feature(s) and label
3) Predict the relationship
"""


def fit_linear_regression(x_values, y_values):
    """
    Fit y = m*x + b using least squares and return (m, b).
    """
    n = len(x_values)
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    denominator = sum((x - mean_x) ** 2 for x in x_values)

    m = numerator / denominator
    b = mean_y - (m * mean_x)
    return m, b


def predict(m, b, x):
    return (m * x) + b


def main():
    # 1) Create dataset
    study_hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    marks = [35, 40, 50, 55, 60, 68, 72, 80, 88, 95]

    dataset = list(zip(study_hours, marks))

    print("Dataset (study_hours, marks):")
    for row in dataset:
        print(row)

    # 2) Identify features and labels
    X = study_hours  # Feature
    y = marks        # Label

    print("\nFeature (X): study_hours")
    print("Label (y): marks")

    # 3) Predict relationship using simple linear regression
    m, b = fit_linear_regression(X, y)
    print(f"\nPredicted relationship: marks = {m:.2f} * study_hours + {b:.2f}")

    # Predictions for new values
    new_hours = [2.5, 5.5, 7.5]
    print("\nPredictions:")
    for hours in new_hours:
        pred = predict(m, b, hours)
        print(f"Study Hours: {hours} -> Predicted Marks: {pred:.2f}")


if __name__ == "__main__":
    main()
