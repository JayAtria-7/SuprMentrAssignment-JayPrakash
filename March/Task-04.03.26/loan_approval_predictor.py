import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


FEATURE_COLUMNS = [
    "Income",
    "Credit Score",
    "Age",
    "Loan Amount",
    "Employment Years",
]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def validate_columns(df: pd.DataFrame, target_column: str) -> None:
    required = FEATURE_COLUMNS + [target_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in dataset: "
            + ", ".join(missing)
            + f"\nExpected features: {FEATURE_COLUMNS}\nTarget: {target_column}"
        )


def prepare_target(y: pd.Series):
    if pd.api.types.is_numeric_dtype(y):
        return y, None
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y.astype(str))
    return encoded, encoder


def print_feature_importance(model_name: str, fitted_pipeline: Pipeline) -> None:
    classifier = fitted_pipeline.named_steps["model"]
    importances = classifier.feature_importances_
    ranking = sorted(
        zip(FEATURE_COLUMNS, importances), key=lambda item: item[1], reverse=True
    )
    print(f"\nFeature importance ({model_name}):")
    for feature, importance in ranking:
        print(f"  {feature:20s} {importance:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Loan Approval Predictor")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--target",
        default="Loan Approved",
        help="Target column name in the dataset. Default: Loan Approved",
    )
    parser.add_argument(
        "--output",
        default="loan_approval_model.pkl",
        help="Output pickle file path. Default: loan_approval_model.pkl",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)

    print(f"Loading dataset from: {data_path}")
    df = load_dataset(data_path)
    validate_columns(df, args.target)

    X = df[FEATURE_COLUMNS]
    y_raw = df[args.target]
    y, label_encoder = prepare_target(y_raw)

    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    decision_tree = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]
    )
    random_forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )

    decision_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    dt_preds = decision_tree.predict(X_test)
    rf_preds = random_forest.predict(X_test)

    dt_acc = accuracy_score(y_test, dt_preds)
    rf_acc = accuracy_score(y_test, rf_preds)

    print("\nModel accuracy comparison:")
    print(f"  Decision Tree: {dt_acc:.4f}")
    print(f"  Random Forest: {rf_acc:.4f}")

    print_feature_importance("Decision Tree", decision_tree)
    print_feature_importance("Random Forest", random_forest)

    if rf_acc >= dt_acc:
        best_name = "Random Forest"
        best_model = random_forest
        best_accuracy = rf_acc
    else:
        best_name = "Decision Tree"
        best_model = decision_tree
        best_accuracy = dt_acc

    payload = {
        "model_name": best_name,
        "model": best_model,
        "accuracy": best_accuracy,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": args.target,
        "label_encoder_classes": (
            label_encoder.classes_.tolist() if label_encoder is not None else None
        ),
    }

    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"\nSaved best model ({best_name}, accuracy={best_accuracy:.4f}) to: {output_path}")


if __name__ == "__main__":
    main()
