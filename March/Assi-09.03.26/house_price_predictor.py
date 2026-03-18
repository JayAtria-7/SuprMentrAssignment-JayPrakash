from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_CSV_PATH = Path("data/house_prices.csv")
DEFAULT_MODEL_PATH = Path("model/house_price_model.joblib")


@dataclass(frozen=True)
class Dataset:
    feature_names: list[str]
    X: np.ndarray
    y: np.ndarray


def _ensure_sample_dataset(csv_path: Path, *, rows: int = 120, seed: int = 42) -> None:
    if csv_path.exists():
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    fieldnames = ["sqft", "bedrooms", "bathrooms", "age", "distance_to_city", "price"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(rows):
            sqft = rng.randint(600, 4200)
            bedrooms = rng.randint(1, 6)
            bathrooms = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
            age = rng.randint(0, 70)
            distance = round(rng.uniform(0.5, 40.0), 1)

            noise = rng.gauss(0, 25_000)
            price = (
                60_000
                + sqft * 280
                + bedrooms * 12_000
                + bathrooms * 18_000
                - age * 1_100
                - distance * 2_200
                + noise
            )
            price = max(price, 35_000)

            writer.writerow(
                {
                    "sqft": sqft,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "age": age,
                    "distance_to_city": distance,
                    "price": round(price, 2),
                }
            )


def _load_dataset(csv_path: Path) -> Dataset:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        fieldnames = [name.strip() for name in reader.fieldnames if name and name.strip()]
        if "price" not in fieldnames:
            raise ValueError("CSV must contain a 'price' column.")

        feature_names = [c for c in fieldnames if c != "price"]
        if not feature_names:
            raise ValueError("CSV must contain at least one feature column besides 'price'.")

        X_rows: list[list[float]] = []
        y_rows: list[float] = []

        for row in reader:
            if row is None:
                continue
            try:
                X_rows.append([float(row[name]) for name in feature_names])
                y_rows.append(float(row["price"]))
            except (TypeError, ValueError, KeyError):
                # Skip bad rows to keep the script friendly for beginners.
                continue

    if not X_rows:
        raise ValueError("No valid rows found in CSV (check for numeric values).")

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=float)
    return Dataset(feature_names=feature_names, X=X, y=y)


def _build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def train_and_save(*, csv_path: Path, model_path: Path, test_size: float = 0.2, seed: int = 7) -> None:
    _ensure_sample_dataset(csv_path)
    dataset = _load_dataset(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=test_size, random_state=seed
    )

    model = _build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": dataset.feature_names}, model_path)

    print("Training complete.")
    print(f"Data:   {csv_path}")
    print(f"Model:  {model_path}")
    print(f"Test:   MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  R^2={r2:.3f}")


def _load_model(model_path: Path) -> tuple[Pipeline, list[str]]:
    payload: Any = joblib.load(model_path)
    model = payload["model"]
    feature_names = payload["feature_names"]
    if not isinstance(feature_names, list) or not all(isinstance(x, str) for x in feature_names):
        raise ValueError("Saved model has invalid 'feature_names'.")
    return model, feature_names


def predict_from_values(*, model_path: Path, values: dict[str, float]) -> float:
    model, feature_names = _load_model(model_path)
    missing = [name for name in feature_names if name not in values]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")

    X = np.array([[values[name] for name in feature_names]], dtype=float)
    pred = float(model.predict(X)[0])
    return pred


def _prompt_for_features(feature_names: list[str]) -> dict[str, float]:
    print("\nEnter new house details to predict price:")
    values: dict[str, float] = {}
    for name in feature_names:
        while True:
            raw = input(f"- {name}: ").strip()
            try:
                values[name] = float(raw)
                break
            except ValueError:
                print("  Please enter a numeric value.")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="House Price Predictor (Linear Regression)")
    subparsers = parser.add_subparsers(dest="cmd")

    train_p = subparsers.add_parser("train", help="Train the model and save it")
    train_p.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    train_p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train_p.add_argument("--test-size", type=float, default=0.2)
    train_p.add_argument("--seed", type=int, default=7)

    pred_p = subparsers.add_parser("predict", help="Predict price using a saved model")
    pred_p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    pred_p.add_argument("--sqft", type=float)
    pred_p.add_argument("--bedrooms", type=float)
    pred_p.add_argument("--bathrooms", type=float)
    pred_p.add_argument("--age", type=float)
    pred_p.add_argument("--distance-to-city", dest="distance_to_city", type=float)

    args = parser.parse_args()

    if args.cmd in (None, "train"):
        csv_path = getattr(args, "csv", DEFAULT_CSV_PATH)
        model_path = getattr(args, "model", DEFAULT_MODEL_PATH)
        test_size = getattr(args, "test_size", 0.2)
        seed = getattr(args, "seed", 7)
        train_and_save(csv_path=csv_path, model_path=model_path, test_size=test_size, seed=seed)

        if args.cmd is None:
            model, feature_names = _load_model(model_path)
            _ = model  # silence "unused" in some editors
            values = _prompt_for_features(feature_names)
            pred = predict_from_values(model_path=model_path, values=values)
            print(f"\nPredicted price: ${pred:,.0f}")
        return 0

    if args.cmd == "predict":
        model_path: Path = args.model
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            print("Run: python house_price_predictor.py train")
            return 2

        _, feature_names = _load_model(model_path)
        cli_values = {
            "sqft": args.sqft,
            "bedrooms": args.bedrooms,
            "bathrooms": args.bathrooms,
            "age": args.age,
            "distance_to_city": args.distance_to_city,
        }
        if any(v is None for v in cli_values.values()):
            values = _prompt_for_features(feature_names)
        else:
            values = {k: float(v) for k, v in cli_values.items()}  # type: ignore[arg-type]

        pred = predict_from_values(model_path=model_path, values=values)
        print(f"Predicted price: ${pred:,.0f}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

