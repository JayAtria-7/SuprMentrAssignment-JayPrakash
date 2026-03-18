import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"


def load_dataset(source: str) -> pd.DataFrame:
    path = Path(source)
    if path.exists():
        return pd.read_csv(path)
    return pd.read_csv(source)


def highest_value_column(df: pd.DataFrame):
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return None, None
    col_max = numeric.max(numeric_only=True)
    col_name = col_max.idxmax()
    return col_name, col_max[col_name]


def strongest_correlation(df: pd.DataFrame):
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr(numeric_only=True).abs()
    corr.values[[range(corr.shape[0])], [range(corr.shape[1])]] = 0
    stacked = corr.stack()
    if stacked.empty:
        return None
    pair = stacked.idxmax()
    value = stacked.max()
    return pair, value


def generate_insights(df: pd.DataFrame, max_col, max_val):
    insights = []

    duplicate_count = int(df.duplicated().sum())
    insights.append(
        f"Dataset has {df.shape[0]} rows, {df.shape[1]} columns, and {duplicate_count} duplicate rows."
    )

    missing = df.isna().sum().sort_values(ascending=False)
    if missing.iloc[0] > 0:
        insights.append(f"Most missing data is in '{missing.index[0]}' with {int(missing.iloc[0])} null values.")
    else:
        insights.append("No missing values were detected in any column.")

    if max_col is not None:
        insights.append(f"Highest numeric value appears in '{max_col}' with a maximum of {max_val}.")
    else:
        insights.append("No numeric columns are available, so no highest numeric value column was found.")

    corr_result = strongest_correlation(df)
    if corr_result:
        (left, right), corr_val = corr_result
        insights.append(
            f"Strongest absolute correlation is between '{left}' and '{right}' at {corr_val:.3f}."
        )
    else:
        insights.append("Not enough numeric columns to compute a meaningful correlation insight.")

    categorical = df.select_dtypes(include=["object", "category", "bool"])
    if not categorical.empty:
        first_cat = categorical.columns[0]
        top_val = df[first_cat].mode(dropna=True).iloc[0]
        pct = (df[first_cat] == top_val).mean() * 100
        insights.append(
            f"Most common value in '{first_cat}' is '{top_val}', appearing in {pct:.2f}% of rows."
        )
    else:
        insights.append("No categorical columns found for a category-distribution insight.")

    return insights[:5]


def run(source: str):
    df = load_dataset(source)
    max_col, max_val = highest_value_column(df)

    print(f"Dataset source: {source}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nTop 5 rows:")
    print(df.head().to_string(index=False))

    print("\nHighest value column (numeric):")
    if max_col is None:
        print("No numeric columns found.")
    else:
        print(f"{max_col}: {max_val}")

    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    print("\n5 insights:")
    for i, insight in enumerate(generate_insights(df, max_col, max_val), start=1):
        print(f"{i}. {insight}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Detective assignment runner.")
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="CSV path or URL. Defaults to Titanic dataset URL.",
    )
    args = parser.parse_args()
    run(args.source)


if __name__ == "__main__":
    main()
