#!/usr/bin/env python3
"""
Data Doctor: clean a dataset by handling missing values, removing duplicates,
and standardizing text fields.

Usage:
  python data_doctor.py input.csv
  python data_doctor.py input.csv --output cleaned.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan", "?"}


def standardize_text(value: object) -> object:
    """Normalize text by trimming, collapsing spaces, and lowercasing."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    text = " ".join(text.split())
    return text.lower()


def clean_dataset(input_path: Path, output_path: Path) -> dict[str, int]:
    df = pd.read_csv(input_path)

    original_rows = len(df)

    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in text_cols:
        df[col] = df[col].apply(standardize_text)
        df[col] = df[col].replace(MISSING_TOKENS, pd.NA)

    rows_before_dedup = len(df)
    df = df.drop_duplicates()
    duplicates_removed = rows_before_dedup - len(df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in non_numeric_cols:
        if df[col].isna().any():
            mode_series = df[col].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else "unknown"
            df[col] = df[col].fillna(fill_value)

    rows_all_missing = int(df.isna().all(axis=1).sum())
    if rows_all_missing:
        df = df.dropna(how="all")

    df.to_csv(output_path, index=False)

    return {
        "original_rows": original_rows,
        "final_rows": len(df),
        "duplicates_removed": duplicates_removed,
        "missing_values_after_cleaning": int(df.isna().sum().sum()),
    }


def why_cleaning_matters() -> str:
    return (
        "Why cleaning matters:\n"
        "1. Missing values can break calculations or bias model output.\n"
        "2. Duplicate records inflate counts and distort analysis.\n"
        "3. Inconsistent text (like 'NY', 'ny ', 'Ny') fragments categories.\n"
        "4. Clean data improves trust, accuracy, and reproducibility."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean a CSV dataset.")
    parser.add_argument("input", type=Path, help="Path to input CSV file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to cleaned CSV file. Defaults to <input>_cleaned.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output or input_path.with_name(f"{input_path.stem}_cleaned.csv")

    stats = clean_dataset(input_path, output_path)

    print("Data Doctor Report")
    print("------------------")
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print(f"Original rows: {stats['original_rows']}")
    print(f"Final rows:    {stats['final_rows']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Missing values after cleaning: {stats['missing_values_after_cleaning']}")
    print()
    print(why_cleaning_matters())


if __name__ == "__main__":
    main()
