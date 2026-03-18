from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

Label = Literal["ham", "spam"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _guess_column(columns: list[str], preferred: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for name in preferred:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _normalize_label(value: object) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Empty label")
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)

    s = str(value).strip().lower()
    if s in {"spam", "junk", "1", "yes", "true"}:
        return 1
    if s in {"ham", "not spam", "not_spam", "0", "no", "false"}:
        return 0
    raise ValueError(f"Unrecognized label: {value!r} (expected spam/ham or 1/0)")


def load_dataset(path: Path, text_col: str | None = None, label_col: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() in {".tsv", ".tab"}:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    if text_col is None:
        text_col = _guess_column(df.columns.tolist(), ["text", "message", "sms", "body", "content"])
    if label_col is None:
        label_col = _guess_column(df.columns.tolist(), ["label", "category", "is_spam", "spam"])

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not infer columns. Found columns: {df.columns.tolist()}. "
            f"Pass --text-col and --label-col."
        )

    out = df[[label_col, text_col]].copy()
    out.rename(columns={label_col: "label", text_col: "text"}, inplace=True)
    out["text"] = out["text"].fillna("").astype(str)

    def safe_norm(v: object) -> float:
        try:
            return float(_normalize_label(v))
        except Exception:
            return float("nan")

    out["y"] = out["label"].apply(safe_norm)
    out = out.dropna(subset=["y", "text"])
    out = out[out["text"].str.len() > 0]
    out = out.drop_duplicates(subset=["text", "y"])
    out["y"] = out["y"].astype(int)
    return out.reset_index(drop=True)


URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")
REPEAT_RE = re.compile(r"(.)\1{2,}")


class TextStatsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: Iterable[str], y: object = None) -> "TextStatsTransformer":
        return self

    def transform(self, X: Iterable[str]) -> csr_matrix:
        rows: list[list[float]] = []
        for raw in X:
            text = "" if raw is None else str(raw)
            lower = text.lower()

            length = float(len(text))
            word_count = float(len(text.split()))
            url_count = float(len(URL_RE.findall(text)))
            email_count = float(len(EMAIL_RE.findall(text)))
            phone_count = float(len(PHONE_RE.findall(text)))
            exclamations = float(text.count("!"))
            question_marks = float(text.count("?"))
            currency = float(sum(text.count(sym) for sym in ["$", "€", "£", "₹"]))

            digit_count = float(sum(ch.isdigit() for ch in text))
            alpha_count = float(sum(ch.isalpha() for ch in text))
            upper_count = float(sum(ch.isupper() for ch in text))
            digit_ratio = digit_count / max(1.0, length)
            upper_ratio = upper_count / max(1.0, alpha_count)

            repeat_runs = [len(m.group(0)) for m in REPEAT_RE.finditer(text)]
            max_repeat_run = float(max(repeat_runs) if repeat_runs else 0.0)

            has_unsubscribe = 1.0 if "unsubscribe" in lower else 0.0
            has_urgent = 1.0 if any(k in lower for k in ["urgent", "act now", "limited time", "last chance"]) else 0.0
            has_prize = 1.0 if any(k in lower for k in ["winner", "won", "prize", "claim", "congrat"]) else 0.0

            rows.append(
                [
                    length,
                    word_count,
                    url_count,
                    email_count,
                    phone_count,
                    exclamations,
                    question_marks,
                    currency,
                    digit_ratio,
                    upper_ratio,
                    max_repeat_run,
                    has_unsubscribe,
                    has_urgent,
                    has_prize,
                ]
            )

        arr = np.asarray(rows, dtype=np.float32)
        return csr_matrix(arr)


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    balanced: bool = True


def build_pipeline(config: TrainConfig) -> Pipeline:
    features = FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    strip_accents="unicode",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    max_df=0.95,
                ),
            ),
            ("stats", TextStatsTransformer()),
        ],
        n_jobs=None,
    )

    clf = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        max_iter=2000,
        class_weight=("balanced" if config.balanced else None),
        random_state=config.random_state,
        n_jobs=1,
    )

    return Pipeline([("features", features), ("clf", clf)])


def fit_model(df: pd.DataFrame, config: TrainConfig) -> tuple[dict, dict]:
    X = df["text"].tolist()
    y = df["y"].astype(int).to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    report = {
        "created_utc": _utc_now_iso(),
        "val_size": int(len(y_val)),
        "class_balance_val": {"ham": int((y_val == 0).sum()), "spam": int((y_val == 1).sum())},
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
        "classification_report": classification_report(
            y_val, y_pred, target_names=["ham", "spam"], digits=4, zero_division=0
        ),
    }
    if len(np.unique(y_val)) == 2:
        report["roc_auc"] = float(roc_auc_score(y_val, y_prob))

    artifact = {
        "pipeline": pipeline,
        "created_utc": report["created_utc"],
        "sklearn": "pipeline: tfidf(word+char)+stats -> logistic_regression",
        "labels": {"ham": 0, "spam": 1},
        "default_threshold": 0.5,
        "train_config": {"test_size": config.test_size, "random_state": config.random_state, "balanced": config.balanced},
    }
    return artifact, report


def save_artifact(artifact: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> dict:
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "pipeline" not in obj:
        raise ValueError("Invalid model artifact (expected a dict with key 'pipeline').")
    return obj


def predict_texts(artifact: dict, texts: list[str], threshold: float) -> pd.DataFrame:
    pipeline: Pipeline = artifact["pipeline"]
    proba = pipeline.predict_proba(texts)[:, 1]
    pred = (proba >= threshold).astype(int)
    labels = np.where(pred == 1, "spam", "ham")
    return pd.DataFrame({"text": texts, "spam_probability": proba, "prediction": labels})


def _cmd_train(args: argparse.Namespace) -> int:
    df = load_dataset(Path(args.data), text_col=args.text_col, label_col=args.label_col)
    config = TrainConfig(test_size=args.test_size, random_state=args.random_state, balanced=(not args.no_balanced))

    artifact, report = fit_model(df, config)
    save_artifact(artifact, Path(args.model_out))

    print(report["classification_report"].rstrip())
    print("Confusion matrix [ham, spam]:")
    print(np.array(report["confusion_matrix"]))
    if "roc_auc" in report:
        print(f"ROC-AUC: {report['roc_auc']:.4f}")

    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    artifact = load_artifact(Path(args.model))
    threshold = float(args.threshold)
    df = load_dataset(Path(args.data), text_col=args.text_col, label_col=args.label_col)

    pipeline: Pipeline = artifact["pipeline"]
    y_true = df["y"].astype(int).to_numpy()
    y_prob = pipeline.predict_proba(df["text"].tolist())[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(classification_report(y_true, y_pred, target_names=["ham", "spam"], digits=4, zero_division=0).rstrip())
    print("Confusion matrix [ham, spam]:")
    print(confusion_matrix(y_true, y_pred))
    if len(np.unique(y_true)) == 2:
        print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    artifact = load_artifact(Path(args.model))
    threshold = float(args.threshold)

    if args.text is not None:
        texts = [args.text]
        out = predict_texts(artifact, texts, threshold)
        print(out.to_string(index=False))
        return 0

    if args.input is None:
        raise SystemExit("Provide either --text or --input.")

    input_path = Path(args.input)
    if input_path.suffix.lower() in {".tsv", ".tab"}:
        df_in = pd.read_csv(input_path, sep="\t")
    else:
        df_in = pd.read_csv(input_path)

    text_col = args.text_col or _guess_column(df_in.columns.tolist(), ["text", "message", "sms", "body", "content"])
    if text_col is None:
        raise ValueError(f"Could not infer text column from: {df_in.columns.tolist()}. Pass --text-col.")

    preds = predict_texts(artifact, df_in[text_col].astype(str).fillna("").tolist(), threshold)
    df_out = df_in.copy()
    df_out["spam_probability"] = preds["spam_probability"].to_numpy()
    df_out["prediction"] = preds["prediction"].to_numpy()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
    else:
        print(df_out.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/evaluate/predict a simple spam classifier.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a model from a labeled CSV/TSV dataset.")
    p_train.add_argument("--data", required=True, help="Path to CSV/TSV containing text + labels.")
    p_train.add_argument("--text-col", default=None, help="Text column name (default: auto-detect).")
    p_train.add_argument("--label-col", default=None, help="Label column name (default: auto-detect).")
    p_train.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction (default: 0.2).")
    p_train.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42).")
    p_train.add_argument("--no-balanced", action="store_true", help="Disable class_weight='balanced'.")
    p_train.add_argument("--model-out", required=True, help="Output path for .joblib model artifact.")
    p_train.add_argument("--report-out", default=None, help="Optional path to write JSON report.")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model on a labeled dataset.")
    p_eval.add_argument("--model", required=True, help="Path to .joblib model artifact.")
    p_eval.add_argument("--data", required=True, help="Path to labeled CSV/TSV dataset.")
    p_eval.add_argument("--text-col", default=None, help="Text column name (default: auto-detect).")
    p_eval.add_argument("--label-col", default=None, help="Label column name (default: auto-detect).")
    p_eval.add_argument("--threshold", type=float, default=0.5, help="Spam probability threshold (default: 0.5).")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_pred = sub.add_parser("predict", help="Predict spam/ham for text or a CSV/TSV.")
    p_pred.add_argument("--model", required=True, help="Path to .joblib model artifact.")
    p_pred.add_argument("--threshold", type=float, default=0.5, help="Spam probability threshold (default: 0.5).")
    p_pred.add_argument("--text", default=None, help="Single text to score.")
    p_pred.add_argument("--input", default=None, help="CSV/TSV with a text column to score.")
    p_pred.add_argument("--text-col", default=None, help="Text column name for --input (default: auto-detect).")
    p_pred.add_argument("--output", default=None, help="Optional output CSV path (default: print).")
    p_pred.set_defaults(func=_cmd_predict)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
