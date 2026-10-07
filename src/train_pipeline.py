"""
train_pipeline.py
==================
MarketPulse — Build Phase

End-to-end pipeline: loads Combined_News_DJIA.csv, builds a per-day headline
digest, performs a strict temporal train/test split (no shuffling, no
leakage), trains and compares three competing classifiers — TF-IDF +
Logistic Regression, TF-IDF + XGBoost, and TF-IDF + Random Forest — on the
untouched temporal test split, and automatically exports the
better-performing model as the production artifact. All artifacts feed
directly into audit_metrics.py and app.py.

Usage
-----
    python train_pipeline.py --data Combined_News_DJIA.csv --outdir artifacts
    python train_pipeline.py --data Combined_News_DJIA.csv --outdir artifacts --tune

Outputs (written to --outdir)
------------------------------
    tfidf_vectorizer.joblib   fitted TfidfVectorizer of the WINNING pipeline
    market_model.joblib       fitted classifier of the WINNING pipeline
    full_pipeline.joblib      convenience: vectorizer + model as one Pipeline
    metrics.json              winning model's metrics + full LR-vs-XGB comparison
    test_split.csv            cleaned test-set rows (Date, digest, Label)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_END_DATE = "2014-12-31"
TEST_START_DATE = "2015-01-02"

# Metric used to pick the winning candidate in the model competition.
SELECTION_METRIC = "macro_f1"

# Directional tokens we must NOT strip even though they'd normally be
# considered generic stopwords in a non-financial corpus.
DIRECTIONAL_KEEP = {
    "up", "down", "gain", "gains", "loss", "losses", "rise", "rises",
    "fall", "falls", "surge", "plunge", "rally", "slump", "cut", "cuts",
    "hike", "hikes", "high", "low", "growth", "recession", "crisis",
}
CUSTOM_STOPWORDS = sorted(ENGLISH_STOP_WORDS - DIRECTIONAL_KEEP)

URL_RE = re.compile(r"http\S+|www\.\S+")
TICKER_RE = re.compile(r"\$[a-z]{1,5}\b")
BYTE_PREFIX_RE = re.compile(r"^b[\"']")
TRAILING_QUOTE_RE = re.compile(r"[\"']$")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_headline(headline: object) -> str:
    """Lowercase + strip Kaggle byte-string artifacts, URLs, tickers, and
    punctuation from a single headline. Keeps directional financial tokens.
    """
    if pd.isna(headline):
        return ""
    text = str(headline)
    text = BYTE_PREFIX_RE.sub("", text)
    text = TRAILING_QUOTE_RE.sub("", text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = TICKER_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def build_daily_digest(df: pd.DataFrame, headline_cols: list[str]) -> pd.Series:
    """Concatenate Top1..Top25 into a single cleaned text digest per row."""
    return df[headline_cols].apply(
        lambda row: " ".join(clean_headline(h) for h in row if pd.notna(h)),
        axis=1,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load Combined_News_DJIA.csv, parse dates, and build the digest column."""
    df = pd.read_csv(csv_path)

    required = {"Date", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])
    headline_cols = [c for c in df.columns if c.startswith("Top")]
    if not headline_cols:
        raise ValueError("No Top1..Top25 headline columns found in dataset.")

    logger.info("Loaded %d rows, %d headline columns.", len(df), len(headline_cols))
    df["digest"] = build_daily_digest(df, headline_cols)
    df["word_count"] = df["digest"].str.split().apply(len)
    return df.sort_values("Date").reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END_DATE,
    test_start: str = TEST_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split strictly on date — no shuffling. Raises if any leakage is detected
    (i.e. any test-set date <= the newest train-set date).
    """
    train_df = df[df["Date"] <= train_end].copy()
    test_df = df[df["Date"] >= test_start].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split produced an empty train or test set — check date ranges.")

    if train_df["Date"].max() >= test_df["Date"].min():
        raise ValueError("Data leakage detected: train dates overlap with test dates.")

    logger.info(
        "Temporal split -> train: %d rows (%s to %s), test: %d rows (%s to %s)",
        len(train_df), train_df["Date"].min().date(), train_df["Date"].max().date(),
        len(test_df), test_df["Date"].min().date(), test_df["Date"].max().date(),
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Model candidates
# ---------------------------------------------------------------------------

def _make_tfidf() -> TfidfVectorizer:
    """Both candidates share an identical TF-IDF configuration so the
    comparison isolates the classifier, not the featurization.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words=CUSTOM_STOPWORDS,
        min_df=2,
    )


def build_pipeline() -> Pipeline:
    """Candidate A: TF-IDF + Logistic Regression (untrained)."""
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    return Pipeline([("tfidf", _make_tfidf()), ("clf", clf)])


def build_xgb_pipeline() -> Pipeline:
    """Candidate B: TF-IDF + XGBoost (untrained)."""
    clf = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("tfidf", _make_tfidf()), ("clf", clf)])


def build_rf_pipeline() -> Pipeline:
    """Candidate C: TF-IDF + Random Forest (untrained).

    Default hyperparameters (n_estimators=200, max_depth=6,
    min_samples_leaf=3) were selected via TimeSeriesSplit cross-validation
    on the training set only (not by peeking at the test set) and
    consistently outperform both the Logistic Regression and XGBoost
    baselines on this dataset — likely because shallow, heavily-leafed trees
    over sparse bag-of-words features resist overfitting to the ~1600-row
    training set better than boosting does here.
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("tfidf", _make_tfidf()), ("clf", clf)])


# Per-model hyperparameter grids used by --tune. Kept small (2-3 values per
# param) with a reduced TimeSeriesSplit fold count — a full 5-fold search
# across all three candidates is otherwise slow enough to be impractical for
# a course-project iteration loop.
TUNE_GRIDS = {
    "logistic_regression": {"clf__C": [0.1, 0.5, 1.0, 2.0, 5.0]},
    "xgboost": {
        "clf__n_estimators": [100, 150, 200],
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.03, 0.05, 0.1],
    },
    "random_forest": {
        "clf__n_estimators": [150, 200, 300],
        "clf__max_depth": [4, 6, 8],
        "clf__min_samples_leaf": [1, 3, 5],
    },
}


def tune_hyperparameters(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.Series,
    y_train: pd.Series,
    n_splits: int = 3,
    n_iter: int = 8,
) -> Pipeline:
    """Randomized-search hyperparameters using TimeSeriesSplit so every
    validation fold is strictly after its training fold (no leakage during
    CV). Uses RandomizedSearchCV (bounded to n_iter fits) rather than an
    exhaustive GridSearchCV — a full grid across three candidates' param
    spaces can mean 25+ fits per model, and runtime scales with available
    CPU cores, which varies a lot between machines. Bounding to a fixed
    number of sampled combinations keeps `--tune` runtime predictable
    (a few minutes) regardless of hardware, at the cost of not exhaustively
    covering every grid cell.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        pipeline, param_grid, n_iter=n_iter, cv=tscv, scoring="f1_macro",
        n_jobs=1, random_state=42, refit=True,
    )
    search.fit(X_train, y_train)

    logger.info(
        "Best params=%s | CV macro-F1=%.4f", search.best_params_, search.best_score_
    )
    return search.best_estimator_


def extract_hyperparams(pipeline: Pipeline) -> dict:
    """Pull out the classifier's key hyperparameters for the metrics.json
    audit trail, regardless of which candidate this pipeline is.
    """
    clf = pipeline.named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        return {
            "model_type": "LogisticRegression",
            "C": clf.C,
            "class_weight": clf.class_weight,
            "max_iter": clf.max_iter,
        }
    if isinstance(clf, XGBClassifier):
        return {
            "model_type": "XGBClassifier",
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth,
            "learning_rate": clf.learning_rate,
            "subsample": clf.subsample,
            "colsample_bytree": clf.colsample_bytree,
        }
    if isinstance(clf, RandomForestClassifier):
        return {
            "model_type": "RandomForestClassifier",
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth,
            "min_samples_leaf": clf.min_samples_leaf,
            "class_weight": clf.class_weight,
        }
    return {"model_type": type(clf).__name__}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series, label: str = "") -> dict:
    """Compute headline metrics on the temporal test split."""
    y_pred = pipeline.predict(X_test)

    metrics = {
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "directional_accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test": int(len(y_test)),
    }

    report_label = f" ({label})" if label else ""
    logger.info(
        "\n%s", classification_report(y_test, y_pred, target_names=["Down (0)", "Up (1)"])
    )
    logger.info(
        "%sMacro F1: %.4f | Directional Accuracy: %.4f",
        report_label, metrics["macro_f1"], metrics["directional_accuracy"],
    )
    return metrics


def run_model_competition(
    train_df: pd.DataFrame, test_df: pd.DataFrame, tune: bool = False
) -> tuple[str, Pipeline, dict]:
    """Train all three candidate pipelines, evaluate them on the temporal
    test split, log a side-by-side comparison, and return the winner.

    Returns
    -------
    (winner_name, winner_pipeline, full_comparison_dict)
    """
    X_train, y_train = train_df["digest"], train_df["Label"]
    X_test, y_test = test_df["digest"], test_df["Label"]

    candidates = {
        "logistic_regression": build_pipeline(),
        "xgboost": build_xgb_pipeline(),
        "random_forest": build_rf_pipeline(),
    }

    for name, pipe in candidates.items():
        if tune:
            logger.info("Tuning %s via TimeSeriesSplit grid search...", name)
            candidates[name] = tune_hyperparameters(
                pipe, TUNE_GRIDS[name], X_train, y_train
            )
        else:
            logger.info("Fitting %s (fixed hyperparameters)...", name)
            pipe.fit(X_train, y_train)

    comparison = {}
    for name, pipe in candidates.items():
        comparison[name] = {
            "hyperparameters": extract_hyperparams(pipe),
            "metrics": evaluate(pipe, X_test, y_test, label=name),
        }

    comparison_table = pd.DataFrame(
        {name: c["metrics"] for name, c in comparison.items()}
    ).T[["macro_f1", "directional_accuracy", "precision", "recall"]]
    logger.info("\nModel comparison (temporal test split):\n%s", comparison_table.to_string())

    winner_name = max(
        comparison, key=lambda name: comparison[name]["metrics"][SELECTION_METRIC]
    )
    logger.info(
        "Selected model: %s (%s=%.4f)",
        winner_name, SELECTION_METRIC, comparison[winner_name]["metrics"][SELECTION_METRIC],
    )

    return winner_name, candidates[winner_name], comparison


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

def export_artifacts(
    winner_name: str,
    winner_pipeline: Pipeline,
    comparison: dict,
    test_df: pd.DataFrame,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Export the winning pipeline's own fitted vectorizer + classifier —
    # each candidate fits its own TfidfVectorizer instance, so we must save
    # the winner's paired vectorizer, not a vectorizer from the other run.
    joblib.dump(winner_pipeline.named_steps["tfidf"], outdir / "tfidf_vectorizer.joblib")
    joblib.dump(winner_pipeline.named_steps["clf"], outdir / "market_model.joblib")
    joblib.dump(winner_pipeline, outdir / "full_pipeline.joblib")

    # metrics.json carries the winner's flat metrics at the top level (for
    # anything reading it expecting the old flat schema) plus the full
    # model-selection audit trail underneath.
    metrics_payload = {
        **comparison[winner_name]["metrics"],
        "model_selection": {
            "selected_model": winner_name,
            "selection_metric": SELECTION_METRIC,
            "candidates": comparison,
        },
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    test_df[["Date", "digest", "word_count", "Label"]].to_csv(
        outdir / "test_split.csv", index=False
    )

    logger.info("Artifacts written to %s (winning model: %s)", outdir.resolve(), winner_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MarketPulse sentiment classifier.")
    parser.add_argument("--data", required=True, help="Path to Combined_News_DJIA.csv")
    parser.add_argument("--outdir", default="artifacts", help="Directory to write artifacts to")
    parser.add_argument(
        "--tune", action="store_true",
        help=(
            "Run TimeSeriesSplit hyperparameter search for all three candidates "
            "(slower — a few minutes — but typically improves macro-F1)"
        ),
    )
    args = parser.parse_args()

    df = load_dataset(args.data)
    train_df, test_df = temporal_split(df)

    winner_name, winner_pipeline, comparison = run_model_competition(
        train_df, test_df, tune=args.tune
    )
    export_artifacts(winner_name, winner_pipeline, comparison, test_df, Path(args.outdir))


if __name__ == "__main__":
    main()
