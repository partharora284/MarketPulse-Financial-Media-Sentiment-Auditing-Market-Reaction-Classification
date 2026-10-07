"""
audit_metrics.py
=================
MarketPulse — Audit Phase

Loads the artifacts produced by train_pipeline.py and:
  1. Computes sliced performance metrics on two subgroup definitions
     (headline volume, and geopolitical vs. monetary/earnings lexicon).
  2. Generates global SHAP token attributions (top Bullish / Bearish drivers).
  3. Generates a local SHAP waterfall plot for a flagged error case
     (a false negative on an up day).

This script never retrains the model — it only reads artifacts/ and writes
new audit outputs alongside them. Works with either winning model type from
train_pipeline.py's competition (LogisticRegression or XGBClassifier) — the
SHAP explainer is picked based on the loaded model's type.

Usage
-----
    python audit_metrics.py --artifacts artifacts

Outputs (written to --artifacts)
---------------------------------
    sliced_metrics.csv          subgroup performance table
    global_shap_tokens.json     top bullish/bearish tokens with signed impact + mean |SHAP|
    shap_explainer.joblib       fitted SHAP explainer (reused by app.py)
    test_shap_values.joblib     precomputed, feature-name-tagged SHAP values for the test set
    local_waterfall_example.png waterfall plot for a selected error case (real token labels)
    error_case.json             metadata (date, headline digest, prediction) for that case
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for script/CI use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Slice definitions
# ---------------------------------------------------------------------------

# Word-count thresholds for the headline-volume slice are computed per-run
# from the test set's own quantiles (see `compute_volume_thresholds` below)
# rather than fixed constants. A fixed "> 500 / <= 300" cutoff (the brief's
# illustrative example) collapses to a near-empty quiet-day slice on the
# actual Combined_News_DJIA.csv distribution (test-set word counts cluster
# tightly around 400-500, min ~224) — quantile-based cutoffs guarantee both
# slices are populated and roughly balanced regardless of the underlying
# distribution.
VOLUME_LOWER_QUANTILE = 0.25
VOLUME_UPPER_QUANTILE = 0.75

GEOPOLITICAL_PATTERN = r"\b(?:war|sanction|sanctions|oil|conflict|military|invasion|troops)\b"
MONETARY_PATTERN = r"\b(?:fed|rate|rates|inflation|earnings|bank|banks|gdp|jobs)\b"

TOP_N_TOKENS = 20


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_artifacts(artifacts_dir: Path):
    tfidf = joblib.load(artifacts_dir / "tfidf_vectorizer.joblib")
    model = joblib.load(artifacts_dir / "market_model.joblib")
    test_df = pd.read_csv(artifacts_dir / "test_split.csv")
    test_df["Date"] = pd.to_datetime(test_df["Date"])
    logger.info(
        "Loaded artifacts and test split (%d rows). Model type: %s",
        len(test_df), type(model).__name__,
    )
    return tfidf, model, test_df


# ---------------------------------------------------------------------------
# Sliced evaluation
# ---------------------------------------------------------------------------

def compute_slice_metrics(
    subset_df: pd.DataFrame, tfidf, model, slice_name: str
) -> dict:
    if subset_df.empty:
        logger.warning("Slice '%s' is empty — skipping.", slice_name)
        return {"slice": slice_name, "n": 0, "macro_f1": None, "accuracy": None}

    X = tfidf.transform(subset_df["digest"].fillna(""))
    y = subset_df["Label"]
    y_pred = model.predict(X)

    return {
        "slice": slice_name,
        "n": int(len(subset_df)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y, y_pred)),
    }


def compute_volume_thresholds(test_df: pd.DataFrame) -> tuple[float, float]:
    """Derive high-volume / quiet-day cutoffs from the test set's own
    word-count distribution (lower and upper quantile), instead of fixed
    constants that may not fit the dataset at hand.
    """
    lower = test_df["word_count"].quantile(VOLUME_LOWER_QUANTILE)
    upper = test_df["word_count"].quantile(VOLUME_UPPER_QUANTILE)
    logger.info(
        "Volume slice thresholds (data-driven): quiet <= %.0f words (p%d), "
        "high-volume > %.0f words (p%d)",
        lower, int(VOLUME_LOWER_QUANTILE * 100), upper, int(VOLUME_UPPER_QUANTILE * 100),
    )
    return lower, upper


def run_slice_audit(test_df: pd.DataFrame, tfidf, model) -> pd.DataFrame:
    """Slice 1: headline volume (quantile-based). Slice 2: lexicon domain."""
    quiet_threshold, high_volume_threshold = compute_volume_thresholds(test_df)

    high_volume = test_df[test_df["word_count"] > high_volume_threshold]
    quiet_days = test_df[test_df["word_count"] <= quiet_threshold]

    geo_days = test_df[test_df["digest"].str.contains(GEOPOLITICAL_PATTERN, regex=True, na=False)]
    macro_days = test_df[test_df["digest"].str.contains(MONETARY_PATTERN, regex=True, na=False)]

    results = [
        compute_slice_metrics(
            high_volume, tfidf, model,
            f"high_volume_breaking_news (>{high_volume_threshold:.0f} words)",
        ),
        compute_slice_metrics(
            quiet_days, tfidf, model,
            f"quiet_days (<={quiet_threshold:.0f} words)",
        ),
        compute_slice_metrics(geo_days, tfidf, model, "geopolitical_crisis"),
        compute_slice_metrics(macro_days, tfidf, model, "monetary_earnings"),
        compute_slice_metrics(test_df, tfidf, model, "overall_test_set"),
    ]

    results_df = pd.DataFrame(results)
    logger.info("\n%s", results_df.to_string(index=False))
    return results_df


# ---------------------------------------------------------------------------
# SHAP explainer dispatch (model-type aware)
# ---------------------------------------------------------------------------

def build_explainer(model, masker):
    """Pick the correct SHAP explainer for whichever model won the
    train_pipeline.py competition. LinearExplainer is exact and fast for
    Logistic Regression; TreeExplainer is exact and fast for tree ensembles
    (XGBoost, Random Forest) and does not need (or accept) an Independent
    masker.
    """
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, masker)
    # Tree ensembles (XGBoost, RandomForestClassifier, etc.)
    return shap.TreeExplainer(model)


def to_explainer_input(X, explainer):
    """TreeExplainer needs dense input for sklearn tree ensembles
    (RandomForestClassifier) — shap's internal np.isnan() check raises a
    casting error on a sparse matrix. XGBoost's TreeExplainer and
    LinearExplainer both accept sparse input directly, and also work fine
    with dense input at this dataset's scale (max ~1989 rows x 5000
    features, ~15 MB dense) — so we densify uniformly for any TreeExplainer
    to keep one code path instead of branching on the specific model class.
    """
    if isinstance(explainer, shap.TreeExplainer):
        return np.asarray(X.todense())
    return X


def attach_feature_names(shap_values: shap.Explanation, feature_names) -> shap.Explanation:
    """Fix for the 'Feature 4771' waterfall bug: shap.Explanation objects
    built from a raw sparse matrix (no DataFrame columns) carry generic
    'Feature N' names by default. Assigning .feature_names directly attaches
    the real TF-IDF vocabulary tokens, and this survives per-row slicing
    (shap_values[i].feature_names), which is what the waterfall plot reads.
    """
    shap_values.feature_names = list(feature_names)
    return shap_values


# ---------------------------------------------------------------------------
# Global SHAP attribution
# ---------------------------------------------------------------------------

def compute_global_shap(tfidf, model, test_df: pd.DataFrame):
    """Fit a SHAP explainer over the full test-set TF-IDF matrix and return
    the explainer + shap_values (reused for the local plot) plus top
    bullish/bearish token rankings.

    Fix for the 'all 0.0000' global-token bug: the *signed* mean SHAP value
    averaged across ALL samples collapses to numerical zero (~1e-17) for
    every token whenever the explainer's background equals the foreground
    (the test set evaluated against itself) — this is a mathematical
    identity of Shapley values, not a display/precision issue, so rounding
    or reformatting alone can't fix it. Two changes fix it:
      1. Importance ranking uses mean ABSOLUTE SHAP value (never cancels).
      2. Direction is the mean SIGNED SHAP value computed only over the rows
         where the token actually appears (TF-IDF > 0) — averaging in the
         many "absent" rows (which are what makes the all-sample mean
         collapse) is what washed out the signal in the original code.
    """
    X_test_vec = tfidf.transform(test_df["digest"].fillna(""))
    feature_names = tfidf.get_feature_names_out()

    masker = shap.maskers.Independent(X_test_vec, max_samples=X_test_vec.shape[0])
    explainer = build_explainer(model, masker)
    shap_values = explainer(to_explainer_input(X_test_vec, explainer))

    if shap_values.values.ndim == 3:
        # RandomForestClassifier's TreeExplainer (and some other
        # explainer/model combinations) return (n_samples, n_features,
        # n_classes) even for binary classification. Slice the Explanation
        # object itself (not just the raw array) down to the "Up" (class 1)
        # output — this keeps shap_values usable for the local waterfall
        # plot later (shap.plots.waterfall needs a single-output row), not
        # just for the aggregate importance computation below.
        shap_values = shap_values[:, :, 1]

    shap_values = attach_feature_names(shap_values, feature_names)
    values = shap_values.values

    present_mask = (X_test_vec > 0).toarray()

    mean_abs_importance = np.abs(values).mean(axis=0)

    masked_for_direction = np.where(present_mask, values, np.nan)
    with warnings.catch_warnings():
        # A token that never appears anywhere in the test set (possible for
        # rare n-grams near the min_df cutoff) yields an all-NaN column here;
        # nanmean's "Mean of empty slice" warning for that case is expected
        # and harmless — nan_to_num below turns it into a direction of 0.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        signed_when_present = np.nanmean(masked_for_direction, axis=0)
    signed_when_present = np.nan_to_num(signed_when_present, nan=0.0)

    direction = pd.Series(signed_when_present, index=feature_names)
    importance = pd.Series(mean_abs_importance, index=feature_names)

    top_bullish = direction[direction > 0].sort_values(ascending=False).head(TOP_N_TOKENS)
    top_bearish = direction[direction < 0].sort_values().head(TOP_N_TOKENS)

    token_summary = {
        "note": (
            "signed_impact is the mean SHAP value computed only over test-set "
            "days where the token appears (not averaged over absent days, "
            "which mathematically washes signed values to ~0). mean_abs_shap "
            "is the overall importance magnitude across all days."
        ),
        "top_bullish_tokens": [
            {
                "token": tok,
                "signed_impact": float(val),
                "mean_abs_shap": float(importance[tok]),
            }
            for tok, val in top_bullish.items()
        ],
        "top_bearish_tokens": [
            {
                "token": tok,
                "signed_impact": float(val),
                "mean_abs_shap": float(importance[tok]),
            }
            for tok, val in top_bearish.items()
        ],
    }
    return explainer, shap_values, X_test_vec, token_summary


# ---------------------------------------------------------------------------
# Local error-case explanation
# ---------------------------------------------------------------------------

def find_false_negative(model, tfidf, test_df: pd.DataFrame) -> int | None:
    """Return the positional index (within test_df) of a false negative:
    predicted Down (0) on a day the index actually closed Up (1).
    Prefers the case with the highest headline word count (most likely to be
    a 'sensationalist negative headlines on an up day' scenario).
    """
    X = tfidf.transform(test_df["digest"].fillna(""))
    y_pred = model.predict(X)

    fn_mask = (y_pred == 0) & (test_df["Label"].values == 1)
    fn_positions = test_df.index[fn_mask].tolist()

    if not fn_positions:
        logger.warning("No false negatives found in test set.")
        return None

    fn_subset = test_df.loc[fn_positions]
    chosen = fn_subset["word_count"].idxmax()
    return test_df.index.get_loc(chosen)


def plot_local_waterfall(
    shap_values, position: int, test_df: pd.DataFrame, outdir: Path
) -> dict:
    row = test_df.iloc[position]

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[position], max_display=15, show=False)
    plt.tight_layout()
    out_path = outdir / "local_waterfall_example.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved local waterfall plot for %s -> %s", row["Date"].date(), out_path)

    return {
        "date": str(row["Date"].date()),
        "actual_label": int(row["Label"]),
        "predicted_label": 0,  # by construction (false negative)
        "word_count": int(row["word_count"]),
        "digest_preview": row["digest"][:300],
        "plot_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the MarketPulse model: slices + SHAP.")
    parser.add_argument("--artifacts", default="artifacts", help="Directory with trained artifacts")
    args = parser.parse_args()
    artifacts_dir = Path(args.artifacts)

    tfidf, model, test_df = load_artifacts(artifacts_dir)

    # 1. Sliced performance
    sliced_df = run_slice_audit(test_df, tfidf, model)
    sliced_df.to_csv(artifacts_dir / "sliced_metrics.csv", index=False)

    # 2. Global SHAP attribution
    explainer, shap_values, X_test_vec, token_summary = compute_global_shap(tfidf, model, test_df)
    with open(artifacts_dir / "global_shap_tokens.json", "w") as f:
        json.dump(token_summary, f, indent=2)
    logger.info(
        "Top bullish tokens: %s",
        [t["token"] for t in token_summary["top_bullish_tokens"][:10]],
    )
    logger.info(
        "Top bearish tokens: %s",
        [t["token"] for t in token_summary["top_bearish_tokens"][:10]],
    )

    # Persist the explainer and the full test-set SHAP values (with feature
    # names attached) so app.py can index straight into them for the SHAP
    # Explorer tab instead of rebuilding an explainer at runtime.
    joblib.dump(explainer, artifacts_dir / "shap_explainer.joblib")
    joblib.dump(shap_values, artifacts_dir / "test_shap_values.joblib")
    logger.info("Serialized SHAP explainer and test SHAP values to disk.")

    # 3. Local error-case explanation
    fn_position = find_false_negative(model, tfidf, test_df)
    if fn_position is not None:
        error_case = plot_local_waterfall(shap_values, fn_position, test_df, artifacts_dir)
        with open(artifacts_dir / "error_case.json", "w") as f:
            json.dump(error_case, f, indent=2)

    logger.info("Audit complete. Outputs written to %s", artifacts_dir.resolve())


if __name__ == "__main__":
    main()
