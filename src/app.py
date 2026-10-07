"""
app.py
======
MarketPulse — Streamlit Web App

Two tabs:
  1. Live Market Radar     — fetches current business headlines via RSS,
                              tags them by sector with a rule-based lexicon,
                              and scores each with the trained model to show
                              aggregated per-sector directional probabilities.
  2. Historical Audit & SHAP Explorer — lets the user browse the temporal
                              test split, see sliced metrics, and inspect a
                              live SHAP waterfall plot for any selected date.

Works with either winning model type from train_pipeline.py's LR-vs-XGBoost
competition — the SHAP explainer (fallback path) is picked based on the
loaded model's type, matching audit_metrics.py's dispatch logic.

Run with:
    streamlit run app.py --  --artifacts artifacts
(Streamlit args go before "--"; script args go after.)
"""

from __future__ import annotations

import html
import json
import logging
import re
import sys
from pathlib import Path

import feedparser
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def resolve_artifacts_dir(default: str = "artifacts") -> Path:
    """Parse --artifacts from sys.argv without crashing on malformed input.

    A bare `sys.argv[sys.argv.index("--artifacts") + 1]` raises an unhandled
    IndexError if `--artifacts` is present but has no following value (e.g.
    it's the last CLI token, or Streamlit's own argument splitting leaves it
    dangling). This falls back to `default` instead.
    """
    if "--artifacts" in sys.argv:
        idx = sys.argv.index("--artifacts")
        if idx + 1 < len(sys.argv):
            return Path(sys.argv[idx + 1])
        logger.warning("--artifacts flag given with no value; falling back to '%s'.", default)
    return Path(default)


ARTIFACTS_DIR = resolve_artifacts_dir()

# Reuters discontinued its public feeds.reuters.com RSS endpoints in 2020 —
# they now return empty results rather than an error, which silently starves
# this tab. Use feeds that are confirmed live instead.
RSS_FEEDS = {
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "CNBC Top Stories": "https://search.cnbc.com/rs/search/combinedlist/view.xml?partnerId=wrss01&id=100003114",
    "MarketWatch Top Stories": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
}

SECTOR_LEXICON = {
    "Banking": ["bank", "rbi", "rate", "loan", "credit", "lender"],
    "IT": ["software", "tech", "it services", "cloud", "ai", "semiconductor"],
    "Energy": ["oil", "crude", "energy", "opec", "gas", "petroleum"],
    "Auto": ["auto", "ev", "vehicle", "car maker", "automaker"],
    "Macro": ["gdp", "inflation", "fed", "jobs", "economy", "recession"],
}

# Theme accent colors, reused across CSS and Plotly charts so both stay
# visually consistent.
COLOR_BULLISH = "#10B981"   # emerald
COLOR_BEARISH = "#F43F5E"   # coral
COLOR_NEUTRAL = "#6366F1"   # indigo
COLOR_MUTED = "#9CA3AF"

# Same cleaning regexes as train_pipeline.py — kept identical so the live
# feed goes through the same transformation the model was trained on.
URL_RE = re.compile(r"http\S+|www\.\S+")
TICKER_RE = re.compile(r"\$[a-z]{1,5}\b")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_headline(headline: str) -> str:
    text = str(headline).lower()
    text = URL_RE.sub(" ", text)
    text = TICKER_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# SHAP explainer dispatch (model-type aware) — mirrors audit_metrics.py
# ---------------------------------------------------------------------------

def build_explainer(model, masker):
    """Pick the correct SHAP explainer for whichever model won the
    train_pipeline.py competition. Only used as a fallback when
    audit_metrics.py hasn't precomputed shap_explainer.joblib yet.
    """
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, masker)
    return shap.TreeExplainer(model)


def attach_feature_names(shap_values, feature_names):
    """Fixes the 'Feature 4771' waterfall bug — see audit_metrics.py's
    identical helper for the full explanation. Assigning .feature_names
    directly attaches real TF-IDF vocabulary tokens and survives per-row
    slicing, which is what the waterfall plot reads its y-axis labels from.
    """
    shap_values.feature_names = list(feature_names)
    return shap_values


def to_explainer_input(X, explainer):
    """TreeExplainer needs dense input for sklearn tree ensembles
    (RandomForestClassifier) — shap's internal np.isnan() check raises a
    casting error on a sparse matrix. See audit_metrics.py's identical
    helper for the full explanation.
    """
    if isinstance(explainer, shap.TreeExplainer):
        return np.asarray(X.todense())
    return X


# ---------------------------------------------------------------------------
# Cached artifact / data loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts(artifacts_dir: Path):
    tfidf = joblib.load(artifacts_dir / "tfidf_vectorizer.joblib")
    model = joblib.load(artifacts_dir / "market_model.joblib")
    test_df = pd.read_csv(artifacts_dir / "test_split.csv")
    test_df["Date"] = pd.to_datetime(test_df["Date"])

    sliced_metrics_path = artifacts_dir / "sliced_metrics.csv"
    sliced_df = pd.read_csv(sliced_metrics_path) if sliced_metrics_path.exists() else None

    tokens_path = artifacts_dir / "global_shap_tokens.json"
    global_tokens = json.loads(tokens_path.read_text()) if tokens_path.exists() else None

    metrics_path = artifacts_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

    # Prefer SHAP artifacts precomputed by audit_metrics.py — avoids rebuilding
    # the explainer's background masker at app startup/interaction time.
    # Falls back to on-the-fly computation (build_shap_explainer, below) if
    # audit_metrics.py hasn't been run yet.
    explainer_path = artifacts_dir / "shap_explainer.joblib"
    shap_values_path = artifacts_dir / "test_shap_values.joblib"
    precomputed_explainer = joblib.load(explainer_path) if explainer_path.exists() else None
    precomputed_shap_values = joblib.load(shap_values_path) if shap_values_path.exists() else None

    return (
        tfidf, model, test_df, sliced_df, global_tokens, metrics,
        precomputed_explainer, precomputed_shap_values,
    )


@st.cache_data(ttl=600, show_spinner="Fetching live headlines...")
def fetch_live_headlines() -> pd.DataFrame:
    rows = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                rows.append({"source": source, "title": entry.title})
        except Exception as exc:  # keep the app alive if one feed is down
            st.warning(f"Could not fetch {source}: {exc}")
    return pd.DataFrame(rows)


def tag_sector(title: str) -> str:
    lowered = title.lower()
    for sector, keywords in SECTOR_LEXICON.items():
        if any(kw in lowered for kw in keywords):
            return sector
    return "Unclassified"


@st.cache_resource(show_spinner=False)
def build_shap_explainer(_model, _tfidf, _test_df):
    """Fallback only: builds an explainer at runtime if audit_metrics.py
    hasn't produced shap_explainer.joblib yet. Underscore-prefixed args tell
    Streamlit's cache not to hash these (unhashable) objects; since this is
    cache_resource-decorated it still only runs once per session.
    """
    X_test_vec = _tfidf.transform(_test_df["digest"].fillna(""))
    masker = shap.maskers.Independent(X_test_vec, max_samples=X_test_vec.shape[0])
    return build_explainer(_model, masker)


def compute_sector_sentiment(live_df: pd.DataFrame, tfidf, model) -> pd.Series:
    """Aggregate each sector's headlines into one pseudo-digest string before
    scoring, rather than scoring individual ~10-word headlines one at a time.

    The model was trained on daily digests of 25 concatenated headlines
    (~450 words). A single headline produces an extremely sparse TF-IDF
    vector (~99.8% zeros), so the model's baseline dominates and predicted
    probabilities compress into a narrow band regardless of content.
    Concatenating same-sector headlines into one longer digest per sector
    brings the input distribution closer to what the model was trained on.
    """
    sector_digests = live_df.groupby("sector")["title"].apply(
        lambda titles: " ".join(clean_headline(t) for t in titles)
    )
    X_sector = tfidf.transform(sector_digests)
    return pd.Series(
        model.predict_proba(X_sector)[:, 1], index=sector_digests.index
    ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def inject_custom_css():
    st.markdown(
        f"""
        <style>
        /* Typography */
        html, body, [class*="css"] {{
            font-family: "Inter", "Segoe UI", -apple-system, sans-serif;
        }}
        h1, h2, h3 {{
            letter-spacing: -0.01em;
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background: rgba(127, 127, 127, 0.06);
            border: 1px solid rgba(127, 127, 127, 0.15);
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
        }}

        /* Generic content cards (used for the headline feed) */
        .mp-card {{
            background: rgba(127, 127, 127, 0.05);
            border: 1px solid rgba(127, 127, 127, 0.15);
            border-radius: 14px;
            padding: 6px 4px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
        }}

        /* Sentiment badges */
        .mp-badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            white-space: nowrap;
        }}
        .mp-badge-bullish {{
            background: rgba(16, 185, 129, 0.14);
            color: {COLOR_BULLISH};
            border: 1px solid rgba(16, 185, 129, 0.35);
        }}
        .mp-badge-bearish {{
            background: rgba(244, 63, 94, 0.14);
            color: {COLOR_BEARISH};
            border: 1px solid rgba(244, 63, 94, 0.35);
        }}
        .mp-badge-neutral {{
            background: rgba(99, 102, 241, 0.14);
            color: {COLOR_NEUTRAL};
            border: 1px solid rgba(99, 102, 241, 0.35);
        }}

        /* Headline feed table */
        table.mp-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        table.mp-table th {{
            text-align: left;
            padding: 10px 14px;
            border-bottom: 1px solid rgba(127, 127, 127, 0.25);
            font-size: 0.72rem;
            color: {COLOR_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        table.mp-table td {{
            padding: 10px 14px;
            border-bottom: 1px solid rgba(127, 127, 127, 0.12);
            font-size: 0.88rem;
            vertical-align: middle;
        }}
        table.mp-table tr:last-child td {{
            border-bottom: none;
        }}
        .mp-source {{
            color: {COLOR_MUTED};
            font-size: 0.78rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sentiment_badge(prob: float) -> str:
    if prob > 0.55:
        return f'<span class="mp-badge mp-badge-bullish">Bullish</span>'
    if prob < 0.45:
        return f'<span class="mp-badge mp-badge-bearish">Bearish</span>'
    return f'<span class="mp-badge mp-badge-neutral">Neutral</span>'


# ---------------------------------------------------------------------------
# Chart builders (Plotly — replaces monochrome st.bar_chart)
# ---------------------------------------------------------------------------

def build_sector_sentiment_chart(sector_probs: pd.Series) -> go.Figure:
    colors = [COLOR_BULLISH if v >= 0.5 else COLOR_BEARISH for v in sector_probs.values]
    labels = [
        f"{v * 100:.1f}% {'Bullish' if v >= 0.5 else 'Bearish'}" for v in sector_probs.values
    ]

    fig = go.Figure(
        go.Bar(
            x=sector_probs.values,
            y=sector_probs.index,
            orientation="h",
            marker_color=colors,
            text=labels,
            textposition="outside",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color=COLOR_MUTED, opacity=0.6)
    fig.update_layout(
        title="Aggregated Sector Sentiment",
        xaxis=dict(title="Bullish probability", range=[0, 1], tickformat=".0%"),
        yaxis=dict(title=None, autorange="reversed"),
        height=110 + 55 * max(len(sector_probs), 1),
        margin=dict(l=10, r=60, t=40, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def build_sector_coverage_chart(sector_counts: pd.Series) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=sector_counts.values,
            y=sector_counts.index,
            orientation="h",
            marker_color=COLOR_NEUTRAL,
            text=sector_counts.values,
            textposition="outside",
            hovertemplate="%{y}: %{x} headlines<extra></extra>",
        )
    )
    fig.update_layout(
        title="Sector Coverage",
        xaxis=dict(title="Headlines parsed"),
        yaxis=dict(title=None, autorange="reversed"),
        height=110 + 55 * max(len(sector_counts), 1),
        margin=dict(l=10, r=40, t=40, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def render_headline_feed_table(display_df: pd.DataFrame):
    rows_html = []
    for _, row in display_df.iterrows():
        rows_html.append(
            "<tr>"
            f'<td class="mp-source">{html.escape(row["source"])}</td>'
            f'<td>{html.escape(row["sector"])}</td>'
            f'<td>{html.escape(row["title"])}</td>'
            f'<td>{sentiment_badge(row["bullish_prob"])}</td>'
            f'<td>{row["bullish_prob"]:.2f}</td>'
            "</tr>"
        )
    table_html = f"""
    <div class="mp-card">
    <table class="mp-table">
        <thead>
            <tr><th>Source</th><th>Sector</th><th>Headline</th><th>Signal</th><th>Prob.</th></tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def render_live_radar_tab(tfidf, model):
    st.header("📡 Live Market Radar")
    st.caption(
        "Sector sentiment is scored from a per-sector pseudo-digest (all of that "
        "sector's headlines concatenated), matching the ~450-word daily-digest "
        "format the model was trained on. Per-headline scores in the feed below "
        "are shown for reading, not for the sector aggregate — a single ~10-word "
        "headline is too sparse for the model to score meaningfully on its own."
    )

    live_df = fetch_live_headlines()
    if live_df.empty:
        st.info("No live headlines available right now — RSS feeds may be unreachable.")
        return

    live_df["sector"] = live_df["title"].apply(tag_sector)

    # Per-headline scores: informative for sorting/reading the feed, but not
    # used for the sector aggregate (see compute_sector_sentiment docstring).
    X_live = tfidf.transform(live_df["title"].apply(clean_headline))
    live_df["bullish_prob"] = model.predict_proba(X_live)[:, 1]

    sector_probs = compute_sector_sentiment(live_df, tfidf, model)
    sector_counts = live_df["sector"].value_counts()

    # --- Quick metric cards -------------------------------------------------
    most_bullish_sector = sector_probs.idxmax()
    most_active_sector = sector_counts.idxmax()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Headlines Parsed", len(live_df))
    m2.metric(
        "Most Bullish Sector", most_bullish_sector,
        f"{sector_probs.loc[most_bullish_sector] * 100:.1f}% bullish",
    )
    m3.metric(
        "Most Active Sector", most_active_sector,
        f"{sector_counts.loc[most_active_sector]} headlines",
    )

    st.write("")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(build_sector_sentiment_chart(sector_probs), width="stretch")
    with col2:
        st.plotly_chart(build_sector_coverage_chart(sector_counts), width="stretch")

    st.subheader("Headline Feed")
    display_df = live_df[["source", "sector", "title", "bullish_prob"]].sort_values(
        "bullish_prob", ascending=False
    )
    render_headline_feed_table(display_df)


def render_audit_explorer_tab(
    tfidf, model, test_df, sliced_df, global_tokens, metrics,
    precomputed_explainer, precomputed_shap_values,
):
    st.header("🔍 Historical Audit & SHAP Explorer")

    if metrics is not None and "model_selection" in metrics:
        selection = metrics["model_selection"]
        st.caption(
            f"Active model: **{selection['selected_model']}** "
            f"(selected on {selection['selection_metric']} across "
            f"{' vs '.join(selection['candidates'].keys())})"
        )

    st.subheader("Sliced Performance (Subgroup Audit)")
    if sliced_df is not None:
        st.dataframe(sliced_df, width="stretch")
    else:
        st.info("Run audit_metrics.py first to generate sliced_metrics.csv.")

    if global_tokens is not None:
        st.subheader("Global Token Drivers")
        st.caption(
            "Signed impact = mean SHAP value on days the token appears "
            "(direction). Mean |SHAP| = overall importance magnitude across "
            "all days. Averaging signed SHAP values over *all* days — "
            "including the many where a token is absent — mathematically "
            "washes every token's signed value to ~0, which is why direction "
            "is computed only over days where the token is present."
        )
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown("**Top Bullish tokens**")
            bullish_df = pd.DataFrame(global_tokens["top_bullish_tokens"]).head(10)
            st.dataframe(
                bullish_df.style.format(
                    {"signed_impact": "{:+.4f}", "mean_abs_shap": "{:.4f}"}
                ),
                width="stretch",
            )
        with gcol2:
            st.markdown("**Top Bearish tokens**")
            bearish_df = pd.DataFrame(global_tokens["top_bearish_tokens"]).head(10)
            st.dataframe(
                bearish_df.style.format(
                    {"signed_impact": "{:+.4f}", "mean_abs_shap": "{:.4f}"}
                ),
                width="stretch",
            )

    st.subheader("Inspect a Test-Split Date")
    date_options = test_df["Date"].dt.date.tolist()
    selected_date = st.selectbox("Select a date", date_options)

    row = test_df[test_df["Date"].dt.date == selected_date].iloc[0]
    row_position = test_df.index.get_loc(row.name)
    X_row = tfidf.transform([row["digest"]])
    pred = model.predict(X_row)[0]
    prob = model.predict_proba(X_row)[0, 1]

    m1, m2, m3 = st.columns(3)
    m1.metric("Actual", "Up" if row["Label"] == 1 else "Down")
    m2.metric("Predicted", "Up" if pred == 1 else "Down")
    m3.metric("Bullish probability", f"{prob:.2f}")

    if pred != row["Label"]:
        st.warning("⚠️ This is a misclassified day — a good candidate for local SHAP inspection.")

    with st.expander("Show headline digest for this date"):
        st.write(row["digest"])

    st.subheader("Local SHAP Attribution")
    if precomputed_shap_values is not None:
        # Fast path: index directly into audit_metrics.py's precomputed,
        # feature-name-tagged test-set SHAP values.
        shap_row = precomputed_shap_values[row_position]
    else:
        st.caption(
            "No precomputed SHAP values found (run audit_metrics.py to speed this up) "
            "— computing on the fly."
        )
        explainer = build_shap_explainer(model, tfidf, test_df)
        shap_row = explainer(to_explainer_input(X_row, explainer))
        if shap_row.values.ndim == 3:
            # RandomForestClassifier's TreeExplainer (and similar) returns
            # (n_samples, n_features, n_classes) even for binary
            # classification — keep the "Up" (class 1) output.
            shap_row = shap_row[:, :, 1]
        shap_row = attach_feature_names(shap_row, tfidf.get_feature_names_out())[0]

    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_row, max_display=15, show=False)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_sidebar():
    st.sidebar.title("MarketPulse")
    st.sidebar.caption("Financial Media Sentiment Auditing & Market Reaction Classification")
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ Model Card Summary"):
        st.markdown(
            """
            **Intended use:** educational/research demo of headline-driven
            sentiment signal extraction. **Not** an investment signal.

            **Training data:** DJIA headlines, 2008–2014.
            **Eval data:** strict temporal holdout, 2015–2016.

            **Known limitation:** bag-of-words digest loses headline order,
            source credibility, and intraday timing.
            """
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="MarketPulse", page_icon="📈", layout="wide")
    inject_custom_css()

    if not (ARTIFACTS_DIR / "tfidf_vectorizer.joblib").exists():
        st.error(
            f"No artifacts found at '{ARTIFACTS_DIR}'. Run train_pipeline.py "
            "(and optionally audit_metrics.py) first."
        )
        st.stop()

    (
        tfidf, model, test_df, sliced_df, global_tokens, metrics,
        precomputed_explainer, precomputed_shap_values,
    ) = load_artifacts(ARTIFACTS_DIR)

    render_sidebar()

    tab1, tab2 = st.tabs(["📡 Live Market Radar", "🔍 Historical Audit & SHAP Explorer"])
    with tab1:
        render_live_radar_tab(tfidf, model)
    with tab2:
        render_audit_explorer_tab(
            tfidf, model, test_df, sliced_df, global_tokens, metrics,
            precomputed_explainer, precomputed_shap_values,
        )


if __name__ == "__main__":
    main()
