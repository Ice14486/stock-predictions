"""Small Streamlit demo for the stock direction project."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import statsmodels.api as sm
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import FEATURES, make_features


ROOT = Path(__file__).resolve().parent
TICKERS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "Meta Platforms (META)": "META",
    "NVIDIA (NVDA)": "NVDA",
    "Alphabet (GOOGL)": "GOOGL",
}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_prices(ticker: str) -> pd.DataFrame:
    """Load cached AAPL history or fetch daily adjusted prices from Yahoo Finance."""
    cache_path = ROOT / "data" / "raw" / "aapl_ohlcv.csv"
    if ticker == "AAPL" and cache_path.exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        prices = yf.download(
            ticker,
            start="2015-01-01",
            end="2026-01-01",
            auto_adjust=True,
            progress=False,
        )
        if prices.empty:
            raise ValueError(f"Yahoo Finance returned no data for {ticker}.")
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)
        prices = prices[["Open", "High", "Low", "Close", "Volume"]].dropna()
        prices.index = pd.to_datetime(prices.index)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.sort_index()


class MomentumRule:
    """Predict an up session when the trailing five-session return is positive."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p_up = (X["return_5d"].to_numpy() > 0).astype(float)
        return np.column_stack([1 - p_up, p_up])

    def predict(self, X):
        return (X["return_5d"].to_numpy() > 0).astype(int)


class StatsmodelsLogitClassifier:
    """Statsmodels Logit adapter using the notebook's compact feature subset."""

    feature_subset = [
        "return_1d", "return_5d", "price_vs_ma20", "volatility_20",
        "volume_change", "high_low_range",
    ]

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X[self.feature_subset])
        scaled = pd.DataFrame(
            self.scaler.transform(X[self.feature_subset]),
            columns=self.feature_subset,
            index=X.index,
        )
        self.result = sm.Logit(
            y, sm.add_constant(scaled, has_constant="add")
        ).fit(method="lbfgs", disp=False, maxiter=1000)
        return self

    def predict_proba(self, X):
        scaled = pd.DataFrame(
            self.scaler.transform(X[self.feature_subset]),
            columns=self.feature_subset,
            index=X.index,
        )
        p_up = self.result.predict(sm.add_constant(scaled, has_constant="add")).to_numpy()
        return np.column_stack([1 - p_up, p_up])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_models():
    """Return the same model set and main settings used in the notebook."""
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Momentum rule": MomentumRule(),
        "Statsmodels Logit": StatsmodelsLogitClassifier(),
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=42
            ),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=15,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }


def evaluate(model, X, y, threshold=0.5):
    """Evaluate a fitted classifier at a probability threshold."""
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y, probability) if y.nunique() == 2 else float("nan"),
    }


@st.cache_data(show_spinner="Training and comparing the notebook's models...")
def train_and_evaluate(
    prices: pd.DataFrame,
    split_mode: str,
    train_share: int,
    valid_share: int,
):
    """Compare notebook models with walk-forward selection and a held-out test."""
    data = make_features(prices)
    if split_mode == "Notebook year split":
        train = data.loc[data.index < "2023-01-01"]
        valid = data.loc[(data.index >= "2023-01-01") & (data.index < "2025-01-01")]
        test = data.loc[(data.index >= "2025-01-01") & (data.index < "2026-01-01")]
        # Purge the last signal in train and validation: its next-session target
        # would otherwise fall across the following split boundary.
        train, valid = train.iloc[:-1], valid.iloc[:-1]
    else:
        n_rows = len(data)
        train_end = int(n_rows * train_share / 100)
        valid_end = int(n_rows * (train_share + valid_share) / 100)
        train = data.iloc[: train_end - 1]
        valid = data.iloc[train_end : valid_end - 1]
        test = data.iloc[valid_end :]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Not enough data for the 2015–2025 train/validation/test periods.")

    X_train, y_train = train[FEATURES], train["target"].astype(int)
    X_valid, y_valid = valid[FEATURES], valid["target"].astype(int)
    X_test, y_test = test[FEATURES], test["target"].astype(int)

    # Fit each candidate on the full training period for validation and test
    # reporting, and separately score expanding-window folds for model selection.
    models = build_models()
    fit_errors = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            fit_errors[name] = str(exc)

    if len(X_train) <= 1000:
        raise ValueError("The training period is too short for four 252-session walk-forward folds.")
    tscv = TimeSeriesSplit(n_splits=4, test_size=252, gap=1)
    fold_rows = []
    fold_errors = {}
    for fold, (fit_idx, score_idx) in enumerate(tscv.split(X_train), start=1):
        for name in models:
            if name in fit_errors or name in fold_errors:
                continue
            fold_model = build_models()[name]
            try:
                fold_model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
                fold_rows.append({
                    "fold": fold,
                    "model": name,
                    **evaluate(fold_model, X_train.iloc[score_idx], y_train.iloc[score_idx]),
                })
            except Exception as exc:
                fold_errors[name] = str(exc)

    fold_metrics = pd.DataFrame(fold_rows)
    if fold_errors:
        fold_metrics = fold_metrics.loc[~fold_metrics["model"].isin(fold_errors)]
    if fold_metrics.empty:
        raise ValueError("No model completed walk-forward validation.")
    walk_forward = (
        fold_metrics.groupby("model")[["balanced_accuracy", "roc_auc"]]
        .mean()
        .sort_values(["balanced_accuracy", "roc_auc"], ascending=False)
    )
    chosen_name = walk_forward.index[0]

    # Pick a threshold on validation only for the selected model.
    chosen_model = models[chosen_name]
    valid_probability = chosen_model.predict_proba(X_valid)[:, 1]
    threshold_rows = []
    for threshold in np.linspace(0.10, 0.90, 161):
        prediction = (valid_probability >= threshold).astype(int)
        threshold_rows.append((
            balanced_accuracy_score(y_valid, prediction),
            abs(threshold - 0.5),
            float(threshold),
        ))
    validation_score, _, threshold = max(threshold_rows, key=lambda row: (row[0], -row[1]))

    test_rows = []
    test_predictions = {}
    for name, model in models.items():
        if name in fit_errors:
            continue
        test_rows.append({"model": name, **evaluate(model, X_test, y_test)})
        test_predictions[name] = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    test_results = pd.DataFrame(test_rows).set_index("model")
    selected_metrics = evaluate(chosen_model, X_test, y_test, threshold=threshold)
    test_results.loc[f"{chosen_name} (validation threshold)"] = selected_metrics
    chosen_prediction = (chosen_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    if fold_errors or fit_errors:
        error_messages = {**fit_errors, **fold_errors}
        error_table = pd.DataFrame(
            [{"model": name, "reason": reason} for name, reason in error_messages.items()]
        ).set_index("model")
    else:
        error_table = pd.DataFrame(columns=["reason"])
    split_summary = {
        "train": (train.index.min().date(), train.index.max().date(), len(train)),
        "validation": (valid.index.min().date(), valid.index.max().date(), len(valid)),
        "test": (test.index.min().date(), test.index.max().date(), len(test)),
    }
    return (
        chosen_name,
        test_results,
        walk_forward,
        fold_metrics,
        error_table,
        y_test,
        chosen_prediction,
        threshold,
        validation_score,
        split_summary,
    )


st.set_page_config(page_title="Stock Direction Explorer", page_icon="📈", layout="wide")
st.title("Stock Direction Explorer")
st.caption("A small interactive companion to the traditional machine-learning notebook.")

with st.sidebar:
    st.header("Explore a stock")
    label = st.selectbox("Stock", list(TICKERS))
    ticker = TICKERS[label]
    st.divider()
    st.header("Data split")
    split_mode = st.radio(
        "Choose split method",
        ["Notebook year split", "Custom chronological split"],
        help="Both options keep observations in date order. The test set is kept after training and validation.",
    )
    train_share, valid_share = 70, 20
    if split_mode == "Custom chronological split":
        train_share = st.slider("Training data", min_value=50, max_value=85, value=70, step=5, format="%d%%")
        max_valid_share = 95 - train_share
        valid_share = st.slider(
            "Validation data",
            min_value=5,
            max_value=max_valid_share,
            value=min(20, max_valid_share),
            step=5,
            format="%d%%",
            help="Used to select the probability threshold. Test data is the remaining share.",
        )
        st.caption(f"Test data: {100 - train_share - valid_share}%")
    st.caption("Price and volume history is sourced from Yahoo Finance. AAPL uses the project's cached data when available.")

try:
    prices = load_prices(ticker)
except Exception as exc:
    st.error(f"Could not load {ticker} data: {exc}")
    st.stop()

first_date, last_date = prices.index.min().date(), prices.index.max().date()
date_range = st.sidebar.date_input(
    "Chart date range",
    value=(max(first_date, last_date - pd.Timedelta(days=365 * 3)), last_date),
    min_value=first_date,
    max_value=last_date,
)
if len(date_range) == 2:
    chart_prices = prices.loc[str(date_range[0]) : str(date_range[1])]
else:
    chart_prices = prices

left, right = st.columns([3, 1])
with left:
    st.subheader(f"{ticker} adjusted closing price")
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=chart_prices.index, y=chart_prices["Close"], name="Close", line={"color": "#2563eb"}))
    price_fig.update_layout(height=390, margin={"l": 10, "r": 10, "t": 10, "b": 10}, xaxis_title=None, yaxis_title="Price (USD)", hovermode="x unified")
    st.plotly_chart(price_fig, use_container_width=True)
with right:
    st.subheader("Daily volume")
    volume_fig = go.Figure()
    volume_fig.add_trace(go.Bar(x=chart_prices.index, y=chart_prices["Volume"], name="Volume", marker_color="#60a5fa"))
    volume_fig.update_layout(height=390, margin={"l": 10, "r": 10, "t": 10, "b": 10}, xaxis_title=None, yaxis_title="Shares", showlegend=False)
    st.plotly_chart(volume_fig, use_container_width=True)

st.subheader("Next-session direction model")
if split_mode == "Notebook year split":
    st.caption("The app compares the notebook's candidate models. Train: through 2022 · model and threshold selection use walk-forward folds and 2023–2024 validation · final test: 2025.")
else:
    st.caption(f"Chronological split: {train_share}% train · {valid_share}% validation · {100 - train_share - valid_share}% test. Walk-forward folds select a model using training data; validation selects its probability threshold; the later test period is held out.")
try:
    (
        chosen_name,
        test_results,
        walk_forward,
        fold_metrics,
        error_table,
        actual,
        prediction,
        threshold,
        validation_score,
        split_summary,
    ) = train_and_evaluate(prices, split_mode, train_share, valid_share)
except Exception as exc:
    st.error(f"Could not evaluate the model: {exc}")
    st.stop()

st.write(f"**Selected by mean walk-forward balanced accuracy:** {chosen_name}")
selected_row = test_results.loc[f"{chosen_name} (validation threshold)"]
metric_cols = st.columns(4)
metric_cols[0].metric("Test balanced accuracy", f"{selected_row['balanced_accuracy']:.3f}")
metric_cols[1].metric("Test ROC-AUC", f"{selected_row['roc_auc']:.3f}")
metric_cols[2].metric("Validation threshold", f"{threshold:.3f}")
metric_cols[3].metric("Validation balanced accuracy", f"{validation_score:.3f}")
with st.expander("Show split date ranges and sample counts"):
    for split_name, (start, end, count) in split_summary.items():
        st.write(f"**{split_name.title()}:** {start} to {end} ({count:,} sessions)")

st.markdown("#### Walk-forward validation")
st.caption("Mean balanced accuracy and ROC-AUC across four expanding-window folds. This determines which model is selected before evaluating the test period.")
st.dataframe(walk_forward.round(3), use_container_width=True)

st.markdown("#### Held-out test results")
st.caption("Every candidate is shown at a 0.50 threshold; the selected model also has a row using its threshold chosen on validation data.")
st.dataframe(test_results.round(3), use_container_width=True)

with st.expander("Show individual walk-forward folds"):
    st.dataframe(
        fold_metrics.pivot(index="fold", columns="model", values="balanced_accuracy").round(3),
        use_container_width=True,
    )
if not error_table.empty:
    with st.expander("Models that could not be fit"):
        st.dataframe(error_table, use_container_width=True)

cm = confusion_matrix(actual, prediction, labels=[0, 1])
cm_fig = go.Figure(data=go.Heatmap(
    z=cm,
    x=["Predicted down/flat", "Predicted up"],
    y=["Actual down/flat", "Actual up"],
    text=cm,
    texttemplate="%{text}",
    colorscale="Blues",
    showscale=False,
))
cm_fig.update_layout(title=f"{chosen_name} at validation threshold: held-out test confusion matrix", height=340, margin={"l": 10, "r": 10, "t": 45, "b": 10})
st.plotly_chart(cm_fig, use_container_width=True)

st.info("The chronological splits leave a one-session gap at each boundary because each label uses the following session's open and close. Results are educational, not investment advice. Selecting the best model based on test metrics would make that test set no longer an independent evaluation. The notebook contains the detailed analysis and cost-aware backtest.")
