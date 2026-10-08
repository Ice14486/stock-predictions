# Stock Movement Prediction

A traditional machine-learning project that predicts whether AAPL’s next-session close is above that session’s open. It demonstrates Pandas, Scikit-learn, PyOD, and Statsmodels, with chronological validation and a costed backtest. It is an educational analysis, not investment advice.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab
```

Open `notebooks/stock_movement_prediction.ipynb` and run the cells from top to bottom. It uses the cached `data/raw/aapl_ohlcv.csv` if present; otherwise, it downloads AAPL data through yfinance.

## Interactive demo

The optional Streamlit app provides interactive price and volume charts for AAPL, MSFT, TSLA, META, NVDA, and GOOGL. It compares the notebook's Majority baseline, Momentum rule, Statsmodels Logit, Logistic Regression, Decision Tree, and Random Forest for the selected ticker. The default split matches the notebook's train (through 2022), validation (2023–2024), and test (2025) periods; an optional custom chronological split lets you set training and validation shares, with the test share using the remainder. Four expanding-window folds select the model, validation data selects its probability threshold, and held-out test metrics are reported for every candidate. A one-session gap is purged at split boundaries because each target uses the following session's open and close. The notebook remains the full analysis and cost-aware backtest.

Run it from the project root:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The AAPL view uses the local cached CSV when available. Other tickers are downloaded from Yahoo Finance when selected, so they require an internet connection.

## Method

- **Target:** 1 when the next trading session closes above its open; otherwise 0. This matches the strategy’s open-to-close return window.
- **Features:** 11 AAPL price, volume, momentum, and volatility features use information through the prediction date’s close.
- **Split:** train through 2022, validation in 2023–2024, final test in 2025.
- **Walk-forward validation:** four expanding-window folds, each about one year, with a one-session gap. The notebook shows both fold-level and mean metrics.
- **Hyperparameter tuning:** a bounded Scikit-learn `GridSearchCV` compares Random Forest `max_depth` values of 4, 6, and 8 with `min_samples_leaf` values of 10 and 15. It uses the same chronological, training-only folds and balanced accuracy scoring; the 2025 test period is excluded from tuning.
- **Tuned model check:** the best training-fold Random Forest is refit on all training data, its probability threshold is selected on 2023–2024 validation, and it is reported as a separate challenger on the final test set. The existing walk-forward-selected model remains the one used in the backtest.
- **Model selection:** mean fold balanced accuracy selects the model; its probability threshold is chosen on validation only. The 2025 test set is not used to select either.
- **Models:** majority baseline, five-session momentum rule, Scikit-learn Logistic Regression/Decision Tree/Random Forest, and Statsmodels Logit.
- **Diagnostics:** Statsmodels ADF and Ljung–Box; PyOD ECOD anomaly detection fit on training features.
- **Backtest:** signal after close, enter next open, exit next close, and charge 5 bps per side. Close-to-close buy-and-hold is shown separately because it includes overnight exposure.

## Current AAPL-only results

In the previously recorded AAPL run, Random Forest was selected in walk-forward validation (mean fold balanced accuracy 0.516 versus 0.500 for the majority baseline). At the threshold selected on 2023–2024 validation, it scored 0.505 balanced accuracy and 0.512 ROC-AUC on the 2025 test set. The costed long/cash open-to-close strategy returned +5.9%, versus -0.7% for always-long open-to-close and +11.9% for close-to-close buy-and-hold. These results were recorded before adding the GridSearchCV challenger; rerun the notebook to produce its metrics. The original classifier was only slightly above chance and its strategy did not outperform buy-and-hold, so those results do not establish a reliable predictive or trading edge.

## Limitations

- Trading costs are fixed at 5 bps per side; actual slippage, taxes, market impact, and financing can be higher.
- PyOD flagged only a handful of test days, too few to support conclusions about anomaly outcomes.
- Features use historical AAPL OHLCV only; there are no fundamentals, news, or macroeconomic variables.
- Model and threshold selection can overfit the folds and validation period. Reserve future data for another independent check.
