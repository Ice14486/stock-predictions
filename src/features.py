"""Feature engineering for next-session AAPL direction."""

import pandas as pd

FEATURES = [
    "return_1d", "return_2d", "return_5d",
    "price_vs_ma5", "price_vs_ma20", "price_vs_ma50",
    "volatility_5", "volatility_20", "volume_change",
    "volume_vs_ma20", "high_low_range",
]

def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Use AAPL data through today's close to predict the next session.

    Target is 1 when the following session closes above its open.
    """
    df = prices.copy().sort_index()
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    daily_return = close.pct_change()

    df["return_1d"] = daily_return
    df["return_2d"] = close.pct_change(2)
    df["return_5d"] = close.pct_change(5)
    for window in (5, 20, 50):
        df[f"price_vs_ma{window}"] = close / close.rolling(window).mean() - 1
    df["volatility_5"] = daily_return.rolling(5).std()
    df["volatility_20"] = daily_return.rolling(20).std()
    df["volume_change"] = volume.pct_change()
    df["volume_vs_ma20"] = volume / volume.rolling(20).mean() - 1
    df["high_low_range"] = (df["High"] - df["Low"]) / close
    df["target"] = (df["Close"].shift(-1) > df["Open"].shift(-1)).astype("int8")

    # The last row has no next-session open/close target.
    return df.iloc[:-1].dropna(subset=FEATURES + ["target"])
