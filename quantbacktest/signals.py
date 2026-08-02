"""Strategy signals: 12-1 momentum, cross-sectional rank, ex-ante volatility, TSMOM trend."""
import numpy as np
import pandas as pd


def momentum_12_1(prices: pd.DataFrame, lookback_days: int = 252, skip_days: int = 21) -> pd.DataFrame:
    """12-minus-1 month return: price 1 month ago vs price 12 months ago.

    r_i = P_{t-skip} / P_{t-lookback} - 1  (formation period ends 1 month before today).
    """
    return prices.shift(skip_days) / prices.shift(lookback_days) - 1.0


def tsmom(prices: pd.DataFrame, lookback_days: int = 252) -> pd.DataFrame:
    """12-month time-series momentum (trend sign): P_t / P_{t-252} - 1."""
    return prices / prices.shift(lookback_days) - 1.0


def annualized_vol(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Trailing ex-ante annualized volatility (daily std x sqrt(252))."""
    returns = prices.pct_change()
    return returns.rolling(window).std() * np.sqrt(252)


def excess_momentum(mom: pd.DataFrame, universe_subset) -> pd.DataFrame:
    """Cross-sectional excess return vs the equal-weighted mean of the universe."""
    subset = list(universe_subset)
    if not subset:
        return mom * np.nan
    ew = mom[subset].mean(axis=1)
    return mom[subset].sub(ew, axis=0)
