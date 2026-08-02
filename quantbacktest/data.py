"""Data layer: universe, price history, fundamentals, and parquet caching."""
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

from .config import BacktestConfig

# --------------------------------------------------------------------------
# Field fallback lists (yfinance statement row names vary slightly by ticker)
# --------------------------------------------------------------------------
FIELDS = {
    "net_income": ["Net Income"],
    "gross_profit": ["Gross Profit"],
    "total_revenue": ["Total Revenue"],
    "total_assets": ["Total Assets"],
    "stockholders_equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    "long_term_debt": ["Long Term Debt And Capital Lease Obligation", "Long Term Debt"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "ordinary_shares": ["Ordinary Shares Number"],
}


def load_universe(universe: str = "nifty500") -> list:
    """Return a list of `.NS`-suffixed tickers for the requested universe."""
    if universe == "nifty500":
        from niftystocks import ns

        return ns.get_nifty500_with_ns()
    return [t.strip().upper() for t in universe.split(",") if t.strip()]


def financial_exclusion_set() -> set:
    """Nifty Financial Services index constituents (banks, NBFCs, insurers)."""
    try:
        from niftystocks import ns

        return set(ns.get_nifty_financial_services_with_ns())
    except Exception:
        return set()


def _ensure_ns(tickers) -> list:
    return [t if str(t).endswith(".NS") else str(t) + ".NS" for t in tickers]


def _cache_key(*parts) -> str:
    """Stable cache key that incorporates the ticker set (via hash) and dates."""
    tickers = parts[0]
    if isinstance(tickers, (list, tuple)):
        tk = "|".join(sorted(str(t) for t in tickers))
        tk_h = hashlib.md5(tk.encode()).hexdigest()[:10]
        rest = "_".join(str(p) for p in parts[1:])
        return f"{tk_h}_{rest}"
    return "_".join(str(p) for p in parts)


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
def _download_start(start_date: str, buffer_days: int = 500) -> str:
    return (pd.Timestamp(start_date) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def download_prices(tickers, start_date: str, end_date: str, cache_dir: str = "cache",
                    refresh: bool = False, universe_limit: int = None) -> pd.DataFrame:
    """Daily adjusted close for all tickers; cached to parquet. Tickers ordered consistently."""
    os.makedirs(cache_dir, exist_ok=True)
    tickers = _ensure_ns(tickers)
    if universe_limit:
        tickers = tickers[:universe_limit]
    key_start = _download_start(start_date)
    cache_path = os.path.join(cache_dir, f"prices_{_cache_key(tickers, key_start, end_date)}.parquet")
    if not refresh and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    print(f"Downloading price history for {len(tickers)} tickers ({key_start} to {end_date})...")
    data = yf.download(tickers, start=key_start, end=end_date, progress=False,
                       threads=True, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            close = data["Close"].copy()
        except KeyError:
            close = data.xs("Close", level=1, axis=1).copy()
    else:
        close = data.copy()
    # Retry tickers that the parallel batch call dropped (rate-limit / transient
    # failures), individually and sequentially.
    missing = [t for t in tickers
               if t not in close.columns or close[t].dropna().empty]
    if missing:
        print(f"Retrying {len(missing)} tickers individually: "
              f"{', '.join(missing[:6])}{'...' if len(missing) > 6 else ''}")
        time.sleep(1.0)
        for t in missing:
            try:
                s = yf.download(t, start=key_start, end=end_date, progress=False,
                                auto_adjust=True)
                if isinstance(s.columns, pd.MultiIndex):
                    s = s["Close"].iloc[:, 0]
                else:
                    s = s["Close"]
                s = s.dropna()
                if len(s) > 0:
                    close[t] = s
            except Exception:
                pass
    close = close.dropna(axis=1, how="all")
    close.columns = [str(c) for c in close.columns]
    close.sort_index(inplace=True)
    close.to_parquet(cache_path)
    print(f"Price data cached to {cache_path}")
    return close


def download_benchmark(symbol: str, start_date: str, end_date: str, cache_dir: str = "cache",
                       refresh: bool = False) -> pd.Series:
    os.makedirs(cache_dir, exist_ok=True)
    key_start = _download_start(start_date)
    cache_path = os.path.join(cache_dir, f"bench_{symbol.replace('^', '_')}_{key_start}_{end_date}.parquet")
    if not refresh and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)["value"]
    raw = yf.download(symbol, start=key_start, end=end_date, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        series = raw["Close"].iloc[:, 0]
    else:
        series = raw["Close"]
    series = series.dropna().sort_index()
    series.name = "value"
    series.to_frame().to_parquet(cache_path)
    return series


def clean_prices(prices: pd.DataFrame, min_price_days: int = 273) -> pd.DataFrame:
    """Drop tickers with too little history and forward-fill small gaps."""
    valid = prices.notna().sum()
    keep = valid[valid >= min_price_days].index
    cleaned = prices[keep].ffill()
    return cleaned.dropna(axis=1, how="any")


# --------------------------------------------------------------------------
# Fundamentals
# --------------------------------------------------------------------------
def _row_value(statement: pd.DataFrame, fy_end, names) -> float:
    """Pull a single field from an annual statement column, trying fallback names."""
    if statement is None or fy_end not in statement.columns:
        return np.nan
    for name in names:
        if name in statement.index:
            v = statement.loc[name, fy_end]
            try:
                if pd.notna(v):
                    return float(v)
            except (TypeError, ValueError):
                pass
    return np.nan


def _fetch_one(ticker: str):
    """Download annual fundamentals for one ticker -> list of rows (one per fiscal year end)."""
    t = yf.Ticker(ticker)
    inc = t.income_stmt
    bal = t.balance_sheet
    cf = t.cash_flow
    if inc is None or inc.empty:
        return []
    rows = []
    for fy_end in sorted(inc.columns):
        rows.append({
            "ticker": ticker,
            "fy_end": pd.Timestamp(fy_end),
            "net_income": _row_value(inc, fy_end, FIELDS["net_income"]),
            "gross_profit": _row_value(inc, fy_end, FIELDS["gross_profit"]),
            "total_revenue": _row_value(inc, fy_end, FIELDS["total_revenue"]),
            "total_assets": _row_value(bal, fy_end, FIELDS["total_assets"]),
            "stockholders_equity": _row_value(bal, fy_end, FIELDS["stockholders_equity"]),
            "long_term_debt": _row_value(bal, fy_end, FIELDS["long_term_debt"]),
            "current_assets": _row_value(bal, fy_end, FIELDS["current_assets"]),
            "current_liabilities": _row_value(bal, fy_end, FIELDS["current_liabilities"]),
            "operating_cash_flow": _row_value(cf, fy_end, FIELDS["operating_cash_flow"]),
            "ordinary_shares": _row_value(bal, fy_end, FIELDS["ordinary_shares"]),
        })
    return rows


def fetch_fundamentals(tickers, cache_dir: str = "cache", refresh: bool = False,
                       universe_limit: int = None, max_workers: int = 8) -> pd.DataFrame:
    """Long-form annual fundamentals panel. Cached to parquet."""
    os.makedirs(cache_dir, exist_ok=True)
    tickers = _ensure_ns(tickers)
    if universe_limit:
        tickers = tickers[:universe_limit]
    cache_path = os.path.join(cache_dir, f"fundamentals_{_cache_key(tickers)}.parquet")
    if not refresh and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    print(f"Downloading annual fundamentals for {len(tickers)} tickers (first run can take several minutes)...")
    all_rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_one, tk): tk for tk in tickers}
        for fut in as_completed(futs):
            tk = futs[fut]
            done += 1
            try:
                all_rows.extend(fut.result())
            except Exception as exc:
                print(f"  skip {tk}: {type(exc).__name__}")
            if done % 100 == 0:
                print(f"  ...{done}/{len(tickers)} fetched")
    if not all_rows:
        raise RuntimeError("No fundamentals could be downloaded.")
    panel = pd.DataFrame(all_rows)
    panel = panel.drop_duplicates(subset=["ticker", "fy_end"]).reset_index(drop=True)
    panel.to_parquet(cache_path)
    print(f"Fundamentals cached to {cache_path}")
    return panel


def filter_financials(fund_panel: pd.DataFrame, index_financials: set):
    """Drop financial firms: those on the Nifty Financial Services index or that
    never report Gross Profit (banks/insurers) across all available years."""
    no_gross = fund_panel.groupby("ticker")["gross_profit"].apply(lambda s: s.isna().all())
    heuristic = set(no_gross[no_gross].index)
    excluded = set(index_financials) | heuristic
    before = fund_panel["ticker"].nunique()
    kept = fund_panel[~fund_panel["ticker"].isin(excluded)].reset_index(drop=True)
    dropped = before - kept["ticker"].nunique()
    return kept, dropped
