#!/usr/bin/env python3
"""Jegadeesh & Titman (1993) momentum strategies for NSE equities.

Implements the two core models from the paper "Returns to Buying Winners and
Selling Losers: Implications for Stock Market Efficiency":

  Model A - J-month / K-month decile strategy
      formation_return_t = P[t-1] / P[t-1-J] - 1      (J=12 -> "12-minus-1")
      Winners = top decile by formation return (long), Losers = bottom decile
      (short). Positions formed at month t are held for K months with
      overlapping monthly rebalancing: at any month s the portfolio averages
      the 1/K slices formed in each of the past K months (JT eq. portfolio).

  Model B - Weighted Relative Strength Strategy (WRSS)
      w_it = r_i,t-1 - rbar_t-1          (deviation from equal-weight market mean)
      Long stocks with positive deviation, short stocks with negative deviation.

Risk overlay / constraints:
  - 200-day Simple Moving Average trend filter: long positions only in a
    confirmed structural uptrend (price above SMA-200).
  - Position cap: at most `max_stocks` active long holdings.
  - Transaction costs applied on monthly weight turnover.

Performance analytics:
  - Final growth, absolute return, CAGR, Sharpe, max drawdown.
  - Market-model regression of strategy returns on the benchmark index to
    extract alpha and beta (Jensen's alpha), via OLS.
"""
import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantbacktest import data  # reuse universe + price download/caching


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class MomentumConfig:
    universe: str = "nifty500"
    start_date: str = "2021-01-01"
    end_date: str = "2026-01-01"
    initial_capital: float = 400000.0
    benchmark: str = "^NSEI"

    model: str = "A"            # "A" = decile strategy, "B" = WRSS
    j_months: int = 12          # formation period J
    k_months: int = 3           # holding period K
    decile_frac: float = 0.10   # top/bottom fraction for Model A
    long_short: bool = False    # False = long winners only (ignore losers)
    max_stocks: int = 6         # active long-holding cap
    sma_trend_filter: bool = True   # 200-day SMA overlay on long legs
    sma_window: int = 200

    transaction_cost: float = 0.0015
    min_price_days: int = 273
    cache_dir: str = "cache"
    universe_limit: Optional[int] = None  # smoke-testing only


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_monthly(universe: str, start_date: str, end_date: str, cache_dir: str,
                 min_price_days: int, universe_limit: Optional[int] = None,
                 sma_window: int = 200):
    """Return (monthly_close, monthly_ret, monthly_sma) aligned DataFrames."""
    tickers = data.load_universe(universe)
    if universe_limit:
        tickers = tickers[:universe_limit]
    daily = data.download_prices(tickers, start_date, end_date,
                                 cache_dir=cache_dir, universe_limit=universe_limit)
    daily = data.clean_prices(daily, min_price_days)

    sma = daily.rolling(sma_window).mean()
    monthly_close = daily.resample("ME").last()
    monthly_sma = sma.resample("ME").last()
    monthly_ret = monthly_close.pct_change()
    return monthly_close, monthly_ret, monthly_sma


def formation_returns(monthly_close: pd.DataFrame, j: int) -> pd.DataFrame:
    """J-month formation return ending one month before the current month.

    formula: f_t = P[t-1] / P[t-1-J] - 1   (12-minus-1 rule when J=12)
    """
    return monthly_close.shift(1) / monthly_close.shift(1 + j) - 1.0


# ---------------------------------------------------------------------------
# Strategy engine
# ---------------------------------------------------------------------------
@dataclass
class MomentumResult:
    config: MomentumConfig
    monthly_returns: pd.Series
    equity: pd.Series
    weights: pd.DataFrame
    formation_weights: Dict[pd.Timestamp, Dict[str, float]] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def _top_decile_sets(formation_ret: pd.Series, decile_frac: float):
    """Return (winners, losers) ticker lists by cross-sectional momentum rank."""
    valid = formation_ret.dropna()
    if valid.empty:
        return [], []
    valid = valid.sort_values(ascending=False)
    n = len(valid)
    n_w = max(1, int(round(n * decile_frac)))
    return list(valid.index[:n_w]), list(valid.index[-n_w:])


def _wrss_weights(formation_ret: pd.Series):
    """WRSS weights: w_it = r_it - rbar_t (deviation from EW market mean)."""
    valid = formation_ret.dropna()
    if valid.empty:
        return {}
    rbar = valid.mean()
    return (valid - rbar).to_dict()


def run_momentum(monthly_close: pd.DataFrame, monthly_ret: pd.DataFrame,
                 monthly_sma: pd.DataFrame, cfg: MomentumConfig) -> MomentumResult:
    """Run the JT (1993) momentum strategy.

    monthly_close/ret : monthly last-close and simple returns (columns = tickers)
    monthly_sma       : monthly-end 200-day SMA for the trend overlay
    """
    formation = formation_returns(monthly_close, cfg.j_months)
    months = monthly_close.index
    # Trend filter must use only data known at the START of formation month m,
    # i.e. end of month m-1 -> shift close and SMA by one month (no lookahead).
    prev_close = monthly_close.shift(1)
    prev_sma = monthly_sma.shift(1)
    valid_formation_idx = formation.index[formation.notna().any(axis=1)]

    # --- Build overlapping formation portfolios (Model A or B) ---
    # formation_weights[m] = {ticker: weight} ; long leg sums to 1, short leg
    # stored under special key "__SHORT__" as {ticker: weight} (sums to 1).
    formation_weights: Dict[pd.Timestamp, Dict] = {}
    for m in valid_formation_idx:
        frow = formation.loc[m]
        trend_ok = None
        if cfg.sma_trend_filter and m in prev_sma.index:
            trend_ok = prev_sma.loc[m]

        if cfg.model.upper() == "A":
            winners, losers = _top_decile_sets(frow, cfg.decile_frac)
            # SMA trend filter on long leg + max-stocks cap (concentrated growth)
            if cfg.sma_trend_filter:
                winners = [t for t in winners if trend_ok is not None
                           and t in prev_close.columns
                           and t in trend_ok.index
                           and not pd.isna(trend_ok[t])
                           and prev_close.loc[m, t] > trend_ok[t]]
            winners = winners[: cfg.max_stocks]
            w = {t: 1.0 / len(winners) for t in winners} if winners else {}
            formation_weights[m] = {"__LONG__": w, "__SHORT__": {}}
            if cfg.long_short and losers:
                formation_weights[m]["__SHORT__"] = {t: 1.0 / len(losers) for t in losers}
        else:  # Model B: WRSS
            raw = _wrss_weights(frow)
            longs = {t: v for t, v in raw.items() if v > 0 and not pd.isna(prev_close.loc[m, t])}
            shorts = {t: -v for t, v in raw.items() if v < 0}
            if cfg.sma_trend_filter:
                longs = {t: v for t, v in longs.items()
                         if trend_ok is not None and not pd.isna(trend_ok.get(t, np.nan))
                         and prev_close.loc[m, t] > trend_ok[t]}
            if cfg.max_stocks:
                longs = dict(sorted(longs.items(), key=lambda kv: kv[1], reverse=True)[: cfg.max_stocks])
            ls = sum(longs.values())
            longs = {t: v / ls for t, v in longs.items()} if ls > 0 else {}
            ss = sum(shorts.values())
            shorts = {t: v / ss for t, v in shorts.items()} if ss > 0 else {}
            formation_weights[m] = {"__LONG__": longs, "__SHORT__": shorts
                                    if cfg.long_short else {}}

    # --- Monthly portfolio return via overlapping K-month holding ---
    monthly_returns = pd.Series(index=months, dtype=float)
    weight_history = {}
    pos = {m: i for i, m in enumerate(months)}
    formation_months = sorted(formation_weights.keys())
    prev_weights = None
    for s in months:
        if s not in pos:
            continue
        # formation windows m covering month s satisfy pos[m] <= pos[s] < pos[m]+K
        windows = [m for m in formation_months
                   if pos[m] <= pos[s] < pos[m] + cfg.k_months]
        if not windows:
            continue
        # target weights = average of the 1/K slices from each covering window,
        # then normalized to be fully invested (long leg sums to 1 each month)
        agg_long, agg_short = {}, {}
        for m in windows:
            for t, w in formation_weights[m]["__LONG__"].items():
                agg_long[t] = agg_long.get(t, 0.0) + w / cfg.k_months
            for t, w in formation_weights[m]["__SHORT__"].items():
                agg_short[t] = agg_short.get(t, 0.0) + w / cfg.k_months
        sl = sum(agg_long.values())
        if sl > 0:
            agg_long = {t: w / sl for t, w in agg_long.items()}
        ss = sum(agg_short.values())
        if ss > 0:
            agg_short = {t: w / ss for t, w in agg_short.items()}
        # cap ACTIVE holdings to max_stocks (aggregate of overlapping slices)
        if cfg.max_stocks and len(agg_long) > cfg.max_stocks:
            top = sorted(agg_long.items(), key=lambda kv: kv[1], reverse=True)[: cfg.max_stocks]
            agg_long = dict(top)
            sl = sum(agg_long.values())
            agg_long = {t: w / sl for t, w in agg_long.items()}

        # return of the aggregate portfolio in month s
        ret = 0.0
        for t, w in agg_long.items():
            r = monthly_ret.loc[s, t]
            if not pd.isna(r):
                ret += w * r
        for t, w in agg_short.items():
            r = monthly_ret.loc[s, t]
            if not pd.isna(r):
                ret -= w * r
        # transaction cost on monthly weight turnover (buys + sells)
        if cfg.transaction_cost > 0 and prev_weights is not None:
            all_tickers = set(agg_long) | set(agg_short) | set(prev_weights)
            turnover = sum(abs(agg_long.get(t, 0.0) + agg_short.get(t, 0.0)
                               - prev_weights.get(t, 0.0)) for t in all_tickers)
            ret -= cfg.transaction_cost * turnover
        prev_weights = {t: agg_long.get(t, 0.0) + agg_short.get(t, 0.0) for t in agg_long}
        prev_weights.update({t: -agg_short.get(t, 0.0) for t in agg_short})
        monthly_returns.loc[s] = ret
        weight_history[s] = agg_long

    monthly_returns = monthly_returns.dropna()
    equity = cfg.initial_capital * (1 + monthly_returns).cumprod()
    weight_df = pd.DataFrame(weight_history).T.fillna(0.0)
    return MomentumResult(cfg, monthly_returns, equity, weight_df, formation_weights)


# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------
def performance_stats(result: MomentumResult, bench_monthly: pd.Series = None) -> dict:
    r = result.monthly_returns
    eq = result.equity
    cfg = result.config

    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    stats = {
        "final_value": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / cfg.initial_capital - 1.0),
        "cagr": float((eq.iloc[-1] / cfg.initial_capital) ** (1 / years) - 1.0),
        "annualized_vol": float(r.std() * np.sqrt(12)),
        "sharpe": float((r.mean() * 12) / (r.std() * np.sqrt(12))) if r.std() > 0 else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1.0).min()),
        "n_months": int(len(r)),
        "avg_long_positions": float(result.weights.gt(0).sum(axis=1).mean()),
    }

    # market-model regression: r_p = alpha + beta * r_m (+ eps)
    if bench_monthly is not None and len(bench_monthly) > 2:
        joined = pd.concat([r.rename("strat"), bench_monthly.rename("bench")],
                           axis=1, join="outer", sort=True).dropna()
        if len(joined) > 2:
            y = joined["strat"].values
            X = np.column_stack([np.ones(len(joined)), joined["bench"].values])
            coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coef
            n, k = X.shape
            sigma2 = resid @ resid / (n - k)
            cov = sigma2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            t_stats = coef / se
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1.0 - (resid @ resid) / ss_tot if ss_tot > 0 else 0.0
            stats["alpha_monthly"] = float(coef[0])
            stats["alpha_annualized"] = float(coef[0] * 12)
            stats["alpha_tstat"] = float(t_stats[0])
            stats["beta"] = float(coef[1])
            stats["beta_tstat"] = float(t_stats[1])
            stats["r2"] = float(r2)
            stats["bench_total_return"] = float(
                joined["bench"].add(1).prod() - 1.0)
            stats["excess_return"] = stats["total_return"] - stats["bench_total_return"]
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Jegadeesh & Titman (1993) momentum backtest (NSE).")
    p.add_argument("--model", default="A", choices=["A", "B"], help="A=decile, B=WRSS")
    p.add_argument("--j", type=int, default=12, help="Formation period J (months).")
    p.add_argument("--k", type=int, default=3, help="Holding period K (months).")
    p.add_argument("--capital", type=float, default=400000.0)
    p.add_argument("--max-stocks", type=int, default=6)
    p.add_argument("--long-short", action="store_true", help="Short bottom decile too.")
    p.add_argument("--no-sma", action="store_true", help="Disable 200-day SMA trend filter.")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--universe-limit", type=int, default=None)
    p.add_argument("--outdir", default="output")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MomentumConfig(model=args.model, j_months=args.j, k_months=args.k,
                         initial_capital=args.capital, max_stocks=args.max_stocks,
                         long_short=args.long_short,
                         sma_trend_filter=not args.no_sma,
                         start_date=args.start, end_date=args.end,
                         universe_limit=args.universe_limit)
    print(f"Model {cfg.model} | J={cfg.j_months} K={cfg.k_months} | max {cfg.max_stocks} stocks | "
          f"SMA filter: {cfg.sma_trend_filter} | long-short: {cfg.long_short}")

    monthly_close, monthly_ret, monthly_sma = load_monthly(
        cfg.universe, cfg.start_date, cfg.end_date, cfg.cache_dir,
        cfg.min_price_days, cfg.universe_limit)
    print(f"Loaded {monthly_close.shape[1]} stocks, {len(monthly_close)} months.")

    result = run_momentum(monthly_close, monthly_ret, monthly_sma, cfg)

    bench = data.download_benchmark(cfg.benchmark, cfg.start_date, cfg.end_date, cfg.cache_dir)
    bench = bench.resample("ME").last().pct_change()
    stats = performance_stats(result, bench)
    result.stats = stats

    print("\n=== JEGADEESH & TITMAN (1993) MOMENTUM BACKTEST ===")
    print(f"Final Portfolio Value : Rs {stats['final_value']:,.2f}")
    print(f"Total Return          : {stats['total_return']*100:.2f}%")
    print(f"CAGR                  : {stats['cagr']*100:.2f}%")
    print(f"Annualized Volatility : {stats['annualized_vol']*100:.2f}%")
    print(f"Sharpe                : {stats['sharpe']:.2f}")
    print(f"Max Drawdown          : {stats['max_drawdown']*100:.2f}%")
    print(f"Avg Long Positions    : {stats['avg_long_positions']:.1f}")
    if "alpha_annualized" in stats:
        print(f"\nMarket model vs {cfg.benchmark}:")
        print(f"  Alpha (annualized) : {stats['alpha_annualized']*100:.2f}%  (t={stats['alpha_tstat']:.2f})")
        print(f"  Beta               : {stats['beta']:.2f}  (t={stats['beta_tstat']:.2f})")
        print(f"  R-squared          : {stats['r2']:.3f}")
        print(f"  Benchmark return   : {stats['bench_total_return']*100:.2f}%")
        print(f"  Excess vs bench    : {stats['excess_return']*100:.2f}%")

    os.makedirs(args.outdir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eq_norm = result.equity / cfg.initial_capital
    bench_norm = bench.cumsum().add(1).loc[result.equity.index[0]:].dropna()
    bench_norm = bench_norm / bench_norm.iloc[0]
    plt.figure(figsize=(12, 6))
    plt.plot(eq_norm.index, eq_norm.values, label=f"Momentum Model {cfg.model} (J={cfg.j_months},K={cfg.k_months})")
    plt.plot(bench_norm.index, bench_norm.values, label=cfg.benchmark, alpha=0.8)
    plt.axhline(1.0, color="gray", ls="--", lw=1)
    plt.title("JT 1993 Momentum: Normalized Growth vs Benchmark")
    plt.xlabel("Date"); plt.ylabel("Growth (1.0 = initial capital)")
    plt.legend(); plt.grid(alpha=0.3)
    chart = os.path.join(args.outdir, "momentum_equity.png")
    plt.savefig(chart, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nChart saved to {chart}")
    result.equity.to_frame("equity").to_csv(os.path.join(args.outdir, "momentum_monthly.csv"))
    return stats


if __name__ == "__main__":
    main()
