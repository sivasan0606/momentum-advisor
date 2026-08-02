#!/usr/bin/env python3
"""Assert-based tests for the Jegadeesh & Titman momentum engine.
Run: python tests/test_momentum.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momentum_engine import (MomentumConfig, formation_returns, run_momentum,
                             performance_stats, _top_decile_sets, _wrss_weights)

passed = 0


def check(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print(f"  ok - {name}")


# ---------------------------------------------------------------------------
# Synthetic monthly data: 2 winners, 1 flat, 2 losers
# ---------------------------------------------------------------------------
def _mk_data(n_months=60):
    dates = pd.date_range(end="2023-12-31", periods=n_months, freq="ME")
    rets = {"W1": 0.05, "W2": 0.04, "M": 0.0, "L1": -0.05, "L2": -0.04}
    close = pd.DataFrame(index=dates)
    for tk, r in rets.items():
        close[tk] = 100.0 * np.cumprod(np.full(n_months, 1 + r))
    mret = close.pct_change()
    msma = close.rolling(5, min_periods=1).mean()
    return close, mret, msma


def test_formation_returns():
    print("Formation returns: 12-minus-1")
    close, _, _ = _mk_data()
    f = formation_returns(close, 12)
    # at month t: P[t-1]/P[t-13] - 1 for W1 = (1.05^12 - 1)
    last = f.index[-1]
    idx = f.index.get_loc(last) - 1  # formation uses t-1 vs t-13
    p_t1 = close["W1"].iloc[idx]
    p_t13 = close["W1"].iloc[idx - 12]
    check("W1 formation return math", abs(f["W1"].iloc[idx] - (p_t1 / p_t13 - 1)) < 1e-9)
    check("W1 momentum > M > L1 momentum",
          f["W1"].iloc[idx] > f["M"].iloc[idx] > f["L1"].iloc[idx])


def test_decile_selection():
    print("Decile selection")
    s = pd.Series({"A": 0.9, "B": 0.6, "C": 0.3, "D": -0.2, "E": -0.5})
    win, lose = _top_decile_sets(s, 0.1)
    check("winners = top decile", win == ["A"])
    check("losers = bottom decile", lose == ["E"])


def test_wrss_weights():
    print("WRSS weights = deviation from EW mean")
    s = pd.Series({"A": 0.1, "B": 0.0, "C": -0.1})
    w = _wrss_weights(s)
    rbar = s.mean()
    check("WRSS A weight", abs(w["A"] - (0.1 - rbar)) < 1e-12)
    check("WRSS C weight", abs(w["C"] - (-0.1 - rbar)) < 1e-12)


def test_model_a_long_only():
    print("Model A long-only: buys winners, equity grows")
    close, mret, msma = _mk_data()
    cfg = MomentumConfig(model="A", j_months=12, k_months=3, max_stocks=2,
                         sma_trend_filter=False, transaction_cost=0.0)
    res = run_momentum(close, mret, msma, cfg)
    stats = performance_stats(res)
    check("equity grew (W1 rises 5%/mo)", stats["total_return"] > 0.2)
    # fully invested in the top-2 winners W1, W2 each month
    first = res.weights.iloc[0]
    check("longs are winners W1,W2",
          set(first[first > 0].index) <= {"W1", "W2"})
    check("long weights sum to 1", abs(first.sum() - 1.0) < 1e-9)


def test_max_stocks_cap():
    print("Max-stocks cap")
    close, mret, msma = _mk_data()
    # wide decile so >2 names qualify, then the cap does the limiting
    base = dict(model="A", j_months=12, k_months=3, decile_frac=0.6,
                sma_trend_filter=False, transaction_cost=0.0)
    cfg1 = MomentumConfig(**base, max_stocks=1)
    r1 = run_momentum(close, mret, msma, cfg1)
    check("cap=1 -> single long (W1)", r1.weights.gt(0).sum(axis=1).max() == 1)
    cfg2 = MomentumConfig(**base, max_stocks=2)
    r2 = run_momentum(close, mret, msma, cfg2)
    check("cap=2 -> up to 2 longs", r2.weights.gt(0).sum(axis=1).max() == 2)


def test_sma_trend_filter():
    print("200-day SMA trend filter excludes below-trend names")
    close, mret, msma = _mk_data()
    cfg = MomentumConfig(model="A", j_months=12, k_months=3, max_stocks=10,
                         sma_trend_filter=True, transaction_cost=0.0)
    res = run_momentum(close, mret, msma, cfg)
    # with a rising-SMA proxy, only W1/W2 are above their trailing mean; M is exactly
    # equal and L1/L2 are below -> M and L names must never appear in longs
    check("below/at-trend names never long",
          not (res.weights.columns.isin(["M", "L1", "L2"]).any()))


def test_model_a_long_short():
    print("Model A long-short: losers shorted")
    close, mret, msma = _mk_data()
    cfg = MomentumConfig(model="A", j_months=12, k_months=3, max_stocks=2,
                         long_short=True, sma_trend_filter=False, transaction_cost=0.0)
    res = run_momentum(close, mret, msma, cfg)
    # every formation has a short leg containing the worst loser
    has_short = all(res.formation_weights[m]["__SHORT__"] for m in res.formation_weights)
    check("short leg populated for all formations", has_short)


def test_overlapping_holding_K():
    print("Overlapping holding: K=3 holds each formation 3 months")
    close, mret, msma = _mk_data(n_months=40)
    cfg = MomentumConfig(model="A", j_months=12, k_months=3, max_stocks=1,
                         sma_trend_filter=False, transaction_cost=0.0)
    res = run_momentum(close, mret, msma, cfg)
    # W1 is held at full weight for K=3 months then rotates; avg position ~1
    check("avg long positions ~1 (single-name overlap)",
          abs(res.weights.gt(0).sum(axis=1).mean() - 1.0) < 1e-6)


def test_alpha_beta_regression():
    print("Market-model regression: beta and alpha")
    close, mret, msma = _mk_data(n_months=40)
    cfg = MomentumConfig(model="A", j_months=12, k_months=3, max_stocks=1,
                         sma_trend_filter=False, transaction_cost=0.0)
    res = run_momentum(close, mret, msma, cfg)
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0.01, 0.03, len(res.monthly_returns)),
                      index=res.monthly_returns.index)
    # strategy returns = 2x benchmark + tiny noise -> beta ~ 2, alpha ~ 0
    res.monthly_returns = 2.0 * bench + rng.normal(0, 1e-4, len(bench))
    stats = performance_stats(res, bench)
    check("beta ~ 2", abs(stats["beta"] - 2.0) < 0.05)
    check("alpha ~ 0 (annualized)", abs(stats["alpha_annualized"]) < 0.01)
    check("r2 near 1", stats["r2"] > 0.99)


if __name__ == "__main__":
    test_formation_returns()
    test_decile_selection()
    test_wrss_weights()
    test_model_a_long_only()
    test_max_stocks_cap()
    test_sma_trend_filter()
    test_model_a_long_short()
    test_overlapping_holding_K()
    test_alpha_beta_regression()
    print(f"\nALL {passed} TESTS PASSED")
