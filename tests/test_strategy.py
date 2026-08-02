#!/usr/bin/env python3
"""Assert-based unit tests for the strategy pipeline. Run: python tests/test_strategy.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantbacktest import fundamentals as fmod
from quantbacktest import signals
from quantbacktest import portfolio
from quantbacktest.config import BacktestConfig

passed = 0


def check(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print(f"  ok - {name}")


# ---------------------------------------------------------------------------
# Piotroski F-Score
# ---------------------------------------------------------------------------
def good_row():
    return {
        "net_income": 120.0, "gross_profit": 250.0, "total_revenue": 600.0,
        "total_assets": 1100.0, "stockholders_equity": 900.0, "long_term_debt": 90.0,
        "current_assets": 450.0, "current_liabilities": 210.0,
        "operating_cash_flow": 180.0, "ordinary_shares": 10.0,
    }


def prev_row():
    return {
        "net_income": 100.0, "gross_profit": 200.0, "total_revenue": 500.0,
        "total_assets": 1000.0, "stockholders_equity": 800.0, "long_term_debt": 100.0,
        "current_assets": 400.0, "current_liabilities": 200.0,
        "operating_cash_flow": 150.0, "ordinary_shares": 10.0,
    }


def test_fscore_all_nine():
    print("F-Score: all 9 flags")
    check("all-improving company scores 9", fmod.piotroski_fscore(good_row(), prev_row()) == 9)


def test_fscore_all_zero():
    print("F-Score: all 0 flags")
    cur = {
        "net_income": -80.0, "gross_profit": 100.0, "total_revenue": 600.0,
        "total_assets": 1500.0, "stockholders_equity": 500.0, "long_term_debt": 300.0,
        "current_assets": 200.0, "current_liabilities": 400.0,
        "operating_cash_flow": -200.0, "ordinary_shares": 20.0,
    }
    prev = {
        "net_income": -50.0, "gross_profit": 150.0, "total_revenue": 500.0,
        "total_assets": 1000.0, "stockholders_equity": 800.0, "long_term_debt": 100.0,
        "current_assets": 300.0, "current_liabilities": 300.0,
        "operating_cash_flow": -150.0, "ordinary_shares": 10.0,
    }
    check("deteriorating company scores 0", fmod.piotroski_fscore(cur, prev) == 0)


def test_fscore_flag_sensitivity():
    print("F-Score: single-flag sensitivity")
    cur = good_row()
    prev = prev_row()

    # 1. F_ROA: negative net income (also kills F_dROA -> drops 2 flags)
    c = dict(cur); c["net_income"] = -1.0
    check("flip F_ROA", fmod.piotroski_fscore(c, prev) == 7)

    # 2. F_CFO: negative operating cash flow (also kills F_ACCRUAL -> drops 2 flags)
    c = dict(cur); c["operating_cash_flow"] = -1.0
    check("flip F_CFO", fmod.piotroski_fscore(c, prev) == 7)

    # 3. F_dROA: ROA falls vs prev while still positive -> drops 1
    c = dict(cur); c["net_income"] = 100.0  # 100/1100 < 100/1000
    check("flip F_dROA", fmod.piotroski_fscore(c, prev) == 8)

    # 4. F_ACCRUAL: CFO below net income -> drops 1
    c = dict(cur); c["operating_cash_flow"] = 100.0
    check("flip F_ACCRUAL", fmod.piotroski_fscore(c, prev) == 8)

    # 5. F_dLEVER: leverage rises
    c = dict(cur); c["long_term_debt"] = 150.0
    check("flip F_dLEVER", fmod.piotroski_fscore(c, prev) == 8)

    # 6. F_dLIQUID: current ratio falls
    c = dict(cur); c["current_assets"] = 400.0
    check("flip F_dLIQUID", fmod.piotroski_fscore(c, prev) == 8)

    # 7. F_EQ: new shares issued
    c = dict(cur); c["ordinary_shares"] = 12.0
    check("flip F_EQ", fmod.piotroski_fscore(c, prev) == 8)

    # 8. F_dMARGIN: gross margin falls
    c = dict(cur); c["gross_profit"] = 220.0
    check("flip F_dMARGIN", fmod.piotroski_fscore(c, prev) == 8)

    # 9. F_dTURN: asset turnover falls (margin still improves at this revenue)
    c = dict(cur); c["total_revenue"] = 500.0
    check("flip F_dTURN", fmod.piotroski_fscore(c, prev) == 8)


def test_fscore_missing_data():
    print("F-Score: missing data is treated as flag 0")
    cur = {"net_income": 120.0}  # only net income -> ROA not computable
    check("mostly-missing row scores only F_ROA", fmod.piotroski_fscore(cur, prev_row()) == 1)


def test_book_to_market():
    print("Book-to-market")
    row = {"stockholders_equity": 900.0}
    bm = fmod.book_to_market(row, price=100.0, shares=10.0)
    check("BM = equity/(price*shares)", abs(bm - 0.9) < 1e-9)
    check("BM NaN when shares missing", pd.isna(fmod.book_to_market(row, price=100.0, shares=None)))


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def test_momentum_12_1():
    print("Signals: 12-1 momentum")
    prices = pd.DataFrame({"X": [100.0, 110.0, 121.0, 133.1, 146.41]})
    m = signals.momentum_12_1(prices, lookback_days=2, skip_days=1)
    # last row: P_{t-1}/P_{t-2} - 1 = 133.1/121 - 1
    check("12-1 momentum math", abs(m.iloc[-1, 0] - (133.1 / 121.0 - 1.0)) < 1e-9)
    check("first values NaN (need lookback)", pd.isna(m.iloc[0, 0]))


def test_tsmom():
    print("Signals: TSMOM sign")
    prices = pd.DataFrame({"X": [100.0, 110.0, 121.0, 133.1, 146.41]})
    ts = signals.tsmom(prices, lookback_days=2)
    check("TSMOM math", abs(ts.iloc[-1, 0] - (146.41 / 121.0 - 1.0)) < 1e-9)
    check("TSMOM positive sign", ts.iloc[-1, 0] > 0)


def test_annualized_vol():
    print("Signals: annualized volatility")
    # Constant growth -> std of returns is 0 -> vol 0; scaling factor sqrt(252).
    prices = pd.DataFrame({"X": 100.0 * (1.01 ** np.arange(60))})
    v = signals.annualized_vol(prices, window=50)
    check("constant-growth vol is 0", abs(v.iloc[-1, 0]) < 1e-9)
    # Alternating returns -> non-zero vol, equals std*sqrt(252)
    alt = pd.Series([100.0, 110.0, 100.0, 110.0, 100.0, 110.0, 100.0, 110.0])
    alt_p = pd.DataFrame({"X": alt})
    av = signals.annualized_vol(alt_p, window=4)
    expected = np.std(alt_p["X"].pct_change().dropna().tail(4), ddof=1) * np.sqrt(252)
    check("alternating-return vol scaling", abs(av.iloc[-1, 0] - expected) < 1e-9)


def test_excess_momentum():
    print("Signals: cross-sectional excess return")
    mom = pd.DataFrame({"A": [0.2], "B": [0.1], "C": [-0.1]})
    ex = signals.excess_momentum(mom, ["A", "B", "C"])
    check("excess == raw - EW mean", abs(ex.loc[0, "A"] - (0.2 - 0.2 / 3)) < 1e-9)


# ---------------------------------------------------------------------------
# Backtest engine on synthetic data
# ---------------------------------------------------------------------------
def _synthetic_config():
    return BacktestConfig(
        start_date="2023-01-01",
        end_date="2023-07-01",
        initial_capital=100000.0,
        transaction_cost=0.0,
        bm_top_quantile=1.0,          # keep all -> F-Score does the filtering here
        fscore_min=7, fscore_max=9,
        mom_lookback_days=60, mom_skip_days=5,
        top_decile=0.6,               # top 60% -> 2 of 3 names
        vol_lookback_days=60, vol_target=0.40,
        tsmom_lookback_days=60,
        fundamental_pub_lag_months=0,
        min_price_days=80, max_positions=30, min_positions=1,
    )


def _synthetic_prices():
    n = 300
    dates = pd.bdate_range(end="2023-06-30", periods=n)
    # A: strong steady uptrend 100 -> 200 (highest momentum)
    a = np.linspace(100.0, 200.0, n)
    # B: rises 100 -> 150 over first 200 days, then declines 150 -> 125 (TSMOM flips negative late)
    b = np.concatenate([np.linspace(100.0, 150.0, 200), np.linspace(150.0, 125.0, n - 200)])
    # C: mild uptrend 100 -> 130 (third in momentum)
    c = np.linspace(100.0, 130.0, n)
    # D: mild uptrend but fails F-Score (score 0)
    d = np.linspace(100.0, 130.0, n)
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=dates)


def _synthetic_fundamentals():
    rows = []
    fy1, fy2 = pd.Timestamp("2021-12-31"), pd.Timestamp("2022-12-31")
    for tk in ["A", "B", "C"]:  # F-Score 9 names
        rows += [
            {"ticker": tk, "fy_end": fy1, "net_income": 100.0, "gross_profit": 200.0,
             "total_revenue": 500.0, "total_assets": 1000.0, "stockholders_equity": 800.0,
             "long_term_debt": 100.0, "current_assets": 400.0, "current_liabilities": 200.0,
             "operating_cash_flow": 150.0, "ordinary_shares": 10.0},
            {"ticker": tk, "fy_end": fy2, "net_income": 120.0, "gross_profit": 250.0,
             "total_revenue": 600.0, "total_assets": 1100.0, "stockholders_equity": 900.0,
             "long_term_debt": 90.0, "current_assets": 450.0, "current_liabilities": 210.0,
             "operating_cash_flow": 180.0, "ordinary_shares": 10.0},
        ]
    # D: F-Score 0 (all flags fail)
    rows += [
        {"ticker": "D", "fy_end": fy1, "net_income": -50.0, "gross_profit": 150.0,
         "total_revenue": 500.0, "total_assets": 1000.0, "stockholders_equity": 800.0,
         "long_term_debt": 100.0, "current_assets": 300.0, "current_liabilities": 300.0,
         "operating_cash_flow": -150.0, "ordinary_shares": 10.0},
        {"ticker": "D", "fy_end": fy2, "net_income": -80.0, "gross_profit": 100.0,
         "total_revenue": 600.0, "total_assets": 1500.0, "stockholders_equity": 500.0,
         "long_term_debt": 300.0, "current_assets": 200.0, "current_liabilities": 400.0,
         "operating_cash_flow": -200.0, "ordinary_shares": 20.0},
    ]
    return pd.DataFrame(rows)


def test_engine_smoke():
    print("Engine: synthetic 4-stock universe")
    cfg = _synthetic_config()
    prices = _synthetic_prices()
    fund = _synthetic_fundamentals()
    res = portfolio.run_backtest(prices, fund, cfg)

    check("history recorded", len(res.history) >= 3)
    # A, B, C pass quality; D never enters.
    d_buys = [t for t in res.trades if t["ticker"] == "D" and t["side"] == "BUY"]
    check("D (F-Score 0) never bought", not d_buys)
    a_buys = [t for t in res.trades if t["ticker"] == "A" and t["side"] == "BUY"]
    check("A (F-Score 9) bought", len(a_buys) > 0)

    # Weights must sum to ~1 every rebalance with positions.
    for _, row in res.history.iterrows():
        if row["weights"]:
            check("weights sum to 1", abs(sum(row["weights"].values()) - 1.0) < 1e-9)

    # B's trend eventually breaks -> full sell occurs.
    b_sells = [t for t in res.trades if t["ticker"] == "B" and t["side"] == "SELL"]
    check("B trend-break sell triggered", len(b_sells) > 0)
    # Final holdings never include a negative-TSMOM name.
    ts = signals.tsmom(prices, cfg.tsmom_lookback_days)
    last_date = res.history.index[-1]
    check("final holdings all TSMOM-positive",
          all(ts.loc[last_date, tk] > 0 for tk in res.final_holdings))
    check("B not in final holdings", "B" not in res.final_holdings)

    # Funnel sanity
    last_funnel = res.funnel.iloc[-1]
    check("funnel quality count excludes D", last_funnel["n_quality"] <= 3)


if __name__ == "__main__":
    test_fscore_all_nine()
    test_fscore_all_zero()
    test_fscore_flag_sensitivity()
    test_fscore_missing_data()
    test_book_to_market()
    test_momentum_12_1()
    test_tsmom()
    test_annualized_vol()
    test_excess_momentum()
    test_engine_smoke()
    print(f"\nALL {passed} TESTS PASSED")
