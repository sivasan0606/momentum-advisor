"""Streamlit app: Jegadeesh & Titman (1993) momentum backtester for NSE.

Run with:  streamlit run app.py

Toggles Model A (J-month/K-month decile) and Model B (Weighted Relative
Strength, WRSS), the 200-day SMA trend overlay, long-only vs long-short, and
a max-stock concentration cap. Reports growth, CAGR, Sharpe, max drawdown,
and market-model alpha/beta against the NIFTY 50 index.
"""
import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from momentum_engine import (MomentumConfig, load_monthly, run_momentum,
                             performance_stats)
from quantbacktest import data as qdata

st.set_page_config(page_title="JT 1993 Momentum Backtester (NSE)", layout="wide")


@st.cache_data(show_spinner=False, ttl=3600)
def load_prices(start, end, universe_limit):
    tickers = qdata.load_universe("nifty500")
    if universe_limit:
        tickers = tickers[:universe_limit]
    daily = qdata.download_prices(tickers, start, end, cache_dir="cache",
                                  universe_limit=universe_limit)
    daily = qdata.clean_prices(daily, 273)
    return daily


@st.cache_data(show_spinner=False, ttl=3600)
def load_bench(start, end):
    b = qdata.download_benchmark("^NSEI", start, end, cache_dir="cache")
    return b


def compute(start, end, capital, model, j, k, max_stocks, long_short,
            sma_filter, cost, universe_limit):
    daily = load_prices(start, end, universe_limit)
    monthly_close = daily.resample("ME").last()
    monthly_sma = daily.rolling(200).mean().resample("ME").last()
    monthly_ret = monthly_close.pct_change()
    cfg = MomentumConfig(model=model, j_months=j, k_months=k,
                         initial_capital=capital, max_stocks=max_stocks,
                         long_short=long_short, sma_trend_filter=sma_filter,
                         transaction_cost=cost, start_date=start, end_date=end)
    result = run_momentum(monthly_close, monthly_ret, monthly_sma, cfg)
    bench = load_bench(start, end).resample("ME").last().pct_change()
    stats = performance_stats(result, bench)
    result.stats = stats
    return result, stats, bench


def plot_equity(result, bench):
    eq_norm = result.equity / result.config.initial_capital
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eq_norm.index, eq_norm.values, linewidth=2, color="navy",
            label=f"Model {result.config.model} (J={result.config.j_months}, K={result.config.k_months})")
    if bench is not None and len(bench) > 2:
        bn = bench.cumsum().add(1)
        bn = bn.loc[eq_norm.index[0]:].dropna()
        bn = bn / bn.iloc[0]
        ax.plot(bn.index, bn.values, color="crimson", alpha=0.8,
                label="^NSEI (buy & hold)")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_title("Jegadeesh & Titman (1993) Momentum - Normalized Growth")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth (1.0 = initial capital)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fmt_pct(x, digits=2):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%"


def main():
    st.title("Jegadeesh & Titman (1993) Momentum Backtester — NSE")
    st.caption("Returns to Buying Winners and Selling Losers (JF, 1993) — Models A & B, "
               "with a 200-day SMA trend overlay and concentration cap.")

    with st.sidebar:
        st.header("Parameters")
        capital = st.number_input("Initial capital (Rs)", 10000.0, 1e9, 400000.0, step=50000.0)
        model = st.radio("Strategy model", ["A (decile, 12-1)", "B (WRSS)"])
        model = model[0]
        j = st.slider("Formation period J (months)", 3, 12, 12)
        k = st.slider("Holding period K (months)", 1, 12, 3)
        max_stocks = st.slider("Max active long holdings", 1, 20, 6)
        long_short = st.checkbox("Long-short (short bottom decile)", value=False)
        sma_filter = st.checkbox("200-day SMA trend filter (longs only)", value=True)
        cost = st.number_input("Transaction cost (per side, %)", 0.0, 2.0, 0.15, step=0.05) / 100.0
        st.divider()
        col1, col2 = st.columns(2)
        start = col1.text_input("Start", "2021-01-01")
        end = col2.text_input("End", "2026-01-01")
        universe_limit = st.slider("Universe limit (0 = full Nifty 500)", 0, 500, 0,
                                   help="Cap tickers for a quick demo run.")
        run_btn = st.button("Run backtest", type="primary")

    if run_btn:
        ul = universe_limit or None
        with st.spinner("Running backtest..."):
            result, stats, bench = compute(start, end, float(capital), model, j, k,
                                           max_stocks, long_short, sma_filter,
                                           float(cost), ul)

        st.header("Performance")
        m = st.columns(6)
        m[0].metric("Final value", f"Rs {stats['final_value']:,.0f}")
        m[1].metric("Total return", fmt_pct(stats["total_return"]))
        m[2].metric("CAGR", fmt_pct(stats["cagr"]))
        m[3].metric("Sharpe", f"{stats['sharpe']:.2f}")
        m[4].metric("Max drawdown", fmt_pct(stats["max_drawdown"]))
        m[5].metric("Avg holdings", f"{stats['avg_long_positions']:.1f}")

        st.pyplot(plot_equity(result, bench))

        if "alpha_annualized" in stats:
            st.subheader("Market-model regression (vs ^NSEI)")
            c = st.columns(4)
            c[0].metric("Alpha (annualized)", fmt_pct(stats["alpha_annualized"]),
                        f"t = {stats['alpha_tstat']:.2f}")
            c[1].metric("Beta", f"{stats['beta']:.2f}", f"t = {stats['beta_tstat']:.2f}")
            c[2].metric("R-squared", f"{stats['r2']:.3f}")
            c[3].metric("Excess vs benchmark", fmt_pct(stats["excess_return"]))

        st.subheader("Last month holdings")
        last = result.weights.iloc[-1]
        holdings = pd.DataFrame({
            "ticker": last.index,
            "weight": last.values,
        }).sort_values("weight", ascending=False)
        holdings = holdings[holdings["weight"] > 0.001]
        holdings["weight_pct"] = (holdings["weight"] * 100).round(2)
        st.dataframe(holdings[["ticker", "weight_pct"]].set_index("ticker"),
                     use_container_width=True)

        st.subheader("Monthly returns (last 12)")
        tail = result.monthly_returns.tail(12).to_frame("strategy_return")
        tail["strategy_pct"] = (tail["strategy_return"] * 100).round(2)
        if bench is not None and len(bench) > 2:
            tail["bench_pct"] = (bench.reindex(tail.index) * 100).round(2)
        st.dataframe(tail[["strategy_pct", "bench_pct"]],
                     use_container_width=True)


if __name__ == "__main__":
    main()
