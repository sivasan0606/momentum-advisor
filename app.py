"""Streamlit app: JT 1993 momentum backtester + advisor + portfolio for NSE.

Tabs:
  Backtester — historical backtest with equity curve, Sharpe, alpha/beta
  Advisor    — current target portfolio, buy/sell recommendations
  Portfolio  — track your actual holdings and P&L
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
from momentum_engine import (MomentumConfig, formation_returns, run_momentum,
                             performance_stats)
from quantbacktest import data as qdata

st.set_page_config(page_title="Momentum Advisor (NSE)", layout="wide")


# ---------------------------------------------------------------------------
# Shared cached helpers
# ---------------------------------------------------------------------------
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
    return qdata.download_benchmark("^NSEI", start, end, cache_dir="cache")


def fmt_pct(x, digits=2):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%"


def fmt_money(v):
    return f"Rs {v:,.0f}"


# ---------------------------------------------------------------------------
# TAB 1: Backtester
# ---------------------------------------------------------------------------
def tab_backtester():
    st.header("Jegadeesh & Titman (1993) Momentum Backtester")
    st.caption("Models A & B, 200-day SMA trend overlay, concentration cap.")

    with st.sidebar:
        st.header("Backtester Parameters")
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
            daily = load_prices(start, end, ul)
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

        st.subheader("Performance")
        m = st.columns(6)
        m[0].metric("Final value", fmt_money(stats["final_value"]))
        m[1].metric("Total return", fmt_pct(stats["total_return"]))
        m[2].metric("CAGR", fmt_pct(stats["cagr"]))
        m[3].metric("Sharpe", f"{stats['sharpe']:.2f}")
        m[4].metric("Max drawdown", fmt_pct(stats["max_drawdown"]))
        m[5].metric("Avg holdings", f"{stats['avg_long_positions']:.1f}")

        eq_norm = result.equity / result.config.initial_capital
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(eq_norm.index, eq_norm.values, linewidth=2, color="navy",
                label=f"Model {result.config.model} (J={j}, K={k})")
        if bench is not None and len(bench) > 2:
            bn = bench.cumsum().add(1)
            bn = bn.loc[eq_norm.index[0]:].dropna()
            bn = bn / bn.iloc[0]
            ax.plot(bn.index, bn.values, color="crimson", alpha=0.8, label="^NSEI (buy & hold)")
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_title("Normalized Growth")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

        if "alpha_annualized" in stats:
            st.subheader("Market-model regression (vs ^NSEI)")
            c = st.columns(4)
            c[0].metric("Alpha", fmt_pct(stats["alpha_annualized"]), f"t = {stats['alpha_tstat']:.2f}")
            c[1].metric("Beta", f"{stats['beta']:.2f}", f"t = {stats['beta_tstat']:.2f}")
            c[2].metric("R-squared", f"{stats['r2']:.3f}")
            c[3].metric("Excess vs benchmark", fmt_pct(stats["excess_return"]))

        st.subheader("Last month holdings")
        last = result.weights.iloc[-1]
        holdings_df = pd.DataFrame({"ticker": last.index, "weight": last.values})
        holdings_df = holdings_df[holdings_df["weight"] > 0.001].sort_values("weight", ascending=False)
        holdings_df["weight_pct"] = (holdings_df["weight"] * 100).round(2)
        st.dataframe(holdings_df[["ticker", "weight_pct"]].set_index("ticker"), use_container_width=True)

        st.subheader("Monthly returns (last 12)")
        tail = result.monthly_returns.tail(12).to_frame("strategy_return")
        tail["strategy_pct"] = (tail["strategy_return"] * 100).round(2)
        if bench is not None and len(bench) > 2:
            tail["bench_pct"] = (bench.reindex(tail.index) * 100).round(2)
        st.dataframe(tail[["strategy_pct", "bench_pct"]], use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 2: Advisor
# ---------------------------------------------------------------------------
def tab_advisor():
    st.header("Quarterly Momentum Advisor")
    st.caption("JT-1993 target portfolio, buy/sell recommendations based on your holdings.")

    with st.sidebar:
        st.header("Advisor Parameters")
        adv_start = st.text_input("Advisor start", "2024-01-01", key="adv_start")
        adv_end = st.text_input("Advisor end", pd.Timestamp.now().strftime("%Y-%m-%d"), key="adv_end")
        adv_cash = st.number_input("New cash this quarter (Rs)", 0.0, 1e9, 0.0, step=50000.0, key="adv_cash")
        adv_max = st.slider("Max stocks", 1, 20, 10, key="adv_max")
        adv_stoploss = st.number_input("Stoploss %", 1.0, 30.0, 7.0, step=0.5, key="adv_stop") / 100.0
        adv_run = st.button("Run advisor scan", type="primary", key="adv_run")

    if adv_run:
        with st.spinner("Downloading prices and computing targets..."):
            daily = load_prices(adv_start, adv_end, None)
            sma = daily.rolling(200).mean()
            monthly_close = daily.resample("ME").last()
            monthly_sma = sma.resample("ME").last()
            monthly_ret = monthly_close.pct_change()

            cfg = MomentumConfig(model="A", j_months=12, k_months=3,
                                 max_stocks=adv_max, sma_trend_filter=True,
                                 sma_window=200, min_price_days=273,
                                 cache_dir="cache",
                                 start_date=adv_start, end_date=adv_end)
            result = run_momentum(monthly_close, monthly_ret, monthly_sma, cfg)

            last_w = result.weights.iloc[-1]
            target_weights = {t: float(w) for t, w in last_w.items() if w > 0.0001}
            formation = formation_returns(monthly_close, 12).iloc[-1].dropna()
            formation = formation.sort_values(ascending=False)
            ranks = {t: i + 1 for i, t in enumerate(formation.index)}
            target_ranks = {t: ranks.get(t, 999) for t in target_weights}

            prices = {t: float(v) for t, v in daily.iloc[-1].items() if pd.isna(v) is False}
            ret_1y = {}
            if len(daily) > 252:
                ret_1y_series = daily.iloc[-1] / daily.shift(252).iloc[-1] - 1.0
                ret_1y = {t: float(v) for t, v in ret_1y_series.items()
                          if pd.notna(v) and np.isfinite(v)}
            as_of = daily.index[-1].strftime("%Y-%m-%d")

        st.success(f"Data as of **{as_of}** — {len(target_weights)} target names")

        st.subheader("Target Portfolio")
        target_df = pd.DataFrame([
            {"Ticker": t.replace(".NS", ""),
             "Rank": target_ranks.get(t, 999),
             "Weight": f"{w:.1%}",
             "Price": fmt_money(prices.get(t, 0)),
             "1Y Return": fmt_pct(ret_1y.get(t))}
            for t, w in sorted(target_weights.items(), key=lambda x: target_ranks.get(x[0], 999))
        ])
        st.dataframe(target_df.set_index("Ticker"), use_container_width=True)

        st.subheader("Rules")
        st.markdown(f"""
- **Stoploss:** Exit if a holding falls **{adv_stoploss:.0%}** below your buy price
- **Entry:** Buy the **top {adv_max}** stocks by 12-month momentum (skip last month), above 200-day SMA
- **Rebalance:** Quarterly, top-up only — never trim over-weighted holdings
- **Exit:** Sell if a stock drops out of top-{adv_max} or below SMA-200
- **Next scan:** 4th of Feb/May/Aug/Nov, after market close
""")


# ---------------------------------------------------------------------------
# TAB 3: Portfolio
# ---------------------------------------------------------------------------
def tab_portfolio():
    st.header("My Portfolio")
    st.caption("Track your actual holdings and P&L. Data comes from the Advisor scan.")

    with st.sidebar:
        st.header("Portfolio Parameters")
        port_start = st.text_input("Portfolio start", "2024-01-01", key="port_start")
        port_end = st.text_input("Portfolio end", pd.Timestamp.now().strftime("%Y-%m-%d"), key="port_end")
        port_stoploss = st.number_input("Stoploss %", 1.0, 30.0, 7.0, step=0.5, key="port_stop") / 100.0
        port_run = st.button("Load prices", type="primary", key="port_run")

    if "holdings" not in st.session_state:
        st.session_state.holdings = pd.DataFrame({
            "ticker": ["", ""],
            "quantity": [0, 0],
            "avg_price": [0.0, 0.0],
            "entry_date": ["", ""],
        })

    st.subheader("What you own")
    c1, c2 = st.columns([1, 3])
    with c1:
        csv_file = st.file_uploader("Load CSV/Excel (Zerodha tradebook)", type=["csv", "xlsx", "xls"])

    if csv_file is not None:
        try:
            name = csv_file.name.lower()
            if name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(csv_file)
            else:
                df = pd.read_csv(csv_file)
            df.columns = [c.strip().lower() for c in df.columns]
            sym_col = next((c for c in ["symbol", "ticker", "stock", "company"] if c in df.columns), None)
            qty_col = next((c for c in ["qty", "quantity"] if c in df.columns), None)
            price_col = next((c for c in ["price", "avg.", "avg price", "average price"] if c in df.columns), None)
            date_col = next((c for c in ["trade_date", "date", "trade date"] if c in df.columns), None)
            type_col = next((c for c in ["trade_type", "type", "side"] if c in df.columns), None)

            if sym_col and qty_col and price_col:
                buys = {}
                for _, row in df.iterrows():
                    ticker = str(row[sym_col]).strip().upper().replace("-BE", "").replace("-EQ", "").replace("-BZ", "")
                    if not ticker.endswith(".NS"):
                        ticker += ".NS"
                    qty = abs(float(row[qty_col]))
                    price = float(row[price_col])
                    trade_type = str(row.get(type_col, "")).upper() if type_col else ""
                    is_sell = trade_type in ("SELL", "S") or float(row[qty_col]) < 0
                    if is_sell:
                        continue
                    if ticker not in buys:
                        buys[ticker] = {"qty": 0, "total_cost": 0, "date": ""}
                    buys[ticker]["qty"] += qty
                    buys[ticker]["total_cost"] += qty * price
                    dt = str(row.get(date_col, "")).strip() if date_col else ""
                    if dt and not buys[ticker]["date"]:
                        buys[ticker]["date"] = dt

                rows = []
                for ticker, b in buys.items():
                    rows.append({
                        "ticker": ticker.replace(".NS", ""),
                        "quantity": int(b["qty"]),
                        "avg_price": round(b["total_cost"] / b["qty"], 2) if b["qty"] else 0.0,
                        "entry_date": b["date"],
                    })
                if rows:
                    st.session_state.holdings = pd.DataFrame(rows)
                    st.success(f"Loaded {len(rows)} holding(s) from {csv_file.name}")
                else:
                    st.warning("No buy trades found in the file.")
            else:
                st.error("Could not find symbol/qty/price columns. Expected Zerodha tradebook format.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.caption("Edit your holdings below. Current price and P&L update after a scan.")

    st.session_state.holdings = st.data_editor(
        st.session_state.holdings,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "quantity": st.column_config.NumberColumn("Qty", min_value=0),
            "avg_price": st.column_config.NumberColumn("Buy price (Rs)", min_value=0.0, format="Rs %.2f"),
            "entry_date": st.column_config.TextColumn("Entry date"),
        },
    )

    if port_run:
        with st.spinner("Loading prices..."):
            daily = load_prices(port_start, port_end, None)
            current_prices = {}
            ret_1y = {}
            for t in daily.columns:
                v = daily[t].iloc[-1]
                if pd.notna(v):
                    current_prices[t] = float(v)
            if len(daily) > 252:
                r1y = daily.iloc[-1] / daily.shift(252).iloc[-1] - 1.0
                ret_1y = {t: float(v) for t, v in r1y.items()
                          if pd.notna(v) and np.isfinite(v)}

        hs = st.session_state.holdings.copy()
        hs["norm_ticker"] = hs["ticker"].str.upper().str.strip().apply(
            lambda x: x if x.endswith(".NS") else x + ".NS" if x else "")
        hs["current_price"] = hs["norm_ticker"].map(current_prices)
        hs["ret_1y"] = hs["norm_ticker"].map(ret_1y)
        hs["inv_value"] = hs["quantity"] * hs["avg_price"]
        hs["cur_value"] = hs["quantity"] * hs["current_price"]
        hs["pnl_rs"] = hs["cur_value"] - hs["inv_value"]
        hs["pnl_pct"] = (hs["current_price"] / hs["avg_price"] - 1.0)
        hs["stoploss_price"] = hs["avg_price"] * (1.0 - port_stoploss)

        display = pd.DataFrame({
            "Ticker": hs["ticker"],
            "Qty": hs["quantity"],
            "Buy Price": hs["avg_price"].apply(lambda x: fmt_money(x) if x else "n/a"),
            "Inv Value": hs.apply(lambda r: fmt_money(r["inv_value"]) if r["quantity"] and r["avg_price"] else "n/a", axis=1),
            "Buy Date": hs["entry_date"].apply(lambda x: x if x else "n/a"),
            "Current": hs["current_price"].apply(lambda x: fmt_money(x) if pd.notna(x) else "n/a"),
            "1Y Return": hs["ret_1y"].apply(lambda x: fmt_pct(x) if pd.notna(x) else "n/a"),
            "P&L %": hs["pnl_pct"].apply(lambda x: fmt_pct(x) if pd.notna(x) else "n/a"),
            "P&L (Rs)": hs.apply(lambda r: fmt_money(r["pnl_rs"]) if pd.notna(r["current_price"]) else "n/a", axis=1),
            "Stoploss": hs["stoploss_price"].apply(lambda x: fmt_money(x) if x else "n/a"),
        })
        st.dataframe(display.set_index("Ticker"), use_container_width=True)

        valid = hs[hs["current_price"].notna() & (hs["avg_price"] > 0)]
        if len(valid) > 0:
            tot_inv = valid["inv_value"].sum()
            tot_val = valid["cur_value"].sum()
            tot_pnl = tot_val - tot_inv

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total invested", fmt_money(tot_inv))
            c2.metric("Current value", fmt_money(tot_val))
            c3.metric("Open P&L", fmt_money(tot_val - tot_inv),
                       f"{(tot_val / tot_inv - 1):.1%}" if tot_inv else "")
            c4.metric("Holdings", f"{len(valid)} held")

            sells = hs[(hs["pnl_pct"].notna()) & (hs["pnl_pct"] < -port_stoploss)]
            if len(sells) > 0:
                st.warning(f"Stoploss triggered: {', '.join(sells['ticker'].tolist())}")
        else:
            st.info("Enter holdings above and click 'Load prices' to see P&L.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tab1, tab2, tab3 = st.tabs(["Backtester", "Advisor", "Portfolio"])

    with tab1:
        tab_backtester()
    with tab2:
        tab_advisor()
    with tab3:
        tab_portfolio()


if __name__ == "__main__":
    main()
