"""Monthly rebalance engine implementing the 4-phase value/quality/momentum/TSMOM strategy."""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import fundamentals as fmod
from . import signals
from .config import BacktestConfig


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float


@dataclass
class ScreenState:
    """Cached per-rebalance screen outcome."""
    date: pd.Timestamp
    n_with_bm: int = 0
    n_value: int = 0
    n_quality: int = 0
    n_trend: int = 0
    n_long: int = 0
    asof_fy: Optional[pd.Timestamp] = None


@dataclass
class BacktestResult:
    history: pd.DataFrame
    trades: List[dict]
    funnel: pd.DataFrame
    final_holdings: Dict[str, Position]
    config: BacktestConfig


def _ceil_quantile(n: int, q: float) -> int:
    if n <= 0:
        return 0
    return max(1, math.ceil(n * q))


def run_backtest(prices: pd.DataFrame, fund_panel: pd.DataFrame,
                 config: BacktestConfig = BacktestConfig()) -> BacktestResult:
    """Run the full 4-phase strategy backtest.

    prices      : daily adjusted close, columns = tickers, DatetimeIndex
    fund_panel  : long-form fundamentals DataFrame (see data.fetch_fundamentals)
    """
    if prices.empty:
        raise ValueError("No price data to backtest.")

    lookup = fmod.panel_to_lookup(fund_panel)

    # --- Precompute vectorized signals across the whole panel ---
    mom = signals.momentum_12_1(prices, config.mom_lookback_days, config.mom_skip_days)
    ts = signals.tsmom(prices, config.tsmom_lookback_days)
    vol = signals.annualized_vol(prices, config.vol_lookback_days)

    # --- Monthly rebalance dates within [start, end] ---
    month_ends = prices.index.to_series().resample("ME").last().dropna().values
    start_ts, end_ts = pd.Timestamp(config.start_date), pd.Timestamp(config.end_date)
    dates = [d for d in month_ends if start_ts <= d <= end_ts]
    if not dates:
        raise ValueError("No rebalance dates in the configured window.")

    cash = config.initial_capital
    holdings: Dict[str, Position] = {}
    history_rows = []
    trade_log: List[dict] = []
    funnel_rows = []

    # Cached per-ticker published-FY state (refreshed only when it changes)
    cur_fy: Dict[str, dict] = {}
    fscore_map: Dict[str, int] = {}
    last_sig = frozenset()

    for date in dates:
        p_row = prices.loc[date]
        mom_row = mom.loc[date]
        ts_row = ts.loc[date]
        vol_row = vol.loc[date]

        # --- Refresh which fiscal year is "published" as of this date ---
        new_cur_fy = {}
        for tk, history in lookup.items():
            fy = fmod.asof_fy(history, pd.Timestamp(date), config.fundamental_pub_lag_months)
            if fy is not None:
                new_cur_fy[tk] = fy
        cur_fy = new_cur_fy

        sig = frozenset((tk, fy["fy_end"]) for tk, fy in cur_fy.items())
        if sig != last_sig:
            last_sig = sig
            fscore_map = {}
            for tk, fy in cur_fy.items():
                prev = fmod.prior_fy(lookup[tk], fy)
                fscore_map[tk] = fmod.piotroski_fscore(fy, prev)

        asof_fy = pd.Timestamp(max((fy["fy_end"] for fy in cur_fy.values()), default=pd.NaT))
        screen = ScreenState(date=pd.Timestamp(date), asof_fy=None if pd.isna(asof_fy) else asof_fy)

        # --- Phase 1: Book-to-market value screen (recomputed monthly with live price) ---
        bm_map = {}
        for tk, fy in cur_fy.items():
            price = p_row.get(tk, np.nan)
            shares = fy.get(fmod.ORDINARY_SHARES)
            bm = fmod.book_to_market(fy, price, shares)
            if bm is not None and not pd.isna(bm):
                bm_map[tk] = bm
        screen.n_with_bm = len(bm_map)

        ranked_by_bm = sorted(bm_map, key=lambda tk: bm_map[tk], reverse=True)
        n_value = _ceil_quantile(len(ranked_by_bm), config.bm_top_quantile)
        value_pass = set(ranked_by_bm[:n_value])
        screen.n_value = len(value_pass)

        # --- Phase 2: Piotroski quality filter (F-Score 7-9) ---
        quality_pass = {tk for tk in value_pass
                        if fscore_map.get(tk, -1) in range(config.fscore_min, config.fscore_max + 1)}
        screen.n_quality = len(quality_pass)

        # --- Phase 3: cross-sectional momentum rank -> long top decile of quality universe ---
        rank_pool = []
        for tk in quality_pass:
            mom_v, vol_v = mom_row.get(tk, np.nan), vol_row.get(tk, np.nan)
            if pd.isna(mom_v) or pd.isna(vol_v) or pd.isna(p_row.get(tk, np.nan)):
                continue
            rank_pool.append(tk)
        rank_pool.sort(key=lambda tk: mom_row[tk], reverse=True)
        n_pick = min(_ceil_quantile(len(rank_pool), config.top_decile), config.max_positions)
        top_decile = rank_pool[:n_pick]
        screen.n_trend = len(top_decile)

        # --- Phase 4: TSMOM trend filter on the selected names (hold only if 12m sign positive) ---
        long_set = [tk for tk in top_decile if ts_row.get(tk, np.nan) > 0]
        screen.n_long = len(long_set)

        # --- Phase 4: volatility-targeted sizing, normalized to 100% invested ---
        raw_w = {tk: config.vol_target / vol_row[tk] for tk in long_set}
        total_w = sum(raw_w.values())
        weights = {tk: (raw_w[tk] / total_w if total_w > 0 else 0.0) for tk in long_set}

        # --- Execute rebalance (sells first, then buys/trims) ---
        pv = cash + sum(pos.shares * p_row[tk] for tk, pos in holdings.items())

        for tk in list(holdings.keys()):
            if tk not in long_set:
                pos = holdings.pop(tk)
                price = p_row[tk]
                proceeds = pos.shares * price * (1 - config.transaction_cost)
                cash += proceeds
                pnl = (price - pos.avg_cost) * pos.shares
                trade_log.append({"date": date, "ticker": tk, "side": "SELL", "qty": pos.shares,
                                  "price": price, "pnl": pnl})

        for tk in long_set:
            target_shares = math.floor(weights[tk] * pv / p_row[tk])
            pos = holdings.get(tk)
            cur_shares = pos.shares if pos else 0
            diff = target_shares - cur_shares
            if diff > 0:
                max_buy = math.floor(cash / (p_row[tk] * (1 + config.transaction_cost)))
                buy = min(diff, max_buy)
                if buy <= 0:
                    continue
                spend = buy * p_row[tk] * (1 + config.transaction_cost)
                cash -= spend
                if pos:
                    new_total = pos.shares + buy
                    pos.avg_cost = (pos.avg_cost * pos.shares + p_row[tk] * buy) / new_total
                    pos.shares = new_total
                else:
                    holdings[tk] = Position(tk, buy, p_row[tk])
                trade_log.append({"date": date, "ticker": tk, "side": "BUY", "qty": buy,
                                  "price": p_row[tk], "pnl": np.nan})
            elif diff < 0:
                qty = -diff
                proceeds = qty * p_row[tk] * (1 - config.transaction_cost)
                cash += proceeds
                pnl = (p_row[tk] - pos.avg_cost) * qty
                pos.shares -= qty
                trade_log.append({"date": date, "ticker": tk, "side": "SELL", "qty": qty,
                                  "price": p_row[tk], "pnl": pnl})
                if pos.shares <= 0:
                    del holdings[tk]

        stock_value = sum(pos.shares * p_row[tk] for tk, pos in holdings.items())
        total_val = cash + stock_value
        history_rows.append({
            "date": date,
            "portfolio_value": total_val,
            "invested": stock_value,
            "cash": cash,
            "n_positions": len(holdings),
            "weights": {tk: weights[tk] for tk in long_set},
        })
        funnel_rows.append(screen)

    history = pd.DataFrame(history_rows).set_index("date")
    funnel = pd.DataFrame(funnel_rows).set_index("date")
    return BacktestResult(history=history, trades=trade_log, funnel=funnel,
                          final_holdings=holdings, config=config)
