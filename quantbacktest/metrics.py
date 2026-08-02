"""Performance metrics: returns, Sharpe, drawdown, benchmark comparison."""
import numpy as np
import pandas as pd


def monthly_returns(history: pd.DataFrame) -> pd.Series:
    return history["portfolio_value"].pct_change().dropna()


def total_return(history: pd.DataFrame) -> float:
    first = history["portfolio_value"].iloc[0]
    last = history["portfolio_value"].iloc[-1]
    return last / first - 1.0


def cagr(history: pd.DataFrame) -> float:
    years = max((history.index[-1] - history.index[0]).days / 365.25, 1e-9)
    return (history["portfolio_value"].iloc[-1] / history["portfolio_value"].iloc[0]) ** (1 / years) - 1.0


def annualized_volatility(history: pd.DataFrame) -> float:
    return monthly_returns(history).std() * np.sqrt(12)


def sharpe(history: pd.DataFrame, rf: float = 0.0) -> float:
    mret = monthly_returns(history)
    vol = mret.std()
    if vol == 0 or pd.isna(vol):
        return 0.0
    return (mret.mean() * 12 - rf) / (vol * np.sqrt(12))


def max_drawdown(history: pd.DataFrame) -> float:
    cummax = history["portfolio_value"].cummax()
    dd = history["portfolio_value"] / cummax - 1.0
    return dd.min()


def benchmark_return(bench: pd.Series, history: pd.DataFrame) -> float:
    """Benchmark buy & hold total return over the same window as the backtest."""
    s, e = history.index[0], history.index[-1]
    sub = bench.loc[s:e]
    if len(sub) < 2:
        return np.nan
    return sub.iloc[-1] / sub.iloc[0] - 1.0


def benchmark_drawdown(bench: pd.Series, history: pd.DataFrame) -> float:
    s, e = history.index[0], history.index[-1]
    sub = bench.loc[s:e]
    if len(sub) < 2:
        return np.nan
    cummax = sub.cummax()
    return (sub / cummax - 1.0).min()


def trade_stats(trades: list) -> dict:
    sells = [t for t in trades if t.get("side") == "SELL"]
    closed = [t for t in sells if t.get("pnl") is not None and not pd.isna(t.get("pnl"))]
    wins = [t for t in closed if t["pnl"] > 0]
    total_buy = sum(t["qty"] * t["price"] for t in trades if t.get("side") == "BUY")
    total_sell = sum(t["qty"] * t["price"] for t in sells)
    return {
        "buys": sum(1 for t in trades if t.get("side") == "BUY"),
        "sells": len(sells),
        "round_trips": len(closed),
        "wins": len(wins),
        "win_rate": (len(wins) / len(closed)) if closed else 0.0,
        "gross_buy_value": total_buy,
        "gross_sell_value": total_sell,
    }


def summarize(history: pd.DataFrame, trades: list, bench: pd.Series = None) -> dict:
    out = {
        "start": history.index[0],
        "end": history.index[-1],
        "final_value": float(history["portfolio_value"].iloc[-1]),
        "initial_value": float(history["portfolio_value"].iloc[0]),
        "total_return": total_return(history),
        "cagr": cagr(history),
        "annualized_volatility": annualized_volatility(history),
        "sharpe": sharpe(history),
        "max_drawdown": max_drawdown(history),
        "avg_positions": float(history["n_positions"].mean()),
    }
    if bench is not None:
        out["benchmark_return"] = benchmark_return(bench, history)
        out["benchmark_max_drawdown"] = benchmark_drawdown(bench, history)
        out["excess_return"] = out["total_return"] - out["benchmark_return"]
    out.update(trade_stats(trades))
    return out
