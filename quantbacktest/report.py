"""Output: equity-curve chart, monthly CSV, funnel audit, console summary."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import metrics


def save_chart(history: pd.DataFrame, bench: pd.Series, config, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history.index, history["portfolio_value"], label="Strategy (Value + Quality + Momentum + TSMOM)",
            color="navy", linewidth=2)
    ax.axhline(config.initial_capital, color="gray", linestyle="--", linewidth=1.2,
               label="Initial Capital")

    if bench is not None:
        s, e = history.index[0], history.index[-1]
        sub = bench.loc[s:e]
        bench_norm = sub / sub.iloc[0] * config.initial_capital
        ax.plot(bench_norm.index, bench_norm.values, label=config.benchmark, color="crimson",
                linewidth=1.2, alpha=0.8)

    ax.set_title("Backtest: Top-20% BM + F-Score>=7 + Top-Decile Momentum + TSMOM Trend, 40%/vol sizing")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (Rs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_history_csv(history: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    history.to_csv(path)
    return path


def save_funnel_csv(funnel: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    funnel.to_csv(path)
    return path


def format_pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.2f}%"


def print_summary(stats: dict):
    print("\n" + "=" * 62)
    print("BACKTEST SUMMARY (Value + F-Score + Momentum + TSMOM, vol-targeted)")
    print("=" * 62)
    print(f"Period                : {stats['start'].date()} to {stats['end'].date()}")
    print(f"Final Portfolio Value : Rs {stats['final_value']:,.2f}")
    print(f"Total Return          : {format_pct(stats['total_return'])}")
    print(f"CAGR                  : {format_pct(stats['cagr'])}")
    print(f"Annualized Volatility : {format_pct(stats['annualized_volatility'])}")
    print(f"Sharpe Ratio          : {stats['sharpe']:.2f}")
    print(f"Max Drawdown          : {format_pct(stats['max_drawdown'])}")
    print(f"Avg Positions         : {stats['avg_positions']:.1f}")
    if "benchmark_return" in stats and not pd.isna(stats["benchmark_return"]):
        print(f"Benchmark Return      : {format_pct(stats['benchmark_return'])}")
        print(f"Benchmark Max DD      : {format_pct(stats['benchmark_max_drawdown'])}")
        print(f"Excess vs Benchmark   : {format_pct(stats['excess_return'])}")
    print("-" * 62)
    print(f"Buys                  : {stats['buys']}")
    print(f"Sells                 : {stats['sells']}")
    print(f"Round trips           : {stats['round_trips']}")
    print(f"Win rate (closed)     : {format_pct(stats['win_rate'])}")
    print("=" * 62)
