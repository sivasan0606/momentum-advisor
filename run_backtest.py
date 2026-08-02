#!/usr/bin/env python3
"""Stage 1 CLI: run the 4-phase value/quality/momentum/TSMOM backtest.

Example:
    python run_backtest.py --years 5
    python run_backtest.py --start 2021-01-01 --end 2026-01-01 --universe-limit 100
    python run_backtest.py --refresh-cache --no-chart
"""
import argparse
import os
import sys

from quantbacktest.config import BacktestConfig
from quantbacktest import data, portfolio, metrics, report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(
        description="Backtest the Value + F-Score + Momentum + TSMOM strategy on the Nifty 500.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--years", type=int, default=5, help="Years back from today (ignored if --start given).")
    p.add_argument("--start", type=str, default="2021-01-01", help="Backtest start YYYY-MM-DD.")
    p.add_argument("--end", type=str, default="2026-01-01", help="Backtest end YYYY-MM-DD.")
    p.add_argument("--initial-capital", type=float, default=400000)
    p.add_argument("--cost", type=float, default=0.0015, help="Per-trade transaction cost fraction.")
    p.add_argument("--bm-quantile", type=float, default=0.20, help="Phase 1: top BM quantile to keep.")
    p.add_argument("--fscore-min", type=int, default=7)
    p.add_argument("--fscore-max", type=int, default=9)
    p.add_argument("--top-decile", type=float, default=0.10, help="Phase 3: long top fraction by momentum.")
    p.add_argument("--vol-target", type=float, default=0.40, help="Phase 4: annualized vol target for sizing.")
    p.add_argument("--universe", type=str, default="nifty500",
                   help="'nifty500' or comma-separated tickers.")
    p.add_argument("--universe-limit", type=int, default=None,
                   help="Limit tickers (smoke tests).")
    p.add_argument("--refresh-cache", action="store_true", help="Re-download prices and fundamentals.")
    p.add_argument("--no-chart", action="store_true")
    p.add_argument("--outdir", type=str, default="output")
    return p.parse_args()


def main():
    args = parse_args()

    config = BacktestConfig(
        universe=args.universe,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.initial_capital,
        transaction_cost=args.cost,
        bm_top_quantile=args.bm_quantile,
        fscore_min=args.fscore_min,
        fscore_max=args.fscore_max,
        top_decile=args.top_decile,
        vol_target=args.vol_target,
        output_dir=args.outdir,
    )

    os.makedirs(args.outdir, exist_ok=True)

    print("Step 1/5: Loading universe...")
    tickers = data.load_universe(config.universe)
    if args.universe_limit:
        tickers = tickers[: args.universe_limit]
    print(f"  {len(tickers)} tickers")

    print("Step 2/5: Downloading prices...")
    prices = data.download_prices(tickers, config.start_date, config.end_date,
                                  cache_dir=config.cache_dir, refresh=args.refresh_cache)
    prices = data.clean_prices(prices, config.min_price_days)
    print(f"  {len(prices.columns)} tickers after cleaning")

    print("Step 3/5: Downloading fundamentals...")
    fund_panel = data.fetch_fundamentals(tickers, cache_dir=config.cache_dir,
                                         refresh=args.refresh_cache, universe_limit=args.universe_limit)
    fin_set = data.financial_exclusion_set()
    fund_panel, n_dropped = data.filter_financials(fund_panel, fin_set)
    print(f"  excluded {n_dropped} financial names; {fund_panel['ticker'].nunique()} with fundamentals")

    print("Step 4/5: Running backtest...")
    result = portfolio.run_backtest(prices, fund_panel, config)

    print("Step 5/5: Computing metrics & reports...")
    bench = None
    try:
        bench = data.download_benchmark(config.benchmark, config.start_date, config.end_date,
                                        cache_dir=config.cache_dir, refresh=args.refresh_cache)
    except Exception as exc:
        print(f"  (benchmark unavailable: {exc})")

    stats = metrics.summarize(result.history, result.trades, bench)
    report.print_summary(stats)

    base = os.path.join(args.outdir, "backtest")
    if not args.no_chart:
        chart = report.save_chart(result.history, bench, config, base + "_equity.png")
        print(f"Chart saved to {chart}")
    csv_path = report.save_history_csv(result.history, base + "_monthly.csv")
    funnel_path = report.save_funnel_csv(result.funnel, base + "_funnel.csv")
    print(f"Monthly history saved to {csv_path}")
    print(f"Screen funnel saved to {funnel_path}")

    # Show the last few rebalances of the funnel for a quick sense of the pipeline
    print("\nLast 8 rebalances (universe -> value -> fscore -> picked -> held):")
    tail = result.funnel.tail(8)
    for date, row in tail.iterrows():
        fy = row["asof_fy"]
        fy_s = fy.date().isoformat() if fy is not None and str(fy) != "NaT" else "-"
        print(f"  {date.date()}  FY={fy_s:>12}  "
              f"BM={int(row['n_with_bm']):>4}  V={int(row['n_value']):>4}  "
              f"Q={int(row['n_quality']):>4}  PICK={int(row['n_trend']):>3}  HELD={int(row['n_long']):>3}")

    return stats


if __name__ == "__main__":
    main()
