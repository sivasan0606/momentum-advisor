# JT 1993 Momentum Backtester (NSE)

Production-ready implementation of **Jegadeesh & Titman (1993), "Returns to
Buying Winners and Selling Losers: Implications for Stock Market Efficiency"**
for the Nifty 500 universe, delivered as a Python backtesting engine
(`momentum_engine.py`), an interactive **Streamlit** app (`app.py`), and a
**monthly momentum advisor** (`advisor.py`) that emits a BUY / HOLD / SELL
HTML report.

> Earlier in this repo you'll also find `quantbacktest/` — a separate 4-phase
> value / Piotroski F-Score / momentum / TSMOM pipeline built from your
> Technical Specification Sheet. This momentum engine is the focused
> Jegadeesh–Titman implementation requested afterward.

## Strategies (per the paper)

**Model A — J-month / K-month decile strategy**
- Formation return (12-minus-1 when J=12):
  `f_t = P[t-1] / P[t-1-J] - 1`
- Rank the universe each month, sort into deciles: **long the top 10%
  (Winners)**, optionally **short the bottom 10% (Losers)**.
- Hold each formation for **K months** using overlapping monthly rebalancing:
  at any month the portfolio averages the `1/K` slices formed over the past
  K months (reduces turnover, JT eq. portfolio construction).

**Model B — Weighted Relative Strength Strategy (WRSS)**
- Continuous weights equal to each stock's deviation from the equal-weighted
  market mean:
  `w_it = r_it - rbar_t`
- Long positive-deviation names, short negative-deviation names (or long-only).

## Risk overlay & constraints
- **200-day SMA trend filter** — long positions only in confirmed uptrends
  (price above SMA-200), evaluated with end-of-previous-month data (no
  look-ahead).
- **Max-stock concentration cap** — aggregate active long holdings capped
  (default 6) for concentrated growth.
- **Transaction costs** charged on monthly weight turnover (default 0.15%).
- **Long-only vs long-short** toggle (India retail usually long-only).

## Performance analytics
- Final value, absolute return, net **CAGR**, annualized volatility, Sharpe,
  max drawdown, average holdings.
- **Market-model regression** against `^NSEI` via OLS → annualized **alpha**
  (Jensen), **beta**, t-stats, R².

## Files
```
momentum_engine.py   # engine + CLI (Models A/B, overlays, alpha/beta)
app.py               # Streamlit UI with configurable J/K/capital/max-stocks
advisor.py           # monthly advisor -> interactive HTML page (Stage 2)
holdings.csv         # optional seed for the HTML holdings table
tests/test_momentum.py  # assert-based unit tests
tests/test_advisor.py   # advisor reconciliation + HTML payload tests
output/momentum_equity.png
advisor.html         # generated interactive report
```

## Requirements
```
pip install numpy pandas matplotlib yfinance niftystocks pyarrow streamlit
```

## Run

CLI backtest (Model A, J=12, K=3, max 6 stocks, long-only, SMA filter):
```bash
python momentum_engine.py
```

Options: `--model A|B`, `--j 12`, `--k 3`, `--max-stocks 6`, `--long-short`,
`--no-sma`, `--capital 400000`, `--start/--end`, `--universe-limit N`.

Tests:
```bash
python tests/test_momentum.py
python tests/test_strategy.py     # the 4-phase value/F-Score engine
python tests/test_advisor.py      # the monthly advisor
```

Interactive app:
```bash
streamlit run app.py
```

## Monthly advisor (Stage 2) — `advisor.html`

Run this once a month on the scan day. It downloads fresh prices, computes the
current JT target portfolio (top-10 by 12-month momentum, above 200-day SMA,
equal weight), and writes ONE self-contained **interactive HTML page**.

```bash
# new cash this month = Rs 50,000
python advisor.py --cash 50000
```

Open `advisor.html` in any browser. The page lets you:

- **Record what you bought** — enter/edit ticker, quantity, buy price and buy
  date in the "What you own" table. Edits save automatically to the browser's
  localStorage and persist between sessions.
- **See everything live** — current price, P&L vs your buy price, and the
  7% stoploss for every holding recompute as you type.
- **Get the recommendation** — BUY / TOP-UP / HOLD / SELL with exact
  whole-share quantities, target weights, momentum rank, and sell reasons,
  all recomputed against your entered holdings.
- **Adjust new cash** — change the cash field at the top and the buy orders
  update instantly.

The CSV is now only a *seed*: `holdings.csv` (optional) pre-fills the table on
first open. Everything after that is edited in the HTML.

Options: `--cash <amount>`, `--max-stocks 10`, `--stoploss 0.07`,
`--model A|B`, `--j 12`, `--k 3`, `--end <date>`, `--holdings <file>`, `--out <file>`. Run it
each month to refresh prices and re-embed the latest momentum targets.

### Run-scan button (live refresh)

For a one-click refresh from the page, host it locally:

```bash
python advisor.py --serve --cash 50000     # default port 8765
```

Then open **http://localhost:8765/advisor.html**. The **"Run scan"** button at
the top re-runs the full scan (downloads fresh prices, recomputes the momentum
targets, rewrites the page) and reloads it automatically. Your entered holdings
and cash stay saved in the browser. Press Ctrl+C to stop the server.

> The button needs the local server. Opening `advisor.html` directly as a file
> still works for editing/review, but "Run scan" will show a hint telling you to
> use `--serve`.

### Automatic monthly run + email report

The scan runs and emails the report to you on the **1st of every month at
19:30** (local time) via a launchd LaunchAgent (`com.momentum.advisor.monthly`).

**One-time email setup** — Gmail requires an App Password (regular password
won't work):

1. Turn on **2-Step Verification** at https://myaccount.google.com/security.
2. Create an App Password at
   https://myaccount.google.com/apppasswords (choose "Mail", generate).
3. Edit `mail_config.json` (chmod 600, git-ignored) and put the 16-char
   password in `app_password`:

   ```json
   { "sender": "you@gmail.com", "app_password": "xxxx xxxx xxxx xxxx", "recipient": "you@gmail.com" }
   ```

4. Send a test email:
   ```bash
   ./monthly_report.sh --test-email
   ```
5. Install the schedule (installs after a successful test email):
   ```bash
   ./install_monthly.sh
   ```

The monthly email contains an HTML summary with **BUY / TOP-UP / HOLD / SELL
tables** (ticker, quantity, price, amount, rank, stoploss, P&L, sell reason)
plus portfolio stats (value, proceeds, cash left, next scan date) and the
interactive `advisor.html` as an attachment.

Manage it:

```bash
./monthly_report.sh              # run the scan + email now (manual)
./install_monthly.sh --test-only # send a test email without scheduling
./install_monthly.sh --remove    # stop the monthly schedule
```

Logs: `logs/monthly_report.log`. To change the day/time, edit `HOUR`/`MINUTE`
(and `Day`) in `install_monthly.sh`, re-run it, or adjust the plist at
`~/Library/LaunchAgents/com.momentum.advisor.monthly.plist` and reload with
`launchctl kickstart -k gui/$(id -u)/com.momentum.advisor.monthly`.

### Rules / playbook (as printed in the report)

- **Scan**: last trading day of each month, after market close. Check prices
  daily (~2 min) only to catch stoploss breaks.
- **Rebalance**: monthly, same scan day. **Top-up only** — keep existing
  shares, never trim an over-weighted holding while it stays in the target
  list. New cash buys the highest-rank unmet orders first.
- **Exit**:
  - **Stoploss (default 7%)**: any holding below `entry × (1 − 0.07)` is sold
    at the next day's open — do not wait for rebalance.
  - **Dropped out**: no longer in the top-10 momentum names, or price below the
    200-day SMA, is sold at rebalance.
- **Buy**: top-decile 12-month momentum names trading above the 200-day SMA,
  up to 10 holdings, whole shares only; leftover cash carries forward.

## Reference results (2021-01 to 2025-12, Nifty 500, Rs 4L, cost 0.15%)

| Model | Config | CAGR | Sharpe | Max DD | Alpha (ann.) | vs NIFTY |
|---|---|---|---|---|---|---|
| A long-only | J=12, K=3, max 6, SMA | **33.1%** | 1.11 | -42.8% | 10.7% | +219 pts |
| B WRSS long-only | J=12, K=3, max 6, SMA | 29.2% | 1.00 | -43.3% | 7.4% | +154 pts |
| A long-short | J=12, K=3, decile | 13.3% | 0.63 | -27.8% | 13.8% | -36 pts |
| NIFTY buy & hold | — | — | — | -17.2% | — | — |

## Reference results (2016-01 to 2025-12, Nifty 500, Rs 4L, cost 0.15%)

NIFTY buy & hold over the window: **+227.8%**. 350 stocks with full 10-yr history.

| Model | Config | CAGR | Sharpe | Max DD | Alpha (ann.) | t | vs NIFTY |
|---|---|---|---|---|---|---|---|
| A long-only | J=12, K=3, max 6, SMA | **32.6%** | 1.01 | -35.4% | **18.8%** | 2.1 | +1475 pts |
| A long-only | J=12, K=3, max 10, SMA | 29.8% | 1.00 | -35.2% | 15.6% | 2.0 | +1126 pts |
| A long-only | J=12, K=3, max 15, SMA | 28.3% | 1.06 | -30.7% | 14.2% | 2.2 | +960 pts |
| B WRSS long-only | J=12, K=3, max 10, SMA | 29.6% | 0.95 | -38.1% | 15.5% | 1.8 | +1102 pts |

Year-by-year (A-10): strong wins in trending years (2017 +188%, 2020 +102%,
2021 +87%, 2023 +55%), losses in choppy/correction years (2016 -16%,
2025 -22%). Alpha becomes statistically significant (t ~ 2) over 10 years.

## Caveats
- **Survivorship bias**: the universe is today's Nifty 500; stocks delisted or
  removed mid-period are absent.
- **Dividend-adjusted** (auto-adjusted) prices are used.
- 5-year window spans a strong Indian equity bull market; momentum can
  whipsaw in sideways regimes.
- Backtests do not guarantee future performance; validate with paper trading
  before deploying capital.
