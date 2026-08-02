"""Piotroski F-Score (9 flags) and book-to-market ratio."""
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Field names that the fundamentals panel uses (must match data.py output)
# --------------------------------------------------------------------------
NET_INCOME = "net_income"
GROSS_PROFIT = "gross_profit"
TOTAL_REVENUE = "total_revenue"
TOTAL_ASSETS = "total_assets"
STOCKHOLDERS_EQUITY = "stockholders_equity"
LONG_TERM_DEBT = "long_term_debt"
CURRENT_ASSETS = "current_assets"
CURRENT_LIABILITIES = "current_liabilities"
OPERATING_CF = "operating_cash_flow"
ORDINARY_SHARES = "ordinary_shares"


def _pos(v):
    """1 if value is a positive number, 0 if missing or <= 0."""
    if v is None or pd.isna(v):
        return 0
    return 1 if float(v) > 0 else 0


def _gt(cur, prev):
    """1 if current strictly exceeds previous (both available), else 0."""
    if cur is None or prev is None or pd.isna(cur) or pd.isna(prev):
        return 0
    return 1 if float(cur) > float(prev) else 0


def piotroski_fscore(cur, prev) -> int:
    """Sum of the 9 binary Piotroski flags for fiscal year `cur` vs `prev`.

    cur / prev : row-like objects exposing the fields named by the module
    constants above (dict, pd.Series, or simple namespace).
    """
    get = lambda r, k: (r.get(k) if hasattr(r, "get") else getattr(r, k, None))

    def roa(r):
        ni, ta = get(r, NET_INCOME), get(r, TOTAL_ASSETS)
        if ni is None or ta is None or pd.isna(ni) or pd.isna(ta) or float(ta) == 0:
            return np.nan
        return float(ni) / float(ta)

    def leverage(r):
        ltd, ta = get(r, LONG_TERM_DEBT), get(r, TOTAL_ASSETS)
        if ltd is None or ta is None or pd.isna(ltd) or pd.isna(ta) or float(ta) == 0:
            return np.nan
        return float(ltd) / float(ta)

    def current_ratio(r):
        ca, cl = get(r, CURRENT_ASSETS), get(r, CURRENT_LIABILITIES)
        if ca is None or cl is None or pd.isna(ca) or pd.isna(cl) or float(cl) == 0:
            return np.nan
        return float(ca) / float(cl)

    def gross_margin(r):
        gp, tr = get(r, GROSS_PROFIT), get(r, TOTAL_REVENUE)
        if gp is None or tr is None or pd.isna(gp) or pd.isna(tr) or float(tr) == 0:
            return np.nan
        return float(gp) / float(tr)

    def asset_turnover(r):
        tr, ta = get(r, TOTAL_REVENUE), get(r, TOTAL_ASSETS)
        if tr is None or ta is None or pd.isna(tr) or pd.isna(ta) or float(ta) == 0:
            return np.nan
        return float(tr) / float(ta)

    def no_new_equity(cur_r, prev_r):
        sc = get(cur_r, ORDINARY_SHARES)
        sp = get(prev_r, ORDINARY_SHARES)
        if sc is None or sp is None or pd.isna(sc) or pd.isna(sp):
            return 0
        return 1 if float(sc) <= float(sp) else 0

    def decreased(a, b):
        """1 if b <= a strictly-not-increased and both available, else 0."""
        if pd.isna(a) or pd.isna(b):
            return 0
        return 1 if float(a) <= float(b) else 0

    flags = [
        # Profitability
        _pos(get(cur, NET_INCOME)),                       # F_ROA: positive ROA (net income > 0)
        _pos(get(cur, OPERATING_CF)),                     # F_CFO: positive operating cash flow
        _gt(roa(cur), roa(prev)),                          # F_dROA: ROA improved
        _gt(get(cur, OPERATING_CF), get(cur, NET_INCOME)), # F_ACCRUAL: CFO > net income
        # Leverage / liquidity / funding
        decreased(leverage(cur), leverage(prev)),          # F_dLEVER: leverage fell (or flat)
        _gt(current_ratio(cur), current_ratio(prev)),      # F_dLIQUID: current ratio rose
        no_new_equity(cur, prev),                          # F_EQ: no new shares issued
        _gt(gross_margin(cur), gross_margin(prev)),        # F_dMARGIN: gross margin rose
        _gt(asset_turnover(cur), asset_turnover(prev)),    # F_dTURN: asset turnover rose
    ]
    return sum(flags)


def book_to_market(row, price: float, shares: float) -> float:
    """BM = book equity / market cap (price x shares). NaN if any input missing."""
    eq = row.get(STOCKHOLDERS_EQUITY, None) if hasattr(row, "get") else getattr(row, STOCKHOLDERS_EQUITY, None)
    if eq is None or shares is None or pd.isna(eq) or pd.isna(shares) or pd.isna(price):
        return np.nan
    if price <= 0 or shares <= 0:
        return np.nan
    return float(eq) / (float(price) * float(shares))


# --------------------------------------------------------------------------
# Panel helpers (long-form fundamentals DataFrame)
# --------------------------------------------------------------------------
def panel_to_lookup(panel: pd.DataFrame) -> dict:
    """ticker -> sorted list of (fy_end, row-dict) pairs, newest last."""
    lookup = {}
    for tk, grp in panel.groupby("ticker", sort=False):
        rows = grp.to_dict("records")
        rows.sort(key=lambda r: r["fy_end"])
        lookup[tk] = rows
    return lookup


def asof_fy(history: list, asof: pd.Timestamp, pub_lag_months: int):
    """Latest fiscal year end whose results are public by `asof`.

    Returns the row-dict, or None. Assumes `history` sorted by fy_end.
    """
    cutoff = asof - pd.DateOffset(months=pub_lag_months)
    candidate = None
    for row in history:
        if row["fy_end"] <= cutoff:
            candidate = row
        else:
            break
    return candidate


def prior_fy(history: list, asof_fy_row):
    """Row immediately before the as-of fiscal year (for F-Score deltas)."""
    if asof_fy_row is None:
        return None
    idx = None
    for i, row in enumerate(history):
        if row["fy_end"] == asof_fy_row["fy_end"]:
            idx = i
            break
    return history[idx - 1] if idx and idx > 0 else None
