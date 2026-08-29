#!/usr/bin/env python3
"""Quarterly momentum advisor (Stage 2) -> single self-contained HTML page.

Scans the market every quarter (4th of Feb/May/Aug/Nov) and emits ONE interactive HTML
file (advisor.html) that:

  * embeds the current JT-1993 momentum target portfolio (top decile by
    12-month momentum, above 200-day SMA, equal weight, capped at N names),
    current prices, stoploss level, and the rules / playbook;
  * lets you RECORD what you have actually bought (ticker, quantity, buy
    price, buy date) directly in the page -- edits persist in your browser's
    localStorage;
  * recomputes live (JavaScript) the BUY / TOP-UP / HOLD / SELL recommendation,
    exact whole-share quantities, per-holding stoploss, and unrealized P&L
    against your entered buy price.

Usage:
    python3 advisor.py [--cash 50000] [--max-stocks 10] [--stoploss 0.07]
                       [--end 2026-08-01] [--out advisor.html]
"""
import argparse
import calendar
import csv
import html as _html
import json
import os
import smtplib
import sys
import time as _time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from momentum_engine import MomentumConfig, formation_returns, run_momentum
from quantbacktest import data as qdata

HOLDINGS_TEMPLATE = "ticker,quantity,avg_price,entry_date\n"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class AdvisorConfig:
    universe: str = "nifty500"
    start_date: str = ""
    end_date: str = ""
    holdings_path: str = "holdings.csv"
    cash: float = 0.0
    max_stocks: int = 10
    stoploss: float = 0.07
    model: str = "A"
    j_months: int = 12
    k_months: int = 3
    sma_window: int = 200
    min_price_days: int = 273
    cache_dir: str = "cache"
    out_path: str = "advisor.html"
    port: int = 8765
    email_config: str = "mail_config.json"


@dataclass
class Position:
    ticker: str
    quantity: float
    avg_price: float
    entry_date: str


@dataclass
class Action:
    ticker: str
    action: str
    reason: str
    quantity: float = 0.0
    current_price: float = 0.0
    amount: float = 0.0
    entry_price: float = 0.0
    pnl_pct: float = 0.0
    target_weight: float = 0.0
    rank: int = 0
    stoploss_price: float = 0.0


# ---------------------------------------------------------------------------
# Holdings file (seed only -- the HTML page is the primary editor)
# ---------------------------------------------------------------------------
def load_holdings(path: str) -> List[Position]:
    """Read holdings.csv (ticker,quantity,avg_price,entry_date) as a seed.

    Creates the template file if it does not exist.
    """
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(HOLDINGS_TEMPLATE)
        return []
    positions = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row or not row.get("ticker"):
                continue
            ticker = str(row["ticker"]).strip()
            if not ticker.endswith(".NS"):
                ticker += ".NS"
            positions.append(Position(
                ticker=ticker,
                quantity=float(row.get("quantity", 0) or 0),
                avg_price=float(row.get("avg_price", 0) or 0),
                entry_date=str(row.get("entry_date", "")).strip(),
            ))
    return positions


# ---------------------------------------------------------------------------
# Reconciliation (used by the CLI summary and tests; the HTML re-implements
# the same rules in JavaScript so it can recompute live as holdings change)
# ---------------------------------------------------------------------------
def reconcile(target_weights: Dict[str, float],
              target_ranks: Dict[str, int],
              holdings: List[Position],
              current_price: Dict[str, float],
              stoploss: float,
              new_cash: float) -> dict:
    """Classify holdings vs the target portfolio (top-up only).

    Returns dict with keys:
      actions, cash_left, sell_proceeds, hold_value, missing_price
    """
    held = {p.ticker: p for p in holdings}
    actions: List[Action] = []
    missing_price: List[str] = []
    sell_proceeds = 0.0
    hold_value = 0.0
    wsum = sum(target_weights.values()) or 1.0

    for p in holdings:
        px = current_price.get(p.ticker)
        if px is None or pd.isna(px) or px <= 0:
            missing_price.append(p.ticker)
            actions.append(Action(p.ticker, "HOLD",
                                  "no current price (delisted or download failed)",
                                  quantity=p.quantity))
            continue
        pnl = px / p.avg_price - 1.0 if p.avg_price > 0 else 0.0
        stop = p.avg_price * (1.0 - stoploss)
        if pnl < -stoploss:
            actions.append(Action(p.ticker, "SELL",
                                  f"stoploss hit ({pnl:.1%} below entry)",
                                  quantity=p.quantity, current_price=px,
                                  amount=p.quantity * px, entry_price=p.avg_price,
                                  pnl_pct=pnl, stoploss_price=stop))
            sell_proceeds += p.quantity * px
        elif p.ticker not in target_weights:
            actions.append(Action(p.ticker, "SELL",
                                  "dropped out of top-N momentum / below SMA-200",
                                  quantity=p.quantity, current_price=px,
                                  amount=p.quantity * px, entry_price=p.avg_price,
                                  pnl_pct=pnl, stoploss_price=stop))
            sell_proceeds += p.quantity * px
        else:
            actions.append(Action(p.ticker, "HOLD",
                                  "in target, above stoploss",
                                  quantity=p.quantity, current_price=px,
                                  amount=p.quantity * px, entry_price=p.avg_price,
                                  pnl_pct=pnl, stoploss_price=stop,
                                  target_weight=target_weights[p.ticker],
                                  rank=target_ranks.get(p.ticker, 999)))
            hold_value += p.quantity * px

    total_target = hold_value + sell_proceeds + new_cash
    remaining_cash = new_cash + sell_proceeds

    held_in_target = {a.ticker: a for a in actions
                      if a.action == "HOLD" and a.ticker in target_weights}
    candidates: List[tuple] = []
    for t, w in target_weights.items():
        target_val = total_target * (w / wsum)
        if t in missing_price:
            continue
        if t in held and t not in held_in_target:
            continue
        if t in held_in_target:
            cur_val = held_in_target[t].amount
            top_up = target_val - cur_val
            if top_up > 0:
                candidates.append((t, "TOP-UP", top_up, target_val))
        else:
            candidates.append((t, "BUY", target_val, target_val))

    candidates.sort(key=lambda c: target_ranks.get(c[0], 999))

    for t, kind, amount, target_val in candidates:
        px = current_price[t]
        if px <= 0:
            continue
        qty = int(amount // px)
        if qty <= 0:
            continue
        cost = qty * px
        if cost > remaining_cash:
            qty = int(remaining_cash // px)
            cost = qty * px
            if qty <= 0:
                continue
        remaining_cash -= cost
        reason = ("new entrant in top-N momentum" if kind == "BUY"
                  else "under target weight, top up")
        actions.append(Action(t, kind, reason, quantity=qty, current_price=px,
                              amount=cost, target_weight=target_weights[t],
                              rank=target_ranks.get(t, 999),
                              stoploss_price=px * (1.0 - stoploss)))

    return {
        "actions": actions,
        "cash_left": remaining_cash,
        "sell_proceeds": sell_proceeds,
        "hold_value": hold_value,
        "missing_price": missing_price,
    }


# ---------------------------------------------------------------------------
# HTML report (single self-contained interactive page)
# ---------------------------------------------------------------------------
def build_payload(cfg: AdvisorConfig, target_weights: Dict[str, float],
                  target_ranks: Dict[str, int], prices: Dict[str, float],
                  as_of: str, seed_holdings: List[Position],
                  scan_time: str, ret_1y: Dict[str, float] = None) -> dict:
    """Everything the HTML page needs to render + recompute recommendations."""
    target = sorted(
        [{"ticker": t, "weight": round(target_weights[t], 6),
          "rank": target_ranks.get(t, 999)} for t in target_weights],
        key=lambda d: d["rank"])
    return {
        "as_of": as_of,
        "scan_time": scan_time,
        "ret_1y": ret_1y or {},
        "model": cfg.model,
        "j": cfg.j_months,
        "k": cfg.k_months,
        "max_stocks": cfg.max_stocks,
        "stoploss": cfg.stoploss,
        "cash": cfg.cash,
        "target": target,
        "prices": {t: float(v) for t, v in prices.items() if pd.notna(v)},
        "seed": [{"ticker": p.ticker, "quantity": p.quantity,
                  "avg_price": p.avg_price, "entry_date": p.entry_date}
                 for p in seed_holdings],
    }


_CSS = """
:root { --buy:#1a7f37; --sell:#cf222e; --hold:#9a6700; --ink:#1f2328;
        --muted:#656d76; --line:#d0d7de; --bg:#ffffff; --panel:#f6f8fa; }
* { box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
       color:var(--ink); background:var(--bg); margin:0; padding:24px; line-height:1.45; }
.wrap { max-width:1040px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 2px; }
.sub { color:var(--muted); font-size:13px; margin-bottom:18px; }
.cards { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px;
        padding:12px 16px; min-width:140px; }
.card .k { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
.card .v { font-size:19px; font-weight:600; margin-top:2px; }
.card .s { font-size:12px; color:var(--muted); margin-top:1px; }
h2 { font-size:16px; margin:24px 0 10px; }
table { width:100%; border-collapse:collapse; font-size:13px; margin-bottom:6px; }
th, td { text-align:right; padding:7px 9px; border-bottom:1px solid var(--line); }
th { background:var(--panel); font-size:11px; text-transform:uppercase;
     letter-spacing:.04em; color:var(--muted); }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align:left; }
tr:hover td { background:#fafbfc; }
.buy { color:var(--buy); } .sell { color:var(--sell); } .hold { color:var(--hold); }
.tag { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; font-weight:600; }
.tag.buy { background:#dafbe1; color:var(--buy); }
.tag.sell { background:#ffebe9; color:var(--sell); }
.tag.hold { background:#fff8c5; color:var(--hold); }
input[type=text], input[type=number], input[type=date] {
  width:100%; padding:5px 7px; border:1px solid var(--line); border-radius:6px;
  font-size:13px; color:var(--ink); background:#fff; }
input.cash { width:160px; font-weight:600; }
.editor td { padding:3px 6px; }
.editor tr td:last-child { text-align:center; }
button { border:1px solid var(--line); background:var(--panel); border-radius:6px;
         padding:6px 14px; font-size:13px; cursor:pointer; color:var(--ink); }
button:hover { background:#eaeef2; }
button.danger { color:var(--sell); }
.scan-status { display:none; margin:6px 0 0; padding:10px 14px; border-radius:8px; font-size:13px; font-weight:500; }
.scan-status.ok { background:#dafbe1; color:var(--buy); border:1px solid #4ac26b; }
.scan-status.err { background:#ffebe9; color:var(--sell); border:1px solid #f85149; }
.notes { background:var(--panel); border:1px solid var(--line); border-radius:8px;
         padding:12px 16px; font-size:13px; color:var(--muted); margin-top:8px; }
.rules { background:var(--panel); border:1px solid var(--line); border-radius:8px;
         padding:4px 20px 14px; font-size:13.5px; }
.rules h3 { font-size:14px; margin:18px 0 4px; }
.rules ol, .rules ul { margin:4px 0 0; padding-left:20px; }
.rules li { margin:3px 0; }
.footer { margin-top:26px; font-size:11.5px; color:var(--muted);
          border-top:1px solid var(--line); padding-top:10px; }
.tab-bar { display:flex; gap:6px; border-bottom:2px solid var(--line); margin:6px 0 18px; }
.tab { border:1px solid var(--line); border-bottom:none; border-radius:8px 8px 0 0;
       padding:9px 22px; font-size:14px; font-weight:600; color:var(--muted); background:var(--panel); }
.tab.active { background:#fff; color:var(--buy); border-color:var(--buy); }
@media print { body { padding:0; } .wrap { max-width:none; } }
"""

_JS = """
const DATA = JSON.parse(document.getElementById('advisor-data').textContent);
const LS_KEY = 'momentum_advisor_holdings_v1';
const LS_CASH = 'momentum_advisor_cash_v1';

let holdings = loadHoldings();
let newCash = loadCash();

function loadHoldings() {
  const raw = localStorage.getItem(LS_KEY);
  if (raw) { try { return JSON.parse(raw); } catch(e) {} }
  return (DATA.seed || []).map(h => ({ ticker: h.ticker, quantity: h.quantity,
                                       avg_price: h.avg_price, entry_date: h.entry_date || '' }));
}
function saveHoldings() { localStorage.setItem(LS_KEY, JSON.stringify(holdings)); }
function loadCash() {
  const raw = localStorage.getItem(LS_CASH);
  return raw !== null ? Number(raw) : (DATA.cash || 0);
}
function saveCash() { localStorage.setItem(LS_CASH, String(newCash)); }

function normTicker(s) {
  let t = String(s || '').trim().toUpperCase().replace(/\\s+/g, '');
  if (t && !t.endsWith('.NS')) t += '.NS';
  return t;
}
function fmtMoney(v) { return 'Rs ' + Math.round(v || 0).toLocaleString('en-IN'); }
function fmtPct(v) { return (v === null || v === undefined || isNaN(v)) ? 'n/a' : (v * 100).toFixed(2) + '%'; }
function fmtQty(v) { return Number(v || 0).toLocaleString('en-IN', { maximumFractionDigits: 2 }); }
function ret1yCell(t) {
  const v = DATA.ret_1y[t];
  if (v === undefined || v === null || isNaN(v)) return '<td>n/a</td>';
  return `<td class="${v >= 0 ? 'buy' : 'sell'}">${fmtPct(v)}</td>`;
}

// ---------- holdings editor ----------
function fmtSignedMoney(v) {
  const n = Math.round(v || 0);
  return (n >= 0 ? '+' : '') + 'Rs ' + Math.abs(n).toLocaleString('en-IN');
}
function renderEditor() {
  const tbody = document.getElementById('holdings-body');
  tbody.innerHTML = '';
  let totQty = 0, totInv = 0, totVal = 0;
  holdings.forEach((h, i) => {
    const tr = document.createElement('tr');
    const px = DATA.prices[h.ticker];
    const cur = px ? fmtMoney(px) : 'n/a';
    const invVal = (h.quantity && h.avg_price) ? h.quantity * h.avg_price : 0;
    const pnlPct = (px && h.avg_price) ? fmtPct(px / h.avg_price - 1) : 'n/a';
    const pnlRs = (px && h.avg_price) ? (px - h.avg_price) * h.quantity : 0;
    const pnlRsFmt = (px && h.avg_price) ? fmtSignedMoney(pnlRs) : 'n/a';
    const stop = (px && h.avg_price) ? fmtMoney(h.avg_price * (1 - DATA.stoploss)) : 'n/a';
    const curVal = (px && h.quantity) ? h.quantity * px : invVal;
    totQty += h.quantity || 0; totInv += invVal; totVal += curVal;
    tr.innerHTML =
      `<td><input type="text" value="${h.ticker.replace('.NS', '')}" data-i="${i}" data-f="ticker"></td>` +
      `<td><input type="number" value="${h.quantity}" data-i="${i}" data-f="quantity"></td>` +
      `<td><input type="number" value="${h.avg_price}" data-i="${i}" data-f="avg_price"></td>` +
      `<td>${fmtMoney(invVal)}</td>` +
      `<td><input type="date" value="${h.entry_date || ''}" data-i="${i}" data-f="entry_date"></td>` +
      `<td>${cur}</td>` + ret1yCell(h.ticker) +
      `<td>${pnlPct}</td><td>${pnlRsFmt}</td><td>${stop}</td>` +
      `<td><button class="danger" data-i="${i}" data-del="1">x</button></td>`;
    tbody.appendChild(tr);
  });
  const tfoot = document.getElementById('holdings-total');
  if (tfoot) {
    const totPnl = totVal - totInv;
    const totPnlPct = totInv ? (totPnl / totInv * 100).toFixed(2) + '%' : '0.00%';
    tfoot.innerHTML = '<tr style="font-weight:700"><td>TOTAL</td><td>' + fmtQty(totQty) +
      '</td><td></td><td>' + fmtMoney(totInv) + '</td><td></td><td>' + fmtMoney(totVal) +
      '</td><td></td><td>' + totPnlPct + '</td><td>' + fmtSignedMoney(totPnl) +
      '</td><td></td><td></td></tr>';
  }
}

function addRow() {
  holdings.push({ ticker: '', quantity: 1, avg_price: 0, entry_date: '' });
  renderEditor(); saveHoldings(); recompute();
}

document.addEventListener('input', (e) => {
  if (e.target.dataset.i === undefined) return;
  const i = +e.target.dataset.i;
  const f = e.target.dataset.f;
  if (f === 'ticker') holdings[i].ticker = normTicker(e.target.value);
  if (f === 'quantity') holdings[i].quantity = Number(e.target.value) || 0;
  if (f === 'avg_price') holdings[i].avg_price = Number(e.target.value) || 0;
  if (f === 'entry_date') holdings[i].entry_date = e.target.value;
  saveHoldings(); recompute();
});
document.addEventListener('click', (e) => {
  if (e.target.dataset.del !== undefined) {
    holdings.splice(+e.target.dataset.i, 1);
    renderEditor(); saveHoldings(); recompute();
  }
});

// ---------- cash ----------
const cashInput = document.getElementById('cash-input');
if (cashInput) {
  cashInput.value = newCash;
  cashInput.addEventListener('input', () => {
    newCash = Number(cashInput.value) || 0;
    saveCash(); recompute();
  });
}

// ---------- reconciliation ----------
function recompute() {
  const tw = DATA.target;
  const prices = DATA.prices;
  const stoploss = DATA.stoploss;
  const wsum = tw.reduce((s, t) => s + t.weight, 0) || 1;

  let sellProceeds = 0, holdValue = 0, costBasis = 0, holdingsValue = 0;
  const missing = [];
  const held = {};        // ticker -> entered holding (kept as HOLD)
  const holds = [], sells = [];

  for (const h of holdings) {
    if (!h.ticker) continue;
    const px = prices[h.ticker];
    holdingsValue += h.quantity * (px || h.avg_price || 0);
    if (h.avg_price) costBasis += h.quantity * h.avg_price;
    const tgt = tw.find(t => t.ticker === h.ticker);
    if (!px) {
      missing.push(h.ticker);
      holds.push({ ticker: h.ticker, qty: h.quantity, price: null, avg: h.avg_price,
                   pnl: null, stop: null, weight: null, rank: null });
      held[h.ticker] = h;
      continue;
    }
    const pnl = px / h.avg_price - 1;
    const stop = h.avg_price * (1 - stoploss);
    if (pnl < -stoploss) {
      sells.push({ ticker: h.ticker, qty: h.quantity, price: px, avg: h.avg_price,
                   pnl, stop, reason: 'stoploss hit (' + fmtPct(pnl) + ' below entry)' });
      sellProceeds += h.quantity * px;
    } else if (!tgt) {
      sells.push({ ticker: h.ticker, qty: h.quantity, price: px, avg: h.avg_price,
                   pnl, stop, reason: 'dropped out of top-' + DATA.max_stocks + ' momentum / below SMA-200' });
      sellProceeds += h.quantity * px;
    } else {
      holds.push({ ticker: h.ticker, qty: h.quantity, price: px, avg: h.avg_price,
                   pnl, stop, weight: tgt.weight, rank: tgt.rank });
      holdValue += h.quantity * px;
      held[h.ticker] = h;
    }
  }

  const totalTarget = holdValue + sellProceeds + newCash;
  let remaining = sellProceeds + newCash;
  const buys = [];
  const sorted = [...tw].sort((a, b) => a.rank - b.rank);
  for (const t of sorted) {
    const px = prices[t.ticker];
    if (!px) continue;
    const targetVal = totalTarget * (t.weight / wsum);
    let amount, kind;
    const h = held[t.ticker];
    if (h) {
      const cur = h.quantity * px;
      const top = targetVal - cur;
      if (top <= 0) continue;
      amount = top; kind = 'TOP-UP';
    } else {
      amount = targetVal; kind = 'BUY';
    }
    let qty = Math.floor(amount / px);
    if (qty <= 0) continue;
    let cost = qty * px;
    const shortfall = cost > remaining;
    if (shortfall) { qty = Math.floor(remaining / px); cost = qty * px; }
    if (qty > 0) remaining -= cost;
    buys.push({ ticker: t.ticker, kind, qty: qty || Math.floor(amount / px), price: px,
                amount: cost || amount, weight: t.weight, rank: t.rank,
                stop: px * (1 - stoploss), shortfall });
  }

  const bookedPnl = sells.reduce((s, sl) => s + (sl.price - sl.avg) * sl.qty, 0);
  const totalPnl = (holdingsValue - costBasis) + bookedPnl;

  renderCards(holdValue, holdingsValue, costBasis, remaining, sellProceeds,
              buys, holds.length, missing, bookedPnl, totalPnl);
  renderTable('buys-body', buys, buysRow);
  renderTable('holds-body', holds, holdRow);
  renderTable('sells-body', sells, sellRow);
  renderMissing(missing);
}

function renderCards(holdValue, holdingsValue, costBasis, cashLeft, sellProceeds,
                     buys, nHolds, missing, bookedPnl, totalPnl) {
  const buyTotal = buys.reduce((s, b) => s + b.amount, 0);
  setText('v-portfolio', fmtMoney(holdingsValue + newCash), 'your holdings + new cash');
  setText('v-invested', fmtMoney(costBasis), nHolds + ' held / ' + DATA.target.length + ' target');
  setText('v-cash', fmtMoney(cashLeft), 'cash after executing the plan');
  setText('v-buys', fmtMoney(buyTotal), buys.length + ' order' + (buys.length === 1 ? '' : 's'));
  setText('v-sells', fmtMoney(sellProceeds), 'proceeds from sells');
  const openPnl = holdingsValue - costBasis;
  setText('v-pnl', fmtMoney(openPnl), fmtPct(costBasis ? openPnl / costBasis : 0) + ' on cost',
          openPnl >= 0 ? 'buy' : 'sell');
  setText('v-advisor-pnl', fmtMoney(openPnl), fmtPct(costBasis ? openPnl / costBasis : 0) + ' on cost',
          openPnl >= 0 ? 'buy' : 'sell');
  setText('v-realized', fmtMoney(bookedPnl), bookedPnl >= 0 ? 'realized gain' : 'realized loss',
          bookedPnl >= 0 ? 'buy' : 'sell');
  setText('v-total-pnl', fmtMoney(totalPnl),
          fmtPct(costBasis ? totalPnl / costBasis : 0) + ' total',
          totalPnl >= 0 ? 'buy' : 'sell');
}
function setText(id, v, sub, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = v;
  const s = el.parentElement ? el.parentElement.querySelector('.s') : null;
  if (s) s.textContent = sub || '';
  if (cls) el.style.color = getComputedStyle(document.body).getPropertyValue('--' + cls).trim();
}
function renderTable(id, rows, rowFn) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = rows.map(rowFn).join('');
}
function buysRow(b) {
  const tag = b.shortfall
    ? `<span class="tag sell">needs cash</span>`
    : `<span class="tag buy">${b.kind}</span>`;
  return `<tr><td>${b.ticker.replace('.NS', '')}</td>` +
    `<td>${tag}</td>` +
    `<td>#${b.rank}</td><td>${fmtMoney(b.price)}</td>` + ret1yCell(b.ticker) +
    `<td>${fmtQty(b.qty)}</td>` +
    `<td>${fmtMoney(b.amount)}</td><td>${fmtPct(b.weight)}</td><td>${fmtMoney(b.stop)}</td></tr>`;
}
function holdRow(h) {
  if (!h.price) {
    return `<tr><td>${h.ticker.replace('.NS', '')}</td><td>${fmtQty(h.qty)}</td>` +
      `<td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td></tr>`;
  }
  return `<tr><td>${h.ticker.replace('.NS', '')}</td><td>${fmtQty(h.qty)}</td>` +
    `<td>${fmtMoney(h.avg)}</td><td>${fmtMoney(h.price)}</td>` + ret1yCell(h.ticker) +
    `<td class="${h.pnl < 0 ? 'sell' : 'buy'}">${fmtPct(h.pnl)}</td>` +
    `<td>${fmtMoney(h.stop)}</td><td>${fmtPct(h.weight)}</td></tr>`;
}
function sellRow(s) {
  return `<tr><td>${s.ticker.replace('.NS', '')}</td><td>${fmtQty(s.qty)}</td>` +
    `<td>${fmtMoney(s.price)}</td>` + ret1yCell(s.ticker) +
    `<td class="${s.pnl < 0 ? 'sell' : 'buy'}">${fmtPct(s.pnl)}</td>` +
    `<td>${fmtMoney(s.qty * s.price)}</td><td style="text-align:left">${s.reason}</td></tr>`;
}
function renderMissing(missing) {
  const el = document.getElementById('missing-note');
  if (missing.length) {
    el.style.display = '';
    el.textContent = 'No current price for: ' + missing.map(t => t.replace('.NS', '')).join(', ') +
      '. Kept HOLD with n/a values — resolve delisting / failed download before next scan.';
  } else {
    el.style.display = 'none';
  }
}

function setStatus(msg, isError) {
  document.querySelectorAll('.scan-status').forEach(el => {
    el.className = 'scan-status ' + (isError ? 'err' : (msg ? 'ok' : ''));
    el.textContent = msg;
    el.style.display = msg ? '' : 'none';
  });
}

// ---------- next scan date (4th of Feb/May/Aug/Nov) ----------
function nextQuarterlyScanDate() {
  const now = new Date();
  const months = [1, 4, 7, 10]; // Feb=1, May=4, Aug=7, Nov=10 (0-indexed)
  const names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  for (let offset = 0; offset < 4; offset++) {
    const m = months[(now.getMonth() + offset) % 4];
    const y = now.getFullYear() + Math.floor((now.getMonth() + offset) / 12);
    // If same month and date already passed, skip to next quarter
    const d = new Date(y, m, 4);
    if (d >= new Date(now.getFullYear(), now.getMonth(), now.getDate())) {
      const iso = d.getFullYear() + '-' +
        String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
      return names[d.getDay()] + ' ' + iso;
    }
  }
  return '';
}
function renderNextScan() {
  const el = document.getElementById('next-scan');
  if (el) el.textContent = nextQuarterlyScanDate();
}

// ---------- CSV / Excel tradebook import ----------
function splitCsvLine(line) {
  const result = []; let cur = ''; let inQ = false;
  for (let c = 0; c < line.length; c++) {
    const ch = line[c];
    if (inQ) {
      if (ch === '"' && line[c+1] === '"') { cur += '"'; c++; }
      else if (ch === '"') inQ = false;
      else cur += ch;
    } else {
      if (ch === '"') inQ = true;
      else if (ch === ',') { result.push(cur); cur = ''; }
      else cur += ch;
    }
  }
  result.push(cur);
  return result;
}
function csvTicker(fields) {
  let t = (fields[0] || '').trim().toUpperCase().replace(/\\s+/g, '');
  t = t.replace(/-BE$|-EQ$|-BZ$/i, '');
  if (t && !t.endsWith('.NS')) t += '.NS';
  return t;
}
function csvNum(fields, idx) {
  const v = (fields[idx] || '').trim().replace(/,/g, '');
  return v ? Number(v) : 0;
}
function normalizeDate(s) {
  if (!s) return '';
  s = s.trim();
  if (/^\\d{4}-\\d{2}-\\d{2}$/.test(s)) return s;
  const m = s.match(/^(\\d{2})\\/(\\d{2})\\/(\\d{4})$/);
  if (m) return m[3] + '-' + m[2] + '-' + m[1];
  return s;
}
function parseTradebookCsv(text) {
  const lines = text.trim().split('\\n').filter(l => l.trim());
  if (!lines.length) return { holdings: [], booked: {}, totalBooked: 0 };
  const hdr = splitCsvLine(lines[0]).map(h => h.trim().toLowerCase());
  const ti = hdr.indexOf('symbol') !== -1 ? hdr.indexOf('symbol') : 0;
  const qi = hdr.indexOf('qty') !== -1 ? hdr.indexOf('qty') : (hdr.indexOf('quantity') !== -1 ? hdr.indexOf('quantity') : 1);
  const pi = hdr.indexOf('price') !== -1 ? hdr.indexOf('price') : (hdr.indexOf('avg.') !== -1 ? hdr.indexOf('avg.') : 2);
  const di = hdr.indexOf('trade_date') !== -1 ? hdr.indexOf('trade_date') : (hdr.indexOf('date') !== -1 ? hdr.indexOf('date') : -1);
  const tti = hdr.indexOf('trade_type') !== -1 ? hdr.indexOf('trade_type') : -1;
  const buys = {}, sells = {};
  for (let i = 1; i < lines.length; i++) {
    const f = splitCsvLine(lines[i]);
    const tk = csvTicker(f);
    if (!tk) continue;
    const qty = Math.abs(csvNum(f, qi));
    const price = csvNum(f, pi);
    const dt = di >= 0 ? normalizeDate(f[di]) : '';
    const tradeType = tti >= 0 ? (f[tti] || '').trim().toUpperCase() : '';
    if (tradeType === 'SELL' || tradeType === 'S' || qty < 0) {
      const sq = Math.abs(qty);
      if (!sells[tk]) sells[tk] = { qty: 0, proceeds: 0, avgBuy: 0, buyQty: 0 };
      sells[tk].qty += sq;
      sells[tk].proceeds += sq * price;
    } else {
      if (!buys[tk]) buys[tk] = { qty: 0, totalCost: 0, avgPrice: 0, date: '' };
      const oldTotal = buys[tk].qty * buys[tk].avgPrice;
      buys[tk].qty += qty;
      buys[tk].totalCost = oldTotal + qty * price;
      buys[tk].avgPrice = buys[tk].totalCost / buys[tk].qty;
      if (dt && !buys[tk].date) buys[tk].date = dt;
    }
  }
  for (const tk in sells) {
    if (buys[tk]) sells[tk].avgBuy = buys[tk].avgPrice;
  }
  const result = [];
  for (const tk in buys) {
    const b = buys[tk];
    const held = b.qty - (sells[tk] ? sells[tk].qty : 0);
    if (held > 0) {
      result.push({ ticker: tk, quantity: held, avg_price: b.avgPrice, entry_date: b.date });
    }
  }
  let totalBooked = 0;
  const booked = {};
  for (const tk in sells) {
    const s = sells[tk];
    const pnl = (s.avgBuy > 0) ? (s.proceeds - s.qty * s.avgBuy) : 0;
    booked[tk] = pnl;
    totalBooked += pnl;
  }
  return { holdings: result, booked: booked, totalBooked: totalBooked };
}
function promptCsvLoad() {
  const fi = document.getElementById('csv-file');
  if (fi) fi.click();
}
const _csvEl = document.getElementById('csv-file');
if (_csvEl) _csvEl.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  e.target.value = '';
  const name = file.name.toLowerCase();
  if (name.endsWith('.xlsx') || name.endsWith('.xls')) {
    setStatus('Uploading Excel file…', false);
    const fd = new FormData(); fd.append('file', file);
    try {
      const r = await fetch('/import-xlsx', { method: 'POST', body: fd });
      if (!r.ok) { const t = await r.text(); setStatus('Excel import failed: ' + t, true); return; }
      const csvText = await r.text();
      applyCsvText(csvText);
    } catch (err) { setStatus('Excel upload error: ' + err.message, true); }
    return;
  }
  const reader = new FileReader();
  reader.onload = (ev) => applyCsvText(ev.target.result);
  reader.readAsText(file);
});
function applyCsvText(text) {
  const res = parseTradebookCsv(text);
  if (!res.holdings.length && !res.totalBooked) {
    setStatus('No valid trade rows found. Expected a Zerodha tradebook with columns: symbol, quantity, price, trade_type, trade_date. The file you uploaded may be a P&L report instead of a tradebook.', true);
    return;
  }
  let replaced = 0, added = 0;
  for (const h of res.holdings) {
    const idx = holdings.findIndex(x => x.ticker === h.ticker);
    if (idx >= 0) { holdings[idx] = h; replaced++; }
    else { holdings.push(h); added++; }
  }
  saveHoldings(); renderEditor(); recompute();
  let msg = 'Loaded ' + res.holdings.length + ' holding' + (res.holdings.length === 1 ? '' : 's') +
            ' (' + added + ' new, ' + replaced + ' updated)';
  if (res.totalBooked) msg += '. Booked P&L: ' + fmtSignedMoney(res.totalBooked);
  setStatus(msg, false);
}

// ---------- export recommendations as CSV (server-side) ----------
function exportCsv() {
  window.location.href = '/export';
}

// ---------- tab switching ----------
function showTab(name) {
  document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  const pane = document.getElementById('tab-' + name);
  if (pane) pane.style.display = '';
}

try { renderNextScan(); renderEditor(); recompute(); showTab('portfolio'); }
catch (e) { setStatus('Page script error: ' + (e && e.message ? e.message : e), true); }

// bind export button via listener (onclick can be unreliable across tabs)
var _exportBtn = document.getElementById('export-btn');
if (_exportBtn) _exportBtn.addEventListener('click', exportCsv);



// ---------- run scan (POST /scan to the local advisor server) ----------
async function runScan() {
  const btn = document.getElementById('scan-btn');
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  btn.textContent = 'Scanning…';
  if (location.protocol !== 'http:' && location.protocol !== 'https:') {
    setStatus('This page was opened as a file. To use Run scan, start the local server ' +
              'with "python3 advisor.py --serve" and open http://localhost:8765/advisor.html.', true);
    btn.disabled = false; btn.textContent = 'Run scan';
    return;
  }
  setStatus('Scanning — refreshing prices and momentum targets…', false);
  try {
    const r = await fetch('/scan', { method: 'POST' });
    const j = await r.json();
    if (j.status === 'ok') {
      setStatus('Scan complete (data as of ' + (j.as_of || '?') + ') — reloading…', false);
      setTimeout(() => { window.location.href = 'advisor.html?t=' + Date.now(); }, 700);
    } else {
      setStatus('Scan failed: ' + (j.message || 'unknown error'), true);
      btn.disabled = false; btn.textContent = 'Run scan';
    }
  } catch (e) {
    setStatus('Could not reach the scan server. Is "python3 advisor.py --serve" running?', true);
    btn.disabled = false; btn.textContent = 'Run scan';
  }
}
"""


def build_html(cfg: AdvisorConfig, payload: dict, as_of: str) -> str:
    """Assemble the self-contained interactive HTML page."""
    rules = f"""
  <h3>When to scan</h3>
  <ol><li>Scan once a month, on the <b>last trading day of the month, after market close</b>.</li>
  <li>Run <code>python3 advisor.py --cash &lt;new cash&gt;</code> to refresh prices and re-embed the
  latest momentum targets into this page.</li>
  <li>Check prices <b>daily</b> (~2 minutes) only to catch stoploss breaks.</li></ol>

  <h3>When to buy — entry timing (per JT 1993)</h3>
  <ol><li><b>Rank once a month, at month-end.</b> At the end of each month <i>t-1</i>, rank the
  universe by the J-month formation return <code>P[t-1] / P[t-1-J] - 1</code> (J=12 here).</li>
  <li><b>Skip the most recent month.</b> The paper deliberately leaves out the last month because
  1-month (short-term) reversal is strong — month-1 returns after formation are not reliably
  positive. This is the "12-minus-1" rule.</li>
  <li><b>Buy the winners at the start of month <i>t</i></b> — the top decile by that return — at the
  <b>month-end close or the next trading day's open</b>. Do not buy mid-month or chase intraday
  moves.</li>
  <li><b>Hold for K months</b> (default 3) with a <b>1/K monthly roll</b>: each month-end sell the
  batch formed K months ago and buy the new winners. Momentum returns are strongest in months
  2-12 after formation, so do not flip early.</li></ol>

  <h3>When to rebalance</h3>
  <ol><li>Rebalance quarterly, on the 4th of Feb/May/Aug/Nov (after close), by executing this page's orders.</li>
  <li>Top-up only: keep existing shares; never trim an over-weighted holding while it remains in the
  target list. This keeps turnover (and tax/cost) low.</li>
  <li>New cash buys the highest-rank unmet BUY / TOP-UP orders first.</li></ol>

  <h3>When to exit (SELL rules)</h3>
  <ol><li><b>Stoploss {payload['stoploss'] * 100:.0f}%:</b> if any holding trades below
  <code>your buy price &times; (1 - {payload['stoploss']:.2f})</code>, exit it at the <b>next day's
  open</b> — do not wait for the monthly rebalance.</li>
  <li><b>Dropped out:</b> if a holding is no longer in the top {payload['max_stocks']} momentum names
  (or its price has fallen below the 200-day SMA), it is sold at the monthly rebalance.</li>
  <li>Never average down a stoploss hit. The stoploss price is shown per holding below.</li></ol>

  <h3>Buy rules</h3>
  <ol><li>Buy only stocks in the current top decile of 12-month momentum that also trade above their
  200-day SMA.</li>
  <li>Target at most {payload['max_stocks']} names, roughly equal weight
  ({100.0 / max(payload['max_stocks'], 1):.2f}% each).</li>
  <li>Whole shares only; leftover cash carries to next month.</li></ol>

  <h3>Risk notes</h3>
  <ul><li>Backtests use Nifty 500 constituents with 10-yr history — survivorship bias inflates
  results. Live results will differ.</li>
  <li>Concentrated momentum is volatile (10-yr max drawdown about -35%). The
  {payload['stoploss'] * 100:.0f}% stoploss is a hard risk floor.</li>
  <li>This is an aid, not investment advice. Prices are cached scan-day closes.</li></ul>
"""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Momentum Advisor - {as_of}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<h1>Momentum Advisor</h1>
<div class="sub">JT-1993 Model {payload['model']} (J={payload['j']}, K={payload['k']}) &middot;
target top {payload['max_stocks']} by 12-month momentum, above 200-day SMA, equal weight &middot;
data as of <b>{as_of}</b> &middot; <b>last scan: {payload['scan_time']}</b>.
<b>Next scan: <span id="next-scan">...</span>, after close</b>.
<button id="scan-btn" onclick="runScan()" style="float:right">Run scan</button>
<button id="export-btn" onclick="exportCsv()" style="float:right; margin-right:8px">Export CSV</button></div>
<div id="scan-status"></div>

<div class="tab-bar">
  <button class="tab active" data-tab="portfolio" onclick="showTab('portfolio')">Portfolio</button>
  <button class="tab" data-tab="advisor" onclick="showTab('advisor')">Advisor</button>
</div>

<div id="tab-portfolio" class="tab-pane" style="display: block;">
<h1>My Portfolio</h1>
<div class="sub">Your actual holdings and overall P&L &middot; data as of <b>{as_of}</b>
(last scan {payload['scan_time']}) &middot;
<a href="#" onclick="showTab('advisor'); return false;">Advisor page</a></div>

<div class="cards">
  <div class="card"><div class="k">Portfolio value</div><div class="v" id="v-portfolio">-</div><div class="s"></div></div>
  <div class="card"><div class="k">Invested</div><div class="v" id="v-invested">-</div><div class="s"></div></div>
  <div class="card"><div class="k">Open P&L (Rs)</div><div class="v" id="v-pnl">-</div><div class="s"></div></div>
  <div class="card"><div class="k">Booked P&L</div><div class="v" id="v-realized">-</div><div class="s"></div></div>
  <div class="card"><div class="k">Total P&L</div><div class="v" id="v-total-pnl">-</div><div class="s"></div></div>
</div>

<h2>What you own</h2>
<p class="notes">Edit rows directly or bulk-load:
<button onclick="addRow()">+ Add holding</button>&nbsp;
<button onclick="promptCsvLoad()">Load CSV/Excel (tradebook)</button>
<input type="file" id="csv-file" accept=".csv,text/csv,.xlsx,.xls" style="display:none">
&nbsp; Current price, Inv value, and P&L come from the last scan.</p>
<table class="editor">
<tr><th>Ticker</th><th>Qty</th><th>Buy price</th><th>Inv value</th><th>Buy date</th>
<th>Current</th><th>1Y ret</th><th>P&L %</th><th>P&L (Rs)</th><th>Stoploss</th><th></th></tr>
<tbody id="holdings-body"></tbody>
<tfoot id="holdings-total"></tfoot>
</table>
<div class="notes" id="missing-note" style="display:none"></div>
</div>

<div id="tab-advisor" class="tab-pane" style="display: none;">
<h1>Quarterly Momentum Advisor</h1>
<div class="sub">JT-1993 Model {payload['model']} (J={payload['j']}, K={payload['k']}) &middot;
target top {payload['max_stocks']} by 12-month momentum, above 200-day SMA, equal weight &middot;
data as of <b>{as_of}</b> &middot; <b>last scan: {payload['scan_time']}</b>.</div>

<div class="cards">
  <div class="card"><div class="k">Cash after plan</div><div class="v" id="v-cash">-</div><div class="s"></div></div>
  <div class="card"><div class="k">Buy orders</div><div class="v" id="v-buys">-</div><div class="s"></div></div>
  <div class="card"><div class="k">Sell proceeds</div><div class="v" id="v-sells">-</div><div class="s"></div></div>
  <div class="card"><div class="k">P&L vs cost</div><div class="v" id="v-advisor-pnl">-</div><div class="s"></div></div>
</div>

<p class="notes">Enter new cash for this quarter:
<input type="number" class="cash" id="cash-input" step="1000" min="0">&nbsp; (used to size the buy orders below)</p>

<h2>BUY / TOP-UP</h2>
<table>
<tr><th>Ticker</th><th>Action</th><th>Mom. rank</th><th>Price</th><th>1Y ret</th><th>Qty</th>
<th>Amount</th><th>Target wt</th><th>Stoploss</th></tr>
<tbody id="buys-body"></tbody>
</table>
<p class="notes">Quantity = whole shares only. New entrants are bought first, then top-ups for
under-weighted holdings, in momentum rank order.</p>

<h2>HOLD</h2>
<table>
<tr><th>Ticker</th><th>Qty</th><th>Avg entry</th><th>Current</th><th>1Y ret</th><th>P&L</th>
<th>Stoploss</th><th>Target wt</th></tr>
<tbody id="holds-body"></tbody>
</table>
<p class="notes">Keep these as-is. No trimming - over-weighted names are left alone until they drop
out of the target list.</p>

<h2>SELL</h2>
<table>
<tr><th>Ticker</th><th>Qty</th><th>Price</th><th>1Y ret</th><th>P&L</th><th>Proceeds</th><th>Reason</th></tr>
<tbody id="sells-body"></tbody>
</table>

<h2>Rules & Instructions</h2>
<div class="rules">{rules}</div>

<div class="footer">Generated {as_of} by advisor.py &middot; holdings live on the
<a href="#" onclick="showTab('portfolio'); return false;">Portfolio page</a> (same browser, shared storage) &middot; run
<code>python3 advisor.py --cash <amount></code> every 3 months (or start the server and click
<b>Run scan</b>) to refresh prices.</div>
</div>
<script type="application/json" id="advisor-data">{json.dumps(payload)}</script>
<script>{_JS}</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    today = date.today().isoformat()
    p = argparse.ArgumentParser(description="Quarterly momentum advisor -> interactive HTML page (NSE).")
    p.add_argument("--cash", type=float, default=0.0, help="New cash this month (default for the page).")
    p.add_argument("--max-stocks", type=int, default=10)
    p.add_argument("--stoploss", type=float, default=0.07)
    p.add_argument("--model", default="A", choices=["A", "B"])
    p.add_argument("--j", type=int, default=12)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--end", default=today)
    p.add_argument("--start", default="")
    p.add_argument("--holdings", default="holdings.csv", help="Seed holdings (optional).")
    p.add_argument("--out", default="advisor.html")
    p.add_argument("--serve", action="store_true",
                   help="Host the page locally and let the 'Run scan' button refresh prices.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--email", action="store_true",
                   help="After scanning, email the report (uses --email-config).")
    p.add_argument("--test-email", action="store_true",
                   help="Send a test email without scanning (verify credentials).")
    p.add_argument("--email-config", default="mail_config.json",
                   help="JSON with sender/app_password/recipient for the quarterly email.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scan + local server
# ---------------------------------------------------------------------------
def run_scan(cfg: AdvisorConfig) -> dict:
    """Full scan: download prices, compute the target portfolio, write the
    interactive HTML page. Returns a dict with the payload + summary info."""
    seed = load_holdings(cfg.holdings_path)
    print(f"Seed holdings from {cfg.holdings_path}: {len(seed)} | new cash: Rs {cfg.cash:,.0f} | "
          f"target: {cfg.max_stocks} | stoploss: {cfg.stoploss:.0%}")

    tickers = qdata.load_universe(cfg.universe)
    daily = qdata.download_prices(tickers, cfg.start_date, cfg.end_date,
                                  cache_dir=cfg.cache_dir)
    daily = qdata.clean_prices(daily, cfg.min_price_days)
    sma = daily.rolling(cfg.sma_window).mean()
    monthly_close = daily.resample("ME").last()
    monthly_sma = sma.resample("ME").last()
    monthly_ret = monthly_close.pct_change()

    cfg_engine = MomentumConfig(model=cfg.model, j_months=cfg.j_months,
                                k_months=cfg.k_months, max_stocks=cfg.max_stocks,
                                sma_trend_filter=True, sma_window=cfg.sma_window,
                                min_price_days=cfg.min_price_days,
                                cache_dir=cfg.cache_dir,
                                start_date=cfg.start_date, end_date=cfg.end_date)
    result = run_momentum(monthly_close, monthly_ret, monthly_sma, cfg_engine)

    last_w = result.weights.iloc[-1]
    target_weights = {t: float(w) for t, w in last_w.items() if w > 0.0001}
    formation = formation_returns(monthly_close, cfg.j_months).iloc[-1].dropna()
    formation = formation.sort_values(ascending=False)
    ranks = {t: i + 1 for i, t in enumerate(formation.index)}
    target_ranks = {t: ranks.get(t, 999) for t in target_weights}

    prices = {t: float(v) for t, v in daily.iloc[-1].items() if pd.notna(v)}
    # 1-year return: latest close vs close ~252 trading days ago
    ret_1y_series = daily.iloc[-1] / daily.shift(252).iloc[-1] - 1.0
    ret_1y = {t: float(v) for t, v in ret_1y_series.items()
              if pd.notna(v) and np.isfinite(v)}
    as_of = daily.index[-1].strftime("%Y-%m-%d")
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M") + " " + _time.tzname[0]

    print(f"Data through {as_of} | {len(target_weights)} target names: "
          f"{', '.join(t.replace('.NS', '') for t in sorted(target_weights, key=lambda t: target_ranks.get(t, 999)))}")

    payload = build_payload(cfg, target_weights, target_ranks, prices, as_of, seed, scan_time, ret_1y)
    html = build_html(cfg, payload, as_of)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.out_path)) or ".", exist_ok=True)
    with open(cfg.out_path, "w") as f:
        f.write(html)

    rec = reconcile(target_weights, target_ranks, seed, prices, cfg.stoploss, cfg.cash)
    return {"payload": payload, "as_of": as_of, "out_path": cfg.out_path,
            "target_ranks": target_ranks, "rec": rec}


class _ScanHandler(SimpleHTTPRequestHandler):
    """Serves the project directory + a POST /scan endpoint that re-runs the scan."""

    server_cfg = None

    def do_POST(self):
        if self.path == "/scan":
            try:
                out = run_scan(self.server_cfg)
                body = json.dumps({"status": "ok", "as_of": out["as_of"],
                                   "out_path": out["out_path"]}).encode()
                code = 200
            except Exception as e:
                body = json.dumps({"status": "error", "message": str(e)}).encode()
                code = 500
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/import-xlsx":
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not length:
                    raise ValueError("Empty upload")
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    raise ValueError("Expected multipart/form-data")
                boundary = content_type.split("boundary=")[1].encode()
                raw = self.rfile.read(length)
                # Extract the file from multipart body
                parts = raw.split(b"--" + boundary)
                csv_text = None
                for part in parts:
                    if b"filename=" in part:
                        # Find double CRLF then the file content
                        header_end = part.find(b"\r\n\r\n")
                        if header_end < 0:
                            continue
                        file_content = part[header_end + 4:]
                        # Strip trailing \r\n-- if present
                        if file_content.endswith(b"\r\n"):
                            file_content = file_content[:-2]
                        # Convert xlsx to csv using pandas
                        import io
                        df = pd.read_excel(io.BytesIO(file_content))
                        csv_text = df.to_csv(index=False)
                        break
                if csv_text is None:
                    raise ValueError("No file found in upload")
                body = csv_text.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                body = str(e).encode()
                self.send_response(400)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path.startswith("/export"):
            self._handle_export()
        else:
            super().do_GET()

    def _handle_export(self):
        """Read advisor.html, extract embedded JSON payload, compute recommendations, return CSV."""
        import io, csv as _csv_mod
        try:
            cfg = self.server_cfg
            out_path = os.path.abspath(cfg.out_path)
            with open(out_path, encoding="utf-8") as f:
                html = f.read()
            marker = 'id="advisor-data">'
            start = html.index(marker) + len(marker)
            end = html.index("</script>", start)
            data = json.loads(html[start:end])

            target = data["target"]
            prices = data["prices"]
            stoploss = data["stoploss"]
            wsum = sum(t["weight"] for t in target) or 1

            # Read holdings from embedded seed + localStorage won't work server-side,
            # so just output the target list with rankings
            buf = io.StringIO()
            w = _csv_mod.writer(buf)
            w.writerow(["Rank", "Ticker", "Target wt %", "Price (Rs)", "1Y ret %",
                         "Stoploss (Rs)", "Action"])
            for t in sorted(target, key=lambda x: x["rank"]):
                tk = t["ticker"]
                px = prices.get(tk, 0)
                ret = data.get("ret_1y", {}).get(tk)
                ret_str = f"{ret*100:.2f}" if ret is not None else ""
                stop = f"{px * (1 - stoploss):.0f}" if px else ""
                w.writerow([t["rank"], tk.replace(".NS", ""),
                            f"{t['weight']*100:.1f}", f"{px:.2f}" if px else "",
                            ret_str, stop, "BUY" if px else "n/a price"])

            body = buf.getvalue().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition",
                             f'attachment; filename="advisor_targets_{data.get("as_of","")}.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            body = str(e).encode()
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args):
        sys.stderr.write(" ".join(str(a) for a in args[:3]) + "\n")


def make_cfg(args) -> AdvisorConfig:
    return AdvisorConfig(
        cash=args.cash, max_stocks=args.max_stocks, stoploss=args.stoploss,
        model=args.model, j_months=args.j, k_months=args.k,
        end_date=args.end, out_path=args.out, holdings_path=args.holdings,
        port=args.port,
        start_date=args.start or (pd.Timestamp(args.end) - pd.Timedelta(days=550)).strftime("%Y-%m-%d"),
        email_config=args.email_config,
    )


# ---------------------------------------------------------------------------
# Monthly email report
# ---------------------------------------------------------------------------
EMAIL_KEYS = ("sender", "app_password", "recipient")


def load_email_config(path: str) -> dict:
    """Read {sender, app_password, recipient} from a JSON file (chmod 600)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Email config not found: {path}. Create it with sender/app_password/recipient.")
    with open(path) as f:
        cfg = json.load(f)
    missing = [k for k in EMAIL_KEYS if not cfg.get(k)]
    if missing:
        raise ValueError(f"Missing {', '.join(missing)} in {path}")
    return cfg


def last_weekday(year: int, month: int) -> date:
    """Last Mon-Fri of a month (the quarterly scan/rebalance date)."""
    d = date(year, month, calendar.monthrange(year, month)[1])
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def next_scan_date_str(today: Optional[date] = None) -> str:
    """Next quarterly scan: 4th of Feb/May/Aug/Nov."""
    today = today or date.today()
    y = today.year
    for m in (2, 5, 8, 11):
        d = date(y, m, 4)
        if d >= today:
            return d.strftime("%A %Y-%m-%d")
    return date(y + 1, 2, 4).strftime("%A %Y-%m-%d")


def summary_text(payload: dict, rec: dict, as_of: str, next_scan: str) -> str:
    """Plain-text body for the quarterly email, mirroring the CLI summary."""
    target = payload.get("target", [])
    lines = [
        "MOMENTUM ADVISOR - QUARTERLY REPORT",
        "=" * 40,
        f"Data as of        : {as_of}",
        f"Last scan         : {payload.get('scan_time', '')}",
        f"Next scan         : {next_scan}",
        f"Model             : {payload.get('model', 'A')} | target {payload.get('max_stocks', 10)} stocks",
        f"Stoploss          : {payload.get('stoploss', 0.07):.0%}",
        f"New cash (plan)   : Rs {payload.get('cash', 0):,.0f}",
        "",
        f"Target names      : {len(target)}",
    ]
    buys = [a for a in rec["actions"] if a.action in ("BUY", "TOP-UP")]
    sells = [a for a in rec["actions"] if a.action == "SELL"]
    holds = [a for a in rec["actions"] if a.action == "HOLD"]
    if buys:
        lines += ["", "BUY / TOP-UP",
                  "  " + ", ".join(f"{a.ticker.replace('.NS', '')} {a.quantity:,.0f} @ Rs {a.current_price:,.0f}"
                                  for a in buys)]
    if holds:
        lines += ["", f"HOLD ({len(holds)})",
                  "  " + ", ".join(a.ticker.replace('.NS', '') for a in holds)]
    if sells:
        lines += ["", f"SELL ({len(sells)})",
                  "  " + ", ".join(f"{a.ticker.replace('.NS', '')} ({a.reason})" for a in sells)]
    lines += [
        "",
        f"Invested value    : Rs {rec.get('hold_value', 0):,.0f}",
        f"Sell proceeds     : Rs {rec.get('sell_proceeds', 0):,.0f}",
        f"Cash left         : Rs {rec.get('cash_left', 0):,.0f}",
        "",
        "NOTE: summary reflects seed holdings.csv; the attached advisor.html is",
        "interactive and reconciles your browser-held positions live.",
        "Open at: http://localhost:8765/advisor.html",
    ]
    return "\n".join(lines)


def _rows_table(title: str, headers: List[str], rows: List[List[str]]) -> str:
    """Render an HTML table for one action group."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    if not rows:
        return (f"<h3>{_html.escape(title)}</h3>"
                f"<table border='1' cellspacing='0' cellpadding='5'>"
                f"<tr>{head}</tr><tr><td colspan='{len(headers)}'><i>none</i></td></tr></table>")
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f"<h3>{_html.escape(title)}</h3>"
            f"<table border='1' cellspacing='0' cellpadding='5'>"
            f"<tr>{head}</tr>{body}</table>")


def summary_html(payload: dict, rec: dict, as_of: str, next_scan: str) -> str:
    """HTML body for the quarterly email with BUY / TOP-UP / HOLD / SELL tables."""
    t = _html.escape
    buys = [a for a in rec["actions"] if a.action in ("BUY", "TOP-UP")]
    holds = [a for a in rec["actions"] if a.action == "HOLD"]
    sells = [a for a in rec["actions"] if a.action == "SELL"]

    buy_rows = [[t(a.ticker.replace(".NS", "")),
                 f"{a.quantity:,.0f}", f"Rs {a.current_price:,.0f}",
                 f"Rs {a.amount:,.0f}", str(a.rank),
                 f"Rs {a.stoploss_price:,.0f}" if a.stoploss_price else "-"]
                for a in buys]
    hold_rows = [[t(a.ticker.replace(".NS", "")),
                  f"{a.quantity:,.0f}", f"Rs {a.entry_price:,.0f}",
                  f"Rs {a.current_price:,.0f}",
                  f"{a.pnl_pct:.1%}", f"Rs {a.stoploss_price:,.0f}" if a.stoploss_price else "-"]
                 for a in holds]
    sell_rows = [[t(a.ticker.replace(".NS", "")),
                  f"{a.quantity:,.0f}", f"Rs {a.current_price:,.0f}",
                  f"Rs {a.amount:,.0f}", t(a.reason)]
                 for a in sells]

    blocks = "".join(x for x in [
        _rows_table(f"BUY / TOP-UP ({len(buys)})",
                    ["Ticker", "Qty", "Price", "Amount", "Rank", "Stoploss"], buy_rows),
        _rows_table(f"HOLD ({len(holds)})",
                    ["Ticker", "Qty", "Entry", "Price", "P&amp;L %", "Stoploss"], hold_rows),
        _rows_table(f"SELL ({len(sells)})",
                    ["Ticker", "Qty", "Price", "Proceeds", "Reason"], sell_rows),
    ])

    summary = "".join(f"<li><b>{t(k)}</b>: {v}</li>" for k, v in [
        ("Data as of", as_of), ("Last scan", str(payload.get("scan_time", ""))),
        ("Next scan", next_scan),
        ("Model", f"{payload.get('model', 'A')} | target {payload.get('max_stocks', 10)} stocks"),
        ("Stoploss", f"{payload.get('stoploss', 0.07):.0%}"),
        ("New cash (plan)", f"Rs {payload.get('cash', 0):,.0f}"),
        ("Target names", str(len(payload.get("target", [])))),
        ("Invested value", f"Rs {rec.get('hold_value', 0):,.0f}"),
        ("Sell proceeds", f"Rs {rec.get('sell_proceeds', 0):,.0f}"),
        ("Cash left", f"Rs {rec.get('cash_left', 0):,.0f}"),
    ])

    return f"""<html><body>
<h2 style="margin-bottom:2px">Momentum Advisor &mdash; Quarterly Report</h2>
<ul>{summary}</ul>
{blocks}
<p><i>Note: summary reflects seed holdings.csv; the attached advisor.html is
interactive and reconciles your browser-held positions live. Open at
<a href="http://localhost:8765/advisor.html">http://localhost:8765/advisor.html</a>.</i></p>
</body></html>"""


def build_email(cfg: AdvisorConfig, payload: dict, rec: dict, as_of: str,
                next_scan: str, attach_path: str, mail: Optional[dict] = None) -> MIMEMultipart:
    """Build the MIME message (no network). Pass mail for tests to avoid I/O."""
    mail = mail or load_email_config(cfg.email_config)
    msg = MIMEMultipart("mixed")
    msg["From"] = mail["sender"]
    msg["To"] = mail["recipient"]
    msg["Subject"] = f"Momentum Advisor report - {as_of} (scan {payload.get('scan_time', '')})"
    body = MIMEMultipart("alternative")
    body.attach(MIMEText(summary_text(payload, rec, as_of, next_scan), "plain"))
    body.attach(MIMEText(summary_html(payload, rec, as_of, next_scan), "html"))
    msg.attach(body)
    if attach_path and os.path.exists(attach_path):
        with open(attach_path, encoding="utf-8") as f:
            part = MIMEText(f.read(), "html")
        part.add_header("Content-Disposition", "attachment", filename="advisor.html")
        msg.attach(part)
    return msg


def send_email(cfg: AdvisorConfig, payload: dict, rec: dict, as_of: str,
               attach_path: str, next_scan: Optional[str] = None) -> str:
    """Send the quarterly report via Gmail SMTP (STARTTLS, App Password)."""
    mail = load_email_config(cfg.email_config)
    next_scan = next_scan or next_scan_date_str()
    msg = build_email(cfg, payload, rec, as_of, next_scan, attach_path, mail=mail)
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=60) as server:
        server.starttls()
        server.login(mail["sender"], mail["app_password"])
        server.sendmail(mail["sender"], [mail["recipient"]], msg.as_string())
    return mail["recipient"]


# ---------------------------------------------------------------------------
# Quarterly reminder email (no scan - step-by-step instructions to run it)
# ---------------------------------------------------------------------------
REMINDER_STEPS = [
    ("Open Terminal", "Launch the Terminal app on your Mac."),
    ("Go to the project folder",
     "cd ~/VSCODE/BK_test"),
    ("Start the advisor server",
     "./serve.sh\n   (or, without the helper script: python3 advisor.py --serve)"),
    ("Open the report in your browser",
     "http://localhost:8765/advisor.html"),
    ("Refresh prices and targets",
     "Click the 'Run scan' button (top-right). Wait for it to finish and reload."),
    ("Verify your holdings",
     "In the 'What you own' table, check/edit your ticker, quantity and buy price. "
     "Set your new cash in the 'New cash this quarter' box."),
    ("Act on the orders",
     "Execute the BUY / TOP-UP / HOLD / SELL tables: sell what it says to sell, "
     "buy what it says to buy in momentum rank order. Whole shares only."),
    ("Save / export",
     "Optionally click 'Export CSV' to keep a copy. Your holdings are saved "
     "automatically in the browser."),
]


def build_reminder_email(mail: dict, next_scan: str,
                         port: int = 8765) -> MIMEMultipart:
    """Build the quarterly reminder message (text + HTML), no scan, no attachment."""
    text_lines = [
        "MOMENTUM ADVISOR - TIME FOR YOUR QUARTERLY SCAN",
        "=" * 45,
        f"Your next scan window: {next_scan}",
        "",
        "This is your once-every-3-months reminder to run the scan yourself. "
        "It takes about 5 minutes. Follow these steps:",
        "",
    ]
    for i, (title, body) in enumerate(REMINDER_STEPS, 1):
        text_lines.append(f"{i}. {title}")
        text_lines.append("   " + body.replace("\n", "\n   "))
        text_lines.append("")

    items = "".join(
        f"<li><b>{_html.escape(title)}</b><br>{_html.escape(body)}</li>"
        for title, body in REMINDER_STEPS)
    html_body = f"""<html><body style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">
<h2 style="margin-bottom:2px">Momentum Advisor &mdash; Time for your quarterly scan</h2>
<p><b>Your next scan window: {_html.escape(next_scan)}</b></p>
<p>This is your once-every-3-months reminder to run the scan yourself (about 5 minutes). Follow these steps:</p>
<ol>{items}</ol>
<p>Or run the scan directly from the terminal:</p>
<pre style="background:#f6f8fa; padding:10px; border-radius:6px">
cd ~/VSCODE/BK_test
python3 advisor.py --cash <new cash></pre>
<p><i>Notes: the scan refreshes prices and momentum targets and rewrites advisor.html.
Check prices daily only to catch stoploss breaks. This is an aid, not investment advice.</i></p>
</body></html>"""

    msg = MIMEMultipart("alternative")
    msg["From"] = mail["sender"]
    msg["To"] = mail["recipient"]
    msg["Subject"] = f"Momentum Advisor - run your quarterly scan ({next_scan})"
    msg.attach(MIMEText("\n".join(text_lines), "plain"))
    msg.attach(MIMEText(html_body, "html"))
    return msg


def main():
    args = parse_args()
    cfg = make_cfg(args)

    if args.test_email:
        mail = load_email_config(cfg.email_config)
        payload = {"scan_time": "test", "target": [], "model": "A",
                   "max_stocks": 10, "stoploss": 0.07, "cash": 0.0}
        rec = {"actions": [], "hold_value": 0, "sell_proceeds": 0, "cash_left": 0}
        msg = build_email(cfg, payload, rec, "test", next_scan_date_str(), cfg.out_path, mail=mail)
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=60) as server:
            server.starttls()
            server.login(mail["sender"], mail["app_password"])
            server.sendmail(mail["sender"], [mail["recipient"]], msg.as_string())
        print(f"Test email sent to {mail['recipient']} from {mail['sender']}.")
        return

    if args.serve:
        run_scan(cfg)  # initial scan so the page is fresh
        serve_dir = os.path.dirname(os.path.abspath(cfg.out_path)) or "."
        _ScanHandler.server_cfg = cfg
        handler = partial(_ScanHandler, directory=serve_dir)
        httpd = ThreadingHTTPServer(("127.0.0.1", cfg.port), handler)
        print(f"\nServing advisor at  http://localhost:{cfg.port}/advisor.html")
        print(f"Press Ctrl+C to stop. Use the 'Run scan' button on the page to refresh.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    out = run_scan(cfg)
    rec = out["rec"]
    target_ranks = out["target_ranks"]

    buys = [a for a in rec["actions"] if a.action in ("BUY", "TOP-UP")]
    sells = [a for a in rec["actions"] if a.action == "SELL"]
    holds = [a for a in rec["actions"] if a.action == "HOLD"]
    print(f"\n=== ADVISOR SUMMARY (seed holdings) ===")
    print(f"Target names   : {len(out['payload']['target'])}")
    print(f"BUY / TOP-UP   : {len(buys)} orders, Rs {sum(a.amount for a in buys):,.0f}")
    for a in buys:
        print(f"   {a.action:<6} {a.ticker.replace('.NS', ''):<12} {a.quantity:,.0f} x Rs {a.current_price:,.0f} = Rs {a.amount:,.0f}")
    print(f"HOLD           : {len(holds)}")
    print(f"SELL           : {len(sells)}, proceeds Rs {rec['sell_proceeds']:,.0f}")
    for a in sells:
        print(f"   {a.ticker.replace('.NS', ''):<12} {a.quantity:,.0f} x Rs {a.current_price:,.0f} = Rs {a.amount:,.0f}  ({a.reason})")
    print(f"Cash left      : Rs {rec['cash_left']:,.0f}")
    print(f"\nInteractive report written to {cfg.out_path} — open it in a browser and "
          f"enter your buys in the 'What you own' table.")
    print(f"Tip: run `python3 advisor.py --serve` and use the in-page 'Run scan' button "
          f"to refresh prices without re-running the command.")

    if args.email:
        recipient = send_email(cfg, out["payload"], rec, out["as_of"], cfg.out_path)
        print(f"Report emailed to {recipient} (next scan: {next_scan_date_str()}).")
    return rec


if __name__ == "__main__":
    main()
