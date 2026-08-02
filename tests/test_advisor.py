"""Unit tests for the monthly momentum advisor (advisor.py).

Assert-based, run with:  python3 tests/test_advisor.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advisor import (Action, AdvisorConfig, Position, build_email,
                     build_html, build_payload, load_holdings, reconcile,
                     next_scan_date_str, summary_html, summary_text,
                     HOLDINGS_TEMPLATE)


def make_target(weights):
    return {t: w for t, w in weights.items()}


def make_ranks(order):
    return {t: i + 1 for i, t in enumerate(order)}


# ---------------------------------------------------------------------------
# load_holdings
# ---------------------------------------------------------------------------
def test_load_holdings_parses_and_adds_ns(tmp="tests/_tmp_h.csv"):
    with open(tmp, "w") as f:
        f.write("ticker,quantity,avg_price,entry_date\nRELIANCE.NS,12,2500,2026-06-30\nTATAMOTORS,150,620,2026-06-30\n")
    pos = load_holdings(tmp)
    assert len(pos) == 2, pos
    assert pos[0].ticker == "RELIANCE.NS"
    assert pos[1].ticker == "TATAMOTORS.NS"  # .NS auto-appended
    assert pos[1].quantity == 150
    os.remove(tmp)


def test_load_holdings_creates_template_when_missing(tmp="tests/_tmp_missing.csv"):
    if os.path.exists(tmp):
        os.remove(tmp)
    pos = load_holdings(tmp)
    assert pos == []
    assert os.path.exists(tmp)
    with open(tmp) as f:
        assert f.read() == HOLDINGS_TEMPLATE
    os.remove(tmp)


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------
def test_first_run_all_buy():
    tw = make_target({"A.NS": 0.5, "B.NS": 0.5})
    tr = make_ranks(["A.NS", "B.NS"])
    prices = {"A.NS": 100.0, "B.NS": 200.0}
    rec = reconcile(tw, tr, [], prices, 0.07, 100000.0)
    buys = [a for a in rec["actions"] if a.action == "BUY"]
    assert len(buys) == 2, rec
    by_ticker = {x.ticker: x for x in buys}
    assert by_ticker["A.NS"].quantity == 500 and by_ticker["A.NS"].amount == 50000.0
    assert by_ticker["B.NS"].quantity == 250 and by_ticker["B.NS"].amount == 50000.0
    assert rec["cash_left"] == 0.0
    assert rec["sell_proceeds"] == 0.0
    assert rec["missing_price"] == []


def test_sell_when_dropped_from_target():
    tw = make_target({"A.NS": 1.0})
    tr = make_ranks(["A.NS"])
    prices = {"A.NS": 100.0, "C.NS": 50.0}
    pos = [Position("C.NS", 100, 40.0, "2026-06-01")]
    rec = reconcile(tw, tr, pos, prices, 0.07, 0.0)
    sells = [a for a in rec["actions"] if a.action == "SELL"]
    assert len(sells) == 1 and sells[0].ticker == "C.NS"
    assert sells[0].quantity == 100
    assert abs(rec["sell_proceeds"] - 5000.0) < 1e-6
    assert "dropped out" in sells[0].reason


def test_sell_when_stoploss_hit():
    tw = make_target({"A.NS": 1.0})
    tr = make_ranks(["A.NS"])
    prices = {"A.NS": 100.0, "C.NS": 45.0}
    pos = [Position("C.NS", 100, 50.0, "2026-06-01")]  # -10% < -7%
    rec = reconcile(tw, tr, pos, prices, 0.07, 0.0)
    sells = [a for a in rec["actions"] if a.action == "SELL"]
    assert len(sells) == 1 and sells[0].ticker == "C.NS"
    assert "stoploss" in sells[0].reason
    assert abs(sells[0].pnl_pct - (-0.10)) < 1e-9


def test_hold_above_stoploss_in_target():
    tw = make_target({"A.NS": 1.0})
    tr = make_ranks(["A.NS"])
    prices = {"A.NS": 110.0}
    pos = [Position("A.NS", 100, 100.0, "2026-06-01")]
    rec = reconcile(tw, tr, pos, prices, 0.07, 0.0)
    holds = [a for a in rec["actions"] if a.action == "HOLD"]
    sells = [a for a in rec["actions"] if a.action == "SELL"]
    assert len(holds) == 1 and holds[0].ticker == "A.NS"
    assert sells == []


def test_topup_only_no_trim_when_overweight():
    tw = make_target({"A.NS": 0.5, "B.NS": 0.5})
    tr = make_ranks(["A.NS", "B.NS"])
    prices = {"A.NS": 300.0, "B.NS": 200.0}
    # A is worth 300*100 = 30000 vs target 25000 -> overweight, must NOT be trimmed
    pos = [Position("A.NS", 100, 100.0, "2026-06-01"), Position("B.NS", 100, 100.0, "2026-06-01")]
    rec = reconcile(tw, tr, pos, prices, 0.07, 0.0)
    actions = {a.ticker: a for a in rec["actions"]}
    assert actions["A.NS"].action == "HOLD"
    assert actions["B.NS"].action == "HOLD"
    # overweight A untouched, underweight B gets top-up only if cash allows (no cash here)
    assert not any(a.action == "SELL" for a in rec["actions"])
    assert rec["cash_left"] == 0.0


def test_buy_prioritized_by_rank_when_cash_short():
    tw = make_target({"A.NS": 0.5, "B.NS": 0.5})
    tr = make_ranks(["A.NS", "B.NS"])
    # B costs more than a single share's worth of its target slice -> unaffordable,
    # so only the higher-ranked A is bought.
    prices = {"A.NS": 100.0, "B.NS": 60000.0}
    rec = reconcile(tw, tr, [], prices, 0.07, 100000.0)
    buys = [a for a in rec["actions"] if a.action == "BUY"]
    assert len(buys) == 1 and buys[0].ticker == "A.NS"
    assert buys[0].quantity == 500 and buys[0].amount == 50000.0
    assert abs(rec["cash_left"] - 50000.0) < 1e-6


def test_missing_price_held_not_bought():
    tw = make_target({"A.NS": 1.0})
    tr = make_ranks(["A.NS"])
    prices = {"A.NS": 100.0}  # D.NS has no price
    pos = [Position("D.NS", 10, 90.0, "2026-06-01")]
    rec = reconcile(tw, tr, pos, prices, 0.07, 50000.0)
    holds = [a for a in rec["actions"] if a.action == "HOLD"]
    assert len(holds) == 1 and holds[0].ticker == "D.NS"
    assert rec["missing_price"] == ["D.NS"]
    assert all(a.action == "BUY" and a.ticker == "A.NS" for a in rec["actions"] if a.action == "BUY")


def test_quantity_rounded_down_to_whole_shares():
    tw = make_target({"A.NS": 1.0})
    tr = make_ranks(["A.NS"])
    prices = {"A.NS": 150.0}
    rec = reconcile(tw, tr, [], prices, 0.07, 100000.0)
    buys = [a for a in rec["actions"] if a.action == "BUY"]
    assert len(buys) == 1
    assert buys[0].quantity == 666  # 100000/150 = 666.67 -> 666
    assert buys[0].amount == 99900.0
    assert abs(rec["cash_left"] - 100.0) < 1e-6


# ---------------------------------------------------------------------------
# payload + interactive HTML
# ---------------------------------------------------------------------------
def test_build_payload_contains_data_js_needs():
    cfg = AdvisorConfig(cash=50000.0, max_stocks=10, stoploss=0.07)
    tw = make_target({"A.NS": 0.5, "B.NS": 0.5})
    tr = make_ranks(["B.NS", "A.NS"])  # B has rank 1
    prices = {"A.NS": 100.0, "B.NS": 200.0}
    seed = [Position("A.NS", 10, 90.0, "2026-06-01")]
    p = build_payload(cfg, tw, tr, prices, "2026-07-31", seed, "2026-08-02 10:30 IST",
                      {"A.NS": 0.25, "B.NS": 0.5})
    assert p["as_of"] == "2026-07-31"
    assert p["scan_time"] == "2026-08-02 10:30 IST"
    assert p["ret_1y"] == {"A.NS": 0.25, "B.NS": 0.5}
    assert p["stoploss"] == 0.07 and p["max_stocks"] == 10 and p["cash"] == 50000.0
    assert p["target"] == [{"ticker": "B.NS", "weight": 0.5, "rank": 1},
                           {"ticker": "A.NS", "weight": 0.5, "rank": 2}]
    assert p["prices"] == {"A.NS": 100.0, "B.NS": 200.0}
    assert p["seed"] == [{"ticker": "A.NS", "quantity": 10,
                          "avg_price": 90.0, "entry_date": "2026-06-01"}]


def test_build_html_embeds_payload_and_interactive_editor():
    cfg = AdvisorConfig(cash=0.0, max_stocks=10, stoploss=0.07)
    p = build_payload(cfg, make_target({"A.NS": 1.0}), make_ranks(["A.NS"]),
                      {"A.NS": 100.0}, "2026-07-31", [], "2026-08-02 10:30 IST", {"A.NS": 0.5})
    html = build_html(cfg, p, "2026-07-31")
    assert 'id="advisor-data"' in html
    assert 'id="holdings-body"' in html          # editable holdings table
    assert 'id="buys-body"' in html
    assert 'id="cash-input"' in html             # new cash field
    assert 'localStorage' in html                # persistence
    assert '"as_of": "2026-07-31"' in html       # embedded JSON payload
    assert 'Rules &amp; Instructions' in html
    assert 'stoploss' in html.lower()
    # 1-year return column in every table
    assert html.count('1Y ret') == 4             # editor + BUY + HOLD + SELL headers
    # when-to-buy rule + next scan date
    assert 'when to buy' in html.lower() or 'Entry timing' in html
    assert 'id="next-scan"' in html
    assert '12-minus-1' in html
    assert 'P[t-1]' in html


def test_build_email_attaches_html_and_correct_recipient():
    cfg = AdvisorConfig(cash=25000.0, max_stocks=10, stoploss=0.07)
    mail = {"sender": "me@gmail.com", "app_password": "abcd", "recipient": "sivasan0606@gmail.com"}
    p = build_payload(cfg, make_target({"A.NS": 0.5, "B.NS": 0.5}),
                      make_ranks(["A.NS", "B.NS"]),
                      {"A.NS": 100.0, "B.NS": 200.0}, "2026-07-31",
                      [], "2026-08-02 10:30 IST", {"A.NS": 0.1, "B.NS": 0.2})
    rec = reconcile(make_target({"A.NS": 0.5, "B.NS": 0.5}), make_ranks(["A.NS", "B.NS"]),
                    [], {"A.NS": 100.0, "B.NS": 200.0}, 0.07, 25000.0)
    html_path = "tests/_tmp_email.html"
    with open(html_path, "w") as f:
        f.write("<html>advisor report</html>")
    msg = build_email(cfg, p, rec, "2026-07-31", "Fri 2026-08-28", html_path, mail=mail)
    os.remove(html_path)
    assert msg["To"] == "sivasan0606@gmail.com"
    assert msg["From"] == "me@gmail.com"
    assert "Momentum Advisor report - 2026-07-31" in msg["Subject"]
    parts = [c for c in msg.walk() if c.get_payload(decode=True)]
    # attachment is the interactive report (named advisor.html)
    attach = [c for c in parts
              if c.get_content_disposition() == "attachment"
              and "advisor.html" in c.get("Content-Disposition")]
    assert attach and "advisor report" in attach[0].get_payload(decode=True).decode()
    # plain-text body still present
    texts = [c.get_payload(decode=True).decode() for c in parts
             if c.get_content_type() == "text/plain"]
    assert texts and "MOMENTUM ADVISOR - MONTHLY REPORT" in texts[0]
    assert all("sivasan0606" not in t for t in texts)  # no credentials leaked
    # HTML body is the table summary, distinct from the attachment
    html_bodies = [c.get_payload(decode=True).decode() for c in parts
                   if c.get_content_type() == "text/html"
                   and c.get_content_disposition() != "attachment"]
    assert html_bodies and "<table" in html_bodies[0]
    assert "advisor report" not in html_bodies[0]


def test_summary_html_has_buy_hold_sell_tables():
    cfg = AdvisorConfig(cash=100000.0, max_stocks=10, stoploss=0.07)
    p = build_payload(cfg, make_target({"A.NS": 1.0}), make_ranks(["A.NS"]),
                      {"A.NS": 100.0}, "2026-07-31", [], "2026-08-02 10:30 IST", {"A.NS": 0.1})
    rec = reconcile(make_target({"A.NS": 1.0}), make_ranks(["A.NS"]),
                    [], {"A.NS": 100.0}, 0.07, 100000.0)
    html = summary_html(p, rec, "2026-07-31", "Fri 2026-08-28")
    assert "BUY / TOP-UP (1)" in html
    assert "<table" in html
    assert "A" in html and "1,000" in html and "Rs 100" in html
    assert "Stoploss" in html
    assert "Cash left" in html


def test_summary_html_shows_hold_and_sell_rows():
    cfg = AdvisorConfig(cash=0.0, max_stocks=10, stoploss=0.07)
    target = make_target({"A.NS": 1.0})
    ranks = make_ranks(["A.NS"])
    prices = {"A.NS": 90.0, "B.NS": 50.0}
    seed = [Position("A.NS", 10, 100.0, "2026-06-30"),   # stoploss hit -> SELL
            Position("B.NS", 20, 40.0, "2026-06-30")]   # not in target -> SELL
    p = build_payload(cfg, target, ranks, prices, "2026-07-31", seed,
                      "2026-08-02 10:30 IST", {"A.NS": -0.1})
    rec = reconcile(target, ranks, seed, prices, 0.07, 0.0)
    html = summary_html(p, rec, "2026-07-31", "Fri 2026-08-28")
    assert "SELL (2)" in html
    assert "stoploss" in html.lower() or "below" in html.lower() or "dropped" in html.lower()
    assert "HOLD (0)" in html
    assert "Proceeds" in html


def test_summary_text_reports_buys_and_cash():
    cfg = AdvisorConfig(cash=100000.0, max_stocks=10, stoploss=0.07)
    p = build_payload(cfg, make_target({"A.NS": 1.0}), make_ranks(["A.NS"]),
                      {"A.NS": 100.0}, "2026-07-31", [], "2026-08-02 10:30 IST", {"A.NS": 0.1})
    rec = reconcile(make_target({"A.NS": 1.0}), make_ranks(["A.NS"]),
                    [], {"A.NS": 100.0}, 0.07, 100000.0)
    text = summary_text(p, rec, "2026-07-31", "Fri 2026-08-28")
    assert "BUY / TOP-UP" in text
    assert "A 1,000 @ Rs 100" in text
    assert "Cash left" in text
    assert next_scan_date_str()  # returns a usable date string


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
