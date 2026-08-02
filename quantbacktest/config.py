from dataclasses import dataclass, asdict, field


@dataclass
class BacktestConfig:
    # --- Data / universe ---
    universe: str = "nifty500"
    start_date: str = "2021-01-01"
    end_date: str = "2026-01-01"
    benchmark: str = "^NSEI"

    # --- Capital / costs ---
    initial_capital: float = 400000.0
    transaction_cost: float = 0.0015  # 0.15% per side

    # --- Phase 1: Value screen (book-to-market) ---
    bm_top_quantile: float = 0.20  # keep top 20% by BM ratio

    # --- Phase 2: Quality screen (Piotroski F-Score) ---
    fscore_min: int = 7
    fscore_max: int = 9

    # --- Phase 3: Cross-sectional momentum ---
    mom_lookback_days: int = 252     # ~12 months
    mom_skip_days: int = 21          # ~1 month skip between formation and holding
    top_decile: float = 0.10         # long top decile of ranked universe

    # --- Phase 4: TSMOM trend filter + volatility targeting ---
    vol_lookback_days: int = 252
    vol_target: float = 0.40         # position weight = 40% / sigma
    tsmom_lookback_days: int = 252   # 12-month time-series momentum sign

    # --- Refresh cadence / data hygiene ---
    fundamental_pub_lag_months: int = 4   # annual results considered available 4 months after FY end
    fundamental_refresh_month: int = 9    # default refresh month (Sep) if FY lag never triggers
    min_price_days: int = 273             # require >= ~13 months of price history
    max_positions: int = 30
    min_positions: int = 1

    # --- Output / cache ---
    output_dir: str = "output"
    cache_dir: str = "cache"

    def to_dict(self):
        return asdict(self)

    def clone(self, **overrides):
        data = asdict(self)
        data.update(overrides)
        return BacktestConfig(**data)
