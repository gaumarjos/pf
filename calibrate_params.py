"""calibrate_params.py — Calibrate vol.csv / mu.csv / corr.csv against
real index and ETF data from Yahoo Finance.

Usage
-----
    python calibrate_params.py           # print diff only (safe, read-only)
    python calibrate_params.py --write   # also save proposed_*.csv files

Design
------
μ, σ  — per-asset, full available history starting from TICKER_MAP[isin].start.
         Uses monthly total-return data where possible (ETF Adj Close).
         Annualisation: σ_annual = σ_monthly × √12
                        μ_annual = (1 + μ_monthly)^12 − 1

ρ     — calibrated assets: COMMON window = max(individual starts) → today.
         Required so every pair shares the same joint distribution.
         Assets listed in CORR_EXCLUDE are skipped; their rows/columns in the
         proposed matrix are kept from your current corr.csv (manual values).
         The full 16×16 matrix is PSD-projected once after blending.
         → Excluded by default: Managed Futures (DBMF only since 2019;
           4-year window dominated by the 2022 rate-hike regime).

Proxy quality is shown alongside every number so you can judge what to trust.

WARNING: price-only tickers (marked ★) exclude dividend yield — μ is
understated by the dividend yield for that asset (typically 0–2 %/yr).
Gold (GC=F) is the exception: gold pays no yield so price = total return.

For assets with no Yahoo proxy (Managed Futures, LifeStrategy 80/20) a
synthetic series is constructed where possible; otherwise the current value
is kept unchanged and flagged.
"""

import sys
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ── ISIN order (must match vol.csv / mu.csv / corr.csv) ───────────────────────
ISINS = [
    "IE000XZSV718", "IE000MLMNYS0", "IE000R4ZNTN3", "LU1681045370",
    "IE00BJQRDN15", "IE00BP3QZB59", "IE00BP3QZ825", "IE00BP3QZ601",
    "IE00BH04GL39", "LU1650491282", "IE00BDBRDM35",
    "IE00B4ND3602", "IE00BZ1NCS44",
    "LU2951555403", "IE00B44Z5B48", "IE00BMVB5R75",
]


# ── Ticker configuration ───────────────────────────────────────────────────────
@dataclass
class TConfig:
    ticker:      Optional[str]   # Yahoo ticker; None → synthetic or unavailable
    start:       Optional[str]   # "YYYY-MM-DD" history start to use
    total_return: bool           # True = Adj Close includes dividends
    note:        str             # human-readable proxy description


TICKER_MAP: dict[str, TConfig] = {
    "IE000XZSV718": TConfig(
        "^SP500TR", "1988-01-01", True,
        "S&P 500 Total Return index (dividends reinvested)"),

    "IE000MLMNYS0": TConfig(
        "RSP", "2003-05-01", True,
        "Invesco S&P 500 Equal Weight ETF — Adj Close (dividends incl.)"),

    "IE000R4ZNTN3": TConfig(
        "EFA", "2001-09-01", True,
        "iShares MSCI EAFE ETF — proxy for MSCI World ex-USA"),

    "LU1681045370": TConfig(
        "EEM", "2003-04-01", True,
        "iShares MSCI EM ETF — proxy for MSCI Emerging Markets"),

    "IE00BJQRDN15": TConfig(
        "ACWI", "2008-03-01", True,
        "MSCI ACWI ETF — rough proxy (IQS Multi-Factor index not on Yahoo)"),

    "IE00BP3QZB59": TConfig(
        "EFV", "2005-04-01", True,
        "iShares MSCI EAFE Value ETF — proxy for MSCI World Value"),

    "IE00BP3QZ825": TConfig(
        "IMTM", "2015-01-01", True,
        "iShares MSCI Intl Momentum Factor ETF — ⚠ limited history (~10y)"),

    "IE00BP3QZ601": TConfig(
        "IQLT", "2015-06-01", True,
        "iShares MSCI Intl Quality Factor ETF — ⚠ limited history (~10y)"),

    "IE00BH04GL39": TConfig(
        "IBGE.L", "2006-01-01", True,
        "iShares € Govt Bond 7-10yr UCITS ETF (London)"),

    "LU1650491282": TConfig(
        "IBCI.L", "2008-01-01", True,
        "iShares € Inflation Linked Govt Bond UCITS ETF (London)"),

    "IE00BDBRDM35": TConfig(
        "AGG", "2003-09-01", True,
        "iShares Core US Aggregate Bond ETF — unhedged USD proxy for Global Agg EUR-hdg; ⚠ currency diff"),

    "IE00B4ND3602": TConfig(
        "GC=F", "1975-01-01", False,          # gold pays no yield → price = TR
        "COMEX Gold Futures continuous — price only (no yield to miss)"),

    "IE00BZ1NCS44": TConfig(
        "^BCOM", "1991-01-01", False,
        "Bloomberg Commodity Index ★ price-only — μ understated by roll yield"),

    "LU2951555403": TConfig(
        "DBMF", "2019-05-01", True,
        "iMGP DBi Managed Futures ETF (US) — ⚠ very limited history (~6y)"),

    "IE00B44Z5B48": TConfig(
        "ACWI", "2008-03-01", True,
        "iShares MSCI ACWI ETF — Adj Close"),

    "IE00BMVB5R75": TConfig(
        None, None, True,
        "Synthetic: 80 % ACWI + 20 % AGGH.L — no direct Yahoo ticker"),
}

# Synthetic compositions: ISIN → [(ticker, weight), ...]
SYNTHETIC: dict[str, list[tuple[str, float]]] = {
    "IE00BMVB5R75": [("ACWI", 0.80), ("AGG", 0.20)],
}

# ── Correlation exclusion ─────────────────────────────────────────────────────
# ISINs here are skipped when calibrating the correlation matrix from data.
# Their rows/columns in the proposed matrix are taken from your current corr.csv.
# Add any asset whose available history is too short or too regime-specific.
CORR_EXCLUDE: set[str] = {
    "LU2951555403",   # Managed Futures — DBMF only since 2019, window collapses to 4y
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _download(ticker: str, start: str) -> Optional[pd.Series]:
    """Return a monthly-resampled Adj-Close price Series, or None on failure."""
    try:
        raw = yf.download(ticker, start=start, auto_adjust=True,
                          progress=False, actions=False)
        if raw.empty:
            return None
        # yfinance ≥0.2 may return a MultiIndex — flatten
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        col = "Close" if "Close" in raw.columns else raw.columns[0]
        prices = raw[col].dropna()
        # resample to month-end
        return prices.resample("ME").last().dropna()
    except Exception:
        return None


def _monthly_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def _annualise(ret_monthly: pd.Series) -> tuple[float, float]:
    """Return (annualised_mu, annualised_vol) from a monthly return Series."""
    mu_annual  = (1 + ret_monthly.mean()) ** 12 - 1
    vol_annual = ret_monthly.std() * np.sqrt(12)
    return mu_annual, vol_annual


def _psd_project(C: np.ndarray) -> np.ndarray:
    """Nearest PSD correlation matrix (eigenvalue clamp, Higham-style)."""
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 1e-8, None)
    C2 = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(C2))
    return C2 / np.outer(d, d)


# ── Fetch all price series ─────────────────────────────────────────────────────
def fetch_all() -> dict[str, Optional[pd.Series]]:
    """Download one monthly price Series per ISIN. Returns dict isin→Series."""
    # collect unique non-None tickers first (avoid double-downloading ACWI etc.)
    unique: dict[str, tuple[str, str]] = {}  # ticker → (isin, start)
    for isin, cfg in TICKER_MAP.items():
        if cfg.ticker and cfg.ticker not in unique:
            unique[cfg.ticker] = cfg.start

    print("Downloading data from Yahoo Finance …")
    ticker_cache: dict[str, Optional[pd.Series]] = {}
    for ticker, start in unique.items():
        p = _download(ticker, start)
        ticker_cache[ticker] = p
        status = f"{len(p)} months" if p is not None else "FAILED"
        print(f"  {ticker:<12}  {status}")

    # build per-ISIN returns, handling synthetics
    result: dict[str, Optional[pd.Series]] = {}
    for isin, cfg in TICKER_MAP.items():
        if cfg.ticker is not None:
            result[isin] = ticker_cache.get(cfg.ticker)
        elif isin in SYNTHETIC:
            # weighted sum of component return series, on common dates
            components = []
            for ticker, weight in SYNTHETIC[isin]:
                cached = ticker_cache.get(ticker)
                p = cached if cached is not None else _download(ticker, "2000-01-01")
                if p is not None:
                    components.append((weight, _monthly_returns(p)))
            if components:
                # align on common index
                base = components[0][1].copy()
                for w, s in components[1:]:
                    base, s = base.align(s, join="inner")
                    base = base * (1 - w) + s * w   # reconstruct from weights
                # rebuild properly: sum of weighted returns
                idx = components[0][1].index
                for w, s in components[1:]:
                    idx = idx.intersection(s.index)
                ret = sum(w * s.loc[idx] for w, s in components)
                result[isin] = ret  # already returns, not prices
                ticker_cache[f"__ret_{isin}"] = ret   # stash for corr
            else:
                result[isin] = None
        else:
            result[isin] = None

    return result


# ── Compute mu / vol per asset ─────────────────────────────────────────────────
def compute_mu_vol(
    series_map: dict[str, Optional[pd.Series]],
) -> tuple[pd.Series, pd.Series]:
    """Return two Series (mu, vol) indexed by ISIN, using each asset's own window."""
    mus, vols = {}, {}
    for isin in ISINS:
        s = series_map.get(isin)
        if s is None:
            mus[isin] = np.nan
            vols[isin] = np.nan
            continue
        # if series is already returns (synthetic), use directly; else pct_change
        if s.min() < -0.5 or s.max() > 3.0:
            # looks like prices, not returns
            ret = _monthly_returns(s)
        else:
            ret = s.dropna()
        mu, vol = _annualise(ret)
        mus[isin] = mu
        vols[isin] = vol
    return pd.Series(mus, name="mu"), pd.Series(vols, name="vol")


# ── Compute correlation on common window ──────────────────────────────────────
def compute_corr(
    series_map: dict[str, Optional[pd.Series]],
    corr_cur: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, set[str]]:
    """
    Compute a correlation matrix on the common overlap window, excluding
    ISINs in CORR_EXCLUDE.  Excluded rows/columns are filled from corr_cur.
    The full blended matrix is PSD-projected once before returning.

    Returns (corr_df, common_start_str, common_end_str, excluded_isins).
    """
    # Build return series only for non-excluded assets
    included = [i for i in ISINS if i not in CORR_EXCLUDE]
    returns: dict[str, pd.Series] = {}
    for isin in included:
        s = series_map.get(isin)
        if s is None:
            continue
        ret = _monthly_returns(s) if (s.min() < -0.5 or s.max() > 3.0) else s.dropna()
        returns[isin] = ret

    if not returns:
        return corr_cur.copy(), "?", "?", CORR_EXCLUDE

    # Align to common overlap of all included, non-excluded assets
    df = pd.DataFrame(returns).dropna()
    common_start = str(df.index.min().date())
    common_end   = str(df.index.max().date())

    # Sub-correlation matrix (only calibrated assets)
    sub_corr = df.corr()

    # Build full 16×16 matrix: start from current (covers excluded rows/cols),
    # then overwrite the calibrated sub-block.
    full = corr_cur.reindex(index=ISINS, columns=ISINS).to_numpy(dtype=float).copy()
    isin_idx = {isin: i for i, isin in enumerate(ISINS)}
    cal_isins = list(sub_corr.index)   # ISINs actually in the sub-matrix
    for r in cal_isins:
        for c in cal_isins:
            full[isin_idx[r], isin_idx[c]] = sub_corr.loc[r, c]

    # PSD-project the full blended matrix
    full_psd = _psd_project(full)
    corr_df  = pd.DataFrame(full_psd, index=ISINS, columns=ISINS)

    # Track which ISINs ended up excluded (CORR_EXCLUDE union those with no data)
    no_data  = set(included) - set(cal_isins)
    excluded = CORR_EXCLUDE | no_data

    return corr_df, common_start, common_end, excluded


# ── Load current CSVs ──────────────────────────────────────────────────────────
def load_current() -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    mu_cur   = pd.read_csv("mu.csv",   index_col="isin")["mu"]
    vol_cur  = pd.read_csv("vol.csv",  index_col="isin")["vol"]
    corr_raw = pd.read_csv("corr.csv", index_col="isin")
    # restore full symmetric matrix (only lower triangle may be stored)
    C = corr_raw.to_numpy(dtype=float)
    i_upper = np.triu_indices_from(C, k=1)
    C[i_upper] = C.T[i_upper]
    corr_cur = pd.DataFrame(C, index=corr_raw.index, columns=corr_raw.index)
    return mu_cur, vol_cur, corr_cur


# ── Pretty diff printers ───────────────────────────────────────────────────────
_SEP = "─" * 90

def _flag(isin: str) -> str:
    cfg = TICKER_MAP[isin]
    flags = []
    if cfg.ticker is None:
        flags.append("SYNTHETIC")
    if not cfg.total_return and cfg.ticker is not None:
        flags.append("★price-only")
    if "⚠" in (cfg.note or ""):
        flags.append("⚠ short history")
    return "  [" + ", ".join(flags) + "]" if flags else ""


def print_mu_vol_diff(
    mu_cur: pd.Series, vol_cur: pd.Series,
    mu_new: pd.Series, vol_new: pd.Series,
    names: dict[str, str],
) -> None:
    print(f"\n{'═'*90}")
    print("  μ (EXPECTED RETURN) DIFF   —   per-asset window, monthly returns annualised")
    print(f"{'═'*90}")
    print(f"  {'Asset':<30}  {'Current':>8}  {'Proposed':>9}  {'Δ':>7}  {'Proxy / Notes'}")
    print(f"  {_SEP}")
    for isin in ISINS:
        cur = mu_cur.get(isin, np.nan)
        new = mu_new.get(isin, np.nan)
        delta_str = f"{(new-cur)*100:+.1f}pp" if not (np.isnan(new) or np.isnan(cur)) else "  n/a  "
        new_str   = f"{new*100:.1f}%" if not np.isnan(new) else "  n/a "
        cfg  = TICKER_MAP[isin]
        note = cfg.note if cfg else "—"
        print(f"  {names[isin]:<30}  {cur*100:>7.1f}%  {new_str:>9}  {delta_str:>7}  {note}{_flag(isin)}")

    print(f"\n{'═'*90}")
    print("  σ (VOLATILITY) DIFF   —   per-asset window, monthly returns annualised")
    print(f"{'═'*90}")
    print(f"  {'Asset':<30}  {'Current':>8}  {'Proposed':>9}  {'Δ':>7}  {'Proxy / Notes'}")
    print(f"  {_SEP}")
    for isin in ISINS:
        cur = vol_cur.get(isin, np.nan)
        new = vol_new.get(isin, np.nan)
        delta_str = f"{(new-cur)*100:+.1f}pp" if not (np.isnan(new) or np.isnan(cur)) else "  n/a  "
        new_str   = f"{new*100:.1f}%" if not np.isnan(new) else "  n/a "
        cfg  = TICKER_MAP[isin]
        note = cfg.note if cfg else "—"
        print(f"  {names[isin]:<30}  {cur*100:>7.1f}%  {new_str:>9}  {delta_str:>7}  {note}{_flag(isin)}")


def print_corr_diff(
    corr_cur: pd.DataFrame,
    corr_new: pd.DataFrame,
    names: dict[str, str],
    common_start: str,
    common_end: str,
    excluded: set[str],
) -> None:
    excl_names = ", ".join(names[i] for i in ISINS if i in excluded)
    print(f"\n{'═'*90}")
    print(f"  CORRELATION MATRIX DIFF   (proposed − current)")
    print(f"  Calibrated window : {common_start} → {common_end}")
    print(f"  Kept from manual  : {excl_names}  (marked M in proposed matrix)")
    print(f"  Full 16×16 matrix PSD-projected after blending calibrated + manual entries.")
    print(f"{'═'*90}")

    labels = [names[i][:14] for i in ISINS]
    # header
    header = f"  {'':20}" + "".join(f"  {l:>7}" for l in labels)
    print(header)
    print(f"  {'─'*20}" + "─" * (9 * len(ISINS)))

    for i, isin_r in enumerate(ISINS):
        row_label = names[isin_r][:20]
        row = f"  {row_label:<20}"
        for j, isin_c in enumerate(ISINS):
            cur_v = corr_cur.loc[isin_r, isin_c] if (isin_r in corr_cur.index and isin_c in corr_cur.columns) else np.nan
            new_v = corr_new.loc[isin_r, isin_c] if (isin_r in corr_new.index and isin_c in corr_new.columns) else np.nan
            if i == j:
                row += f"  {'  —  ':>7}"
            elif np.isnan(new_v) or np.isnan(cur_v):
                row += f"  {'  n/a':>7}"
            else:
                d = new_v - cur_v
                row += f"  {d:>+7.2f}"
        print(row)

    print()
    print("  (positive = proposed higher than current, negative = lower)")
    print()
    # also print the full proposed matrix for reference
    print(f"  PROPOSED CORRELATION MATRIX   (after PSD projection)")
    print(f"  {_SEP}")
    print(header)
    print(f"  {'─'*20}" + "─" * (9 * len(ISINS)))
    for i, isin_r in enumerate(ISINS):
        row_label = names[isin_r][:20]
        row = f"  {row_label:<20}"
        for j, isin_c in enumerate(ISINS):
            v = corr_new.loc[isin_r, isin_c] if (isin_r in corr_new.index and isin_c in corr_new.columns) else np.nan
            is_manual = (isin_r in excluded or isin_c in excluded) and i != j
            if i == j:
                row += f"  {'  1.00':>7}"
            elif np.isnan(v):
                row += f"  {'  n/a':>7}"
            elif is_manual:
                row += f"  {v:>6.2f}M"
            else:
                row += f"  {v:>7.2f}"
        print(row)


# ── Save proposed CSVs ─────────────────────────────────────────────────────────
def write_proposed(mu_new: pd.Series, vol_new: pd.Series, corr_new: pd.DataFrame) -> None:
    out = {
        "proposed_mu.csv":   mu_new.rename_axis("isin"),
        "proposed_vol.csv":  vol_new.rename_axis("isin"),
    }
    for fname, s in out.items():
        s.to_csv(fname)
        print(f"  Saved {fname}")
    corr_new.rename_axis("isin").to_csv("proposed_corr.csv")
    print("  Saved proposed_corr.csv")
    print()
    print("  To adopt, review the diffs above and manually copy values into")
    print("  vol.csv / mu.csv / corr.csv  (or run _gen_params.py with updated numbers).")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    write = "--write" in sys.argv

    # Load asset names for display
    try:
        import tomllib
        with open("assets.toml", "rb") as f:
            assets_meta = tomllib.load(f)["assets"]
        names = {isin: assets_meta[isin]["name"] for isin in ISINS}
    except Exception:
        names = {isin: isin for isin in ISINS}

    # Fetch data
    series_map = fetch_all()

    # μ / σ per asset (own window)
    mu_new, vol_new = compute_mu_vol(series_map)

    # Load current (needed before compute_corr to fill excluded rows/cols)
    mu_cur, vol_cur, corr_cur = load_current()

    # Correlation on common window (excluded ISINs taken from corr_cur)
    corr_new, common_start, common_end, excluded = compute_corr(series_map, corr_cur)

    # Print diffs
    print_mu_vol_diff(mu_cur, vol_cur, mu_new, vol_new, names)
    print_corr_diff(corr_cur, corr_new, names, common_start, common_end, excluded)

    # Window summary
    print(f"{'═'*90}")
    print("  DATA SOURCES & WINDOWS")
    print(f"{'═'*90}")
    print(f"  {'Asset':<30}  {'Ticker':<12}  {'Hist. start':>12}  {'Notes'}")
    print(f"  {_SEP}")
    for isin in ISINS:
        cfg = TICKER_MAP[isin]
        ticker = cfg.ticker or "(synthetic)"
        start  = cfg.start  or "—"
        print(f"  {names[isin]:<30}  {ticker:<12}  {start:>12}  {cfg.note}")

    print(f"\n  Correlation calibrated window : {common_start} → {common_end}")
    print(f"  Correlation excluded (manual) : {', '.join(names[i] for i in ISINS if i in excluded)}")
    print(f"  μ / σ price-only note (★) : μ is understated by ~dividend yield for that asset.")
    print(f"  n/a entries               : ticker download failed or history too short.")
    print()

    if write:
        print("Writing proposed_*.csv …")
        write_proposed(mu_new, vol_new, corr_new)
    else:
        print("  Run with --write to save proposed_vol.csv / proposed_mu.csv / proposed_corr.csv")


if __name__ == "__main__":
    main()
