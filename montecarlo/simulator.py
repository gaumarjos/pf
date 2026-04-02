"""simulator.py — Portfolio Monte Carlo simulation engine.

Can be used standalone from the command line:
    python simulator.py [scenario_file]

or imported by other modules:
    from simulator import simulate, print_results

All matplotlib / charting dependencies live in charts.py, not here.
"""

import tomllib
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm


# ── Default simulation constants ──────────────────────────────────────────────
# These are used when simulate() is called without overrides.
# The web app (app.py) passes its own values via kwargs.

N_SIMS       = 100_000
RAND_SEED    = 42
STUDENT_T_NU = 4
RF_RATE      = 0.025   # risk-free rate for Sharpe ratio (nominal)
INFLATION    = 0.02    # annual CPI assumption; set to 0.0 for nominal output


# ── MC statistics helpers ──────────────────────────────────────────────────────

def _max_drawdown_paths(paths: np.ndarray) -> np.ndarray:
    """Return the maximum drawdown for each simulated path (shape: N_SIMS,).

    Drawdown at step t = (peak_up_to_t - value_t) / peak_up_to_t.
    """
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns   = (running_max - paths) / running_max
    return drawdowns.max(axis=1)


def _ulcer_index_paths(paths: np.ndarray) -> np.ndarray:
    """Return the Ulcer Index for each simulated path (shape: N_SIMS,).

    UI = sqrt( mean( ((price - peak) / peak)^2 ) )
    """
    running_max = np.maximum.accumulate(paths, axis=1)
    pct_drawdown = (running_max - paths) / running_max   # ≥ 0
    return np.sqrt((pct_drawdown ** 2).mean(axis=1))


def _cagr_paths(paths: np.ndarray, initial: float, n_years: int) -> np.ndarray:
    """Annualised return for each simulated path."""
    return (paths[:, -1] / initial) ** (1.0 / n_years) - 1.0


# ── simulate() ────────────────────────────────────────────────────────────────

def simulate(
    scenario_file: str,
    *,
    n_sims:       int   = N_SIMS,
    student_t_nu: int   = STUDENT_T_NU,
    rf_rate:      float = RF_RATE,
    inflation:    float = INFLATION,
) -> dict:
    """Run a full portfolio simulation and return a results dict.

    Parameters
    ----------
    scenario_file : str
        Path to a TOML scenario file.
    n_sims : int
        Number of Monte Carlo paths (default: N_SIMS).
    student_t_nu : int
        Degrees of freedom for Student-t distribution (default: STUDENT_T_NU).
    rf_rate : float
        Nominal risk-free rate for Sharpe ratio (default: RF_RATE).
    inflation : float
        Annual CPI assumption; 0.0 → nominal output (default: INFLATION).

    Returns
    -------
    dict — see key listing in the body below.
    """

    # ── Load scenario ─────────────────────────────────────────────────────────
    with open(scenario_file, "rb") as _f:
        _scenario = tomllib.load(_f)

    initial_capital = _scenario["capital"].get("initial", 0.0)
    contributions   = np.array(_scenario["capital"]["contributions"])
    n_years         = len(contributions)

    _dist = "student-t"

    # ── Portfolio holdings ────────────────────────────────────────────────────
    data = {
        "isin": [
            "IE000XZSV718", "IE000MLMNYS0", "IE000R4ZNTN3", "LU1681045370",
            "IE00BJQRDN15", "IE00BP3QZB59", "IE00BP3QZ825", "IE00BP3QZ601",
            "IE00BH04GL39", "LU1650491282", "IE00BDBRDM35",
            "IE00B4ND3602", "IE00BZ1NCS44",
            "LU2951555403", "IE00B44Z5B48", "IE00BMVB5R75",
        ],
    }
    df = pd.DataFrame(data).set_index("isin")

    _weights = _scenario["weights"]
    _unknown = set(_weights) - set(df.index)
    if _unknown:
        raise ValueError(f"scenario.toml references unknown ISINs: {_unknown}")
    df["weight"] = df.index.map(lambda isin: _weights.get(isin, 0.0))
    _wsum = df["weight"].sum()
    if abs(_wsum - 1.0) > 1e-4:
        raise ValueError(f"Weights in scenario.toml sum to {_wsum:.6f}, expected 1.0")

    # Resolve paths relative to the directory of this script so simulate()
    # works regardless of the caller's cwd.
    _here = Path(__file__).parent

    with open(_here / "assets.toml", "rb") as _f:
        _assets = tomllib.load(_f)["assets"]
    _missing_meta = set(df.index) - set(_assets)
    if _missing_meta:
        raise ValueError(f"assets.toml is missing metadata for ISINs: {_missing_meta}")
    for _col in ("name", "etf", "index", "asset_class"):
        df[_col] = df.index.map(lambda isin, col=_col: _assets[isin][col])

    # ── Load model parameters ─────────────────────────────────────────────────
    _vol_df  = pd.read_csv(_here / "vol.csv",  index_col="isin")
    _corr_df = pd.read_csv(_here / "corr.csv", index_col="isin")
    _mu_df   = pd.read_csv(_here / "mu.csv",   index_col="isin")

    _missing = set(df.index) - set(_vol_df.index)
    if _missing:
        raise ValueError(f"vol.csv is missing ISINs: {_missing}")

    vol = _vol_df.loc[df.index, "vol"].to_numpy()

    # corr.csv stores only the lower triangle; reconstruct the full symmetric matrix
    _corr_raw = _corr_df.loc[df.index, df.index].to_numpy(dtype=float)
    _i_upper  = np.triu_indices_from(_corr_raw, k=1)
    _corr_raw[_i_upper] = _corr_raw.T[_i_upper]
    C = _corr_raw

    mu = _mu_df.loc[df.index, "mu"].to_numpy()

    # Project C to nearest PSD matrix (clip negative eigenvalues → 0)
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 1e-8, None)
    C = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(C))
    C /= np.outer(d, d)

    # ── Portfolio-level analytics ─────────────────────────────────────────────
    D     = np.diag(vol)
    Sigma = D @ C @ D
    w     = df["weight"].to_numpy()

    pf_variance    = w @ Sigma @ w
    pf_vol         = np.sqrt(pf_variance)
    pf_return_nom  = w @ mu
    pf_sharpe_nom  = (pf_return_nom - rf_rate) / pf_vol

    # Fisher equation: real = (1+nominal)/(1+inflation) − 1
    pf_return_real = (1.0 + pf_return_nom) / (1.0 + inflation) - 1.0
    rf_rate_real   = (1.0 + rf_rate)       / (1.0 + inflation) - 1.0
    pf_sharpe_real = (pf_return_real - rf_rate_real) / pf_vol

    # ── Analytical projection (no contributions, V0=1) ────────────────────────
    mu_log  = pf_return_nom - 0.5 * pf_vol**2
    mu_T    = n_years * mu_log
    sigma_T = pf_vol * np.sqrt(n_years)
    pcts    = [5, 25, 50, 75, 95]
    z_scores = norm.ppf([p / 100 for p in pcts])
    analytical_terminals = np.exp(mu_T + sigma_T * z_scores)
    analytical_mean      = np.exp(mu_T + 0.5 * sigma_T**2)
    prob_loss_analytical = norm.cdf(-mu_T / sigma_T) * 100

    # ── Monte Carlo with contributions ────────────────────────────────────────
    rng = np.random.default_rng()  # non-deterministic; results vary each run
    _scale      = pf_vol / np.sqrt(student_t_nu / (student_t_nu - 2))
    log_returns = mu_log + _scale * rng.standard_t(
        df=student_t_nu, size=(n_sims, n_years)
    )

    ret_factors = np.exp(log_returns)

    paths_nom = np.zeros((n_sims, n_years))
    V         = np.full(n_sims, initial_capital, dtype=float)
    for t in range(n_years):
        rf = ret_factors[:, t]
        V  = V * rf + contributions[t] * np.sqrt(rf)   # mid-year timing
        paths_nom[:, t] = V

    total_invested_nom = initial_capital + contributions.sum()

    # ── Deflate to real terms (today's purchasing power) ─────────────────────
    # Each year-t nominal value is divided by (1+π)^t so everything is
    # expressed in year-0 euros.  When inflation=0 real ≡ nominal.
    deflators          = (1.0 + inflation) ** np.arange(1, n_years + 1)
    paths              = paths_nom / deflators[np.newaxis, :]
    contributions_real = contributions / deflators
    total_invested     = initial_capital + contributions_real.sum()

    terminal = paths[:, -1]
    bands    = np.percentile(paths, pcts, axis=0)   # shape (5, n_years)

    # ── Path-level risk statistics (computed on real paths) ──────────────────
    _ref   = total_invested
    cagr   = _cagr_paths(paths, _ref, n_years)
    max_dd = _max_drawdown_paths(paths)
    ulcer  = _ulcer_index_paths(paths)

    def _pct_stats(arr: np.ndarray, name: str) -> dict:
        return {
            f"{name}_p05":  float(np.percentile(arr, 5)),
            f"{name}_p25":  float(np.percentile(arr, 25)),
            f"{name}_p50":  float(np.percentile(arr, 50)),
            f"{name}_p75":  float(np.percentile(arr, 75)),
            f"{name}_p95":  float(np.percentile(arr, 95)),
            f"{name}_mean": float(arr.mean()),
        }

    # ── Assemble results dict ─────────────────────────────────────────────────
    results = {
        # metadata
        "scenario_file":        scenario_file,
        "n_years":              n_years,
        "n_sims":               n_sims,
        "initial_capital":      initial_capital,
        "inflation":            inflation,
        "rf_rate":              rf_rate,
        "student_t_nu":         student_t_nu,
        "total_invested":       total_invested,       # real (today's €)
        "total_invested_nom":   total_invested_nom,   # nominal
        "return_distribution":  _dist,

        # portfolio analytics
        "pf_return":      float(pf_return_real),  # real (Fisher-adjusted)
        "pf_return_nom":  float(pf_return_nom),
        "pf_vol":         float(pf_vol),
        "pf_sharpe":      float(pf_sharpe_real),
        "pf_sharpe_nom":  float(pf_sharpe_nom),

        # MC terminal value — absolute (real)
        "p05":  float(np.percentile(terminal,  5)),
        "p25":  float(np.percentile(terminal, 25)),
        "p50":  float(np.percentile(terminal, 50)),
        "p75":  float(np.percentile(terminal, 75)),
        "p95":  float(np.percentile(terminal, 95)),
        "mean": float(terminal.mean()),

        # MC terminal value — multiples of total invested (real)
        "p05_x":  float(np.percentile(terminal,  5) / total_invested),
        "p25_x":  float(np.percentile(terminal, 25) / total_invested),
        "p50_x":  float(np.percentile(terminal, 50) / total_invested),
        "p75_x":  float(np.percentile(terminal, 75) / total_invested),
        "p95_x":  float(np.percentile(terminal, 95) / total_invested),
        "mean_x": float(terminal.mean()              / total_invested),

        # probabilities
        "prob_loss":       float((terminal < total_invested).mean() * 100),
        "prob_loss_50pct": float((terminal < 0.5 * total_invested).mean() * 100),
        "prob_double":     float((terminal > 2.0 * total_invested).mean() * 100),

        # raw arrays
        "terminal_values": terminal,          # real
        "paths":           paths,             # real
        "paths_nom":       paths_nom,         # nominal (for custom analysis)
        "bands":           bands,             # real, shape (5, n_years)
        "pcts":            pcts,
        "weights":         w,
        "mu_vec":          mu,
        "vol_vec":         vol,
        "corr_matrix":     C,
        "labels":          df["name"].tolist(),
        "asset_classes":   df["asset_class"].tolist(),
        "contributions":   contributions_real,  # real deflated contributions

        # analytical (kept for reference)
        "analytical_terminals":  analytical_terminals,
        "analytical_mean":       float(analytical_mean),
        "prob_loss_analytical":  float(prob_loss_analytical),
    }

    results.update(_pct_stats(cagr,   "cagr"))
    results.update(_pct_stats(max_dd, "max_drawdown"))
    results.update(_pct_stats(ulcer,  "ulcer_index"))

    return results


# ── print helpers ─────────────────────────────────────────────────────────────

def print_results(r: dict) -> None:
    """Pretty-print the results dictionary returned by simulate()."""
    sep   = "=" * 80
    real  = r["inflation"] > 0
    tag   = f"real (today's €, π={r['inflation']*100:.1f}%)" if real else "nominal €"
    rf    = r["rf_rate"]
    nsims = r["n_sims"]

    print(f"\n{sep}")
    print(f"SCENARIO: {r['scenario_file']}")
    print(sep)
    print(f"  Distribution:     {r['return_distribution']}")
    print(f"  Years:            {r['n_years']}")
    print(f"  Simulations:      {nsims:,}")
    print(f"  Inflation (π):    {r['inflation']*100:.1f}%  — all monetary values in {tag}")
    print(f"  Initial capital:  {r['initial_capital']:>14,.0f}")
    print(f"  Total invested:   {r['total_invested']:>14,.0f}  ({tag})")
    if real:
        print(f"  Total invested:   {r['total_invested_nom']:>14,.0f}  (nominal, for reference)")

    print(f"\n{sep}")
    print("PORTFOLIO ANALYTICS")
    print(sep)
    if real:
        print(f"  Expected return — nominal (μ):  {r['pf_return_nom']*100:.2f}%")
        print(f"  Expected return — real    (μ):  {r['pf_return']*100:.2f}%")
    else:
        print(f"  Expected return (μ):            {r['pf_return']*100:.2f}%")
    print(f"  Volatility (σ):                 {r['pf_vol']*100:.2f}%")
    if real:
        rf_real = (1 + rf) / (1 + r["inflation"]) - 1
        print(f"  Sharpe — nominal (rf={rf*100:.1f}% nom):  {r['pf_sharpe_nom']:.3f}")
        print(f"  Sharpe — real    (rf={rf_real*100:.1f}% real):  {r['pf_sharpe']:.3f}")
    else:
        print(f"  Sharpe ratio (rf={rf*100:.1f}%):          {r['pf_sharpe']:.3f}")

    print(f"\n{sep}")
    print(f"MONTE CARLO — TERMINAL VALUE  ({nsims:,} sims, {r['n_years']} years, {tag})")
    print(sep)
    print(f"  {'Pct':<6}  {'Value':>14}  {'× invested':>12}  {'Real total return':>18}")
    print("  " + "-" * 58)
    for lbl, key in [("5th", "p05"), ("25th", "p25"), ("50th", "p50"),
                     ("75th", "p75"), ("95th", "p95"), ("Mean", "mean")]:
        v = r[key]
        print(f"  {lbl:<6}  {v:>14,.0f}  {r[key+'_x']:>12.2f}×  "
              f"{(r[key+'_x']-1)*100:>17.1f}%")

    print(f"\n{sep}")
    print(f"MONTE CARLO — PATH RISK STATISTICS  ({tag})")
    print(sep)
    print(f"  {'Statistic':<20}  {'p05':>8}  {'p25':>8}  {'p50':>8}  "
          f"{'p75':>8}  {'p95':>8}  {'mean':>8}")
    print("  " + "-" * 68)
    for label, key in [("CAGR (real)", "cagr"), ("Max Drawdown", "max_drawdown"),
                        ("Ulcer Index", "ulcer_index")]:
        vals = [f"{r[f'{key}_{s}']*100:>7.1f}%"
                for s in ("p05", "p25", "p50", "p75", "p95", "mean")]
        print(f"  {label:<20}  {'  '.join(vals)}")

    print(f"\n{sep}")
    print("MONTE CARLO — PROBABILITIES")
    print(sep)
    print(f"  P(terminal < total invested):        {r['prob_loss']:.1f}%")
    print(f"  P(terminal < 50% of total invested): {r['prob_loss_50pct']:.1f}%")
    print(f"  P(terminal > 2× total invested):     {r['prob_double']:.1f}%")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from charts import save_charts

    scenario = sys.argv[1] if len(sys.argv) > 1 else "scenarios/scenario_edo_now.toml"
    r = simulate(scenario)

    vol_vec = r["vol_vec"]
    labels  = r["labels"]
    C       = r["corr_matrix"]

    print("=" * 80)
    print("ANNUAL VOLATILITY VECTOR")
    print("=" * 80)
    for name, v in zip(labels, vol_vec):
        print(f"  {name:<30}  {v*100:.1f}%")
    print(f"\nCorrelation matrix min eigenvalue: {np.linalg.eigvalsh(C).min():.6f}  ✓ PSD")

    print_results(r)

    sim_dir       = Path("output")
    sim_dir.mkdir(exist_ok=True)
    scenario_name = Path(scenario).stem
    save_charts(
        r,
        fan_path=str(sim_dir / f"{scenario_name}__fan.png"),
        dist_path=str(sim_dir / f"{scenario_name}__distribution.png"),
    )
