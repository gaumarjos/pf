import tomllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from pathlib import Path
from scipy.stats import norm, t as student_t


# ── Simulation constants ───────────────────────────────────────────────────────
N_SIMS    = 100_000
RAND_SEED = 42
STUDENT_T_NU = 6
RF_RATE   = 0.025   # risk-free rate for Sharpe ratio


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


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_correlation_matrix(C: np.ndarray, labels: list) -> None:
    """Save a correlation heatmap to correlation_matrix.png."""
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        C, annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=labels,
        cmap="RdYlGn", vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Asset Class Correlation Matrix\n(long-run index estimates)", fontsize=13, pad=12)
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=150)
    print("\nSaved: correlation_matrix.png")
    plt.close()


def plot_volatility(vol: np.ndarray, labels: list, asset_classes: list) -> None:
    """Save a horizontal bar chart of annualised volatilities to volatility_vector.png."""
    order  = np.argsort(vol)
    colors = ["#e67e22" if "Equity" in asset_classes[i] else
              "#2980b9" if "Bond"   in asset_classes[i] else
              "#27ae60" for i in order]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh([labels[i] for i in order], vol[order] * 100, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xlabel("Annual Volatility (%)")
    ax.set_title("Annual Volatility by Asset (long-run index estimates)", fontsize=12)
    ax.set_xlim(0, vol.max() * 100 * 1.18)
    ax.legend(handles=[
        Patch(facecolor="#e67e22", label="Equity"),
        Patch(facecolor="#2980b9", label="Bonds"),
        Patch(facecolor="#27ae60", label="Alternatives"),
    ], loc="lower right")
    plt.tight_layout()
    plt.savefig("volatility_vector.png", dpi=150)
    print("Saved: volatility_vector.png")
    plt.close()


def plot_projection(
    bands_contrib:    np.ndarray,
    terminal_contrib: np.ndarray,
    initial_capital:  float,
    contributions:    np.ndarray,
    pcts:             list,
    out_path:         str = "projection.png",
) -> None:
    """Save fan chart and terminal-value histogram to *out_path*."""
    n_years        = len(contributions)
    years          = np.arange(1, n_years + 1)
    years_with_0   = np.concatenate([[0], years])
    total_invested = initial_capital + contributions.sum()
    fan_colors     = ["#c0392b", "#e67e22", "#27ae60", "#e67e22", "#c0392b"]
    labels_fan     = ["5th", "25th", "50th (median)", "75th", "95th"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Fan chart
    ax = axes[0]
    cumulative_invested = np.concatenate([[initial_capital],
        initial_capital + np.cumsum(contributions)])
    for i in range(len(pcts)):
        vals = np.concatenate([[initial_capital], bands_contrib[i]])
        ax.plot(years_with_0, vals, color=fan_colors[i],
                lw=2.5 if i == 2 else 1.5,
                linestyle="--" if i in (0, 4) else "-",
                label=labels_fan[i])
    ax.fill_between(years_with_0,
                    np.concatenate([[initial_capital], bands_contrib[0]]),
                    np.concatenate([[initial_capital], bands_contrib[4]]),
                    alpha=0.10, color="steelblue")
    ax.fill_between(years_with_0,
                    np.concatenate([[initial_capital], bands_contrib[1]]),
                    np.concatenate([[initial_capital], bands_contrib[3]]),
                    alpha=0.20, color="steelblue")
    ax.plot(years_with_0, cumulative_invested, color="black", lw=1.2,
            linestyle=":", label="Cumulative invested")
    ax.set_title(f"Portfolio Fan Chart — {n_years}y with contributions", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value (€)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=8)
    ax.set_xlim(0, n_years)

    # Terminal-value distribution
    ax2 = axes[1]
    ax2.hist(terminal_contrib, bins=120, color="steelblue", edgecolor="none", alpha=0.75)
    for p, v in zip(pcts, bands_contrib[:, -1]):
        ax2.axvline(v, color="#c0392b" if p in (5, 95) else
                       "#e67e22" if p in (25, 75) else "#27ae60",
                    lw=1.5, linestyle="--", label=f"{p}th: {v:,.0f}")
    ax2.axvline(total_invested, color="black", lw=1.2, linestyle=":",
                label=f"Total invested: {total_invested:,.0f}")
    ax2.set_title(f"Terminal Value Distribution (Year {n_years})", fontsize=12)
    ax2.set_xlabel("Portfolio Value (€)")
    ax2.set_ylabel("Frequency")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    plt.close()




# ── simulate() ────────────────────────────────────────────────────────────────

def simulate(scenario_file: str) -> dict:
    """Run a full portfolio simulation for *scenario_file* and return a stats dict.

    Parameters
    ----------
    scenario_file : str
        Path to a TOML scenario file (e.g. ``"scenarios/scenario_edo_now.toml"``).

    Returns
    -------
    dict with keys:

    Scenario / portfolio metadata
        scenario_file, n_years, initial_capital, total_invested,
        return_distribution, pf_return, pf_vol, pf_sharpe,

    Monte Carlo — terminal value percentiles
        p05, p25, p50, p75, p95, mean,

    Monte Carlo — terminal value ratios (relative to total invested)
        p05_x, p25_x, p50_x, p75_x, p95_x, mean_x,

    Monte Carlo — path-level risk statistics (mean across sims)
        cagr_p05, cagr_p25, cagr_p50, cagr_p75, cagr_p95, cagr_mean,
        max_drawdown_p05, max_drawdown_p25, max_drawdown_p50,
        max_drawdown_p75, max_drawdown_p95, max_drawdown_mean,
        ulcer_index_p05, ulcer_index_p25, ulcer_index_p50,
        ulcer_index_p75, ulcer_index_p95, ulcer_index_mean,

    Probabilities
        prob_loss,          — P(terminal < total invested)
        prob_loss_50pct,    — P(terminal < 0.5 × total invested)
        prob_double,        — P(terminal > 2 × total invested)

    Raw arrays (for further analysis / plotting)
        terminal_values, paths, bands, pcts,
        weights, mu_vec, vol_vec, corr_matrix, labels,
    """

    # ── Load scenario ─────────────────────────────────────────────────────────
    with open(scenario_file, "rb") as _f:
        _scenario = tomllib.load(_f)

    initial_capital = _scenario["capital"].get("initial", 0.0)
    contributions   = np.array(_scenario["capital"]["contributions"])
    n_years         = len(contributions)

    _dist = _scenario.get("simulation", {}).get("return_distribution", "normal")
    if _dist not in ("normal", "student-t"):
        raise ValueError(
            f"scenario.toml: return_distribution must be 'normal' or 'student-t', got '{_dist}'"
        )

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

    with open("assets.toml", "rb") as _f:
        _assets = tomllib.load(_f)["assets"]
    _missing_meta = set(df.index) - set(_assets)
    if _missing_meta:
        raise ValueError(f"assets.toml is missing metadata for ISINs: {_missing_meta}")
    for _col in ("name", "etf", "index", "asset_class"):
        df[_col] = df.index.map(lambda isin, col=_col: _assets[isin][col])

    # ── Load model parameters ─────────────────────────────────────────────────
    _vol_df  = pd.read_csv("vol.csv",  index_col="isin")
    _corr_df = pd.read_csv("corr.csv", index_col="isin")
    _mu_df   = pd.read_csv("mu.csv",   index_col="isin")

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

    pf_variance = w @ Sigma @ w
    pf_vol      = np.sqrt(pf_variance)
    pf_return   = w @ mu
    pf_sharpe   = (pf_return - RF_RATE) / pf_vol

    # ── Analytical projection (no contributions, V0=1) ────────────────────────
    mu_log  = pf_return - 0.5 * pf_vol**2
    mu_T    = n_years * mu_log
    sigma_T = pf_vol * np.sqrt(n_years)
    pcts    = [5, 25, 50, 75, 95]
    z_scores = norm.ppf([p / 100 for p in pcts])
    analytical_terminals = np.exp(mu_T + sigma_T * z_scores)
    analytical_mean      = np.exp(mu_T + 0.5 * sigma_T**2)
    prob_loss_analytical = norm.cdf(-mu_T / sigma_T) * 100

    # ── Monte Carlo with contributions ────────────────────────────────────────
    rng = np.random.default_rng(RAND_SEED)
    if _dist == "student-t":
        _scale      = pf_vol / np.sqrt(STUDENT_T_NU / (STUDENT_T_NU - 2))
        log_returns = mu_log + _scale * rng.standard_t(
            df=STUDENT_T_NU, size=(N_SIMS, n_years)
        )
    else:
        log_returns = rng.normal(loc=mu_log, scale=pf_vol, size=(N_SIMS, n_years))

    ret_factors = np.exp(log_returns)

    paths  = np.zeros((N_SIMS, n_years))
    V      = np.full(N_SIMS, initial_capital, dtype=float)
    for t in range(n_years):
        rf = ret_factors[:, t]
        V  = V * rf + contributions[t] * np.sqrt(rf)   # mid-year timing
        paths[:, t] = V

    terminal     = paths[:, -1]
    bands        = np.percentile(paths, pcts, axis=0)   # (5, n_years)
    total_invested = initial_capital + contributions.sum()

    # ── Path-level risk statistics ────────────────────────────────────────────
    # CAGR reference: use total_invested so the metric is meaningful even when
    # initial_capital is 0 (pure contribution scenario).
    _ref = total_invested
    cagr          = _cagr_paths(paths, _ref, n_years)
    max_dd        = _max_drawdown_paths(paths)
    ulcer         = _ulcer_index_paths(paths)

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
        # ── metadata ─────────────────────────────────────────────────────────
        "scenario_file":       scenario_file,
        "n_years":             n_years,
        "initial_capital":     initial_capital,
        "total_invested":      total_invested,
        "return_distribution": _dist,

        # ── portfolio analytics ───────────────────────────────────────────────
        "pf_return":  float(pf_return),
        "pf_vol":     float(pf_vol),
        "pf_sharpe":  float(pf_sharpe),

        # ── MC terminal value — absolute ──────────────────────────────────────
        "p05":  float(bands[-1, -1] if False else np.percentile(terminal, 5)),
        "p25":  float(np.percentile(terminal, 25)),
        "p50":  float(np.percentile(terminal, 50)),
        "p75":  float(np.percentile(terminal, 75)),
        "p95":  float(np.percentile(terminal, 95)),
        "mean": float(terminal.mean()),

        # ── MC terminal value — multiples of total invested ───────────────────
        "p05_x":  float(np.percentile(terminal, 5)  / total_invested),
        "p25_x":  float(np.percentile(terminal, 25) / total_invested),
        "p50_x":  float(np.percentile(terminal, 50) / total_invested),
        "p75_x":  float(np.percentile(terminal, 75) / total_invested),
        "p95_x":  float(np.percentile(terminal, 95) / total_invested),
        "mean_x": float(terminal.mean()             / total_invested),

        # ── probabilities ─────────────────────────────────────────────────────
        "prob_loss":       float((terminal < total_invested).mean() * 100),
        "prob_loss_50pct": float((terminal < 0.5 * total_invested).mean() * 100),
        "prob_double":     float((terminal > 2.0 * total_invested).mean() * 100),

        # ── raw arrays ───────────────────────────────────────────────────────
        "terminal_values": terminal,
        "paths":           paths,
        "bands":           bands,
        "pcts":            pcts,
        "weights":         w,
        "mu_vec":          mu,
        "vol_vec":         vol,
        "corr_matrix":     C,
        "labels":          df["name"].tolist(),
        "asset_classes":   df["asset_class"].tolist(),
        "contributions":   contributions,

        # ── analytical (kept for reference) ──────────────────────────────────
        "analytical_terminals":   analytical_terminals,
        "analytical_mean":        float(analytical_mean),
        "prob_loss_analytical":   float(prob_loss_analytical),
    }

    # ── path-level risk stats ─────────────────────────────────────────────────
    results.update(_pct_stats(cagr,   "cagr"))
    results.update(_pct_stats(max_dd, "max_drawdown"))
    results.update(_pct_stats(ulcer,  "ulcer_index"))

    return results


# ── print helpers ─────────────────────────────────────────────────────────────

def print_results(r: dict) -> None:
    """Pretty-print the results dictionary returned by simulate()."""
    sep = "=" * 80

    print(f"\n{sep}")
    print(f"SCENARIO: {r['scenario_file']}")
    print(sep)
    print(f"  Distribution:     {r['return_distribution']}")
    print(f"  Years:            {r['n_years']}")
    print(f"  Initial capital:  {r['initial_capital']:>14,.0f}")
    print(f"  Total invested:   {r['total_invested']:>14,.0f}")

    print(f"\n{sep}")
    print("PORTFOLIO ANALYTICS")
    print(sep)
    print(f"  Expected return (μ):   {r['pf_return']*100:.2f}%")
    print(f"  Volatility (σ):        {r['pf_vol']*100:.2f}%")
    print(f"  Sharpe ratio (rf={RF_RATE*100:.1f}%): {r['pf_sharpe']:.3f}")

    print(f"\n{sep}")
    print(f"MONTE CARLO — TERMINAL VALUE  ({N_SIMS:,} sims, {r['n_years']} years)")
    print(sep)
    print(f"  {'Pct':<6}  {'Value':>14}  {'× invested':>12}  {'Total return':>13}")
    print("  " + "-" * 52)
    for lbl, key in [("5th", "p05"), ("25th", "p25"), ("50th", "p50"),
                     ("75th", "p75"), ("95th", "p95"), ("Mean", "mean")]:
        v = r[key]
        print(f"  {lbl:<6}  {v:>14,.0f}  {r[key+'_x']:>12.2f}×  "
              f"{(r[key+'_x']-1)*100:>12.1f}%")

    print(f"\n{sep}")
    print("MONTE CARLO — PATH RISK STATISTICS")
    print(sep)
    print(f"  {'Statistic':<20}  {'p05':>8}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'p95':>8}  {'mean':>8}")
    print("  " + "-" * 68)
    for label, key in [("CAGR", "cagr"), ("Max Drawdown", "max_drawdown"), ("Ulcer Index", "ulcer_index")]:
        vals = [f"{r[f'{key}_{s}']*100:>7.1f}%" for s in ("p05", "p25", "p50", "p75", "p95", "mean")]
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

    scenario = sys.argv[1] if len(sys.argv) > 1 else "scenarios/scenario_edo_now.toml"
    r = simulate(scenario)

    # Print portfolio metadata
    labels       = r["labels"]
    asset_classes = r["asset_classes"]
    w            = r["weights"]
    mu_vec       = r["mu_vec"]
    vol_vec      = r["vol_vec"]
    C            = r["corr_matrix"]

    print("=" * 80)
    print("ANNUAL VOLATILITY VECTOR")
    print("=" * 80)
    for name, v in zip(labels, vol_vec):
        print(f"  {name:<30}  {v*100:.1f}%")

    print(f"\nCorrelation matrix min eigenvalue: {np.linalg.eigvalsh(C).min():.6f}  ✓ PSD")

    print_results(r)

    # ── Plots ─────────────────────────────────────────────────────────────────
    sim_dir = Path("simulations")
    sim_dir.mkdir(exist_ok=True)
    scenario_name = Path(scenario).stem          # e.g. "scenario_edo_now"
    proj_path     = sim_dir / f"{scenario_name}__projection.png"

    plot_correlation_matrix(C, labels)
    plot_volatility(vol_vec, labels, asset_classes)
    plot_projection(r["bands"], r["terminal_values"],
                    r["initial_capital"], r["contributions"], r["pcts"],
                    out_path=str(proj_path))
