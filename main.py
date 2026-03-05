import tomllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import norm, t as student_t


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
) -> None:
    """Save fan chart and terminal-value histogram to projection.png."""
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
    plt.savefig("projection.png", dpi=150)
    print("\nSaved: projection.png")
    plt.close()


# ── Simulation parameters ─────────────────────────────────────────────────────
N_SIMS    = 100_000
RAND_SEED = 42

# ── Scenario inputs (scenario.toml) ─────────────────────────────────────────
# Edit scenario.toml to change weights, initial capital, or contributions.
# Capital config:
#   initial       — current portfolio value (0 if starting fresh)
#   contributions — one entry per year (annual total; length → N_YEARS)
#     Monthly investors use annual total (e.g. €3k/month → €36k/year).
#     Mid-year timing: each contribution earns (1+r)^0.5 in its arrival year
#     (actuarial standard for uniform monthly cash-flows).
# SCENARIO_FILE = "scenario_ste_mf.toml"
# SCENARIO_FILE = "scenario_ste_mf.toml"
SCENARIO_FILE = "scenario_edo_now.toml"
with open(SCENARIO_FILE, "rb") as _f:
    _scenario = tomllib.load(_f)

INITIAL_CAPITAL = _scenario["capital"].get("initial", 0.0)
CONTRIBUTIONS   = np.array(_scenario["capital"]["contributions"])
N_YEARS         = len(CONTRIBUTIONS)

_dist = _scenario.get("simulation", {}).get("return_distribution", "normal")
if _dist not in ("normal", "student-t"):
    raise ValueError(f"scenario.toml: return_distribution must be 'normal' or 'student-t', got '{_dist}'")
RETURN_DIST = _dist

# ── Portfolio (single source of truth for holdings & metadata) ────────────────
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

# ── Inject weights from scenario.toml ────────────────────────────────────────
_weights = _scenario["weights"]
_unknown = set(_weights) - set(df.index)
if _unknown:
    raise ValueError(f"scenario.toml references unknown ISINs: {_unknown}")
df["weight"] = df.index.map(lambda isin: _weights.get(isin, 0.0))
_wsum = df["weight"].sum()
if abs(_wsum - 1.0) > 1e-4:
    raise ValueError(f"Weights in scenario.toml sum to {_wsum:.6f}, expected 1.0")

df["name"] = [
    "S&P 500 (cap-wtd)", "S&P 500 (equal-wtd)", "MSCI World ex USA",
    "MSCI Emerging Mkts", "Global Multi-Factor", "MSCI World Value",
    "MSCI World Momentum", "MSCI World Quality", "EUR Govt Bonds",
    "EUR Inflation-Linked", "Global Agg (EUR hdg)", "Gold", "Broad Commodities",
    "Managed Futures", "MSCI ACWI", "LifeStrategy 80/20",
]
df["etf"] = [
    "SPDR S&P 500 UCITS ETF", "iShares S&P 500 Equal Weight UCITS ETF",
    "iShares MSCI World ex-USA UCITS ETF", "Amundi MSCI Emerging Markets Swap ETF",
    "Invesco Global Equity Multi-Factor ESG ETF", "iShares MSCI World Value Factor UCITS ETF",
    "iShares MSCI World Momentum Factor UCITS ETF", "iShares MSCI World Quality Factor UCITS ETF",
    "Vanguard EUR Eurozone Govt Bond UCITS ETF", "Amundi Euro Govt Inflation-Linked Bond ETF",
    "iShares Core Global Aggregate Bond EUR Hdg ETF", "iShares Physical Gold ETC",
    "iShares Bloomberg Roll Select Commodity ETF",
    "iMGP DBi Managed Futures R EUR ETF", "SPDR MSCI ACWI UCITS ETF",
    "Vanguard LifeStrategy 80% Equity UCITS ETF",
]
df["index"] = [
    "S&P 500", "S&P 500 Equal Weight", "MSCI World ex USA", "MSCI Emerging Markets",
    "IQS Global Multi-Factor (proprietary)", "MSCI World Enhanced Value",
    "MSCI World Momentum", "MSCI World Sector Neutral Quality",
    "Bloomberg Euro Aggregate Treasury", "Bloomberg Euro Govt Inflation-Linked",
    "Bloomberg Global Aggregate (EUR hdg)", "Gold Spot", "Bloomberg Roll Select Commodity",
    "iMGP DBi Managed Futures (CTA replication)", "MSCI All Country World (ACWI)",
    "Vanguard LifeStrategy 80% Equity (fund-of-ETFs)",
]
df["asset_class"] = [
    "Equity – US", "Equity – US", "Equity – Developed ex-US", "Equity – Emerging Markets",
    "Equity – Global Factor", "Equity – Global Factor",
    "Equity – Global Factor", "Equity – Global Factor",
    "Bonds – EUR Govts", "Bonds – EUR Inflation-Linked", "Bonds – Global Aggregate",
    "Alternatives – Gold", "Alternatives – Commodities",
    "Alternatives – Managed Futures", "Equity – Global", "Multi-Asset – 80/20",
]

print("=" * 80)
print("PORTFOLIO HOLDINGS")
print("=" * 80)
print(df[["name", "etf", "index", "asset_class", "weight"]].to_string())
print(f"\nTotal weight: {df['weight'].sum():.4f}")

# ── Load model parameters from CSV ──────────────────────────────────────────
# Edit vol.csv, corr.csv, mu.csv to update estimates — no code changes needed.
# corr.csv stores the raw (intended) correlations; PSD projection is always
# applied below so the matrix remains valid even after manual edits.
_vol_df  = pd.read_csv("vol.csv",  index_col="isin")
_corr_df = pd.read_csv("corr.csv", index_col="isin")
_mu_df   = pd.read_csv("mu.csv",   index_col="isin")

_missing = set(df.index) - set(_vol_df.index)
if _missing:
    raise ValueError(f"vol.csv is missing ISINs: {_missing}")

vol = _vol_df.loc[df.index, "vol"].to_numpy()
C   = _corr_df.loc[df.index, df.index].to_numpy()
mu  = _mu_df.loc[df.index, "mu"].to_numpy()

print("\n" + "=" * 80)
print("ANNUAL VOLATILITY VECTOR (long-run index estimates)")
print("=" * 80)
for name, v in zip(df["name"], vol):
    print(f"  {name:<30}  {v*100:.1f}%")

# Project to nearest PSD matrix (clip negative eigenvalues → 0, rescale diagonal)
vals, vecs = np.linalg.eigh(C)
vals = np.clip(vals, 1e-8, None)
C = vecs @ np.diag(vals) @ vecs.T
d = np.sqrt(np.diag(C))
C /= np.outer(d, d)

print(f"\nCorrelation matrix min eigenvalue after projection: {np.linalg.eigvalsh(C).min():.6f}  ✓ PSD")

# ── Covariance matrix ─────────────────────────────────────────────────────────
# Σ = D · C · D   where D = diag(σ)
D = np.diag(vol)
Sigma = D @ C @ D

print("\n" + "=" * 80)
print("COVARIANCE MATRIX (annual)")
print("=" * 80)
labels = df["name"].tolist()
col_w = 22
header = f"{'':>{col_w}}" + "".join(f"  {l[:10]:>10}" for l in labels)
print(header)
for i, row_label in enumerate(labels):
    row = f"{row_label:>{col_w}}" + "".join(f"  {Sigma[i, j]:>10.6f}" for j in range(len(labels)))
    print(row)

# ── Plots ─────────────────────────────────────────────────────────────────────
labels = df["name"].tolist()
plot_correlation_matrix(C, labels)
plot_volatility(vol, labels, df["asset_class"].tolist())

# ── Portfolio variance & volatility ───────────────────────────────────────────
# σ_p² = wᵀ Σ w
w = df["weight"].to_numpy()
pf_variance = w @ Sigma @ w
pf_vol      = np.sqrt(pf_variance)

# Marginal risk contribution: MRC_i = (Σw)_i * w_i / σ_p
marginal    = Sigma @ w
risk_contrib = w * marginal / pf_vol          # absolute contribution to σ_p
risk_pct     = risk_contrib / pf_vol * 100    # % of total variance explained

print("\n" + "=" * 80)
print("PORTFOLIO RISK SUMMARY")
print("=" * 80)
print(f"  Portfolio variance (σ²):   {pf_variance:.6f}")
print(f"  Portfolio volatility (σ):  {pf_vol*100:.2f}%")

print("\n  Risk contribution by asset:")
print(f"  {'Asset':<30}  {'Weight':>8}  {'σ (ann)':>8}  {'RC (pp)':>8}  {'RC (%)':>8}")
print("  " + "-" * 70)
for i, (name, wi, vi, rci, rpci) in enumerate(
        zip(df["name"], w, vol, risk_contrib, risk_pct)):
    print(f"  {name:<30}  {wi*100:>7.2f}%  {vi*100:>7.1f}%  {rci*100:>7.2f}pp  {rpci:>7.2f}%")
print("  " + "-" * 70)
print(f"  {'TOTAL':<30}  {w.sum()*100:>7.2f}%  {'':>8}  {risk_contrib.sum()*100:>7.2f}pp  {risk_pct.sum():>7.2f}%")

# ── Expected returns loaded from mu.csv ──────────────────────────────────────
# (building-block methodology documented in _gen_params.py)
pf_return = w @ mu

print("\n" + "=" * 80)
print("EXPECTED RETURNS (nominal, EUR, annual)")
print("=" * 80)
print(f"  {'Asset':<30}  {'Weight':>8}  {'E[r]':>8}")
print("  " + "-" * 52)
for name, wi, mui in zip(df["name"], w, mu):
    print(f"  {name:<30}  {wi*100:>7.2f}%  {mui*100:>7.2f}%")
print("  " + "-" * 52)
print(f"  {'PORTFOLIO':<30}  {w.sum()*100:>7.2f}%  {pf_return*100:>7.2f}%")

print("\n" + "=" * 80)
print("RISK / RETURN SUMMARY")
print("=" * 80)
print(f"  Expected return (μ_p):     {pf_return*100:.2f}%")
print(f"  Volatility (σ_p):          {pf_vol*100:.2f}%")
print(f"  Sharpe ratio (rf=2.5%):    {(pf_return - 0.025)/pf_vol:.3f}")

# ── Analytical projection (no contributions, normalised V0=1) ─────────────────
# Terminal value is EXACTLY log-normal under the GBM / log-normal return model:
#   log(V_T/V_0) ~ N(T·μ_log, T·σ²)
# This holds because we model log-returns as normal (the standard assumption).
# Arithmetic returns r_t are then approximately normal for small σ, but the
# fundamental distributional claim lives in log-return space.
# Once contributions are added the terminal value becomes a sum of correlated
# log-normals — no closed form → Monte Carlo is the right tool.
INITIAL = 1.0   # normalised reference for the analytical section
mu_log  = pf_return - 0.5 * pf_vol**2   # annual log-drift (Itô correction)
mu_T    = N_YEARS * mu_log
sigma_T = pf_vol * np.sqrt(N_YEARS)      # σ·√T

pcts    = [5, 25, 50, 75, 95]
z_scores = norm.ppf([p / 100 for p in pcts])
analytical_terminals = INITIAL * np.exp(mu_T + sigma_T * z_scores)
analytical_mean      = INITIAL * np.exp(mu_T + 0.5 * sigma_T**2)

print("\n" + "=" * 80)
print(f"ANALYTICAL PROJECTION — no contributions, V0=1  ({N_YEARS} years)")
print("=" * 80)
print(f"  μ_log = μ - ½σ²: {mu_log*100:.3f}%    "
      f"T·μ_log: {mu_T*100:.2f}%    σ·√T: {sigma_T*100:.2f}%")
print(f"  {'Percentile':<10}  {'Growth (×)':>12}  {'Total return':>13}")
print("  " + "-" * 40)
for p, v in zip(pcts, analytical_terminals):
    print(f"  {str(p)+'th':<10}  {v:>12.4f}  {(v - 1)*100:>12.1f}%")
print(f"  {'Mean':<10}  {analytical_mean:>12.4f}  {(analytical_mean - 1)*100:>12.1f}%")
prob_loss_analytical = norm.cdf(-mu_T / sigma_T) * 100
print(f"  Probability of loss after {N_YEARS}y: {prob_loss_analytical:.1f}%")

# ── Monte Carlo projection WITH contributions ─────────────────────────────────
# Log-returns are drawn (consistent with the log-normal model):
#   log(1+r_t) ~ N(μ_log, σ²)  →  terminal V_T is exactly log-normal (no contrib)
# Student-t variant: t(ν=6) innovations rescaled to unit variance so σ_p is
#   preserved.  Var(t_ν) = ν/(ν-2), so scale = σ_p / √(ν/(ν-2)).  The mean
#   is still μ_log (t is symmetric, zero-mean before shift).
#   Fat tails → lower 5th percentile, heavier left tail vs normal.
# Mid-year contribution timing:
#   V_t = V_{t-1}·(1+r_t) + C_t·(1+r_t)^0.5
STUDENT_T_NU = 6
rng = np.random.default_rng(RAND_SEED)
if RETURN_DIST == "student-t":
    _scale = pf_vol / np.sqrt(STUDENT_T_NU / (STUDENT_T_NU - 2))  # rescale to unit variance
    log_returns = mu_log + _scale * rng.standard_t(df=STUDENT_T_NU, size=(N_SIMS, N_YEARS))
else:
    log_returns = rng.normal(loc=mu_log, scale=pf_vol, size=(N_SIMS, N_YEARS))
ret_factors = np.exp(log_returns)                 # (1+r_t) from log-return draw

paths_contrib = np.zeros((N_SIMS, N_YEARS))
V = np.full(N_SIMS, INITIAL_CAPITAL, dtype=float)
for t in range(N_YEARS):
    rf = ret_factors[:, t]
    V  = V * rf + CONTRIBUTIONS[t] * np.sqrt(rf)   # mid-year: ×rf^0.5 = ×√rf
    paths_contrib[:, t] = V

terminal_contrib = paths_contrib[:, -1]
bands_contrib    = np.percentile(paths_contrib, pcts, axis=0)  # (5, N_YEARS)

total_invested = INITIAL_CAPITAL + CONTRIBUTIONS.sum()
_dist_label = f"Student-t (ν={STUDENT_T_NU})" if RETURN_DIST == "student-t" else "Normal"
print("\n" + "=" * 80)
print(f"MONTE CARLO WITH CONTRIBUTIONS  ({N_SIMS:,} sims, {N_YEARS} years, {_dist_label})")
print("=" * 80)
print(f"  Initial capital:       {INITIAL_CAPITAL:>12,.0f}")
print(f"  Annual contributions:  {CONTRIBUTIONS.sum()/N_YEARS:>12,.0f}  (×{N_YEARS} years)")
print(f"  Total invested:        {total_invested:>12,.0f}")
print()
print(f"  {'Percentile':<10}  {'Terminal':>14}  {'Total return':>13}  {'On invested':>12}")
print("  " + "-" * 56)
for p, v in zip(pcts, bands_contrib[:, -1]):
    print(f"  {str(p)+'th':<10}  {v:>14,.0f}"
          f"  {(v/total_invested - 1)*100:>12.1f}%"
          f"  {'×'+f'{v/total_invested:.2f}':>12}")
mc_mean = terminal_contrib.mean()
print(f"  {'Mean':<10}  {mc_mean:>14,.0f}"
      f"  {(mc_mean/total_invested - 1)*100:>12.1f}%"
      f"  {'×'+f'{mc_mean/total_invested:.2f}':>12}")
mc_prob_loss = (terminal_contrib < total_invested).mean() * 100
print(f"\n  Probability that terminal value < total invested: {mc_prob_loss:.1f}%")

# ── Plots ─────────────────────────────────────────────────────────────────────
plot_projection(bands_contrib, terminal_contrib, INITIAL_CAPITAL, CONTRIBUTIONS, pcts)
