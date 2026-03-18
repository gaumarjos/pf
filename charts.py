"""charts.py — All matplotlib/seaborn visualisation for the portfolio simulator.

Main entry points
-----------------
chart_projection(r)         → base64 PNG string
chart_correlation(r)        → base64 PNG string
chart_volatility(r)         → base64 PNG string
save_charts(r, proj_path)   → saves the three PNGs to disk (CLI use)

All chart_* functions accept the dict returned by simulator.simulate() and
return a base64-encoded PNG so the Flask app can embed them directly in HTML
without writing temporary files.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server use

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch


# ── Internal helper ────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    """Encode a matplotlib Figure to a base64 PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


# ── Public chart functions ─────────────────────────────────────────────────────

def chart_projection(r: dict) -> str:
    """Fan chart + terminal-value histogram.  Returns base64 PNG string."""
    bands        = r["bands"]
    terminal     = r["terminal_values"]
    initial      = r["initial_capital"]
    contributions = r["contributions"]   # already real-deflated
    pcts         = r["pcts"]
    inflation    = r["inflation"]

    n_years        = len(contributions)
    years          = np.arange(1, n_years + 1)
    years_with_0   = np.concatenate([[0], years])
    total_invested = initial + contributions.sum()

    fan_colors = ["#c0392b", "#e67e22", "#27ae60", "#e67e22", "#c0392b"]
    labels_fan = ["5th", "25th", "50th (median)", "75th", "95th"]
    value_label = (f"real (today's €, π={inflation*100:.1f}%)"
                   if inflation > 0 else "nominal €")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Fan chart ──────────────────────────────────────────────────────────────
    ax = axes[0]
    cumulative_invested = np.concatenate(
        [[initial], initial + np.cumsum(contributions)]
    )
    for i in range(len(pcts)):
        vals = np.concatenate([[initial], bands[i]])
        ax.plot(years_with_0, vals,
                color=fan_colors[i],
                lw=2.5 if i == 2 else 1.5,
                linestyle="--" if i in (0, 4) else "-",
                label=labels_fan[i])
    ax.fill_between(years_with_0,
                    np.concatenate([[initial], bands[0]]),
                    np.concatenate([[initial], bands[4]]),
                    alpha=0.10, color="steelblue")
    ax.fill_between(years_with_0,
                    np.concatenate([[initial], bands[1]]),
                    np.concatenate([[initial], bands[3]]),
                    alpha=0.20, color="steelblue")
    ax.plot(years_with_0, cumulative_invested,
            color="black", lw=1.2, linestyle=":", label="Cumulative invested")
    ax.set_title(f"Portfolio Fan Chart — {n_years}y with contributions\n({value_label})",
                 fontsize=11)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Portfolio Value ({value_label})")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=8)
    ax.set_xlim(0, n_years)

    # ── Terminal-value histogram ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(terminal, bins=120, color="steelblue", edgecolor="none", alpha=0.75)
    for p, v in zip(pcts, bands[:, -1]):
        ax2.axvline(v,
                    color="#c0392b" if p in (5, 95) else
                          "#e67e22" if p in (25, 75) else "#27ae60",
                    lw=1.5, linestyle="--", label=f"{p}th: {v:,.0f}")
    ax2.axvline(total_invested, color="black", lw=1.2, linestyle=":",
                label=f"Total invested: {total_invested:,.0f}")
    ax2.set_title(f"Terminal Value Distribution (Year {n_years})\n({value_label})",
                  fontsize=11)
    ax2.set_xlabel(f"Portfolio Value ({value_label})")
    ax2.set_ylabel("Frequency")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.legend(fontsize=8)

    plt.tight_layout()
    return _fig_to_b64(fig)


def chart_correlation(r: dict) -> str:
    """Correlation heatmap.  Returns base64 PNG string."""
    C      = r["corr_matrix"]
    labels = r["labels"]

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        C, annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=labels,
        cmap="RdYlGn", vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Asset Class Correlation Matrix\n(long-run index estimates)",
                 fontsize=13, pad=12)
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    return _fig_to_b64(fig)


def chart_volatility(r: dict) -> str:
    """Horizontal volatility bar chart.  Returns base64 PNG string."""
    vol          = r["vol_vec"]
    labels       = r["labels"]
    asset_classes = r["asset_classes"]

    order  = np.argsort(vol)
    colors = [
        "#e67e22" if "Equity" in asset_classes[i] else
        "#2980b9" if "Bond"   in asset_classes[i] else
        "#27ae60"
        for i in order
    ]
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
    return _fig_to_b64(fig)


# ── CLI convenience wrapper ────────────────────────────────────────────────────

def save_charts(r: dict, proj_path: str = "projection.png") -> None:
    """Save the three standard charts to disk.  Used by simulator.py CLI."""
    import matplotlib
    matplotlib.use("Agg")

    # Correlation
    _save_b64_to_file(chart_correlation(r), "correlation_matrix.png")
    print("Saved: correlation_matrix.png")

    # Volatility
    _save_b64_to_file(chart_volatility(r), "volatility_vector.png")
    print("Saved: volatility_vector.png")

    # Projection
    _save_b64_to_file(chart_projection(r), proj_path)
    print(f"Saved: {proj_path}")


def _save_b64_to_file(b64: str, path: str) -> None:
    """Decode a base64 PNG string and write it to *path*."""
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
