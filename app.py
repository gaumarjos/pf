"""app.py — Flask web interface for the portfolio Monte Carlo simulator.

Usage
-----
    python app.py          # starts on http://127.0.0.1:5000
"""

import os
from pathlib import Path

from flask import Flask, render_template, request

import simulator
import charts

app = Flask(__name__)

# ── Resolve paths relative to this file ───────────────────────────────────────
_HERE = Path(__file__).parent


def _list_scenarios() -> list[str]:
    """Return sorted list of scenario file stems (e.g. 'scenario_edo_now')."""
    scen_dir = _HERE / "scenarios"
    return sorted(p.stem for p in scen_dir.glob("*.toml"))


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template(
        "index.html",
        scenarios=_list_scenarios(),
        defaults=_form_defaults(),
        results=None,
        charts=None,
        error=None,
    )


@app.post("/run")
def run():
    form = request.form

    # ── Parse form values ──────────────────────────────────────────────────────
    scenario_stem = form.get("scenario", _list_scenarios()[0])
    scenario_file = str(_HERE / "scenarios" / f"{scenario_stem}.toml")

    try:
        n_sims       = int(form.get("n_sims", simulator.N_SIMS))
        student_t_nu = int(form.get("student_t_nu", simulator.STUDENT_T_NU))
        rf_rate      = float(form.get("rf_rate", simulator.RF_RATE * 100)) / 100.0
        inflation    = float(form.get("inflation", simulator.INFLATION * 100)) / 100.0
    except (ValueError, TypeError) as exc:
        return render_template(
            "index.html",
            scenarios=_list_scenarios(),
            defaults=_form_defaults(),
            results=None,
            charts=None,
            error=f"Invalid parameter value: {exc}",
        )

    # ── Run simulation ─────────────────────────────────────────────────────────
    try:
        r = simulator.simulate(
            scenario_file,
            n_sims=n_sims,
            student_t_nu=student_t_nu,
            rf_rate=rf_rate,
            inflation=inflation,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            scenarios=_list_scenarios(),
            defaults=_form_defaults(),
            results=None,
            charts=None,
            error=str(exc),
        )

    # ── Generate charts ────────────────────────────────────────────────────────
    chart_data = {
        "projection": charts.chart_projection(r),
    }

    # Convert numpy arrays to plain Python so Jinja2 can iterate them
    r_safe = _make_serialisable(r)

    current = {
        "scenario":     scenario_stem,
        "n_sims":       n_sims,
        "student_t_nu": student_t_nu,
        "rf_rate":      rf_rate * 100,
        "inflation":    inflation * 100,
    }

    return render_template(
        "index.html",
        scenarios=_list_scenarios(),
        defaults=current,
        results=r_safe,
        charts=chart_data,
        error=None,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _form_defaults() -> dict:
    return {
        "scenario":     _list_scenarios()[0] if _list_scenarios() else "",
        "n_sims":       simulator.N_SIMS,
        "student_t_nu": simulator.STUDENT_T_NU,
        "rf_rate":      simulator.RF_RATE * 100,
        "inflation":    simulator.INFLATION * 100,
    }


def _make_serialisable(r: dict) -> dict:
    """Strip numpy arrays from results dict — Jinja2 doesn't need the raw paths."""
    import numpy as np
    skip = {"terminal_values", "paths", "paths_nom", "bands",
            "weights", "mu_vec", "vol_vec", "corr_matrix",
            "contributions", "analytical_terminals"}
    out = {}
    for k, v in r.items():
        if k in skip:
            continue
        if isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        else:
            out[k] = v
    # Keep per-asset info for the weights table
    out["asset_labels"]  = r["labels"]
    out["asset_classes"] = r["asset_classes"]
    out["weights"]       = [float(x) for x in r["weights"]]
    out["mu_vec"]        = [float(x) for x in r["mu_vec"]]
    out["vol_vec"]       = [float(x) for x in r["vol_vec"]]
    return out


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
