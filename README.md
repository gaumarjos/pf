# pf — Portfolio Simulation

Monte Carlo projection model for a multi-asset ETF portfolio.
Log-normal GBM returns, mid-year contribution timing, optional Student-t fat tails.

---

## How to run

```bash
# 1. Pick (or create) a scenario file and set SCENARIO_FILE in main.py
# 2. Run
python main.py
```

Outputs: `projection.png`, `correlation_matrix.png`, `volatility_vector.png`.

---

## File reference

### Code

| File | Purpose |
|---|---|
| `main.py` | Entry point. Loads all inputs, runs analytics and MC simulation, saves plots. Contains only ISINs (structural backbone) and computation logic — no numeric parameters. |
| `_gen_params.py` | One-shot generator: builds `vol.csv`, `corr.csv`, `mu.csv` from documented estimates. Re-run to reset the baseline. Contains the rationale comments for every number. |

### Scenario inputs — edit these between runs

| File | What to edit |
|---|---|
| `scenario_*.toml` | **Weights** (must sum to 1.0), initial capital, annual contributions, return distribution (`"normal"` or `"student-t"`). One file per scenario; set `SCENARIO_FILE` in `main.py` to switch. |

### Model parameters — shared across all scenarios

| File | Contents |
|---|---|
| `assets.toml` | Human-readable metadata per ISIN: `name`, `etf`, `index`, `asset_class`. One `[assets.ISIN]` block per asset. |
| `vol.csv` | Annual volatility per ISIN (`isin`, `vol`). Long-run index estimates. |
| `mu.csv` | Expected annual return per ISIN (`isin`, `mu`). Building-block approach (see `_gen_params.py`). |
| `corr.csv` | 16×16 correlation matrix, ISIN-indexed rows and columns. PSD projection applied at runtime — small manual edits cannot break the simulation. |

### Generated outputs (not committed)

| File | Produced by |
|---|---|
| `projection.png` | Fan chart + terminal-value histogram (MC with contributions) |
| `correlation_matrix.png` | Correlation heatmap |
| `volatility_vector.png` | Volatility bar chart by asset |

---

## Adding a new asset

1. Add the ISIN to the `data["isin"]` list in `main.py`.
2. Add a `[assets.ISIN]` block to `assets.toml`.
3. Add entries to `vol.csv` and `mu.csv`, and a row/column to `corr.csv` (or update and re-run `_gen_params.py`).
4. Add the ISIN to every `scenario_*.toml` under `[weights]` (set to `0.0` to keep it inactive).
