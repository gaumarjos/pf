import yfinance as yf
import quantstats as qs
import matplotlib.pyplot as plt
import pandas as pd

TICKERS = ["SP5A.MI", "XUSE.MI"]
WEIGHTS = [0.5, 0.5]
START = "2018-01-01"
PLOT_YEARS = 5

prices = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"].dropna()

returns = prices.pct_change().dropna()
port_returns = (returns * WEIGHTS).sum(axis=1)
port_returns.name = "Portfolio"

qs.reports.html(port_returns, output="backtest.html", title="SP5A + XUSE Portfolio")
qs.reports.metrics(port_returns, mode="full")

# --- 5-year performance plot ---
cutoff = pd.Timestamp.today() - pd.DateOffset(years=PLOT_YEARS)
recent_prices = prices[prices.index >= cutoff]
normalized = recent_prices / recent_prices.iloc[0] * 100

fig, ax = plt.subplots(figsize=(12, 5))
for ticker in TICKERS:
    ax.plot(normalized.index, normalized[ticker], linewidth=1.5, label=ticker)

recent_returns = returns[returns.index >= cutoff]
port_recent = (recent_returns * WEIGHTS).sum(axis=1)
port_norm = (1 + port_recent).cumprod() * 100
ax.plot(port_norm.index, port_norm, linewidth=2.5, color="black", linestyle="--", label="Portfolio")

ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")
ax.set_title(f"Performance — last {PLOT_YEARS} years (base 100)")
ax.set_ylabel("Indexed value")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("performance_5y.png", dpi=150)
plt.show()

