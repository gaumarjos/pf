"""
One-shot script: generate vol.csv, corr.csv, mu.csv from hard-coded estimates.
Run once (or whenever you want to update the baseline estimates).
main.py reads these files at runtime — edit them there instead of here.
"""
import numpy as np
import pandas as pd

ISINS = [
    "IE000XZSV718", "IE000MLMNYS0", "IE000R4ZNTN3", "LU1681045370",
    "IE00BJQRDN15", "IE00BP3QZB59", "IE00BP3QZ825", "IE00BP3QZ601",
    "IE00BH04GL39", "LU1650491282", "IE00BDBRDM35",
    "IE00B4ND3602", "IE00BZ1NCS44",
    "LU2951555403", "IE00B44Z5B48", "IE00BMVB5R75",
]

# ── Volatility ────────────────────────────────────────────────────────────────
vol = np.array([
    0.155,  # S&P 500 cap-wtd
    0.170,  # S&P 500 equal-wtd
    0.160,  # MSCI World ex USA
    0.210,  # MSCI EM
    0.145,  # Global multi-factor
    0.165,  # MSCI World Value
    0.155,  # MSCI World Momentum
    0.135,  # MSCI World Quality
    0.060,  # EUR Govt Bonds
    0.070,  # EUR Inflation-Linked
    0.045,  # Global Agg EUR-hedged
    0.155,  # Gold
    0.175,  # Broad Commodities
    0.130,  # Managed Futures
    0.150,  # MSCI ACWI
    0.115,  # LifeStrategy 80/20
])

# ── Expected returns ──────────────────────────────────────────────────────────
mu = np.array([
    0.075,  # S&P 500 cap-wtd
    0.085,  # S&P 500 equal-wtd
    0.085,  # MSCI World ex USA
    0.100,  # MSCI EM
    0.085,  # Global multi-factor
    0.090,  # MSCI World Value
    0.090,  # MSCI World Momentum
    0.080,  # MSCI World Quality
    0.027,  # EUR Govt Bonds
    0.025,  # EUR Inflation-Linked
    0.032,  # Global Agg EUR hdg
    0.040,  # Gold
    0.045,  # Broad Commodities
    0.055,  # Managed Futures
    0.080,  # MSCI ACWI
    0.070,  # LifeStrategy 80/20
])

# ── Correlation matrix ────────────────────────────────────────────────────────
n   = len(ISINS)
idx = {isin: i for i, isin in enumerate(ISINS)}
C   = np.eye(n)

def sc(a, b, v):
    C[idx[a], idx[b]] = v
    C[idx[b], idx[a]] = v

EQ   = ISINS[:8]
GOLD = "IE00B4ND3602"
COMM = "IE00BZ1NCS44"
MF   = "LU2951555403"
ACWI = "IE00B44Z5B48"
LS   = "IE00BMVB5R75"

# Equity – equity
sc("IE000XZSV718", "IE000MLMNYS0", 0.97)
sc("IE000XZSV718", "IE000R4ZNTN3", 0.72)
sc("IE000XZSV718", "LU1681045370", 0.62)
sc("IE000XZSV718", "IE00BJQRDN15", 0.93)
sc("IE000XZSV718", "IE00BP3QZB59", 0.82)
sc("IE000XZSV718", "IE00BP3QZ825", 0.83)
sc("IE000XZSV718", "IE00BP3QZ601", 0.88)
sc("IE000MLMNYS0", "IE000R4ZNTN3", 0.73)
sc("IE000MLMNYS0", "LU1681045370", 0.63)
sc("IE000MLMNYS0", "IE00BJQRDN15", 0.91)
sc("IE000MLMNYS0", "IE00BP3QZB59", 0.84)
sc("IE000MLMNYS0", "IE00BP3QZ825", 0.80)
sc("IE000MLMNYS0", "IE00BP3QZ601", 0.86)
sc("IE000R4ZNTN3", "LU1681045370", 0.72)
sc("IE000R4ZNTN3", "IE00BJQRDN15", 0.90)
sc("IE000R4ZNTN3", "IE00BP3QZB59", 0.83)
sc("IE000R4ZNTN3", "IE00BP3QZ825", 0.78)
sc("IE000R4ZNTN3", "IE00BP3QZ601", 0.84)
sc("LU1681045370", "IE00BJQRDN15", 0.78)
sc("LU1681045370", "IE00BP3QZB59", 0.72)
sc("LU1681045370", "IE00BP3QZ825", 0.68)
sc("LU1681045370", "IE00BP3QZ601", 0.73)
sc("IE00BJQRDN15", "IE00BP3QZB59", 0.88)
sc("IE00BJQRDN15", "IE00BP3QZ825", 0.85)
sc("IE00BJQRDN15", "IE00BP3QZ601", 0.90)
sc("IE00BP3QZB59", "IE00BP3QZ825", 0.70)
sc("IE00BP3QZB59", "IE00BP3QZ601", 0.78)
sc("IE00BP3QZ825", "IE00BP3QZ601", 0.78)

# Equity – bonds
for eq in EQ:
    sc(eq, "IE00BH04GL39", -0.15)
    sc(eq, "LU1650491282", -0.10)
    sc(eq, "IE00BDBRDM35", -0.15)

# Bond – bond
sc("IE00BH04GL39", "LU1650491282", 0.82)
sc("IE00BH04GL39", "IE00BDBRDM35", 0.80)
sc("LU1650491282", "IE00BDBRDM35", 0.68)

# Gold
for eq in EQ:
    sc(eq, GOLD, 0.05)
sc("IE00BH04GL39", GOLD, 0.12)
sc("LU1650491282", GOLD, 0.22)
sc("IE00BDBRDM35", GOLD, 0.10)

# Commodities
for eq in EQ:
    sc(eq, COMM, 0.18)
sc("IE00BH04GL39", COMM, -0.02)
sc("LU1650491282", COMM,  0.15)
sc("IE00BDBRDM35", COMM,  0.00)
sc(GOLD, COMM, 0.25)

# Managed Futures
for eq in EQ:
    sc(eq, MF, 0.00)
sc("IE00BH04GL39", MF,  0.15)
sc("LU1650491282", MF,  0.10)
sc("IE00BDBRDM35", MF,  0.12)
sc(GOLD, MF,  0.15)
sc(COMM, MF,  0.20)

# MSCI ACWI
for eq in EQ:
    sc(eq, ACWI, 0.93)
sc("IE000XZSV718", ACWI, 0.97)
sc("IE000R4ZNTN3", ACWI, 0.92)
sc("LU1681045370", ACWI, 0.82)
sc("IE00BH04GL39", ACWI, -0.14)
sc("LU1650491282", ACWI, -0.10)
sc("IE00BDBRDM35", ACWI, -0.14)
sc(GOLD, ACWI,  0.05)
sc(COMM, ACWI,  0.18)
sc(MF,   ACWI,  0.00)

# LifeStrategy 80/20
for eq in EQ:
    sc(eq, LS, 0.90)
sc("IE000XZSV718", LS, 0.93)
sc("IE000R4ZNTN3", LS, 0.89)
sc("LU1681045370", LS, 0.78)
sc("IE00BH04GL39", LS, -0.05)
sc("LU1650491282", LS, -0.02)
sc("IE00BDBRDM35", LS,  0.02)
sc(GOLD, LS,  0.05)
sc(COMM, LS,  0.16)
sc(MF,   LS,  0.00)
sc(ACWI, LS,  0.98)

# ── Write CSVs ────────────────────────────────────────────────────────────────
pd.Series(vol, index=ISINS, name="vol").rename_axis("isin").to_csv("vol.csv")
pd.Series(mu,  index=ISINS, name="mu" ).rename_axis("isin").to_csv("mu.csv")
pd.DataFrame(C, index=ISINS, columns=ISINS).rename_axis("isin").to_csv("corr.csv")

print("Written: vol.csv, mu.csv, corr.csv")
print(f"  vol shape : {vol.shape}")
print(f"  mu  shape : {mu.shape}")
print(f"  corr shape: {C.shape}")
