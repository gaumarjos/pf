# https://github.com/pytr-org/pytr

# Before starting, run
# pytr export_transactions
# to get all transactions in account_transactions.csv, then this script

import subprocess
import csv
import re
import tempfile
import os
import pandas as pd
from scipy.optimize import brentq


def export_portfolio_csv(output_path="portfolio.csv"):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # `script` gives pytr a real pseudo-TTY so interactive prompts work normally,
        # while mirroring all output to tmp_path.
        subprocess.run(["script", "-q", tmp_path, "pytr", "portfolio"])
        with open(tmp_path, errors="replace") as f:
            output = f.read()
    finally:
        os.unlink(tmp_path)

    lines = output.splitlines()

    # Find the two header lines (lines starting with "Name")
    header_indices = [i for i, l in enumerate(lines) if l.startswith("Name")]
    if len(header_indices) < 2:
        raise ValueError("Could not find table boundaries in pytr portfolio output")

    table_lines = lines[header_indices[0]: header_indices[1] + 1]

    # Parse fixed-width columns using the header to determine column positions
    header = table_lines[0]
    # Column names and their start positions
    col_names = ["Name", "ISIN", "avgCost", "quantity", "buyCost", "netValue", "price", "diff", "%-diff"]

    # Match the row structure explicitly: name (variable length) then ISIN (12 chars), then numbers
    row_re = re.compile(
        r"(.+?)\s+([A-Z]{2}[A-Z0-9]{10})\s+([\d.]+)\s+\*\s+([\d.]+)\s+=\s+([\d.]+)\s+->\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([-\d.]+%)"
    )

    rows = []
    for line in table_lines[1:-1]:  # skip both header lines
        m = row_re.search(line)
        if m:
            rows.append(list(m.groups()))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(col_names)
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows to {output_path}")


def _xirr(dates, cashflows):
    """Date-aware IRR: solves sum(CF_i / (1+r)^t_i) = 0 where t_i is in years from first date."""
    d0 = dates.iloc[0]
    years = [(d - d0).days / 365.25 for d in dates]

    def npv(r):
        return sum(cf / (1 + r) ** t for cf, t in zip(cashflows, years))

    try:
        return brentq(npv, -0.9999, 100.0)
    except ValueError:
        return float("nan")


def irr(transactions_path="account_transactions.csv", portfolio_path="portfolio.csv"):
    txn = pd.read_csv(transactions_path, sep=";", parse_dates=["Date"])
    txn = txn[txn["Type"].isin(["Buy", "Sell"])].copy()

    portfolio = pd.read_csv(portfolio_path)

    today = pd.Timestamp.today().normalize()
    results = []

    for _, pos in portfolio.iterrows():
        isin = pos["ISIN"]
        pos_txn = txn[txn["ISIN"] == isin].sort_values("Date")

        if pos_txn.empty:
            results.append({"Name": pos["Name"], "ISIN": isin, "IRR": float("nan")})
            continue

        # Buy values are already negative, Sell positive — append current value as terminal inflow
        dates = pd.concat([pos_txn["Date"], pd.Series([today])], ignore_index=True)
        cashflows = pd.concat([pos_txn["Value"], pd.Series([pos["netValue"]])], ignore_index=True)

        results.append({"Name": pos["Name"], "ISIN": isin, "IRR": _xirr(dates, cashflows)})

    return pd.DataFrame(results)


if __name__ == "__main__":
    #export_portfolio_csv()
    print(irr())

