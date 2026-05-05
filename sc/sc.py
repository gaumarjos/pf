# https://github.com/ScalableCapital/scalable-cli/blob/main/README.md


import subprocess
import json
import os
import requests
import pandas as pd
import xlwings as xw
from dotenv import load_dotenv
from scipy.optimize import brentq

SC_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SC_DIR, "..", ".env"))

ACCOUNT_ID = os.environ["ACCOUNT_ID"]
PORTFOLIO_ID = os.environ["PORTFOLIO_ID"]
SC_BASE_URL = "https://de.scalable.capital"
SC_TOKEN = os.environ.get("SC_TOKEN")  # TODO: find where the CLI stores the auth token
PF_SPREADSHEET = os.environ["PF_SPREADSHEET"]


def _path(filename: str) -> str:
    return os.path.join(SC_DIR, filename)


def get_analytics(output_file: str = "analytics") -> str:
    result = subprocess.run(["sc", "broker", "analytics"], capture_output=True, text=True)
    data = json.loads(result.stdout)
    with open(_path(f"{output_file}.json"), "w") as f:
        json.dump(data, f, indent=2)

    r = data["result"]

    lines = ["# Portfolio Analytics\n"]

    # Allocations
    lines.append("## Allocations")
    for alloc in r["allocations"]:
        for pos in alloc["positions"]:
            lines.append(f"\n**{pos['name'].upper()}** — total valuation: €{pos['valuation']:,.2f}")
            for c in sorted(pos["contributors"], key=lambda x: x["weight"], reverse=True):
                asset = c["underlying_asset"]
                lines.append(f"- {asset['name']} ({asset['isin']}): {c['weight']*100:.1f}%  (qty: {asset['filled_quantity']})")

    # Health checks
    lines.append("\n## Health Checks")
    for hc in r["health_checks"]:
        lines.append(f"- **{hc['type']}**: {hc['state']} — score {hc['health_score']} ({hc['number_of_items_in_portfolio']}/{hc['max_items']} items)")

    # Coverage & invalid securities
    lines.append(f"\n## Portfolio Coverage")
    lines.append(f"{r['portfolio_coverage']*100:.1f}% of the portfolio is covered by analytics.")
    if r["invalid_securities"]:
        lines.append(f"\nThe following {r['invalid_securities_count']} holding(s) are excluded (unsupported security type):")
        for s in r["invalid_securities"]:
            lines.append(f"- {s['name']} ({s['isin']}) — type: {s['security_type']}")

    # Payments
    lines.append("\n## Payments")
    p = r["payments"]
    lines.append(f"- Total distributions (dividends): €{p['total_distributions']:,.2f}")
    lines.append(f"- Total interest: €{p['total_interest']:,.2f}")

    # Scenarios
    lines.append("\n## Stress Test Scenarios")
    lines.append("> Note: the `securities` list in each scenario is of **uncertain meaning** — could be benchmark constituents, scenario outperformers, or hedge suggestions. Needs clarification.\n")
    scenario_labels = {
        "EURO_INFLATION_UP": "Euro inflation up",
        "US_RATES_UP": "US rates up",
        "EURO_RATES_UP": "Euro rates up",
        "EURO_MARKET_DOWN": "Euro market down",
        "WORLD_DOWN": "World down",
    }
    for s in r["scenarios"]:
        label = scenario_labels.get(s["type"], s["type"])
        pp = s["portfolio_performance"] * 100
        bp = s["benchmark_performance"] * 100
        verdict = "outperforms" if pp > bp else "underperforms"
        lines.append(f"### {label}")
        lines.append(f"- Portfolio: {pp:+.2f}%  |  Benchmark: {bp:+.2f}%  → portfolio **{verdict}** benchmark")
        lines.append(f"- Associated securities: {', '.join(s2['name'] for s2 in s['securities'])}")

    # Last updated
    lines.append(f"\n---\n_Last updated: {r['last_updated_utc']}_")

    md = "\n".join(lines)
    with open(_path(f"{output_file}.md"), "w") as f:
        f.write(md)
    return md


def download_document(doc_id: str, label: str, output_dir: str = "documents") -> str:
    if not SC_TOKEN:
        raise RuntimeError("SC_TOKEN not set in .env — auth token still pending")
    os.makedirs(_path(output_dir), exist_ok=True)
    url = f"{SC_BASE_URL}/api/download?id={doc_id}"
    response = requests.get(url, headers={"Authorization": f"Bearer {SC_TOKEN}"})
    response.raise_for_status()
    ext = response.headers.get("Content-Type", "").split("/")[-1].split(";")[0] or "bin"
    filename = f"{label.replace(' ', '_')}_{doc_id}.{ext}"
    filepath = _path(os.path.join(output_dir, filename))
    with open(filepath, "wb") as f:
        f.write(response.content)
    return filepath


def run_sc_command(command: str, output_file: str) -> None:
    result = subprocess.run(command.split(), capture_output=True, text=True)
    data = json.loads(result.stdout)
    with open(_path(output_file), "w") as f:
        json.dump(data, f, indent=2)


def get_holdings(output_file: str = "holdings", dry_run: bool = False) -> pd.DataFrame:
    result = subprocess.run(["sc", "broker", "holdings"], capture_output=True, text=True)
    data = json.loads(result.stdout)
    if data["account_id"] != ACCOUNT_ID or data["result"]["portfolio_id"] != PORTFOLIO_ID:
        raise ValueError(f"Response account/portfolio does not match requested ids")
    df = pd.DataFrame(data["result"]["items"])

    print("")
    print("RETRIEVING UPDATED HOLDINGS")
    csv_path = _path(f"{output_file}.csv")
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        old_vals = old_df.set_index("isin")["valuation"] if "isin" in old_df.columns and "valuation" in old_df.columns else pd.Series(dtype=float)
        new_vals = df.set_index("isin")["valuation"]
        new_names = df.set_index("isin")["name"]
        for isin in new_vals.index.union(old_vals.index):
            name = new_names.get(isin, old_df.set_index("isin")["name"].get(isin, isin) if "name" in old_df.columns else isin)
            if isin not in old_vals.index:
                print(f"INFO: {name} ({isin}) added to holdings")
            elif isin not in new_vals.index:
                print(f"INFO: {name} ({isin}) removed from holdings")
            elif old_vals[isin] != new_vals[isin]:
                print(f"INFO: {name} ({isin}) valuation changed {old_vals[isin]:,.2f} → {new_vals[isin]:,.2f} ({new_vals[isin] - old_vals[isin]:+,.2f})")

    if not dry_run:
        with open(_path(f"{output_file}.json"), "w") as f:
            json.dump(data, f, indent=2)
        df.to_csv(csv_path, index=False)

    print("")
    print("UPDATED HOLDINGS")
    view = df[["name", "isin", "valuation"]].copy()
    total = pd.DataFrame([{"name": "TOTAL", "isin": "", "valuation": view["valuation"].sum()}])
    print(pd.concat([view, total], ignore_index=True).to_string(index=False))
    return df


def update_pf_spreadsheet(holdings_csv: str, xlsx_path: str, start_row: int = 5, end_row: int = 20, dry_run: bool = False) -> None:
    holdings_df = pd.read_csv(holdings_csv)
    holdings_map = dict(zip(holdings_df["isin"], zip(holdings_df["name"], holdings_df["valuation"])))

    wb = xw.Book(xlsx_path)
    ws = wb.sheets.active

    excel_isins = {}
    for row in range(start_row, end_row + 1):
        isin = ws.range(f"H{row}").value
        if isin:
            excel_isins[isin] = row

    for isin, (name, valuation) in holdings_map.items():
        if isin not in excel_isins:
            print(f"WARNING: {name} ({isin}) from holdings not found in spreadsheet (rows {start_row}–{end_row})")
        else:
            old_val = ws.range(f"O{excel_isins[isin]}").value
            if old_val != valuation:
                delta = valuation - old_val
                print(f"INFO: {name} ({isin}) valuation changed {old_val:,.2f} → {valuation:,.2f} ({delta:+,.2f})")
            if not dry_run:
                ws.range(f"O{excel_isins[isin]}").value = valuation

    for isin, row in excel_isins.items():
        if isin not in holdings_map:
            print(f"WARNING: {isin} found in spreadsheet (row {row}) but not in holdings")

    if not dry_run:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ws.range("O2").value = f"Updated from script on {ts}"
        wb.save()
    wb.close()


def get_associated_transactions(holdings_df: pd.DataFrame, output_file: str = "transactions") -> pd.DataFrame:
    result = subprocess.run(["sc", "broker", "transactions"], capture_output=True, text=True)
    data = json.loads(result.stdout)
    if data["account_id"] != ACCOUNT_ID or data["result"]["portfolio_id"] != PORTFOLIO_ID:
        raise ValueError(f"Response account/portfolio does not match requested ids")
    with open(_path(f"{output_file}.json"), "w") as f:
        json.dump(data, f, indent=2)
    isins = set(holdings_df["isin"])
    items = [item for item in data["result"]["items"] if item.get("isin") in isins]
    df = pd.DataFrame(items)
    df.to_csv(_path(f"{output_file}.csv"), index=False)
    return df


def irr(holdings_df: pd.DataFrame, transactions_df: pd.DataFrame, output_file: str = "irr") -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()
    transactions_df = transactions_df.copy()
    transactions_df["date"] = pd.to_datetime(transactions_df["last_event_datetime"], utc=True).dt.tz_convert(None).dt.normalize()

    print("")
    print("IRR")
    results = []
    for _, holding in holdings_df.iterrows():
        isin = holding["isin"]
        txns = transactions_df[transactions_df["isin"] == isin].sort_values("date")

        if txns.empty:
            results.append({"name": holding["name"], "isin": isin, "irr": float("nan")})
            continue

        dates = list(txns["date"]) + [today]
        cashflows = list(txns["amount"]) + [holding["valuation"]]

        d0 = dates[0]
        years = [(d - d0).days / 365.25 for d in dates]

        def npv(r):
            return sum(cf / (1 + r) ** t for cf, t in zip(cashflows, years))

        try:
            rate = brentq(npv, -0.9999, 100.0)
        except ValueError:
            rate = float("nan")

        results.append({"name": holding["name"], "isin": isin, "irr": 100 * rate})

    df = pd.DataFrame(results)
    df.to_csv(_path(f"{output_file}.csv"), index=False)
    return df


if __name__ == "__main__":
    # run_sc_command("sc broker overview", "overview.json")         # useless

    get_updates_from_sc = 1
    write_pf_spreadsheet = 0

    if get_updates_from_sc:
        holdings_df = get_holdings(dry_run=False)
        transactions_df = get_associated_transactions(holdings_df)
        irr_df = irr(holdings_df, transactions_df)
        print(irr_df)

        # Update analytics.md, not really useful, just doing it because it's there
        get_analytics()

    if write_pf_spreadsheet:
        update_pf_spreadsheet("holdings.csv", PF_SPREADSHEET, dry_run=False)
