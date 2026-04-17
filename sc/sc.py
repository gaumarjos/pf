import subprocess
import json
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import brentq

SC_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SC_DIR, "..", ".env"))

ACCOUNT_ID = os.environ["ACCOUNT_ID"]
PORTFOLIO_ID = os.environ["PORTFOLIO_ID"]
SC_BASE_URL = "https://de.scalable.capital"
SC_TOKEN = os.environ.get("SC_TOKEN")  # TODO: find where the CLI stores the auth token


def _path(filename: str) -> str:
    return os.path.join(SC_DIR, filename)


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


def get_holdings(output_file: str = "holdings") -> pd.DataFrame:
    result = subprocess.run(["sc", "broker", "holdings"], capture_output=True, text=True)
    data = json.loads(result.stdout)
    if data["account_id"] != ACCOUNT_ID or data["result"]["portfolio_id"] != PORTFOLIO_ID:
        raise ValueError(f"Response account/portfolio does not match requested ids")
    with open(_path(f"{output_file}.json"), "w") as f:
        json.dump(data, f, indent=2)
    df = pd.DataFrame(data["result"]["items"])
    df.to_csv(_path(f"{output_file}.csv"), index=False)
    return df


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
    transactions_df["date"] = pd.to_datetime(transactions_df["last_event_datetime"], utc=True).dt.tz_localize(None)

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
    # run_sc_command("sc broker analytics", "analytics.json")
    #run_sc_command("sc broker transactions", "transactions.json")  # implemented as function
    #run_sc_command("sc broker holdings", "holdings.json")          # implemented as function

    if 0:
        holdings_df = get_holdings()
        transactions_df = get_associated_transactions(holdings_df)
        irr_df = irr(holdings_df, transactions_df)
        print(irr_df)

    run_sc_command("sc broker analytics", "analytics.json")