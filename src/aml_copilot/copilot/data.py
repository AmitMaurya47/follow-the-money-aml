"""Load IBM AML transaction CSVs (HI-Small, LI-Small, or the Kaggle export).

All of these files share the same 11-column layout:
Timestamp, From Bank, Account, To Bank, Account, Amount Received,
Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read and normalize a transaction CSV.

    Account identity is the pair (bank, account) because the same account
    number can exist at two banks. ``src`` and ``dst`` hold that composite id.
    """
    raw = pd.read_csv(path, header=0, dtype=str)
    if raw.shape[1] != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} columns, got {raw.shape[1]} in {path}")
    raw.columns = COLUMNS

    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT)
    df["amount_received"] = df["amount_received"].astype(float)
    df["amount_paid"] = df["amount_paid"].astype(float)
    df["is_laundering"] = df["is_laundering"].astype(int)
    df["src"] = df["from_bank"] + ":" + df["from_account"]
    df["dst"] = df["to_bank"] + ":" + df["to_account"]
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def account_labels(df: pd.DataFrame) -> pd.Series:
    """Account-level label: 1 if the account takes part in any laundering transfer.

    The source data labels transactions, not accounts. Counting both senders and
    receivers is a deliberate, documented choice (mules receive funds too).
    """
    accounts = pd.Index(pd.concat([df["src"], df["dst"]]).unique(), name="account")
    laundering = df.loc[df["is_laundering"] == 1]
    illicit = set(laundering["src"]) | set(laundering["dst"])
    return pd.Series([int(a in illicit) for a in accounts], index=accounts, name="label")
