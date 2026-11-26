"""
Convert an IBM AMLSim sample archive into the HI-Small file layout the pipeline reads.

AMLSim sample output has:
    transactions.csv : sourceNodeId, targetNodeId, value, time
    nodes.csv        : nodeid, isFraud, init_balance, fraudStep

HI-Small (what the pipeline expects) has:
    data/raw/HI-Small_Trans.csv      11 columns, see HEADER below
    data/raw/HI-Small_Patterns.txt   "<typology> <account> <account> ..." per line

Mapping (approximations — the sample lacks these fields, so they are synthesized):
    Timestamp          : 2022/09/01 00:00 + (time step) minutes
    From/To Bank       : node id // 100   (synthetic bank grouping)
    From/To Account    : node id as string
    Amount Received/Paid : value (USD)
    Receiving/Payment Currency : US Dollar
    Payment Format     : ACH
    Is Laundering      : 1 if BOTH endpoints have isFraud == 1, else 0
    Patterns           : one typology line "AMLSIM_FRAUD" listing every isFraud node

Results from a run on this data are sample-only and are NOT comparable to the
README numbers, which were produced on IBM HI-Small.

Usage
-----
    python scripts/sample_to_hismall.py \\
        --sample-dir data/raw/amlsim_sample/20K_fanin200cycle200 \\
        --out-dir    data/raw
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

HEADER = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]
BASE_TIME = datetime(2022, 9, 1, 0, 0)


def convert(sample_dir: Path, out_dir: Path) -> dict[str, int]:
    nodes_path = sample_dir / "nodes.csv"
    trans_path = sample_dir / "transactions.csv"
    if not nodes_path.exists() or not trans_path.exists():
        raise FileNotFoundError(f"nodes.csv / transactions.csv not found in {sample_dir}")

    fraud: dict[str, int] = {}
    with nodes_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            fraud[row["nodeid"]] = int(row["isFraud"])

    out_dir.mkdir(parents=True, exist_ok=True)
    trans_out = out_dir / "HI-Small_Trans.csv"
    n_rows = n_laundering = 0
    with trans_path.open(newline="") as fin, trans_out.open("w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(HEADER)
        for row in csv.DictReader(fin):
            src, dst = row["sourceNodeId"], row["targetNodeId"]
            amount = f"{float(row['value']):.2f}"
            ts = (BASE_TIME + timedelta(minutes=int(row["time"]))).strftime("%Y/%m/%d %H:%M")
            laundering = 1 if fraud.get(src, 0) == 1 and fraud.get(dst, 0) == 1 else 0
            writer.writerow([
                ts, int(src) // 100, src, int(dst) // 100, dst,
                amount, "US Dollar", amount, "US Dollar", "ACH", laundering,
            ])
            n_rows += 1
            n_laundering += laundering

    fraud_nodes = sorted((n for n, f in fraud.items() if f == 1), key=int)
    patterns_out = out_dir / "HI-Small_Patterns.txt"
    with patterns_out.open("w") as fh:
        fh.write("# Synthesized from AMLSim sample — see scripts/sample_to_hismall.py\n")
        if fraud_nodes:
            fh.write("AMLSIM_FRAUD " + " ".join(fraud_nodes) + "\n")

    return {"rows": n_rows, "laundering_rows": n_laundering, "fraud_accounts": len(fraud_nodes)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample-dir", required=True, type=Path)
    p.add_argument("--out-dir", default=Path("data/raw"), type=Path)
    args = p.parse_args()
    stats = convert(args.sample_dir, args.out_dir)
    print(f"Wrote {args.out_dir / 'HI-Small_Trans.csv'} and HI-Small_Patterns.txt")
    print(f"  rows={stats['rows']}  laundering_rows={stats['laundering_rows']}  fraud_accounts={stats['fraud_accounts']}")


if __name__ == "__main__":
    main()
