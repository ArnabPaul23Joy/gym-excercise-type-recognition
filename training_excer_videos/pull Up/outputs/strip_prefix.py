#!/usr/bin/env python3
"""Remove a text prefix (default "pull up_") from every CSV in a directory, in place.

The replacement is applied to every text column (in practice only vid_id), so
"pull up_1" becomes "1". Numeric columns are untouched.

    python strip_prefix.py                 # all *.csv next to this script
    python strip_prefix.py --prefix "x_"   # a different string
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def strip_prefix(csv: Path, prefix: str) -> int:
    table = pd.read_csv(csv)
    text_columns = [c for c in table.columns if not pd.api.types.is_numeric_dtype(table[c])]
    changed = 0
    for column in text_columns:
        mask = table[column].astype(str).str.contains(prefix, regex=False)
        changed += int(mask.sum())
        table[column] = table[column].astype(str).str.replace(prefix, "", regex=False)
    table.to_csv(csv, index=False)
    return changed


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", type=Path, default=here, help="folder containing the CSV files")
    parser.add_argument("--prefix", default="pull up_", help="text to remove")
    args = parser.parse_args()

    files = sorted(args.directory.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files in {args.directory}")
    for csv in files:
        changed = strip_prefix(csv, args.prefix)
        print(f"{csv.name}: replaced in {changed} cells")


if __name__ == "__main__":
    main()
