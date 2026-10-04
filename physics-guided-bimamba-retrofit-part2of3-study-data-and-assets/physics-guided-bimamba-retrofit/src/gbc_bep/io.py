from __future__ import annotations
from pathlib import Path
import yaml
import pandas as pd


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if name.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.xz", ".csv.zip")):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        df.to_parquet(path, index=False)
    elif name.endswith(".csv.gz"):
        df.to_csv(path, index=False, compression="gzip")
    elif name.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")
