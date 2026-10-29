"""IMF PortWatch Data Ingestion Engine.

Responsibilities (Data Engineer):
- Ingest IMF PortWatch global disruptions database (GDACS-verified weather & geopolitical disruptions)
- Ingest global port directory (UN/LOCODE, vessel stats, coordinates)
- Persist structured datasets to data/raw/ and data/reference/
"""

from pathlib import Path
from typing import Dict, List
import requests
import pandas as pd

PORTWATCH_BASE_URL = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"


def fetch_portwatch_disruptions(output_path: Path) -> pd.DataFrame:
    """Fetches all 132 recorded natural disaster and geopolitical disruptions from PortWatch."""
    print("Fetching IMF PortWatch Disruptions Database...")
    url = f"{PORTWATCH_BASE_URL}/portwatch_disruptions_database/FeatureServer/0/query"
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "json",
    }
    r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    data = r.json()

    records = [f["attributes"] for f in data.get("features", [])]
    df = pd.DataFrame(records)

    if "fromdate" in df.columns:
        df["start_datetime"] = pd.to_datetime(df["fromdate"], unit="ms", errors="coerce")
    if "todate" in df.columns:
        df["end_datetime"] = pd.to_datetime(df["todate"], unit="ms", errors="coerce")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    df.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"-> Saved {len(df):,} disruptions to {output_path}")
    return df


def fetch_portwatch_ports(output_path: Path) -> pd.DataFrame:
    """Fetches global ports directory (UN/LOCODE, vessel traffic, coordinates).

    Uses a reliable page size of 500 to avoid server query limits.
    """
    print("Fetching IMF PortWatch Global Ports Reference...")
    url = f"{PORTWATCH_BASE_URL}/PortWatch_ports_database/FeatureServer/0/query"
    out_fields = "portid,portname,country,ISO3,continent,fullname,lat,lon,vessel_count_total,vessel_count_container,LOCODE"

    all_ports = []
    offset = 0
    page_size = 500

    while True:
        params = {
            "where": "1=1",
            "outFields": out_fields,
            "resultRecordCount": page_size,
            "resultOffset": offset,
            "returnGeometry": "false",
            "f": "json",
        }
        r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        features = r.json().get("features", [])
        if not features:
            break
        all_ports.extend([f["attributes"] for f in features])
        print(f"  Fetched {len(features)} ports (offset {offset})...")
        if len(features) < page_size:
            break
        offset += page_size

    df = pd.DataFrame(all_ports)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    df.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"-> Saved {len(df):,} ports to {output_path}")
    return df


def main():
    raw_dir = Path("data/raw")
    ref_dir = Path("data/reference")

    fetch_portwatch_disruptions(raw_dir / "portwatch_disruptions.parquet")
    fetch_portwatch_ports(ref_dir / "portwatch_ports.parquet")
    print("PortWatch ingestion successfully completed!")


if __name__ == "__main__":
    main()
