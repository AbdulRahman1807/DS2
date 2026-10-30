"""IMF PortWatch Data Ingestion Engine.

Responsibilities (Data Engineer):
- Ingest IMF PortWatch global disruptions database (GDACS-verified weather & geopolitical disruptions)
- Ingest global port directory (UN/LOCODE, vessel stats, coordinates)
- Persist structured datasets to data/raw/ and data/reference/
"""

import time
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


def fetch_portwatch_daily_port(output_path: Path, portid: str = "port1237") -> pd.DataFrame:
    """Fetches IMF PortWatch daily port activity (AIS-based port calls and trade volume estimates).

    Default portid 'port1237' = Surabaya (LOCODE ID SUB). The layer rejects paged queries
    without an explicit ordering, so results are ordered by ObjectId.
    """
    print(f"Fetching IMF PortWatch daily activity for {portid}...")
    url = f"{PORTWATCH_BASE_URL}/Daily_Ports_Data/FeatureServer/0/query"
    rows = []
    offset = 0
    page_size = 1000
    while True:
        params = {
            "where": f"portid='{portid}'",
            "outFields": "*",
            "orderByFields": "ObjectId",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "json",
        }
        # The service intermittently answers valid queries with HTTP-200 error payloads; retry.
        for attempt in range(6):
            r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            r.raise_for_status()
            payload = r.json()
            if "error" not in payload:
                break
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"PortWatch query failed: {payload['error']}")
        features = payload.get("features", [])
        rows.extend(f["attributes"] for f in features)
        if len(features) < page_size:
            break
        offset += page_size

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"PortWatch returned no daily records for {portid}")
    df["date"] = pd.to_datetime(df["date"], unit="ms", errors="coerce") if df["date"].dtype != object else pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    df.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"-> Saved {len(df):,} daily records ({df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}) to {output_path}")
    return df


def main():
    raw_dir = Path("data/raw")
    ref_dir = Path("data/reference")

    fetch_portwatch_disruptions(raw_dir / "portwatch_disruptions.parquet")
    fetch_portwatch_ports(ref_dir / "portwatch_ports.parquet")
    fetch_portwatch_daily_port(raw_dir / "portwatch_daily_ports.parquet")
    print("PortWatch ingestion successfully completed!")


if __name__ == "__main__":
    main()
