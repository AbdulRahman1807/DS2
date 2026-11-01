#!/usr/bin/env python3
"""
scripts/process_supplementary_datasets.py

Processes and integrates supplementary port datasets:
  1. ADB AIS Port Traffic Database (AISData.csv: 19 ports, weekly 2019-2022)
  2. Indian DGQI Port Dwell Dataset (DGQI.csv: 13 ports + 3 regions, monthly 2020)

Generates:
  - Cleaned & enriched CSV and Parquet files in DL/data/staging/supplementary_acquisition/processed/
  - Multi-source port bridge cross-walk table
  - Target metric definition and semantic comparability matrix
  - LA/LB and NY/NJ empirical overlap analysis table
"""

import sys
import os
import json
import calendar
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING_RAW_DIR = PROJECT_ROOT / "data" / "staging" / "supplementary_acquisition" / "raw"
STAGING_PROCESSED_DIR = PROJECT_ROOT / "data" / "staging" / "supplementary_acquisition" / "processed"
CANONICAL_REGISTRY_PATH = PROJECT_ROOT / "reports" / "port_identity_resolution" / "v1_3_final_semantics" / "canonical_port_registry.csv"
ZENODO_DWELL_PATH = PROJECT_ROOT / "data" / "raw" / "ais_dwell" / "monthly_dwell_2015_2025.csv"

# Ensure target directories exist
STAGING_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Part 1: Process ADB AIS Port Traffic Database
# ---------------------------------------------------------------------------

def process_adb_dataset():
    print(">>> Processing ADB AIS Port Traffic Database...")
    raw_path = STAGING_RAW_DIR / "AISData.csv"
    assert raw_path.exists(), f"Raw ADB file missing: {raw_path}"

    # Read with utf-8-sig to handle UTF-8 BOM
    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    print(f"    Raw shape: {df.shape}")

    # Check and remove trailing all-NaN row
    if df.iloc[-1].isnull().sum() >= 65:
        print("    Dropping 1 trailing empty row...")
        df = df.iloc[:-1].copy()

    assert len(df) == 3990, f"Expected exactly 3990 clean rows, got {len(df)}"

    # Normalize column names: strip whitespace, handle specific typos
    col_map = {}
    for col in df.columns:
        clean_col = col.strip()
        # Handle known typos/inconsistencies
        if clean_col == "Port":
            clean_col = "port_name"
        elif clean_col == "Year":
            clean_col = "year"
        elif clean_col == "Week":
            clean_col = "week"
        elif clean_col == "Month":
            clean_col = "month"
        elif clean_col == "country":
            clean_col = "country"
        elif clean_col == "avgservice_time_feed":
            clean_col = "avgservice_time_feeder"
        elif clean_col == "avgwaiting_time_feed":
            clean_col = "avgwaiting_time_feeder"
        elif clean_col == "share_neopan":
            clean_col = "share_neopanamax"
        col_map[col] = clean_col

    df = df.rename(columns=col_map)

    # Cast temporal integers
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)
    df["month"] = df["month"].astype(int)

    # Sort deterministically
    df = df.sort_values(by=["port_name", "year", "week"]).reset_index(drop=True)

    # Verify 19 ports with exactly 210 weeks each
    ports = df["port_name"].unique().tolist()
    assert len(ports) == 19, f"Expected 19 ports, got {len(ports)}"
    for p in ports:
        port_rows = len(df[df["port_name"] == p])
        assert port_rows == 210, f"Port {p} has {port_rows} rows, expected 210"

    # Compute continuous weekly temporal boundaries (strictly partitioning 2019-2022)
    # The 210 weekly intervals exactly cover 1,461 calendar days from 2019-01-01 to 2022-12-31.
    time_step_indices = []
    iso_year_weeks = []
    week_start_dates = []
    week_end_dates = []
    week_midpoint_dates = []

    # Precompute date schedule for the 210 chronological intervals
    first_port_df = df[df["port_name"] == ports[0]].copy()
    schedule = []
    for idx, r in first_port_df.iterrows():
        y = int(r["year"])
        w = int(r["week"])
        start_d = date(y, 1, 1) + timedelta(days=(w - 1) * 7)
        schedule.append({"year": y, "week": w, "start_date": start_d})

    for i in range(len(schedule)):
        start_d = schedule[i]["start_date"]
        if i < len(schedule) - 1:
            end_d = schedule[i + 1]["start_date"] - timedelta(days=1)
        else:
            end_d = date(2022, 12, 31)
        mid_d = start_d + timedelta(days=(end_d - start_d).days // 2)
        schedule[i]["end_date"] = end_d
        schedule[i]["midpoint_date"] = mid_d
        schedule[i]["iso_year_week"] = f"{schedule[i]['year']}-W{schedule[i]['week']:02d}"

    schedule_lookup = {(s["year"], s["week"]): s for s in schedule}

    # Total days partitioned verification
    total_days_partitioned = sum((s["end_date"] - s["start_date"]).days + 1 for s in schedule)
    assert total_days_partitioned == 1461, f"Expected 1461 days, got {total_days_partitioned}"

    for idx, r in df.iterrows():
        s = schedule_lookup[(int(r["year"]), int(r["week"]))]
        time_step_indices.append((int(r["year"]) - 2019) * 52 + (int(r["week"]) - 1)) # chronological sequence
        iso_year_weeks.append(s["iso_year_week"])
        week_start_dates.append(str(s["start_date"]))
        week_end_dates.append(str(s["end_date"]))
        week_midpoint_dates.append(str(s["midpoint_date"]))

    df["time_step_index"] = df.groupby("port_name").cumcount()
    df["iso_year_week"] = iso_year_weeks
    df["week_start_date"] = week_start_dates
    df["week_end_date"] = week_end_dates
    df["week_midpoint_date"] = week_midpoint_dates

    # Add derived vessel operational measures
    # Total vessel port turnaround time = waiting time + service time
    df["avg_turnaround_time_hours"] = df["avgwaiting_time"] + df["avgservice_time"]
    df["avg_turnaround_time_days"] = df["avg_turnaround_time_hours"] / 24.0

    # Feeder turnaround time
    df["avg_turnaround_time_feeder_hours"] = df["avgwaiting_time_feeder"] + df["avgservice_time_feeder"]
    df["avg_turnaround_time_feeder_days"] = df["avg_turnaround_time_feeder_hours"] / 24.0

    # Intermediate turnaround time
    df["avg_turnaround_time_intermediate_hours"] = df["avgwaiting_time_intermediate"] + df["avgservice_time_intermediate"]
    df["avg_turnaround_time_intermediate_days"] = df["avg_turnaround_time_intermediate_hours"] / 24.0

    # Neopanamax turnaround time
    df["avg_turnaround_time_neopanamax_hours"] = df["avgwaiting_time_neopanamax"] + df["avgservice_time_neopanamax"]
    df["avg_turnaround_time_neopanamax_days"] = df["avg_turnaround_time_neopanamax_hours"] / 24.0

    # Add port identifiers bridge columns
    adb_bridge_info = {
        "Antwerp": {"portwatch_id": "port57", "canonical_name": "Antwerp", "unlocode": "BEANR", "iso3": "BEL", "zenodo_name": None, "join_status": "DIRECT_MATCH"},
        "Busan": {"portwatch_id": "port1065", "canonical_name": "Busan", "unlocode": "KRPUS", "iso3": "KOR", "zenodo_name": None, "join_status": "DIRECT_MATCH"},
        "Dubai": {"portwatch_id": "port744", "canonical_name": "Jebel Ali", "unlocode": "AEJEA", "iso3": "ARE", "zenodo_name": None, "join_status": "COMPLEX_PROXY_JEBEL_ALI"},
        "Guangzhou": {"portwatch_id": "port425", "canonical_name": "Guangzhou (Nansha)", "unlocode": "CNNSA", "iso3": "CHN", "zenodo_name": None, "join_status": "COMPLEX_PROXY_NANSHA"},
        "Hamburg": {"portwatch_id": "port446", "canonical_name": "Hamburg", "unlocode": "DEHAM", "iso3": "DEU", "zenodo_name": None, "join_status": "DIRECT_MATCH"},
        "Hong Kong": {"portwatch_id": "port474", "canonical_name": "Hong Kong", "unlocode": "HKHKG", "iso3": "HKG", "zenodo_name": None, "join_status": "DIRECT_MATCH"},
        "Kaohsiung": {"portwatch_id": "port541", "canonical_name": "Kaohsiung", "unlocode": "TWKHH", "iso3": "TWN", "zenodo_name": None, "join_status": "DIRECT_MATCH"},
        "LALB": {"portwatch_id": "port664", "canonical_name": "Los Angeles-Long Beach", "unlocode": "USLAX;USLGB", "iso3": "USA", "zenodo_name": "LA_Long_Beach", "join_status": "COMPLEX_MATCH_SAN_PEDRO_BAY"},
        "NYNJ": {"portwatch_id": "port815", "canonical_name": "New York-New Jersey", "unlocode": "USNYC;USNWK", "iso3": "USA", "zenodo_name": "NY_NJ", "join_status": "COMPLEX_MATCH_PANYNJ"},
        "Ningbo": {"portwatch_id": "port824", "canonical_name": "Ningbo", "unlocode": "CNNBO", "iso3": "CHN", "zenodo_name": None, "join_status": "DIRECT_MATCH"},
        "Port Klang": {"portwatch_id": "port960", "canonical_name": "Port Klang", "unlocode": "MYPKG", "iso3": "MYS", "zenodo_name": None, "join_status": "DIRECT_MATCH"},
        "Qingdao": {"portwatch_id": "port1069", "canonical_name": "Qingdao Port", "unlocode": "CNTAO", "iso3": "CHN", "zenodo_name": None, "join_status": "DIRECT_MATCH"},
        "Rotterdam": {"portwatch_id": "port1114", "canonical_name": "Rotterdam", "unlocode": "NLRTM", "iso3": "NLD", "zenodo_name": None, "join_status": "DIRECT_MATCH"},
        "Shanghai": {"portwatch_id": "port1188", "canonical_name": "Shanghai", "unlocode": "CNSGH", "iso3": "CHN", "zenodo_name": None, "join_status": "COMPLEX_MATCH_SHANGHAI_PORT_CLUSTER"},
        "Shenzhen": {"portwatch_id": "port1189", "canonical_name": "Shekou (Shenzhen)", "unlocode": "CNSZX", "iso3": "CHN", "zenodo_name": None, "join_status": "COMPLEX_MATCH_SHENZHEN_PORT_CLUSTER"},
        "Singapore": {"portwatch_id": "port1201", "canonical_name": "Singapore", "unlocode": "SGSIN", "iso3": "SGP", "zenodo_name": None, "join_status": "DIRECT_MATCH"},
        "Tanjung Pelepas": {"portwatch_id": "port1269", "canonical_name": "Tanjung Pelepas", "unlocode": "MYTPP", "iso3": "MYS", "zenodo_name": None, "join_status": "DIRECT_MATCH"},
        "Tianjin": {"portwatch_id": "port1297", "canonical_name": "Tianjin Xin Gang", "unlocode": "CNTSN", "iso3": "CHN", "zenodo_name": None, "join_status": "DIRECT_MATCH"},
        "Xiamen": {"portwatch_id": "port1404", "canonical_name": "Xiamen", "unlocode": "CNXAM", "iso3": "CHN", "zenodo_name": None, "join_status": "DIRECT_MATCH"}
    }

    df["portwatch_id"] = df["port_name"].map(lambda p: adb_bridge_info[p]["portwatch_id"])
    df["canonical_port_name"] = df["port_name"].map(lambda p: adb_bridge_info[p]["canonical_name"])
    df["unlocode"] = df["port_name"].map(lambda p: adb_bridge_info[p]["unlocode"])
    df["country_iso3"] = df["port_name"].map(lambda p: adb_bridge_info[p]["iso3"])
    df["zenodo_port_name"] = df["port_name"].map(lambda p: adb_bridge_info[p]["zenodo_name"])
    df["portwatch_join_status"] = df["port_name"].map(lambda p: adb_bridge_info[p]["join_status"])

    # Reorder columns logically: Identifiers -> Temporal -> Operational Totals -> Vessel Classes -> Moving Averages
    id_cols = [
        "port_name", "canonical_port_name", "portwatch_id", "unlocode", "country", "country_iso3",
        "zenodo_port_name", "portwatch_join_status"
    ]
    time_cols = [
        "year", "week", "month", "time_step_index", "iso_year_week",
        "week_start_date", "week_end_date", "week_midpoint_date"
    ]
    core_ops = [
        "count_entrance", "numofcall", "avgservice_time", "avgwaiting_time",
        "avg_turnaround_time_hours", "avg_turnaround_time_days"
    ]
    feeder_cols = [
        "count_entrance_feeder", "numofcall_feeder", "avgservice_time_feeder", "avgwaiting_time_feeder",
        "avg_turnaround_time_feeder_hours", "avg_turnaround_time_feeder_days"
    ]
    intermediate_cols = [
        "count_entrance_intermediate", "numofcall_intermediate",
        "avgservice_time_intermediate", "avgwaiting_time_intermediate",
        "avg_turnaround_time_intermediate_hours", "avg_turnaround_time_intermediate_days"
    ]
    neopanamax_cols = [
        "count_entrance_neopanamax", "numofcall_neopanamax",
        "avgservice_time_neopanamax", "avgwaiting_time_neopanamax",
        "avg_turnaround_time_neopanamax_hours", "avg_turnaround_time_neopanamax_days",
        "share_neopanamax"
    ]
    ma_cols = [c for c in df.columns if c.startswith("ma")]

    ordered_cols = id_cols + time_cols + core_ops + feeder_cols + intermediate_cols + neopanamax_cols + ma_cols
    assert len(ordered_cols) == len(df.columns), f"Column count mismatch: {len(ordered_cols)} vs {len(df.columns)}"
    df = df[ordered_cols]

    # Save to CSV and Parquet
    csv_out = STAGING_PROCESSED_DIR / "adb_port_traffic_weekly.csv"
    parquet_out = STAGING_PROCESSED_DIR / "adb_port_traffic_weekly.parquet"
    df.to_csv(csv_out, index=False)
    df.to_parquet(parquet_out, index=False)
    print(f"    Saved: {csv_out} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"    Saved: {parquet_out}")

    return df

# ---------------------------------------------------------------------------
# Part 2: Process Indian DGQI Port Dwell Dataset
# ---------------------------------------------------------------------------

def process_dgqi_dataset():
    print(">>> Processing Indian DGQI Port Dwell Dataset...")
    raw_path = STAGING_RAW_DIR / "DGQI.csv"
    assert raw_path.exists(), f"Raw DGQI file missing: {raw_path}"

    df = pd.read_csv(raw_path)
    print(f"    Raw shape: {df.shape}")
    assert len(df) == 363, f"Expected 363 rows, got {len(df)}"

    # Clean column names
    df = df.rename(columns={
        "portName": "port_name_raw",
        "month": "month_name",
        "year": "year",
        "category": "trade_direction",
        "dwellTime": "container_dwell_hours"
    })

    # Standardize trade direction (Import / Export)
    df["trade_direction"] = df["trade_direction"].str.strip().str.capitalize()
    df["year"] = df["year"].astype(int)

    # Flag regional averages vs individual ports
    regional_patterns = ["SOUTHERN REGION AVG.", "EASTERN REGION AVG.", "WESTERN REGION AVG."]
    df["is_regional_average"] = df["port_name_raw"].isin(regional_patterns)
    df["entity_type"] = np.where(df["is_regional_average"], "regional_average", "individual_port")

    # Map month names to numbers and date ranges (Year 2020 is a leap year!)
    month_name_to_num = {
        "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4, "MAY": 5, "JUNE": 6,
        "JULY": 7, "AUGUST": 8, "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12
    }
    df["month_num"] = df["month_name"].str.strip().str.upper().map(month_name_to_num)
    assert df["month_num"].isnull().sum() == 0, "Failed to parse month names"

    df["year_month"] = df["year"].astype(str) + "-" + df["month_num"].astype(str).str.zfill(2)
    df["month_start_date"] = df["year_month"] + "-01"

    # End dates for 2020 (leap year: Feb has 29 days)
    month_days_2020 = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    df["month_end_date"] = df.apply(lambda r: f"2020-{r['month_num']:02d}-{month_days_2020[r['month_num']]:02d}", axis=1)

    # Add dwell time in days
    df["container_dwell_days"] = df["container_dwell_hours"] / 24.0

    # Clean port name and region classification
    def get_clean_name_and_region(raw_name):
        raw = raw_name.strip()
        if raw == "SOUTHERN REGION AVG.":
            return "Southern Region Average", "Southern", "REGIONAL_AGGREGATE"
        elif raw == "EASTERN REGION AVG.":
            return "Eastern Region Average", "Eastern", "REGIONAL_AGGREGATE"
        elif raw == "WESTERN REGION AVG.":
            return "Western Region Average", "Western", "REGIONAL_AGGREGATE"
        elif raw == "APSEZ":
            return "Mundra (APSEZ)", "Western", "COMPLEX_PROXY_MUNDRA"
        elif raw == "KANDLA":
            return "Kandla (Deendayal)", "Western", "DIRECT_MATCH"
        elif raw == "HAZIRA":
            return "Hazira", "Western", "DIRECT_MATCH"
        elif raw == "NMPT":
            return "New Mangalore", "Western", "DIRECT_MATCH"
        elif raw == "JNPT":
            return "Jawaharlal Nehru Port (Nhava Sheva)", "Western", "COMPLEX_PROXY_MUMBAI"
        elif raw == "COCHIN":
            return "Cochin (Kochi)", "Southern", "DIRECT_MATCH"
        elif raw == "TUTICORIN":
            return "Tuticorin (V.O.C.)", "Southern", "DIRECT_MATCH"
        elif raw == "CHENNAI":
            return "Chennai (Madras)", "Southern", "DIRECT_MATCH"
        elif raw == "KATTUPALLI":
            return "Kattupalli", "Southern", "DIRECT_MATCH"
        elif raw == "KRISHNAPATNAM":
            return "Krishnapatnam", "Southern", "DIRECT_MATCH"
        elif raw == "VIZAG":
            return "Visakhapatnam", "Eastern", "DIRECT_MATCH"
        elif raw == "HALDIA":
            return "Haldia", "Eastern", "DIRECT_MATCH"
        elif raw == "KOLKATA":
            return "Kolkata (Syama Prasad)", "Eastern", "DIRECT_MATCH"
        else:
            return raw, "Unknown", "UNKNOWN"

    names_regions_status = [get_clean_name_and_region(n) for n in df["port_name_raw"]]
    df["port_name_clean"] = [x[0] for x in names_regions_status]
    df["region"] = [x[1] for x in names_regions_status]
    df["portwatch_join_status"] = [x[2] for x in names_regions_status]

    # PortWatch and UN/LOCODE mapping
    dgqi_bridge_info = {
        "SOUTHERN REGION AVG.": {"pw_id": None, "pw_name": None, "unlocode": None},
        "EASTERN REGION AVG.": {"pw_id": None, "pw_name": None, "unlocode": None},
        "WESTERN REGION AVG.": {"pw_id": None, "pw_name": None, "unlocode": None},
        "APSEZ": {"pw_id": "port777", "pw_name": "Mundra", "unlocode": "INMUN"},
        "KANDLA": {"pw_id": "port540", "pw_name": "Kandla", "unlocode": "INIXY"},
        "HAZIRA": {"pw_id": "port2039", "pw_name": "Hazira", "unlocode": "INHZA"},
        "NMPT": {"pw_id": "port811", "pw_name": "New Mangalore", "unlocode": "INNML"},
        "JNPT": {"pw_id": "port776", "pw_name": "Mumbai", "unlocode": "INNSA"},
        "COCHIN": {"pw_id": "port583", "pw_name": "Cochin (Kochi)", "unlocode": "INCOK"},
        "TUTICORIN": {"pw_id": "port1331", "pw_name": "Tuticorin", "unlocode": "INTUT"},
        "CHENNAI": {"pw_id": "port235", "pw_name": "Chennai (Madras)", "unlocode": "INMAA"},
        "KATTUPALLI": {"pw_id": "port2038", "pw_name": "Kattupalli", "unlocode": "INKAT"},
        "KRISHNAPATNAM": {"pw_id": "port599", "pw_name": "Krishnapatnam Port", "unlocode": "INKRI"},
        "VIZAG": {"pw_id": "port1367", "pw_name": "Visakhapatnam", "unlocode": "INVTZ"},
        "HALDIA": {"pw_id": "port442", "pw_name": "Haldia", "unlocode": "INHAL"},
        "KOLKATA": {"pw_id": "port207", "pw_name": "Kolkata (Syama Prasad Mookerje Port)", "unlocode": "INCCU"},
    }

    df["portwatch_id"] = df["port_name_raw"].map(lambda p: dgqi_bridge_info[p]["pw_id"])
    df["portwatch_name"] = df["port_name_raw"].map(lambda p: dgqi_bridge_info[p]["pw_name"])
    df["unlocode"] = df["port_name_raw"].map(lambda p: dgqi_bridge_info[p]["unlocode"])
    df["country"] = "India"
    df["country_iso3"] = "IND"

    # Reorder columns
    ordered_cols = [
        "port_name_raw", "port_name_clean", "entity_type", "is_regional_average", "region",
        "portwatch_id", "portwatch_name", "unlocode", "country", "country_iso3",
        "portwatch_join_status",
        "year", "month_num", "month_name", "year_month", "month_start_date", "month_end_date",
        "trade_direction", "container_dwell_hours", "container_dwell_days"
    ]
    df = df[ordered_cols].sort_values(by=["entity_type", "region", "port_name_raw", "month_num", "trade_direction"]).reset_index(drop=True)

    csv_out = STAGING_PROCESSED_DIR / "indian_port_dwell_monthly.csv"
    parquet_out = STAGING_PROCESSED_DIR / "indian_port_dwell_monthly.parquet"
    df.to_csv(csv_out, index=False)
    df.to_parquet(parquet_out, index=False)
    print(f"    Saved: {csv_out} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"    Saved: {parquet_out}")

    return df

# ---------------------------------------------------------------------------
# Part 3: Build Unified Port Cross-Walk Bridge Table
# ---------------------------------------------------------------------------

def build_port_bridge_table(df_adb, df_dgqi):
    print(">>> Building Unified Multi-Source Port Bridge Cross-Walk...")

    bridge_rows = []

    # 1. ADB Ports (19 ports)
    adb_meta = {
        "Antwerp": ("port57", "Antwerp", "BEANR", "Belgium", "BEL", None, "DIRECT_MATCH", "Direct identity match with PortWatch Antwerp."),
        "Busan": ("port1065", "Busan", "KRPUS", "South Korea", "KOR", None, "DIRECT_MATCH", "Direct identity match with PortWatch Busan."),
        "Dubai": ("port744", "Jebel Ali", "AEJEA", "United Arab Emirates", "ARE", None, "COMPLEX_PROXY_JEBEL_ALI", "ADB records Dubai container traffic; container operations shifted to Jebel Ali (port744 / AEJEA) in 2018."),
        "Guangzhou": ("port425", "Guangzhou (Nansha)", "CNNSA", "China", "CHN", None, "COMPLEX_PROXY_NANSHA", "Container traffic concentrates at Nansha deepwater terminal (port425); PortWatch also has Huangpu (port2401)."),
        "Hamburg": ("port446", "Hamburg", "DEHAM", "Germany", "DEU", None, "DIRECT_MATCH", "Direct identity match with PortWatch Hamburg."),
        "Hong Kong": ("port474", "Hong Kong", "HKHKG", "Hong Kong SAR", "HKG", None, "DIRECT_MATCH", "Direct identity match with PortWatch Hong Kong."),
        "Kaohsiung": ("port541", "Kaohsiung", "TWKHH", "Taiwan", "TWN", None, "DIRECT_MATCH", "Direct identity match with PortWatch Kaohsiung."),
        "LALB": ("port664", "Los Angeles-Long Beach", "USLAX;USLGB", "United States", "USA", "LA_Long_Beach", "COMPLEX_MATCH_SAN_PEDRO_BAY", "San Pedro Bay twin-port complex. Matches PortWatch port664 and Zenodo LA_Long_Beach exactly."),
        "NYNJ": ("port815", "New York-New Jersey", "USNYC;USNWK", "United States", "USA", "NY_NJ", "COMPLEX_MATCH_PANYNJ", "Port Authority of NY & NJ multi-terminal complex. Matches PortWatch port815 and Zenodo NY_NJ exactly."),
        "Ningbo": ("port824", "Ningbo", "CNNBO", "China", "CHN", None, "DIRECT_MATCH", "Direct identity match with PortWatch Ningbo-Zhoushan complex."),
        "Port Klang": ("port960", "Port Klang", "MYPKG", "Malaysia", "MYS", None, "DIRECT_MATCH", "Direct identity match with PortWatch Port Klang (Northport/Westports)."),
        "Qingdao": ("port1069", "Qingdao Port", "CNTAO", "China", "CHN", None, "DIRECT_MATCH", "Direct identity match with PortWatch Qingdao Port."),
        "Rotterdam": ("port1114", "Rotterdam", "NLRTM", "Netherlands", "NLD", None, "DIRECT_MATCH", "Direct identity match with PortWatch Rotterdam."),
        "Shanghai": ("port1188", "Shanghai", "CNSGH", "China", "CHN", None, "COMPLEX_MATCH_SHANGHAI_PORT_CLUSTER", "Includes Yangshan Deepwater (port2027) and Waigaoqiao river terminals. PortWatch port1188 represents Shanghai."),
        "Shenzhen": ("port1189", "Shekou (Shenzhen)", "CNSZX", "China", "CHN", None, "COMPLEX_MATCH_SHENZHEN_PORT_CLUSTER", "Shenzhen multi-terminal complex (Shekou port1189, Yantian, Chiwan). UN/LOCODE CNSZX encompasses all terminals."),
        "Singapore": ("port1201", "Singapore", "SGSIN", "Singapore", "SGP", None, "DIRECT_MATCH", "Direct identity match with PortWatch Singapore mega-hub."),
        "Tanjung Pelepas": ("port1269", "Tanjung Pelepas", "MYTPP", "Malaysia", "MYS", None, "DIRECT_MATCH", "Direct identity match with PortWatch Tanjung Pelepas (PTP)."),
        "Tianjin": ("port1297", "Tianjin Xin Gang", "CNTSN", "China", "CHN", None, "DIRECT_MATCH", "Direct identity match with PortWatch Tianjin Xin Gang."),
        "Xiamen": ("port1404", "Xiamen", "CNXAM", "China", "CHN", None, "DIRECT_MATCH", "Direct identity match with PortWatch Xiamen.")
    }

    for p in df_adb["port_name"].unique():
        meta = adb_meta[p]
        bridge_rows.append({
            "source_dataset": "ADB_AIS_PORT_TRAFFIC",
            "source_entity_name": p,
            "entity_type": "individual_port" if meta[6] == "DIRECT_MATCH" else "port_complex",
            "country_name": meta[3],
            "country_iso3": meta[4],
            "portwatch_id": meta[0],
            "portwatch_name": meta[1],
            "unlocode": meta[2],
            "zenodo_dwell_port_name": meta[5],
            "join_confidence": "EXACT" if meta[6] == "DIRECT_MATCH" else "HIGH_PROXY",
            "mapping_status": meta[6],
            "mapping_notes": meta[7]
        })

    # 2. Indian DGQI Ports (13 ports + 3 regional averages)
    dgqi_meta = {
        "SOUTHERN REGION AVG.": (None, None, None, "India", "IND", None, "REGIONAL_AGGREGATE", "Regional benchmark aggregate for Southern Indian major container ports."),
        "EASTERN REGION AVG.": (None, None, None, "India", "IND", None, "REGIONAL_AGGREGATE", "Regional benchmark aggregate for Eastern Indian major container ports."),
        "WESTERN REGION AVG.": (None, None, None, "India", "IND", None, "REGIONAL_AGGREGATE", "Regional benchmark aggregate for Western Indian major container ports."),
        "APSEZ": ("port777", "Mundra", "INMUN", "India", "IND", None, "COMPLEX_PROXY_MUNDRA", "Adani Ports and SEZ flagship container hub at Mundra Port (port777 / INMUN)."),
        "KANDLA": ("port540", "Kandla", "INIXY", "India", "IND", None, "DIRECT_MATCH", "Deendayal Port Trust at Kandla (port540 / INIXY)."),
        "HAZIRA": ("port2039", "Hazira", "INHZA", "India", "IND", None, "DIRECT_MATCH", "Direct identity match with PortWatch Hazira Port (port2039 / INHZA)."),
        "NMPT": ("port811", "New Mangalore", "INNML", "India", "IND", None, "DIRECT_MATCH", "New Mangalore Port Trust (port811 / INNML)."),
        "JNPT": ("port776", "Mumbai", "INNSA", "India", "IND", None, "COMPLEX_PROXY_MUMBAI", "Jawaharlal Nehru Port Trust / Nhava Sheva (INNSA). PortWatch maps under Mumbai complex (port776)."),
        "COCHIN": ("port583", "Cochin (Kochi)", "INCOK", "India", "IND", None, "DIRECT_MATCH", "Cochin / Vallarpadam International Container Transshipment Terminal (port583 / INCOK)."),
        "TUTICORIN": ("port1331", "Tuticorin", "INTUT", "India", "IND", None, "DIRECT_MATCH", "V.O. Chidambaranar Port at Tuticorin (port1331 / INTUT)."),
        "CHENNAI": ("port235", "Chennai (Madras)", "INMAA", "India", "IND", None, "DIRECT_MATCH", "Chennai Port Trust container terminals (port235 / INMAA)."),
        "KATTUPALLI": ("port2038", "Kattupalli", "INKAT", "India", "IND", None, "DIRECT_MATCH", "Adani Kattupalli Port north of Chennai (port2038 / INKAT)."),
        "KRISHNAPATNAM": ("port599", "Krishnapatnam Port", "INKRI", "India", "IND", None, "DIRECT_MATCH", "Krishnapatnam Port container terminal (port599 / INKRI)."),
        "VIZAG": ("port1367", "Visakhapatnam", "INVTZ", "India", "IND", None, "DIRECT_MATCH", "Visakhapatnam Port Trust container terminal (port1367 / INVTZ)."),
        "HALDIA": ("port442", "Haldia", "INHAL", "India", "IND", None, "DIRECT_MATCH", "Haldia Dock Complex (port442 / INHAL)."),
        "KOLKATA": ("port207", "Kolkata (Syama Prasad Mookerje Port)", "INCCU", "India", "IND", None, "DIRECT_MATCH", "Syama Prasad Mookerjee Port at Kolkata / Netaji Subhash Dock (port207 / INCCU).")
    }

    for p in df_dgqi["port_name_raw"].unique():
        meta = dgqi_meta[p]
        bridge_rows.append({
            "source_dataset": "INDIAN_DGQI_PORT_DWELL",
            "source_entity_name": p,
            "entity_type": "regional_average" if meta[6] == "REGIONAL_AGGREGATE" else "individual_port",
            "country_name": meta[3],
            "country_iso3": meta[4],
            "portwatch_id": meta[0],
            "portwatch_name": meta[1],
            "unlocode": meta[2],
            "zenodo_dwell_port_name": meta[5],
            "join_confidence": "NOT_APPLICABLE" if meta[6] == "REGIONAL_AGGREGATE" else ("EXACT" if meta[6] == "DIRECT_MATCH" else "HIGH_PROXY"),
            "mapping_status": meta[6],
            "mapping_notes": meta[7]
        })

    bridge_df = pd.DataFrame(bridge_rows)
    bridge_out = STAGING_PROCESSED_DIR / "supplementary_port_bridge.csv"
    bridge_df.to_csv(bridge_out, index=False)
    print(f"    Saved: {bridge_out} ({len(bridge_df)} rows)")
    return bridge_df

# ---------------------------------------------------------------------------
# Part 4: Target Metric Definition & Semantic Comparability Matrix
# ---------------------------------------------------------------------------

def build_metric_matrix():
    print(">>> Building Target Metric Definition Matrix...")

    matrix_rows = [
        {
            "dataset_name": "ADB AIS Port Traffic Database",
            "metric_field": "avgwaiting_time",
            "target_concept": "vessel_waiting_time",
            "observation_target": "Vessel (Container Ships only)",
            "unit": "Hours",
            "temporal_resolution": "Weekly",
            "spatial_boundary": "Offshore anchorage to pilot station / berth entry",
            "operational_definition": "Average elapsed time container vessels spend idle at anchorage or drifting waiting for berth allocation.",
            "comparability_with_zenodo": "Complementary component (Waiting is a subset of total port dwell time).",
            "interchangeable_flag": False
        },
        {
            "dataset_name": "ADB AIS Port Traffic Database",
            "metric_field": "avgservice_time",
            "target_concept": "vessel_service_time",
            "observation_target": "Vessel (Container Ships only)",
            "unit": "Hours",
            "temporal_resolution": "Weekly",
            "spatial_boundary": "Quay wall / berth mooring pocket",
            "operational_definition": "Average elapsed time container vessels spend moored alongside berth undergoing container handling (stevedoring).",
            "comparability_with_zenodo": "Complementary component (Service is a subset of total port dwell time).",
            "interchangeable_flag": False
        },
        {
            "dataset_name": "ADB AIS Port Traffic Database",
            "metric_field": "avg_turnaround_time_hours (derived)",
            "target_concept": "vessel_turnaround_time",
            "observation_target": "Vessel (Container Ships only)",
            "unit": "Hours (or Days)",
            "temporal_resolution": "Weekly",
            "spatial_boundary": "Port anchorage arrival to berth departure",
            "operational_definition": "Sum of waiting time and service time representing full port turnaround duration for container ships.",
            "comparability_with_zenodo": "Strong proxy: conceptually corresponds to vessel dwell time inside port limits.",
            "interchangeable_flag": False
        },
        {
            "dataset_name": "Zenodo AIS Port Dwell Dataset",
            "metric_field": "MeanDwellDays / MedianDwellDays",
            "target_concept": "vessel_dwell_time",
            "observation_target": "Vessel (All Commercial Cargo: Bulk, Tanker, Container)",
            "unit": "Days",
            "temporal_resolution": "Monthly",
            "spatial_boundary": "PortWatch GIS port polygon boundary (entry-to-exit)",
            "operational_definition": "Total elapsed duration from the moment a vessel enters the PortWatch geographic polygon until it exits.",
            "comparability_with_zenodo": "Baseline ground truth for maritime port polygon dwell.",
            "interchangeable_flag": False
        },
        {
            "dataset_name": "Indian DGQI Port Dwell Dataset",
            "metric_field": "container_dwell_hours",
            "target_concept": "container_dwell_time",
            "observation_target": "Cargo / Container (TEU box logistics)",
            "unit": "Hours",
            "temporal_resolution": "Monthly",
            "spatial_boundary": "Container terminal yard, container freight station (CFS), and port gates",
            "operational_definition": "Import: Time from vessel discharge to gate-out/CFS dispatch. Export: Time from gate-in/CFS arrival to vessel loading.",
            "comparability_with_zenodo": "Completely non-interchangeable. Measures cargo/box dwell on land, NOT vessel time at sea/berth.",
            "interchangeable_flag": False
        }
    ]

    matrix_df = pd.DataFrame(matrix_rows)
    matrix_out = STAGING_PROCESSED_DIR / "target_metric_definition_matrix.csv"
    matrix_df.to_csv(matrix_out, index=False)
    print(f"    Saved: {matrix_out} ({len(matrix_df)} rows)")
    return matrix_df

# ---------------------------------------------------------------------------
# Part 5: Empirical LA/LB and NY/NJ Overlap Comparison
# ---------------------------------------------------------------------------

def build_overlap_comparison(df_adb):
    print(">>> Building LA/LB and NY/NJ Empirical Overlap Comparison Table...")

    zen_path = ZENODO_DWELL_PATH
    assert zen_path.exists(), f"Zenodo dwell file missing: {zen_path}"
    df_zen = pd.read_csv(zen_path)

    # Filter Zenodo to LA/LB and NY/NJ for 2019-2022
    df_zen_sub = df_zen[(df_zen["Port"].isin(["LA_Long_Beach", "NY_NJ"])) & (df_zen["YearMonth"].between("2019-01", "2022-12"))].copy()
    df_zen_sub["adb_port_key"] = df_zen_sub["Port"].map({"LA_Long_Beach": "LALB", "NY_NJ": "NYNJ"})

    # Filter ADB to LALB and NYNJ and aggregate monthly
    adb_sub = df_adb[df_adb["port_name"].isin(["LALB", "NYNJ"])].copy()
    adb_sub["year_month"] = adb_sub["year"].astype(str) + "-" + adb_sub["month"].astype(str).str.zfill(2)

    adb_monthly = adb_sub.groupby(["port_name", "year_month"]).agg({
        "count_entrance": "sum",
        "numofcall": "sum",
        "avgwaiting_time": "mean",
        "avgservice_time": "mean",
        "avg_turnaround_time_hours": "mean",
        "avg_turnaround_time_days": "mean",
        "count_entrance_neopanamax": "sum",
        "share_neopanamax": "mean"
    }).reset_index().rename(columns={
        "count_entrance": "adb_monthly_entrances",
        "numofcall": "adb_monthly_calls",
        "avgwaiting_time": "adb_mean_waiting_hours",
        "avgservice_time": "adb_mean_service_hours",
        "avg_turnaround_time_hours": "adb_mean_turnaround_hours",
        "avg_turnaround_time_days": "adb_mean_turnaround_days",
        "count_entrance_neopanamax": "adb_neopanamax_entrances",
        "share_neopanamax": "adb_mean_neopanamax_share"
    })

    # Join ADB and Zenodo on (port_key, year_month)
    overlap_df = pd.merge(
        df_zen_sub,
        adb_monthly,
        left_on=["adb_port_key", "YearMonth"],
        right_on=["port_name", "year_month"],
        how="inner"
    )

    assert len(overlap_df) == 96, f"Expected exactly 96 overlap rows (48 months x 2 ports), got {len(overlap_df)}"

    overlap_df["zenodo_mean_dwell_hours"] = overlap_df["MeanDwellDays"] * 24.0
    overlap_df["zenodo_median_dwell_hours"] = overlap_df["MedianDwellDays"] * 24.0
    overlap_df["dwell_ratio_zenodo_to_adb"] = overlap_df["MeanDwellDays"] / overlap_df["adb_mean_turnaround_days"]
    overlap_df["waiting_share_of_turnaround_pct"] = (overlap_df["adb_mean_waiting_hours"] / overlap_df["adb_mean_turnaround_hours"]) * 100.0

    cols_to_keep = [
        "Port", "port_name", "YearMonth",
        "UniqueVessels", "MeanDwellDays", "MedianDwellDays", "StdDwellDays", "TotalObservations",
        "zenodo_mean_dwell_hours", "zenodo_median_dwell_hours",
        "adb_monthly_entrances", "adb_monthly_calls",
        "adb_mean_waiting_hours", "adb_mean_service_hours",
        "adb_mean_turnaround_hours", "adb_mean_turnaround_days",
        "adb_neopanamax_entrances", "adb_mean_neopanamax_share",
        "dwell_ratio_zenodo_to_adb", "waiting_share_of_turnaround_pct"
    ]
    overlap_df = overlap_df[cols_to_keep].sort_values(by=["Port", "YearMonth"]).reset_index(drop=True)

    overlap_out = STAGING_PROCESSED_DIR / "lalb_nynj_overlap_comparison.csv"
    overlap_df.to_csv(overlap_out, index=False)
    print(f"    Saved: {overlap_out} ({len(overlap_df)} rows)")

    # Compute correlation metrics
    for p in ["LA_Long_Beach", "NY_NJ"]:
        p_sub = overlap_df[overlap_df["Port"] == p]
        corr_dwell = p_sub["MeanDwellDays"].corr(p_sub["adb_mean_turnaround_days"])
        corr_wait = p_sub["MeanDwellDays"].corr(p_sub["adb_mean_waiting_hours"])
        corr_serv = p_sub["MeanDwellDays"].corr(p_sub["adb_mean_service_hours"])
        print(f"    Correlation for {p} (N=48):")
        print(f"        MeanDwellDays vs ADB Turnaround Days: {corr_dwell:.4f}")
        print(f"        MeanDwellDays vs ADB Waiting Hours:    {corr_wait:.4f}")
        print(f"        MeanDwellDays vs ADB Service Hours:    {corr_serv:.4f}")

    return overlap_df

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    print("=================================================================")
    print("  PHASE 2 SUPPLEMENTARY DATA INTEGRATION & STAGING PIPELINE     ")
    print("=================================================================")

    df_adb = process_adb_dataset()
    df_dgqi = process_dgqi_dataset()
    bridge_df = build_port_bridge_table(df_adb, df_dgqi)
    matrix_df = build_metric_matrix()
    overlap_df = build_overlap_comparison(df_adb)

    print("\n>>> All supplementary datasets processed and validated successfully!")
    print(f"    ADB Processed:           {STAGING_PROCESSED_DIR / 'adb_port_traffic_weekly.csv'}")
    print(f"    DGQI Processed:          {STAGING_PROCESSED_DIR / 'indian_port_dwell_monthly.csv'}")
    print(f"    Cross-Walk Bridge:       {STAGING_PROCESSED_DIR / 'supplementary_port_bridge.csv'}")
    print(f"    Target Metric Matrix:    {STAGING_PROCESSED_DIR / 'target_metric_definition_matrix.csv'}")
    print(f"    LA/LB & NY/NJ Overlap:   {STAGING_PROCESSED_DIR / 'lalb_nynj_overlap_comparison.csv'}")

if __name__ == "__main__":
    main()

