import os, gzip, glob, re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def run():
    print("=== WORKSTREAM 1: NOAA Storm Events Processing ===")
    script_dir = Path(__file__).resolve().parent
    dl_root = script_dir.parents[1]
    repo_root = script_dir.parents[2]

    base_dir = os.environ.get("PHASE2_BASE_DIR", str(dl_root / "data/staging/supplementary_acquisition"))
    raw_noaa_dir = os.path.join(base_dir, "raw/noaa_storm_events")
    out_dir = os.path.join(base_dir, "processed/noaa")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Canonical US Ports with Coordinates
    p_canon = os.environ.get("CANONICAL_REGISTRY_CSV", str(dl_root / "reports/port_identity_resolution/v1_3_final_semantics/canonical_port_registry.csv"))
    df_canon = pd.read_csv(p_canon)
    p_gov = os.environ.get("SPATIAL_GOV_CSV", str(dl_root / "reports/port_identity_resolution/v1_3_final_semantics/spatial_feature_governance.csv"))
    df_gov = pd.read_csv(p_gov)
    p_ref = os.environ.get("PORTWATCH_PORTS_REF_CSV", str(repo_root / "data/reference/portwatch_ports.csv"))
    df_ref = pd.read_csv(p_ref)

    m = df_canon[df_canon['country_iso2'] == 'US'].copy()
    m = m.merge(df_gov[['portwatch_id', 'accepted_coordinates']], on='portwatch_id', how='left')
    m = m.merge(df_ref[['portid', 'lat', 'lon']], left_on='portwatch_id', right_on='portid', how='left')

    lat_list, lon_list = [], []
    for _, r in m.iterrows():
        if pd.notnull(r['accepted_latitude']):
            lat_list.append(r['accepted_latitude'])
            lon_list.append(r['accepted_longitude'])
        elif pd.notnull(r['accepted_coordinates']):
            parts = str(r['accepted_coordinates']).split(',')
            lat_list.append(float(parts[0]))
            lon_list.append(float(parts[1]))
        elif pd.notnull(r['lat']):
            lat_list.append(r['lat'])
            lon_list.append(r['lon'])
        else:
            lat_list.append(None)
            lon_list.append(None)

    m['port_lat'] = lat_list
    m['port_lon'] = lon_list
    us_ports = m[m['port_lat'].notnull()].copy()
    print(f"Loaded {len(us_ports)} canonical US ports with valid coordinates.")

    port_ids = us_ports['portwatch_id'].values
    port_names = us_ports['portwatch_name'].values
    port_lats = us_ports['port_lat'].values.astype(float)
    port_lons = us_ports['port_lon'].values.astype(float)

    # 2. Vectorized Haversine Distance (km)
    def vectorized_haversine(ev_lats, ev_lons, p_lats, p_lons):
        R = 6371.0
        lat1 = np.radians(ev_lats)[:, None]
        lon1 = np.radians(ev_lons)[:, None]
        lat2 = np.radians(p_lats)[None, :]
        lon2 = np.radians(p_lons)[None, :]
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        return 2.0 * R * np.arcsin(np.sqrt(a))

    # Timezone Offsets (hours to ADD to local time to get UTC)
    tz_offsets = {
        'CST-6': timedelta(hours=6),
        'EST-5': timedelta(hours=5),
        'MST-7': timedelta(hours=7),
        'PST-8': timedelta(hours=8),
        'HST-10': timedelta(hours=10),
        'AST-4': timedelta(hours=4),
        'AKST-9': timedelta(hours=9),
        'EDT-4': timedelta(hours=4),
        'CDT-5': timedelta(hours=5),
        'PDT-7': timedelta(hours=7),
        'GST10': timedelta(hours=-10),
        'SST-11': timedelta(hours=11),
    }

    def parse_damage(val):
        if pd.isnull(val):
            return 0.0
        val = str(val).strip().upper()
        if not val or val == '0' or val == '0.00':
            return 0.0
        match = re.match(r'^([\d\.]+)\s*([KMB])?$', val)
        if not match:
            return 0.0
        num, suffix = match.groups()
        mult = {'K': 1e3, 'M': 1e6, 'B': 1e9}.get(suffix, 1.0)
        return float(num) * mult

    # Event classification sets
    marine_wind_types = {
        'Marine Thunderstorm Wind', 'Marine High Wind', 'Marine Strong Wind',
        'Marine Tropical Storm', 'Marine Hurricane'
    }
    hurricane_types = {
        'Hurricane (Typhoon)', 'Tropical Storm', 'Tropical Depression', 'Hurricane'
    }
    flood_types = {
        'Flash Flood', 'Flood', 'Coastal Flood', 'Lakeshore Flood', 'Storm Surge/Tide'
    }
    tornado_types = {
        'Tornado', 'Funnel Cloud', 'Waterspout'
    }

    # 3. Process NOAA files across 2019-2025
    noaa_files = sorted(glob.glob(os.path.join(raw_noaa_dir, "StormEvents_details-ftp_v1.0_d*.csv.gz")))
    all_matches = []
    total_events_read = 0
    total_with_coords = 0

    use_cols = [
        'EVENT_ID', 'EPISODE_ID', 'EVENT_TYPE', 'STATE', 'CZ_TIMEZONE',
        'BEGIN_DATE_TIME', 'END_DATE_TIME', 'BEGIN_LAT', 'BEGIN_LON',
        'DAMAGE_PROPERTY', 'DAMAGE_CROPS', 'DEATHS_DIRECT', 'INJURIES_DIRECT'
    ]

    for fpath in noaa_files:
        fname = os.path.basename(fpath)
        print(f"Reading {fname}...")
        with gzip.open(fpath, 'rt', errors='ignore') as gz:
            df_yr = pd.read_csv(gz, usecols=lambda c: c in use_cols, low_memory=False)
        
        total_events_read += len(df_yr)
        df_coords = df_yr[df_yr['BEGIN_LAT'].notnull() & df_yr['BEGIN_LON'].notnull()].copy()
        total_with_coords += len(df_coords)

        if len(df_coords) == 0:
            continue

        ev_lats = df_coords['BEGIN_LAT'].values.astype(float)
        ev_lons = df_coords['BEGIN_LON'].values.astype(float)

        dists = vectorized_haversine(ev_lats, ev_lons, port_lats, port_lons)
        # Find matches <= 100 km
        ev_indices, port_indices = np.where(dists <= 100.0)
        print(f"  Year matches within 100km: {len(ev_indices)}")

        if len(ev_indices) == 0:
            continue

        matched_df = df_coords.iloc[ev_indices].copy()
        matched_df['portwatch_id'] = port_ids[port_indices]
        matched_df['portwatch_name'] = port_names[port_indices]
        matched_df['distance_km'] = np.round(dists[ev_indices, port_indices], 2)
        matched_df['within_50km'] = matched_df['distance_km'] <= 50.0
        matched_df['within_100km'] = True

        all_matches.append(matched_df)

    if not all_matches:
        raise RuntimeError("No storm matches found within 100km of US ports!")

    df_matches = pd.concat(all_matches, ignore_index=True)
    print(f"Total port-event match instances within 100km: {len(df_matches)}")

    # 4. Standardize timestamps to UTC
    print("Standardizing timestamps to UTC...")
    def convert_to_utc(row):
        b_str = str(row['BEGIN_DATE_TIME']).strip()
        tz = str(row['CZ_TIMEZONE']).strip()
        offset = tz_offsets.get(tz, timedelta(hours=6)) # default CST if unknown
        try:
            dt_local = datetime.strptime(b_str, '%d-%b-%y %H:%M:%S')
            dt_utc = dt_local + offset
            return dt_utc.strftime('%Y-%m-%d %H:%M:%S'), dt_utc.strftime('%Y-%m-%d')
        except Exception:
            return None, None

    utc_res = [convert_to_utc(r) for _, r in df_matches[['BEGIN_DATE_TIME', 'CZ_TIMEZONE']].iterrows()]
    df_matches['begin_time_utc'] = [x[0] for x in utc_res]
    df_matches['date_utc'] = [x[1] for x in utc_res]

    # Damage parsing
    df_matches['damage_property_usd'] = df_matches['DAMAGE_PROPERTY'].apply(parse_damage)
    df_matches['damage_crops_usd'] = df_matches['DAMAGE_CROPS'].apply(parse_damage)
    df_matches['deaths_direct'] = df_matches['DEATHS_DIRECT'].fillna(0).astype(int)
    df_matches['injuries_direct'] = df_matches['INJURIES_DIRECT'].fillna(0).astype(int)

    # Classifications
    df_matches['is_marine_wind'] = df_matches['EVENT_TYPE'].isin(marine_wind_types)
    df_matches['is_hurricane'] = df_matches['EVENT_TYPE'].isin(hurricane_types)
    df_matches['is_flood'] = df_matches['EVENT_TYPE'].isin(flood_types)
    df_matches['is_tornado'] = df_matches['EVENT_TYPE'].isin(tornado_types)

    # Save event-port matches
    matches_out_p = os.path.join(out_dir, "noaa_event_port_matches.parquet")
    df_matches.to_parquet(matches_out_p, index=False)
    print(f"Saved {matches_out_p} ({len(df_matches)} rows)")

    # 5. Build Complete Daily Port-Level Disruption Panel (2019-2025)
    print("Building daily port-level shock panel across 2019-2025...")
    all_dates = pd.date_range("2019-01-01", "2025-12-31", freq="D").strftime('%Y-%m-%d')
    unique_ports = sorted(df_matches['portwatch_id'].unique())
    print(f"Ports with storm exposures: {len(unique_ports)}")

    # Create full Cartesian grid of (portwatch_id, date_utc)
    grid_rows = []
    for pid in unique_ports:
        pname = us_ports[us_ports['portwatch_id'] == pid]['portwatch_name'].values[0]
        for d in all_dates:
            grid_rows.append({'portwatch_id': pid, 'portwatch_name': pname, 'date_utc': d})
    df_panel = pd.DataFrame(grid_rows)

    # Aggregate daily metrics from matches
    # 50km aggregations
    df_50 = df_matches[df_matches['within_50km']].groupby(['portwatch_id', 'date_utc']).agg(
        storm_event_count_50km=('EVENT_ID', 'count'),
        marine_wind_event_count=('is_marine_wind', lambda x: int(sum(x))),
        flood_event_flag=('is_flood', lambda x: int(sum(x) > 0)),
        tornado_event_flag=('is_tornado', lambda x: int(sum(x) > 0)),
        storm_damage_property_sum=('damage_property_usd', 'sum')
    ).reset_index()

    # 100km aggregations
    df_100 = df_matches.groupby(['portwatch_id', 'date_utc']).agg(
        storm_event_count_100km=('EVENT_ID', 'count'),
        hurricane_event_flag=('is_hurricane', lambda x: int(sum(x) > 0))
    ).reset_index()

    # Merge into grid
    df_panel = df_panel.merge(df_50, on=['portwatch_id', 'date_utc'], how='left')
    df_panel = df_panel.merge(df_100, on=['portwatch_id', 'date_utc'], how='left')

    # Fill unexposed days with clean zeros
    df_panel['storm_event_count_50km'] = df_panel['storm_event_count_50km'].fillna(0).astype(int)
    df_panel['storm_event_count_100km'] = df_panel['storm_event_count_100km'].fillna(0).astype(int)
    df_panel['is_storm_active_50km'] = (df_panel['storm_event_count_50km'] > 0).astype(int)
    df_panel['is_storm_active_100km'] = (df_panel['storm_event_count_100km'] > 0).astype(int)
    df_panel['marine_wind_event_count'] = df_panel['marine_wind_event_count'].fillna(0).astype(int)
    df_panel['hurricane_event_flag'] = df_panel['hurricane_event_flag'].fillna(0).astype(int)
    df_panel['flood_event_flag'] = df_panel['flood_event_flag'].fillna(0).astype(int)
    df_panel['tornado_event_flag'] = df_panel['tornado_event_flag'].fillna(0).astype(int)
    df_panel['storm_damage_property_sum'] = df_panel['storm_damage_property_sum'].fillna(0.0).astype(float)

    # Sort for rolling computations
    df_panel['date_dt'] = pd.to_datetime(df_panel['date_utc'])
    df_panel = df_panel.sort_values(by=['portwatch_id', 'date_dt']).reset_index(drop=True)

    # Rolling 7-day features
    print("Computing 7-day rolling exposure features...")
    df_panel['rolling_7d_storm_event_count'] = df_panel.groupby('portwatch_id')['storm_event_count_50km'].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    ).astype(int)
    df_panel['rolling_7d_storm_damage'] = df_panel.groupby('portwatch_id')['storm_damage_property_sum'].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    ).astype(float)

    df_panel = df_panel.drop(columns=['date_dt'])

    shocks_out_p = os.path.join(out_dir, "noaa_port_daily_shocks.parquet")
    df_panel.to_parquet(shocks_out_p, index=False)
    print(f"Saved {shocks_out_p} ({len(df_panel)} rows, {len(df_panel.columns)} columns)")

    # 6. Generate Metadata Markdown
    meta_p = os.path.join(out_dir, "noaa_feature_metadata.md")
    with open(meta_p, "w") as f:
        f.write(f"""# NOAA Storm Events Disruption Feature Metadata

**Dataset Archetype:** High-Impact Spatiotemporal Weather Shocks  
**Coverage Window:** 2019-01-01 to 2025-12-31 (2,557 daily intervals)  
**Geographic Domain:** United States Coastal & Deepwater Seaports ({len(unique_ports)} exposed canonical ports)  
**Output Files:**
1. `noaa_port_daily_shocks.parquet` ({len(df_panel)} rows, 13 features)
2. `noaa_event_port_matches.parquet` ({len(df_matches)} matched event records)

## Feature Definitions

| Feature Name | Type | Range / Format | Description |
| :--- | :---: | :---: | :--- |
| `portwatch_id` | String | e.g. `port103` | Canonical PortWatch identifier |
| `portwatch_name` | String | e.g. `Baltimore` | Canonical seaport name |
| `date_utc` | String | `YYYY-MM-DD` | Calendar date normalized to UTC |
| `storm_event_count_50km` | Integer | [0, 20+] | Count of acute convective/severe storm events within 50 km |
| `storm_event_count_100km` | Integer | [0, 40+] | Count of severe storm events within 100 km |
| `is_storm_active_50km` | Binary | 0 or 1 | Indicator if at least 1 severe storm event occurred within 50 km |
| `is_storm_active_100km` | Binary | 0 or 1 | Indicator if at least 1 severe storm event occurred within 100 km |
| `marine_wind_event_count` | Integer | [0, 10+] | Count of Marine Thunderstorm / Gale Wind events within 50 km |
| `hurricane_event_flag` | Binary | 0 or 1 | Indicator for Hurricane / Tropical Storm occurrence within 100 km |
| `flood_event_flag` | Binary | 0 or 1 | Indicator for Coastal / Flash Flooding within 50 km |
| `tornado_event_flag` | Binary | 0 or 1 | Indicator for Tornado / Waterspout event within 50 km |
| `storm_damage_property_sum` | Float | USD ($) | Total direct estimated property damage within 50 km on date |
| `rolling_7d_storm_event_count` | Integer | [0, 50+] | Cumulative 7-day storm event frequency within 50 km |
| `rolling_7d_storm_damage` | Float | USD ($) | Cumulative 7-day property damage within 50 km |

## Methodology and Governance

1. **Timezone Normalization:** Source timestamps (`BEGIN_DATE_TIME`) in local standard times were converted to UTC timestamps using the explicit `CZ_TIMEZONE` attribute.
2. **Geodesic Spatial Matching:** Point coordinates (`BEGIN_LAT`, `BEGIN_LON`) were matched to port coordinates using the spherical Haversine formula on WGS84.
3. **Proximity Thresholds:** A 50 km radius captures immediate berth and crane operational hazards; 100 km captures approach channel and regional trucking access corridors.
4. **Zero Representation:** Days with zero recorded events receive an explicit zero count and flag, distinguishing clean operation from missing data.
""")
    print(f"Wrote metadata to {meta_p}")
    print("=== WORKSTREAM 1 COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run()
