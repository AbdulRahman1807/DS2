"""Build Unified Temporal Feature Dataset for Phase 2 Data Engineering.

Generates:
- DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet
- DL/data/staging/supplementary_acquisition/processed/unified_temporal/temporal_join_coverage.csv
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DL_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]

BASE_DIR = os.environ.get("PHASE2_BASE_DIR", str(DL_ROOT / "data/staging/supplementary_acquisition"))
OUT_DIR = os.path.join(BASE_DIR, "processed/unified_temporal")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    t0 = time.time()
    print("=== STARTING WORKSTREAM A & B: UNIFIED TEMPORAL PIPELINE ===")
    
    # -------------------------------------------------------------
    # 1. Load Primary PortWatch Data (2019-2025)
    # -------------------------------------------------------------
    pw_path = os.environ.get("PORTWATCH_PATH", str(DL_ROOT / "data/raw/portwatch/Daily_Ports_Data.csv"))
    print(f"Loading PortWatch from {pw_path}...")
    
    cols = [
        'portid', 'portname', 'date', 'year', 'month', 'day', 'country', 'ISO3',
        'portcalls', 'portcalls_container', 'portcalls_cargo', 'portcalls_tanker',
        'portcalls_dry_bulk', 'portcalls_general_cargo', 'portcalls_roro',
        'import_cargo', 'export_cargo', 'import_container', 'export_container'
    ]
    
    chunks = []
    for c in pd.read_csv(pw_path, usecols=cols, chunksize=1000000):
        c = c[c['year'] <= 2025]
        chunks.append(c)
    
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    print(f"PortWatch loaded: {len(df):,} rows. Sorting chronologically per port...")
    
    # Standardize column names and format date
    df.rename(columns={
        'portid': 'portwatch_id',
        'portname': 'portwatch_name',
        'country': 'country_name',
        'ISO3': 'country_iso3'
    }, inplace=True)
    
    df['date_utc'] = df['date'].str.replace('/', '-')
    df.drop(columns=['date'], inplace=True)
    
    # Sort strictly by (portwatch_id, date_utc)
    df = df.sort_values(['portwatch_id', 'date_utc']).reset_index(drop=True)
    print("Sorting complete. Adding calendar features...")
    
    dt_series = pd.to_datetime(df['date_utc'])
    df['day_of_week'] = dt_series.dt.dayofweek.astype(np.int8)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(np.int8)
    df['day_of_year'] = dt_series.dt.dayofyear.astype(np.int16)
    del dt_series
    gc.collect()
    
    # -------------------------------------------------------------
    # 2. Rolling Historical Activity Features (Vectorized per port)
    # -------------------------------------------------------------
    print("Computing trailing rolling activity features...")
    df['pw_calls_rolling_7d_mean'] = df.groupby('portwatch_id')['portcalls'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    ).astype(np.float32)
    
    df['pw_calls_rolling_14d_mean'] = df.groupby('portwatch_id')['portcalls'].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    ).astype(np.float32)
    
    df['pw_calls_rolling_30d_mean'] = df.groupby('portwatch_id')['portcalls'].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    ).astype(np.float32)
    
    df['pw_container_rolling_7d_mean'] = df.groupby('portwatch_id')['portcalls_container'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    ).astype(np.float32)
    
    df['pw_container_rolling_14d_mean'] = df.groupby('portwatch_id')['portcalls_container'].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    ).astype(np.float32)
    
    # Total cargo tonnage = import_cargo + export_cargo
    df['total_cargo_tonnage'] = (df['import_cargo'] + df['export_cargo']).astype(np.float32)
    
    # -------------------------------------------------------------
    # 3. Workstream B: Forecasting Targets (7-Day Horizon, Zero Leakage)
    # -------------------------------------------------------------
    print("Computing 7-day future forecasting targets and lead vectors...")
    for i in range(1, 8):
        df[f'target_portcalls_lead_{i}d'] = df.groupby('portwatch_id')['portcalls'].shift(-i).astype(np.float32)
    
    df['target_portcalls_7d_future_sum'] = sum([df[f'target_portcalls_lead_{i}d'] for i in range(1, 8)]).astype(np.float32)
    
    # Container calls future sum
    cnt_leads = [df.groupby('portwatch_id')['portcalls_container'].shift(-i).astype(np.float32) for i in range(1, 8)]
    df['target_container_calls_7d_future_sum'] = sum(cnt_leads).astype(np.float32)
    del cnt_leads
    
    # Cargo tonnage future sum
    cargo_leads = [df.groupby('portwatch_id')['total_cargo_tonnage'].shift(-i).astype(np.float32) for i in range(1, 8)]
    df['target_cargo_tonnage_7d_future_sum'] = sum(cargo_leads).astype(np.float32)
    del cargo_leads
    gc.collect()
    
    # -------------------------------------------------------------
    # 4. Integrate ERA5 Daily Features
    # -------------------------------------------------------------
    era5_path = os.environ.get("ERA5_DAILY_PATH", str(DL_ROOT / "data/processed/era5/era5_port_daily_v2.parquet"))
    print(f"Integrating ERA5 features from {era5_path}...")
    df_era5 = pd.read_parquet(era5_path)
    df_era5['date_utc'] = df_era5['date_utc'].dt.strftime('%Y-%m-%d')
    df_era5.rename(columns={'port_id': 'portwatch_id'}, inplace=True)
    
    # Classify wave proxy type based on distance
    df_era5['era5_wave_proxy_type'] = np.where(
        df_era5['wave_distance_km'] <= 40.0,
        'NEARBY_COASTAL_PROXY',
        'CONDITIONAL_OFFSHORE_PROXY'
    )
    
    era5_cols = [
        'portwatch_id', 'date_utc',
        'wind_speed_mean_ms', 'wind_speed_max_ms', 't2m_mean_c', 't2m_min_c', 't2m_max_c',
        'msl_mean_hpa', 'msl_min_hpa', 'msl_max_hpa', 'precipitation_mm',
        'wind_direction_mean_deg', 'swh_mean_m', 'swh_max_m', 'mwp_mean_s',
        'wave_direction_mean_deg', 'wave_distance_km', 'era5_wave_proxy_type'
    ]
    df_era5 = df_era5[era5_cols].copy()
    df_era5.rename(columns={
        'wind_speed_mean_ms': 'era5_wind_speed_mean_ms',
        'wind_speed_max_ms': 'era5_wind_speed_max_ms',
        't2m_mean_c': 'era5_t2m_mean_c',
        't2m_min_c': 'era5_t2m_min_c',
        't2m_max_c': 'era5_t2m_max_c',
        'msl_mean_hpa': 'era5_msl_mean_hpa',
        'msl_min_hpa': 'era5_msl_min_hpa',
        'msl_max_hpa': 'era5_msl_max_hpa',
        'precipitation_mm': 'era5_precipitation_mm',
        'wind_direction_mean_deg': 'era5_wind_direction_mean_deg',
        'swh_mean_m': 'era5_swh_mean_m',
        'swh_max_m': 'era5_swh_max_m',
        'mwp_mean_s': 'era5_mwp_mean_s',
        'wave_direction_mean_deg': 'era5_wave_direction_mean_deg',
        'wave_distance_km': 'era5_wave_distance_km'
    }, inplace=True)
    
    era5_ports = set(df_era5['portwatch_id'].unique())
    df['has_era5_coverage'] = df['portwatch_id'].isin(era5_ports).astype(np.int8)
    
    df = df.merge(df_era5, on=['portwatch_id', 'date_utc'], how='left')
    del df_era5
    gc.collect()
    print("ERA5 merged successfully.")
    
    # -------------------------------------------------------------
    # 5. Integrate NOAA Port Daily Shocks
    # -------------------------------------------------------------
    noaa_path = os.path.join(BASE_DIR, "processed/noaa/noaa_port_daily_shocks.parquet")
    print(f"Integrating NOAA shocks from {noaa_path}...")
    df_noaa = pd.read_parquet(noaa_path)
    noaa_ports = set(df_noaa['portwatch_id'].unique())
    df['has_noaa_coverage'] = df['portwatch_id'].isin(noaa_ports).astype(np.int8)
    
    df_noaa = df_noaa.drop(columns=['portwatch_name'], errors='ignore')
    df_noaa.rename(columns={
        'storm_event_count_50km': 'noaa_storm_event_count_50km',
        'marine_wind_event_count': 'noaa_marine_wind_event_count',
        'flood_event_flag': 'noaa_flood_event_flag',
        'tornado_event_flag': 'noaa_tornado_event_flag',
        'storm_damage_property_sum': 'noaa_storm_damage_property_sum',
        'storm_event_count_100km': 'noaa_storm_event_count_100km',
        'hurricane_event_flag': 'noaa_hurricane_event_flag',
        'is_storm_active_50km': 'noaa_is_storm_active_50km',
        'is_storm_active_100km': 'noaa_is_storm_active_100km',
        'rolling_7d_storm_event_count': 'noaa_rolling_7d_storm_event_count',
        'rolling_7d_storm_damage': 'noaa_rolling_7d_storm_damage'
    }, inplace=True)
    
    df = df.merge(df_noaa, on=['portwatch_id', 'date_utc'], how='left')
    del df_noaa
    gc.collect()
    print("NOAA merged successfully.")
    
    # -------------------------------------------------------------
    # 6. Integrate EM-DAT Country Daily Shocks
    # -------------------------------------------------------------
    emdat_path = os.path.join(BASE_DIR, "processed/emdat/emdat_country_daily_shocks.parquet")
    print(f"Integrating EM-DAT disaster shocks from {emdat_path}...")
    df_emdat = pd.read_parquet(emdat_path)
    emdat_countries = set(df_emdat['country_iso3'].unique())
    df['has_emdat_coverage'] = df['country_iso3'].isin(emdat_countries).astype(np.int8)
    
    df_emdat.rename(columns={
        'active_disaster_count': 'emdat_active_disaster_count',
        'natural_disaster_count': 'emdat_natural_disaster_count',
        'technological_disaster_count': 'emdat_technological_disaster_count',
        'macro_disaster_active': 'emdat_macro_disaster_active',
        'rolling_7d_disaster_count': 'emdat_rolling_7d_disaster_count',
        'rolling_30d_disaster_count': 'emdat_rolling_30d_disaster_count',
        'total_deaths': 'emdat_total_deaths',
        'total_affected': 'emdat_total_affected',
        'total_damage_usd': 'emdat_total_damage_usd',
        'log_affected_population': 'emdat_log_affected_population'
    }, inplace=True)
    
    df = df.merge(df_emdat, on=['country_iso3', 'date_utc'], how='left')
    del df_emdat
    gc.collect()
    print("EM-DAT merged successfully.")
    
    # -------------------------------------------------------------
    # 7. Integrate BTS Port Capacity Features & Pressure Ratio
    # -------------------------------------------------------------
    bts_path = os.path.join(BASE_DIR, "processed/bts/bts_port_capacity_features.parquet")
    print(f"Integrating BTS capacity features from {bts_path}...")
    df_bts = pd.read_parquet(bts_path)
    bts_ports = set(df_bts['canonical_portwatch_id'].unique())
    df['has_bts_coverage'] = df['portwatch_id'].isin(bts_ports).astype(np.int8)
    
    # Prepare BTS features: 2016-2023 exact; forward-fill 2023 baseline for 2024-2025
    # Extract unique ports and create full (canonical_portwatch_id, year) grid for 2019-2025
    grid_years = pd.MultiIndex.from_product(
        [sorted(list(bts_ports)), list(range(2019, 2026))],
        names=['canonical_portwatch_id', 'year']
    ).to_frame().reset_index(drop=True)
    
    # Merge existing BTS data
    bts_grid = grid_years.merge(
        df_bts.rename(columns={'reporting_year': 'year'}),
        on=['canonical_portwatch_id', 'year'],
        how='left'
    )
    
    # Sort and forward-fill within each port
    bts_grid = bts_grid.sort_values(['canonical_portwatch_id', 'year']).reset_index(drop=True)
    cols_to_ffill = [
        'annual_container_teu_baseline', 'annual_vessel_call_baseline',
        'annual_cargo_short_tons_baseline', 'daily_expected_vessel_calls',
        'daily_expected_teu', 'daily_expected_short_tons',
        'annual_teu_growth_pct', 'annual_tonnage_growth_pct', 'port_national_rank'
    ]
    
    # Mark forward-fill status
    bts_grid['is_bts_forward_filled'] = np.where(bts_grid['year'] > 2023, 1, 0).astype(np.int8)
    bts_grid['bts_baseline_year'] = np.where(bts_grid['year'] > 2023, 2023, bts_grid['year']).astype(np.int16)
    
    # Forward fill the baseline attributes
    bts_grid[cols_to_ffill] = bts_grid.groupby('canonical_portwatch_id')[cols_to_ffill].ffill()
    
    bts_grid = bts_grid[[
        'canonical_portwatch_id', 'year', 'is_bts_forward_filled', 'bts_baseline_year',
        'annual_container_teu_baseline', 'annual_vessel_call_baseline',
        'annual_cargo_short_tons_baseline', 'daily_expected_vessel_calls',
        'daily_expected_teu', 'daily_expected_short_tons',
        'annual_teu_growth_pct', 'annual_tonnage_growth_pct', 'port_national_rank'
    ]].copy()
    
    bts_grid.rename(columns={
        'canonical_portwatch_id': 'portwatch_id',
        'annual_container_teu_baseline': 'bts_annual_container_teu_baseline',
        'annual_vessel_call_baseline': 'bts_annual_vessel_call_baseline',
        'annual_cargo_short_tons_baseline': 'bts_annual_cargo_short_tons_baseline',
        'daily_expected_vessel_calls': 'bts_daily_expected_vessel_calls',
        'daily_expected_teu': 'bts_daily_expected_teu',
        'daily_expected_short_tons': 'bts_daily_expected_short_tons',
        'annual_teu_growth_pct': 'bts_annual_teu_growth_pct',
        'annual_tonnage_growth_pct': 'bts_annual_tonnage_growth_pct',
        'port_national_rank': 'bts_port_national_rank'
    }, inplace=True)
    
    df = df.merge(bts_grid, on=['portwatch_id', 'year'], how='left')
    del bts_grid, df_bts
    gc.collect()
    
    # Compute Capacity Pressure Ratio: rolling 14-day calls / daily expected vessel calls
    # Safe division: if expected calls is 0, NaN, or port not covered by BTS, result is NaN
    valid_expected_mask = (df['bts_daily_expected_vessel_calls'] > 0) & df['bts_daily_expected_vessel_calls'].notnull()
    df['bts_capacity_pressure_ratio_14d'] = np.where(
        valid_expected_mask,
        df['pw_calls_rolling_14d_mean'] / df['bts_daily_expected_vessel_calls'],
        np.nan
    ).astype(np.float32)
    print("BTS features merged and capacity pressure ratio computed.")
    
    # -------------------------------------------------------------
    # 8. Write Parquet in Partitioned Streaming Fashion
    # -------------------------------------------------------------
    out_parquet = os.path.join(OUT_DIR, "port_daily_unified.parquet")
    print(f"Writing unified temporal parquet to {out_parquet}...")
    
    # Ensure correct dtypes for pyarrow schema stability
    for c in df.select_dtypes(include='float64').columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include='int64').columns:
        if c in ['year']:
            df[c] = df[c].astype(np.int16)
        elif c in ['month', 'day']:
            df[c] = df[c].astype(np.int8)
        else:
            df[c] = df[c].astype(np.int32)
            
    # Write using pyarrow Table
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_parquet, compression='snappy')
    del table
    print(f"Parquet written: {os.path.getsize(out_parquet) / (1024*1024):.2f} MB")
    
    # -------------------------------------------------------------
    # 9. Compute Temporal Join Coverage Summary
    # -------------------------------------------------------------
    cov_path = os.path.join(OUT_DIR, "temporal_join_coverage.csv")
    print(f"Computing join coverage metrics and writing to {cov_path}...")
    
    total_ports = df['portwatch_id'].nunique()
    total_countries = df['country_iso3'].nunique()
    total_rows = len(df)
    
    cov_data = [
        {
            'dataset_source': 'PortWatch Daily (Base)',
            'join_key': 'portwatch_id, date_utc',
            'covered_entities': total_ports,
            'total_possible_entities': total_ports,
            'coverage_entity_pct': 100.0,
            'covered_rows': total_rows,
            'total_rows': total_rows,
            'row_coverage_pct': 100.0,
            'temporal_range': '2019-01-01 to 2025-12-31',
            'coverage_indicator_col': 'has_portwatch_coverage'
        },
        {
            'dataset_source': 'ERA5 Daily Environmental v2',
            'join_key': 'portwatch_id, date_utc',
            'covered_entities': len(era5_ports),
            'total_possible_entities': total_ports,
            'coverage_entity_pct': round(len(era5_ports) / total_ports * 100, 3),
            'covered_rows': int(df['has_era5_coverage'].sum()),
            'total_rows': total_rows,
            'row_coverage_pct': round(df['has_era5_coverage'].sum() / total_rows * 100, 3),
            'temporal_range': '2019-01-01 to 2025-12-31',
            'coverage_indicator_col': 'has_era5_coverage'
        },
        {
            'dataset_source': 'NOAA Storm Events Shocks',
            'join_key': 'portwatch_id, date_utc',
            'covered_entities': len(noaa_ports),
            'total_possible_entities': total_ports,
            'coverage_entity_pct': round(len(noaa_ports) / total_ports * 100, 3),
            'covered_rows': int(df['has_noaa_coverage'].sum()),
            'total_rows': total_rows,
            'row_coverage_pct': round(df['has_noaa_coverage'].sum() / total_rows * 100, 3),
            'temporal_range': '2019-01-01 to 2025-12-31',
            'coverage_indicator_col': 'has_noaa_coverage'
        },
        {
            'dataset_source': 'EM-DAT Macro Disaster Shocks',
            'join_key': 'country_iso3, date_utc',
            'covered_entities': len(df[df['has_emdat_coverage'] == 1]['country_iso3'].unique()),
            'total_possible_entities': total_countries,
            'coverage_entity_pct': round(len(df[df['has_emdat_coverage'] == 1]['country_iso3'].unique()) / total_countries * 100, 3),
            'covered_rows': int(df['has_emdat_coverage'].sum()),
            'total_rows': total_rows,
            'row_coverage_pct': round(df['has_emdat_coverage'].sum() / total_rows * 100, 3),
            'temporal_range': '2019-01-01 to 2025-12-31',
            'coverage_indicator_col': 'has_emdat_coverage'
        },
        {
            'dataset_source': 'BTS Port Capacity Features',
            'join_key': 'portwatch_id, year',
            'covered_entities': len(bts_ports),
            'total_possible_entities': total_ports,
            'coverage_entity_pct': round(len(bts_ports) / total_ports * 100, 3),
            'covered_rows': int(df['has_bts_coverage'].sum()),
            'total_rows': total_rows,
            'row_coverage_pct': round(df['has_bts_coverage'].sum() / total_rows * 100, 3),
            'temporal_range': '2019 to 2025 (2024-2025 baseline carry-forward)',
            'coverage_indicator_col': 'has_bts_coverage'
        }
    ]
    
    df_cov = pd.DataFrame(cov_data)
    df_cov.to_csv(cov_path, index=False)
    print("Temporal join coverage CSV written.")
    print(df_cov.to_string())
    
    print(f"\nPipeline successfully completed in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
