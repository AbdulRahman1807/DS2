"""Generate final_feature_dictionary.csv for all Phase 2 data engineering assets."""

import os
import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DL_ROOT = SCRIPT_DIR.parents[1]

BASE_DIR = os.environ.get("PHASE2_BASE_DIR", str(DL_ROOT / "data/staging/supplementary_acquisition"))
OUT_DIR = os.path.join(BASE_DIR, "final_handoff")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    rows = []
    
    # 1. port_daily_unified.parquet
    p_unified = os.path.join(BASE_DIR, "processed/unified_temporal/port_daily_unified.parquet")
    df_u = pd.read_parquet(p_unified)
    
    # Read descriptions from port_daily_feature_dictionary.md if available or define systematically
    u_desc = {
        'year': ('Calendar year (2019-2025)', 'PortWatch Primary', 'Global (100%)', 'Never null'),
        'month': ('Calendar month (1-12)', 'PortWatch Primary', 'Global (100%)', 'Never null'),
        'day': ('Day of month (1-31)', 'PortWatch Primary', 'Global (100%)', 'Never null'),
        'portwatch_id': ('Canonical alphanumeric port ID', 'PortWatch Primary', 'Global (100%)', 'Never null'),
        'portwatch_name': ('Official port name', 'PortWatch Primary', 'Global (100%)', 'Never null'),
        'country_name': ('Sovereign host country', 'PortWatch Primary', 'Global (100%)', 'Never null'),
        'country_iso3': ('3-letter ISO-3 country code', 'PortWatch Primary', 'Global (100%)', 'Never null'),
        'portcalls_container': ('Container vessel calls on day t', 'PortWatch AIS', 'Global (100%)', '0 for zero calls'),
        'portcalls_dry_bulk': ('Dry bulk vessel calls on day t', 'PortWatch AIS', 'Global (100%)', '0 for zero calls'),
        'portcalls_general_cargo': ('General cargo calls on day t', 'PortWatch AIS', 'Global (100%)', '0 for zero calls'),
        'portcalls_roro': ('Ro-Ro vehicle carrier calls on day t', 'PortWatch AIS', 'Global (100%)', '0 for zero calls'),
        'portcalls_tanker': ('Tanker vessel calls on day t', 'PortWatch AIS', 'Global (100%)', '0 for zero calls'),
        'portcalls_cargo': ('Combined cargo calls on day t', 'PortWatch AIS', 'Global (100%)', '0 for zero calls'),
        'portcalls': ('Total vessel calls on day t', 'PortWatch AIS', 'Global (100%)', '0 for zero calls'),
        'import_container': ('Estimated import containers on day t', 'PortWatch AIS', 'Global (100%)', '0 for zero calls'),
        'import_cargo': ('Estimated import cargo tonnage on day t', 'PortWatch AIS', 'Global (100%)', '0 for zero calls'),
        'export_container': ('Estimated export containers on day t', 'PortWatch AIS', 'Global (100%)', '0 for zero calls'),
        'export_cargo': ('Estimated export cargo tonnage on day t', 'PortWatch AIS', 'Global (100%)', '0 for zero calls'),
        'date_utc': ('Universal calendar date (YYYY-MM-DD)', 'PortWatch Primary', 'Global (100%)', 'Never null'),
        'day_of_week': ('Day of week index (0=Mon, 6=Sun)', 'Derived Calendar', 'Global (100%)', 'Never null'),
        'is_weekend': ('Weekend binary indicator {0,1}', 'Derived Calendar', 'Global (100%)', 'Never null'),
        'day_of_year': ('Julian day of year (1-366)', 'Derived Calendar', 'Global (100%)', 'Never null'),
        'pw_calls_rolling_7d_mean': ('Trailing 7-day rolling mean calls', 'Derived Rolling', 'Global (100%)', 'Never null'),
        'pw_calls_rolling_14d_mean': ('Trailing 14-day rolling mean calls', 'Derived Rolling', 'Global (100%)', 'Never null'),
        'pw_calls_rolling_30d_mean': ('Trailing 30-day rolling mean calls', 'Derived Rolling', 'Global (100%)', 'Never null'),
        'pw_container_rolling_7d_mean': ('Trailing 7-day rolling container calls', 'Derived Rolling', 'Global (100%)', 'Never null'),
        'pw_container_rolling_14d_mean': ('Trailing 14-day rolling container calls', 'Derived Rolling', 'Global (100%)', 'Never null'),
        'total_cargo_tonnage': ('Total waterborne cargo (import + export)', 'Derived Activity', 'Global (100%)', 'Never null'),
        'target_portcalls_lead_1d': ('1-day ahead vessel calls (t+1)', 'Forecasting Target', 'Global (100%)', 'Null for final 1 day'),
        'target_portcalls_lead_2d': ('2-day ahead vessel calls (t+2)', 'Forecasting Target', 'Global (100%)', 'Null for final 2 days'),
        'target_portcalls_lead_3d': ('3-day ahead vessel calls (t+3)', 'Forecasting Target', 'Global (100%)', 'Null for final 3 days'),
        'target_portcalls_lead_4d': ('4-day ahead vessel calls (t+4)', 'Forecasting Target', 'Global (100%)', 'Null for final 4 days'),
        'target_portcalls_lead_5d': ('5-day ahead vessel calls (t+5)', 'Forecasting Target', 'Global (100%)', 'Null for final 5 days'),
        'target_portcalls_lead_6d': ('6-day ahead vessel calls (t+6)', 'Forecasting Target', 'Global (100%)', 'Null for final 6 days'),
        'target_portcalls_lead_7d': ('7-day ahead vessel calls (t+7)', 'Forecasting Target', 'Global (100%)', 'Null for final 7 days'),
        'target_portcalls_7d_future_sum': ('7-day cumulative total vessel calls sum', 'Forecasting Target', 'Global (100%)', 'Null for final 7 days'),
        'target_container_calls_7d_future_sum': ('7-day cumulative container calls sum', 'Forecasting Target', 'Global (100%)', 'Null for final 7 days'),
        'target_cargo_tonnage_7d_future_sum': ('7-day cumulative total cargo tonnage sum', 'Forecasting Target', 'Global (100%)', 'Null for final 7 days'),
        'has_era5_coverage': ('Indicator for ERA5 reanalysis coverage', 'Spatial Governance', '5 US Ports (0.24%)', 'Never null (0 or 1)'),
        'era5_wind_speed_mean_ms': ('Daily mean 10m wind speed (m/s)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_wind_speed_max_ms': ('Daily max 10m wind speed (m/s)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_t2m_mean_c': ('Daily mean 2m air temp (deg C)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_t2m_min_c': ('Daily min 2m air temp (deg C)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_t2m_max_c': ('Daily max 2m air temp (deg C)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_msl_mean_hpa': ('Daily mean barometric pressure (hPa)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_msl_min_hpa': ('Daily min barometric pressure (hPa)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_msl_max_hpa': ('Daily max barometric pressure (hPa)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_precipitation_mm': ('Daily total precipitation (mm)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_wind_direction_mean_deg': ('Daily mean wind direction (0-360 deg)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_swh_mean_m': ('Daily mean significant wave height (m)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_swh_max_m': ('Daily max significant wave height (m)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_mwp_mean_s': ('Daily mean wave period (seconds)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_wave_direction_mean_deg': ('Daily mean wave propagation dir (deg)', 'ERA5 Reanalysis v2', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_wave_distance_km': ('Distance from port to wave grid (km)', 'Spatial Governance', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'era5_wave_proxy_type': ('Proxy classification (COASTAL/OFFSHORE)', 'Spatial Governance', '5 Ports (12,785 rows)', 'Null if unmapped'),
        'has_noaa_coverage': ('Indicator for NOAA storm spatial match', 'Spatial Matching', '56 US Ports (2.71%)', 'Never null (0 or 1)'),
        'noaa_storm_event_count_50km': ('Severe storm events within 50km', 'NOAA Storm Events', '56 Ports (143,192 rows)', '0 if calm, null if unmapped'),
        'noaa_marine_wind_event_count': ('Marine wind / gale events within 50km', 'NOAA Storm Events', '56 Ports (143,192 rows)', '0 if calm, null if unmapped'),
        'noaa_flood_event_flag': ('Coastal/flash flood flag within 50km', 'NOAA Storm Events', '56 Ports (143,192 rows)', '0 if calm, null if unmapped'),
        'noaa_tornado_event_flag': ('Tornado touchdown flag within 50km', 'NOAA Storm Events', '56 Ports (143,192 rows)', '0 if calm, null if unmapped'),
        'noaa_storm_damage_property_sum': ('Total property damage USD within 50km', 'NOAA Storm Events', '56 Ports (143,192 rows)', '0 if calm, null if unmapped'),
        'noaa_storm_event_count_100km': ('Severe storm events within 100km', 'NOAA Storm Events', '56 Ports (143,192 rows)', '0 if calm, null if unmapped'),
        'noaa_hurricane_event_flag': ('Tropical cyclone flag within 100km', 'NOAA Storm Events', '56 Ports (143,192 rows)', '0 if calm, null if unmapped'),
        'noaa_is_storm_active_50km': ('Binary active storm flag within 50km', 'NOAA Storm Events', '56 Ports (143,192 rows)', '0 if calm, null if unmapped'),
        'noaa_is_storm_active_100km': ('Binary active storm flag within 100km', 'NOAA Storm Events', '56 Ports (143,192 rows)', '0 if calm, null if unmapped'),
        'noaa_rolling_7d_storm_event_count': ('Trailing 7-day storm count within 50km', 'NOAA Storm Events', '56 Ports (143,192 rows)', '0 if calm, null if unmapped'),
        'noaa_rolling_7d_storm_damage': ('Trailing 7-day property damage USD', 'NOAA Storm Events', '56 Ports (143,192 rows)', '0 if calm, null if unmapped'),
        'has_emdat_coverage': ('Indicator for sovereign EM-DAT disaster panel', 'Sovereign Match', '150 Countries (97.2%)', 'Never null (0 or 1)'),
        'emdat_active_disaster_count': ('Total active disasters on date t', 'EM-DAT Database', '150 Countries (5.13M rows)', '0 if calm, null if unmapped'),
        'emdat_natural_disaster_count': ('Active natural disasters on date t', 'EM-DAT Database', '150 Countries (5.13M rows)', '0 if calm, null if unmapped'),
        'emdat_technological_disaster_count': ('Active tech disasters on date t', 'EM-DAT Database', '150 Countries (5.13M rows)', '0 if calm, null if unmapped'),
        'emdat_total_deaths': ('Recorded disaster fatalities', 'EM-DAT Database', '150 Countries', 'NaN if missing/unreported'),
        'emdat_total_affected': ('Recorded affected persons', 'EM-DAT Database', '150 Countries', 'NaN if missing/unreported'),
        'emdat_total_damage_usd': ('Recorded direct damage USD', 'EM-DAT Database', '150 Countries', 'NaN if missing/unreported'),
        'emdat_macro_disaster_active': ('Binary macro disaster active flag', 'EM-DAT Database', '150 Countries (5.13M rows)', '0 if calm, null if unmapped'),
        'emdat_log_affected_population': ('Log10(1 + affected population)', 'Derived Macro', '150 Countries', '0.0 if zero, NaN if missing'),
        'emdat_rolling_7d_disaster_count': ('Trailing 7-day disaster event-days', 'Derived Macro', '150 Countries (5.13M rows)', '0 if calm, null if unmapped'),
        'emdat_rolling_30d_disaster_count': ('Trailing 30-day disaster event-days', 'Derived Macro', '150 Countries (5.13M rows)', '0 if calm, null if unmapped'),
        'has_bts_coverage': ('Indicator for BTS port capacity mapping', 'Structural Crosswalk', '39 US Ports (1.74%)', 'Never null (0 or 1)'),
        'is_bts_forward_filled': ('Binary flag: 1 if 2024-2025 carry-forward', 'Provenance Flag', '39 Ports (92,052 rows)', 'Null if unmapped'),
        'bts_baseline_year': ('Reporting year of BTS baseline', 'Provenance Flag', '39 Ports (92,052 rows)', 'Null if unmapped'),
        'bts_annual_container_teu_baseline': ('Annual container throughput baseline (TEU)', 'BTS Port Catalog', '39 Ports (92,052 rows)', 'Null if unmapped'),
        'bts_annual_vessel_call_baseline': ('Annual vessel arrivals baseline', 'BTS Port Catalog', '39 Ports (92,052 rows)', 'Null if unmapped'),
        'bts_annual_cargo_short_tons_baseline': ('Annual cargo tonnage baseline (short tons)', 'BTS Port Catalog', '39 Ports (92,052 rows)', 'Null if unmapped'),
        'bts_daily_expected_vessel_calls': ('Design operational arrivals/day (annual/365)', 'Derived Capacity', '39 Ports (92,052 rows)', 'Null if unmapped'),
        'bts_daily_expected_teu': ('Design operational TEU/day (annual/365)', 'Derived Capacity', '39 Ports (92,052 rows)', 'Null if unmapped'),
        'bts_daily_expected_short_tons': ('Design operational tons/day (annual/365)', 'Derived Capacity', '39 Ports (92,052 rows)', 'Null if unmapped'),
        'bts_annual_teu_growth_pct': ('YoY TEU capacity growth percentage', 'Derived Capacity', '39 Ports (92,052 rows)', 'Null if unmapped'),
        'bts_annual_tonnage_growth_pct': ('YoY tonnage capacity growth percentage', 'Derived Capacity', '39 Ports (92,052 rows)', 'Null if unmapped'),
        'bts_port_national_rank': ('US national tonnage rank (1-50)', 'BTS Port Catalog', '39 Ports (92,052 rows)', 'Null if unmapped'),
        'bts_capacity_pressure_ratio_14d': ('Rolling 14d calls / daily expected calls', 'Derived Bottleneck', '39 Ports (77,085 rows)', 'Null if unmapped or zero base')
    }
    
    for col in df_u.columns:
        dtype = str(df_u[col].dtype)
        desc, src, cov, null_sem = u_desc.get(col, (col, 'Derived', 'Various', 'Null if missing'))
        rows.append({
            'dataset_name': 'port_daily_unified.parquet',
            'column_name': col,
            'data_type': dtype,
            'description': desc,
            'source_provenance': src,
            'entity_coverage': cov,
            'missingness_handling': null_sem
        })
        
    # 2. sequence_metadata.parquet
    p_seq = os.path.join(BASE_DIR, "processed/temporal_sequences/sequence_metadata.parquet")
    df_s = pd.read_parquet(p_seq)
    for col in df_s.columns:
        rows.append({
            'dataset_name': 'sequence_metadata.parquet',
            'column_name': col,
            'data_type': str(df_s[col].dtype),
            'description': f'Sequence pointer index / partition label: {col}',
            'source_provenance': 'Workstream C Window Generator',
            'entity_coverage': '5,181,085 clean sequences',
            'missingness_handling': 'Never null'
        })
        
    # 3. North American Graph Nodes & Edges
    p_nan = os.path.join(BASE_DIR, "processed/graph_ready/north_american_graph_nodes.parquet")
    df_nan = pd.read_parquet(p_nan)
    for col in df_nan.columns:
        rows.append({
            'dataset_name': 'north_american_graph_nodes.parquet',
            'column_name': col,
            'data_type': str(df_nan[col].dtype),
            'description': f'Multimodal hinterland node attribute: {col}',
            'source_provenance': 'NTAD Rail / Intermodal / Canonical Ports',
            'entity_coverage': '251,646 nodes',
            'missingness_handling': 'Null if unmapped for modality'
        })
        
    p_nae = os.path.join(BASE_DIR, "processed/graph_ready/north_american_graph_edges.parquet")
    df_nae = pd.read_parquet(p_nae)
    for col in df_nae.columns:
        rows.append({
            'dataset_name': 'north_american_graph_edges.parquet',
            'column_name': col,
            'data_type': str(df_nae[col].dtype),
            'description': f'Rail and drayage connector edge attribute: {col}',
            'source_provenance': 'NTAD Rail / Intermodal Spatial Linkage',
            'entity_coverage': '303,016 edges',
            'missingness_handling': 'Null if unmapped for edge type'
        })
        
    # 4. Caribbean Graph Nodes & Edges
    p_cn = os.path.join(BASE_DIR, "processed/graph_ready/caribbean_graph_nodes.parquet")
    df_cn = pd.read_parquet(p_cn)
    for col in df_cn.columns:
        rows.append({
            'dataset_name': 'caribbean_graph_nodes.parquet',
            'column_name': col,
            'data_type': str(df_cn[col].dtype),
            'description': f'Caribbean regional maritime port attribute: {col}',
            'source_provenance': 'Caribbean Port Calls and Vessel AIS',
            'entity_coverage': '528 ports',
            'missingness_handling': 'Null if unrecorded'
        })
        
    p_ce = os.path.join(BASE_DIR, "processed/graph_ready/caribbean_graph_edges.parquet")
    df_ce = pd.read_parquet(p_ce)
    for col in df_ce.columns:
        rows.append({
            'dataset_name': 'caribbean_graph_edges.parquet',
            'column_name': col,
            'data_type': str(df_ce[col].dtype),
            'description': f'Directed empirical shipping corridor attribute: {col}',
            'source_provenance': 'Caribbean Port Calls and Vessel AIS',
            'entity_coverage': '9,969 directed lanes',
            'missingness_handling': 'Never null'
        })
        
    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "final_feature_dictionary.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"final_feature_dictionary.csv written: {len(df_out)} features documented across 6 datasets.")

if __name__ == "__main__":
    main()
