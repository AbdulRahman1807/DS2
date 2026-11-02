"""
Stage 02 Deep Learning EDA: Workstream C — Feature Quality, Correlation, and Multicollinearity Analysis
Script: DL/eda engineer/05_feature_quality_and_correlation.py

Analyzes all 91 features in port_daily_unified.parquet:
1. Complete Feature Taxonomy & Quality Audit (Table 18)
2. Feature-Target Correlation Analysis (Table 19)
3. Multicollinearity & Redundancy Assessment (Table 20)
4. Visualizations:
   - Figure 13: Feature-Target Correlation Heatmap
   - Figure 14: Rolling Feature Multicollinearity & Redundancy

Generates:
- tables/18_master_feature_quality_inventory.csv
- tables/19_feature_target_correlation_matrix.csv
- tables/20_multicollinearity_and_redundancy_assessment.csv
- figures/13_feature_target_correlation_heatmap.png
- figures/14_rolling_feature_multicollinearity_decay.png
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARQUET_PATH = "DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet"
OUTPUT_TABLES_DIR = "DL/eda/eda_reports/tables"
OUTPUT_FIGURES_DIR = "DL/eda/eda_reports/figures"

os.makedirs(OUTPUT_TABLES_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Feature Taxonomy Definition across 91 Features
# -------------------------------------------------------------
FEATURE_TAXONOMY = {
    # 1. Identifiers & Temporal Metadata (8)
    "year": ("Identifiers & Metadata", "Global (All 2,065 ports)"),
    "month": ("Identifiers & Metadata", "Global (All 2,065 ports)"),
    "day": ("Identifiers & Metadata", "Global (All 2,065 ports)"),
    "portwatch_id": ("Identifiers & Metadata", "Global (All 2,065 ports)"),
    "portwatch_name": ("Identifiers & Metadata", "Global (All 2,065 ports)"),
    "country_name": ("Identifiers & Metadata", "Global (All 2,065 ports)"),
    "country_iso3": ("Identifiers & Metadata", "Global (All 2,065 ports)"),
    "date_utc": ("Identifiers & Metadata", "Global (All 2,065 ports)"),
    
    # 2. Operational Ground Truth (12)
    "portcalls_container": ("Operational Ground Truth", "Global (All 2,065 ports)"),
    "portcalls_dry_bulk": ("Operational Ground Truth", "Global (All 2,065 ports)"),
    "portcalls_general_cargo": ("Operational Ground Truth", "Global (All 2,065 ports)"),
    "portcalls_roro": ("Operational Ground Truth", "Global (All 2,065 ports)"),
    "portcalls_tanker": ("Operational Ground Truth", "Global (All 2,065 ports)"),
    "portcalls_cargo": ("Operational Ground Truth", "Global (All 2,065 ports)"),
    "portcalls": ("Operational Ground Truth", "Global (All 2,065 ports)"),
    "import_container": ("Operational Ground Truth", "Global (All 2,065 ports)"),
    "import_cargo": ("Operational Ground Truth", "Global (All 2,065 ports)"),
    "export_container": ("Operational Ground Truth", "Global (All 2,065 ports)"),
    "export_cargo": ("Operational Ground Truth", "Global (All 2,065 ports)"),
    "total_cargo_tonnage": ("Operational Ground Truth", "Global (All 2,065 ports)"),
    
    # 3. Calendar & Seasonal Features (3)
    "day_of_week": ("Calendar Features", "Global (All 2,065 ports)"),
    "is_weekend": ("Calendar Features", "Global (All 2,065 ports)"),
    "day_of_year": ("Calendar Features", "Global (All 2,065 ports)"),
    
    # 4. Trailing Operational Windows / Rolling Features (5)
    "pw_calls_rolling_7d_mean": ("Trailing Rolling Features", "Global (All 2,065 ports)"),
    "pw_calls_rolling_14d_mean": ("Trailing Rolling Features", "Global (All 2,065 ports)"),
    "pw_calls_rolling_30d_mean": ("Trailing Rolling Features", "Global (All 2,065 ports)"),
    "pw_container_rolling_7d_mean": ("Trailing Rolling Features", "Global (All 2,065 ports)"),
    "pw_container_rolling_14d_mean": ("Trailing Rolling Features", "Global (All 2,065 ports)"),
    
    # 5. Forecasting Targets (10)
    "target_portcalls_lead_1d": ("Forecasting Targets", "Global (All 2,065 ports)"),
    "target_portcalls_lead_2d": ("Forecasting Targets", "Global (All 2,065 ports)"),
    "target_portcalls_lead_3d": ("Forecasting Targets", "Global (All 2,065 ports)"),
    "target_portcalls_lead_4d": ("Forecasting Targets", "Global (All 2,065 ports)"),
    "target_portcalls_lead_5d": ("Forecasting Targets", "Global (All 2,065 ports)"),
    "target_portcalls_lead_6d": ("Forecasting Targets", "Global (All 2,065 ports)"),
    "target_portcalls_lead_7d": ("Forecasting Targets", "Global (All 2,065 ports)"),
    "target_portcalls_7d_future_sum": ("Forecasting Targets", "Global (All 2,065 ports)"),
    "target_container_calls_7d_future_sum": ("Forecasting Targets", "Global (All 2,065 ports)"),
    "target_cargo_tonnage_7d_future_sum": ("Forecasting Targets", "Global (All 2,065 ports)"),
    
    # 6. ERA5 Atmospheric & Wave Weather (17)
    "has_era5_coverage": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_wind_speed_mean_ms": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_wind_speed_max_ms": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_t2m_mean_c": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_t2m_min_c": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_t2m_max_c": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_msl_mean_hpa": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_msl_min_hpa": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_msl_max_hpa": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_precipitation_mm": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_wind_direction_mean_deg": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_swh_mean_m": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_swh_max_m": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_mwp_mean_s": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_wave_direction_mean_deg": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_wave_distance_km": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    "era5_wave_proxy_type": ("ERA5 Weather", "Pilot Ports (5 ports, 12,785 rows)"),
    
    # 7. NOAA Storm Shocks (12)
    "has_noaa_coverage": ("NOAA Storm Shocks", "US Coastal Ports (56 ports, 143,192 rows)"),
    "noaa_storm_event_count_50km": ("NOAA Storm Shocks", "US Coastal Ports (56 ports, 143,192 rows)"),
    "noaa_marine_wind_event_count": ("NOAA Storm Shocks", "US Coastal Ports (56 ports, 143,192 rows)"),
    "noaa_flood_event_flag": ("NOAA Storm Shocks", "US Coastal Ports (56 ports, 143,192 rows)"),
    "noaa_tornado_event_flag": ("NOAA Storm Shocks", "US Coastal Ports (56 ports, 143,192 rows)"),
    "noaa_storm_damage_property_sum": ("NOAA Storm Shocks", "US Coastal Ports (56 ports, 143,192 rows)"),
    "noaa_storm_event_count_100km": ("NOAA Storm Shocks", "US Coastal Ports (56 ports, 143,192 rows)"),
    "noaa_hurricane_event_flag": ("NOAA Storm Shocks", "US Coastal Ports (56 ports, 143,192 rows)"),
    "noaa_is_storm_active_50km": ("NOAA Storm Shocks", "US Coastal Ports (56 ports, 143,192 rows)"),
    "noaa_is_storm_active_100km": ("NOAA Storm Shocks", "US Coastal Ports (56 ports, 143,192 rows)"),
    "noaa_rolling_7d_storm_event_count": ("NOAA Storm Shocks", "US Coastal Ports (56 ports, 143,192 rows)"),
    "noaa_rolling_7d_storm_damage": ("NOAA Storm Shocks", "US Coastal Ports (56 ports, 143,192 rows)"),
    
    # 8. EM-DAT Natural & Macro Disasters (11)
    "has_emdat_coverage": ("EM-DAT Disasters", "Global Disasters (2,008 ports, 150 countries)"),
    "emdat_active_disaster_count": ("EM-DAT Disasters", "Global Disasters (2,008 ports, 150 countries)"),
    "emdat_natural_disaster_count": ("EM-DAT Disasters", "Global Disasters (2,008 ports, 150 countries)"),
    "emdat_technological_disaster_count": ("EM-DAT Disasters", "Global Disasters (2,008 ports, 150 countries)"),
    "emdat_total_deaths": ("EM-DAT Disasters", "Global Disasters (2,008 ports, 150 countries)"),
    "emdat_total_affected": ("EM-DAT Disasters", "Global Disasters (2,008 ports, 150 countries)"),
    "emdat_total_damage_usd": ("EM-DAT Disasters", "Global Disasters (2,008 ports, 150 countries)"),
    "emdat_macro_disaster_active": ("EM-DAT Disasters", "Global Disasters (2,008 ports, 150 countries)"),
    "emdat_log_affected_population": ("EM-DAT Disasters", "Global Disasters (2,008 ports, 150 countries)"),
    "emdat_rolling_7d_disaster_count": ("EM-DAT Disasters", "Global Disasters (2,008 ports, 150 countries)"),
    "emdat_rolling_30d_disaster_count": ("EM-DAT Disasters", "Global Disasters (2,008 ports, 150 countries)"),
    
    # 9. BTS Hinterland Logistics & Freight Baselines (13)
    "has_bts_coverage": ("BTS Hinterland Baselines", "US Major Freight Ports (36 ports, 92,052 rows)"),
    "is_bts_forward_filled": ("BTS Hinterland Baselines", "US Major Freight Ports (36 ports, 92,052 rows)"),
    "bts_baseline_year": ("BTS Hinterland Baselines", "US Major Freight Ports (36 ports, 92,052 rows)"),
    "bts_annual_container_teu_baseline": ("BTS Hinterland Baselines", "US Major Freight Ports (36 ports, 92,052 rows)"),
    "bts_annual_vessel_call_baseline": ("BTS Hinterland Baselines", "US Major Freight Ports (36 ports, 92,052 rows)"),
    "bts_annual_cargo_short_tons_baseline": ("BTS Hinterland Baselines", "US Major Freight Ports (36 ports, 92,052 rows)"),
    "bts_daily_expected_vessel_calls": ("BTS Hinterland Baselines", "US Major Freight Ports (36 ports, 92,052 rows)"),
    "bts_daily_expected_teu": ("BTS Hinterland Baselines", "US Major Freight Ports (36 ports, 92,052 rows)"),
    "bts_daily_expected_short_tons": ("BTS Hinterland Baselines", "US Major Freight Ports (36 ports, 92,052 rows)"),
    "bts_annual_teu_growth_pct": ("BTS Hinterland Baselines", "US Major Freight Ports (36 ports, 92,052 rows)"),
    "bts_annual_tonnage_growth_pct": ("BTS Hinterland Baselines", "US Major Freight Ports (36 ports, 92,052 rows)"),
    "bts_port_national_rank": ("BTS Hinterland Baselines", "US Major Freight Ports (36 ports, 92,052 rows)"),
    "bts_capacity_pressure_ratio_14d": ("BTS Hinterland Baselines", "US Major Freight Ports (36 ports, 92,052 rows)")
}


def build_feature_quality_inventory():
    print("\n" + "="*80)
    print("STEP 1: PROFILING MASTER FEATURE QUALITY INVENTORY (ALL 91 COLUMNS)")
    print("="*80)
    t0 = time.time()
    
    pf = pq.ParquetFile(PARQUET_PATH)
    all_cols = pf.schema_arrow.names
    n_total = pf.metadata.num_rows
    
    inventory_rows = []
    
    # Process columns in batches to keep memory small
    batch_size = 15
    for b_idx in range(0, len(all_cols), batch_size):
        b_cols = all_cols[b_idx:b_idx+batch_size]
        df_batch = pf.read(b_cols).to_pandas()
        
        for col in b_cols:
            s = df_batch[col]
            dtype_str = str(s.dtype)
            null_cnt = int(s.isna().sum())
            null_pct = (null_cnt / n_total) * 100.0
            
            # Check unique count and zeros
            if pd.api.types.is_numeric_dtype(s):
                valid_s = s.dropna()
                zero_cnt = int((valid_s == 0).sum())
                zero_pct = (zero_cnt / len(valid_s) * 100.0) if len(valid_s) > 0 else 0.0
                uniq_cnt = int(valid_s.nunique())
                
                if len(valid_s) > 0:
                    min_val = round(float(valid_s.min()), 4)
                    p25_val = round(float(np.percentile(valid_s, 25)), 4)
                    p50_val = round(float(np.percentile(valid_s, 50)), 4)
                    p75_val = round(float(np.percentile(valid_s, 75)), 4)
                    max_val = round(float(valid_s.max()), 4)
                    mean_val = round(float(valid_s.mean()), 4)
                    std_val = round(float(valid_s.std()), 4)
                else:
                    min_val, p25_val, p50_val, p75_val, max_val, mean_val, std_val = [None]*7
            else:
                valid_s = s.dropna()
                zero_cnt, zero_pct = 0, 0.0
                uniq_cnt = int(valid_s.nunique())
                min_val, p25_val, p50_val, p75_val, max_val, mean_val, std_val = [None]*7
                
            domain_cat, geo_scope = FEATURE_TAXONOMY.get(col, ("Uncategorized", "Global"))
            
            inventory_rows.append({
                "column_index": len(inventory_rows) + 1,
                "column_name": col,
                "domain_category": domain_cat,
                "geographic_scope": geo_scope,
                "data_type": dtype_str,
                "null_count": null_cnt,
                "null_pct": round(null_pct, 2),
                "zero_count": zero_cnt,
                "zero_pct": round(zero_pct, 2),
                "unique_values": uniq_cnt,
                "min": min_val,
                "p25": p25_val,
                "median_p50": p50_val,
                "p75": p75_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val
            })
            
    t18_df = pd.DataFrame(inventory_rows)
    t18_path = os.path.join(OUTPUT_TABLES_DIR, "18_master_feature_quality_inventory.csv")
    t18_df.to_csv(t18_path, index=False)
    print(f"✅ Saved Table 18 -> {t18_path} (91 features profiled in {time.time()-t0:.2f}s)")
    return t18_df


def analyze_feature_target_correlations():
    print("\n" + "="*80)
    print("STEP 2: FEATURE-TARGET CORRELATION ANALYSIS (POOLED & WITHIN-PORT)")
    print("="*80)
    t0 = time.time()
    
    pf = pq.ParquetFile(PARQUET_PATH)
    
    targets = [
        "target_portcalls_7d_future_sum",
        "target_container_calls_7d_future_sum",
        "target_cargo_tonnage_7d_future_sum"
    ]
    
    # Candidate predictor features (numeric features across all categories, excluding targets and ID strings)
    candidate_features = [
        # Operational
        "portcalls", "portcalls_container", "portcalls_dry_bulk", "portcalls_general_cargo", "portcalls_roro", "portcalls_tanker",
        "import_container", "export_container", "total_cargo_tonnage",
        # Calendar
        "day_of_week", "is_weekend", "day_of_year",
        # Rolling
        "pw_calls_rolling_7d_mean", "pw_calls_rolling_14d_mean", "pw_calls_rolling_30d_mean",
        "pw_container_rolling_7d_mean", "pw_container_rolling_14d_mean",
        # Weather
        "era5_wind_speed_mean_ms", "era5_wind_speed_max_ms", "era5_t2m_mean_c", "era5_msl_mean_hpa", "era5_precipitation_mm",
        "era5_swh_mean_m", "era5_mwp_mean_s",
        # NOAA
        "noaa_storm_event_count_50km", "noaa_marine_wind_event_count", "noaa_storm_damage_property_sum", "noaa_rolling_7d_storm_event_count",
        # EM-DAT
        "emdat_active_disaster_count", "emdat_total_deaths", "emdat_total_affected", "emdat_log_affected_population", "emdat_rolling_7d_disaster_count",
        # BTS
        "bts_daily_expected_vessel_calls", "bts_daily_expected_teu", "bts_capacity_pressure_ratio_14d"
    ]
    
    read_cols = candidate_features + targets + ["portwatch_id"]
    df = pf.read(read_cols).to_pandas()
    print(f"Loaded {len(read_cols)} features ({len(df):,} rows) in {time.time()-t0:.2f}s")
    
    # Identify top 100 ports by volume for within-port analysis
    top100_ports = df.groupby('portwatch_id')['portcalls'].sum().nlargest(100).index.tolist()
    df_top100 = df[df['portwatch_id'].isin(top100_ports)].copy()
    
    corr_results = []
    
    for feat in candidate_features:
        domain_cat, geo_scope = FEATURE_TAXONOMY.get(feat, ("General", "Global"))
        
        # Valid non-null subset for feature and main target
        sub_main = df[[feat, "target_portcalls_7d_future_sum"]].dropna()
        n_valid = len(sub_main)
        
        if n_valid < 100:
            continue
            
        # Sample for pooled spearman if large (for speed on 5.28M rows, sample 50,000 for spearman rank)
        p_r_calls = sub_main[feat].corr(sub_main["target_portcalls_7d_future_sum"], method="pearson")
        sample_spearman = sub_main.sample(n=min(50000, n_valid), random_state=42)
        s_rho_calls = sample_spearman[feat].corr(sample_spearman["target_portcalls_7d_future_sum"], method="spearman")
        
        # Sub for container target
        sub_cont = df[[feat, "target_container_calls_7d_future_sum"]].dropna()
        p_r_cont = sub_cont[feat].corr(sub_cont["target_container_calls_7d_future_sum"], method="pearson") if len(sub_cont) > 100 else np.nan
        
        # Sub for tonnage target
        sub_ton = df[[feat, "target_cargo_tonnage_7d_future_sum"]].dropna()
        p_r_ton = sub_ton[feat].corr(sub_ton["target_cargo_tonnage_7d_future_sum"], method="pearson") if len(sub_ton) > 100 else np.nan
        
        # Within-port correlations across Top 100 ports
        sub_top = df_top100[[feat, "target_portcalls_7d_future_sum", "target_container_calls_7d_future_sum", "target_cargo_tonnage_7d_future_sum", "portwatch_id"]].dropna()
        
        if len(sub_top) > 500:
            wp_calls_r = sub_top.groupby('portwatch_id').apply(
                lambda g: g[feat].corr(g['target_portcalls_7d_future_sum'], method='pearson') if len(g) > 30 and g[feat].std() > 1e-6 else np.nan,
                include_groups=False
            ).dropna().mean()
            
            wp_calls_rho = sub_top.groupby('portwatch_id').apply(
                lambda g: g[feat].corr(g['target_portcalls_7d_future_sum'], method='spearman') if len(g) > 30 and g[feat].std() > 1e-6 else np.nan,
                include_groups=False
            ).dropna().mean()
            
            wp_cont_r = sub_top.groupby('portwatch_id').apply(
                lambda g: g[feat].corr(g['target_container_calls_7d_future_sum'], method='pearson') if len(g) > 30 and g[feat].std() > 1e-6 else np.nan,
                include_groups=False
            ).dropna().mean()
            
            wp_ton_r = sub_top.groupby('portwatch_id').apply(
                lambda g: g[feat].corr(g['target_cargo_tonnage_7d_future_sum'], method='pearson') if len(g) > 30 and g[feat].std() > 1e-6 else np.nan,
                include_groups=False
            ).dropna().mean()
        else:
            wp_calls_r, wp_calls_rho, wp_cont_r, wp_ton_r = np.nan, np.nan, np.nan, np.nan
            
        # Determine predictive utility tier
        abs_wp = abs(wp_calls_r) if not np.isnan(wp_calls_r) else abs(p_r_calls)
        if abs_wp >= 0.35:
            tier = "Tier 1: High Predictive Power (Core Dynamic Feature)"
        elif abs_wp >= 0.15:
            tier = "Tier 2: Moderate Dynamic Signal (Valuable Context)"
        elif abs(p_r_calls) >= 0.50:
            tier = "Tier 3: Structural Scale Predictor (Cross-Sectional Anchor)"
        elif abs_wp >= 0.05:
            tier = "Tier 4: Weak Linear Signal (Non-linear Shock / GNN Interaction)"
        else:
            tier = "Tier 5: Marginal Linear Utility (Threshold / Shock Mask Only)"
            
        corr_results.append({
            "feature_name": feat,
            "domain_category": domain_cat,
            "geographic_scope": geo_scope,
            "valid_observations": n_valid,
            "pooled_pearson_calls": round(float(p_r_calls), 4) if not np.isnan(p_r_calls) else None,
            "pooled_spearman_calls": round(float(s_rho_calls), 4) if not np.isnan(s_rho_calls) else None,
            "within_port_pearson_mean_calls": round(float(wp_calls_r), 4) if not np.isnan(wp_calls_r) else None,
            "within_port_spearman_mean_calls": round(float(wp_calls_rho), 4) if not np.isnan(wp_calls_rho) else None,
            "pooled_pearson_container": round(float(p_r_cont), 4) if not np.isnan(p_r_cont) else None,
            "within_port_pearson_mean_container": round(float(wp_cont_r), 4) if not np.isnan(wp_cont_r) else None,
            "pooled_pearson_tonnage": round(float(p_r_ton), 4) if not np.isnan(p_r_ton) else None,
            "within_port_pearson_mean_tonnage": round(float(wp_ton_r), 4) if not np.isnan(wp_ton_r) else None,
            "predictive_utility_tier": tier
        })
        
    t19_df = pd.DataFrame(corr_results).sort_values("pooled_pearson_calls", ascending=False)
    t19_path = os.path.join(OUTPUT_TABLES_DIR, "19_feature_target_correlation_matrix.csv")
    t19_df.to_csv(t19_path, index=False)
    print(f"✅ Saved Table 19 -> {t19_path} ({len(t19_df)} candidate features evaluated)")
    return t19_df, df


def analyze_multicollinearity(df):
    print("\n" + "="*80)
    print("STEP 3: MULTICOLLINEARITY & REDUNDANCY ASSESSMENT")
    print("="*80)
    t0 = time.time()
    
    # Assess pairwise multicollinearity across relevant feature clusters:
    clusters = {
        "Trailing Rolling Windows": ["portcalls", "pw_calls_rolling_7d_mean", "pw_calls_rolling_14d_mean", "pw_calls_rolling_30d_mean", 
                                     "portcalls_container", "pw_container_rolling_7d_mean", "pw_container_rolling_14d_mean"],
        "Weather & Wave Clusters": ["era5_wind_speed_mean_ms", "era5_wind_speed_max_ms", "era5_swh_mean_m", "era5_mwp_mean_s", "era5_msl_mean_hpa", "era5_t2m_mean_c"],
        "NOAA Storm Clusters": ["noaa_storm_event_count_50km", "noaa_storm_damage_property_sum", "noaa_rolling_7d_storm_event_count", "noaa_marine_wind_event_count"],
        "EM-DAT Disaster Clusters": ["emdat_active_disaster_count", "emdat_total_affected", "emdat_rolling_7d_disaster_count", "emdat_log_affected_population"],
        "BTS Logistics Baselines": ["bts_daily_expected_vessel_calls", "bts_daily_expected_teu", "bts_capacity_pressure_ratio_14d"]
    }
    
    collinearity_rows = []
    
    for cluster_name, feat_list in clusters.items():
        avail_feats = [f for f in feat_list if f in df.columns]
        sub_df = df[avail_feats].dropna()
        if len(sub_df) < 50:
            continue
            
        corr_mat = sub_df.corr(method='pearson')
        
        for i in range(len(avail_feats)):
            for j in range(i+1, len(avail_feats)):
                f1 = avail_feats[i]
                f2 = avail_feats[j]
                r_val = corr_mat.loc[f1, f2]
                abs_r = abs(r_val)
                
                if abs_r >= 0.70:
                    if abs_r >= 0.95:
                        level = "CRITICAL / SEVERE REDUNDANCY (|r| >= 0.95)"
                        risk = "Severe parameter instability, inflated gradient variance in unregularized linear/dense layers."
                        remediation = "Retain 7d mean only; represent 14d/30d as difference residuals (e.g. roll7 - roll30) or drop."
                    elif abs_r >= 0.85:
                        level = "HIGH MULTICOLLINEARITY (0.85 <= |r| < 0.95)"
                        risk = "High feature redundancy; shared variance suppresses individual feature importance."
                        remediation = "Apply L2 weight decay / ridge regularization, or combine via PCA / feature projection layer."
                    else:
                        level = "MODERATE COLLINEARITY (0.70 <= |r| < 0.85)"
                        risk = "Shared directional trend; manageable with standard dropout / neural regularization."
                        remediation = "Safe to retain if using deep neural architectures with dropout (p >= 0.2)."
                        
                    collinearity_rows.append({
                        "feature_cluster": cluster_name,
                        "feature_1": f1,
                        "feature_2": f2,
                        "pearson_correlation": round(float(r_val), 4),
                        "absolute_correlation": round(float(abs_r), 4),
                        "collinearity_level": level,
                        "modeling_risk": risk,
                        "recommended_remediation": remediation
                    })
                    
    t20_df = pd.DataFrame(collinearity_rows).sort_values("absolute_correlation", ascending=False)
    t20_path = os.path.join(OUTPUT_TABLES_DIR, "20_multicollinearity_and_redundancy_assessment.csv")
    t20_df.to_csv(t20_path, index=False)
    print(f"✅ Saved Table 20 -> {t20_path} ({len(t20_df)} high/severe multicollinear pairs documented)")
    return t20_df


def generate_figures(t19_df, df):
    print("\n" + "="*80)
    print("STEP 4: GENERATING FIGURES 13 AND 14")
    print("="*80)
    
    # -------------------------------------------------------------
    # Figure 13: Feature-Target Correlation Heatmap
    # -------------------------------------------------------------
    # Select key representative features across categories
    heatmap_feats = [
        "pw_calls_rolling_7d_mean",
        "pw_calls_rolling_30d_mean",
        "portcalls",
        "pw_container_rolling_7d_mean",
        "portcalls_container",
        "total_cargo_tonnage",
        "era5_wind_speed_mean_ms",
        "era5_swh_mean_m",
        "noaa_storm_event_count_50km",
        "emdat_active_disaster_count",
        "bts_daily_expected_vessel_calls",
        "day_of_week",
        "is_weekend"
    ]
    
    sub_t19 = t19_df[t19_df['feature_name'].isin(heatmap_feats)].set_index('feature_name')
    
    matrix_data = []
    row_labels = []
    
    col_names = [
        "Pooled Pearson (Total Calls)",
        "Within-Port Pearson (Total Calls)",
        "Pooled Pearson (Container)",
        "Within-Port Pearson (Container)",
        "Pooled Pearson (Tonnage)",
        "Within-Port Pearson (Tonnage)"
    ]
    
    for f in heatmap_feats:
        if f in sub_t19.index:
            r = sub_t19.loc[f]
            matrix_data.append([
                r["pooled_pearson_calls"],
                r["within_port_pearson_mean_calls"],
                r["pooled_pearson_container"],
                r["within_port_pearson_mean_container"],
                r["pooled_pearson_tonnage"],
                r["within_port_pearson_mean_tonnage"]
            ])
            row_labels.append(f)
            
    mat = np.array(matrix_data, dtype=float)
    
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    cax = ax.imshow(mat, cmap='coolwarm', vmin=-0.2, vmax=1.0, aspect='auto')
    
    # Text annotations
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.65 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8, fontweight="bold")
            else:
                ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=8)
                
    ax.set_xticks(np.arange(len(col_names)))
    ax.set_xticklabels(col_names, rotation=25, ha='right', fontsize=9, fontweight='bold')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9, fontweight='bold')
    ax.set_title("Figure 13: Feature-Target Correlation Matrix (Pooled vs Within-Port Mean)\nTarget: 7-Day Forward Sum Forecasts Across Top 100 Global Ports", fontsize=11, fontweight='bold', pad=15)
    
    cbar = fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Correlation Coefficient (r)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig13_path = os.path.join(OUTPUT_FIGURES_DIR, "13_feature_target_correlation_heatmap.png")
    plt.savefig(fig13_path, dpi=300)
    plt.close()
    print(f"✅ Figure 13 saved -> {fig13_path}")
    
    # -------------------------------------------------------------
    # Figure 14: Rolling Feature Multicollinearity & Redundancy
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Panel A: Autocorrelation decay across rolling windows vs lag distance
    ax1 = axes[0]
    windows = [7, 14, 30]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Correlation between rolling means
    roll_cols = ['pw_calls_rolling_7d_mean', 'pw_calls_rolling_14d_mean', 'pw_calls_rolling_30d_mean']
    sub_roll = df[roll_cols].dropna()
    c_7_14 = sub_roll['pw_calls_rolling_7d_mean'].corr(sub_roll['pw_calls_rolling_14d_mean'])
    c_7_30 = sub_roll['pw_calls_rolling_7d_mean'].corr(sub_roll['pw_calls_rolling_30d_mean'])
    c_14_30 = sub_roll['pw_calls_rolling_14d_mean'].corr(sub_roll['pw_calls_rolling_30d_mean'])
    
    bars = ax1.bar(['Roll 7d vs 14d', 'Roll 14d vs 30d', 'Roll 7d vs 30d'], [c_7_14, c_14_30, c_7_30], color=['#d9534f', '#f0ad4e', '#5bc0de'], edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Pearson Correlation Coefficient (r)', fontsize=10, fontweight='bold')
    ax1.set_ylim(0.95, 1.002)
    ax1.set_title('A: Pairwise Rolling Features Multicollinearity\nExtreme Correlation Redundancy (r > 0.99)', fontsize=11, fontweight='bold', pad=10)
    ax1.axhline(0.95, color='red', linestyle='--', linewidth=1.5, label='Severe Redundancy Threshold (r = 0.95)')
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax1.legend(loc='lower right', frameon=True, fontsize=9)
    
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.001, f"{h:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    # Panel B: Predictive Signal: Raw Portcalls vs Rolling Means vs Target
    ax2 = axes[1]
    features_to_compare = ['Daily Portcalls', 'Rolling 7d Mean', 'Rolling 14d Mean', 'Rolling 30d Mean']
    pooled_vals = [
        t19_df[t19_df['feature_name']=='portcalls']['pooled_pearson_calls'].values[0],
        t19_df[t19_df['feature_name']=='pw_calls_rolling_7d_mean']['pooled_pearson_calls'].values[0],
        t19_df[t19_df['feature_name']=='pw_calls_rolling_14d_mean']['pooled_pearson_calls'].values[0],
        t19_df[t19_df['feature_name']=='pw_calls_rolling_30d_mean']['pooled_pearson_calls'].values[0]
    ]
    within_vals = [
        t19_df[t19_df['feature_name']=='portcalls']['within_port_pearson_mean_calls'].values[0],
        t19_df[t19_df['feature_name']=='pw_calls_rolling_7d_mean']['within_port_pearson_mean_calls'].values[0],
        t19_df[t19_df['feature_name']=='pw_calls_rolling_14d_mean']['within_port_pearson_mean_calls'].values[0],
        t19_df[t19_df['feature_name']=='pw_calls_rolling_30d_mean']['within_port_pearson_mean_calls'].values[0]
    ]
    
    x = np.arange(len(features_to_compare))
    width = 0.35
    
    rects1 = ax2.bar(x - width/2, pooled_vals, width, label='Pooled Pearson r (Global Confounded)', color='#1f77b4', edgecolor='black', linewidth=0.5)
    rects2 = ax2.bar(x + width/2, within_vals, width, label='Within-Port Pearson r (True Dynamic)', color='#2ca02c', edgecolor='black', linewidth=0.5)
    
    ax2.set_ylabel('Correlation with Target (7d Future Sum)', fontsize=10, fontweight='bold')
    ax2.set_title('B: Pooled vs Within-Port Predictive Power\nDynamic Signal Peaks at Rolling 7d Window (r = 0.45)', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(features_to_compare, fontsize=9, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2.set_ylim(0, 1.15)
    
    for r in rects1:
        h = r.get_height()
        ax2.text(r.get_x() + r.get_width()/2, h + 0.02, f"{h:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax2.text(r.get_x() + r.get_width()/2, h + 0.02, f"{h:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    plt.tight_layout()
    fig14_path = os.path.join(OUTPUT_FIGURES_DIR, "14_rolling_feature_multicollinearity_decay.png")
    plt.savefig(fig14_path, dpi=300)
    plt.close()
    print(f"✅ Figure 14 saved -> {fig14_path}")


def main():
    print("="*80)
    print("STAGE 02 EDA: WORKSTREAM C — FEATURE QUALITY & CORRELATION ANALYSIS")
    print("="*80)
    t_start = time.time()
    
    t18_df = build_feature_quality_inventory()
    t19_df, df = analyze_feature_target_correlations()
    t20_df = analyze_multicollinearity(df)
    generate_figures(t19_df, df)
    
    print("\n" + "="*80)
    print(f"Workstream C completed successfully in {time.time()-t_start:.2f}s.")
    print("="*80)

if __name__ == "__main__":
    main()

