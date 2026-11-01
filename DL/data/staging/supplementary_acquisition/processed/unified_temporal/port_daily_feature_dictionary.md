# Port Daily Unified Feature Dictionary

**Dataset Path:** `DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet`  
**Dataset Format:** Apache Parquet (snappy compression)  
**Row Count:** 5,280,205  
**Column Count:** 91  
**Entity Population:** 2,065 Global Seaports  
**Temporal Extent:** 2019-01-01 to 2025-12-31 (2,557 contiguous calendar days per port)  
**Primary Key:** `(portwatch_id, date_utc)`  

---

## 1. Identifiers and Calendar Metadata

| Column Name | Data Type | Physical Domain / Units | Null Semantics | Source Provenance | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `portwatch_id` | `string` | e.g. `port1325`, `port664` | Never null | PortWatch Primary | Unique canonical PortWatch alphanumeric identifier. |
| `portwatch_name` | `string` | e.g. `Tsuruga`, `Los Angeles-Long Beach` | Never null | PortWatch Primary | Human-readable port facility name. |
| `country_name` | `string` | e.g. `Japan`, `United States` | Never null | PortWatch Primary | Sovereign host country name. |
| `country_iso3` | `string` | 3-letter ISO 3166-1 alpha-3 | Never null | PortWatch Primary | ISO-3 country code used for national-level joins. |
| `date_utc` | `string` | `YYYY-MM-DD` | Never null | PortWatch Primary | Universal Time Coordinated calendar date. |
| `year` | `int16` | 2019 – 2025 | Never null | PortWatch Primary | Calendar year. |
| `month` | `int8` | 1 – 12 | Never null | PortWatch Primary | Calendar month. |
| `day` | `int8` | 1 – 31 | Never null | PortWatch Primary | Day of month. |
| `day_of_week` | `int8` | 0 – 6 (0=Monday, 6=Sunday) | Never null | Derived Calendar | Day of the week index. |
| `is_weekend` | `int8` | {0, 1} | Never null | Derived Calendar | Binary flag: 1 if Saturday or Sunday, 0 otherwise. |
| `day_of_year` | `int16` | 1 – 366 | Never null | Derived Calendar | Julian day of year for seasonal sinusoidal encoding. |

---

## 2. Contemporaneous Port Activity (PortWatch AIS Observables)

All metrics represent observed vessel activity inside the designated port spatial geofence during day $t$.

| Column Name | Data Type | Physical Domain / Units | Null Semantics | Source Provenance | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `portcalls` | `int32` | Non-negative integer | Never null (0 = zero calls) | PortWatch AIS | Total commercial vessel calls recorded on day $t$. |
| `portcalls_container` | `int32` | Non-negative integer | Never null (0 = zero calls) | PortWatch AIS | Fully cellular container vessel calls on day $t$. |
| `portcalls_cargo` | `int32` | Non-negative integer | Never null (0 = zero calls) | PortWatch AIS | Combined dry bulk and general cargo vessel calls. |
| `portcalls_tanker` | `int32` | Non-negative integer | Never null (0 = zero calls) | PortWatch AIS | Liquid bulk / tanker vessel calls on day $t$. |
| `portcalls_dry_bulk` | `int32` | Non-negative integer | Never null (0 = zero calls) | PortWatch AIS | Dry bulk carrier calls on day $t$. |
| `portcalls_general_cargo` | `int32` | Non-negative integer | Never null (0 = zero calls) | PortWatch AIS | General / break-bulk cargo carrier calls on day $t$. |
| `portcalls_roro` | `int32` | Non-negative integer | Never null (0 = zero calls) | PortWatch AIS | Roll-on/roll-off vehicle carrier calls on day $t$. |
| `import_container` | `int32` | Non-negative integer | Never null | PortWatch AIS | Estimated inbound container activity on day $t$. |
| `export_container` | `int32` | Non-negative integer | Never null | PortWatch AIS | Estimated outbound container activity on day $t$. |
| `import_cargo` | `int32` | Metric tons / call proxy | Never null | PortWatch AIS | Estimated inbound cargo volume on day $t$. |
| `export_cargo` | `int32` | Metric tons / call proxy | Never null | PortWatch AIS | Estimated outbound cargo volume on day $t$. |
| `total_cargo_tonnage` | `float32` | Metric tons / proxy | Never null | Derived Activity | Total cargo tonnage: `import_cargo + export_cargo`. |

---

## 3. Trailing Historical Rolling Features (Strictly Backward-Looking)

Computed strictly over past window $[t-k+1, t]$ per port. Zero forward leakage.

| Column Name | Data Type | Physical Domain / Units | Null Semantics | Lineage Formula | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `pw_calls_rolling_7d_mean` | `float32` | Calls / day | Never null (min_periods=1) | $\frac{1}{7}\sum_{\tau=0}^6 \text{calls}_{t-\tau}$ | Trailing 7-day rolling mean of total port calls. |
| `pw_calls_rolling_14d_mean` | `float32` | Calls / day | Never null (min_periods=1) | $\frac{1}{14}\sum_{\tau=0}^{13} \text{calls}_{t-\tau}$ | Trailing 14-day rolling mean of total port calls. |
| `pw_calls_rolling_30d_mean` | `float32` | Calls / day | Never null (min_periods=1) | $\frac{1}{30}\sum_{\tau=0}^{29} \text{calls}_{t-\tau}$ | Trailing 30-day rolling mean of total port calls. |
| `pw_container_rolling_7d_mean` | `float32` | Container calls / day | Never null (min_periods=1) | $\frac{1}{7}\sum_{\tau=0}^6 \text{cont}_{t-\tau}$ | Trailing 7-day rolling mean of container vessel calls. |
| `pw_container_rolling_14d_mean` | `float32` | Container calls / day | Never null (min_periods=1) | $\frac{1}{14}\sum_{\tau=0}^{13} \text{cont}_{t-\tau}$ | Trailing 14-day rolling mean of container vessel calls. |

---

## 4. Forecasting Targets & Future Lead Vectors (Workstream B)

Formulated over future forecast horizon $[t+1, t+7]$. Null exclusively for the terminal 7 calendar days of 2025 ($t \in [2025\text{-}12\text{-}25, 2025\text{-}12\text{-}31]$) because future truth is outside the dataset temporal horizon.

| Column Name | Data Type | Physical Domain / Units | Null Semantics | Mathematical Formula | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `target_portcalls_lead_1d` | `float32` | Vessel calls | Null for final 1 day | $\text{portcalls}_{t+1}$ | 1-day ahead total vessel call ground truth. |
| `target_portcalls_lead_2d` | `float32` | Vessel calls | Null for final 2 days | $\text{portcalls}_{t+2}$ | 2-day ahead total vessel call ground truth. |
| `target_portcalls_lead_3d` | `float32` | Vessel calls | Null for final 3 days | $\text{portcalls}_{t+3}$ | 3-day ahead total vessel call ground truth. |
| `target_portcalls_lead_4d` | `float32` | Vessel calls | Null for final 4 days | $\text{portcalls}_{t+4}$ | 4-day ahead total vessel call ground truth. |
| `target_portcalls_lead_5d` | `float32` | Vessel calls | Null for final 5 days | $\text{portcalls}_{t+5}$ | 5-day ahead total vessel call ground truth. |
| `target_portcalls_lead_6d` | `float32` | Vessel calls | Null for final 6 days | $\text{portcalls}_{t+6}$ | 6-day ahead total vessel call ground truth. |
| `target_portcalls_lead_7d` | `float32` | Vessel calls | Null for final 7 days | $\text{portcalls}_{t+7}$ | 7-day ahead total vessel call ground truth. |
| `target_portcalls_7d_future_sum` | `float32` | Cumulative vessel calls | Null for final 7 days | $\sum_{k=1}^7 \text{portcalls}_{t+k}$ | **Primary Target:** 7-day cumulative total vessel throughput. |
| `target_container_calls_7d_future_sum` | `float32` | Cumulative container calls | Null for final 7 days | $\sum_{k=1}^7 \text{portcalls\_container}_{t+k}$ | **Secondary Target:** 7-day cumulative cellular container throughput. |
| `target_cargo_tonnage_7d_future_sum` | `float32` | Cumulative metric tons | Null for final 7 days | $\sum_{k=1}^7 \text{total\_cargo\_tonnage}_{t+k}$ | **Secondary Target:** 7-day cumulative total waterborne cargo tonnage. |

---

## 5. ERA5 Port-Level Environmental Features

Governed environmental observations for key global maritime hubs.

| Column Name | Data Type | Physical Domain / Units | Null Semantics | Source Provenance | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `has_era5_coverage` | `int8` | {0, 1} | Never null | Governance Indicator | 1 for 5 governed US gateways (`port1170`, `port1175`, `port481`, `port664`, `port815`); 0 otherwise. |
| `era5_wind_speed_mean_ms` | `float32` | $\text{m/s}$ | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily mean 10-meter wind speed. |
| `era5_wind_speed_max_ms` | `float32` | $\text{m/s}$ | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily maximum 10-meter wind speed. |
| `era5_t2m_mean_c` | `float32` | $^{\circ}\text{C}$ | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily mean 2-meter air temperature. |
| `era5_t2m_min_c` | `float32` | $^{\circ}\text{C}$ | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily minimum 2-meter air temperature. |
| `era5_t2m_max_c` | `float32` | $^{\circ}\text{C}$ | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily maximum 2-meter air temperature. |
| `era5_msl_mean_hpa` | `float32` | $\text{hPa}$ | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily mean mean sea level pressure. |
| `era5_msl_min_hpa` | `float32` | $\text{hPa}$ | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily minimum barometric pressure. |
| `era5_msl_max_hpa` | `float32` | $\text{hPa}$ | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily maximum barometric pressure. |
| `era5_precipitation_mm` | `float32` | $\text{mm/day}$ | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily total accumulated precipitation. |
| `era5_wind_direction_mean_deg` | `float32` | Degrees ($0^{\circ} - 360^{\circ}$) | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily mean wind direction from north. |
| `era5_swh_mean_m` | `float32` | Meters | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily mean significant wave height. |
| `era5_swh_max_m` | `float32` | Meters | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily maximum significant wave height. |
| `era5_mwp_mean_s` | `float32` | Seconds | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily mean wave period. |
| `era5_wave_direction_mean_deg` | `float32` | Degrees ($0^{\circ} - 360^{\circ}$) | Null if `has_era5_coverage`=0 | ERA5 Reanalysis v2 | Daily mean wave propagation direction. |
| `era5_wave_distance_km` | `float32` | Kilometers | Null if `has_era5_coverage`=0 | Spatial Governance | Geodesic distance from port node to governed wave offshore grid point. |
| `era5_wave_proxy_type` | `string` | Categorical | Null if `has_era5_coverage`=0 | Spatial Governance | `NEARBY_COASTAL_PROXY` ($\le 40\text{km}$) or `CONDITIONAL_OFFSHORE_PROXY` ($>40\text{km}$). |

---

## 6. NOAA Port-Level Disruption Features

Spatially matched high-impact severe convective, flood, and marine wind storm events from NOAA Details (2019–2025).

| Column Name | Data Type | Physical Domain / Units | Null Semantics | Source Provenance | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `has_noaa_coverage` | `int8` | {0, 1} | Never null | Spatial Match Indicator | 1 for 56 US canonical ports within NOAA spatial coverage; 0 otherwise. |
| `noaa_storm_event_count_50km` | `float32` | Event count | Null if `has_noaa_coverage`=0; 0 for zero events | NOAA Storm Events | Number of severe storm events occurring within 50 km on date $t$. |
| `noaa_storm_event_count_100km` | `float32` | Event count | Null if `has_noaa_coverage`=0; 0 for zero events | NOAA Storm Events | Number of severe storm events occurring within 100 km on date $t$. |
| `noaa_is_storm_active_50km` | `float32` | {0, 1} | Null if `has_noaa_coverage`=0 | NOAA Storm Events | Binary indicator: 1 if $\ge 1$ storm active within 50 km, else 0. |
| `noaa_is_storm_active_100km` | `float32` | {0, 1} | Null if `has_noaa_coverage`=0 | NOAA Storm Events | Binary indicator: 1 if $\ge 1$ storm active within 100 km, else 0. |
| `noaa_marine_wind_event_count` | `float32` | Event count | Null if `has_noaa_coverage`=0 | NOAA Storm Events | Count of high marine wind and gale warnings within 50 km. |
| `noaa_flood_event_flag` | `float32` | {0, 1} | Null if `has_noaa_coverage`=0 | NOAA Storm Events | Binary indicator of flash flood / coastal flood within 50 km. |
| `noaa_tornado_event_flag` | `float32` | {0, 1} | Null if `has_noaa_coverage`=0 | NOAA Storm Events | Binary indicator of tornado touchdown within 50 km. |
| `noaa_hurricane_event_flag` | `float32` | {0, 1} | Null if `has_noaa_coverage`=0 | NOAA Storm Events | Binary indicator of tropical cyclone / hurricane exposure within 100 km. |
| `noaa_storm_damage_property_sum` | `float32` | USD | Null if `has_noaa_coverage`=0 | NOAA Storm Events | Property damage attributed to matched events on date $t$. |
| `noaa_rolling_7d_storm_event_count`| `float32`| Event count | Null if `has_noaa_coverage`=0 | Trailing Exposure Window | Trailing 7-day cumulative severe storm events within 50 km. |
| `noaa_rolling_7d_storm_damage` | `float32` | USD | Null if `has_noaa_coverage`=0 | Trailing Exposure Window | Trailing 7-day cumulative property damage within 50 km. |

---

## 7. EM-DAT Country Macro Disaster Shocks

Country-level macro disaster disruptions from EM-DAT (191 countries, 2019–2025).

| Column Name | Data Type | Physical Domain / Units | Null Semantics | Source Provenance | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `has_emdat_coverage` | `int8` | {0, 1} | Never null | Sovereign Indicator | 1 for 150 covered port-host countries; 0 for 30 overseas dependencies. |
| `emdat_active_disaster_count` | `float32` | Active events | Null if `has_emdat_coverage`=0; 0 for calm days | EM-DAT Global Database | Total disaster events actively ongoing in host country on date $t$. |
| `emdat_natural_disaster_count` | `float32` | Active events | Null if `has_emdat_coverage`=0; 0 for calm days | EM-DAT Global Database | Natural disaster events (earthquake, flood, cyclone, etc.) active on date $t$. |
| `emdat_technological_disaster_count`| `float32`| Active events | Null if `has_emdat_coverage`=0; 0 for calm days | EM-DAT Global Database | Technological disasters (industrial explosions, transport wrecks) on date $t$. |
| `emdat_macro_disaster_active` | `float32` | {0, 1} | Null if `has_emdat_coverage`=0 | EM-DAT Global Database | Binary indicator: 1 if any disaster active in country, 0 otherwise. |
| `emdat_rolling_7d_disaster_count` | `float32` | Active event-days | Null if `has_emdat_coverage`=0 | Trailing Macro Window | Cumulative active disaster event-days over past 7 calendar days. |
| `emdat_rolling_30d_disaster_count`| `float32` | Active event-days | Null if `has_emdat_coverage`=0 | Trailing Macro Window | Cumulative active disaster event-days over past 30 calendar days. |
| `emdat_total_deaths` | `float32` | Fatalities | **NaN if missing/unreported** | EM-DAT Global Database | Documented disaster fatalities (never imputed as zero when missing). |
| `emdat_total_affected` | `float32` | Persons | **NaN if missing/unreported** | EM-DAT Global Database | Documented affected population (never imputed as zero when missing). |
| `emdat_total_damage_usd` | `float32` | USD | **NaN if missing/unreported** | EM-DAT Global Database | Documented direct economic damages (never imputed as zero when missing). |
| `emdat_log_affected_population` | `float32` | $\log_{10}(1 + \text{affected})$ | 0.0 if zero affected; NaN if unmeasured | Derived Macro Feature | Log-transformed affected population intensity. |

---

## 8. BTS Structural Port Capacity Features & Pressure Ratio

Annual capacity baselines and congestion pressure ratios for 39 primary US deepwater port complexes.

| Column Name | Data Type | Physical Domain / Units | Null Semantics | Source Provenance | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `has_bts_coverage` | `int8` | {0, 1} | Never null | Structural Indicator | 1 for 39 resolved US deepwater port complexes; 0 otherwise. |
| `is_bts_forward_filled` | `float32` | {0, 1} | Null if `has_bts_coverage`=0 | Provenance Indicator | 0 for 2019–2023 contemporaneous annual data; 1 for 2024–2025 carry-forward. |
| `bts_baseline_year` | `float32` | Year | Null if `has_bts_coverage`=0 | Provenance Indicator | Actual reporting year of capacity baseline (2019–2023 exact; 2023 for 2024–2025). |
| `bts_annual_container_teu_baseline`| `float32`| Twenty-Foot Equivalent Units | Null if `has_bts_coverage`=0 | BTS Port Data Catalog | Official annual container throughput baseline. |
| `bts_annual_vessel_call_baseline` | `float32` | Vessel dock arrivals | Null if `has_bts_coverage`=0 | BTS Port Data Catalog | Official annual commercial vessel arrival baseline. |
| `bts_annual_cargo_short_tons_baseline`| `float32`| Short Tons (2,000 lbs) | Null if `has_bts_coverage`=0 | BTS Port Data Catalog | Official annual waterborne commerce cargo tonnage. |
| `bts_daily_expected_vessel_calls` | `float32` | Calls / day | Null if `has_bts_coverage`=0 | Derived Capacity | Design operational daily vessel call baseline: $\frac{\text{annual calls}}{365}$. |
| `bts_daily_expected_teu` | `float32` | TEU / day | Null if `has_bts_coverage`=0 | Derived Capacity | Design operational daily container throughput: $\frac{\text{annual TEU}}{365}$. |
| `bts_daily_expected_short_tons` | `float32` | Short Tons / day | Null if `has_bts_coverage`=0 | Derived Capacity | Design operational daily tonnage baseline: $\frac{\text{annual tonnage}}{365}$. |
| `bts_annual_teu_growth_pct` | `float32` | Percentage | Null if `has_bts_coverage`=0 | Derived Capacity | Year-over-year annual TEU percentage change. |
| `bts_annual_tonnage_growth_pct` | `float32` | Percentage | Null if `has_bts_coverage`=0 | Derived Capacity | Year-over-year annual tonnage percentage change. |
| `bts_port_national_rank` | `float32` | National Rank (1 – 50) | Null if `has_bts_coverage`=0 | BTS Port Data Catalog | Official US national port ranking by total waterborne tonnage. |
| `bts_capacity_pressure_ratio_14d` | `float32` | Dimensionless Ratio | Null if unmapped / zero expected | Derived Congestion Metric | **Key Bottleneck Indicator:** $\frac{\text{pw\_calls\_rolling\_14d\_mean}}{\text{bts\_daily\_expected\_vessel\_calls}}$. Values $>1.0$ indicate operational pressure exceeding design baseline. |

