# EM-DAT Country-Level Disruption Feature Metadata

**Dataset Archetype:** Macroeconomic & Socio-Natural Disaster Shocks  
**Coverage Window:** 2019-01-01 to 2025-12-31 (2,557 daily intervals)  
**Geographic Domain:** 191 ISO3 Nations (191 countries with active disasters in project window)  
**Output Files:**
1. `emdat_country_daily_shocks.parquet`: 488387 daily country-level panel records
2. `emdat_event_normalized.parquet`: 3521 normalized disaster events

## Feature Definitions

| Feature Name | Type | Range / Format | Description |
| :--- | :---: | :---: | :--- |
| `country_iso3` | String | ISO 3166-1 alpha-3 | Country identifier compatible with PortWatch |
| `date_utc` | String | `YYYY-MM-DD` | Calendar date normalized to UTC |
| `macro_disaster_active` | Binary | 0 or 1 | Indicator if at least 1 disaster is active on date |
| `active_disaster_count` | Integer | [0, 10+] | Count of concurrent active disasters |
| `natural_disaster_count` | Integer | [0, 8+] | Count of active natural disasters |
| `technological_disaster_count` | Integer | [0, 5+] | Count of active technological/industrial disasters |
| `total_deaths` | Float | Counts / NaN | Sum of reported deaths across active events (NaN preserved) |
| `total_affected` | Float | Persons / NaN | Sum of reported affected population (NaN preserved) |
| `log_affected_population` | Float | [0.0, 9.0+] | Log10(1 + total_affected) |
| `total_damage_usd` | Float | USD ($) / NaN | Sum of estimated economic damages (NaN preserved) |
| `rolling_7d_disaster_count` | Integer | [0, 50+] | Cumulative 7-day active disaster frequency |
| `rolling_30d_disaster_count` | Integer | [0, 150+] | Cumulative 30-day active disaster frequency |

## Data Governance & Integrity Safeguards

1. **Missing Impact Handling:** In strict compliance with guidelines, missing damage, deaths, and affected counts are **not** imputed to zero. NaN is explicitly preserved to reflect unquantified impact.
2. **Date Interval Interpolation:** Missing start/end days default to month-boundary ranges and are formally flagged in `emdat_event_normalized.parquet` (`MONTH_DEFAULT`, `MONTH_END_DEFAULT`).
3. **Integration Mechanism:** Downstream models join this table to PortWatch on `(portwatch.country_iso3, portwatch.date)`.
