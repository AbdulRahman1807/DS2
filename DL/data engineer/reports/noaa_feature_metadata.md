# NOAA Storm Events Disruption Feature Metadata

**Dataset Archetype:** High-Impact Spatiotemporal Weather Shocks  
**Coverage Window:** 2019-01-01 to 2025-12-31 (2,557 daily intervals)  
**Geographic Domain:** United States Coastal & Deepwater Seaports (56 exposed canonical ports)  
**Output Files:**
1. `noaa_port_daily_shocks.parquet` (143192 rows, 13 features)
2. `noaa_event_port_matches.parquet` (80244 matched event records)

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
