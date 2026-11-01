# Phase 2 — Dataset Manifest

## Purpose

This document defines the source datasets approved for Phase 2 deep
learning work. Phase 2 data are kept separate from Phase 1 ML data.

No dataset should be treated as evidence of broader geographic,
temporal, or operational coverage than its actual observations support.

## 1. Primary temporal datasets

### 1.1 IMF PortWatch — Daily Ports Data

- File: `data/raw/portwatch/Daily_Ports_Data.csv`
- Source: IMF PortWatch
- Granularity: Daily, port-level
- Coverage: 2,065 ports across 180 countries
- Period: 2019-01-01 to 2026-07-24
- Records: 5,703,530
- Role: Primary daily maritime activity and cargo-volume time series
- Potential use:
  - Daily port activity forecasting
  - Cargo-volume forecasting
  - Port-level temporal feature extraction
  - Maritime activity signals for network analysis

Limitations:
- AIS-derived estimates, not customs-verified throughput.
- Does not directly measure container dwell time, vessel waiting
  time, queue length, or berth occupancy.
- Does not provide rail, road, or warehouse operational observations.
- The downloaded snapshot ends on 2026-07-24; it is not a live feed.
- Zero values must be preserved unless a documented data-quality
  rule establishes otherwise.

### 1.2 AIS-derived monthly vessel dwell dataset

Files:
- `data/raw/ais_dwell/monthly_dwell_2009_2014.csv`
- `data/raw/ais_dwell/monthly_dwell_2015_2025.csv`

- Source: Zenodo-derived AIS port congestion dataset
- Granularity: Monthly, port-complex level
- Geographic coverage: Five US port complexes
  - Houston
  - Los Angeles / Long Beach
  - New York / New Jersey
  - Savannah
  - Seattle
- Period: January 2009 to December 2025
- Records: 1,016 across the two files
- Role: Primary monthly vessel dwell/congestion target source

Potential use:
- Monthly vessel dwell forecasting
- Historical congestion analysis
- Port-level target variables for aligned PortWatch features

Limitations:
- Five US port complexes only; not global coverage.
- Vessel dwell is not equivalent to container dwell.
- Monthly observations provide a relatively small sample for
  deep sequence models.
- One anomalous LA/Long Beach observation in June 2009 requires
  explicit review before modelling.
- Four missing port-months occur in the 2009–2014 period.

### 1.3 Derived AIS analytical dataset

- File: `data/raw/ais_dwell/analysis_dataset_dwell.csv`
- Granularity: Monthly
- Period: January 2001 to September 2025
- Role: Supplementary analytical and macroeconomic context

Limitations:
- Dwell observations begin in January 2009.
- Dwell variables overlap with the LA/Long Beach AIS series.
- This is a derived analytical dataset, not an independent AIS source.
- Do not count it as an independent dataset when evaluating
  sample size or evidence.

## 2. Supplementary operational dataset

### 2.1 Mendeley container event log

- File: `data/raw/mendeley/Dataset of Dwelling Time Process Fix.xlsx`
- Granularity: Container activity events
- Records: 952,069 events
- Cases: 95,648 container cases
- Activities: 20 activity types
- Period: July–September 2021
- Geographic coverage: One terminal
- Role: Supplementary operational event-sequence dataset

Potential use:
- Event-sequence reconstruction
- Activity-duration analysis
- Container-level dwell analysis, subject to milestone definition

Limitations:
- One terminal and a short observation period.
- Activity duration is not automatically total container dwell.
- 54 records contain spreadsheet error values.
- Six timestamps appear corrupted into 1954.
- Event-specific missing descriptive fields require care.

## 3. Geographic reference datasets

### 3.1 UN/LOCODE

Files:
- `data/reference/unlocode/UNLOCODE CodeListPart1.csv`
- `data/reference/unlocode/UNLOCODE CodeListPart2.csv`
- `data/reference/unlocode/UNLOCODE CodeListPart3.csv`

- Role: Global location and transport-function reference
- Raw records: 116,533
- Location records: 116,232
- Countries: 249
- Port-function locations: 17,600
- Port-function locations with coordinates: 11,833

Limitations:
- A location code is not necessarily an active terminal.
- Missing coordinates must remain explicit.
- Function codes must be interpreted positionally.
- Duplicate or updated records must not be blindly discarded.
- UN/LOCODE locations do not establish network connectivity.

### 3.2 Upply seaports

- File: `data/reference/upply/UPPLY-SEAPORTS.csv`
- Records: 14,269
- Role: Supplementary port-coordinate reference

Limitations:
- Coordinates are reference points, not berth or terminal coordinates.
- 12,255 records matched UN/LOCODE by country and location code
  in the audited edition.
- Estonia and Sweden are excluded from the planned integration,
  but their source records remain preserved.
- Country-code disagreements must be retained and documented,
  not silently overwritten.

## 4. Approved PortWatch–AIS alignment

Five provisional port mappings:

| AIS port complex | PortWatch ID | PortWatch name |
|---|---|---|
| Los Angeles / Long Beach | port664 | Los Angeles-Long Beach |
| New York / New Jersey | port815 | New York-New Jersey |
| Houston | port481 | Houston (US-TX) |
| Savannah | port1170 | Savannah |
| Seattle | port1175 | Seattle |

The common period is January 2019 through December 2025.

The mappings are provisional entity alignments based on matching
port identities. They do not prove identical geographic boundaries
or measurement methodology.

Planned integration:
- Aggregate daily PortWatch observations to monthly port-level
  features for the five mapped ports.
- Join to monthly AIS dwell targets by port and month.
- Preserve source values, source identifiers, and provenance.
- Do not create dwell targets for months without AIS observations.
- Keep the integrated table separate from the source datasets.

The overlap contains at most 420 port-month observations
(5 ports × 84 months). This is a small dataset for deep learning.

## 5. Modelling and reporting constraints

1. Do not mix Phase 1 ML datasets with Phase 2 DL datasets.
2. Do not describe five-port AIS dwell observations as global dwell
   coverage.
3. Do not describe static global reference locations as global
   dynamic operational coverage.
4. Do not treat PortWatch cargo/activity estimates as direct dwell
   measurements.
5. Do not treat vessel dwell and container dwell as interchangeable.
6. Do not count derived analytical data as independent observations.
7. Use chronological validation for temporal prediction.
8. Prevent future information from entering historical features.
9. Preserve source provenance and document all transformations.
10. Evaluate simple baselines before claiming deep learning gains.

## 6. Overall coverage assessment

The combined sources provide:
- Broad global maritime activity observations from PortWatch.
- Long-term monthly vessel dwell observations for five US port
  complexes.
- One-terminal container activity event sequences.
- Global static location and transport-function references.

They do not provide globally distributed, dynamic, multimodal
operational observations across ports, rail hubs, roads, warehouses,
and distribution centres.

Any global network or multimodal capability must therefore distinguish
observed data from inferred, reference, or synthetic components.
