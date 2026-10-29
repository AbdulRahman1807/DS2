# EDA Engineer Workspace - Stage 01

## Mission
Identify noise floor variations across seasonal demand cycles, audit telemetry reporting drift, and establish empirical distributions for bottleneck thresholds.

## Key Files & Tasks
1. [`profiler.py`](file:///Users/abdul/Desktop/DS/ML/eda_engineer/profiler.py):
   - Cardinally checks (95,648 cases, 95,568 distinct containers).
   - Frequency counts of all milestones (`DISCHARGE`, `TRUCK_OUT`, `CUSTOMS_DEL`).
   - Dwell percentiles (p10, p25, p50, p75, p85, p90, p95, p99).
   - **Status:** Stub generated.

2. [`drift_analysis.py`](file:///Users/abdul/Desktop/DS/ML/eda_engineer/drift_analysis.py):
   - Analyze arrival window (2021-04-19 to 2021-09-30).
   - Flag sparse dates (< June 10) vs dense window (104 call dates).
   - Investigate 9 missing call dates in the dense window (06-15, 06-22, 06-25, 07-01, 07-08, 07-29, 09-13, 09-24, 09-29).
   - **Status:** Stub generated.

3. [`eda_runner.py`](file:///Users/abdul/Desktop/DS/ML/eda_engineer/eda_runner.py):
   - Command-line runner to generate distribution figures and report tables for the squad.
   - **Status:** Stub generated.
