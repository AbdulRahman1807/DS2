# Downstream Migration Impact Report — Spatial Governance v1.3 Final

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Milestone:** Phase 2 Data Engineering — Final Governance Semantics  
**Date:** 2026-11-01 03:04:41 UTC  

---

## 1. Audit of Existing Scripts Consuming Legacy Fields

A comprehensive static scan across `/Users/abdul/Desktop/DS/DL/scripts/` and `/Users/abdul/Desktop/DS/ML/` was performed to identify any consumption of `candidate_is_accepted` and legacy boolean `wave_modeling_eligibility`:

### Ingestion of `candidate_is_accepted`
- `scripts/build_canonical_port_registry.py` (Line 716): Generator script that originally constructed `port_identity_candidates.csv`.
- `scripts/build_spatial_feature_governance.py` (Lines 263-270): Generator script for interim governance.
- `scripts/remediate_spatial_governance_v1_3.py`: Remediation script.
- **Downstream ML / Extraction Consumers:** **ZERO scripts** currently consume `candidate_is_accepted`. Bulk ERA5 extraction and global machine learning pipelines have not yet been initiated.

### Risk Assessment of Legacy `candidate_is_accepted`
In legacy candidate tables (v1.1 through v1.3), `candidate_is_accepted` evaluated only whether an identifier matched an accepted port record. Consequently, records with valid identities but missing or corrupted coordinates (Gary `USGYY`, Monroe `USMOI`, Delfzijl `NLDZL`) were marked `candidate_is_accepted = True`. If a downstream extraction script relied on this flag to initiate geospatial queries, it would attempt to extract weather at `NaN` coordinates or accidentally fall back to unverified municipal centroids.

**Action Taken:**
- Retained `candidate_is_accepted` solely as a deprecated legacy alias for `identity_candidate_accepted`.
- Added unambiguous authoritative boolean fields:
  - `identity_candidate_accepted`: Strictly indicates whether the identifier matches the port.
  - `coordinate_candidate_accepted`: Strictly indicates whether the candidate's coordinates are approved for extraction and modeling.

### Ingestion of Legacy `wave_modeling_eligibility`
- In earlier iterations, `wave_modeling_eligibility = True` was assigned to both Primary ports (714) and Conditional regional proxies (226).
- **Risk:** A downstream supervised model filtering `df[df['wave_modeling_eligibility'] == True]` would silently ingest distant regional proxies (displaced 40–100 km) as local port predictors without conducting ablation experiments.

**Action Taken:**
- `wave_modeling_eligibility_status` (`str`) is now the authoritative categorical field (`ELIGIBLE_PRIMARY`, `ELIGIBLE_CONDITIONAL`, `INELIGIBLE`).
- Replaced the ambiguous boolean with two non-overlapping, explicit booleans:
  - `wave_modeling_primary_eligible` (`bool`): `True` for exactly 714 ports; `False` for all others.
  - `wave_modeling_conditional_eligible` (`bool`): `True` for exactly 226 ports; `False` for all others.
- Any downstream script building baseline dwell-time models must filter exclusively on `wave_modeling_primary_eligible == True`.

---

## 2. Mandatory Downstream Migration Instructions

When authoring the upcoming global ERA5 bulk extraction and feature engineering pipelines, developers must strictly adhere to the following ingestion rules:

```python
# 1. INGESTING CANDIDATE DATA
# DO NOT USE: cand['candidate_is_accepted'] (Deprecated)
# USE INSTEAD:
valid_coords = cand[cand['coordinate_candidate_accepted'] == True]
valid_identities = cand[cand['identity_candidate_accepted'] == True]

# 2. INGESTING SURFACE EXTRACTION TARGETS
surface_targets = gov[gov['surface_extraction_eligibility'] == 'ELIGIBLE']  # Exactly 976 ports

# 3. INGESTING WAVE EXTRACTION TARGETS
wave_targets = gov[gov['wave_extraction_eligibility'].isin(['ELIGIBLE', 'ELIGIBLE_REGIONAL_TRANSIT_ONLY'])]  # Exactly 961 ports

# 4. INGESTING PRIMARY DWELL-TIME MODELING FEATURES
primary_features = gov[gov['wave_modeling_primary_eligible'] == True]  # Exactly 714 ports

# 5. INGESTING CONDITIONAL FEATURES FOR ABLATION EXPERIMENTS
# Requires explicit flag and ablation against primary baseline
conditional_features = gov[gov['wave_modeling_conditional_eligible'] == True]  # Exactly 226 ports
```