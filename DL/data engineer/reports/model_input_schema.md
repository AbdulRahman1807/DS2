# Model Input Schema & Tensor Interface Specification

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Final Data Engineering Stage  
**Artifact Path:** `DL/data/staging/supplementary_acquisition/final_handoff/model_input_schema.md`  
**Date:** 2026-11-01  
**Status:** Approved for Downstream ML/DL Consumption  

---

## 1. Feature Group Classifications

To ensure clean feature selection and prevent data leakage, all features across the unified panel are categorized into five functional groups:

```
                                  FEATURE TAXONOMY
                                  
  ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
  │ 1. DYNAMIC TEMPORAL OBSERVABLES │      │ 2. EXOGENOUS DISRUPTION SHOCKS  │
  │ • Daily vessel calls by type    │      │ • NOAA convective storms (50km) │
  │ • Import/export cargo tonnages  │      │ • NOAA marine wind & floods     │
  │ • Trailing rolling means (7/14d)│      │ • EM-DAT active disaster counts │
  │ • Calendar sine/cosine encoding │      │ • ERA5 wind, waves, barometric  │
  └─────────────────────────────────┘      └─────────────────────────────────┘
                   │                                        │
                   └───────────────────┬────────────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │ 3. STRUCTURAL CAPACITY (STATIC│
                       │ • Annual TEU/Tonnage baselines│
                       │ • Daily expected vessel calls │
                       │ • Capacity pressure ratio     │
                       │ • Port national rank          │
                       └───────────────────────────────┘
                                       │
                   ┌───────────────────┴────────────────────┐
                   ▼                                        ▼
  ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
  │ 4. SPATIAL GRAPH ADJACENCY      │      │ 5. SUPERVISED TARGET HORIZON    │
  │ • Rail track network topology   │      │ • target_portcalls_7d_future_sum│
  │ • Port-to-intermodal drayage    │      │ • target_container_7d_future_sum│
  │ • Caribbean directed feeder lane│      │ • Daily leads: lead_1d..lead_7d │
  └─────────────────────────────────┘      └─────────────────────────────────┘
```

---

## 2. Model Family Input Schemas

### 2.1 Family 1: Recurrent Neural Networks (LSTM / GRU)

* **Tensor Shape:** $\mathbf{X} \in \mathbb{R}^{B \times L \times F_{\text{dyn}}}$, $\mathbf{y} \in \mathbb{R}^{B}$
  - $B$: Batch size (e.g., 256)
  - $L$: Sequence lookback length ($L = 30$ calendar days)
  - $F_{\text{dyn}}$: Dynamic feature dimension ($F_{\text{dyn}} = 24$)
* **Input Features ($F_{\text{dyn}}$):**
  1. `portcalls` (standardized or log1p)
  2. `portcalls_container`
  3. `portcalls_tanker`
  4. `portcalls_cargo`
  5. `import_cargo`
  6. `export_cargo`
  7. `total_cargo_tonnage`
  8. `pw_calls_rolling_7d_mean`
  9. `pw_calls_rolling_14d_mean`
  10. `pw_container_rolling_7d_mean`
  11. `day_of_week` (one-hot or continuous)
  12. `is_weekend`
  13. $\sin(2\pi \cdot \text{day\_of\_year} / 365.25)$
  14. $\cos(2\pi \cdot \text{day\_of\_year} / 365.25)$
  15. `has_noaa_coverage`
  16. `noaa_storm_event_count_50km` (zero-imputed for non-US)
  17. `noaa_marine_wind_event_count`
  18. `noaa_is_storm_active_50km`
  19. `has_emdat_coverage`
  20. `emdat_active_disaster_count` (zero-imputed)
  21. `emdat_macro_disaster_active`
  22. `has_bts_coverage`
  23. `bts_daily_expected_vessel_calls`
  24. `bts_capacity_pressure_ratio_14d` (zero-imputed)
* **Target Output:** `target_portcalls_7d_future_sum` (Scalar) or multi-horizon vector $\mathbf{y} \in \mathbb{R}^{B \times 7}$ (`target_portcalls_lead_1d` to `lead_7d`).

---

### 2.2 Family 2: Temporal Fusion Transformer (TFT)

TFT partitions features into distinct architectural pathways:
* **Static Context Vectors ($\mathbf{s}_i \in \mathbb{R}^{F_{\text{stat}}}$):**
  - `portwatch_id` (Entity embedding)
  - `country_iso3` (Entity embedding)
  - `bts_port_national_rank`
  - `bts_annual_container_teu_baseline`
  - `bts_daily_expected_vessel_calls`
* **Past-Only Dynamic Inputs ($\mathbf{z}_{i, t-\tau} \in \mathbb{R}^{F_{\text{past}}}$ for $\tau \in [1, 30]$):**
  - Historical vessel calls (`portcalls`, `container_calls`, `cargo_tonnage`)
  - Historical rolling averages (`pw_calls_rolling_7d_mean`, `14d_mean`, `30d_mean`)
  - Observed weather & disaster shocks (`noaa_storm_event_count_50km`, `emdat_active_disaster_count`)
  - Dynamic capacity pressure (`bts_capacity_pressure_ratio_14d`)
* **Known Future Inputs ($\mathbf{u}_{i, t+k} \in \mathbb{R}^{F_{\text{future}}}$ for $k \in [1, 7]$):**
  - Calendar metadata: `day_of_week`, `is_weekend`, `day_of_year` sine/cosine
  - Seasonal holiday flags

---

### 2.3 Family 3: Spatio-Temporal Graph Neural Networks (ST-GNN / ST-GCN / DCRNN)

* **Spatial Node Tensor:** $\mathbf{X}_t \in \mathbb{R}^{N \times F}$ for $N$ graph nodes at each time step $t$.
* **Static Graph Adjacency Matrix:** $\mathbf{A} \in \mathbb{R}^{N \times N}$
  - **Caribbean Network:** $N = 528$ nodes, derived from `caribbean_graph_edges.parquet` (9,969 edges).
    $$\mathbf{A}_{ij} = \frac{\text{voyages\_total}_{ij}}{\sum_k \text{voyages\_total}_{ik}}$$
  - **Hinterland Network:** $N = 251,646$ nodes, derived from `north_american_graph_edges.parquet` (303,016 edges). Unweighted or inverse-distance weighted: $\mathbf{A}_{ij} = \frac{1}{\text{distance\_km}_{ij}}$.
* **Dynamic Node States:** Sliced per timestamp $t$ from `port_daily_unified.parquet` for overlapping canonical ports (`canonical_portwatch_id`).

---

### 2.4 Family 4: Tabular Gradient Boosted Trees (LightGBM / XGBoost / CatBoost)

* **Matrix Dimensions:** $\mathbf{X} \in \mathbb{R}^{M \times K}$, $\mathbf{y} \in \mathbb{R}^M$
  - $M$: Number of sequence instances ($M_{\text{train}} = 3,696,350$)
  - $K$: Tabular feature vector ($K \approx 65$ features)
* **Lagged Feature Construction:**
  - Lags: $y_t, y_{t-1}, y_{t-2}, y_{t-7}, y_{t-14}, y_{t-28}$
  - Rolling aggregations: Mean, Std, Min, Max over 7d, 14d, 30d
  - Macro shock indicators: `noaa_rolling_7d_storm_event_count`, `emdat_rolling_7d_disaster_count`
  - Structural features: `bts_capacity_pressure_ratio_14d`, `is_bts_forward_filled`
  - Coverage masks: `has_noaa_coverage`, `has_emdat_coverage`, `has_bts_coverage`

---

## 3. Recommended Preprocessing & Scaling Conventions

1. **Target Normalization:**
   - Vessel call counts exhibit heavy right tails (skewness $> 3.5$).
   - Recommended transform: Box-Cox or Log-Transform:
     $$\tilde{y} = \log(1 + y)$$
   - Downstream metrics (MAE, RMSE, WAPE) should be computed on inverted predictions: $\hat{y} = \exp(\tilde{\hat{y}}) - 1$.
2. **Missing Value Gating:**
   - **Never impute missing values with global column means** for spatially restricted features (NOAA, BTS, ERA5). Doing so corrupts non-covered ports with fictitious US weather values.
   - For neural networks: Pass the binary coverage indicators (`has_noaa_coverage`, `has_era5_coverage`, `has_bts_coverage`) alongside zero-imputed feature values. The coverage flag acts as an inductive bias gate.
   - For Tree models (LightGBM/XGBoost): Native NaN handling is preserved and automatically routes missing observations through optimal split directions.
3. **Temporal Split Enforcement:**
   - Compute all scaler parameters (e.g., `StandardScaler`, `MinMaxScaler`, `QuantileTransformer`) **strictly on the Train partition (2019–2023)**.
   - Apply fitted train scalers to Validation (2024) and Test (2025) without re-fitting to prevent test leakage.

