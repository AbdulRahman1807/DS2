# ML Engineer Phase — Deep Learning Model Development

**Phase:** Phase 2 — Machine Learning & Deep Learning Model Development  
**Role:** Senior ML / DL Engineer  
**Status:** **AWAITING EDA COMPLETION & DRIFT ANALYSIS**  

---

## 1. Scope & Objective

The ML Engineer is responsible for designing, training, and benchmarking model families to predict global supply chain bottleneck shocks across the 2,065 ports.

### Assigned Model Families:
1. **Stage 01 Baselines:**
   - Persistence & Trailing Rolling Mean benchmarks.
   - Tabular Gradient Boosted Decision Trees (LightGBM, CatBoost, XGBoost) using lagged feature vectors.
2. **Stage 02 Recurrent Sequence Models:**
   - Bidirectional LSTM / GRU networks consuming 30-day lookback sequences ($L=30$).
3. **Stage 03 Attention & Temporal Transformers:**
   - Temporal Fusion Transformer (TFT), PatchTST, and Informer architectures.
4. **Stage 04 Spatio-Temporal Graph Neural Networks (ST-GNN):**
   - Spatio-Temporal Graph Convolutional Networks (ST-GCN) and Graph Attention Networks (GAT) propagating bottleneck signals along the Caribbean maritime network and North American hinterland rail network.

---

## 2. Input Dependencies

All models must consume the finalized data engineering deliverables:
* Primary Temporal Panel: `DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet`
* Sliding Sequence Index: `DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet`
* Split Partitions: Train (2019–2023), Validation (2024), Test (2025) strictly quarantined.
* Spatial Graphs: `DL/data/staging/supplementary_acquisition/processed/graph_ready/`

