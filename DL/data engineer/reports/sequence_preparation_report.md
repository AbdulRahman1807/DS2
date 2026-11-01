# Temporal Sliding-Window Sequence Preparation & Partitioning Report

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Final Data Engineering Stage  
**Pipeline:** Workstream C — Temporal Sequence Generation & Split Quarantine  
**Metadata Artifact:** `DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet`  
**Split Manifest:** `DL/data/staging/supplementary_acquisition/processed/temporal_sequences/temporal_split_manifest.csv`  
**Date:** 2026-11-01  
**Status:** Complete, Verified, Leakage-Quarantined  

---

## 1. Sequence Window Formulation & Parametric Geometry

To support recurrent neural networks (LSTM, GRU), temporal transformers (Temporal Fusion Transformer, Informer, PatchTST), and spatio-temporal graph neural networks (ST-GCN, DCRNN), the daily panel is indexed into standard sliding sequences:

* **Historical Lookback ($L$):** 30 calendar days ($[t - 29, t]$). Captures monthly cyclicities, trailing rolling volatility, and environmental shock decay.
* **Forecast Horizon ($H$):** 7 calendar days ($[t + 1, t + 7]$). Matches standard weekly maritime scheduling and intermodal drayage planning horizons.
* **Total Sequence Envelope:** 37 contiguous calendar days per sequence instance.
* **Total Candidate Instances:** 5,205,865 sliding instances.
* **Total Quarantined Clean Sequences:** **5,181,085** unique instances across all 2,065 ports (24,780 boundary transition sequences spanning across partition cutoff dates have been strictly excluded).

---

## 2. Chronological Split Manifest & Boundary Quarantine

To enforce zero future-to-past data leakage, sequence partitions are assigned based on strict temporal target boundaries:

| Split Partition | Sequence Count | Sequence Share (%) | Port Population | Input Window Span | Target Window Span | Mean 7d Calls (Std) | P50 (P90) | NOAA Pct | ERA5 Pct | EMDAT Pct | BTS Pct |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **3,696,350** | 71.34% | 2,065 | 2019-01-01 to 2023-12-24 | 2019-01-31 to 2023-12-31 | 16.87 (44.44) | 5.0 (38.0) | 2.71% | 0.24% | 97.24% | 1.74% |
| **Validation** | **743,400** | 14.35% | 2,065 | 2023-12-02 to 2024-12-24 | 2024-01-01 to 2024-12-31 | 17.43 (45.79) | 5.0 (40.0) | 2.71% | 0.24% | 97.24% | 1.74% |
| **Test** | **741,335** | 14.31% | 2,065 | 2024-12-02 to 2025-12-24 | 2025-01-01 to 2025-12-31 | 17.02 (44.70) | 5.0 (40.0) | 2.71% | 0.24% | 97.24% | 1.74% |

### Boundary Quarantine Guarantees:
1. **Train Boundary Quarantine:** The target window for all training sequences ends strictly on or before **2023-12-31** (reference date $t \le 2023\text{-}12\text{-}24$). Not a single target ground-truth observation from 2024 or 2025 enters the training set.
2. **Validation Boundary Quarantine:** The target window for validation sequences is strictly contained within **2024-01-01 to 2024-12-31**. Input windows can look back into December 2023 for autoregressive state warm-up with zero forward leakage.
3. **Test Boundary Quarantine:** The target window for test sequences is strictly contained within **2025-01-01 to 2025-12-31**. Test targets are fully quarantined until final model freezing.
4. **Transition Window Exclusion:** Sequences whose target windows span across partition boundaries (e.g., target starting on 2023-12-28 and ending on 2024-01-03) are pruned from the evaluation sets (24,780 records pruned) to prevent label contamination across folds.

---

## 3. High-Performance Indexing Architecture (Zero Memory Duplication)

Creating contiguous 3D NumPy arrays `(N, 30, Features)` in system RAM for 5,181,085 sequences would require over **35 Gigabytes of memory**, triggering immediate system crash.

Instead, we implemented a **Lean Index Pointer Architecture**:
- `port_daily_unified.parquet` stores the full balanced panel (2,065 ports $\times$ 2,557 days) indexed contiguously.
- `sequence_metadata.parquet` (only 50.77 MB) stores pointer records referencing the port ID and the exact slice boundaries.
- Downstream deep learning engineers use our high-speed PyTorch `Dataset` generator (detailed below) which slices tensors directly on the fly or pre-loads small batches in parallel workers.

---

## 4. PyTorch DataLoader Integration Guide for Downstream Engineers

Downstream model engineers can directly instantiate training batches using the following production-grade pattern:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class PortTimeSeriesDataset(Dataset):
    def __init__(self, panel_parquet_path, meta_parquet_path, split='train', feature_cols=None):
        self.meta = pd.read_parquet(meta_parquet_path)
        self.meta = self.meta[self.meta['split_tag'] == split].reset_index(drop=True)
        
        # Load panel and index by (portwatch_id, date_utc)
        df_panel = pd.read_parquet(panel_parquet_path)
        if feature_cols is None:
            # Default feature set
            feature_cols = [
                'portcalls', 'portcalls_container', 'total_cargo_tonnage',
                'pw_calls_rolling_7d_mean', 'pw_calls_rolling_14d_mean',
                'day_of_week', 'is_weekend'
            ]
        self.feature_cols = feature_cols
        
        # Build memory-efficient fast numpy matrix (ports, days, features)
        self.ports = sorted(df_panel['portwatch_id'].unique().tolist())
        self.port2idx = {p: i for i, p in enumerate(self.ports)}
        self.dates = sorted(df_panel['date_utc'].unique().tolist())
        self.date2idx = {d: i for i, d in enumerate(self.dates)}
        
        # Reshape into fast tensor [P, D, F]
        P, D, F = len(self.ports), len(self.dates), len(self.feature_cols)
        self.tensor = np.zeros((P, D, F), dtype=np.float32)
        
        # Fill tensor vectorized
        df_sorted = df_panel.sort_values(['portwatch_id', 'date_utc'])
        for f_idx, col in enumerate(self.feature_cols):
            self.tensor[:, :, f_idx] = df_sorted[col].values.reshape(P, D)
            
    def __len__(self):
        return len(self.meta)
        
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        p_idx = self.port2idx[row['portwatch_id']]
        start_d = self.date2idx[row['input_start_date']]
        ref_d = self.date2idx[row['reference_date']]
        
        # Historical lookback tensor [30, F]
        x = torch.from_numpy(self.tensor[p_idx, start_d:ref_d+1, :])
        
        # Target scalar or vector
        y = torch.tensor(row['target_portcalls_7d_future_sum'], dtype=torch.float32)
        return x, y

# Example Usage:
# train_dataset = PortTimeSeriesDataset('DL/.../port_daily_unified.parquet', 'DL/.../sequence_metadata.parquet', split='train')
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
```

