# Downstream ML/DL Engineer Handoff Guide

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Final Data Engineering Stage  
**Artifact Path:** `DL/data/staging/supplementary_acquisition/final_handoff/downstream_engineer_handoff.md`  
**Date:** 2026-11-01  
**Status:** Ready for Immediate Model Development  

---

## 1. Quick-Start File Directory & Access Matrix

All model-ready data assets and manifests are organized under `DL/data/staging/supplementary_acquisition/`:

| Artifact Name | File Path | Format | Size | Records | Primary Usage |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Unified Daily Panel** | `processed/unified_temporal/port_daily_unified.parquet` | Parquet | 97.7 MB | 5,280,205 | Core tabular & time-series feature matrix (2,065 ports, 2019–2025). |
| **Sequence Metadata** | `processed/temporal_sequences/sequence_metadata.parquet` | Parquet | 50.8 MB | 5,181,085 | Pointer index for sliding sequences ($L=30, H=7$) with split tags. |
| **Split Manifest** | `processed/temporal_sequences/temporal_split_manifest.csv` | CSV | 1 KB | 3 splits | Statistical summary and verification of Train, Val, Test partitions. |
| **Hinterland Nodes** | `processed/graph_ready/north_american_graph_nodes.parquet` | Parquet | 3.0 MB | 251,646 | Multimodal rail, intermodal facility, and port node attributes. |
| **Hinterland Edges** | `processed/graph_ready/north_american_graph_edges.parquet` | Parquet | 11.5 MB | 303,016 | Rail track network and port-to-intermodal drayage connections. |
| **Caribbean Nodes** | `processed/graph_ready/caribbean_graph_nodes.parquet` | Parquet | 31 KB | 528 | Regional Caribbean maritime port berths and UN/LOCODE crosswalks. |
| **Caribbean Edges** | `processed/graph_ready/caribbean_graph_edges.parquet` | Parquet | 148 KB | 9,969 | Directed maritime shipping lanes with empirical voyage distributions. |
| **Feature Dictionary**| `final_handoff/final_feature_dictionary.csv` | CSV | 28 KB | 166 features | Full data dictionary across all 6 production Parquet tables. |

---

## 2. Production Code Snippets for Immediate Ingestion

### 2.1 Tabular Modeling (LightGBM / XGBoost / CatBoost)

```python
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load sequence metadata to filter splits
seq_meta = pd.read_parquet("DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet")

# 2. Load unified panel
df_panel = pd.read_parquet("DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet")

# 3. Create training dataset (Train: <= 2023-12-31)
train_ref_dates = seq_meta[seq_meta['split_tag'] == 'train']['reference_date'].unique()
val_ref_dates = seq_meta[seq_meta['split_tag'] == 'val']['reference_date'].unique()

features = [
    'portcalls', 'portcalls_container', 'total_cargo_tonnage',
    'pw_calls_rolling_7d_mean', 'pw_calls_rolling_14d_mean', 'pw_calls_rolling_30d_mean',
    'day_of_week', 'is_weekend', 'day_of_year',
    'has_noaa_coverage', 'noaa_storm_event_count_50km', 'noaa_is_storm_active_50km',
    'has_emdat_coverage', 'emdat_active_disaster_count', 'emdat_macro_disaster_active',
    'has_bts_coverage', 'bts_capacity_pressure_ratio_14d'
]
target = 'target_portcalls_7d_future_sum'

train_df = df_panel[df_panel['date_utc'].isin(train_ref_dates)]
val_df = df_panel[df_panel['date_utc'].isin(val_ref_dates)]

# Impute coverage-gated nulls safely for LightGBM
X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]

# 4. Train Baseline LightGBM Regressor
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])

preds = model.predict(X_val)
print(f"Validation MAE: {mean_absolute_error(y_val, preds):.2f}")
print(f"Validation R2:  {r2_score(y_val, preds):.4f}")
```

---

### 2.2 Deep Sequence Modeling (PyTorch Temporal DataLoader)

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class MemoryEfficientPortDataset(Dataset):
    def __init__(self, panel_path, meta_path, split='train'):
        self.meta = pd.read_parquet(meta_path)
        self.meta = self.meta[self.meta['split_tag'] == split].reset_index(drop=True)
        
        # Load panel into memory-contiguous 3D array: [Ports, Days, Features]
        df_panel = pd.read_parquet(panel_path)
        self.ports = sorted(df_panel['portwatch_id'].unique())
        self.port2idx = {p: i for i, p in enumerate(self.ports)}
        self.dates = sorted(df_panel['date_utc'].unique())
        self.date2idx = {d: i for i, d in enumerate(self.dates)}
        
        self.feature_cols = [
            'portcalls', 'portcalls_container', 'total_cargo_tonnage',
            'pw_calls_rolling_7d_mean', 'pw_calls_rolling_14d_mean',
            'day_of_week', 'is_weekend'
        ]
        
        P, D, F = len(self.ports), len(self.dates), len(self.feature_cols)
        self.data_tensor = np.zeros((P, D, F), dtype=np.float32)
        
        df_sorted = df_panel.sort_values(['portwatch_id', 'date_utc'])
        for f_idx, col in enumerate(self.feature_cols):
            self.data_tensor[:, :, f_idx] = df_sorted[col].fillna(0).values.reshape(P, D)
            
    def __len__(self):
        return len(self.meta)
        
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        p_idx = self.port2idx[row['portwatch_id']]
        start_t = self.date2idx[row['input_start_date']]
        ref_t = self.date2idx[row['reference_date']]
        
        # Slice [30, F]
        x = torch.from_numpy(self.data_tensor[p_idx, start_t:ref_t+1, :])
        y = torch.tensor(row['target_portcalls_7d_future_sum'], dtype=torch.float32)
        return x, y

# Instantiate PyTorch DataLoader:
# train_loader = DataLoader(MemoryEfficientPortDataset('DL/.../port_daily_unified.parquet', 'DL/.../sequence_metadata.parquet', 'train'), batch_size=256, shuffle=True)
```

---

### 2.3 Graph Neural Networks (PyTorch Geometric ST-GCN)

```python
import torch
import pandas as pd
from torch_geometric.data import Data

def get_caribbean_st_graph(graph_dir):
    df_nodes = pd.read_parquet(f"{graph_dir}/caribbean_graph_nodes.parquet")
    df_edges = pd.read_parquet(f"{graph_dir}/caribbean_graph_edges.parquet")
    
    node2idx = {nid: i for i, nid in enumerate(df_nodes['node_id'])}
    
    src = [node2idx[s] for s in df_edges['source_node']]
    dst = [node2idx[t] for t in df_edges['target_node']]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    edge_attr = torch.tensor(
        df_edges[['voyages_total', 'voyages_container', 'voyages_tanker', 'is_inter_canonical_edge']].values,
        dtype=torch.float32
    )
    
    # Static node features (berth capacity, quay length, draft)
    x_static = torch.tensor(
        df_nodes[['latitude', 'longitude', 'quay_meters', 'wharf_meters', 'offshore_berths']].fillna(0).values,
        dtype=torch.float32
    )
    
    return Data(x=x_static, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(df_nodes))
```

---

## 3. Recommended Next Steps for Model Development

1. **Establish Stage 01 Baselines:**
   - Fit a persistence baseline ($\hat{y}_{t+7} = \text{calls}_t \times 7$) and trailing rolling mean baseline ($\hat{y}_{t+7} = \text{pw\_calls\_rolling\_7d\_mean} \times 7$).
   - Train a global LightGBM regressor on `target_portcalls_7d_future_sum` to establish standard tabular benchmarks (MAE, RMSE, WAPE).
2. **Train Deep Temporal Architectures:**
   - Train a 2-layer Bidirectional LSTM and Temporal Fusion Transformer (TFT) on the 30-day lookback sequences.
   - Evaluate feature importance across exogenous weather shocks (NOAA) and macro disaster counts (EM-DAT).
3. **Deploy Spatio-Temporal Graph Neural Networks (ST-GNN):**
   - Use the Caribbean directed shipping graph (528 nodes, 9,969 edges) to model regional congestion shock diffusion.
   - Use the North American multimodal hinterland graph (251,646 nodes, 303,016 edges) to test port-to-rail intermodal bottleneck transmission.
4. **Quarantine Test Split Until Final Sign-off:**
   - Keep the Test split (2025 calendar year, 741,335 sequences) strictly unobserved during model architecture exploration, hyperparameter tuning, and feature ablation.

