# Stage 02 Deep Learning Architecture & Data-Readiness Plan

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Stage:** Stage 02 — Sequence Modeling, Graph Neural Networks & Spatio-Temporal Congestion Forecasting  
**Author:** Lead AI & Deep Learning Architect  
**Status:** Approved Architectural Blueprint (Implementation Frozen Pending Review)  
**Target Milestone:** Beat Frozen Classical LightGBM Baseline on Out-of-Time Validation  

---

> [!IMPORTANT]
> **Stage 01 Freeze Commitment:**
> The classical ML pipeline—including model weights ([`best_macro_f1_model.joblib`](file:///Users/abdul/Desktop/DS/models/best_macro_f1_model.joblib), [`xgboost_baseline_model.joblib`](file:///Users/abdul/Desktop/DS/models/xgboost_baseline_model.joblib)), 13-feature arrival contracts, operational alert thresholds ($\tau^* = 0.22$, default $\tau = 0.25$), and FastAPI REST contracts—is **STRICTLY FROZEN**. No Stage 02 experiments will modify, retrain, or overwrite these production baseline artifacts.

---

## 1. Deep Learning Architecture Evaluation Against Available Data

Classical tabular gradient boosted trees (L ightGBM/XGBoost) treat each container arrival as an independent and identically distributed (IID) tabular sample. However, terminal bottlenecks are inherently **spatio-temporal queueing phenomena**: a container delays because the crane gang is backlogged, its specific yard block is physically saturated, or an upstream customs inspection corridor is throttled.

We evaluate three deep learning architectures against the raw Mendeley TOS event stream (952,069 events, 95,647 container stays, 28-week span):

```
+--------------------------------------------------------------------------------------------------+
|                                    CANDIDATE ARCHITECTURES                                       |
+--------------------------------+--------------------------------+--------------------------------+
|          1. LSTM / GRU         |    2. Temporal Transformer     |    3. Spatio-Temporal GNN      |
|  Recurrent Milestone Sequencer |  Causal Self-Attention Stream  |   Yard Block Topology Network  |
+--------------------------------+--------------------------------+--------------------------------+
| - Intra-stay event sequences   | - Continuous temporal encodings| - Graph nodes = Yard Blocks     |
| - Recurrent memory cell        | - Multi-head attention across  | - Edges = physical moves/paths |
| - Degenerates at arrival (L=1) |   preceding container arrivals | - Aggregates yard saturation   |
| - Feasible at +24h / +48h      | - Captures vessel queue waves  | - Highest physical fidelity    |
+--------------------------------+--------------------------------+--------------------------------+
```

### 1.1 Comparative Architectural Trade-Off Matrix

| Dimension | Recurrent Neural Network (LSTM / GRU) | Temporal Attention Transformer | Spatio-Temporal Graph Neural Network (ST-GNN) |
| :--- | :--- | :--- | :--- |
| **Primary Structural Unit** | Ordered sequence of activities within a container stay ($\Delta t_1 \to \Delta t_2$) | Sliding causal window of successive vessel container discharges ($W = 256-512$) | Spatial graph of terminal yard blocks ($\mathcal{G} = (\mathcal{V}, \mathcal{E})$) + temporal GCN |
| **Applicability at Arrival ($t=0$)** | **POOR (Degenerate):** Only 1 token (`[DISCHARGE]`) is observed at arrival; equivalent to an MLP. | **HIGH:** Can attend to the preceding 500 containers discharged from the same vessel/berth. | **EXCELLENT:** Ingests the current occupancy and dwell load of the target `YARD_BLOCK`. |
| **Applicability at Mid-Stay (+24h, +48h)**| **STRONG:** Naturally processes variable-length inspection milestone chains (3 to 8 events). | **STRONG:** Attends across both intra-container milestones and concurrent yard arrivals. | **HIGH:** Updates graph node embeddings with incoming inspection and handling churn. |
| **Additional Data Requirements** | Event token vocabulary $\mathcal{V}_{\text{events}}$, padded sequence tensors, interval durations. | Continuous temporal sinusoidal encodings, vessel call groupings, causal sequence masking. | Yard block adjacency matrix $\mathbf{A} \in \mathbb{R}^{B \times B}$, physical distance proxies. |
| **Computational Footprint** | Low (CPU / lightweight GPU inference: $\sim 2-5\text{ ms}$). | Moderate (GPU acceleration required for $O(W^2)$ attention: $\sim 8-15\text{ ms}$). | Moderate (Sparse matrix message passing on GPU: $\sim 5-10\text{ ms}$). |
| **Interpretability** | Moderate (Hidden state saliency maps). | High (Attention weight visualization across preceding containers). | High (Spatial heatmap of yard block bottleneck contagion). |

---

## 2. Sequence & Graph Data Representation Specifications

### 2.1 Intra-Stay vs. Inter-Stay Sequence Framing

To unlock signal beyond static tabular attributes, Stage 02 designs two distinct sequence representations:

#### Representation A: Inter-Stay Vessel Stream (For Arrival Triage at $t = 0$)
Every container $i$ arriving from vessel call $V_k$ is placed into a chronological crane sequence:
$$\mathbf{S}_{V_k} = [x_1, x_2, \dots, x_N], \quad t_{\text{discharge}}(x_1) \le t_{\text{discharge}}(x_2) \le \dots \le t_{\text{discharge}}(x_N)$$
Each token $x_i \in \mathbb{R}^d$ combines:
1. Static physical embeddings: `carrier_prefix` (dense embedding $\in \mathbb{R}^8$), `CTR_SIZE`, `CTR_TYPE` ($\in \mathbb{R}^4$), `GROSS` weight.
2. Continuous temporal encoding: Positional encoding of discharge timestamp $\tau_i - \tau_{\text{berth}}$ and inter-arrival headway $\Delta \tau = \tau_i - \tau_{i-1}$.
3. Target yard location embedding: `YARD_BLOCK` categorical embedding ($\in \mathbb{R}^8$).

#### Representation B: Intra-Stay Event History (For Dynamic In-Yard Triage at $+24\text{h}$, $+48\text{h}$)
For containers already dwelled in the yard, the sequence represents historical completed milestones:
$$\mathbf{H}_c(t) = \left[ (e_1, \delta t_1), (e_2, \delta t_2), \dots, (e_m, \delta t_m) \right], \quad \delta t_j \le t$$
where $e_j \in \{\text{DISCHARGE}, \text{STACK}, \text{BEHANDLE}, \text{CUSTOMS\_HOLD}, \text{JOB\_DEL}, \dots\}$.

---

### 2.2 Yard Block Spatial Graph Formulation (ST-GNN)

The terminal physical layout is formalized as a directed, weighted spatial graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{W})$:
- **Nodes $\mathcal{V}$ ($|\mathcal{V}| = B \approx 28$ blocks):** Unique `YARD_BLOCK` zones observed in the terminal operating system (e.g. Block 01A through Block 05D).
- **Edges $\mathcal{E}$:** Direct physical connectivity between adjacent yard blocks.
  - Edge weights $W_{uv} = \exp(-d(u, v) / \sigma)$, where $d(u, v)$ is the physical grid distance between block centroids.
- **Dynamic Node State Matrix $\mathbf{H}^{(t)} \in \mathbb{R}^{B \times F}$:**
  At any query timestamp $t$, node $v$ maintains a dynamic feature vector:
  - $N_v(t)$: Current active container population in block $v$.
  - $\bar{D}_v(t)$: Mean dwell time of containers currently residing in block $v$.
  - $C_v(t)$: Critical delay concentration ratio in block $v$ ($\% \text{ stays } > 160.9\text{h}$).
  - $U_v(t)$: Crane unstacking transaction velocity in trailing 24 hours.

---

## 3. Strict Temporal Isolation & Leakage Prevention Protocol

To guarantee mathematical integrity and preserve benchmark comparability with Stage 01, Stage 02 implements strict temporal isolation gates:

```
+----------------------------------------------------------------------------------------------------+
|                                    TEMPORAL PARTITIONING GATES                                     |
+--------------------------------+--------------------------------+----------------------------------+
|      TRAINING PARTITION        |      VALIDATION PARTITION      |        QUARANTINED HOLDOUT       |
|    2021-06-10 to 2021-08-26    |    2021-08-27 to 2021-09-11    |     2021-09-12 to 2021-09-30     |
|      60,698 Stays (Real)       |      20,879 Stays (Real)       |       13,959 Stays (Real)        |
+--------------------------------+--------------------------------+----------------------------------+
| - Dynamic graph states fitted  | - Strict causal roll-forward   | - STRICTLY FORBIDDEN             |
|   strictly on historical moves | - Zero target lookahead        | - Never accessed for tuning,     |
| - Sequence windows masked at t | - Max(t_val) strictly precedes |   pruning, or hyperparameter     |
| - Scalers fitted on Train only |   Min(t_holdout) by ~7 hours   |   selection                      |
+--------------------------------+--------------------------------+----------------------------------+
```

### 3.1 Strict Prevention Rules:
1. **Zero Future Token Attention (Causal Masking):**
   In Transformer self-attention, upper-triangular attention mask $\mathbf{M}_{ij} = -\infty$ for $j > i$ is enforced. No container can attend to a container discharged at a later timestamp.
2. **As-Of Dynamic Graph State Aggregation:**
   Yard block states $\mathbf{H}^{(t)}$ must only aggregate departure and milestone events whose completion timestamp $\tau \le t$. Future departures cannot leak into current yard density.
3. **Quarantine Preservation:**
   The held-out test split (`data/processed/test.parquet`, `2021-09-12` to `2021-09-30`) remains completely quarantined. No deep learning model will be evaluated against this partition during architecture research.

---

## 4. Benchmarking Protocol Against Frozen LightGBM Baseline

### 4.1 Frozen Baseline Metrics to Beat
Stage 02 architectures will be benchmarked directly against the frozen Stage 01 LightGBM baseline on the **identical out-of-time real validation split** (`val.parquet`, $N = 20,879$):

| Horizon / Task | Frozen Model | Real Val Macro-F1 | Balanced Acc | Critical Recall | Critical PR-AUC | Inference SLA |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Arrival ($t=0$)** | Frozen LightGBM ([`best_macro_f1_model.joblib`](file:///Users/abdul/Desktop/DS/models/best_macro_f1_model.joblib)) | **0.3905** | **40.07%** | **11.38%** | **0.0894** | < 25.0 ms |
| **Arrival ($t=0$)** | Frozen XGBoost ([`xgboost_baseline_model.joblib`](file:///Users/abdul/Desktop/DS/models/xgboost_baseline_model.joblib)) | **0.3968** | **40.76%** | **11.18%** | **0.0937** | < 25.0 ms |
| **+48h In-Yard** | Reassessment LightGBM (+48h as-of features) | **0.5515** | **60.65%** | **60.42%** | **0.3003** | < 50.0 ms |

---

## 5. Adoption Decision Gates: What Measurable Gain Justifies Deep Learning?

Deep learning architectures introduce substantial operational overhead: GPU compute requirements, larger package dependencies (PyTorch/PyG), complex serialization pipelines, and reduced decision transparency.

To justify adopting a Deep Learning model into the production API over the 0.29 ms LightGBM baseline, the candidate model must satisfy **three non-negotiable quantitative gates**:

```
                                  +------------------------------------+
                                  |     STAGE 02 ADOPTION GATES        |
                                  +------------------------------------+
                                                    |
         +------------------------------------------+-----------------------------------------+
         |                                          |                                         |
         v                                          v                                         v
+------------------------+              +------------------------+              +------------------------+
|      GATE 1: MACRO-F1  |              |    GATE 2: CRITICAL    |              |      GATE 3: SLA &     |
|   STATISTICAL LIFT     |              |     PR-AUC & RECALL    |              |     LATENCY BUDGET     |
+------------------------+              +------------------------+              +------------------------+
| Must achieve:          |              | Must achieve:          |              | Must maintain:         |
| Delta Macro-F1 >= +0.05|              | Critical PR-AUC >= 0.15|              | P95 Latency < 25.0 ms  |
| over LightGBM (>= 0.45 |              | (vs 0.089 baseline)    |              | Model binary size      |
| at t=0, or >= 0.60 at  |              | Critical Recall >= 25% |              | < 100 MB               |
| +48h).                 |              | at Tau = 0.25.         |              | Deterministic fallback |
+------------------------+              +------------------------+              +------------------------+
```

### Rejection Thresholds:
- If a Deep Learning model achieves **$< +0.03$ Macro-F1 improvement** over LightGBM, it will be **REJECTED** for production serving, and LightGBM will remain the production standard.
- If inference latency exceeds **25.0 ms on CPU**, it cannot replace the real-time arrival hook and will be restricted to hourly background batch processing.

---

## 6. Phase Execution Roadmap

1. **Phase 2.1 (Data Tensor Pipelines):**
   - Implement `SequenceDataset` and `YardGraphBuilder` PyTorch loaders under `DL/data/`.
   - Compute spatial coordinate adjacency matrix $\mathbf{A}$ for all 28 yard blocks from empirical movement logs.
2. **Phase 2.2 (Architecture Prototyping):**
   - Model 1: Bidirectional GRU with attention pooling on intra-stay milestones.
   - Model 2: Temporal Transformer with causal multi-head attention on vessel streams.
   - Model 3: Spatio-Temporal Graph Convolutional Network (ST-GCN) on yard block graphs.
3. **Phase 2.3 (Rigorous Out-of-Time Benchmarking):**
   - Train on Train partition (`2021-06-10` to `2021-08-26`).
   - Validate on Val partition (`2021-08-27` to `2021-09-11`).
   - Run 1,000 bootstrap iterations to establish 95% confidence intervals against LightGBM.
4. **Phase 2.4 (Production Gate Review):**
   - Formal evaluation against Gates 1, 2, and 3 before any integration with the FastAPI service.
