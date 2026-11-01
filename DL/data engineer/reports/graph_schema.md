# Graph-Ready Data Architecture & Topological Schema

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Final Data Engineering Stage  
**Pipeline:** Workstream D — Graph Packaging for Graph Neural Networks (GNNs)  
**Output Directory:** `DL/data/staging/supplementary_acquisition/processed/graph_ready/`  
**Date:** 2026-11-01  
**Status:** Packaged, Normalized, and Documented  

---

## 1. Executive Summary & Graph Topology Overview

Graph Neural Networks (GNNs, ST-GCN, GAT, Spatio-Temporal Transformers) require standardized node index arrays, edge adjacency matrices (`edge_index`), and edge feature tensors (`edge_attr`). 

Workstream D delivers two distinct, fully realized topological graphs:
1. **North American Multimodal Hinterland Network:**
   - **Nodes (251,646):** 250,204 rail network junctions + 1,401 freight intermodal transfer terminals + 41 canonical US port gateways.
   - **Edges (303,016):** 302,774 physical freight rail line segments + 242 geodesic intermodal transfer connectors ($\le 50\text{ km}$).
   - **Operational Scope:** Inland supply chain bottleneck propagation, rail corridor choke points, intermodal transfer latency.
2. **Caribbean Regional Maritime Shipping Graph:**
   - **Nodes (528):** 528 commercial seaports spanning 38 Caribbean and Central American territories (including 18 canonical PortWatch gateways).
   - **Edges (9,969):** 9,969 empirical directed maritime shipping lanes with observed transit duration distributions (median, P25, P75, IQR) and voyage frequencies.
   - **Operational Scope:** Maritime feeder-to-hub bottleneck transmission, transshipment delays, regional shock contagion.

---

## 2. North American Hinterland Graph Schema

### 2.1 Nodes: `north_american_graph_nodes.parquet`
* **Primary Key:** `node_id` (string)
* **Count:** 251,646 nodes
* **Size:** 3.04 MB

| Column Name | Data Type | Physical Domain | Description |
| :--- | :--- | :--- | :--- |
| `node_id` | `string` | `rail_{id}`, `im_{id}`, `port_{id}` | Unique global node identifier with type prefix. |
| `node_type` | `string` | `RAIL_JUNCTION`, `INTERMODAL_FACILITY`, `PORT_GATEWAY` | Heterogeneous node classification. |
| `node_name` | `string` | Text string | Human-readable name or facility descriptor. |
| `country` | `string` | `USA`, `CAN`, `MEX` | Sovereign country code. |
| `state_province` | `string` | 2-letter state/province code | State or jurisdiction. |
| `latitude` | `float64` | Degrees North | WGS84 latitude (available for ports and intermodal facilities). |
| `longitude` | `float64` | Degrees East | WGS84 longitude (available for ports and intermodal facilities). |
| `degree` | `int32` | Non-negative integer | Topological node degree in the rail/intermodal graph. |
| `is_canonical_port`| `int8` | {0, 1} | Binary indicator: 1 if canonical PortWatch deepwater node. |
| `canonical_portwatch_id`| `string` | `port{id}` or None | Foreign key linking to primary `port_daily_unified.parquet`. |
| `operator_name` | `string` | Text string | Terminal operator, railroad company, or port authority. |
| `source_provenance`| `string` | Source identifier | Primary source dataset lineage. |

### 2.2 Edges: `north_american_graph_edges.parquet`
* **Primary Key:** `edge_id` (string)
* **Adjacency Keys:** `source_node`, `target_node`
* **Count:** 303,016 edges
* **Size:** 11.49 MB

| Column Name | Data Type | Physical Domain | Description |
| :--- | :--- | :--- | :--- |
| `edge_id` | `string` | `rail_edge_{id}`, `transfer_{port}_{im}` | Unique global edge identifier. |
| `source_node` | `string` | Foreign key to `node_id` | Origin node identifier. |
| `target_node` | `string` | Foreign key to `node_id` | Destination node identifier. |
| `source_name` | `string` | Text string | Name of source facility or junction. |
| `target_name` | `string` | Text string | Name of destination facility or junction. |
| `edge_type` | `string` | `RAIL_LINE`, `PORT_INTERMODAL_TRANSFER` | Edge archetype (physical track vs drayage connector). |
| `distance_km` | `float32` | Kilometers | Segment physical or geodesic distance. |
| `distance_miles` | `float32` | Miles | Segment distance in statute miles. |
| `tracks` | `int16` | Non-negative integer | Number of parallel main tracks (rail). |
| `operator` | `string` | Railroad reporting mark | Primary railroad carrier or drayage entity. |
| `subdivision` | `string` | Text string | Operating railroad subdivision or state line. |
| `net_type` | `string` | `M` (Main), `S` (Side), `Y` (Yard), etc. | Network track classification. |
| `stracnet` | `string` | Categorical / String | Strategic Rail Corridor Network defense designation. |
| `is_inter_canonical_edge` | `int8` | {0, 1} | 1 if edge links two canonical nodes. |
| `source_provenance` | `string` | Lineage string | Primary source provenance. |

---

## 3. Caribbean Maritime Graph Schema

### 3.1 Nodes: `caribbean_graph_nodes.parquet`
* **Primary Key:** `node_id` (string: `carib_{port_id}`)
* **Count:** 528 nodes
* **Size:** 31 KB

| Column Name | Data Type | Physical Domain | Description |
| :--- | :--- | :--- | :--- |
| `node_id` | `string` | `carib_{port_id}` | Unique regional maritime node identifier. |
| `port_name` | `string` | Text string | Port facility name. |
| `country_name` | `string` | Text string | Host country or territory. |
| `country_iso3` | `string` | 3-letter code | ISO-3 country designation. |
| `unlocode` | `string` | 5-character UN/LOCODE | United Nations Code for Trade and Transport Locations. |
| `latitude` | `float64` | Degrees North | WGS84 centroid latitude. |
| `longitude` | `float64` | Degrees East | WGS84 centroid longitude. |
| `quay_meters` | `float32` | Meters | Total commercial quay wall length. |
| `wharf_meters` | `float32` | Meters | Wharf length in meters. |
| `offshore_berths` | `int32` | Integer | Count of operational offshore mooring berths. |
| `is_canonical_port`| `int8` | {0, 1} | 1 if cross-referenced to canonical global registry. |
| `canonical_portwatch_id`| `string` | `port{id}` or None | Foreign key linking to primary `port_daily_unified.parquet`. |
| `canonical_portwatch_name`| `string` | Text string or None | Canonical PortWatch display name. |
| `node_type` | `string` | `REGIONAL_MARITIME_PORT` | Node archetype. |
| `source_provenance`| `string` | Lineage string | Primary data source. |

### 3.2 Edges: `caribbean_graph_edges.parquet`
* **Primary Key:** `edge_id` (string: `carib_edge_{id}`)
* **Adjacency Keys:** `source_node`, `target_node`
* **Count:** 9,969 directed edges
* **Size:** 148 KB

| Column Name | Data Type | Physical Domain | Description |
| :--- | :--- | :--- | :--- |
| `edge_id` | `string` | `carib_edge_{id}` | Unique directed maritime edge identifier. |
| `source_node` | `string` | Foreign key to `node_id` | Departure port node. |
| `target_node` | `string` | Foreign key to `node_id` | Arrival port node. |
| `source_name` | `string` | Text string | Origin port name. |
| `target_name` | `string` | Text string | Destination port name. |
| `source_canonical_id`| `string` | `port{id}` or None | Canonical PortWatch ID of departure port. |
| `target_canonical_id`| `string` | `port{id}` or None | Canonical PortWatch ID of arrival port. |
| `is_inter_canonical_edge` | `int8` | {0, 1} | 1 if edge directly connects two canonical global gateways (1,179 edges). |
| `voyages_total` | `int32` | Integer | Total verified historical commercial voyages on lane. |
| `voyages_container` | `int32` | Integer | Number of container vessel voyages. |
| `voyages_tanker` | `int32` | Integer | Number of tanker vessel voyages. |
| `voyages_dry_bulk` | `int32` | Integer | Number of dry bulk carrier voyages. |
| `voyages_general_cargo` | `int32` | Integer | Number of general cargo carrier voyages. |
| `voyages_island_feeder` | `int32` | Integer | Number of regional island feeder voyages. |
| `directionality` | `string` | `DIRECTED` | Graph edge directionality. |
| `edge_type` | `string` | `DIRECTED_MARITIME_CORRIDOR` | Edge classification. |
| `source_provenance` | `string` | Lineage string | Primary data source. |

---

## 4. PyTorch Geometric (PyG) Integration Pattern

Downstream graph engineers can convert these Parquet files into standard PyG `Data` or `HeteroData` objects in under 5 lines of code:

```python
import torch
import pandas as pd
from torch_geometric.data import Data

def load_caribbean_pyg(graph_dir):
    df_nodes = pd.read_parquet(f"{graph_dir}/caribbean_graph_nodes.parquet")
    df_edges = pd.read_parquet(f"{graph_dir}/caribbean_graph_edges.parquet")
    
    # 1. Map node string IDs to contiguous integer indices [0, N-1]
    node2idx = {nid: i for i, nid in enumerate(df_nodes['node_id'])}
    
    # 2. Construct edge_index tensor [2, E]
    src = [node2idx[s] for s in df_edges['source_node']]
    dst = [node2idx[t] for t in df_edges['target_node']]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # 3. Construct edge_attr tensor [E, D]
    edge_attr = torch.tensor(
        df_edges[['voyages_total', 'voyages_container', 'voyages_tanker', 'is_inter_canonical_edge']].values,
        dtype=torch.float32
    )
    
    # 4. Construct node static feature tensor [N, F]
    x_static = torch.tensor(
        df_nodes[['latitude', 'longitude', 'quay_meters', 'wharf_meters', 'offshore_berths']].fillna(0).values,
        dtype=torch.float32
    )
    
    return Data(x=x_static, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(df_nodes))
```

