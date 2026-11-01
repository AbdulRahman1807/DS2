# Graph Network Coverage, Connectivity & Topological Analysis Report

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Final Data Engineering Stage  
**Pipeline:** Workstream D — Graph Packaging & Topological Audit  
**Artifact Directory:** `DL/data/staging/supplementary_acquisition/processed/graph_ready/`  
**Date:** 2026-11-01  
**Status:** Complete, Verified, Topologically Consistent  

---

## 1. Network Topology & Structural Metric Summary

| Graph Domain | Node Count | Edge Count | Density | Weakly Connected Components | Largest Component Share | Diameter / Radius Proxy | Mean Node Degree |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **North American Hinterland** | **251,646** | **303,016** | $9.58 \times 10^{-6}$ | 148 | 98.42% (247,671 nodes) | Unweighted ~84 hops | 2.41 |
| **Caribbean Regional Maritime** | **528** | **9,969** | $3.58 \times 10^{-2}$ | 4 | 99.24% (524 nodes) | 4 maritime hops | 37.76 |

---

## 2. North American Hinterland Network Analysis

### 2.1 Node Composition & Modality Breakdown
* **Rail Junctions (`RAIL_JUNCTION`):** 250,204 nodes (99.43% of network). Represents the full physical rail track topology across USA, Canada, and Mexico from the NTAD North American Rail Network Lines.
* **Intermodal Facilities (`INTERMODAL_FACILITY`):** 1,401 freight terminals (0.56% of network). Spatially resolved multimodal freight transfer hubs with reported handling capabilities for trucks, rail, water, and crude/petroleum energy products.
* **Port Gateways (`PORT_GATEWAY`):** 41 canonical US seaport complexes (0.01% of network) linked directly to the rail hinterland.
* **Intermodal Transfer Connectors:** 242 directed geodesic edges connecting deepwater seaports to intermodal rail/truck transfer facilities within 50 km (147 within 10 km, 67 within 25 km, 28 within 50 km).

### 2.2 Connectivity and Corridor Choke Points
* The giant connected component spans **247,671 nodes (98.42%)**, providing unbroken transcontinental routing from the Pacific Coast (LA/Long Beach, Seattle, Tacoma) across Class I rail networks (BNSF, Union Pacific, Norfolk Southern, CSX) to the Atlantic seaboard (NY/NJ, Savannah, Norfolk) and the Gulf of Mexico (Houston, New Orleans).
* Minor disconnected components (147 isolated clusters totaling 3,975 nodes) consist of private industrial spurs, isolated mining branch lines, and island rail systems (e.g., Oahu, Alaska Railroad isolated sections).

---

## 3. Caribbean Maritime Graph Analysis

### 3.1 Network Density & Transshipment Hub Topology
* **Dense Feeder-Hub Topology:** With 528 nodes and 9,969 directed edges, the Caribbean graph exhibits a remarkably high edge density of **3.58%**, characteristic of intense regional feeder services.
* **Primary Transshipment Hubs (Highest In/Out Degree):**
  1. **Kingston Container Terminal, Jamaica (`carib_312` / `port587`):** 184 directed connections. Dominant regional transshipment hub.
  2. **Freeport Container Port, Bahamas (`carib_108` / `port397`):** 152 connections. Gateway for North America / Transatlantic flows.
  3. **Caucedo / Rio Haina, Dominican Republic (`carib_420` / `port1116`):** 138 connections.
  4. **San Juan, Puerto Rico (`carib_401` / `port1150`):** 126 connections.
* **Inter-Canonical Gateway Lanes:** **1,179 directed edges** directly link two canonical global gateways from the primary PortWatch registry, enabling direct shock transmission modeling between primary international ports.

### 3.2 Voyage Volume & Vessel Type Breakdown
* **Total Documented Historical Voyages:** 1,294,831 commercial voyages across 9,969 lanes.
* **Container Vessel Voyages:** 481,209 voyages across 3,412 active container feeder corridors.
* **Tanker Vessel Voyages:** 312,840 voyages across oil product and bunkering routes.
* **Dry Bulk Carrier Voyages:** 198,450 voyages.
* **General / Breakbulk Cargo Voyages:** 187,212 voyages.
* **Island Feeder Voyages:** 115,120 voyages servicing small island nations.

---

## 4. Multi-Modal Cross-Graph Integration Readiness

Both graph datasets are fully keyed to the primary canonical identifiers:
1. Canonical ports in the Caribbean graph carry `canonical_portwatch_id` matching `port_daily_unified.parquet` (`portwatch_id`).
2. US ports in the North American hinterland graph carry `canonical_portwatch_id` matching `port_daily_unified.parquet`.
3. Downstream engineers can execute Spatio-Temporal Graph Neural Networks (ST-GNN) where:
   - Dynamic node features $\mathbf{X}_t \in \mathbb{R}^{N \times F}$ are indexed from `port_daily_unified.parquet`.
   - Spatial adjacency $\mathbf{A} \in \mathbb{R}^{N \times N}$ is derived from `caribbean_graph_edges.parquet` or `north_american_graph_edges.parquet`.
   - Message passing propagates congestion shocks along empirical maritime lanes and inland rail corridors.

