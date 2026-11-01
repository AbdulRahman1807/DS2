# North American Hinterland Rail & Intermodal Graph Metadata

**Graph Architecture:** Heterogeneous Multimodal Transport Network  
**Geographic Domain:** Continental North America (US, Canada, Mexico)  
**Components:**
1. `rail_nodes.parquet`: 250204 topological track vertices
2. `rail_edges.parquet`: 302774 physical track segments (264,651 miles)
3. `intermodal_nodes.parquet`: 1401 intermodal bulk transfer facilities
4. `port_intermodal_candidates.parquet`: 242 evaluated seaport-to-facility candidate linkages

## Topological Principles

- **Explicit Node Identifiers:** `FRFRANODE` and `TOFRANODE` represent ground-truth endpoint vertices established by the Federal Railroad Administration. Zero geometric snapping or node synthesis was performed.
- **Directionality:** Undirected physical rail infrastructure. Edge table preserves source from/to endpoint convention.
- **Intermodal Layer:** Intermodal nodes connect to rail infrastructure via physical rail service indicators (`mode_rail == 'Y'`).
- **Seaport Connection Governance:** Seaport-to-intermodal edges are strictly categorized:
  - `VERIFIED_ADJACENT_MARINE` (Distance <= 5km + Marine berth connection)
  - `STRONG_SPATIAL_CANDIDATE` (Distance <= 10km + Rail/Water access)
  - `METROPOLITAN_PROXIMITY_ONLY` (Spatial proximity within 30km, unverified physical link)
- **Warning on Graph Synthesis:** Distance proximity must not be converted into guaranteed physical connectivity without verifying local spur rights.
