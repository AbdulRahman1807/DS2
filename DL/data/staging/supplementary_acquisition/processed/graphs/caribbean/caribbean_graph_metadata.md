# Caribbean Observed Maritime Graph Metadata

**Graph Architecture:** Directed Weighted Empirical Shipping Network  
**Temporal Domain:** Calendar Year 2019 (Observed AIS Trajectories)  
**Geographic Domain:** Caribbean Sea, Gulf of Mexico, Florida Straits, Northern South America  
**Output Files:**
1. `caribbean_maritime_nodes.parquet`: 528 regional ports
2. `caribbean_maritime_edges.parquet`: 9969 directed voyage segments

## Graph Attributes and Edge Structure

- **Edge Weight Definitions:** Every edge represents observed 2019 vessel passages broken down by vessel type (`voyages_container`, `voyages_tanker`, `voyages_dry_bulk`, `voyages_general_cargo`, `voyages_cruise`, `voyages_service`, `voyages_island_feeder`).
- **Inter-Canonical Subgraph:** 1179 directed edges connect canonical seaports resolved in the global registry (e.g. Miami, San Juan, Port of Spain, Kingston, Caucedo).
- **Regional Berth Retainage:** 8,790 edges link smaller island berths, fuel jetties, and local anchorages, preserving the fine-grained feeder topology of the Caribbean basin.
- **Critical Modeling Constraint:** This graph is **strictly regional (2019)**. It cannot be combined into a global graph without introducing severe regional sampling bias.
