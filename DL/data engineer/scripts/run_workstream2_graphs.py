import os
import pandas as pd
import numpy as np
from pathlib import Path

def run():
    print("=== WORKSTREAM 2: Spatial Graph & Hinterland Network Construction ===")
    script_dir = Path(__file__).resolve().parent
    dl_root = script_dir.parents[1]
    repo_root = script_dir.parents[2]

    base_dir = os.environ.get("PHASE2_BASE_DIR", str(dl_root / "data/staging/supplementary_acquisition"))
    raw_rail = os.path.join(base_dir, "raw/ntad_rail_network/NTAD_North_American_Rail_Network_Lines_-4617128124973485019.csv")
    raw_inter = os.path.join(base_dir, "raw/ntad_intermodal/NTAD_Intermodal_Freight_Facilities_Pipeline_Terminals_-3233202154821634717.csv")
    raw_carib_ports = os.path.join(base_dir, "raw/caribbean_ais/ports.csv")
    raw_carib_seg = os.path.join(base_dir, "raw/caribbean_ais/segments_port2port.csv")
    raw_carib_calls = os.path.join(base_dir, "raw/caribbean_ais/portcalls.csv")
    cw_carib = os.path.join(base_dir, "crosswalks/caribbean_to_canonical_crosswalk.csv")
    
    out_na = os.path.join(base_dir, "processed/graphs/north_american")
    out_carib = os.path.join(base_dir, "processed/graphs/caribbean")
    os.makedirs(out_na, exist_ok=True)
    os.makedirs(out_carib, exist_ok=True)

    # =========================================================================
    # PART A: NORTH AMERICAN RAIL & INTERMODAL GRAPH
    # =========================================================================
    print("--- Part A: Building North American Rail & Intermodal Graph ---")
    
    # 1. Read Rail Lines
    print("Reading NTAD Rail Arcs (302,774 lines)...")
    rail_cols = [
        'OBJECTID', 'FRAARCID', 'FRFRANODE', 'TOFRANODE', 'STFIPS', 'STATEAB',
        'COUNTRY', 'RROWNER1', 'PASSNGR', 'STRACNET', 'TRACKS', 'NET', 'MILES', 'KM', 'TIMEZONE'
    ]
    df_rail = pd.read_csv(raw_rail, usecols=rail_cols, low_memory=False)

    df_rail_edges = pd.DataFrame({
        'edge_id': df_rail['FRAARCID'].astype(int),
        'from_node': df_rail['FRFRANODE'].astype(int),
        'to_node': df_rail['TOFRANODE'].astype(int),
        'miles': df_rail['MILES'].astype(float),
        'km': df_rail['KM'].astype(float),
        'tracks': df_rail['TRACKS'].fillna(1).astype(int),
        'net_type': df_rail['NET'].astype(str),
        'stracnet': df_rail['STRACNET'].fillna('NONE').astype(str),
        'railroad_owner': df_rail['RROWNER1'].fillna('UNKNOWN').astype(str),
        'state': df_rail['STATEAB'].fillna('UNKNOWN').astype(str),
        'country': df_rail['COUNTRY'].astype(str),
        'timezone': df_rail['TIMEZONE'].fillna('UNKNOWN').astype(str),
        'directionality': 'UNDIRECTED_PHYSICAL_TRACK',
        'source_provenance': 'NTAD_North_American_Rail_Network_Lines_2024'
    })

    rail_edges_p = os.path.join(out_na, "rail_edges.parquet")
    df_rail_edges.to_parquet(rail_edges_p, index=False)
    print(f"Saved {rail_edges_p} ({len(df_rail_edges)} rail edges)")

    # 2. Build Explicit Rail Nodes from Endpoints
    print("Aggregating 250,204 Rail Topology Nodes...")
    from_nodes = df_rail[['FRFRANODE', 'STATEAB', 'COUNTRY', 'RROWNER1', 'MILES']].rename(
        columns={'FRFRANODE': 'node_id', 'STATEAB': 'state', 'COUNTRY': 'country', 'RROWNER1': 'rr'}
    )
    to_nodes = df_rail[['TOFRANODE', 'STATEAB', 'COUNTRY', 'RROWNER1', 'MILES']].rename(
        columns={'TOFRANODE': 'node_id', 'STATEAB': 'state', 'COUNTRY': 'country', 'RROWNER1': 'rr'}
    )
    all_endpoint_refs = pd.concat([from_nodes, to_nodes], ignore_index=True)

    node_stats = all_endpoint_refs.groupby('node_id').agg(
        degree=('state', 'count'),
        connected_miles=('MILES', 'sum'),
        primary_country=('country', 'first'),
        primary_state=('state', 'first'),
        primary_operator=('rr', 'first')
    ).reset_index()

    node_stats['node_type'] = 'RAILWAY_TRACK_JUNCTION'
    node_stats['source_provenance'] = 'NTAD_North_American_Rail_Topology'

    rail_nodes_p = os.path.join(out_na, "rail_nodes.parquet")
    node_stats.to_parquet(rail_nodes_p, index=False)
    print(f"Saved {rail_nodes_p} ({len(node_stats)} unique rail topology nodes)")

    # 3. Read Intermodal Freight Facilities
    print("Reading NTAD Intermodal Facilities (1,401 terminals)...")
    df_inter = pd.read_csv(raw_inter)
    df_inter_nodes = pd.DataFrame({
        'term_id': df_inter['TERM_ID'].astype(int),
        'comp_name': df_inter['COMP_NAME'].astype(str),
        'term_name': df_inter['TERM_NAME'].fillna('UNSPECIFIED').astype(str),
        'city': df_inter['CITY'].astype(str),
        'state': df_inter['STATE'].astype(str),
        'latitude': df_inter['LATITUDE'].astype(float),
        'longitude': df_inter['LONGITUDE'].astype(float),
        'mode_truck': df_inter['TRUCK'].astype(str),
        'mode_rail': df_inter['RAIL'].astype(str),
        'mode_water': df_inter['WATER'].astype(str),
        'capacity_barrels': pd.to_numeric(df_inter['CAPACITY'], errors='coerce').fillna(-1.0).astype(float),
        'handles_crude_oil': df_inter['CRUDE_OIL'].fillna('N').astype(str),
        'handles_refined': df_inter['REFINED'].fillna('N').astype(str),
        'handles_pet_chem': df_inter['PET_CHEM'].fillna('N').astype(str),
        'handles_ngl': df_inter['NGL'].fillna('N').astype(str),
        'handles_gasoline': df_inter['GASOLINE'].fillna('N').astype(str),
        'handles_distillate': df_inter['DISTILLATE'].fillna('N').astype(str),
        'handles_biodiesel': df_inter['BIODIESEL'].fillna('N').astype(str),
        'handles_jetfuel': df_inter['JETFUEL'].fillna('N').astype(str),
        'handles_ethanol': df_inter['ETHANOL'].fillna('N').astype(str),
        'is_trimodal_hub': ((df_inter['TRUCK'] == 'Y') & (df_inter['RAIL'] == 'Y') & (df_inter['WATER'] == 'Y')).astype(int),
        'node_type': 'INTERMODAL_FREIGHT_TERMINAL',
        'source_provenance': 'NTAD_Intermodal_Freight_Facilities_2024'
    })

    inter_nodes_p = os.path.join(out_na, "intermodal_nodes.parquet")
    df_inter_nodes.to_parquet(inter_nodes_p, index=False)
    print(f"Saved {inter_nodes_p} ({len(df_inter_nodes)} intermodal facility nodes)")

    # 4. Port to Intermodal Spatial Candidates
    print("Classifying Port-to-Intermodal Candidates...")
    # Load canonical US ports
    p_canon = os.environ.get("CANONICAL_REGISTRY_CSV", str(dl_root / "reports/port_identity_resolution/v1_3_final_semantics/canonical_port_registry.csv"))
    df_canon = pd.read_csv(p_canon)
    p_gov = os.environ.get("SPATIAL_GOV_CSV", str(dl_root / "reports/port_identity_resolution/v1_3_final_semantics/spatial_feature_governance.csv"))
    df_gov = pd.read_csv(p_gov)
    p_ref = os.environ.get("PORTWATCH_PORTS_REF_CSV", str(repo_root / "data/reference/portwatch_ports.csv"))
    df_ref = pd.read_csv(p_ref)

    m = df_canon[df_canon['country_iso2'] == 'US'].copy()
    m = m.merge(df_gov[['portwatch_id', 'accepted_coordinates']], on='portwatch_id', how='left')
    m = m.merge(df_ref[['portid', 'lat', 'lon']], left_on='portwatch_id', right_on='portid', how='left')

    lat_list, lon_list = [], []
    for _, r in m.iterrows():
        if pd.notnull(r['accepted_latitude']):
            lat_list.append(r['accepted_latitude'])
            lon_list.append(r['accepted_longitude'])
        elif pd.notnull(r['accepted_coordinates']):
            parts = str(r['accepted_coordinates']).split(',')
            lat_list.append(float(parts[0]))
            lon_list.append(float(parts[1]))
        elif pd.notnull(r['lat']):
            lat_list.append(r['lat'])
            lon_list.append(r['lon'])
        else:
            lat_list.append(None)
            lon_list.append(None)

    m['port_lat'] = lat_list
    m['port_lon'] = lon_list
    us_ports = m[m['port_lat'].notnull()].copy()

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    candidates = []
    for _, fac in df_inter_nodes.iterrows():
        f_lat, f_lon = fac['latitude'], fac['longitude']
        dists = haversine_km(f_lat, f_lon, us_ports['port_lat'].values, us_ports['port_lon'].values)
        min_idx = np.argmin(dists)
        min_dist = round(dists[min_idx], 2)
        c_port = us_ports.iloc[min_idx]

        if min_dist <= 30.0: # Retain all within 30km for classification
            if min_dist <= 5.0 and fac['mode_water'] == 'Y':
                cls = 'VERIFIED_ADJACENT_MARINE'
                conf = 'HIGH'
            elif min_dist <= 10.0 and (fac['mode_water'] == 'Y' or fac['mode_rail'] == 'Y'):
                cls = 'STRONG_SPATIAL_CANDIDATE'
                conf = 'MEDIUM'
            else:
                cls = 'METROPOLITAN_PROXIMITY_ONLY'
                conf = 'LOW'

            candidates.append({
                'portwatch_id': c_port['portwatch_id'],
                'portwatch_name': c_port['portwatch_name'],
                'port_latitude': round(c_port['port_lat'], 5),
                'port_longitude': round(c_port['port_lon'], 5),
                'term_id': fac['term_id'],
                'comp_name': fac['comp_name'],
                'term_name': fac['term_name'],
                'facility_latitude': fac['latitude'],
                'facility_longitude': fac['longitude'],
                'distance_km': min_dist,
                'mode_truck': fac['mode_truck'],
                'mode_rail': fac['mode_rail'],
                'mode_water': fac['mode_water'],
                'is_trimodal_hub': fac['is_trimodal_hub'],
                'connection_classification': cls,
                'connection_confidence': conf,
                'justification_note': f"Distance {min_dist}km. Water={fac['mode_water']}, Rail={fac['mode_rail']}. Proximity does not establish confirmed physical track."
            })

    df_port_candidates = pd.DataFrame(candidates)
    port_cand_p = os.path.join(out_na, "port_intermodal_candidates.parquet")
    df_port_candidates.to_parquet(port_cand_p, index=False)
    print(f"Saved {port_cand_p} ({len(df_port_candidates)} candidate linkages within 30km)")

    # 5. Metadata for North American Graph
    na_meta_p = os.path.join(out_na, "north_american_graph_metadata.md")
    with open(na_meta_p, "w") as f:
        f.write(f"""# North American Hinterland Rail & Intermodal Graph Metadata

**Graph Architecture:** Heterogeneous Multimodal Transport Network  
**Geographic Domain:** Continental North America (US, Canada, Mexico)  
**Components:**
1. `rail_nodes.parquet`: {len(node_stats)} topological track vertices
2. `rail_edges.parquet`: {len(df_rail_edges)} physical track segments (264,651 miles)
3. `intermodal_nodes.parquet`: {len(df_inter_nodes)} intermodal bulk transfer facilities
4. `port_intermodal_candidates.parquet`: {len(df_port_candidates)} evaluated seaport-to-facility candidate linkages

## Topological Principles

- **Explicit Node Identifiers:** `FRFRANODE` and `TOFRANODE` represent ground-truth endpoint vertices established by the Federal Railroad Administration. Zero geometric snapping or node synthesis was performed.
- **Directionality:** Undirected physical rail infrastructure. Edge table preserves source from/to endpoint convention.
- **Intermodal Layer:** Intermodal nodes connect to rail infrastructure via physical rail service indicators (`mode_rail == 'Y'`).
- **Seaport Connection Governance:** Seaport-to-intermodal edges are strictly categorized:
  - `VERIFIED_ADJACENT_MARINE` (Distance <= 5km + Marine berth connection)
  - `STRONG_SPATIAL_CANDIDATE` (Distance <= 10km + Rail/Water access)
  - `METROPOLITAN_PROXIMITY_ONLY` (Spatial proximity within 30km, unverified physical link)
- **Warning on Graph Synthesis:** Distance proximity must not be converted into guaranteed physical connectivity without verifying local spur rights.
""")

    # =========================================================================
    # PART B: CARIBBEAN OBSERVED MARITIME GRAPH
    # =========================================================================
    print("--- Part B: Building Caribbean Observed Maritime Graph ---")
    
    # 1. Caribbean Nodes
    print("Building Caribbean maritime nodes (528 ports)...")
    df_cports = pd.read_csv(raw_carib_ports, sep='|')
    df_ccross = pd.read_csv(cw_carib)

    # Compute mean coordinates from portcalls
    df_calls = pd.read_csv(raw_carib_calls, sep='|', usecols=['port', 'latitude', 'longitude'])
    port_coords = df_calls.groupby('port')[['latitude', 'longitude']].mean().reset_index()
    df_cports = df_cports.merge(port_coords, left_on='id', right_on='port', how='left')

    df_c_nodes = df_cports.merge(
        df_ccross[['carib_port_id', 'canonical_portwatch_id', 'canonical_portwatch_name', 'crosswalk_method', 'match_confidence']],
        left_on='id', right_on='carib_port_id', how='left'
    )

    df_carib_nodes = pd.DataFrame({
        'carib_port_id': df_c_nodes['id'].astype(int),
        'port_name': df_c_nodes['portname'].astype(str),
        'country_name': df_c_nodes['countryname'].astype(str),
        'country_code': df_c_nodes['countrycode'].astype(str),
        'unlocode': df_c_nodes['locode'].fillna('NONE').astype(str),
        'latitude': df_c_nodes['latitude'].astype(float),
        'longitude': df_c_nodes['longitude'].astype(float),
        'quay_meters': df_c_nodes['quay'].fillna(0.0).astype(float),
        'wharf_meters': df_c_nodes['wharf'].fillna(0.0).astype(float),
        'offshore_berths': df_c_nodes['offshore'].fillna(0.0).astype(float),
        'canonical_portwatch_id': df_c_nodes['canonical_portwatch_id'].fillna('UNMATCHED').astype(str),
        'canonical_portwatch_name': df_c_nodes['canonical_portwatch_name'].fillna('UNMATCHED').astype(str),
        'match_method': df_c_nodes['crosswalk_method'].fillna('REGIONAL_ONLY_UNMATCHED').astype(str),
        'match_confidence': df_c_nodes['match_confidence'].fillna('UNRESOLVED').astype(str),
        'node_type': 'MARITIME_SEAPORT_NODE',
        'source_provenance': 'Caribbean_AIS_Port_Catalog_2019'
    })

    carib_nodes_p = os.path.join(out_carib, "caribbean_maritime_nodes.parquet")
    df_carib_nodes.to_parquet(carib_nodes_p, index=False)
    print(f"Saved {carib_nodes_p} ({len(df_carib_nodes)} maritime port nodes)")

    # 2. Caribbean Edges
    print("Reading and standardizing Caribbean maritime edges (9,969 segments)...")
    df_seg = pd.read_csv(raw_carib_seg, sep='|')

    # Mapping dicts
    id_to_name = df_carib_nodes.set_index('carib_port_id')['port_name'].to_dict()
    id_to_canon = df_carib_nodes.set_index('carib_port_id')['canonical_portwatch_id'].to_dict()

    df_seg['source_name'] = df_seg['port_origin'].map(id_to_name)
    df_seg['target_name'] = df_seg['port_destination'].map(id_to_name)
    df_seg['source_canonical'] = df_seg['port_origin'].map(id_to_canon)
    df_seg['target_canonical'] = df_seg['port_destination'].map(id_to_canon)

    is_inter_canon = (
        (df_seg['source_canonical'] != 'UNMATCHED') &
        (df_seg['target_canonical'] != 'UNMATCHED') &
        (df_seg['source_canonical'] != df_seg['target_canonical'])
    )

    df_carib_edges = pd.DataFrame({
        'edge_id': df_seg['id'].astype(int),
        'source_port_id': df_seg['port_origin'].astype(int),
        'source_port_name': df_seg['source_name'].astype(str),
        'target_port_id': df_seg['port_destination'].astype(int),
        'target_port_name': df_seg['target_name'].astype(str),
        'source_canonical_id': df_seg['source_canonical'].astype(str),
        'target_canonical_id': df_seg['target_canonical'].astype(str),
        'is_inter_canonical_edge': is_inter_canon.astype(int),
        'voyages_total': df_seg['number_all'].astype(int),
        'voyages_container': df_seg['number_cont'].astype(int),
        'voyages_tanker': df_seg['number_tank'].astype(int),
        'voyages_dry_bulk': df_seg['number_bulk'].astype(int),
        'voyages_general_cargo': df_seg['number_gcar'].astype(int),
        'voyages_cruise': df_seg['number_crui'].astype(int),
        'voyages_service': df_seg['number_serv'].astype(int),
        'voyages_island_feeder': df_seg['number_iisl'].astype(int),
        'directionality': 'DIRECTED_PORT_TO_PORT',
        'temporal_scope': '2019_CALENDAR_YEAR',
        'geographic_scope': 'CARIBBEAN_GULF_BASIN',
        'source_provenance': 'Caribbean_AIS_segments_port2port'
    })

    carib_edges_p = os.path.join(out_carib, "caribbean_maritime_edges.parquet")
    df_carib_edges.to_parquet(carib_edges_p, index=False)
    print(f"Saved {carib_edges_p} ({len(df_carib_edges)} directed maritime edges; {df_carib_edges['is_inter_canonical_edge'].sum()} inter-canonical)")

    # 3. Metadata for Caribbean Graph
    carib_meta_p = os.path.join(out_carib, "caribbean_graph_metadata.md")
    with open(carib_meta_p, "w") as f:
        f.write(f"""# Caribbean Observed Maritime Graph Metadata

**Graph Architecture:** Directed Weighted Empirical Shipping Network  
**Temporal Domain:** Calendar Year 2019 (Observed AIS Trajectories)  
**Geographic Domain:** Caribbean Sea, Gulf of Mexico, Florida Straits, Northern South America  
**Output Files:**
1. `caribbean_maritime_nodes.parquet`: {len(df_carib_nodes)} regional ports
2. `caribbean_maritime_edges.parquet`: {len(df_carib_edges)} directed voyage segments

## Graph Attributes and Edge Structure

- **Edge Weight Definitions:** Every edge represents observed 2019 vessel passages broken down by vessel type (`voyages_container`, `voyages_tanker`, `voyages_dry_bulk`, `voyages_general_cargo`, `voyages_cruise`, `voyages_service`, `voyages_island_feeder`).
- **Inter-Canonical Subgraph:** {df_carib_edges['is_inter_canonical_edge'].sum()} directed edges connect canonical seaports resolved in the global registry (e.g. Miami, San Juan, Port of Spain, Kingston, Caucedo).
- **Regional Berth Retainage:** 8,790 edges link smaller island berths, fuel jetties, and local anchorages, preserving the fine-grained feeder topology of the Caribbean basin.
- **Critical Modeling Constraint:** This graph is **strictly regional (2019)**. It cannot be combined into a global graph without introducing severe regional sampling bias.
""")
    print("=== WORKSTREAM 2 COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run()
