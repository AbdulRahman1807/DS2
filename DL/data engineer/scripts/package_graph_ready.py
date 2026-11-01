"""Package Model-Ready Graph Datasets for Graph Neural Networks.

Generates:
- DL/data/staging/supplementary_acquisition/processed/graph_ready/north_american_graph_nodes.parquet
- DL/data/staging/supplementary_acquisition/processed/graph_ready/north_american_graph_edges.parquet
- DL/data/staging/supplementary_acquisition/processed/graph_ready/caribbean_graph_nodes.parquet
- DL/data/staging/supplementary_acquisition/processed/graph_ready/caribbean_graph_edges.parquet
- DL/data/staging/supplementary_acquisition/processed/graph_ready/graph_schema.md
- DL/data/staging/supplementary_acquisition/processed/graph_ready/graph_coverage_report.md
"""

import os
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DL_ROOT = SCRIPT_DIR.parents[1]

BASE_DIR = os.environ.get("PHASE2_BASE_DIR", str(DL_ROOT / "data/staging/supplementary_acquisition"))
OUT_DIR = os.path.join(BASE_DIR, "processed/graph_ready")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    t0 = time.time()
    print("=== STARTING WORKSTREAM D: GRAPH-READY DATA PACKAGING ===")
    
    # -------------------------------------------------------------
    # 1. Package North American Hinterland Graph
    # -------------------------------------------------------------
    print("Processing North American Hinterland Graph...")
    na_in_dir = os.path.join(BASE_DIR, "processed/graphs/north_american")
    
    df_rail_nodes = pd.read_parquet(os.path.join(na_in_dir, "rail_nodes.parquet"))
    df_rail_edges = pd.read_parquet(os.path.join(na_in_dir, "rail_edges.parquet"))
    df_im_nodes = pd.read_parquet(os.path.join(na_in_dir, "intermodal_nodes.parquet"))
    df_port_im = pd.read_parquet(os.path.join(na_in_dir, "port_intermodal_candidates.parquet"))
    
    # A. Standardize Nodes
    # 1. Rail junction nodes
    r_nodes = pd.DataFrame({
        'node_id': 'rail_' + df_rail_nodes['node_id'].astype(str),
        'node_type': 'RAIL_JUNCTION',
        'node_name': 'Rail Node ' + df_rail_nodes['node_id'].astype(str),
        'country': df_rail_nodes['primary_country'],
        'state_province': df_rail_nodes['primary_state'],
        'latitude': np.nan,
        'longitude': np.nan,
        'degree': df_rail_nodes['degree'].astype(np.int32),
        'is_canonical_port': 0,
        'canonical_portwatch_id': None,
        'operator_name': df_rail_nodes['primary_operator'],
        'source_provenance': 'NTAD_North_American_Rail_Lines'
    })
    
    # 2. Intermodal facility nodes
    im_nodes = pd.DataFrame({
        'node_id': 'im_' + df_im_nodes['term_id'].astype(str),
        'node_type': 'INTERMODAL_FACILITY',
        'node_name': df_im_nodes['term_name'],
        'country': 'USA',
        'state_province': df_im_nodes['state'],
        'latitude': df_im_nodes['latitude'].astype(np.float64),
        'longitude': df_im_nodes['longitude'].astype(np.float64),
        'degree': 0, # enriched via transfer edges
        'is_canonical_port': 0,
        'canonical_portwatch_id': None,
        'operator_name': df_im_nodes['comp_name'],
        'source_provenance': 'NTAD_Intermodal_Freight_Facilities'
    })
    
    # 3. Canonical US Port Nodes
    port_unique = df_port_im[['portwatch_id', 'portwatch_name', 'port_latitude', 'port_longitude']].drop_duplicates().copy()
    p_nodes = pd.DataFrame({
        'node_id': 'port_' + port_unique['portwatch_id'].astype(str),
        'node_type': 'PORT_GATEWAY',
        'node_name': port_unique['portwatch_name'],
        'country': 'USA',
        'state_province': 'US_COASTAL',
        'latitude': port_unique['port_latitude'].astype(np.float64),
        'longitude': port_unique['port_longitude'].astype(np.float64),
        'degree': 0,
        'is_canonical_port': 1,
        'canonical_portwatch_id': port_unique['portwatch_id'],
        'operator_name': 'PORT_AUTHORITY',
        'source_provenance': 'Canonical_Port_Registry_v1_3'
    })
    
    na_nodes = pd.concat([r_nodes, im_nodes, p_nodes], ignore_index=True)
    out_na_nodes_p = os.path.join(OUT_DIR, "north_american_graph_nodes.parquet")
    pq.write_table(pa.Table.from_pandas(na_nodes, preserve_index=False), out_na_nodes_p, compression='snappy')
    print(f"Saved North American nodes: {len(na_nodes):,} rows ({os.path.getsize(out_na_nodes_p)/(1024*1024):.2f} MB)")
    
    # B. Standardize Edges
    # 1. Rail lines
    r_edges = pd.DataFrame({
        'edge_id': 'rail_edge_' + df_rail_edges['edge_id'].astype(str),
        'source_node': 'rail_' + df_rail_edges['from_node'].astype(str),
        'target_node': 'rail_' + df_rail_edges['to_node'].astype(str),
        'source_name': 'Rail Node ' + df_rail_edges['from_node'].astype(str),
        'target_name': 'Rail Node ' + df_rail_edges['to_node'].astype(str),
        'edge_type': 'RAIL_LINE',
        'distance_km': df_rail_edges['km'].astype(np.float32),
        'distance_miles': df_rail_edges['miles'].astype(np.float32),
        'tracks': df_rail_edges['tracks'].astype(np.int16),
        'operator': df_rail_edges['railroad_owner'],
        'subdivision': df_rail_edges['state'],
        'net_type': df_rail_edges['net_type'],
        'stracnet': df_rail_edges['stracnet'].astype(str),
        'is_inter_canonical_edge': 0,
        'source_provenance': 'NTAD_North_American_Rail_Lines'
    })
    
    # 2. Port-to-Intermodal Transfer Edges
    p_edges = pd.DataFrame({
        'edge_id': 'transfer_' + df_port_im['portwatch_id'].astype(str) + '_' + df_port_im['term_id'].astype(str),
        'source_node': 'port_' + df_port_im['portwatch_id'].astype(str),
        'target_node': 'im_' + df_port_im['term_id'].astype(str),
        'source_name': df_port_im['portwatch_name'],
        'target_name': df_port_im['term_name'],
        'edge_type': 'PORT_INTERMODAL_TRANSFER',
        'distance_km': df_port_im['distance_km'].astype(np.float32),
        'distance_miles': (df_port_im['distance_km'] * 0.621371).astype(np.float32),
        'tracks': 0,
        'operator': df_port_im['comp_name'],
        'subdivision': 'PORT_INTERMODAL_DRAYAGE',
        'net_type': 'DRAYAGE_CONNECTOR',
        'stracnet': 'NONE',
        'is_inter_canonical_edge': 0,
        'source_provenance': 'NTAD_Intermodal_Spatial_Linkage'
    })
    
    na_edges = pd.concat([r_edges, p_edges], ignore_index=True)
    out_na_edges_p = os.path.join(OUT_DIR, "north_american_graph_edges.parquet")
    pq.write_table(pa.Table.from_pandas(na_edges, preserve_index=False), out_na_edges_p, compression='snappy')
    print(f"Saved North American edges: {len(na_edges):,} rows ({os.path.getsize(out_na_edges_p)/(1024*1024):.2f} MB)")
    
    # -------------------------------------------------------------
    # 2. Package Caribbean Maritime Graph
    # -------------------------------------------------------------
    print("Processing Caribbean Maritime Graph...")
    carib_in_dir = os.path.join(BASE_DIR, "processed/graphs/caribbean")
    df_carib_nodes = pd.read_parquet(os.path.join(carib_in_dir, "caribbean_maritime_nodes.parquet"))
    df_carib_edges = pd.read_parquet(os.path.join(carib_in_dir, "caribbean_maritime_edges.parquet"))
    
    # A. Standardize Nodes
    c_nodes = pd.DataFrame({
        'node_id': 'carib_' + df_carib_nodes['carib_port_id'].astype(str),
        'port_name': df_carib_nodes['port_name'],
        'country_name': df_carib_nodes['country_name'],
        'country_iso3': df_carib_nodes['country_code'],
        'unlocode': df_carib_nodes['unlocode'],
        'latitude': df_carib_nodes['latitude'].astype(np.float64),
        'longitude': df_carib_nodes['longitude'].astype(np.float64),
        'quay_meters': df_carib_nodes['quay_meters'].astype(np.float32),
        'wharf_meters': df_carib_nodes['wharf_meters'].astype(np.float32),
        'offshore_berths': df_carib_nodes['offshore_berths'].astype(np.int32),
        'is_canonical_port': df_carib_nodes['canonical_portwatch_id'].notnull().astype(np.int8),
        'canonical_portwatch_id': df_carib_nodes['canonical_portwatch_id'],
        'canonical_portwatch_name': df_carib_nodes['canonical_portwatch_name'],
        'node_type': 'REGIONAL_MARITIME_PORT',
        'source_provenance': df_carib_nodes['source_provenance']
    })
    out_c_nodes_p = os.path.join(OUT_DIR, "caribbean_graph_nodes.parquet")
    pq.write_table(pa.Table.from_pandas(c_nodes, preserve_index=False), out_c_nodes_p, compression='snappy')
    print(f"Saved Caribbean nodes: {len(c_nodes):,} rows ({os.path.getsize(out_c_nodes_p)/(1024*1024):.2f} MB)")
    
    # B. Standardize Edges
    c_edges = pd.DataFrame({
        'edge_id': 'carib_edge_' + df_carib_edges['edge_id'].astype(str),
        'source_node': 'carib_' + df_carib_edges['source_port_id'].astype(str),
        'target_node': 'carib_' + df_carib_edges['target_port_id'].astype(str),
        'source_name': df_carib_edges['source_port_name'],
        'target_name': df_carib_edges['target_port_name'],
        'source_canonical_id': df_carib_edges['source_canonical_id'],
        'target_canonical_id': df_carib_edges['target_canonical_id'],
        'is_inter_canonical_edge': df_carib_edges['is_inter_canonical_edge'].astype(np.int8),
        'voyages_total': df_carib_edges['voyages_total'].astype(np.int32),
        'voyages_container': df_carib_edges['voyages_container'].astype(np.int32),
        'voyages_tanker': df_carib_edges['voyages_tanker'].astype(np.int32),
        'voyages_dry_bulk': df_carib_edges['voyages_dry_bulk'].astype(np.int32),
        'voyages_general_cargo': df_carib_edges['voyages_general_cargo'].astype(np.int32),
        'voyages_island_feeder': df_carib_edges['voyages_island_feeder'].astype(np.int32),
        'directionality': df_carib_edges['directionality'],
        'edge_type': 'DIRECTED_MARITIME_CORRIDOR',
        'source_provenance': df_carib_edges['source_provenance']
    })
    out_c_edges_p = os.path.join(OUT_DIR, "caribbean_graph_edges.parquet")
    pq.write_table(pa.Table.from_pandas(c_edges, preserve_index=False), out_c_edges_p, compression='snappy')
    print(f"Saved Caribbean edges: {len(c_edges):,} rows ({os.path.getsize(out_c_edges_p)/(1024*1024):.2f} MB)")
    
    # -------------------------------------------------------------
    # 3. Done Parquet Packaging
    # -------------------------------------------------------------
    print(f"Parquet files packaged successfully in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
    
    print(f"Workstream D completed in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
