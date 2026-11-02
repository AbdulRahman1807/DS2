# Evaluation Engineer — Figures & Visual Diagnostics

This directory documents the visual diagnostic assets supporting the Stage 02 Deep Learning evaluations.

---

## 1. Visual Diagnostics Inventory

| Figure Identifier | Primary Focus | Evaluated Dimension | Canonical Location |
|---|---|---|---|
| `fig_09_temporal_acf_pacf` | Temporal Autocorrelation & Memory | Decay of port-call persistence across port archetypes | `DL/eda/eda_reports/figures/09_temporal_acf_pacf_by_port_archetype.png` |
| `fig_10_rolling_volatility` | Regime Shifts & Shocks | Volatility clusters during 2020-2022 supply chain disruptions | `DL/eda/eda_reports/figures/10_rolling_volatility_and_regime_shifts.png` |
| `fig_11_multimodal_degrees` | Spatial Graph Topology | Network degree distribution and hub concentration | `DL/eda/eda_reports/figures/11_graph_degree_distributions_multimodal.png` |
| `fig_12_intermodal_reachability`| Inland Reachability | Port-to-corridor topological hop distributions | `DL/eda/eda_reports/figures/12_port_to_intermodal_reachability_hops.png` |
| `fig_13_target_correlation` | Feature-Target Alignment | Cross-correlation between trailing 7d/14d means and 7d future sums | `DL/eda/eda_reports/figures/13_feature_target_correlation_heatmap.png` |
| `fig_15_vessel_congestion` | Vessel Class Congestion Dynamics | Bottleneck build-up across container vs dry bulk carriers | `DL/eda/eda_reports/figures/15_adb_vessel_class_congestion_dynamics.png` |
| `fig_16_dwell_calibration` | Supplementary Anchorage Dwell | Cross-source calibration between PortWatch calls and AIS anchorage waiting time | `DL/eda/eda_reports/figures/16_cross_source_operational_dwell_calibration.png` |

---

## 2. Key Diagnostic Findings
1. **Autoregressive Concentration:** The ACF decays slowly across high-volume container gateways, confirming that trailing 7-day persistence provides an exceptionally strong analytical anchor ($R^2 > 0.96$).
2. **Anchorage Waiting-Time Surges:** Figure 16 confirms that peak outer anchorage waiting times at mega-hubs (e.g. San Pedro Bay / LALB at 79.1h mean) are episodic, explaining why residual formulation over persistence is critical to prevent explosive variance.

