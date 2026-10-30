# Dataset Suitability Assessment

**Question:** is the current dataset an appropriate foundation for three-class dwell prediction at a 0.90 Macro-F1 ambition, and which data would change the answer?
**Short answer:** the event log is the *right kind* of data (a real terminal process log with customs and delivery milestones), and it is substantially richer than Stage 01–02 used. As currently extracted it is **not adequate for a credible 0.90 claim**, for three reasons: (1) it is selected on the outcome (gate-out window); (2) it covers ~9 usable weeks containing one congestion episode, so regime generalisation cannot be tested; (3) the drivers of dwell (customs-declaration timing, delivery-order release, trucking appointment) are recorded only as outcome milestones, without their upstream causes. A better foundation is **the same terminal log, extended in time and joined to customs-declaration (PIB) and truck-appointment records**. No public dataset is a better substitute.

---

## 1. What each row represents and where fields come from

Source: Mendeley Data "Event Log Dwelling Time Dataset" (DOI 10.17632/yvp2b4rtp3, v2 2024-06-26; Prasetyo, Sarno, Sungkono, Waspada, Budiraharjo; *Data in Brief* 57:110906, Dec 2024). The dataset page says the data were withdrawn over **6 months** from the terminal information system but the published log covers **3 months**, and the source institution is anonymised. The repository file `Dataset of Dwelling Time Process Fix.xlsx` matches that structure.

| Field / activity | Provenance (best evidence) | Notes |
|---|---|---|
| `case_id` | container number + vessel call date | 95,648 stays, 81 repeat containers |
| `BAPLIE` (+ `CTR_SIZE`, `CTR_TYPE`) | EDI stowage plan from the vessel | median 28 h before discharge |
| `VESSEL_ATB` | actual time of berthing | 235 distinct values = 235 vessel calls |
| `DISCHARGE` (+ `GROSS`, `YARD_BLOCK`, `YARD_SLOT`) | quay crane discharge with planned yard position | quarantine blocks (Z, Y, YY, W1) used only for flagged boxes |
| `STACK`, `STACK_MAIN_YARD`, `STACK_TRK_BF_MAIN_YARD` | yard moves (RTG) | |
| `HAS_QUARANTINE_FLAG`, `FIRST_STACK_QUARANTINE_BLK`, `QUARANTINE_RELEASE` | agricultural/health quarantine hold | 28% of stays flagged, 67% of flags before discharge |
| `SP_RED_FLAG_PIB` | customs red-lane designation of the import declaration (PIB) | 5.4% of stays, 54% before discharge |
| `FIRST_JOB_CUSTOMS_BEHANDLE`, `FIRST_STACK_BEHANDLE`, `LAST_COMPLETE_BEHANDLE` | customs physical examination ("behandle") | 4.1% of stays |
| `CUSTOMS_DEL` | customs release (SPPB) date | **date-only**, 100% of stays, 40% before discharge by raw date |
| `JOB_DEL` (+ `JOB_DEL_DOCTYPE`: SPPB 93%, BC23 6%, …) | delivery job created in TOS after documents are complete | median +32 h after discharge |
| `JOB_PLP` | transfer to another bonded area (PLP) | 0.5% |
| `TRUCK_IN`, `UNSTACK_TO_TRUCK`, `TRUCK_OUT` | gate-in of truck, yard pick, gate-out | dwell ends at TRUCK_OUT end |
| `Duration`, start timestamps | start = end − U(17.6, 90.9) min | looks synthetic (likely anonymisation or simulation of service time); do not use sub-1.5 h timing |
| `Role` | resource role per activity | not used; constant per activity |

Interpretation of activity semantics comes from Indonesian customs practice (PIB, SPPB, BC 2.3, behandle) and is consistent with the observed ordering, but the *Data in Brief* article should be used to confirm it before any operational claim.

---

## 2. Representativeness, exclusions and censoring

| Question | Answer (evidence) |
|---|---|
| Missing observations / exclusions | Systematic exclusion of every stay whose gate-out is outside 2021-07-01 → 2021-09-30. Only 1 stay lacks TRUCK_OUT. No duplicated case IDs. |
| Only completed stays? | **Yes.** No unfinished stays exist in the extract, so survival methods cannot recover censored information that was never extracted. |
| Survivorship / censoring effect | June discharges: only long stays (100% CRITICAL before 06-21). September discharges: long stays missing (~53% of CRITICAL stays missing in the legacy test window). |
| Representative of the prediction population? | Only inside the selection-aware cohort (discharges 07-01 → 09-05, 69,453 stays), where at most ~2.3% of CRITICAL stays can be missing. |
| Class prevalence over time | CRITICAL 27.9% (week of 07-12) → 7.0–8.0% (late August), inside the clean cohort. One congestion episode only. |
| Regime changes | July congestion (genuine). Vessel-call-level CRITICAL rate ranges from 2.8% (p10) to 30.1% (p90) across calls with ≥ 50 boxes, i.e. vessel/consignee mix matters more than container attributes. |
| Temporal patterns the legacy split misses | Label embargo (labels resolve up to 161 h after discharge), discharge-week regime, vessel-call clustering of outcomes. |
| Thresholds appropriate? | They are p50/p85 of a biased window, not SLAs. Unbiased cohort p50/p85 = 60.2 / 150.9 h. Indonesian practice tracks dwell against 3-day targets, so 72 h is a natural operational ON_TIME cut, but changing the target is out of scope unless the business defines it. |
| Classes distinguishable? | At discharge, weakly (see `predictability_and_ceiling_analysis.md`). Once customs-release timing is known, almost deterministically (release at day 5–8 after discharge → 83% CRITICAL; at or before discharge day → 85% ON_TIME). |
| Useful fields discarded | Customs release date, quarantine and red-flag *timing* (only "ever" flags were used), vessel-call identity, yard inventory, delivery-order and truck-in milestones (for landmark/event formulations). |

---

## 3. Candidate external and supplementary data

Ranked by expected value to the 3-class problem multiplied by feasibility. "Join key" is how it attaches to the event log.

| Rank | Source | Contribution mechanism | Availability / feasibility | Join key and timestamps | Expected value |
|---|---|---|---|---|---|
| 1 | **Full 6-month withdrawal / extended extract of the same TOS log** (dataset authors, ITS Surabaya) | Removes outcome selection (include stays still in yard at extract end, i.e. right-censored rows), adds regimes beyond July 2021 so temporal generalisation can actually be tested, and gives an uncensored independent test window. | High: authors state 6 months were withdrawn; contact the corresponding author. | identical schema | **Necessary for any credible final claim.** |
| 2 | **Customs declaration (PIB) lifecycle from CEISA / INSW**: PIB submission time, lane (green/yellow/red/MITA), document requests (SPJM/SPJK), payment, SPPB time | Customs release timing is the dominant driver of dwell (§2 and the oracle experiments). The *submission* time and lane are upstream causes known before release, often before discharge. This turns customs from an outcome into a predictor. | Medium: needs a data-sharing agreement with the terminal or a customs broker. The terminal itself receives SPPB electronically. | container number + B/L; precise timestamps | **Highest predictive value.** |
| 3 | **Delivery order (DO) and invoice/payment status** (shipping-line portal / terminal billing) | JOB_DEL→TRUCK_OUT median 13 h, but customs→JOB_DEL varies from hours to days because of DO and payment. DO release time narrows AT_RISK vs CRITICAL. | Medium: terminal billing holds payment; DO from line or terminal portal. | container number; timestamps | High |
| 4 | **Truck appointment / gate booking** | TRUCK_IN→TRUCK_OUT is ~2 h. A booked appointment almost fixes the exit time. Most useful for landmark predictions. | Medium-high where a TAS exists. | container number; booking time | High at landmarks, little at discharge |
| 5 | **Consignee / importer and B/L identifiers** (anonymised IDs are enough) | Importer behaviour (pre-clearance habit, broker quality) explains much of the vessel-call heterogeneity. Supports temporally-safe target encoding of past dwell for the same importer. | Medium: in the same TOS, removed by anonymisation. | case_id | High |
| 6 | **Real shipping-line / service ID and vessel identity** | `carrier_prefix` is the container owner (often a lessor). Line and service carry customs/DO process differences. | High (TOS); public vessel IDs from AIS. | vessel call | Medium |
| 7 | **Yard occupancy and equipment moves** (RTG moves, reshuffles, block occupancy) | Captures physical congestion and retrieval delay; partly reconstructable already from the log (inventory). | High within TOS | timestamps | Low-medium (gate/doc delays dominate) |
| 8 | **IMF PortWatch daily port activity, Surabaya (portid `port1237`, LOCODE ID SUB)** | Daily vessel calls and trade volumes; an external congestion/regime signal with long history (2019+). The repository's `portwatch_daily_ports.parquet` is **empty (0 rows)**: the fetch failed silently. | High, public API | date (lag ≥ 1 day) | Low-medium; only regime-level |
| 9 | Public holidays (Indonesian national/religious calendar) and customs working days | Customs and DO offices close; clearance stalls over long weekends. | High, public | date | Low-medium, cheap |
| 10 | Weather (BMKG / ERA5 for Tanjung Perak) | Crane stoppages, gate closures | High, public | hour/date | Low (dwell is document-driven) |
| — | PortWatch/GDACS disruptions (already ingested) | No Indonesian events in 2021 | — | — | None; drop |
| — | Other public dwell datasets (US BTS port dwell, MarineTraffic vessel dwell) | Vessel-level or aggregate only, no container-level customs milestones | Public | none | Not a substitute |

**Do not** download unrelated public data to grow row counts: more rows from the same regime do not address the information gap.

---

## 4. Verdict

1. Keep the Mendeley/terminal event log as the foundation. It is the only source here with container-level customs and delivery milestones.
2. Immediately rebuild the modelling table from the event log with availability timestamps (done: `ML/ml_engineer/reassessment/cohort.py`, `asof_features.py`), restricted to the selection-aware cohort.
3. Request the full 6-month withdrawal **including stays still in the yard at extraction time**, then customs-declaration (PIB) timestamps and lanes. These two requests decide whether a high-Macro-F1 prediction can be made *early*. Without (2), high Macro-F1 is only reachable at prediction times after customs release (see the strategy document).
4. Treat the legacy `test.parquet` as permanently unsuitable for CRITICAL-class evaluation.
