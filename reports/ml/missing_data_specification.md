# Missing Data Specification

**Goal:** list the process information that determines dwell time and is absent from the published event log. For each field: why it matters, when it becomes available, how to represent it, how it joins, priority, and which prediction stage it enables.
**Evidence for priorities:** `predictability_and_ceiling_analysis.md` (oracles: exact customs date → 0.71; + delivery-job time → 0.83), `formulation_feasibility.csv` (CRITICAL stays leave the open population before their customs/DO events occur), `value_of_information` results in `data_recovery_and_0.90_feasibility.md`.
**Join keys available today:** `case_id` = container number + vessel call date. `CONTAINER_KEY` and **Importer ID** exist in the raw capture (article §3, §4.2) but were not published.

**Key structural finding.** Among stays whose three classes are still open, CRITICAL is 9.4% at discharge but only 3.5% when customs release becomes known and 1.2% at delivery-job creation. A CRITICAL stay has to be recognised **while it is still stuck**. The most valuable fields are therefore **status fields observable before release** (declaration filed? lane? inspection ordered? payment made? DO requested?), not the release timestamps themselves.

Priority: **P0** = essential to move beyond ≈ 0.55; **P1** = high value; **P2** = optional.
Stage: A = discharge, L = landmarks (12/24/48 h), E = event-driven, M = monitoring/cohort.

---

## 1. Customs process (holder: DJBC / CEISA; the terminal receives release permits via TPS Online)

| Field | Why it matters | Available when | Representation | Join | Priority | Stage |
|---|---|---|---|---|---|---|
| Customs declaration (PIB, BC 2.0) **registration timestamp** | Not-yet-filed declarations are the main reason a box cannot leave. Filing before arrival predicts fast exit. The single strongest *upstream* cause still missing | At filing, often before arrival | `pib_registered_at`; as-of features `pib_filed_by_t`, `hours_since_pib`, `pib_lead_vs_discharge_h` | PIB lists container numbers; join on container + B/L | **P0** | A, L |
| Declaration **status history** (registered, document request, payment of duties, approved, rejected) | Distinguishes "waiting for importer" from "waiting for customs" | Each status change | Long table `(container, status, ts)` → as-of last status, age, count of returns | as above | **P0** | A, L |
| **Lane assignment** (green / yellow / red / MITA priority) and its timestamp | Red/yellow lanes add document review or physical inspection (days). Only 5.4% of stays have a red-flag event today and the lane of the rest is unknown | Right after PIB registration | categorical `lane`, `lane_assigned_at` | PIB → container | **P0** | A, L |
| Inspection notice (SPJM/SPJK) timestamp | Formal trigger for physical inspection or document review | Notice issued | `spjm_at`, `spjk_at` | PIB → container | **P0** | L |
| Inspection initiation / completion (behandle) | Partly present (`FIRST_JOB_CUSTOMS_BEHANDLE`, `LAST_COMPLETE_BEHANDLE`, 4.1% of stays); needs completeness and cause | Event time | already in log; add `inspection_result` | case_id | P1 | L |
| **Customs release (SPPB/BC23) exact timestamp** | Current `CUSTOMS_DEL` is date-only, which forces a +24 h availability rule and blurs the 65 h boundary. TPS receives SPPB electronically, with receipt times | At release | `sppb_received_at` (terminal receipt time) | container | **P0** (cheap: terminal-held) | L, E |
| Quarantine (Barantin / IQFAST) status history: permit application, inspection, KT-9 release | 28% of stays carry a quarantine flag; flagged boxes go to dedicated blocks | Each status | long table → as-of status, age | container / permit no. | P1 | A, L |
| Red-flag (`SP_RED_FLAG_PIB`) full history with precise time | 5% of red-flag timestamps are date-only; lane history covers this | Event | as lane | container | P1 (subsumed by lane) | A, L |
| Bonded transfer (BC 2.3, PLP) and other non-gate exits (re-export, auction `LELANG`) | Different exit mechanism; must not be mislabelled | Event | `exit_type` | container | P1 | M |

## 2. Delivery order and payment (holders: shipping line / INSW-NLE DO Online; TPS billing/SP2)

| Field | Why it matters | Available when | Representation | Join | Priority | Stage |
|---|---|---|---|---|---|---|
| **DO application timestamp** | Importer has started the release process; absence at 24–48 h signals CRITICAL | At request | `do_requested_at` → as-of flag, age | B/L → containers | **P0** | A, L |
| **DO issuance / release timestamp** | Customs-to-delivery-job lag varies from hours to days; the DO is one cause | At release | `do_released_at` | B/L → containers | **P0** | L |
| **Terminal delivery request (SP2 / booking) timestamp** | Probably what `JOB_DEL` records now. The request time precedes `JOB_DEL` | At request | `sp2_requested_at` | container | **P0** | L |
| **Invoice issued / payment initiated / payment completed** | Unpaid storage/handling blocks release; payment usually precedes pickup by hours | Each step | `invoice_at`, `paid_at`, `amount_band` | container | **P0** | L |
| Outstanding payment or document flags | Direct "stuck" indicator | Continuous | as-of boolean + age | container | P1 | L |
| **Importer / consignee ID (anonymised)** and customs broker (PPJK) ID | Explains vessel-call heterogeneity (CRITICAL 2.8%–30.1% across calls). Enables temporally safe history features (importer's past dwell) | Known at PIB | hashed ID; history computed from completed stays only | PIB / raw capture | **P0** (already captured; publication omitted it) | A, L |
| Cost attributes from the raw capture (storage, handling) | Accrued cost, a demurrage proxy | Continuous | as-of accrued cost | case_id | P2 | L |

## 3. Trucking and collection (holder: terminal gate / TOS; Terminal Booking System from July 2025)

| Field | Why it matters | Available when | Representation | Join | Priority | Stage |
|---|---|---|---|---|---|---|
| **Truck appointment booking timestamp** and **scheduled slot** | Delivery job → gate-out takes 13 h median, p90 52 h: pickup timing is the residual after the 0.83 oracle | At booking | `appt_booked_at`, `slot_start`, `slot_end` | container | **P0 for ≥ 0.85**, but **no TPS appointment system existed in 2021** (TBS trial began 7 July 2025) | L, E |
| Appointment modification / cancellation / no-show history | Reliability of the slot | Each change | long table → count of changes, last slot | container | P1 (2025+ only) | L |
| Trucking company ID (anonymised) | Carrier reliability | At booking / gate-in | hashed ID | container | P2 | L |
| Truck arrival / gate-in / gate-out | Present (`TRUCK_IN`, `TRUCK_OUT`) | Event | as is | case_id | present | E |

## 4. Terminal operations (holder: TOS; partly reconstructable now)

| Field | Why it matters | Available when | Representation | Join | Priority | Stage |
|---|---|---|---|---|---|---|
| Yard occupancy by block | Congestion regime; slow retrieval | Continuous | as-of occupancy % per block | time + block | P2 (inventory already reconstructed) | A, L |
| Full container movement history (re-handles, shifts) | Retrieval delay, buried boxes | Event | count/age of moves | case_id | P2 | L |
| Yard block/slot **history** (not only discharge position) | Moves to quarantine/behandle blocks | Event | last block at t | case_id | P2 | L |
| Crane and equipment availability, RTG breakdowns | Operational disruptions | Continuous | daily availability | time | P2 | A, L |
| Vessel discharge rate, berth congestion | Regime | Per call | moves/hour, waiting time | vessel call | P2 | A |
| Terminal disruptions, holidays, customs working days | Calendar stalls | Daily | flags | date | P1 (cheap, public) | A, L |
| PortWatch daily calls / volumes (port1237) | External regime | Daily, ≥ 1 day lag | daily series | date | P2 | A |

## 5. Dataset construction fields (all P0)

`extraction_cutoff_ts`; stays **unfinished at the cutoff**; arrivals selected by **discharge date**; context rows for stays already in the yard at the window start; months beyond July–September 2021 (raw capture ran 4 June – 9 November 2021); exit type for non-gate exits.

---

## 6. Minimum viable request (smallest set that tests the 0.90 hypothesis)

1. Arrival-based extract with unfinished stays (June – early November 2021 raw capture). *Holder: dataset authors / TPS.*
2. Importer ID (hashed) and `CONTAINER_KEY` from the raw capture. *Authors.*
3. Exact SPPB/BC23 receipt timestamp and SP2/billing request and payment timestamps. *TPS (terminal-held).*
4. PIB registration time, lane and SPJM time per container. *KPPBC Tanjung Perak / DJBC, or TPS if it stores the TPS Online messages.*
5. For any 0.90 attempt beyond the customs/DO bound: a **2025+ period** extract that includes Terminal Booking System appointments.
