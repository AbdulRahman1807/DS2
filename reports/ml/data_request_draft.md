# Data Request — DRAFT (not sent)

**To:** Prof. Riyanarto Sarno, Department of Informatics, Institut Teknologi Sepuluh Nopember (corresponding author; `riyanarto@if.its.ac.id`, as published in *Data in Brief* 57:110906)
**Cc:** Dr. Hanung Nindito Prasetyo (ITS / Telkom University), first author and data curator
**From:** [name, role, institution, supervisor] — *to be completed by the project team*
**Subject:** Research request: extended version of the "Event Log Dwelling Time Dataset" (DOI 10.17632/yvp2b4rtp3) for dwell-time prediction

---

Dear Prof. Sarno and Dr. Prasetyo,

We are using your Event Log Dwelling Time Dataset (Mendeley Data v1/v2; *Data in Brief* 57:110906, 2024) in an academic project on predicting import-container dwell time from process milestones (three classes: ≤ 65 h, 65–161 h, > 161 h). The dataset has been very valuable, and we would like to share a finding and ask whether a slightly extended version could be made available under your existing agreement with the data owner.

**What we found.** Every stay in the published file has its gate-out between 1 July and 30 September 2021, while discharges go back to 12 May 2021. This matches the monthly extraction convention described in Jokonowo, Sarno et al. (IJEECS 16(1), 2019). It is ideal for monthly KPI analysis, but for prediction it means long stays are over-represented among June arrivals and under-represented among September arrivals, and containers still in the yard on 30 September are absent. In our simulations this removes essentially all > 161 h stays among the most recent week of arrivals at any cutoff. We are happy to share the analysis and code.

**What we are asking for, in order of priority.** We understand that some items may be impossible under the ethics agreement; any subset helps.

### Tier 1 — items your article indicates were already captured (4 June – 9 November 2021)
1. The same event log, but **selected by discharge date** (all import containers discharged 1 June – 31 October 2021, or the widest available range), **including containers still in the yard at the extraction cutoff** (no gate-out event), with the extraction cutoff timestamp stated.
2. The **exact time of day** for `CUSTOMS_DEL` (currently date only) and for `SP_RED_FLAG_PIB` where available.
3. `CONTAINER_KEY` and an **anonymised (hashed) Importer ID**, which the article (§3) says is recorded. If permitted, also an anonymised customs-broker ID and B/L grouping key, so containers in the same consignment can be linked.
4. The **cost attributes** mentioned in the article (§3), and any additional activities present in the raw capture but not in the 20 published activities (e.g. billing / SP2 / payment, document or status updates).
5. If the start timestamps / durations were generated during anonymisation, a short note on the method, so we do not model artefacts.

### Tier 2 — terminal-held records (would need the data owner's consent)
6. Per container: **SPPB / BC 2.3 receipt time** (TPS Online), **delivery request (SP2 / booking) time**, **invoice and payment time**, and **delivery-order release time** if recorded.
7. Quarantine permit/release times, if the terminal receives them.

### Tier 3 — customs-side records (likely only via the data owner or the customs office)
8. Per container: **PIB registration time, lane (green / yellow / red / MITA), SPJM/SPJK time**, inspection start/end. We would apply separately through the appropriate channel if you advise.

### Optional
9. Access to the 2015 TOS extract used in Jokonowo et al. (2019) (102,328 cases, gate-out July–September 2015) for out-of-period validation, ideally re-extracted by discharge date.
10. Advice on whether a more recent period (2025 onward, after the Terminal Booking System trial) could ever be considered, since truck-appointment data appear to be the remaining missing piece for high accuracy.

**Format.** Same structure as the published file (one row per event: case_id, activity, start/end timestamp, attributes), CSV/XLSX/Parquet, plus a short field dictionary. Identifiers may be hashed with a key we never see.

**Data protection and use.**
- Use limited to non-commercial academic research.
- No redistribution. Stored on access-controlled storage and deleted on request.
- No attempt to re-identify the terminal, companies or persons. We would never name the data owner, matching your ethics agreement.
- We will sign any data-use agreement you or the data owner require, cite the dataset and article, and offer acknowledgement or co-authorship as you consider appropriate.
- We will share our code, the bias analysis and any derived results with you before publication.

**Why it matters.** With the published file, our best honest model reaches Macro-F1 ≈ 0.40 at discharge and ≈ 0.54 at 48 h after discharge. Exact knowledge of customs-release and delivery-job times raises a model's ceiling to ≈ 0.83, so the information in Tier 1–2 decides whether early, accurate dwell prediction is possible for this terminal.

Thank you very much for considering this. We would be glad to discuss on a short call.

Kind regards,
[name, title, institution, email]

---

### Internal notes (not part of the message)
- **Contact route:** the corresponding author's email is published in the article. Send from an institutional address, with a supervisor in copy and a one-page project summary attached. If no reply in 3 weeks, follow up once, then contact Dr. Prasetyo (Telkom University, Department of Information System Diploma).
- **Alternative routes, if the authors cannot share:** (a) a direct research agreement with the terminal operator (Pelindo Terminal Petikemas / TPS Surabaya) through its corporate secretary, citing the ITS precedent; (b) KPPBC TMP Tanjung Perak (customs) for PIB/lane/SPJM data under a research MoU. A DJBC PPID public-information request is unlikely to yield record-level data.
- **Licences:** dataset CC BY 4.0 (Mendeley); article CC BY-NC 4.0. The ethics agreement forbids naming the owning institution; keep that in all outputs.
- **Funding acknowledged by the authors:** ADB HETI, ITS Flagship, PTUPT. Collaboration fits these programmes.
