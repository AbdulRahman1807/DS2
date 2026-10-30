# Data Provenance and Acquisition Routes

**Status of evidence:** facts below come from the published article full text (Europe PMC, PMC11424786), the Mendeley dataset pages, the ITS repository, and the authors' 2019 IJEECS paper (full PDF). Items marked *unverified* rest on search snippets or could not be confirmed.

---

## 1. The dataset and its publication history

| Item | Fact | Source |
|---|---|---|
| Dataset | "Event Log Dwelling Time Dataset", Mendeley Data, DOI 10.17632/yvp2b4rtp3. v1 published 10 Nov 2023, v2 published 26 Jun 2024. Licence CC BY 4.0. One .xlsx file. No changelog between versions | data.mendeley.com/datasets/yvp2b4rtp3/1 and /2 |
| Article | Prasetyo, Sarno, Wijaya, Budiraharjo, Waspada, Sungkono, "Data of dwelling time process at container terminal: Multi-perspective based dataset", *Data in Brief* 57:110906 (Dec 2024; online 4 Sep 2024), DOI 10.1016/j.dib.2024.110906, PMID 39328964, PMC11424786. Licence CC BY-NC 4.0 | Europe PMC |
| Corresponding author | **Riyanarto Sarno**, `riyanarto@if.its.ac.id` (published in the article) | article front matter |
| Affiliations | Prasetyo: ITS Informatics and Telkom University (Information System Diploma). Sarno, Sungkono: ITS Informatics. Wijaya: Telkom University. Budiraharjo: Institut Teknologi Nasional Bandung. Waspada: Universitas Diponegoro | article |
| Roles | Prasetyo: data curation. Sarno, Waspada, Sungkono: data curation. **Budiraharjo and Waspada: "Data Collector"** | CRediT statement |
| Matching numbers | 952,069 events, 95,648 cases, 20 activities, 12 attributes: identical to our file | article Table 1 |
| Terminal | Not named in the article ("not permitted to mention the name of the agency"). Prasetyo's doctoral thesis (ITS, 2024, repository 114740) states the event log is from **PT Terminal Petikemas Surabaya (TPS)**; the IMARA paper (IEEE Access 2023, same authors) uses TPS as its case study | repository.its.ac.id/114740; doaj.org IEEE Access 2023 |
| Funding | ADB HETI project; ITS Flagship; Ministry PTUPT; ITS PPHKI | acknowledgements |

## 2. How the data were captured (and why it is selected on gate-out)

- "Collection of traffic network and port information system web server logs in a three months period, namely **July, August, and September 2021** where the collection dates are **November 4 and 9 2021**."
- Withdrawal "carried out … for approximately 6 months … from **4 June 2021 to 9 November 2021**." "Raw data … stored in **3 files** … monthly data withdrawals, namely data for July, August, and September 2021."
- The system "records … **Importer ID**, … red route, green route or quarantine, and custom", and "officers performing activities and **costs** can be captured". The published file has none of Importer ID, cost or `CONTAINER_KEY` (the raw container key, mentioned in §4.2).
- Activities without timestamps were removed, i.e. events that did not occur.
- Anonymisation: officer names → roles, owner not named. The synthetic-looking start-time jitter we observed is **not documented**.
- The 2019 IJEECS paper by the same group (Jokonowo, Sarno, Rochimah, Priambodo; TPS TOS extract, **102,328 cases in three gate-out monthly groups, July–September 2015** (earliest discharge 2 Jan 2015)) states the convention explicitly: a process "must be completed in the month" in which it is evaluated. That is the gate-out selection we detected.

**Conclusions.**
1. The raw capture probably contains **June and October–early November 2021** records, which would fix left truncation and give follow-up for September arrivals. Whether stays unfinished on 9 November were kept is **not stated**.
2. **Importer ID, costs and the raw container key were captured but omitted.**
3. A second, older TPS extract exists in the same group: July–September 2015 by gate-out month, with the same right-censoring design as 2021.
4. Redistribution is constrained by a signed ethics agreement with the terminal. Any extension needs the authors' cooperation and probably the owner's consent.

## 3. Who holds what (external process data)

| Data | Holder / system | Researcher route | Confidence |
|---|---|---|---|
| Arrival-based extended extract, Importer ID, costs | Authors (ITS, Sarno group) under agreement with TPS | Email corresponding author (draft in `data_request_draft.md`) | High |
| SPPB/BC23 receipt times; gate realisation | TPS, via **TPS Online** (DJBC–terminal electronic exchange, PER-01/BC/2019, PER-10/BC/2020) | TPS research agreement via the authors | High that the terminal holds it; message names unverified |
| SP2 / billing / payment, import booking | TPS e-service (import booking since 2018); NLE "SP2 Online" covered Tanjung Perak in 2020–2021 | TPS | Medium |
| Delivery order | Shipping lines through INSW/NLE DO Online (Tanjung Perak pilot under PM 120/2017) | LNSW or lines; only aggregates published | High that it exists; low accessibility |
| PIB registration, lane, SPJM, inspection | DJBC **CEISA**; local office KPPBC TMP Tanjung Perak | Research MoU with KPPBC (precedent: PKN STAN study used KPPBC stage times 2020–2022). PPID requests are often refused for record-level data | High (process), Medium/low (access) |
| Quarantine permits | Barantin, IQFAST / PPK Online | Barantin PPID or UPT Karantina Surabaya | Medium |
| Truck appointments | **No TPS appointment system found for 2021**. TPS Terminal Booking System trial started 7 Jul 2025 | Only for 2025+ periods, via TPS | Medium |
| Daily port activity | IMF PortWatch (port1237, AIS-based, 2019+) | Public API; the repository's `portwatch_daily_ports.parquet` is empty because the fetch failed | Medium (2021 coverage unverified) |

## 4. Cheapest realistic route

1. **Email the corresponding author** with the tiered request (`data_request_draft.md`). Tier 1 (already captured: June–November raw capture, unfinished stays, exact customs time, Importer ID, costs) needs no new extraction from the terminal, only re-processing of existing raw files under the existing agreement. Cost: one email plus a data-use agreement.
2. If Tier 1 is granted, ask the authors to sponsor a **Tier 2 request to TPS** for terminal-held SPPB receipt, SP2 and payment times. These live in systems the terminal already exports to customs, but the terminal must consent.
3. Customs-side PIB/lane data (Tier 3) only via a formal MoU with KPPBC Tanjung Perak. It's slow and uncertain, so start in parallel, not first.
4. Truck appointments only exist for 2025+. Pursue only if Tiers 1–3 show the 0.83 bound is approached.

**Do not assume the authors will provide the data.** The article allows "limitations on the form of data used can be distributed" under the ethics agreement, and the dataset was released only after anonymisation. The fallback if they decline is the direct TPS route; failing that, the project must accept the current-data ceiling (≈ 0.55 at landmarks) and state it.
