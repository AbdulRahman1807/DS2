# Container-level port/terminal datasets that could replace or complement the TPS Surabaya dwell-time event log

Scope: container- or shipment-level datasets (any country) with customs/clearance or terminal-release timestamps plus gate-out. National customs open-data portals and WCO Time Release Studies are excluded (covered by another researcher). About 25 tool calls were used. Several publisher pages (Taylor & Francis, ScienceDirect, MDPI) returned HTTP 403, so details for those papers come from search snippets or citing papers and are marked as such.

## Q1. Which repository datasets (Mendeley, Zenodo, Figshare, Kaggle, UCI, Dataverse, data.world, IEEE DataPort, 4TU, BPI/process-mining) contain container/port/terminal event logs?

### Takeaway
Only one public, container-level terminal event log with gate-out was found: the TPS Surabaya Mendeley dataset already in use. No repository dataset found contains customs declaration, customs release, delivery order or appointment timestamps per container. The other hits are vessel-level, synthetic, or unrelated.

### Cited Findings
- **Mendeley "Event Log Dwelling Time Dataset"** (DOI 10.17632/yvp2b4rtp3, v1 and v2; v2 dated June 2024; Prasetyo, Sarno, Wijaya et al., ITS Surabaya). This is the current dataset. It covers the span from discharge to exit through the main gate. Logs were extracted from the terminal information system over a 6-month window and cover about 3 months of activity. — [Mendeley v2](https://data.mendeley.com/datasets/yvp2b4rtp3/2); [v1](https://data.mendeley.com/datasets/yvp2b4rtp3/1)
- The companion paper is in Data in Brief, vol. 57 (Dec 2024), article 110906 (DOI 10.1016/j.dib.2024.110906). It has case, activity and timestamp fields plus worker, waiting-time and cost attributes. Truck_In and Truck_Out each have 95,647 events and Unstack_To_Truck has 95,540. The authors state that collection covers "only 3 months a year". — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11424786/); [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340924008692); [PubMed](https://pubmed.ncbi.nlm.nih.gov/39328964/)
- **Zenodo "Dataset Container Tracking"** (record 17875110; Sucianna Rabiha, Binus University; CC BY 4.0). It holds vessel operation logs, one row per vessel cycle from berthing to departure, with 80 attributes over 18 months. It is a single 222 kB xlsx file. The port is not specified. It is vessel-level and not usable for container dwell. — [Zenodo](https://zenodo.org/records/17875110)
- **BPI Challenge logs on 4TU** (2012 loans, 2017 loans, 2018 EU agricultural payments, 2019 purchase-to-pay with 251,654 orders). None is a port or container log. — [BPI 2018](https://data.4tu.nl/articles/dataset/BPI_Challenge_2018/12688355); [BPI 2012](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204)
- **Kaggle:** searches found no container-terminal dwell dataset. Kaggle's search page could not be rendered by the fetch tool, so this was not exhaustive. Logistics datasets such as "Smart Logistics Supply Chain Dataset" (ziya07) are generic shipment data, not port data. — [Kaggle smart logistics](https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset)
- **GitHub maelfabien/Cargo-Dwell-Time** is a small project that predicts dwell from container attributes (reefer, hazardous, weight). Its data provenance and licence were not verified, and no customs fields are mentioned. — [GitHub](https://github.com/maelfabien/Cargo-Dwell-Time); [project page](https://maelfabien.github.io/project/cargo/)
- **Aggregate-only US sources** have no per-container data: BTS monthly average container vessel dwell for the top 25 US ports (vessel dwell, not container dwell), and the Vizion API (near-real-time averages, commercial, no history). — [data.gov BTS](https://catalog.data.gov/dataset/monthly-average-container-vessel-dwell-times-at-the-top-25-u-s-container-ports-januar-2023); [Vizion](https://vizionapi.com/supply-chain-optimzation/port-activity-monitoring)

### Inferences
- The public repository landscape for container-level terminal logs is essentially one dataset, the one already in use. Any dataset with customs, DO or appointment timestamps will have to come from a data-sharing agreement with a terminal or research group, not from a download.
- No hits were found on Figshare, UCI, Harvard Dataverse, data.world or IEEE DataPort for container dwell or terminal event logs. This is an absence of search hits, not a verified exhaustive check of each site.

### Gaps
- Searches inside the Figshare, Dataverse, data.world and IEEE DataPort sites were not run separately, so a niche upload could have been missed.
- Kaggle could not be browsed directly.

## Q2. Which academic dwell-time studies released their data or describe obtainable data, and what fields do they hold?

### Takeaway
The closest functional match to the target ("full chain") is the Pusan National University / Busan terminal study (arXiv 2602.20540, 2026). It has 394,386 import-container records (2024–2025) with EDI states for customs release (CR, which includes DO issue and DO due date), carrier pickup pre-notice (CP, a quasi-appointment) and gate-out. The data is not released. The Veracruz (Mexico) study has about 2M containers with consignee, cargo text and HS codes but no customs timestamps, and is accessible only on the terminal's infrastructure. No study found publishes its data.

### Cited Findings
**Busan, Korea: Kim, Kim, Park, Park & Bae (Pusan National University), arXiv 2602.20540 (Feb 2026)**
- 394,386 records from "an operational container terminal at Busan port between 2024 and 2025". Records with dwell over 10 days were excluded from training. — [arXiv PDF](https://arxiv.org/pdf/2602.20540)
- EDI states (Table 2):
  - IN: unloaded.
  - CR (Container Release): "Owner completes customs procedures and obtains a release order". A delivery order is issued with a delivery due date t_do.
  - CP (Container Pre-Information Notice): "Carrier submits pickup job information". Typically valid 24–48 h, so it acts as a pickup pre-notice or quasi-appointment.
  - OUT: gate-out.
  - Features include elapsed time since unloading at each EDI state and DO due date minus the EDI time. Base variables are size, type, B/L, country, carrier, day and holiday. Cargo and owner free text was mapped by an LLM to HS codes (4,346 level-3 codes) and KSIC codes (343). — [arXiv PDF](https://arxiv.org/pdf/2602.20540)
- MAE (hours) for CatBoost with standardisation: 33.25 at IN, 23.71 at CR, 1.89 at CP. The authors say customs-release information "reduces uncertainty". They also suggest a TAS could refine predictions once the pickup truck's arrival is confirmed. — [arXiv PDF](https://arxiv.org/pdf/2602.20540)
- No data availability statement was found in the extracted text. Treat the data as not public; the contact route is the corresponding author group (Hyerim Bae, Pusan National University, Dept. of Industrial Engineering).
- Related work from the same group: Lee, Park, Lee, Son, Kim & Bae (2024), Maritime Transport Research, DOI 10.1016/j.martra.2024.100116. It applies process mining and XAI to import dwell at a Busan terminal. The top factors were temperature, weight, voyage number, block, shipping company and discharge month, and the features include union strikes and weather at discharge. Data availability was not verified because the page returned 403. — [DOAJ](https://doaj.org/article/bd1cd4c32e0b4023a6027f6eb6dac83d); [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666822X24000145)

**Veracruz, Mexico: Villalobos, De Unánue, Sobrino, Aké, Cisneros, Lecona, Matadamaz (Tec de Monterrey + Container Terminal Operations, Veracruz), arXiv 2604.06251 (Apr 2026)**
- The data is about 2M containers (operations view) and about 13M crane-movement rows over three years. The terminal and years are anonymised.
- Fields: loading port, dimensions and type, cargo description, consignee, vessel arrival, yard entry and exit dates, country of origin, net and gross weight, hazardous flag, liner client, shipping line and route, HS chapter and section.
- There are no customs, DO or appointment timestamps. Pre-clearance (customs broker inspection service before release) is used only as a target label.
- Access was via controlled extracts processed on the terminal's infrastructure, with sensitive fields removed. The data is not public. — [arXiv HTML](https://arxiv.org/html/2604.06251); [abstract](https://arxiv.org/abs/2604.06251)

**Other studies**
- **Kourounioti et al. (2016), ANN dwell prediction.** A secondary account says it used 13,733 import containers from a Middle Eastern terminal's TOS, with arrival and departure times, a customs-inspection flag and the exact customs inspection date (40.68% inspected). This contradicts the brief's "Piraeus" label; the review places it in the Middle East. Cargo text was grouped into 8 categories. The data is not public. — Account via [Heilig, Stahlbock & Voß, arXiv 1904.13251](https://arxiv.org/pdf/1904.13251) (search snippet; the abstract page did not confirm) and [arXiv 2602.20540](https://arxiv.org/pdf/2602.20540)
- **De Armas Jacomino et al. (2021), Applied Sciences 11(20):9380, ordinal regression.** It used two datasets from one container terminal and is linked to Tec de Monterrey and the University of Salamanca. The terminal was not confirmed (the MDPI page returned 403). The data is not known to be released. — [DOI](https://doi.org/10.3390/app11209380); [Tec record](https://research.tec.mx/vivo-tec/display/AcademicArticleSCO_85117061083)
- **Gaete et al. (2017), Random Forest dwell prediction feeding a positioning DSS.** The port is unverified; from memory it is likely San Antonio, Chile. — [SciTePress](https://scitepress.org/PublishedPapers/2017/61930/pdf/index.html); [ResearchGate](https://www.researchgate.net/publication/314522056_A_Dwell_Time-based_Container_Positioning_Decision_Support_System_at_a_Port_Terminal)
- **Saini & Lerher (2024), Business: Theory and Practice, DOI 10.3846/btp.2024.19205.** It analyses 2.8M container entries from 14 major ports with OLS. Variables are cycle, size, mode, status, delivery and tracking technology, and dwell is measured from discharge to yard exit. Ports are anonymised (Port A, G, …); India is likely but unverified. No customs timestamps are reported, and the data is not public. — [Journal](https://journals.vilniustech.lt/index.php/BTP/article/view/19205); [DOAJ](https://doaj.org/article/ef7a2bccc4884de08f305b627cba7891)
- **Mombasa: "Container dwell time predictive modelling: an application of ML algorithms", Maritime Policy & Management (2025), DOI 10.1080/03088839.2025.2501010.** It uses ML (RF; ordinal approach reported best MAE) on Mombasa container data, filters dwell over 48 h, and finds seasonal and weekly effects. Fields and data access could not be verified (403). — [TRID](https://trid.trb.org/View/2709465); [T&F](https://www.tandfonline.com/doi/full/10.1080/03088839.2025.2501010)
- **Mombasa/KPA Port Community Charter reports (2014–2017)** define a customs-release-to-exit metric but are monthly aggregates, not row-level data. — [TTCA report](https://www.ttcanc.org/sites/default/files/2023-01/Monthly_Port_CommunityCharter_Report_June2015.pdf)
- **Valencia:** a 2015 Valenciaport Foundation TAS study used historical planning and execution data. No public download exists; access would be through Fundación Valenciaport. — [Fundación Valenciaport](https://www.fundacionvalenciaport.com/en/project/analysis-on-the-implementation-of-an-appointment-system-in-container-terminals-in-the-port-of-valencia/)
- "Long Container Dwell Time at Seaport Terminals: An Investigation from a Consignee Perspective" (IJEMS) is a survey or consignee study, not a container log. — [IJEMS](https://ojs.lib.unideb.hu/IJEMS/article/view/9774)

### Inferences
- The Busan dataset is the best-documented analogue of the target feature set: customs release, DO, pickup pre-notice and gate-out per container, at about 4x the size of TPS. It is obtainable, if at all, only through collaboration with the PNU Bae lab. Its reported MAE drop from IN to CR (33 h to 24 h) is direct evidence that customs-release timestamps carry large signal. The near-zero MAE at CP is leakage-like, since pickup is imminent, so CP should be treated as an as-of stage rather than a discharge-time feature.
- The Veracruz data would add HS codes and consignee but not customs timestamps. It is a weaker complement than Busan.

### Gaps
- Not found or not verified: Antwerp, Felixstowe, Rotterdam, Hamburg, Santos, Indian ports (JNPT/ICEGATE), Chittagong, Lagos/Tema and Durban container-level research datasets. No study with released container-level data was found for any of them in this search budget.
- Kourounioti 2016's original paper was not accessed. The terminal location (Middle East, as reported in a review, vs Piraeus) is unresolved.
- Data availability statements for the MPM 2025 Mombasa and Maritime Transport Research 2024 Busan papers could not be read (403).

## Q3. Are there datasets with truck appointment system (TAS) records alongside container release?

### Takeaway
No public TAS-plus-release dataset was found for LA/Long Beach, Valencia, Antwerp, Vancouver or Santos. TAS data appears only inside papers that used private terminal extracts, or behind operator APIs that are limited to authorised users.

### Cited Findings
- A JMSE 2025 paper (DOI 10.3390/jmse13030405) predicts actual truck arrival under a TAS from TOS data (appointment slots, no-shows, actual arrivals). No public download was found. — [DOI](https://doi.org/10.3390/jmse13030405)
- A Sustainability 2025 paper on a novel TAS uses synthetic data. — [MDPI](https://www.mdpi.com/2071-1050/17/13/5740)
- The APM Terminals Import Availability API exposes appointment dates, holds and container status but needs a bearer token and is for authorised partners. — [APMT developer](https://developer.apmterminals.com/api/5)
- The APMT Onne (WACT, Nigeria) TAS portal lets users export only their own appointments. — [APMT Onne](https://www.apmterminals.com/en/onne/e-tools/truck-appointment-system)
- An arXiv paper (2503.11728) on empty-container forecasting for a Vehicle Booking System gives no data link in its abstract; the journal version is in Procedia Computer Science 2024. — [arXiv](https://arxiv.org/abs/2503.11728)
- Busan's CP state ("carrier submits pickup job information", valid 24–48 h) functions as a pickup pre-notice in that dataset, which is not public. — [arXiv 2602.20540](https://arxiv.org/pdf/2602.20540)

### Inferences
- If TAS records are required, the realistic route is a terminal partnership, for example through the TPS operator's own booking system, or a research collaboration such as PNU Busan or Fundación Valenciaport.

### Gaps
- Port Optimizer (LA/LB), Vancouver VBS and Santos appointment data were not investigated in depth. These are believed, unverified, to be aggregate or partner-only.

## Q4. Does any dataset contain the full chain (discharge, declaration, lane/inspection, release, DO/payment, appointment, gate-out)?

### Takeaway
No public dataset was found. The nearest is the private Busan 2024–25 terminal data: discharge, customs release with DO, pickup pre-notice and gate-out, but no declaration filing time or lane. Kourounioti 2016 had a customs inspection date and flag but no release or DO.

### Cited Findings
- Busan EDI chain: IN, then CR (customs done, DO issued with due date), then CP (pickup pre-notice), then OUT. No declaration-filing or lane field is described. — [arXiv 2602.20540](https://arxiv.org/pdf/2602.20540)
- Kourounioti 2016 data: arrival and departure times, a customs-inspection flag and an inspection date for 13,733 containers. — via [arXiv 1904.13251](https://arxiv.org/pdf/1904.13251) (secondary, snippet-level)
- The TPS Mendeley log has terminal activities and truck in/out but no customs events. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11424786/)

### Inferences
- A full-chain dataset would almost certainly need customs data (declaration time, lane) joined to terminal data (discharge, gate-out). Outside national customs systems, that join exists only in proprietary port community systems such as Korea's PORT-MIS/KL-Net EDI or Indonesia's INSW. This inference is not verified for any specific system.

### Gaps
- No evidence was found of any research release combining a customs lane with terminal gate-out at container level.

## Q5. Is the Kaggle "INCOM 2024 delay example dataset" a port dwell dataset?

### Takeaway
No. It is the example (about 15k-row) release of the synthetic supply-chain delivery-delay dataset from the INCOM 2024 Data Challenge, Segment 2, organised by Cambridge's Supply Chain AI Lab. The task is procurement delivery status (early / on time / delayed) from about 40 features. It is not port or container data.

### Cited Findings
- INCOM 2024 Data Challenge, Segment 2 (Supply Chain Competition): organised by Prof. Alexandra Brintrup and Dr. Liming Xu (Cambridge), Dr. George Baryannis (Huddersfield) and Prof. Dmitry Ivanov (Berlin School of Economics and Law). Participants act as procurement professionals predicting delivery arrival status (early, on time or delay), described as multi-label. The contact is Liming Xu. — [INCOM 2024 data challenge](https://www.incom2024.org/data-challenge/)
- Per the IfM page, the final dataset was synthetic, generated from a real dataset, with about 150k samples, 40 features and 1 outcome. A 15k-sample example set was released for testing. — [IfM Supply Chain Data Competition](https://www.ifm.eng.cam.ac.uk/research/supply-chain-ai-lab/data-competition/) (via search snippet)
- The same lab's SynDelay dataset ("generated using an advanced generative model trained on real-world data") was used by the delivery-delay segments of INCOM 2024 and MIM 2025. Code and data are at github.com/lx249/syndelay, with baselines (CatBoost, RF, XGBoost). No licence is stated. — [SynDelay page](https://supplychaindatahub.org/datasets/syndelay/); [GitHub](https://github.com/lx249/syndelay)

### Inferences
- The exact Kaggle page "incom2024_delay_example_dataset" was not found by search. It is very likely a re-upload of the 15k example set. It has no port, discharge, customs or gate-out semantics and should not be used as a replacement or second dataset.

### Gaps
- The Kaggle uploader, licence and exact columns were not verified, and the real source behind the synthetic generator is undisclosed.

## Summary assessment (replacement vs second dataset)

| Dataset | Level | Customs / release / DO / appt | Size | Access | Verdict |
|---|---|---|---|---|---|
| TPS Surabaya Mendeley (current) | container event log | none; truck in/out | ~95k stays | public | baseline |
| Busan terminal (PNU, arXiv 2602.20540) | container | customs release + DO due date, pickup pre-notice, gate-out; HS/KSIC via LLM | 394,386 (2024–25) | private; collaborate with Bae lab | best replacement or second dataset if obtainable |
| Veracruz terminal (Tec de Monterrey, arXiv 2604.06251) | container | HS, consignee, cargo text; pre-clearance label; no timestamps | ~2M | private, on-premise | weak complement |
| Busan XAI (Lee et al. 2024, MTR) | container | weather, strike etc.; customs not verified | unknown | not verified | possible |
| Kourounioti 2016 (Middle East) | container | customs inspection date and flag | 13,733 | not public, old | low value |
| Saini & Lerher 2024 (14 ports) | container | delivery mode/status; no customs timestamps | 2.8M | not public | low |
| Mombasa MPM 2025 | container | not verified | not verified | not verified | unknown |
| Zenodo Container Tracking | vessel | none | small | CC BY 4.0 | not usable |
| INCOM 2024 / SynDelay | synthetic procurement | none | 15k / ~150k | public | not a port dataset |
