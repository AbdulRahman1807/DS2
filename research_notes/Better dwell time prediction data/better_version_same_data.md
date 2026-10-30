# Routes to a better version of the same TPS (Terminal Petikemas Surabaya) container dwell-time data

Scope: publications, theses and datasets built on TPS or Tanjung Perak container-level data; what each used; whether anything was shared publicly; realistic ways to get access. Research done 2026-10-30. Nobody was contacted. Several publisher sites (IEEE Xplore, DOAJ, Hindawi PDF, JISEBI, ITS full-text PDFs) returned 401, 403 or 502 errors, so some details come from indexes or abstracts only and are marked that way.

## Q1. Papers by Sarno, Prasetyo, Waspada, Budiraharjo, Sungkono, Wijaya, Jokonowo and Rochimah on container dwell-time event logs: data used and extra attributes

### Takeaway
There are two separate TPS extracts:
- **Extract A (Jokonowo era):** a 2015 TOS extract obtained directly from the TPS TOS database administrator. It has 102,328 cases and 531,524 events, with 16 activities and 6 activity groups.
- **Extract B (Prasetyo/Sarno era):** a 2021 web-server extract. 95,648 cases and 952,069 events were published on Mendeley. The 2021 papers (ACISC 2023, the thesis, IMARA) describe richer resource, cost and waiting-time perspectives than the published columns contain.

Neither extract was published with importer or cost fields.

### Cited Findings
**Extract A: 2015 TPS TOS (Jokonowo, Sarno, Rochimah, Priambodo; IJEECS 16(1), 2019)**
- The event data come from the Petikemas Surabaya (TPS) terminal. The authors were "introduced to TOS Database Administrator to retrieve historical data", working with the table owners to understand the code semantics. Data were "taken to measure key performance indicators for three months, i.e., July, August, and September of 2015." — [IJEECS PDF](https://ijeecs.iaescore.com/index.php/IJEECS/article/download/18024/13000); [landing page](https://ijeecs.iaescore.com/index.php/IJEECS/article/view/18024)
- Table 3 gives the size. Coarse scoping: 102,340 cases, 531,585 events, 49 trace variants. Fine scoping: 102,328 cases, 531,524 events, 41 variants. Range is 02/01/2015 15:26 to 30/09/2015 23:47. — [IJEECS PDF](https://ijeecs.iaescore.com/index.php/IJEECS/article/download/18024/13000)
- **Correction to the brief.** The "2 Jan to 30 Sep 2015" range is not a 9-month window of all containers. Table 7 splits the cases by gate-out month: July has 26,920 cases (earliest event 02/01/2015), August 40,515 (earliest 15/02/2015), September 34,893 (earliest 02/05/2015). These are three monthly gate-out cohorts, the same design as the 2021 Mendeley files. Median dwell is 5.5 days and mean 6.07 days. — [IJEECS PDF](https://ijeecs.iaescore.com/index.php/IJEECS/article/download/18024/13000)
- There are 16 activities:
  - Discharge, Stack Date
  - Job BHD, Move_to_BHD_LOC, Job_BHD_Complete, Return_BHD_move_to_Yard (behandle = customs physical inspection)
  - Job_Change_Status
  - Move_to_CFS, JComplete_Stripping, Return_RS_Move_to_Yard
  - Quarantine_Job_BHD, Move_to_TPFT, Quarantine_Job_BHD_Complete
  - Job_DEL_Date, Truck IN (Ext), Truck OUT

  Each event has a Groups label (G1 to G6), and the formal definition includes a resource S(e). — [IJEECS PDF](https://ijeecs.iaescore.com/index.php/IJEECS/article/download/18024/13000)
- The acknowledgements thank TPS's IT Manager, the Business Process Analyst Manager, the TOS Database Administrator, the Port Operation Team and the TPS Board of Directors. This is direct evidence of board-level sign-off for academic extraction. — [IJEECS PDF](https://ijeecs.iaescore.com/index.php/IJEECS/article/download/18024/13000)
- The paper has no data availability statement and no supplementary data (seen in the PDF text). — [IJEECS PDF](https://ijeecs.iaescore.com/index.php/IJEECS/article/download/18024/13000)
- A companion paper is Jokonowo, Sarno, Rochimah, "Extracting audit trail data of port container terminal for process mining", ICTS 2019, DOI 10.1109/ICTS.2019.8850984. A search snippet says it was tested on four port case studies in eastern Indonesia, which is **unverified**: the IEEE page did not load. — [IEEE Xplore](https://ieeexplore.ieee.org/document/8850984)
- Jokonowo, Fatonah, Akhir, "Conformance checking of dwelling time using a token-based method", JISEBI 7(2):129-137, Oct 2021, DOI 10.20473/jisebi.7.2.129-137. It uses an event log from a TOS that records container dwelling time. The full text returned a 502 error, so the period and case count are **unverified**; the authors and design suggest it reuses Extract A. — [JISEBI](https://e-journal.unair.ac.id/JISEBI/article/view/28515)
- A related student work is Firmansyah, Riyaldy, Jokonowo, "Process Mining Dwelling Time at the Container Port using the Alpha Miner Method" (Sistemasi, UNISI), with PM4Py fitness around 0.99. There is also a Mercu Buana repository item with the same Indonesian title. The data source was not verified because the page reset the connection. — [Sistemasi](https://sistemasi.ftik.unisi.ac.id/index.php/stmsi/article/view/1501); [Mercu Buana repo](https://repository.mercubuana.ac.id/89053)

**Extract B: 2021 TPS web-server extract (Prasetyo, Sarno and others)**
- Data in Brief 57:110906 (2024). Raw data were collected 4 June to 9 November 2021. The published log covers July to September 2021 in three monthly files: 952,069 events, 95,648 cases, 12 attributes:
  - Case_ID, Activity, Start Timestamp, End Timestamp, Duration
  - CTR_SIZE, CTR_TYPE, GROSS
  - Yard_Block, Yard_Slot
  - JOB_DEL_DOCTYPE, Role

  The abstract claims cost and waiting-time perspectives, but no cost column is published. The source system records Importer ID, which is not published. Personal names were replaced by roles. The authors signed an ethics agreement with the data owner and do not name the agency. The article concedes the limitation "The data collection time is only 3 months a year." — [PMC11424786](https://pmc.ncbi.nlm.nih.gov/articles/PMC11424786/)
- The Data in Brief article says the dataset underlies four papers:
  - Waspada et al., IEEE Access 2022 (graph-based token replay)
  - Waspada et al., IEEE Access 2023 (hierarchical block detection)
  - Budiraharjo et al., IEEE Access 2023 (IMARA)
  - Prasetyo et al., ACISC 2023 (job shop and resource clustering)

  — [PMC11424786](https://pmc.ncbi.nlm.nih.gov/articles/PMC11424786/)
- Prasetyo, Sarno, Wijaya, Budiraharjo, Waspada, Sungkono, Septiyanto, "Optimizing Decision Making on Business Processes Using a Combination of Process Mining, Job Shop, and Multivariate Resource Clustering", ACISC 2023, DOI 10.1155/2023/3392012, CC BY. The abstract says the event log records "resources such as employee working time and service processing costs", which suggests cost and officer-time variables existed in the working data. The PDF returned 403, so the period, case count and data statement are **unverified**. — [OpenAlex record](https://api.openalex.org/works/doi:10.1155%2F2023%2F3392012); [DOI](https://doi.org/10.1155/2023/3392012)
- Budiraharjo et al., "IMARA: A New Approach to Multi-Attribute Risk Assessment Based on Event Data Weighting (Case Study in a Container Terminal)", IEEE Access, Jan 2023. It takes event logs from information systems as input and evaluates delay risk in dwelling-time processes at TPS Surabaya. Attribute list and size are **unverified**. — [DOAJ](https://doaj.org/article/5e731f945a104884a9a0515fcee3130f)
- Waspada et al., "Graph-Based Token Replay for Online Conformance Checking", IEEE Access 2022, DOI 10.1109/ACCESS.2022.3208098, first author at Universitas Diponegoro. Data details are **unverified**. — [OpenAlex](https://api.openalex.org/works/doi:10.1109%2FACCESS.2022.3208098)
- Indra Waspada's ITS doctoral thesis (2023), "Graph-Based Online Conformance Checking Menggunakan Multi-Perspektif Untuk Deteksi Anomali Pada Proses Bisnis", is listed in the ITS repository. — [ITS repository author list](https://repository.its.ac.id/view/creators/Waspada=3AIndra=3A=3A.html)

**Other TPS or Tanjung Perak process-mining items**
- Yutika Amelia Effendi and Fitri Retrialisca (UNAIR), "Transforming Timestamp of a CSV Database to an Event Log", ICCCE 2021, DOI 10.1109/ICCCE50029.2021.9467226. It converts a TPS TOS database exported as CSV into an import-process event log with single and double timestamps. Sarno is not an author. — [UNAIR scholar](https://scholar.unair.ac.id/en/publications/transforming-timestamp-of-a-csv-database-to-an-event-log/); [UNAIR news](https://unair.ac.id/?p=111337)
- **Not found:** the "Effendi UNAIR 2025, TOS log Jan to Mar 2015, 159,056 cases" paper did not appear in any search. Its figures are **unverified**. It may be a later publication on the same CSV TOS export that ICCCE 2021 used.
- A Telkom University book or thesis, "Analisis dwelling time petikemas di terminal petikemas menggunakan process mining, studi kasus Terminal Peti Kemas Surabaya" (PDF metadata dated 2025-07). It identifies SOP-versus-log mismatches and bottlenecks in document checks and container moves. Author, period and size are not in the abstract; Prasetyo is Telkom faculty, so it is probably his work, which is **unverified**. — [Telkom abstract PDF](https://repositori.telkomuniversity.ac.id/pustaka/files/241293/abstraksi/analisis-dwelling-time-petikemas-di-terminal-petikemas-menggunakan-process-mining-studi-kasus-terminal-peti-kemas-surabaya-dalam-bentuk-buku-karya-ilmiah.pdf)
- An ITS study of Terminal Nilam (Tanjung Perak, a Pelindo terminal, not TPS) used an event log of 341 cases, 16 activities and 21,382 events from Nov 2018 to Mar 2019. Source is a search snippet only, **unverified**. — [ITS iptek JPS](https://iptek.its.ac.id/index.php/jps/article/download/6364/4206)
- An earlier ITS undergraduate thesis (Rizkikurniadi, 2014) ran a simulation of TPS import dwell time: 8.49-day average in 2013, reduced to 4.52 days in scenarios. Full text is restricted. — [ITS repo 98292](https://repository.its.ac.id/98292)

### Inferences
- Extract A (2015) has customs and quarantine behandle, CFS stripping and job-delivery-date activities, which the 2021 Mendeley log either lacks or encodes differently. Combining the two would give two years with consistent gate-out monthly cohorts, but both are right-censored the same way: only containers that exited in the window are included.
- The ACISC abstract and the thesis abstract (cost efficiency up 8.47%, staff performance by TPS, customs, quarantine and PLP roles) show that the working copy of Extract B carried cost, officer and workload data and more than 3 months of raw data (June to November 2021). That unpublished superset is the most realistic "better version" to request.

### Gaps
- Exact attributes, periods and data statements for the three IEEE Access papers and the ACISC paper: full texts were blocked (403/401/502).
- The Effendi 2025 paper with 159,056 cases could not be located.
- Whether ICTS 2019 covers TPS or other eastern-Indonesia ports.

## Q2. Did any paper publish extra data (supplements, GitHub, Mendeley with costs, resources, other months, 2015 logs)?

### Takeaway
No public release beyond Mendeley yvp2b4rtp3 (v1 and v2, July to September 2021) was found. The 2015 extract and the cost and importer fields were never published.

### Cited Findings
- The Data in Brief availability statement points only to Mendeley DOI 10.17632/yvp2b4rtp3 (cites v1; the link goes to v2). — [PMC11424786](https://pmc.ncbi.nlm.nih.gov/articles/PMC11424786/); [Mendeley v2](https://data.mendeley.com/datasets/yvp2b4rtp3/2)
- A web search for GitHub repositories on TPS dwell-time event logs returned no repository. — search results only; [Mendeley dataset](https://data.mendeley.com/datasets/yvp2b4rtp3)
- The IJEECS 2019 paper, with the 2015 data, has no data availability statement or supplement. — [IJEECS PDF](https://ijeecs.iaescore.com/index.php/IJEECS/article/download/18024/13000)

### Inferences
- ACISC (Hindawi) and IEEE Access papers usually carry a data availability statement. Their likely wording is "available on request from the corresponding author", but this was not verified. If so, that statement is a formal hook for a request.

### Gaps
- Mendeley author pages for Prasetyo and Sarno were not checked for other deposits. GitHub was searched only through a general web search, not GitHub's own search.

## Q3. Does Prasetyo's ITS thesis (repository.its.ac.id/114740) describe the full raw schema?

### Takeaway
The thesis is "Optimasi pada Proses Bisnis Berdasarkan Multi-Perspektif Menggunakan Metode Hybrid untuk Memperbaiki Proses Dwelling Time di PT Terminal Petikemas Surabaya" (PhD, ITS, 2024). Its 8 MB PDF is still listed as "restricted to repository staff", embargoed until 1 Oct 2026. As of 30 Oct 2026 the page did not show a public file. The abstract confirms cost, officer and waiting-time perspectives but gives no schema, period or case count.

### Cited Findings
- The full text (7025201008-Dissertation.pdf) is restricted until 1 October 2026, and the page offers a "request a copy" route. — [ITS repo 114740](https://repository.its.ac.id/114740/)
- The abstract describes:
  - Methods: multi-perspective optimisation using process mining, job shop scheduling, multivariate clustering, hybrid simulation (system dynamics, discrete-event and agent-based) and MCDM.
  - Results: asynchronous waiting time down 75.24%, process cost efficiency up 8.47%, TPS staff performance up 3.33%, customs and quarantine staff up 9.52%, PLP staff up 3.84%.
  - Data source: the TPS dwelling-time event log.

  — [ITS repo 114740](https://repository.its.ac.id/114740/)

### Inferences
- The thesis models customs, quarantine and PLP officer performance and process costs, so its data chapter very likely documents officer-level and cost fields that were left out of Mendeley. The repository's "request a copy" button, or simply re-checking after the embargo is lifted in the system, is a legitimate low-cost route to the schema.

### Gaps
- The table of contents and data chapter are not visible. The embargo appears not yet lifted in the repository UI, which may lag behind the date.

## Q4. TPS and Pelindo research collaboration, internship data programmes, ITS-TPS MoU, Pelindo data-access policies

### Takeaway
No published ITS-TPS MoU or formal Pelindo research data-access policy was found. Both TPS extracts were obtained through personal, board-approved arrangements: the TOS DBA and the board in 2015, and an ethics agreement with the data owner in 2021.

### Cited Findings
- The 2015 extract was obtained with TPS IT management, the TOS DBA and Board of Directors involvement. — [IJEECS PDF](https://ijeecs.iaescore.com/index.php/IJEECS/article/download/18024/13000)
- In 2021 the authors negotiated the data format and period with the owner and signed an ethics agreement on data use. — [PMC11424786](https://pmc.ncbi.nlm.nih.gov/articles/PMC11424786/)
- Searches found no ITS-TPS or ITS-Pelindo research MoU. ITS lists other MoUs, such as one with PELNI in 2022. — [ITS DKPU](https://www.its.ac.id/dkpu/?p=4082)
- Student research at TPS often goes through PKL internships or questionnaire and interview studies, not data releases. One example is a Widya Mandala skripsi based on PKL observation. — [UKWMS abstract](https://repositori.ukwms.ac.id/id/eprint/30712/1/ABSTRAK_SKRIPSI.pdf); [UM Surabaya thesis](https://repository.um-surabaya.ac.id/8820/5/Zayd%20Abigail%20Johansyah-20191221120-Bab4..pdf)
- TPS was historically a JV of Pelindo III (51%) and DP World (49%); the DP World partnership ended around 2019. — [Bisnis.com 2019](https://ekonomi.bisnis.com/read/20190430/98/917063/kata-pakar-soal-rencana-pelindo-iii-tak-lagi-bermitra-dengan-asing)

### Inferences
- The realistic route is collaboration with the ITS Sarno group, or with the Telkom or Undip co-authors, under their existing ethics agreement, not a fresh request to Pelindo. A new direct request would probably need an Indonesian academic partner and a PKL or research-permit letter.

### Gaps
- No public Pelindo research-permit procedure was found. Pelindo and TPS websites were not checked directly.

## Q5. Other universities' theses using TPS container-level data with customs or trucking timestamps (precedent)

### Takeaway
There is precedent at UNAIR (Effendi, TPS TOS CSV), Mercu Buana and UNISI (Jokonowo group), Telkom (Prasetyo) and ITS (simulation theses, Terminal Nilam). None of the hits found publish data, and the full texts mostly sit behind repository restrictions.

### Cited Findings
- UNAIR ICCCE 2021: TPS TOS CSV converted to an import event log. — [UNAIR scholar](https://scholar.unair.ac.id/en/publications/transforming-timestamp-of-a-csv-database-to-an-event-log/)
- Telkom University process-mining study of TPS dwell time. — [Telkom PDF](https://repositori.telkomuniversity.ac.id/pustaka/files/241293/abstraksi/analisis-dwelling-time-petikemas-di-terminal-petikemas-menggunakan-process-mining-studi-kasus-terminal-peti-kemas-surabaya-dalam-bentuk-buku-karya-ilmiah.pdf)
- ITS 2014 simulation thesis on TPS import dwell time. — [ITS 98292](https://repository.its.ac.id/98292)
- Mercu Buana and UNISI Alpha Miner dwell-time work. — [Mercu Buana](https://repository.mercubuana.ac.id/89053); [Sistemasi](https://sistemasi.ftik.unisi.ac.id/index.php/stmsi/article/view/1501)

### Gaps
- No PENS, Hang Tuah or Poltekpel Surabaya theses with TPS container-level timestamps were found. These were searched only lightly.

## Q6. Is there an Indonesian NLE or INSW research data-access programme?

### Takeaway
No public academic data-access programme for INSW, LNSW or NLE data was found.

### Cited Findings
- A search for an LNSW or NLE research data request returned only foreign analogues, such as India's NLDS platform open to faculty and students. — [IIMA NLDS event](https://iima.ac.in/events/unlocking-research-potential-with-nlds-exploring-data-sources-and-opportunities-in-logistics)

### Gaps
- The insw.go.id and LNSW (Kemenkeu) sites and PMK regulations were not checked directly. This needs a targeted follow-up.
