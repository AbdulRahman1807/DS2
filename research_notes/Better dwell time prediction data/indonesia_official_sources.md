# Indonesian official statistics and data services that could be merged with the TPS (Tanjung Perak) 2021 per-container import dwell dataset

Scope note: this covers about 17 tool calls. Several primary government sites (insw.go.id, jurnal.pknstan.ac.id) refused connections from the research environment, and bisnis.com and jatim.bps.go.id returned HTTP 403, so some findings rely on search-result abstracts and secondary news. Items marked **[UNVERIFIED]** come from background knowledge and were not confirmed in this session. Check them before relying on them.

## 1. INSW / LNSW dwelling-time statistics: is there a per-port 2021 series split into pre-clearance, customs clearance and post-clearance? Is there an open API or dashboard archive?

### Takeaway
LNSW (Lembaga National Single Window) does compute per-port dwelling time for the 5 main ports, including Tanjung Perak, from terminal-operator data. Monthly and annual per-port figures reach the public mainly through press releases and news reports quoting the "INSW dashboard". I found no open API, downloadable archive or public 2021 Tanjung Perak series split into pre-clearance, customs clearance and post-clearance. In practice, a 2021 monthly series split into those three stages exists at KPPBC TMP Tanjung Perak: a 2024 PKN STAN study used it (see section 2).

### Cited Findings
- Bisnis.com cites the "INSW dashboard" for monthly dwelling time: Tanjung Priok was 3.95 days in May 2022 and 3.11 days in June 2022. This shows monthly per-port values exist on the dashboard — [Bisnis.com 2022-06-29](https://ekonomi.bisnis.com/read/20220629/98/1549389/dwelling-time-naik-lagi-ada-apa-dengan-pelabuhan-utama-tanjung-priok)
- LNSW per-port annual figures for 2024: Tanjung Priok 2.73 days, Makassar 2.26, Belawan 2.97, Tanjung Perak 3.04, Tanjung Emas 3.45 — [DDTCNews](https://news.ddtc.co.id/berita/nasional/1808265/dwelling-time-tercatat-rata-rata-288-hari-pada-2024)
- LNSW per-port monthly figures for March 2025: Tanjung Perak 3.37 days. LNSW collects the data from 10 terminal operators across 5 main ports (Priok, Belawan, Makassar, Tanjung Emas, Tanjung Perak) — [DDTCNews](https://news.ddtc.co.id/berita/nasional/1810353/dwelling-time-tercatat-rata-rata-26-hari-pada-maret-2025)
- The 2025 annual national figure conflicts between sources: 3.02 days ([DDTCNews, Jan 2026](https://news.ddtc.co.id/berita/nasional/1816572/2025-dwelling-time-tercatat-rata-rata-302-hari)) versus 2.9 days quoted from LNSW ([DDTCNews](https://news.ddtc.co.id/berita/nasional/1807414/dwelling-time-285-hari-kepala-lnsw-ini-tergolong-sudah-bagus)). The 2026 target is 2.85 days — [DDTCNews](https://news.ddtc.co.id/berita/nasional/1817834/dwelling-time-ditarget-susut-jadi-285-hari-tahun-ini)
- LNSW has publicly attributed dwelling-time spikes to long holidays (Dec 2025 article; full text returned 403) — [Bisnis.com 2025-12-04](https://ekonomi.bisnis.com/read/20251204/98/1934244/libur-panjang-bikin-dwelling-time-molor-lnsw-ungkap-biang-keroknya)
- The standard three-stage decomposition is pre-clearance, customs clearance and post-clearance. The 2016 target for Priok, Perak and Belawan was 1 day + 0.5 day + 1 day — [PKN STAN study](https://jurnal.pknstan.ac.id/index.php/PBC/article/view/2631); [Kontan](https://nasional.kontan.co.id/news/dwelling-time-di-priok-harus-25-hari-ini-caranya)
- Older Antara framing of the stages: pre-clearance is tied to bank payment, customs clearance to customs and excise procedures, post-clearance to audit — [Antara Jatim](https://jatim.antaranews.com/berita/159467/pelindo-iii-bertekad-tekan-dwelling-time)
- **[UNVERIFIED]** A public dwelling-time dashboard has existed at insw.go.id, historically at a path like `insw.go.id/index.php/home/dwelling_time` or `dwellingtime.insw.go.id`. It showed the latest month and could not be queried for history. The site refused connections during this session, so the current URL and any 2021 archive are unconfirmed.
- I found no public procedure for research data requests to LNSW. The route would be the LNSW/Kemenkeu PPID (Public Information Officer) channel under UU 14/2008 (KIP) — **[UNVERIFIED]** that LNSW runs its own PPID desk ([search](https://repository.mercubuana.ac.id/62066/) returned only secondary work).

### Inferences
- Join key for merging: **date (month, possibly week) × port = Tanjung Perak**. This gives at best a monthly port-level context feature, such as the monthly stage-level dwell average, which can be lagged so it does not leak the target. It cannot fill any per-declaration gap.
- The LNSW measure is probably built from container-level data (terminal gate events joined to customs SPPB timestamps through the INSW portal). So INSW/LNSW internally holds per-container PIB-to-SPPB times. Releasing them would require an MoU or a PPID request, and confidentiality rules (UU Kepabeanan art. 34 **[UNVERIFIED article number]**) make release of importer-identifiable rows unlikely.

### Gaps
- I could not confirm whether a 2021 monthly Tanjung Perak LNSW series is publicly archived anywhere (dashboard snapshots, Kemenkeu newsletters, Wayback Machine). Next step: check web.archive.org for insw.go.id dwelling-time pages from 2021.
- I found no open API.

## 2. DJBC / KPPBC TMP Tanjung Perak: published 2021 statistics on PIB counts by lane, SPPB times, behandle counts and clearance time; research data-sharing precedent

### Takeaway
I found no public 2021 LAKIN for KPPBC TMP Tanjung Perak, although other KPPBC offices publish theirs on beacukai.go.id CDN files. There is a clear **research-sharing precedent**: a PKN STAN study (Perspektif Bea dan Cukai, vol. 8 no. 1, 2024) used KPPBC TMP Tanjung Perak's **monthly pre-clearance, customs clearance, post-clearance and total dwelling-time series for Jan 2020 to Dec 2022**, so the 2021 months are covered. This is the most concrete official stage-level series found for 2021. Lane shares are only available as old press quotes (2015: 91% green, 9% red).

### Cited Findings
- The PKN STAN study "A Study of Dwelling Time at Tanjung Perak Harbor Surabaya Indonesia" uses secondary data from KPPBC TMP Tanjung Perak: monthly pre-clearance, customs clearance, post-clearance and dwelling-time processing times, January 2020 to December 2022 (36 months). It finds pre- and post-clearance significantly and positively affect dwelling time, while customs clearance does not. It appears in Perspektif Bea dan Cukai 8(1), 2024 — [PKN STAN journal](https://jurnal.pknstan.ac.id/index.php/PBC/article/view/2631); [PDF](https://jurnal.pknstan.ac.id/index.php/PBC/article/download/2631/1385/13614) (the site refused direct fetch, so details come from search abstracts)
- In 2015 a KPPBC Tanjung Perak official said 91% of inspections were green lane and 9% red lane — [Bisnis.com 2015](https://ekonomi.bisnis.com/read/20150622/98/445959/operator-tanjung-perak-pastikan-dwelling-time-tak-separah-priok)
- An academic study looked at red-lane physical inspection service quality at KPPBC TMP Tanjung Perak using a survey-based design. It contains no lane statistics — [Unitomo SBR](https://ejournal.unitomo.ac.id/index.php/sbr/article/view/5415/2610)
- A study of the effect of dwelling time on import-tax receipts at KPPBC TMP Tanjung Perak indicates the office has shared revenue and dwell aggregates with external researchers — [UWKS Equilibrium](https://journal.uwks.ac.id/index.php/equilibrium/article/view/2102)
- DJBC hosts office LAKIN (performance reports) for 2021 as CDN files. For example, TMP Tanjung Emas, Ngurah Rai and Balikpapan 2021 reports turned up in search. These usually contain IKU (key performance indicators) such as service time and dwelling-time targets — [beacukai CDN example](https://oneweb-file.beacukai.go.id/cdn/download/file/6800be6cf547e628a791c10a); [KPPBC Bekasi LAKIN listing](https://bekasi.beacukai.go.id/laporan-kinerja-file/57048cda-e604-4e43-92dc-ed47d146a406)
- An older PKN STAN-affiliated article analysed green and yellow lane dwelling time at Tanjung Priok (2015 data). This is a precedent for lane-level analysis with DJBC data, but at Priok — [PKN STAN PBC article 125](https://jurnal.pknstan.ac.id/index.php/PBC/article/download/125/142/837)
- Definition: the red lane means physical inspection plus document research before SPPB is issued — [Unitomo](https://ejournal.unitomo.ac.id/index.php/sbr/article/view/5415/2610)
- **[UNVERIFIED]** KPPBC TMP Tanjung Perak's site is likely `bcperak.beacukai.go.id` or `tanjungperak.beacukai.go.id`. The office pattern `<office>.beacukai.go.id/laporan-kinerja-file/...` (as in Bekasi) suggests the Tanjung Perak 2021 LAKIN may sit under a similar path. Not checked.
- The 2024 PKN STAN Tanjung Perak study is the only DJBC data-sharing precedent found for this port in this window — [PKN STAN](https://jurnal.pknstan.ac.id/index.php/PBC/article/view/2631).

### Inferences
- **Monthly stage-level series (2020 to 2022)**: obtainable by contacting the PKN STAN authors or KPPBC TMP Tanjung Perak directly. Join key: month. It adds a monthly "customs-stage regime" feature. It is not per-declaration.
- **Per-declaration PIB filing time and lane for TPS 2021**: CEISA (the DJBC system) holds PIB submission timestamps, lane (H/K/M/MITA), SPJM/SPJK and SPPB timestamps keyed by PIB number/date and BL, with container numbers in the PIB container section. The realistic routes for a university team are:
  1. **TPS itself**: the terminal receives SPPB/BC 2.0 and, for red lane, SPJM/behandle messages through the CEISA–terminal host-to-host link. TPS's own TOS may already hold the PIB number and date, SPPB time and lane-derived behandle flags per container. This is likely the cheapest source because it is the same data owner as the current event log. Ask TPS for the customs-document table (PIB no, PIB date, SPPB no/time, SPJM flag) behind JOB_DEL. **[Inference, not verified]**
  2. **KPPBC TMP Tanjung Perak or the DJBC Directorate of Information Technology (Dit. IKC)** through a formal research permit (surat izin penelitian from the university to the Kepala Kantor). Per-declaration data would need anonymisation: importer NPWP removed, HS kept at 2 to 4 digits. A PKN STAN co-author is the most plausible facilitator, since PKN STAN is the DJBC training school.
  3. **LNSW** through an MoU or PPID request for its dwell-time computation table.
- Customs confidentiality makes importer identity and HS at the 8-digit level the hardest fields to get. Lane, timestamps and HS2 are more plausibly releasable.

### Gaps
- I did not locate the KPPBC TMP Tanjung Perak 2021 LAKIN or its IKU values (search found only other offices).
- I found no published 2021 lane split (H/K/M/MITA) or behandle counts for Tanjung Perak.
- I could not read the full PKN STAN PDF (connection refused), so the 2021 monthly values and the exact data-request procedure are unverified.

## 3. BPS: monthly imports by port of entry and HS chapter for Tanjung Perak in 2021; resolution; is port × HS × month available?

### Takeaway
BPS publishes monthly import bulletins with breakdowns by HS 2-digit, SITC, country, province and port of unloading. **Port × month** is available in publications. **Port × HS2 × month** as a single cross-tab is not confirmed open. BPS Jatim publishes an annual "Import Statistics of Jawa Timur Province 2021" compiled from PIB documents. These are aggregate features only (join key: month × port, optionally HS2 if the TPS dataset had HS, which it does not).

### Cited Findings
- BPS monthly import bulletins ("Statistik Perdagangan Luar Negeri Bulanan Impor") cover imports by HS 2-digit, SITC, country of origin, province and port of import — [BPS Dec 2024 bulletin](https://www.bps.go.id/id/publication/2025/02/28/51230ea405830664d784ba0b/foreign-trade-statistical-bulletin-imports--december-2024.html); [BPS 2011 bulletin description](https://www.bps.go.id/en/publication/2011/12/15/f6ce3b8cb58d5e2ceb5816aa)
- BPS Jatim published "Import Statistics of Jawa Timur Province 2021" (released 2022-08-02), compiled from PIB (Goods Import Notification) documents — [BPS Jatim](https://jatim.bps.go.id/en/publication/2022/08/02/0707516eef81ffd2fc241ce6/import-statistics-of-jawa-timur-province-2021.html) (page returned 403, so table-of-contents detail is unverified)
- Third-party aggregators show BPS monthly imports by port × HS chapter for specific chapters (for example HS 49 by "provinsi asal pelabuhan"). This suggests BPS dynamic tables have HS × port-province × month — [Databoks](https://databoks.katadata.co.id/en/demographics/multiple/d24800eac45d5cd/nilai-impor-produk-industri-percetakan-menurut-provinsi-asal-pelabuhan-kode-hs-49-bulanan)
- Regional open-data portals redistribute BPS "imports by port of unloading" CSVs at annual level — [Aceh open data](https://data.acehprov.go.id/dataset/ac976519-8665-49cc-b704-bde373a2e5a2/resource/784a3647-6b4c-4ee0-97a0-df8905a73491)
- **[UNVERIFIED]** The BPS WebAPI (`https://webapi.bps.go.id/v1/api/...`, free key from webapi.bps.go.id/developer) exposes dynamic tables ("var") and a dedicated export-import endpoint (`/v1/api/dataexim/`) with parameters for import/export, HS digit level (2-digit or full) and year. The BPS "Exim" portal (bps.go.id/exim) allows HS × country × month queries. I could not confirm that port of unloading is a dimension in the exim API.

### Inferences
- The most likely open feature is **monthly Tanjung Perak (or Jawa Timur) import value and net weight**, total or by HS2. As a congestion or volume proxy it partly duplicates TPS's own throughput. Low expected value for per-container prediction.
- Port × HS × month for Tanjung Perak probably exists in BPS microdata. It can be requested through BPS's paid/request data service (PST, Pelayanan Statistik Terpadu, silastik.bps.go.id **[UNVERIFIED]**).

### Gaps
- I did not confirm whether the BPS API exim endpoint has a port dimension.
- I did not confirm the exact 2021 Tanjung Perak monthly values.

## 4. Barantin (quarantine) statistics for Tanjung Perak 2021; IQFAST

### Takeaway
I found no open 2021 quarantine statistics for Surabaya/Tanjung Perak. IQFAST is Barantan's (now Barantin's) internal real-time traffic and certification system. Per-certificate data, which would be keyed by container or BL and carry KT-9/KH-12 release times, would only come on request. Tanjung Perak is cited as the port with the largest detained-import volume (2022).

### Cited Findings
- IQFAST allows real-time monitoring of agricultural commodity traffic at all entry and exit points and is used by quarantine units nationwide — [Kontan](https://industri.kontan.co.id/news/digitalisasi-operasional-badan-karantina-kembangkan-sistem-iqfast); [Bisnis 2018](https://ekonomi.bisnis.com/read/20181120/99/861408/badan-karantina-pertanian-terapkan-layanan-digital)
- Barantan data put the largest volume of detained imports at Tanjung Perak: 969,000 kg in 57 containers (2022 report) — [Jakarta Post 2022-10-03](https://newgelora.thejakartapost.com/business/2022/10/03/govt-puts-end-to-detained-imports-drama-as-kpk-calls-for-updated-database)
- Barantan has stated it wants to cut its share of dwelling time — [Kontan](https://industri.kontan.co.id/news/badan-karantina-pangkas-waktu-dwelling-time)
- A comparable 2021 study of agricultural import activity used BBKP Makassar working-area data. It sets a precedent for quarantine-office data sharing with academics, at a different port — [Filogeni UIN Alauddin](https://journal.uin-alauddin.ac.id/index.php/filogeni/article/view/29537)
- **[UNVERIFIED]** In 2021 the Surabaya units were BBKP Surabaya (agriculture, Kementan) and BKIPM Surabaya I (fish, KKP). Both merged into Barantin in 2023 (karantinaindonesia.go.id). Their 2021 LAKIN reports may contain monthly import-frequency counts.

### Inferences
- The TPS event log already has quarantine flags. A Barantin per-certificate extract (certificate issue time, commodity) would add the timing of quarantine release, but the access route is a formal request to Barantin or the former BBKP Surabaya. Aggregate monthly counts add little.

### Gaps
- No 2021 Surabaya quarantine statistics or LAKIN located.
- No IQFAST public API found.

## 5. Pelindo / TPS: 2021 annual reports (dwell, YOR, monthly throughput), research cooperation, e-service / online booking, NLE dashboards (DO online, SP2 online)

### Takeaway
I did not find official 2021 TPS dwell or YOR monthly series in open sources. Pelindo III gave half-year 2021 throughput to the press. Pelindo merged into a single Pelindo in October 2021, so the 2021 annual report is Pelindo-level. Academic work shows TPS shares operational data with researchers: YOR studies and a Telkom University process-mining study on the TPS dwell event log. DO Online has been mandatory at Tanjung Perak since 2017/2018. NLE offers SP2 online, but I found no public NLE dashboard with Tanjung Perak data.

### Cited Findings
- Pelindo III H1 2021 throughput was 2.61 million TEU (versus 2.47 million in H1 2020). International volume was stagnant and domestic volume drove growth. This is company-level, not TPS-level — [Bisnis.com 2021-07-21](https://ekonomi.bisnis.com/read/20210721/98/1420277/pelindo-iii-arus-peti-kemas-internasional-masih-stagnan)
- An academic YOR evaluation of TPS (Mitrans, UNESA) uses YOR (%) as the performance parameter — [UNESA Mitrans](https://journal.unesa.ac.id/index.php/mitrans/article/view/38182)
- A TPS yard/dwell study notes that container flow at TPS rose from 2016 to 2019 and fell in 2020, and links YOR above roughly 80% to longer dwell — [Akamigas JTLM](https://jtlm.akamigas.ac.id/index.php/jtlmig/article/download/20/12/30)
- A Telkom University study, "Analisis dwelling time petikemas ... menggunakan process mining, studi kasus Terminal Peti Kemas Surabaya" (PDF dated 2025-07), used TPS's terminal information-system log data. It found mismatches between port SOPs and the data log, plus bottlenecks in document checks and container moves. This is a direct precedent for TPS releasing event logs to researchers — [Telkom University repository abstract](https://repositori.telkomuniversity.ac.id/pustaka/files/241293/abstraksi/analisis-dwelling-time-petikemas-di-terminal-petikemas-menggunakan-process-mining-studi-kasus-terminal-peti-kemas-surabaya-dalam-bentuk-buku-karya-ilmiah.pdf)
- DO Online was mandated by PM 120/2017 for imports at 4 ports including Tanjung Perak, and carried forward in PM 42/2020 — [PM 120/2017](https://peraturan.bpk.go.id/Download/93833/PM_120_TAHUN_2017.pdf); [PM 42/2020](https://peraturan.bpk.go.id/Download/141502/PM_42_TAHUN_2020.pdf)
- Logistics associations reported in 2018 that DO Online was not yet effective — [Bisnis 2018](https://ekonomi.bisnis.com/read/20180713/98/816256/implementasi-do-online-dinilai-belum-efektif); [ALFI](https://ekonomi.bisnis.com/read/20180712/98/815880/alfi-persoalkan-implementasi-delivery-order-online-di-4-pelabuhan)
- The NLE Indonesia app offers online SP2 for moving goods into and out of TPS (temporary storage, line 1 and line 2) without paper — [App Store listing](https://apps.apple.com/us/app/id1604699920)
- Pelindo III announced "gate-in online" at Tanjung Perak — [Kontan](https://industri.kontan.co.id/news/pelindo-iii-segera-terapkan-gate-in-online-di-pelabuhan-tanjung-perak)
- In 2016 Pelindo III stated Tanjung Perak dwell was 6 days (historical context) — [Detik](https://finance.detik.com/berita-ekonomi-bisnis/d-3297224/i-dwell-time-i-di-tanjung-perak-6-hari-ini-kata-pelindo-iii)

### Inferences
- **The single most valuable action is to request extra tables from TPS itself.** The TOS and billing systems should already hold, per container: SP2 / proforma issue and payment time (billing), DO-online receipt, PIB/SPPB document numbers and times received through the customs host-to-host link, behandle/SPJM order times, and truck booking or gate-pass times. These are exactly the missing timestamps. The Telkom University precedent shows TPS releases log data for research. **[Inference]**
- TPS/Pelindo annual reports give annual, not monthly, figures. A monthly YOR series is better derived from the existing event log (yard inventory reconstruction) than from publications.
- NLE (nle.kemenkeu.go.id) aggregates DO and SP2 transactions, but I found no public Tanjung Perak data. Access would be through an LNSW request.

### Gaps
- I did not find the Pelindo 2021 annual report URL or the TPS-specific 2021 dwell/YOR numbers (Pelindo investor relations pages were not reached).
- I found no public NLE dashboard.
- I did not confirm TPS's formal research cooperation program or its e-service (e.g. "TPS Online" / i-Booking) data-access policy.

## 6. Cheap exogenous series: Bank Indonesia JISDOR and BMKG Perak weather station

### Takeaway
Both are open and joinable by date. Neither was fetched in this session; the details below are from background knowledge and need checking.

### Cited Findings
- **[UNVERIFIED]** Bank Indonesia JISDOR (USD/IDR reference rate) is published daily on business days and downloadable as Excel from `https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx`. Join key: date (forward-fill weekends).
- **[UNVERIFIED]** BMKG publishes daily station observations (rain, wind, temperature) through `https://dataonline.bmkg.go.id` after free registration. Relevant stations are Stasiun Meteorologi Maritim Tanjung Perak (WMO ID 96937 **[unverified]**) and Juanda (96935). Data are daily, with some synoptic hourly data on request. Join key: date (or hour).

### Inferences
- Low effort, low expected lift. JISDOR may proxy for duty or payment-timing behaviour. Rain or wind may affect gate and trucking operations. Both are useful mainly as controls.

### Gaps
- Neither site was fetched, so URL paths, station IDs and registration requirements need checking.

## 7. Summary: what is per-container or per-declaration versus aggregate, and the most realistic route to per-declaration PIB filing time and lane

### Takeaway
Every open source found is **aggregate** (month × port at best). The per-declaration fields (PIB filing time, lane, SPJM/SPJK, importer/HS, SP2/DO, truck appointments) exist only in closed systems: CEISA (DJBC), INSW/NLE (LNSW), the TPS TOS/billing system, and IQFAST (Barantin). The most realistic route for a university team is: (1) ask TPS for the customs-document and billing tables tied to the containers it already shared; (2) in parallel, apply for a research permit from KPPBC TMP Tanjung Perak for an anonymised PIB extract (PIB no/date/time, lane, SPJM/SPPB times, HS2), ideally through a PKN STAN co-author; (3) ask LNSW through PPID for monthly stage-split dwell for Tanjung Perak in 2021.

### Cited Findings
| Source | Open or on request | Resolution | Join key | 2021 Perak coverage | URL |
|---|---|---|---|---|---|
| LNSW/INSW dwelling-time dashboard | Open (current month only, as reported); history on request | Monthly × port (total; stage split internal) | month | Yes (Perak is one of the 5 monitored ports) | [DDTCNews](https://news.ddtc.co.id/berita/nasional/1810353/dwelling-time-tercatat-rata-rata-26-hari-pada-maret-2025), insw.go.id **[unreachable]** |
| KPPBC TMP Tanjung Perak stage series | On request (research precedent) | Monthly; pre/customs/post-clearance and total | month | Yes (Jan 2020–Dec 2022) | [PKN STAN 2024](https://jurnal.pknstan.ac.id/index.php/PBC/article/view/2631) |
| DJBC CEISA PIB records | On request / MoU only | Per declaration (per container via PIB container section) | PIB no, BL, container no | Yes (system of record) | Inference |
| KPPBC LAKIN 2021 | Open (where published) | Annual IKU | year | Not found for Perak | [beacukai CDN example](https://oneweb-file.beacukai.go.id/cdn/download/file/6800be6cf547e628a791c10a) |
| BPS monthly import bulletins / BPS Jatim 2021 | Open | Monthly × port; HS2 × month; annual Jatim | month (+HS2) | Yes | [BPS](https://www.bps.go.id/id/publication/2025/02/28/51230ea405830664d784ba0b/foreign-trade-statistical-bulletin-imports--december-2024.html), [BPS Jatim](https://jatim.bps.go.id/en/publication/2022/08/02/0707516eef81ffd2fc241ce6/import-statistics-of-jawa-timur-province-2021.html) |
| Barantin IQFAST | On request | Per certificate (internal) | BL / container (likely) | Yes (internal) | [Kontan](https://industri.kontan.co.id/news/digitalisasi-operasional-badan-karantina-kembangkan-sistem-iqfast) |
| TPS TOS / billing (SP2, DO, booking) | On request (precedent: Telkom Univ. process mining) | Per container, timestamped | container no + visit | Yes | [Telkom Univ.](https://repositori.telkomuniversity.ac.id/pustaka/files/241293/abstraksi/analisis-dwelling-time-petikemas-di-terminal-petikemas-menggunakan-process-mining-studi-kasus-terminal-peti-kemas-surabaya-dalam-bentuk-buku-karya-ilmiah.pdf) |
| NLE (DO/SP2 online) | Closed / on request | Per transaction | BL / DO no | Tanjung Perak in DO Online scope since PM 120/2017 | [PM 120/2017](https://peraturan.bpk.go.id/Download/93833/PM_120_TAHUN_2017.pdf) |
| BI JISDOR | Open | Daily | date | Yes | **[UNVERIFIED URL]** bi.go.id |
| BMKG station data | Open (registration) | Daily / hourly | date | Yes | **[UNVERIFIED URL]** dataonline.bmkg.go.id |

### Inferences
- Open aggregates can only add monthly or daily regime features. They cannot fix the per-container information gap that limits classification. Any meaningful improvement depends on the closed per-declaration data, and TPS is the lowest-friction holder of most of it.

### Gaps
- None of the on-request routes has a published formal procedure in the sources found. Turnaround and the exact field lists are unknown.
