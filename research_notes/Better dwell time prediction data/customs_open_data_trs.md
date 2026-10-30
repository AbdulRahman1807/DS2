# Customs open data and Time Release Study (TRS) data for import-container dwell prediction

Scope: national customs open data and TRS data with declaration-level or container-level clearance timestamps. Port/terminal event logs and Kaggle/Mendeley datasets are out of scope (covered elsewhere).
Verification legend: **[V]** = verified this session from a fetched or searched primary/official page; **[S]** = only from search-result snippets or secondary sources; **[U]** = unverified (from prior knowledge, not confirmed this session; check before relying on it).
Budget note: about 17 tool calls were used, so several portals (datos.gob.cl field dictionaries, Mexico, Argentina, Ecuador, Costa Rica, Philippines, Thailand, Vietnam, Kenya, South Africa) were not checked directly. They are listed under Gaps.

## Q1. Peru SUNAT: what is published, at what granularity?

### Takeaway
I found no open SUNAT microdata file that has the control channel (verde/naranja/rojo) and the levante date-time together. SUNAT does track both per DAM, but publicly it offers only (a) a one-at-a-time lookup per declaration and (b) the "Tiempo de Importación" microsite, which gives **average** import times in hours by operator. Declaration-level data with channel and release times has been used in research only under data-sharing agreements.

### Cited Findings
- [V] SUNAT's "Tiempo de Importación" microsite (https://tiempodeimportacion.sunat.gob.pe/) shows **average import times in hours** by operator (importer, customs broker, customs warehouse, port/airport) and by AEO. It identifies the operators with the shortest times per customs office or per clearance mode (anticipado vs diferido). This is aggregate, not microdata. — [SUNAT comunicado 14-01-2025](https://www.sunat.gob.pe/operatividadaduanera/novedades/aduanas/comunicados/2025/enero/co140125.pdf)
- [S] SUNAT's online "Consulta de Declaración" shows the control channel, levante, and regularization status for **one DAM at a time** (an interactive lookup, not a bulk dataset). — [SUNAT Consulta de Declaración](https://www.sunat.gob.pe/cl-ad-itorientacionconsmovil/entrada.htm)
- [S] D.S. 226-2009-EF sets out which customs declaration fields SUNAT may publish as an exception to tax confidentiality (e.g., port and date of shipment). — [MEF D.S. 226-2009-EF](https://www.mef.gob.pe/es/por-instrumento/decreto-supremo/4853-d-s-n-226-2009-ef/file)
- [S] A SUNAT legal report (2021) looks at whether names and ID numbers of individual importers/consignees can be published, which signals privacy limits on any microdata release. — [SUNAT Informe 040-2021](https://www.sunat.gob.pe/legislacion/oficiosAd/2021/2021-INF-040-340000.pdf)
- [S] Levante is granted per declaration. For partial withdrawals, the recorded date is the actual withdrawal of the last package. — [SUNAT Informe 028-2014](https://www.sunat.gob.pe/legislacion/oficiosAd/2014/informes/2014-INF-028-5D1000.pdf)
- [U] SUNAT's statistics area (sunat.gob.pe "Estadísticas y Estudios" / "Operatividad Aduanera") has historically published downloadable declaration-line import files (DAM number, date of numbering, customs office, HS 10-digit, FOB/CIF, country, possibly an encrypted importer RUC). Commercial resellers (e.g., Veritrade) are built on these. I could not confirm whether the files carry the channel or the levante **time**. My prior belief is that they carry dates only and no channel.

### Inferences
- Peru is the strongest Latin American candidate in principle, because SUNAT computes arrival→levante→exit times internally (the microsite reports hours). Getting declaration-level data would still likely need a formal request or a research agreement with SUNAT's Intendencia Nacional de Desarrollo e Innovación Aduanera, which owns the microsite.

### Gaps
- I could not open datosabiertos.gob.pe to see whether any SUNAT customs dataset exists there, or what its fields are.
- I did not inspect the microsite's internals: whether it exposes API/JSON per-operator series, and which stage boundaries it uses (arrival vs numbering vs levante vs terminal exit).

## Q2. Other Latin American customs (Chile, Colombia, Mexico, Brazil, Argentina, Ecuador, Uruguay, Costa Rica)

### Takeaway
Several countries publish **declaration-level trade microdata** (Colombia DIAN, Chile Aduanas DIN files, Peru), but none I could verify include the **inspection channel plus release timestamp plus physical exit** needed for dwell prediction. Brazil's Receita Federal holds day-hour-minute registration and clearance stamps per DI and channel, but publishes them only as aggregate studies and TRS reports. Uruguay's transaction-level clearance times exist but were accessed under research agreements.

### Cited Findings
**Colombia (DIAN)**
- [V] DIAN publishes declaration-level import files (2020 to Sept 2026) and export files (2019 to Sept 2026) as zipped flat text files. They are grouped weekly or every few days for recent periods and monthly for older ones. The page highlights casilla 91 (the free-text goods description, split into 250-character segments). It says the data is "no oficial", not statistically validated, and published under Ley 1712/2014 and Ley 1581/2012. The page states no open licence and no field list mentioning levante, channel or times. — [DIAN Registro de las Declaraciones de Importación/Exportación](https://www.dian.gov.co/dian/cifras/Paginas/Registro-de-las-Declaraciones-de-Impo-Expo.aspx)
- [U] DIAN/DANE import microdata (DANE "Importaciones" anonymized microdata) include declaration date, customs office, HS code and values. I believe they include no levante time or inspection channel.

**Chile (Servicio Nacional de Aduanas)**
- [S] Aduanas publishes downloadable DIN (Declaración de Ingreso) and export declaration data "as registered in its systems at the date of publication" via its Transparency/Datos Abiertos subsite and datos.gob.cl. — [Aduana: datos estadísticos mensuales](https://www.aduana.cl/transparencia/2016/datosestadmens.html); [Aduana: Información por Ley de Transparencia y Datos Abiertos](https://www.aduana.cl/informacion-por-ley-de-transparencia-y-datos-abiertos/aduana/2014-04-24/155536.html); [Aduanas publica información de importaciones y exportaciones](https://www.aduana.cl/aduanas-publica-informacion-de-importaciones-y-exportaciones-en-su-web/aduana/2016-09-30/162849.html)
- [S] A third-party scraper describes a "Registro de Importaciones DIN 2026" dataset on datos.gob.cl with fields such as goods, brand, quantities, country of origin and FOB/CIF, distributed as ZIP/RAR. This is not an official source. — [Apify scraper description](https://apify.com/scraperschile/aduanas-chile.md)
- [U] The Chilean DIN files are believed to be line-level with ~100+ columns. These include acceptance date (FECACEP, **date-only**), customs office, port of embarkation/landing, transport, HS 8-digit, an encrypted importer ID, and values. I know of no inspection-channel ("aforo") or levante time field. Verify against the datos.gob.cl data dictionary.

**Brazil (Receita Federal / Siscomex)**
- [V] Siscomex itself generates the **day, hour and minute** of DI registration and of desembaraço (clearance). RFB's clearance-time statistics are computed directly from these stamps. The official clearance time is measured from **registration to desembaraço** only. It excludes pre-registration licensing and post-clearance state ICMS steps, and terminal operator records are not synchronized online with Siscomex. — [RFB, "Variabilidade dos tempos de despacho aduaneiro de importação – Alfândega do Porto de Santos 2016"](https://www.gov.br/receitafederal/pt-br/acesso-a-informacao/dados-abertos/resultados/estatisticas_comercio_exterior/arquivos-e-imagens/Variabilidadedostemposdedespachoaduaneirodeimportao2.pdf)
- [V] Same study: the Port of Santos customs office processed 382,803 DIs in 2016 (21.5% of Brazil's DIs) across 43 customs premises ("recintos"). It analyzes times by DI type, parametrization channel (verde/amarelo/vermelho/cinza) and premises type. It is an aggregate analytic report; I saw no microdata release. — same source
- [S] RFB published a WCO-methodology TRS (June 2020) that maps import flows into stages; the simplest 4-stage flow covers 87.25% of operations. — [RFB TRS Executive Report](https://www.gov.br/receitafederal/pt-br/acesso-a-informacao/dados-abertos/resultados/estatisticas_comercio_exterior/estudos-e-analises/TRSExecutiveReport.pdf); [Relatório Executivo (PT)](https://www.gov.br/receitafederal/pt-br/acesso-a-informacao/dados-abertos/resultados/estatisticas_comercio_exterior/estudos-e-analises/TRSRelatrioExecutivo.pdf)
- [S] Balanço Aduaneiro gives an average of 11.47 h from registration to clearance, with 93.82% of DIs cleared in under 24 h (2024 edition per snippet). The 2019 edition gives a gross average of 19.53 h. Channel targets: green <24 h, yellow <72 h, red <120 h. Monthly times by channel sit in the internal "Gerencial de Desempenho Aduaneiro"/DW Aduaneiro, which is not public. — [RFB Balanço Aduaneiro 2019](https://www.gov.br/receitafederal/pt-br/centrais-de-conteudo/publicacoes/relatorios/aduana/2019-balanco-aduaneiro.pdf); [ENAP "Gerencial do Despacho Aduaneiro"](https://repositorio.enap.gov.br/handle/1/4608?mode=full)
- [U] Comex Stat (comexstat.mdic.gov.br) is aggregate monthly trade statistics (NCM × country × state/URF × month). It has no declaration-level times.

**Uruguay (DNA)**
- [S] Volpe Martincus, Carballo & Graziano ("Customs", *JIE* 2015; IDB WP-705) use the **universe of Uruguayan export transactions 2002–2011**, with the actual customs processing time per shipment and the risk-based verification channel. Export processing times ranged 1–31 days in 2011. The data came from the Uruguayan customs agency for research. This is **exports**, not imports, and not public. — [IDB "Customs" PDF](https://publications.iadb.org/publications/english/document/Customs.pdf); [IDB page](https://publications.iadb.org/en/customs)
- [U] Related IDB work on Peru (Volpe Martincus et al., e.g. "Out of the border, out of the way" / customs and exports using SUNAT transaction data with control channels) relied on SUNAT data-sharing. It was not public.

**Mexico, Argentina, Ecuador, Costa Rica**
- [U] Not verified this session. Prior belief: Mexico's SAT/ANAM publishes aggregate statistics only (pedimento-level data is commercial, via resellers, without the semáforo fiscal outcome). Argentina's AFIP/ARCA publishes no declaration-level clearance times. Ecuador's SENAE has historically published declaration-level import data (DAI) through its "Sistema de consulta"/open-data portal, possibly including aforo type. **Verify.** Costa Rica: no knowledge of open TICA microdata with times.

### Inferences
- Across Latin America the published microdata is trade-value data (date, HS, value, origin, sometimes encrypted importer). Channel and release timestamps are held internally and appear only in aggregate reports, TRS reports, or research collaborations.
- Brazil and Peru are the most promising for a **formal data request**. Both systems record hour-precision registration, channel and release per declaration. Neither public output includes physical gate-out, though. Brazil explicitly notes that terminal records are not synchronized online.

### Gaps
- Field dictionaries for the Chile datos.gob.cl DIN files and the Ecuador SENAE open data were not retrieved.
- I could not confirm whether any Brazilian dados.gov.br dataset exposes per-DI channel and times. I found none.

## Q3. India, US, EU, Indonesia, and other Asian/African administrations

### Takeaway
None publish declaration-level clearance timestamps openly. India's NTRS and Indonesia's INSW dwelling-time dashboard are **aggregate** (port × stage averages). US commercial bill-of-lading data (Panjiva, ImportGenius) gives vessel **arrival date** but no CBP entry/release timestamps. Indonesia's per-stage dwell data for Tanjung Perak exists at **monthly aggregate** level and was obtained by researchers from the customs office.

### Cited Findings
**India (CBIC NTRS)**
- [S] NTRS 2024 (the 4th annual study) covered bills of entry and shipping bills filed in the first week of January 2024, tracked to 7 Feb 2024, across 15 ports. It reports average release times by port category: e.g., export seaports 22:49 h, ICDs 30:20 h, ACCs 3:50 h, ICPs 5:28 h. Reports are aggregate. The 2025 (5th) edition adds stage-wise, process-specific breakdowns for seaports. — [Maritime Gateway](https://www.maritimegateway.com/cbic-releases-national-time-release-study-ntrs-2024/); [TaxGuru](https://taxguru.in/corporate-law/cbic-releases-national-time-release-study-ntrs-2024.html); [India Seatrade News, June 2026](https://indiaseatradenews.com/efficiency-gains-across-indian-ports-cbic-study-reveals-faster-cargo-clearance-times/)
- [U] The NTRS stages are arrival (IGM/inward entry) → BE filing → assessment/duty payment → out-of-charge → gate-out at seaports. They are computed from ICEGATE plus custodian data, so the underlying microdata has the exact fields needed, but no per-consignment release is public.

**United States (CBP)**
- [S] Panjiva US data comes from CBP vessel manifest (AMS) records, updated weekly. Fields include date/arrival date, shipper, consignee, vessel, voyage, B/L and master B/L, number of containers and weight, and sometimes HTS codes. Full field definitions come from Panjiva's schema endpoint or a representative. — [Panjiva shipments API schema](https://cn.panjiva.com/api-guide/data-schemas/shipments)
- [U] Manifest data is public under 19 CFR 103.31 (press/vessel manifest release). ACE entry data (entry date, release date, exam holds) is **not** public, and FOIA requests for entry-level data are typically denied as confidential business information. So US commercial data supports "arrival" only, with no customs release or gate-out.
- [S] For context, BTS publishes only aggregate monthly **vessel** dwell (not container) at the top 25 US ports. — [data.gov BTS vessel dwell](https://catalog.data.gov/dataset/monthly-average-container-vessel-dwell-times-at-the-top-25-u-s-container-ports-januar-2023)

**EU**
- [U] Eurostat Comext and TARIC are aggregate trade statistics and tariff reference data. They hold no declaration times. Not researched further.

**Indonesia (DJBC / INSW / LNSW)**
- [S] The INSW dashboard reports dwell time by port, split into **pre-clearance, customs clearance and post-clearance**: e.g., Tanjung Priok 3.95 days (May 2022) and 3.11 days (Jun 2022). One report gives Priok stage times of pre 2.3 d, customs 0.47 d, post 1.45 d. These are dashboard aggregates and I found no download. — [Bisnis.com, 1 Jul 2022](https://ekonomi.bisnis.com/read/20220701/98/1550282/dwelling-time-di-pelabuhan-tanjung-priok-naik-ini-biang-keroknya); [Antara](https://en.antaranews.com/news/101965/tanjung-priok-port-improves-dwelling-time)
- [S] LNSW/Ministry of Finance figures: national average dwell 3.02 days in 2025 (3.52 in 2024), customs clearance 0.42 days in 2025. The 2026 target is 2.85 days. Tanjung Perak is among the measured ports. — [DDTC News](https://news.ddtc.co.id/berita/nasional/1816572/2025-dwelling-time-tercatat-rata-rata-302-hari); [DDTC News target 2.85](https://news.ddtc.co.id/berita/nasional/1817834/dwelling-time-ditarget-susut-jadi-285-hari-tahun-ini)
- [S] A PKN STAN study of Tanjung Perak used **monthly** pre-clearance, customs-clearance, post-clearance and total dwell from KPPBC TMP Tanjung Perak, Jan 2020 – Dec 2022. Pre- and post-clearance significantly affected dwell time, and customs clearance did not. — [PKN STAN, "A Study of Dwelling Time at Tanjung Perak Harbor Surabaya"](https://jurnal.pknstan.ac.id/index.php/PBC/article/view/2631)
- [S] A UWKS study used Tanjung Perak customs-office dwell data for 2016–2020 (regression on import tax). — [UWKS Equilibrium](https://journal.uwks.ac.id/index.php/equilibrium/article/view/2102)
- [U] Per-container stage stamps (SPPB release, gate-out) exist inside INSW/CEISA and the TPS/Pelindo systems. Per-container "pre/customs/post" decomposition is what the INSW dwell dashboard is computed from. I found no public microdata. An authorized request to LNSW/DJBC (KPPBC TMP Tanjung Perak) is the realistic route, and it would directly complement the existing TPS 2021 data (same port, same year possible).

**Philippines, Thailand, Vietnam, Kenya KenTrade, South Africa SARS**
- [U] Not researched this session (call budget). Prior belief: all publish TRS reports or aggregate trade statistics only, and none publishes declaration-level release timestamps.

### Inferences
- For our Surabaya problem, the highest-value source is not a foreign dataset but **Indonesian CEISA/INSW declaration data for Tanjung Perak**: PIB filing time, lane (jalur hijau/kuning/merah/MITA), SPPB time, HS and importer type. Researchers have obtained monthly aggregates from KPPBC Tanjung Perak, which suggests a data request is feasible.

### Gaps
- Whether LNSW exposes a downloadable INSW dwelling-time dataset (even daily aggregates) is unconfirmed.
- I did not retrieve the NTRS 2025 stage-wise tables.

## Q4. WCO / World Bank TRS: are microdata ever published? Indonesia coverage? Stages?

### Takeaway
TRS reports are almost always published as **aggregate** findings (mean/median time per stage, per channel, per port). The WCO methodology collects per-consignment timing data, but I found no case where the consignment-level sample was released publicly. I found no formal WCO-style TRS covering Tanjung Perak. Indonesia's equivalent is the INSW pre/customs/post dwell decomposition.

### Cited Findings
- [S] WCO TRS Guide: three phases (preparation/planning, data collection/recording, analysis/conclusions). It aims to capture all transactions in the sample period and supports national, bilateral or multilateral studies. — [WCO TRS page](https://www.wcoomd.org/es-es/topics/facilitation/instrument-and-tools/tools/time-release-study.aspx); [WCO TRS Methodology Overview (TFA Facility, 2025)](https://tfafacility.org/sites/default/files/2025-06/WCO%20TRS%20Methodology%20Overview_Final%20%28ID%20193554%29.pdf); [CAREC WCO TRS workshop deck](https://www.carecprogram.org/uploads/03-WCO-TRS-Overview-Purpose-and-Methodology.pdf)
- [S] Published national TRS examples, all with aggregate results: Brazil 2020 (4-stage flow = 87.25% of operations), Sri Lanka 2014 (average release times by channel), Uganda 2008, Bhutan 2024. — [Brazil TRS](https://www.gov.br/receitafederal/pt-br/acesso-a-informacao/dados-abertos/resultados/estatisticas_comercio_exterior/estudos-e-analises/TRSExecutiveReport.pdf); [Sri Lanka TRS](https://ereg.customs.gov.lk/public/files/trs/TimeReleaseStudy1.pdf); [Bhutan TRS 2024](https://www.drc.gov.bt/wp-content/uploads/2025/05/TRS-report-for-publishing.pdf)
- [U] Typical TRS stages for sea imports: vessel arrival → manifest submission → container discharge → declaration lodgement → risk lane assignment → documentary/physical inspection → duty payment → customs release → other-agency permits → gate-out/exit from port. Reports give mean/median hours per stage and per lane.
- [S] No Tanjung Perak TRS was found. Dwell studies for Perak use the INSW pre/customs/post scheme from customs office data. — [PKN STAN](https://jurnal.pknstan.ac.id/index.php/PBC/article/view/2631)

### Inferences
- TRS reports are useful as **priors and validation targets**: for example, per-lane release-time distributions to sanity-check a simulated lane feature. They are not training data.
- [U] World Bank/ADB-supported TRS in Indonesia (e.g., under APEC/ASEAN TRS initiatives) may exist, but I could not confirm one.

### Gaps
- No consolidated WCO list of TRS countries was found.
- No publicly released per-consignment TRS sample was found. If any exists, it is not prominently indexed.

## Q5. Which could realistically support 3-class dwell prediction?

### Takeaway
**None of the open sources verified here can, on their own, support container-level 3-class dwell prediction.** None publish both a per-declaration release time and a physical arrival/gate-out. The realistic options are formal data requests to (1) DJBC/LNSW for Tanjung Perak CEISA/INSW data, which can be joined to the existing TPS data, (2) SUNAT Peru, or (3) RFB Brazil (Santos). Open declaration-level files from Colombia and Chile add only HS/importer/value features with date-only stamps.

### Cited Findings
- Brazil records hour-precision registration and clearance per DI, but measures only registration→clearance, and terminal records are not synchronized. — [RFB Santos study](https://www.gov.br/receitafederal/pt-br/acesso-a-informacao/dados-abertos/resultados/estatisticas_comercio_exterior/arquivos-e-imagens/Variabilidadedostemposdedespachoaduaneirodeimportao2.pdf)
- Peru publishes only average hours by operator. — [SUNAT comunicado 2025](https://www.sunat.gob.pe/operatividadaduanera/novedades/aduanas/comunicados/2025/enero/co140125.pdf)
- Colombia publishes declaration-level files without documented time/channel fields. — [DIAN](https://www.dian.gov.co/dian/cifras/Paginas/Registro-de-las-Declaraciones-de-Impo-Expo.aspx)
- Indonesia: monthly stage aggregates for Tanjung Perak exist and were released to researchers. — [PKN STAN](https://jurnal.pknstan.ac.id/index.php/PBC/article/view/2631)

### Inferences
Summary table (granularity / timestamps / dwell-usable):

| Source | Granularity | Channel | Release time | Arrival / gate-out | Precision | Open? | Usable for 3-class dwell? |
|---|---|---|---|---|---|---|---|
| SUNAT Tiempo de Importación | Aggregate (operator averages) | No | Avg hours | No | Hours (avg) | Yes, web | No (priors only) |
| SUNAT DAM lookup | Single declaration | Yes | Levante status/date | No | Unverified | Lookup only | No |
| DIAN declaration files | Declaration (line) | Not documented | Not documented | No | Date [U] | Yes, no licence stated | No (feature enrichment only) |
| Chile Aduanas DIN (datos.gob.cl) | Declaration line [S] | No [U] | No [U] | No | Date [U] | Yes | No |
| Brazil RFB studies/TRS | Aggregate | By channel (aggregate) | Avg hours | No | Hours (internal) | Reports only | No; request needed |
| Uruguay (IDB research) | Transaction (exports) | Yes | Yes | No | Days | No | No (exports; restricted) |
| India NTRS | Aggregate | Partly | Avg h:mm | Arrival→release/exit (avg) | h:mm (avg) | Reports | No (priors only) |
| US Panjiva/ImportGenius | B/L | No | No | Arrival date only | Date | Commercial | No |
| Indonesia INSW dashboard | Port × month aggregate | No | Customs-clearance avg | Pre/post avg | Days | Dashboard | No; request needed |
| WCO/WB TRS reports | Aggregate | By lane | Avg | Avg | Hours | Reports | No (validation/priors) |

- Best next action: request CEISA/INSW per-container data (PIB filing time, jalur, SPPB time) from KPPBC TMP Tanjung Perak / LNSW for the same 2021 period, and join it to TPS gate-out records.

### Gaps
- Unverified countries (Mexico, Argentina, Ecuador, Costa Rica, Philippines, Thailand, Vietnam, Kenya, South Africa) should be checked for completeness. Ecuador SENAE is the one with some prior likelihood of declaration-level open data.
