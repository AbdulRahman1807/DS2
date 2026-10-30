# Global / port-level datasets mergeable with a Surabaya (ID SUB, Tanjung Perak) container dwell-time dataset, Jun–Nov 2021

Scope note: per-container import dwell (discharge to gate-out), 3 classes. Already merged: IMF PortWatch daily port calls/imports for Surabaya (port1237) and an Indonesian holiday/COVID calendar (gain <0.01 Macro-F1). Indonesian government statistics (INSW/LNSW/DJBC/BPS) are out of scope here. Research done 2026-10-30; roughly 15 tool calls, so several items are marked unverified.

## 1. Global Port Congestion Index (Bai et al.) and similar congestion indices

### Takeaway
I could not find a paper or data release titled "A global port congestion index" by Bai et al. Xiwen Bai's AIS papers cover the 30 largest container ports (lockdown/port calls), 25 major ports (connectivity) or the top 20 ports (congestion). Surabaya is almost certainly not in any of them, and I found no public data release. Commercial daily congestion indices (Spire, GoComet, Portcast) exist but cost money, and none confirms Surabaya coverage for 2021.

### Cited Findings
- Bai, Xu, Han & Yang, "Quantifying the impact of pandemic lockdown policies on global port calls", *Transportation Research Part A* 164 (2022) 224–241, DOI 10.1016/j.tra.2022.08.002. It uses AIS data for the **world's 30 largest container ports** with RDD and DiD models, and studies spillover from China's lockdowns to high- and low-connectivity foreign ports — [RePEc](https://ideas.repec.org/a/eee/transa/v164y2022icp224-241.html); [PolyU portal](https://research.polyu.edu.hk/en/publications/quantifying-the-impact-of-pandemic-lockdown-policies-on-global-po/)
- Bai, Cheng, Yang & Cai (J. Transport Geography 102, 2022) build a **monthly** container port connectivity index from 2020 AIS data for **25 major container ports** — [RePEc](https://ideas.repec.org/a/eee/jotrge/v102y2022ics0966692322001089.html)
- Peng, Bai, Yang, Yuen & Wu, "A deep learning approach for port congestion estimation and prediction", *Maritime Policy & Management* 50(7) (2023) 835–860. It builds **hourly** congestion measures from AIS (DBSCAN plus convex hull) for the **top 20 container ports** — [RePEc](https://ideas.repec.org/a/taf/marpmg/v50y2023i7p835-860.html); [NTU record](https://nanyangtechnologicaluniv.demo.elsevierpure.com/en/publications/a-deep-learning-approach-for-port-congestion-estimation-and-predi)
- Bai, Ma, Hou, Li & Yang, "A Data-Driven Iterative Multi-Attribute Clustering Algorithm and Its Application in Port Congestion Estimation" (IEEE T-ITS 2023). It is AIS-based, with a Los Angeles Aug–Dec 2020 case study — [PolyU](https://research.polyu.edu.hk/en/publications/a-data-driven-iterative-multi-attribute-clustering-algorithm-and-/)
- Spire's Port Congestion Index is a daily per-port time series built from Spire AIS, available "since the beginning of May 2021" for the "world's top ports". It is a commercial product — [Spire](https://spire.com/?p=9266)
- TRADLINX claims a congestion index across 1,253 ports. GoComet and Portcast publish live congestion dashboards. These are commercial and current-state tools, and I found no historical 2021 export — [TRADLINX](https://blogs.tradlinx.com/tradlinx-tracks-global-congestion-across-1253-ports-with-the-port-congestion-index/); [GoComet](https://www.gocomet.com/real-time-port-congestion?=PCA_Button1)

### Inferences
- Even if a Bai index covered Surabaya, it would be a port-level daily or hourly anchorage/berth congestion measure. Your terminal data already contains the 235 actual berthing timestamps, so you can compute berth occupancy, call arrival rate and inter-arrival gaps for the terminal itself, at higher fidelity than any external index. **Expected value for per-container dwell: very low.** It would largely duplicate PortWatch, which already added <0.01 F1.
- The only congestion signal not visible in terminal data is **anchorage waiting time before berthing**. It mainly affects vessel turnaround, not import dwell after discharge.

### Gaps
- I did not find the exact title "A global port congestion index" (Bai et al.). It may be a working paper, a thesis chapter ([PolyU thesis 6657](https://theses.lib.polyu.edu.hk/handle/200/12201)) or a misremembered title. Not verified.
- No data release or supplementary dataset was found for any Bai paper. Contacting the author is the only route.
- Spire coverage of Surabaya was not verified.

## 2. Global port-to-port network / supply-chain datasets (Verschuur, Koks & Hall) and CPPI

### Takeaway
Verschuur, Koks & Hall (Nature Communications 2022) publish a static port-to-port trade network (CC BY 4.0, Mendeley). It is annual and pre-2021, so it gives at most one constant row for Surabaya. CPPI 2021 does include Tanjung Perak, but only as an annual port-level rank and score: Administrative approach rank 107, 41.3 points, 291 calls. **This is a constant for your single-port dataset, so its value for prediction is zero.**

### Cited Findings
- "Global port supply-chains" dataset by J. Verschuur (Oxford), Mendeley Data, DOI 10.17632/vzzy3b9gg4.1, published 27 Jun 2022, **CC BY 4.0**. Folders: Modal_split, Network, Maritime_transport, Port_to_port_network, Port_statistics, Port_IO_tables. It supports "Ports' criticality in international trade and global supply-chains" (Nature Comms 2022) — [Mendeley Data](https://data.mendeley.com/datasets/vzzy3b9gg4)
- The paper's supplement refers to "1176 ports" and a **2015** trade network. The paper finds that for some countries 43.5% of economic activity depends on trade through a single port — [Research Square SI](https://assets-eu.researchsquare.com/files/rs-106378/v1/10d0779cadcbda77ab6fda97.docx); [Oxford Intelligent Earth](https://intelligent-earth.ox.ac.uk/node/2570381)
- CPPI 2021 (World Bank and S&P Global Market Intelligence) uses **full calendar-year 2021** data for 370 ports. It ranks ports on total port hours per call, measured from arrival at port limits or anchorage to departure from berth — [World Bank PDF](https://thedocs.worldbank.org/en/doc/66e3aa5c3be4647addd01845ce353992-0190062022/original/Container-Port-Performance-Index-2021.pdf); [S&P press release](https://press.spglobal.com/2022-05-25-Middle-East-container-ports-are-the-most-efficient-in-the-world)
- **Tanjung Perak values in CPPI 2021** (I extracted these from the PDF text):
  - Administrative approach: rank **107**, **41.333** points.
  - Detailed table: **291 calls** in 2021; per-ship-size-band ranks 97 (<1,500 TEU), 156 (1,501–5,000 TEU) and 31 (5,001–8,500 TEU); 2020H1 rank **172**, a change of +65.
  - Statistical approach in the executive-summary table: rank **103**, index 16.338.
  - Statistical approach in Table A.2: rank **100**, index 17.77. **This conflicts with the executive summary inside the same report. Flag it.**
  - For comparison: Tanjung Priok statistical rank 115 (14.21) and Tanjung Emas 140 (8.21) in Table A.2 — [World Bank CPPI 2021 PDF](https://thedocs.worldbank.org/en/doc/66e3aa5c3be4647addd01845ce353992-0190062022/original/Container-Port-Performance-Index-2021.pdf)
- The CPPI 2021 fuel-consumption weighting uses "Singapore–Surabaya" as the reference feeder sea leg for vessels under 1,500 TEU — same PDF.
- Per-call port-hour data behind CPPI is S&P Global's proprietary port-call product (Sea-web Vessel Movements). The report publishes only aggregates — [S&P Sea-web Vessel Movements](https://ssl.ihsmarkit.com/marketintelligence/en/mi/products/sea-web-vessel-movements.html)

### Inferences
- CPPI and the Verschuur network are **cross-sectional across ports**. In a single-port, single-terminal, 6-month dataset they are constants and cannot add any F1.
- The 291 CPPI calls (full year, whole Tanjung Perak port, ships of at least 1,000 GT under S&P's definition) are consistent with your 235 terminal calls in 6 months only if your terminal handles part of the traffic or the definitions differ. This is a useful sanity check, not a feature.
- The Verschuur network could in principle give *origin-port* attributes (e.g. trade volume, centrality of the port of loading). However, your data has no port of loading, and the vessel is anonymised. Without a vessel or service key, it is unusable.

### Gaps
- I did not confirm whether the Mendeley port statistics include Surabaya explicitly, or which port ID scheme they use. It is probably the World Port Index / Verschuur port IDs that underpin PortWatch, but this is unverified.
- I did not verify whether S&P will sell per-call Sea-web movement records for Surabaya in 2021, or at what price.

## 3. AIS-based recovery of vessel identity/service for each berthing timestamp

### Takeaway
This is **the only external merge that could plausibly add per-container signal**, because it could de-anonymise the vessel. With the vessel identified, you could derive vessel size, service, line/operator and previous port of call, which act as proxies for consignee mix and transshipment versus direct import. The cheapest feasible route is the **Global Fishing Watch port-visits API**. It is free with a registration token and covers all vessel types from 2012 onward. You would time-match GFW port visits at Surabaya to your 235 berthing timestamps. Paid fallbacks are VesselFinder or MarineTraffic historical port calls (credit-based) and Spire or exactEarth raw AIS.

### Cited Findings
- The GFW Events API dataset `public-global-port-visits-events` provides AIS port visits **for all vessel types**, from 2012 to about 72 h before the present. Each visit has a confidence score from 1 to 4. Supported vessel types include CARGO. A separate vessel-identity dataset draws on AIS plus public registries — [GFW Events API docs](https://globalfishingwatch.org/our-apis/documentation/docs/v3/events.md); [GFW product differences PDF](https://globalfishingwatch.org/our-apis/assets/APIs_gfwr_and_Data_Downloads_Products_Differences.pdf)
- GFW warns that its events are algorithmic estimates ("apparent events") — [GFW data caveats](https://api-doc.globalfishingwatch.org/our-apis/documentation/docs/v3/general-api-doc/data-caveats)
- VesselFinder Port Calls API: query by port LOCODE for vessels with ATA and ATD, at 2 credits per record. Prices: 10,000 credits = €330, 20,000 = €625, 50,000 = €1,470 (excl. VAT); credits are valid for 12 months. The page does not state how far back history goes — [VesselFinder Port Calls API](https://www.vesselfinder.com/port-calls-api)
- MarineTraffic (Kpler) historical port calls are credit-metered, with prices only through sales. The Enterprise "Port Call Trends" product offers up to 5 years of history — [MarineTraffic support](https://support.marinetraffic.com/en/articles/9552785); [Kpler](https://www.kpler.com/solutions/derived-insights/port-call-trends)
- BPS Indonesia obtained exactEarth satellite AIS for Indonesia's EEZ (2022) through the UN Global Platform. This shows the UN Global Platform AIS covers Indonesian waters, but access is for official statistics offices — [UNSD presentation](https://unstats.un.org/bigdata/events/2025/ai-data-science/webinar1/presentations/Setia%20-%20UN%20AI%20Sprint%20BPS%20Indonesia%20short.pdf)
- A UGM study found AIS-derived ship-call counts correlate well with official data, but **AIS-derived stay times did not**. So berth timestamps from AIS are noisy, and matching should use generous windows — [UGM ETD](https://etd.repository.ugm.ac.id/penelitian/detail/187209)
- Tanjung Perak is a very busy port, with about 38,800 ship calls in 2008. An ITS Surabaya AIS receiver was used in academic work — [Kobe University dissertation](https://da.lib.kobe-u.ac.jp/da/kernel/D1006490/D1006490y.pdf). Many of these calls are domestic or small craft, so many candidate vessels may overlap any given berthing window.
- Maritime Optima publishes free vessel and port pages for Tanjung Perak-area ports (e.g. IDTPP) — [Maritime Optima](https://maritimeoptima.com/public/ports/pages/IDTPP)

### Inferences
- **Proposed matching procedure:**
  1. Pull GFW port visits whose anchorage or port polygon is Surabaya / Tanjung Perak, for Jun–Nov 2021, filtered to CARGO.
  2. Restrict further to container ships via the GFW vessel-identity dataset or IMO lookup.
  3. Match each of the 235 terminal berthing timestamps to the visit whose start or end is within ±N hours.
  4. Check a candidate by comparing the number of containers discharged per call against vessel TEU capacity.

  Tanjung Perak has several container facilities (TPS, Terminal Teluk Lamong, Berlian / domestic terminals), so ambiguous matches are likely. A uniqueness rate of maybe 50–80% is a guess and has not been tested.
- **Data-governance caution:** vessel identity was deliberately anonymised. Re-identification may breach the data provider's terms, so check with the data owner before doing it.
- **Ceiling on gain:** within 235 calls, a "vessel/service" key is a high-cardinality, low-sample categorical. Its main benefit is enabling *service-level* features (e.g. intra-Asia feeder versus direct mainline, a carrier's free-time policy). The repo's existing call-level features (call ID, call size) may already capture most call-level variance. Run a quick test first: compute between-call versus within-call variance of the dwell class in your data. If most variance is within-call, vessel identity cannot help much.
- **Free European AIS will not help:** the Danish Maritime Authority AIS and NOAA MarineCadastre cover only their own waters, not Indonesia (general knowledge, not re-verified here).

### Gaps
- GFW licence terms (believed to be CC BY-NC 4.0 for API data, so non-commercial) were not verified in this session.
- I did not verify GFW's anchorage polygon naming for Surabaya, or how well it separates TPS from Teluk Lamong.
- The historical depth of VesselFinder's port-call API for 2021 is not stated. Spire, exactEarth and MarineTraffic prices for a 6-month Surabaya extract were not found.

## 4. Shipping-line schedules and container owner-prefix → operator/lessor mapping

### Takeaway
I found no public archive of 2021 Surabaya liner service schedules. Sea-Intelligence schedule-reliability data is trade-lane or carrier level, paid, and not port-specific (unverified). The **owner prefix → company mapping is the cheapest high-value merge**. BIC's owner-code lookup is available through the free web search and a registered fair-use API. It turns the ISO 6346 prefix into owner name and country, from which you can hand-label carrier versus lessor and carrier alliance.

### Cited Findings
- BIC exposes three APIs: owner-code lookup ("Identify the owner of the container from a BIC Code"), facility-code lookup, and BoxTech container technical data. Registration is required, and use is on a "fair use" basis — [BIC API page](https://www.bic-code.org/api/); [API registration](https://bic-code.org/api-registration/)
- BoxTech offers web and API access to container technical data for 11m+ containers, including Maersk and HMM fleets — [BoxTech about](https://www.bic-boxtech.org/about); [BIC news PDF](https://www.bic-code.org/wp-content/uploads/2018/07/BoxTech-database-exceeds-11m-containers-following-HMM-fleet-upload.pdf); [BoxTech GitHub](https://github.com/bic-org/Boxtech)
- A third-party listing says BoxTech has a free tier plus paid tiers. Unverified — [Apispine](https://apispine.com/bic-boxtech/pricing)
- BIC registry fees (for registrants, not readers): €2,000 registration plus €475 per year renewal — [BIC tariff schedule](https://www.bic-code.org/wp-content/uploads/2026/01/Schedule-of-Charges_Tariffs.pdf)
- Services at TPS:
  - New weekly services RSU1 (RCL, from 22 Mar 2022) and SSE (PIL, from 25 Apr 2022), plus monthly KPG (Kaiso) — [Beritajatim](https://beritajatim.com/?p=875782). These are post-period and show that the service mix changes over time.
  - TPS international container volume fell in mid-2021 — [Bisnis.com Jul 2021](https://ekonomi.bisnis.com/read/20210721/98/1420277/pelindo-iii-arus-peti-kemas-internasional-masih-stagnan)
  - A 2014 list of international carriers at Surabaya includes Evergreen, CMA, CSCL, MOL, MSC and OOCL — [Prasetyadi 2014](https://transportevents.com/presentations/Jakarta2014/Prasetyadi.pdf)

### Inferences
- **Owner-prefix mapping, expected value: low to moderate.**
  - The model can already learn prefix as a categorical, so mapping mainly helps in two ways. First, it pools rare prefixes into owner-type groups (carrier-owned versus leasing company such as TRIU, TGHU, CAIU, SEGU, TCNU). Second, it groups carriers by alliance.
  - Lessor-owned boxes under a carrier's operation hide the operator, which is a known limitation.
  - The work is cheap: one lookup per unique prefix, probably fewer than 200.
- **Schedule data:** if vessels were de-anonymised (section 3), carrier and service could be attached from the vessel's operator in public registries. Without that, schedule-reliability series are trade-lane level and would be near-constant over 6 months.

### Gaps
- Sea-Intelligence GLP coverage and price were not researched in this session. Its port-level coverage for Surabaya is unverified, and I believe it covers only major trade lanes.
- I found no archive of 2021 TPS berth windows or service lists. Pelindo or TPS annual reports might list them, but this is unverified.
- The exact BIC owner-code API response fields (name, country, lessor flag) were not confirmed.

## 5. UNCTAD port-call statistics, World Bank LPI, CPPI values

### Takeaway
UNCTADstat publishes AIS-derived **country-level, annual or semi-annual** median time in port by ship type. LPI is country-level and roughly biennial (2018, then 2023). CPPI 2021 values for Tanjung Perak are given in section 2. **All are constant over Jun–Nov 2021 for one port, so their predictive value is zero.** They are useful only as narrative context.

### Cited Findings
- UNCTAD port-call and time-in-port statistics come from AIS combined with port mapping intelligence (MarineTraffic), for ships of 1,000 GT and above — [UNCTAD](https://unctad.org/node/30289); [UN iLibrary](https://www.un-ilibrary.org/content/books/9789210019156c006)
- UNCTADstat maritime country profile for Indonesia (M49 code 360), with port calls and time in port. The 2023 all-ships median was 1.26 days — [UNCTADstat profile](https://unctadstat.unctad.org/CountryProfile/MaritimeProfile/en-GB/360/index.html)
- Indonesia container-ship median time in port:
  - Review of Maritime Transport 2021 snippet: Indonesia at about **0.99 days** among the top-25 container countries. The snippet is truncated and needs verification — [UNCTAD](https://unctad.org/node/35916)
  - 2018: Indonesia was among 4 of the top 25 countries with median container turnaround above 1 day — [UNCTAD](https://unctad.org/node/3939)
- An Indonesian minister cited a median container waiting time of 24.9 h for H1 2022. This is a government claim, and the metric's definition is unclear — [Hellenic Shipping News](https://www.hellenicshippingnews.com/?p=961875)
- CPPI 2021 Tanjung Perak: Administrative approach rank 107 (41.3 points). Statistical approach rank 100 or 103, which conflict within the report — see section 2.

### Inferences
- Use these only to describe the operating context in the report, e.g. "Tanjung Perak ranked 107/370 in CPPI 2021, improved from 172 in 2020H1". They cannot be features.

### Gaps
- I did not extract the exact UNCTADstat semi-annual 2021 container median time in port for Indonesia (it requires an interactive query).
- I did not fetch the LPI 2018 and 2023 scores for Indonesia, because they are irrelevant as features.

## 6. Weather: ERA5 hourly, BMKG Perak station, alternatives

### Takeaway
Weather is the one external **time-varying, sub-daily, local** signal not already present in terminal data. Rain stops quay and yard crane work and can slow gate and trucking. It is still a port-wide shock, so its effect on a 65 h / 161 h dwell class boundary is probably small. The data is free and easy to get:
- **ERA5:** hourly, 0.25°, 1940–present, CC-BY, from Copernicus CDS.
- **BMKG Data Online:** Stasiun Meteorologi Maritim Tanjung Perak, free registration. Daily resolution is likely; hourly is unverified.
- **GPM IMERG:** half-hourly at 0.1°, a better rainfall product for the tropics (from general knowledge; not verified in this session).

### Cited Findings
- ERA5 single levels: 1940 to present, **hourly**, 0.25° × 0.25° atmosphere, **CC-BY** licence, about 5-day latency, download through the Copernicus CDS with login — [Copernicus CDS ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
- BMKG Data Online (dataonline.bmkg.go.id) is BMKG's official data portal. Users report downloading monthly chunks of station data. Contact: dataonline@bmkg.go.id — [Discourse post](https://discourse.weareopen.coop/news/bmkg-online-your-go-to); [WMO PMO Indonesia doc](https://old.wmo.int/extranet/pages/prog/amp/mmop/documents/sot/PMO4/PMO4-15-NR-Indonesia.pdf)
- The Stasiun Meteorologi Maritim Tanjung Perak is on Jl. Kalimas Baru, Surabaya. BMKG Regulation 7/2023 lists rainfall among maritime station observations — [WMO PMO doc](https://old.wmo.int/extranet/pages/prog/amp/mmop/documents/sot/PMO4/PMO4-15-NR-Indonesia.pdf); [Peraturan BMKG 7/2023](https://www.peraturan.go.id/files/peraturan-bmkg-no-7-tahun-2023.pdf)
- Surabaya climate statistics use the Perak I, Perak II and Juanda stations — [Surabaya govt geography chapter](https://surabaya.go.id/uploads/attachments/2019/11/45085/BAB_I_Keadaan_Geografis.pdf?1573615373)
- BMKG maritime route weather for Surabaya — [peta-maritim BMKG](https://peta-maritim.bmkg.go.id/marine-data/rute/pdf/surabaya.pdf)

### Inferences
- **Recommended features** (as-of, no leakage):
  - Rainfall hours or mm in the 24 h and 72 h after discharge, computed at prediction time from data available up to the discharge time.
  - Note: rain *after* discharge is future information unless a forecast is used. For as-of prediction, only past or at-discharge weather is legitimate. That weakens the feature further.
- Jun–Nov is mostly the East Java dry season (roughly Jun–Oct), with rains starting around Nov. So there will be little rain variation in the window, and **expected F1 gain is very low (<0.005)**.
- Free alternative for hourly station data: METAR/SYNOP for Juanda airport (WARR), via NOAA ISD or Meteostat. Station IDs were not verified in this session.

### Gaps
- I did not confirm whether BMKG Data Online offers hourly rainfall for the Perak station or only daily values.
- NOAA ISD / Meteostat station IDs for Perak II and Juanda were not verified.
- I did not verify ERA5 total-precipitation skill over Surabaya's coast. ERA5 is known to be weak on convective tropical rain (general knowledge).

## Overall ranking for per-container dwell prediction (synthesis; inference)
1. **Owner prefix → owner/lessor/alliance (BIC):** cheap, container-level key. Low to moderate value.
2. **Vessel re-identification via AIS (GFW free API, VesselFinder paid):** the only route to service, line and origin features, but it has governance caveats and ambiguous matching. Value is moderate only if between-call variance of dwell is large, so test that first.
3. **Weather (ERA5 / BMKG / IMERG):** free, hourly and local, but dry-season and port-wide. Very low value.
4. **Congestion indices, CPPI, UNCTAD, LPI, Verschuur network:** port-level or annual constants, or duplicates of terminal data and PortWatch. Essentially zero value. Use only for context.
