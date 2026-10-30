"""Indonesian Operational Calendar Reference (2021).

Assigned Role: Data Engineer

Responsibilities:
- Persist national public holidays (customs offices and banks closed) for 2021
- Persist Surabaya COVID-19 activity-restriction (PPKM) levels for the study window
- Every row carries its source so the reference can be audited

All dates were announced in advance (holiday decree SKB 3 Menteri, revised June 2021; PPKM
instructions announced before each period), so they are legitimately known at prediction time.
Output: data/reference/indonesia_calendar_2021.csv
"""

from pathlib import Path

import pandas as pd

HOLIDAYS_2021 = [
    ("2021-01-01", "New Year"),
    ("2021-02-12", "Chinese New Year"),
    ("2021-03-11", "Isra Mi'raj"),
    ("2021-03-14", "Nyepi"),
    ("2021-04-02", "Good Friday"),
    ("2021-05-01", "Labour Day"),
    ("2021-05-13", "Idul Fitri / Ascension Day"),
    ("2021-05-14", "Idul Fitri"),
    ("2021-05-26", "Waisak"),
    ("2021-06-01", "Pancasila Day"),
    ("2021-07-20", "Idul Adha"),
    ("2021-08-11", "Islamic New Year (moved from 10 Aug)"),
    ("2021-08-17", "Independence Day"),
    ("2021-10-19", "Maulid Nabi (moved from 20 Oct)"),
    ("2021-12-25", "Christmas"),
]
HOLIDAY_SOURCE = "SKB 3 Menteri on 2021 national holidays, revised June 2021"

# (start, end inclusive, level, label, source)
PPKM_SURABAYA_2021 = [
    ("2021-06-01", "2021-07-02", 1, "PPKM Mikro", "pre-emergency micro-scale restrictions"),
    ("2021-07-03", "2021-07-20", 4, "PPKM Darurat",
     "https://www.jawapos.com/nasional/2107010108/selain-jakarta-dan-surabaya-ini-daftar-wilayah-ppkm-darurat"),
    ("2021-07-21", "2021-08-23", 4, "PPKM Level 4",
     "Inmendagri 22/2021 and extensions 21-25 Jul, 26 Jul-2 Aug, 3-9 Aug, 10-16 Aug, 17-23 Aug"),
    ("2021-08-24", "2021-09-06", 3, "PPKM Level 3",
     "https://kabar24.bisnis.com/read/20210823/15/1433043/jokowi-jabodetabek-bandung-raya-dan-surabaya-raya-terapkan-ppkm-level-3"),
    ("2021-09-07", "2021-10-31", 2, "PPKM Level 2",
     "https://en.antaranews.com/news/187690/surabaya-targets-lowering-its-ppkm-level-from-2-to-1 (Kemenkes assessment 4 Sep)"),
]


def build_calendar(start: str = "2021-05-01", end: str = "2021-11-30") -> pd.DataFrame:
    days = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    holidays = pd.DataFrame(HOLIDAYS_2021, columns=["date", "holiday_name"])
    holidays["date"] = pd.to_datetime(holidays["date"])
    days = days.merge(holidays, on="date", how="left")
    days["is_public_holiday"] = days["holiday_name"].notna().astype(int)
    days["is_weekend"] = (days["date"].dt.dayofweek >= 5).astype(int)
    days["is_working_day"] = ((days["is_public_holiday"] == 0) & (days["is_weekend"] == 0)).astype(int)
    days["holiday_source"] = days["holiday_name"].where(days["holiday_name"].isna(), HOLIDAY_SOURCE)

    days["ppkm_level"] = pd.NA
    days["ppkm_label"] = pd.NA
    days["ppkm_source"] = pd.NA
    for s, e, level, label, source in PPKM_SURABAYA_2021:
        mask = (days["date"] >= s) & (days["date"] <= e)
        days.loc[mask, ["ppkm_level", "ppkm_label", "ppkm_source"]] = [level, label, source]
    return days


def run_calendar_ingest(output_path: str = "data/reference/indonesia_calendar_2021.csv") -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cal = build_calendar()
    cal.to_csv(out, index=False)
    print(f"Saved {len(cal)} calendar days to {out}")
    return out


if __name__ == "__main__":
    run_calendar_ingest()
