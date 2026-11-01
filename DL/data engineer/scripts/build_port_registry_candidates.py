
from pathlib import Path
import json
import re
import unicodedata
from collections import defaultdict

import pandas as pd
import pycountry
from rapidfuzz import fuzz, process


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

PORTWATCH_PATH = ROOT / "data/raw/portwatch/Daily_Ports_Data.csv"
UNLOCODE_DIR = ROOT / "data/reference/unlocode"
UPPLY_PATH = ROOT / "data/reference/upply/UPPLY-SEAPORTS.csv"

OUTPUT_DIR = ROOT / "reports/port_registry"

TOP_K = 5
FUZZY_REVIEW_THRESHOLD = 75
HIGH_SIMILARITY_THRESHOLD = 95


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_name(value):
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )

    value = value.upper()
    value = re.sub(r"[^A-Z0-9]+", " ", value)

    return " ".join(value.split())


def iso2_from_iso3(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    try:
        country = pycountry.countries.get(alpha_3=value)
        return country.alpha_2 if country else None
    except (KeyError, LookupError):
        return None


def valid_country_name(code):
    if not code or pd.isna(code):
        return ""

    try:
        country = pycountry.countries.get(alpha_2=str(code).upper())
        return country.name if country else ""
    except (KeyError, LookupError):
        return ""


def classify_match(score, exact):
    if exact:
        return "EXACT_NORMALIZED_NAME_COUNTRY"

    if score >= HIGH_SIMILARITY_THRESHOLD:
        return "HIGH_SIMILARITY_REVIEW"

    if score >= FUZZY_REVIEW_THRESHOLD:
        return "FUZZY_REVIEW"

    return "LOW_SIMILARITY"


# =========================================================
# 1. LOAD PORTWATCH PORTS
# =========================================================

def load_portwatch():

    print("Loading distinct PortWatch ports...")

    frames = []

    for chunk in pd.read_csv(
        PORTWATCH_PATH,
        usecols=[
            "portid",
            "portname",
            "country",
            "ISO3",
        ],
        dtype=str,
        chunksize=250_000,
    ):
        frames.append(
            chunk.drop_duplicates("portid")
        )

    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("portid")
        .reset_index(drop=True)
    )

    df["portwatch_country_iso2"] = (
        df["ISO3"].map(iso2_from_iso3)
    )

    df["portwatch_name_norm"] = (
        df["portname"].map(normalize_name)
    )

    print("PortWatch ports:", len(df))
    print(
        "Unresolved PortWatch country codes:",
        int(df["portwatch_country_iso2"].isna().sum()),
    )

    return df


# =========================================================
# 2. LOAD ALL UN/LOCODE PARTS
# =========================================================

def load_unlocode():

    print("\nLoading UN/LOCODE records...")

    files = sorted(UNLOCODE_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No UN/LOCODE CSV files found in {UNLOCODE_DIR}"
        )

    frames = []

    for path in files:

        raw = pd.read_csv(
            path,
            header=None,
            dtype=str,
        )

        # Columns:
        # 1 country code
        # 2 location code
        # 3 location name
        # 4 ASCII location name
        # 6 function codes
        # 10 coordinates

        valid = raw[
            raw[1].notna()
            & raw[2].notna()
            & raw[2].str.strip().ne("")
        ].copy()

        df = pd.DataFrame({
            "unlocode_country_iso2": (
                valid[1].str.strip().str.upper()
            ),
            "unlocode_location_code": (
                valid[2].str.strip().str.upper()
            ),
            "unlocode_name": valid[3].fillna("").str.strip(),
            "unlocode_name_ascii": (
                valid[4].fillna("").str.strip()
            ),
            "unlocode_functions": (
                valid[6].fillna("").str.strip()
            ),
            "unlocode_coordinates_raw": (
                valid[10].fillna("").str.strip()
            ),
        })

        df["unlocode"] = (
            df["unlocode_country_iso2"]
            + df["unlocode_location_code"]
        )

        df["unlocode_country_name"] = (
            df["unlocode_country_iso2"]
            .map(valid_country_name)
        )

        df["unlocode_name_norm"] = (
            df["unlocode_name"].map(normalize_name)
        )

        df["unlocode_name_ascii_norm"] = (
            df["unlocode_name_ascii"].map(normalize_name)
        )

        # UN/LOCODE function code:
        # First character 1 means port.
        df["has_port_function"] = (
            df["unlocode_functions"].str[0] == "1"
        )

        frames.append(df)

        print(
            path.name,
            "| valid locations:",
            len(df),
            "| port-function:",
            int(df["has_port_function"].sum()),
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = result.drop_duplicates(
        subset=["unlocode"]
    ).reset_index(drop=True)

    print("Unique UN/LOCODE locations:", len(result))
    print(
        "Unique port-function locations:",
        int(result["has_port_function"].sum()),
    )

    return result


# =========================================================
# 3. LOAD UPPLY
# =========================================================

def load_upply():

    print("\nLoading Upply seaports...")

    df = pd.read_csv(
        UPPLY_PATH,
        sep=";",
        dtype={
            "code": str,
            "country_code": str,
        },
    )

    df["code"] = df["code"].str.strip().str.upper()

    df["country_code"] = (
        df["country_code"].str.strip().str.upper()
    )

    df["upply_country_name"] = (
        df["country_code"].map(valid_country_name)
    )

    df["upply_name_norm"] = (
        df["name"].map(normalize_name)
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    print("Upply records:", len(df))

    return df


# =========================================================
# 4. EXACT UPPLY -> UN/LOCODE IDENTIFIER MATCHING
# =========================================================

def match_upply_identifiers(upply, unlocode):

    print("\nMatching Upply identifiers against UN/LOCODE...")

    unlo_lookup = (
        unlocode.drop_duplicates("unlocode")
        .set_index("unlocode")
    )

    unlo_codes = set(unlo_lookup.index)

    result = upply.copy()

    result["unlocode_exact_match"] = (
        result["code"].isin(unlo_codes)
    )

    result["matched_unlocode"] = (
        result["code"].where(
            result["unlocode_exact_match"],
            None,
        )
    )

    result["unlocode_name"] = (
        result["code"].map(
            unlo_lookup["unlocode_name"]
        )
    )

    result["unlocode_country_iso2"] = (
        result["code"].map(
            unlo_lookup["unlocode_country_iso2"]
        )
    )

    result["country_code_agreement"] = (
        result["country_code"]
        == result["unlocode_country_iso2"]
    )

    result["identifier_match_status"] = "UNMATCHED"

    result.loc[
        result["unlocode_exact_match"]
        & result["country_code_agreement"],
        "identifier_match_status",
    ] = "EXACT_CODE_AND_COUNTRY_AGREEMENT"

    result.loc[
        result["unlocode_exact_match"]
        & ~result["country_code_agreement"],
        "identifier_match_status",
    ] = "EXACT_CODE_COUNTRY_DISAGREEMENT"

    return result


# =========================================================
# 5. COUNTRY-RESTRICTED FUZZY CANDIDATE GENERATION
# =========================================================

def build_country_indexes(reference, country_col, name_columns):

    indexes = defaultdict(dict)

    for row in reference.itertuples(index=False):

        record = row._asdict()
        country = record.get(country_col)

        if not country:
            continue

        for name_col in name_columns:

            name = record.get(name_col)

            if not name:
                continue

            # Retain a deterministic representative for each
            # normalized string; the complete reference dataset
            # remains available in the source.
            indexes[country].setdefault(
                name,
                record,
            )

    return indexes


def generate_candidates(
    portwatch,
    reference,
    reference_country_col,
    reference_name_columns,
    reference_id_col,
    source_label,
):

    print(f"\nGenerating {source_label} candidates...")

    indexes = build_country_indexes(
        reference,
        reference_country_col,
        reference_name_columns,
    )

    results = []

    for port in portwatch.itertuples(index=False):

        port_record = port._asdict()

        country = port_record.get(
            "portwatch_country_iso2"
        )

        port_name = port_record.get(
            "portwatch_name_norm"
        )

        base = {
            "portid": port_record["portid"],
            "portwatch_name": port_record["portname"],
            "portwatch_country": port_record["country"],
            "portwatch_iso3": port_record["ISO3"],
            "portwatch_country_iso2": country,
        }

        if not country or not port_name:
            results.append({
                **base,
                "candidate_rank": None,
                "candidate_id": None,
                "candidate_name": None,
                "similarity_score": None,
                "match_method": "UNMATCHABLE_MISSING_COUNTRY_OR_NAME",
            })
            continue

        choices = indexes.get(country, {})

        if not choices:
            results.append({
                **base,
                "candidate_rank": None,
                "candidate_id": None,
                "candidate_name": None,
                "similarity_score": None,
                "match_method": "NO_REFERENCE_IN_COUNTRY",
            })
            continue

        matches = process.extract(
            port_name,
            list(choices.keys()),
            scorer=fuzz.token_sort_ratio,
            limit=TOP_K,
            score_cutoff=FUZZY_REVIEW_THRESHOLD,
        )

        if not matches:
            results.append({
                **base,
                "candidate_rank": None,
                "candidate_id": None,
                "candidate_name": None,
                "similarity_score": None,
                "match_method": "NO_CANDIDATE_ABOVE_THRESHOLD",
            })
            continue

        for rank, (matched_name, score, _) in enumerate(
            matches,
            start=1,
        ):

            matched_record = choices[matched_name]

            exact = port_name == matched_name

            candidate_id = matched_record.get(
                reference_id_col
            )

            results.append({
                **base,
                "candidate_rank": rank,
                "candidate_id": candidate_id,
                "candidate_name": matched_name,
                "similarity_score": round(float(score), 2),
                "match_method": classify_match(
                    score,
                    exact,
                ),
            })

    return pd.DataFrame(results)


# =========================================================
# 6. MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    portwatch = load_portwatch()
    unlocode = load_unlocode()
    upply = load_upply()

    # Exact Upply identifier matching.
    upply_matches = match_upply_identifiers(
        upply,
        unlocode,
    )

    upply_matches.to_csv(
        OUTPUT_DIR / "upply_unlocode_identifier_matches.csv",
        index=False,
    )

    # PortWatch -> UN/LOCODE.
    unlo_ports = unlocode[
        unlocode["has_port_function"]
    ].copy()

    pw_unlo = generate_candidates(
        portwatch=portwatch,
        reference=unlo_ports,
        reference_country_col="unlocode_country_iso2",
        reference_name_columns=[
            "unlocode_name_norm",
            "unlocode_name_ascii_norm",
        ],
        reference_id_col="unlocode",
        source_label="PortWatch -> UN/LOCODE",
    )

    pw_unlo.to_csv(
        OUTPUT_DIR / "portwatch_unlocode_candidates.csv",
        index=False,
    )

    # PortWatch -> Upply.
    # Upply's country_code is ISO alpha-2.
    pw_upply = generate_candidates(
        portwatch=portwatch,
        reference=upply,
        reference_country_col="country_code",
        reference_name_columns=[
            "upply_name_norm",
        ],
        reference_id_col="code",
        source_label="PortWatch -> Upply",
    )

    pw_upply.to_csv(
        OUTPUT_DIR / "portwatch_upply_candidates.csv",
        index=False,
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    def summarize_candidates(df):

        counts = df["match_method"].value_counts(
            dropna=False
        )

        return {
            str(key): int(value)
            for key, value in counts.items()
        }

    exact_count = int(
        upply_matches["unlocode_exact_match"].sum()
    )

    country_disagreements = int(
        (
            upply_matches["unlocode_exact_match"]
            & ~upply_matches["country_code_agreement"]
        ).sum()
    )

    summary = {
        "status": "CANDIDATES_GENERATED_PENDING_REVIEW",
        "portwatch_port_count": len(portwatch),
        "unlocode_location_count": len(unlocode),
        "unlocode_port_function_count": len(unlo_ports),
        "upply_record_count": len(upply),
        "upply_exact_unlocode_identifier_matches": exact_count,
        "upply_identifier_country_disagreements": (
            country_disagreements
        ),
        "portwatch_unlocode_candidate_status": (
            summarize_candidates(pw_unlo)
        ),
        "portwatch_upply_candidate_status": (
            summarize_candidates(pw_upply)
        ),
        "matching_policy": {
            "country_restricted": True,
            "top_k": TOP_K,
            "fuzzy_review_threshold": FUZZY_REVIEW_THRESHOLD,
            "high_similarity_threshold": (
                HIGH_SIMILARITY_THRESHOLD
            ),
            "automatic_fuzzy_merges": False,
            "unmatched_records_preserved": True,
        },
        "outputs": [
            "upply_unlocode_identifier_matches.csv",
            "portwatch_unlocode_candidates.csv",
            "portwatch_upply_candidates.csv",
        ],
    }

    summary_path = (
        OUTPUT_DIR / "matching_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("PORT REGISTRY CANDIDATE GENERATION COMPLETE")
    print("=" * 65)

    print(json.dumps(summary, indent=2))

    print("\nOutput directory:", OUTPUT_DIR)
    print("Source files modified: NO")


if __name__ == "__main__":
    main()
