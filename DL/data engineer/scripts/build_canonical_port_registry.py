
from pathlib import Path
from collections import defaultdict
import json
import re
import unicodedata

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

OUTPUT_DIR = ROOT / "reports/port_registry/v1_2_identifier_bridge"

TOP_K = 5
FUZZY_THRESHOLD = 75

OUTPUT_FILES = [
    "canonical_port_registry.csv",
    "port_identity_candidates.csv",
    "unresolved_port_matches.csv",
    "source_reconciliation.csv",
    "registry_metadata.json",
]


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_name(value):
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        c for c in value if not unicodedata.combining(c)
    )
    value = value.upper()
    value = re.sub(r"[^A-Z0-9]+", " ", value)

    return " ".join(value.split())


def iso2_from_iso3(value):
    if pd.isna(value):
        return None

    country = pycountry.countries.get(
        alpha_3=str(value).strip().upper()
    )

    return country.alpha_2 if country else None


def valid_iso2(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    country = pycountry.countries.get(alpha_2=value)

    return country.alpha_2 if country else None


def unique_values(values):
    return sorted({
        str(v) for v in values
        if pd.notna(v) and str(v).strip()
    })


# =========================================================
# 1. PORTWATCH
# =========================================================

def load_portwatch():

    print("Loading PortWatch...")

    frames = []

    for chunk in pd.read_csv(
        PORTWATCH_PATH,
        usecols=["portid", "portname", "country", "ISO3"],
        dtype=str,
        chunksize=250_000,
    ):
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)

    # A port ID should have one stable identity.
    identity_counts = (
        df.groupby("portid")
        .agg(
            names=("portname", "nunique"),
            countries=("country", "nunique"),
            iso3_codes=("ISO3", "nunique"),
        )
    )

    unstable = identity_counts[
        (identity_counts["names"] > 1)
        | (identity_counts["countries"] > 1)
        | (identity_counts["iso3_codes"] > 1)
    ]

    if len(unstable):
        raise ValueError(
            "PortWatch IDs have inconsistent identity fields. "
            f"Inspect these {len(unstable)} IDs before proceeding:\n"
            f"{unstable.head(20)}"
        )

    df = (
        df.drop_duplicates("portid")
        .reset_index(drop=True)
    )

    df["portwatch_country_iso2"] = (
        df["ISO3"].map(iso2_from_iso3)
    )

    df["portwatch_name_norm"] = (
        df["portname"].map(normalize_name)
    )

    if df["portid"].duplicated().any():
        raise ValueError("Duplicate PortWatch IDs remain.")

    if df["portwatch_country_iso2"].isna().any():
        raise ValueError(
            "Unresolved PortWatch ISO3 country code detected."
        )

    print("PortWatch ports:", len(df))

    return df


# =========================================================
# 2. UN/LOCODE
# =========================================================

def load_unlocode():

    print("\nLoading UN/LOCODE...")

    frames = []

    files = sorted(UNLOCODE_DIR.glob("*.csv"))

    if len(files) != 3:
        raise ValueError(
            f"Expected 3 UN/LOCODE parts, found {len(files)}."
        )

    for path in files:

        raw = pd.read_csv(
            path,
            header=None,
            dtype=str,
        )

        valid = raw[
            raw[1].notna()
            & raw[2].notna()
            & raw[2].str.strip().ne("")
        ].copy()

        part = pd.DataFrame({
            "unlocode_country_iso2": (
                valid[1].str.strip().str.upper()
            ),
            "unlocode_location_code": (
                valid[2].str.strip().str.upper()
            ),
            "unlocode_name": (
                valid[3].fillna("").str.strip()
            ),
            "unlocode_name_ascii": (
                valid[4].fillna("").str.strip()
            ),
            "unlocode_functions": (
                valid[6].fillna("").str.strip()
            ),
            "unlocode_coordinates_raw": (
                valid[10].fillna("").str.strip()
            ),
            "unlocode_source_file": path.name,
            "unlocode_source_row": valid.index + 1,
        })

        part["unlocode"] = (
            part["unlocode_country_iso2"]
            + part["unlocode_location_code"]
        )

        part["unlocode_name_norm"] = (
            part["unlocode_name"].map(normalize_name)
        )

        part["unlocode_ascii_norm"] = (
            part["unlocode_name_ascii"].map(normalize_name)
        )

        part["has_port_function"] = (
            part["unlocode_functions"].str[0] == "1"
        )

        frames.append(part)

        print(
            path.name,
            "| records:", len(part),
            "| port-function:",
            int(part["has_port_function"].sum()),
        )

    # IMPORTANT: Do not deduplicate identifiers.
    result = pd.concat(
        frames,
        ignore_index=True,
    )

    print("Total UN/LOCODE records:", len(result))
    print("Unique identifiers:", result["unlocode"].nunique())
    print(
        "Port-function records:",
        int(result["has_port_function"].sum()),
    )

    return result


# =========================================================
# 3. UPPLY
# =========================================================

def load_upply():

    print("\nLoading Upply...")

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
        df["country_code"].map(valid_iso2)
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

    if df["code"].duplicated().any():
        raise ValueError(
            "Duplicate Upply codes detected. "
            "Inspect before registry construction."
        )

    print("Upply records:", len(df))
    print("Unique Upply codes:", df["code"].nunique())

    return df


# =========================================================
# 4. BUILD MULTI-VALUE LOOKUPS
# =========================================================

def build_unlo_indexes(unlo):

    exact_lookup = defaultdict(list)
    fuzzy_lookup = defaultdict(lambda: defaultdict(list))
    identifier_lookup = defaultdict(list)

    for row in unlo.itertuples(index=False):

        record = row._asdict()

        identifier_lookup[row.unlocode].append(record)

        if not row.has_port_function:
            continue

        country = row.unlocode_country_iso2

        for name in {
            row.unlocode_name_norm,
            row.unlocode_ascii_norm,
        }:
            if not name:
                continue

            exact_lookup[(country, name)].append(record)
            fuzzy_lookup[country][name].append(record)

    return exact_lookup, fuzzy_lookup, identifier_lookup


def build_upply_indexes(upply):

    exact_lookup = defaultdict(list)
    fuzzy_lookup = defaultdict(lambda: defaultdict(list))
    identifier_lookup = {}

    for row in upply.itertuples(index=False):

        record = row._asdict()

        identifier_lookup[row.code] = record

        if not row.country_code or not row.upply_name_norm:
            continue

        exact_lookup[
            (row.country_code, row.upply_name_norm)
        ].append(record)

        fuzzy_lookup[
            row.country_code
        ][row.upply_name_norm].append(record)

    return exact_lookup, fuzzy_lookup, identifier_lookup


# =========================================================
# 5. CANDIDATE GENERATION
# =========================================================

def get_candidates(
    port,
    exact_lookup,
    fuzzy_lookup,
    source,
):

    country = port["portwatch_country_iso2"]
    name = port["portwatch_name_norm"]

    exact_records = exact_lookup.get(
        (country, name),
        [],
    )

    candidates = []

    if exact_records:

        # Preserve every reference row, even if identifiers
        # are duplicated or names collide.
        for record in exact_records:

            candidate_id = (
                record["unlocode"]
                if source == "UNLOCODE"
                else record["code"]
            )

            candidate_name = (
                record["unlocode_name"]
                if source == "UNLOCODE"
                else record["name"]
            )

            candidates.append({
                "candidate_source": source,
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "similarity_score": 100.0,
                "match_method": "EXACT_NORMALIZED_NAME_COUNTRY",
                "source_record": record,
            })

        return candidates

    choices = fuzzy_lookup.get(country, {})

    if not choices or not name:
        return []

    matches = process.extract(
        name,
        list(choices.keys()),
        scorer=fuzz.ratio,
        limit=TOP_K,
        score_cutoff=FUZZY_THRESHOLD,
    )

    for matched_name, score, _ in matches:

        for record in choices[matched_name]:

            candidate_id = (
                record["unlocode"]
                if source == "UNLOCODE"
                else record["code"]
            )

            candidate_name = (
                record["unlocode_name"]
                if source == "UNLOCODE"
                else record["name"]
            )

            candidates.append({
                "candidate_source": source,
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "similarity_score": round(float(score), 2),
                "match_method": "FUZZY_REVIEW",
                "source_record": record,
            })

    return candidates


# =========================================================
# 6. CANONICAL REGISTRY AND CANDIDATE TABLE
# =========================================================

def build_registry(portwatch, unlo, upply):

    print("\nBuilding multi-value matching indexes...")

    unlo_exact_lookup, unlo_fuzzy_lookup, unlo_ids = (
    build_unlo_indexes(unlo)
    )

    upply_exact_lookup, upply_fuzzy_lookup, upply_ids = (
        build_upply_indexes(upply)
    )

    unlo_id_set = set(unlo_ids)
    candidates_output = []
    registry_output = []

    for port in portwatch.to_dict("records"):

        unlo_candidates = get_candidates(
            port,
            unlo_exact_lookup,
            unlo_fuzzy_lookup,
            "UNLOCODE",
        )

        upply_candidates = get_candidates(
            port,
            upply_exact_lookup,
            upply_fuzzy_lookup,
            "UPPLY",
        )

        # Preserve cross-source identifier evidence even when the
        # Upply country conflicts with the PortWatch country.
        # Such records are evidence only and must never be auto-accepted.
        existing_upply_ids = {
            c["candidate_id"]
            for c in upply_candidates
        }

        for unlo_candidate in unlo_candidates:
            if (
                unlo_candidate["match_method"]
                != "EXACT_NORMALIZED_NAME_COUNTRY"
            ):
                continue

            candidate_id = unlo_candidate["candidate_id"]
            up_record = upply_ids.get(candidate_id)

            if not up_record or candidate_id in existing_upply_ids:
                continue

            up_country = up_record.get("country_code")

            if (
                up_country
                and up_country != port["portwatch_country_iso2"]
            ):
                upply_candidates.append({
                    "candidate_source": "UPPLY",
                    "candidate_id": candidate_id,
                    "candidate_name": up_record["name"],
                    "similarity_score": None,
                    "match_method": "IDENTIFIER_BRIDGE_COUNTRY_CONFLICT",
                    "source_record": up_record,
                })

                existing_upply_ids.add(candidate_id)

        def distinct_ids(candidates):
            return sorted({
                c["candidate_id"]
                for c in candidates
                if c["candidate_id"]
            })

        unlo_candidate_ids = distinct_ids(unlo_candidates)
        upply_candidate_ids = distinct_ids(upply_candidates)

        # A fuzzy candidate is never automatically accepted.
        unlo_exact = bool(
            unlo_candidates
            and all(
                c["match_method"]
                == "EXACT_NORMALIZED_NAME_COUNTRY"
                for c in unlo_candidates
            )
        )

        upply_exact = bool(
            upply_candidates
            and all(
                c["match_method"]
                == "EXACT_NORMALIZED_NAME_COUNTRY"
                for c in upply_candidates
            )
        )

        accepted_unlo = (
            unlo_candidate_ids[0]
            if unlo_exact and len(unlo_candidate_ids) == 1
            else None
        )

        accepted_upply = (
            upply_candidate_ids[0]
            if upply_exact and len(upply_candidate_ids) == 1
            else None
        )

        # If an unambiguous UN/LOCODE identity exists,
        # the exact same Upply code can be linked by identifier.
        if accepted_unlo and accepted_unlo in upply_ids:

            if accepted_upply is None:
                accepted_upply = accepted_unlo

        # Conversely, an accepted Upply identifier can identify
        # its exact UN/LOCODE code, but only if that identifier
        # exists in the UN/LOCODE source.
        if accepted_upply and accepted_upply in unlo_id_set:

            if accepted_unlo is None:
                accepted_unlo = accepted_upply

        # Validate each accepted source association independently.
        country_conflict = False

        if accepted_upply and accepted_upply in upply_ids:
            up_record = upply_ids[accepted_upply]
            up_country = up_record.get("country_code")

            if (
                up_country
                and up_country != port["portwatch_country_iso2"]
            ):
                country_conflict = True
                accepted_upply = None

        if accepted_unlo:
            unlo_records = unlo_ids[accepted_unlo]

            unlo_countries = {
                r["unlocode_country_iso2"]
                for r in unlo_records
            }

            if unlo_countries != {port["portwatch_country_iso2"]}:
                country_conflict = True
                accepted_unlo = None

        if country_conflict:
            overall_status = "COUNTRY_CONFLICT"
        elif len(unlo_candidate_ids) > 1 or len(upply_candidate_ids) > 1:
            overall_status = "AMBIGUOUS"
        elif accepted_unlo or accepted_upply:
            overall_status = "EXACT_MATCH"
        elif unlo_candidates or upply_candidates:
            overall_status = "FUZZY_CANDIDATE_ONLY"
        else:
            overall_status = "UNMATCHED"

        # Preserve source reference details.
        unlo_names = []
        unlo_coordinates = []

        if accepted_unlo:
            for record in unlo_ids[accepted_unlo]:
                unlo_names.extend([
                    record["unlocode_name"],
                    record["unlocode_name_ascii"],
                ])
                unlo_coordinates.append(
                    record["unlocode_coordinates_raw"]
                )

        up_record = (
            upply_ids.get(accepted_upply)
            if accepted_upply
            else None
        )

        registry_output.append({
            "portwatch_id": port["portid"],
            "portwatch_name": port["portname"],
            "portwatch_country": port["country"],
            "portwatch_iso3": port["ISO3"],
            "country_iso2": port["portwatch_country_iso2"],
            "portwatch_name_normalized": port["portwatch_name_norm"],

            "unlocode": accepted_unlo,
            "unlocode_match_status": (
                "UNIQUE_EXACT"
                if accepted_unlo
                else "AMBIGUOUS_OR_UNMATCHED"
            ),
            "unlocode_candidate_count": len(unlo_candidate_ids),
            "unlocode_source_names": " | ".join(
                unique_values(unlo_names)
            ),
            "unlocode_coordinate_strings": " | ".join(
                unique_values(unlo_coordinates)
            ),

            "upply_code": accepted_upply,
            "upply_match_status": (
                "UNIQUE_EXACT_OR_IDENTIFIER_LINK"
                if accepted_upply
                else "AMBIGUOUS_OR_UNMATCHED"
            ),
            "upply_candidate_count": len(upply_candidate_ids),
            "upply_name": (
                up_record["name"]
                if up_record else None
            ),
            "upply_latitude": (
                up_record["latitude"]
                if up_record else None
            ),
            "upply_longitude": (
                up_record["longitude"]
                if up_record else None
            ),
            "upply_country_code": (
                up_record["country_code"]
                if up_record else None
            ),
            "country_conflict": country_conflict,
            "identity_status": overall_status,
        })

        # Keep every candidate row.
        for source_candidates in (
            unlo_candidates,
            upply_candidates,
        ):

            for candidate in source_candidates:

                record = candidate["source_record"]

                candidates_output.append({
                    "portwatch_id": port["portid"],
                    "portwatch_name": port["portname"],
                    "portwatch_country": port["country"],
                    "portwatch_iso3": port["ISO3"],
                    "candidate_source": candidate["candidate_source"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_name": candidate["candidate_name"],
                    "similarity_score": candidate["similarity_score"],
                    "match_method": candidate["match_method"],
                    "candidate_country": (
                        record.get("unlocode_country_iso2")
                        if candidate["candidate_source"] == "UNLOCODE"
                        else record.get("country_code")
                    ),
                    "source_file": record.get(
                        "unlocode_source_file"
                    ),
                    "source_row": record.get(
                        "unlocode_source_row"
                    ),
                    "candidate_is_accepted": (
                        candidate["candidate_id"] == accepted_unlo
                        if candidate["candidate_source"] == "UNLOCODE"
                        else candidate["candidate_id"] == accepted_upply
                    ),
                })

    registry = pd.DataFrame(registry_output)
    candidates = pd.DataFrame(candidates_output)

    return registry, candidates


# =========================================================
# 7. UPPLY <-> UN/LOCODE IDENTIFIER RECONCILIATION
# =========================================================

def reconcile_sources(unlo, upply, registry):

    unlo_ids = set(unlo["unlocode"])
    upply_ids = set(upply["code"])

    intersection = unlo_ids & upply_ids

    unlo_duplicates = int(
        unlo.duplicated("unlocode", keep=False).sum()
    )

    duplicate_groups = int(
        unlo.loc[
            unlo.duplicated("unlocode", keep=False),
            "unlocode",
        ].nunique()
    )

    disagreements = []

    for code in sorted(intersection):

        up_rows = upply[upply["code"] == code]
        un_rows = unlo[unlo["unlocode"] == code]

        up_countries = set(
            up_rows["country_code"].dropna()
        )

        un_countries = set(
            un_rows["unlocode_country_iso2"].dropna()
        )

        if up_countries and un_countries and up_countries != un_countries:
            disagreements.append(code)

    rows = [
        ("PortWatch", len(registry), "Primary registry spine"),
        ("UN/LOCODE", len(unlo), "All source records preserved"),
        ("UN/LOCODE unique identifiers", len(unlo_ids), "Identifier cardinality"),
        ("UN/LOCODE duplicate rows", unlo_duplicates, "Preserved, not deduplicated"),
        ("UN/LOCODE duplicate identifier groups", duplicate_groups, "Requires provenance review"),
        ("Upply", len(upply), "All source records preserved"),
        ("Upply unique identifiers", len(upply_ids), "Identifier cardinality"),
        ("Upply-UN/LOCODE exact identifier intersection", len(intersection), "Exact code overlap"),
        ("Upply unmatched identifiers", len(upply_ids - unlo_ids), "Not automatically invalid"),
        ("Upply-UN/LOCODE country disagreements", len(disagreements), "Explicit conflicts"),
        ("PortWatch unique accepted UN/LOCODE", int(registry["unlocode"].notna().sum()), "Conservative associations"),
        ("PortWatch unique accepted Upply", int(registry["upply_code"].notna().sum()), "Conservative associations"),
        ("PortWatch ambiguous", int((registry["identity_status"] == "AMBIGUOUS").sum()), "Manual review"),
        ("PortWatch fuzzy-only", int((registry["identity_status"] == "FUZZY_CANDIDATE_ONLY").sum()), "Manual review"),
        ("PortWatch unmatched", int((registry["identity_status"] == "UNMATCHED").sum()), "No candidate"),
        ("PortWatch country conflicts", int(registry["country_conflict"].sum()), "Manual review"),
    ]

    return pd.DataFrame(
        rows,
        columns=["metric", "value", "interpretation"],
    ), disagreements


# =========================================================
# 8. MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Avoid accidental overwrites.
    existing = [
        name for name in OUTPUT_FILES
        if (OUTPUT_DIR / name).exists()
    ]

    if existing:
        raise FileExistsError(
            "Registry outputs already exist. "
            "Move or version them before rerunning:\n"
            + "\n".join(existing)
        )

    portwatch = load_portwatch()
    unlo = load_unlocode()
    upply = load_upply()

    registry, candidates = build_registry(
        portwatch,
        unlo,
        upply,
    )

    reconciliation, disagreements = reconcile_sources(
        unlo,
        upply,
        registry,
    )

    # Unresolved records: every port without a fully accepted,
    # conflict-free identity association.
    unresolved = registry[
        registry["identity_status"] != "EXACT_MATCH"
    ].copy()

    # Save all artifacts.
    registry.to_csv(
        OUTPUT_DIR / "canonical_port_registry.csv",
        index=False,
    )

    candidates.to_csv(
        OUTPUT_DIR / "port_identity_candidates.csv",
        index=False,
    )

    unresolved.to_csv(
        OUTPUT_DIR / "unresolved_port_matches.csv",
        index=False,
    )

    reconciliation.to_csv(
        OUTPUT_DIR / "source_reconciliation.csv",
        index=False,
    )

    metadata = {
        "dataset": "Global PortWatch-spined canonical port registry",
        "version": "1.2",
        "status": "IDENTIFIER_BRIDGE_EVIDENCE_PRESERVED_PENDING_REVIEW",
        "primary_source": "IMF PortWatch",
        "portwatch_records": len(registry),
        "unlocode_source_records": len(unlo),
        "unlocode_unique_identifiers": int(
            unlo["unlocode"].nunique()
        ),
        "upply_source_records": len(upply),
        "upply_unique_identifiers": int(
            upply["code"].nunique()
        ),
        "candidate_rows": len(candidates),
        "unresolved_port_records": len(unresolved),
        "fuzzy_matches_automatically_accepted": False,
        "duplicate_unlocode_records_preserved": True,
        "unmatched_source_records_discarded": False,
        "country_disagreement_codes": disagreements,
        "matching_policy": {
            "primary_spine": "PortWatch port IDs",
            "exact_name_country": "Accepted only if one distinct identifier",
            "exact_identifier_bridge": "Upply code equals UN/LOCODE",
            "fuzzy_threshold": FUZZY_THRESHOLD,
            "fuzzy_top_k": TOP_K,
            "fuzzy_auto_merge": False,
            "ambiguous_candidates_preserved": True,
        },
        "limitations": [
            "PortWatch port universe is limited to its 2,065 records.",
            "A PortWatch port is not necessarily a single terminal or berth.",
            "Name and country matching does not prove identical geographic boundaries.",
            "Upply coordinates are reference points, not terminal or berth coordinates.",
            "UN/LOCODE duplicate identifiers and name collisions are preserved.",
            "Fuzzy matches require manual review.",
            "This registry does not establish maritime route connectivity.",
        ],
        "outputs": OUTPUT_FILES,
    }

    (OUTPUT_DIR / "registry_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("CANONICAL PORT REGISTRY GENERATED")
    print("=" * 65)

    print("\nRegistry shape:", registry.shape)
    print("Candidate rows:", len(candidates))
    print("Unresolved ports:", len(unresolved))

    print("\nIdentity status:")
    print(
        registry["identity_status"]
        .value_counts()
        .to_string()
    )

    print("\nAccepted UN/LOCODE associations:",
          registry["unlocode"].notna().sum())

    print("Accepted Upply associations:",
          registry["upply_code"].notna().sum())

    print("\nSource reconciliation:")
    print(reconciliation.to_string(index=False))

    print("\nOutputs:")
    for name in OUTPUT_FILES:
        path = OUTPUT_DIR / name
        print(
            name,
            "|",
            round(path.stat().st_size / 1024, 1),
            "KB",
        )

    print("\nRaw source files modified: NO")
    print("ERA5 frozen artifacts modified: NO")
    print("Registry status: INITIAL / PENDING REVIEW")


if __name__ == "__main__":
    main()
