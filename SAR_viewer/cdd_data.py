"""
CDD Vault download + local cache for the SAR app.

Port of Export_HiBiT_Jess_SMILES.ipynb: connect to the vault, find the HiBiT and
Jess protocols, pull their protocol data, join molecule SMILES and the molecule
fields the SAR matrix needs (Target, Combination synonym, Target protein
synonym, E3 ligase synonym), and cache the result as a CSV next to this file.

The CSV in this directory is the source the app reads. A download happens only
when there is nothing usable on disk, or when it is explicitly asked for
(`force=True`, the app's "Refresh from CDD" button, `--force` here). Age never
triggers a download on its own; `cache_is_fresh()` only reports whether the copy
is less than MAX_CACHE_AGE old, so the status line can say so.

Environment:
    cddVaultId    numeric CDD Vault ID
    cddAPIToken   CDD API token with read access

CLI:
    python cdd_data.py            # download only if there is no usable copy
    python cdd_data.py --force    # download regardless of what is on disk
    python cdd_data.py --status   # just report the age of the local copy
"""

import datetime as dt
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Where the downloaded copy lives: next to the code unless SAR_DATA_DIR says
# otherwise (a mounted volume when deployed in a container).
DATA_DIR = Path(os.environ.get("SAR_DATA_DIR") or Path(__file__).resolve().parent)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_CSV = DATA_DIR / "cdd_sar_data.csv"
CACHE_META = DATA_DIR / "cdd_sar_data.meta.json"

# Not a download trigger: how old a copy may be before the status line calls it
# stale and suggests refreshing.
MAX_CACHE_AGE = dt.timedelta(days=1)
# After a failed attempt, wait this long before trying the vault again, so a
# vault that is down does not get hit on every callback or every restart.
RETRY_AFTER = dt.timedelta(hours=1)

# Case-insensitive patterns used to find the protocols in the vault.
ASSAY_PROTOCOL_PATTERNS = {
    "HiBiT": r"hibit|hi\s*bit",
    "Jess": r"jess",
}

# Optional project scoping, e.g. PROJECT_IDS = [12345]
PROJECT_IDS = []

# Molecule fields the SAR matrix is built from; a cache without them is refetched.
REQUIRED_COLUMNS = (
    "molecule_id",
    "Synonyms",
    "Target",
    "Combination synonyms",
    "Target protein synonym",
    "E3 ligase synonym",
    "Concentration",
)


# --------------------------------------------------------------------------
# notebook helpers
# --------------------------------------------------------------------------
def project_query_kwargs():
    if not PROJECT_IDS:
        return {}
    return {"projects": ",".join(str(pid) for pid in PROJECT_IDS)}


def normalize_records(records):
    """Flatten CDD JSON objects into a DataFrame, preserving every field."""
    if records is None:
        return pd.DataFrame()
    if isinstance(records, dict):
        records = records.get("objects", records)
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records, sep=".")


def first_existing_column(df, candidates):
    """First available column out of `candidates`, matched case-insensitively."""
    exact = {col: col for col in df.columns}
    lowered = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        match = lowered.get(str(candidate).lower())
        if match is not None:
            return match
    return None


def find_columns_containing(df, words):
    return [
        col for col in df.columns
        if all(w.lower() in str(col).lower() for w in words)
    ]


def readout_definition_map(protocol):
    """{readout_definition_id: readout_name} for one protocol row.

    Names are made unique: a protocol with several curve fits repeats the
    generic readouts (N, R squared, Minimum measured, ...) once per fit. The
    first definition keeps the plain name, later ones get " #2", " #3", ...
    """
    definitions = protocol.get("readout_definitions", [])
    if not isinstance(definitions, list):
        return {}
    mapping, seen = {}, {}
    for d in definitions:
        if not (isinstance(d, dict) and "id" in d and "name" in d):
            continue
        name = str(d["name"])
        seen[name] = seen.get(name, 0) + 1
        mapping[str(d["id"])] = name if seen[name] == 1 else f"{name} #{seen[name]}"
    return mapping


def rename_readout_columns(df, protocol):
    """readouts.<id>.<field> -> the readable readout name."""
    mapping = readout_definition_map(protocol)
    if not mapping:
        return df

    rename = {}
    for col in df.columns:
        match = re.fullmatch(r"readouts\.(\d+)\.(.+)", str(col))
        if not match:
            continue
        readout_id, attribute = match.groups()
        name = mapping.get(readout_id)
        if not name:
            continue
        rename[col] = name if attribute == "value" else f"{name} {attribute}"
    return df.rename(columns=rename)


def run_date_map(protocol):
    """{run_id: "YYYY-MM-DD"} from a protocol's run list (its "runs" field)."""
    runs = protocol.get("runs", [])
    if not isinstance(runs, list):
        return {}
    return {
        int(run["id"]): run.get("run_date")
        for run in runs
        if isinstance(run, dict) and "id" in run
    }


def add_molecule_id(df):
    """Normalise whichever molecule-id field this protocol returned."""
    if df.empty:
        return df

    molecule_col = first_existing_column(
        df,
        ["molecule_id", "molecule.id", "molecule.id.int", "molecule",
         "molecule.id.number", "Molecule ID", "Molecule Id"],
    )
    if molecule_col is None:
        candidates = find_columns_containing(df, ["molecule", "id"])
        molecule_col = candidates[0] if candidates else None

    if molecule_col is not None:
        df = df.copy()
        df["molecule_id"] = pd.to_numeric(df[molecule_col], errors="coerce").astype("Int64")
    return df


# --------------------------------------------------------------------------
# vault calls
# --------------------------------------------------------------------------
def connect():
    from cdd_python_sdk.VaultClient import VaultClient

    try:
        vault_id = int(os.environ["cddVaultId"])
        token = os.environ["cddAPIToken"]
    except KeyError as exc:
        raise RuntimeError(
            f"Environment variable {exc.args[0]} is not set; "
            "export cddVaultId and cddAPIToken before downloading."
        ) from exc
    return VaultClient(vault_id, token)


def build_smiles_lookup(vault):
    """Molecule table: id, SMILES and the molecule fields used by the matrix."""
    molecules_df = normalize_records(
        vault.getMolecules(
            asDataFrame=False,
            include_original_structures="true",
            **project_query_kwargs(),
        )
    )
    if molecules_df.empty:
        raise RuntimeError("No molecules were returned from CDD Vault.")

    id_col = first_existing_column(molecules_df, ["id", "molecule.id", "molecule_id"])
    if id_col is None:
        raise RuntimeError(
            f"Could not find a molecule ID column. Columns: {molecules_df.columns.tolist()}"
        )

    # Structure columns in preference order, then the molecule fields we need.
    wanted = [
        "smiles", "cxsmiles", "canonical_smiles", "structure.smiles",
        "molecule.smiles", "original_structure",
        "molecule_fields.Combination synonym",
        "molecule_fields.Target",
        "molecule_fields.Target protein synonym",
        "molecule_fields.E3 ligase synonym",
        "synonyms",
    ]
    keep = []
    for candidate in wanted:
        col = first_existing_column(molecules_df, [candidate])
        if col is not None and col not in keep:
            keep.append(col)
    for col in find_columns_containing(molecules_df, ["smiles"]):
        if col not in keep:
            keep.append(col)

    structure_cols = [c for c in keep if "smiles" in str(c).lower()
                      or c == "original_structure"]
    if not structure_cols:
        raise RuntimeError(
            f"Could not find a SMILES-like column. Columns: {molecules_df.columns.tolist()}"
        )

    name_col = first_existing_column(molecules_df, ["name", "molecule.name", "primary_name"])
    cols = [id_col] + keep + ([name_col] if name_col else [])
    lookup = molecules_df[list(dict.fromkeys(cols))].copy()

    lookup["molecule_smiles"] = lookup[structure_cols].bfill(axis=1).iloc[:, 0]
    lookup = lookup.rename(columns={id_col: "molecule_id",
                                    **({name_col: "molecule_name"} if name_col else {})})
    lookup["molecule_id"] = pd.to_numeric(lookup["molecule_id"], errors="coerce").astype("Int64")

    # A molecule with two synonyms is a building block: synonyms[0] is the
    # warhead code (A095), synonyms[1] the compound name. One synonym means the
    # code is absent (notebook cell 18).
    lookup["Synonyms"] = [
        syn[0] if isinstance(syn, (list, tuple)) and len(syn) > 1 else np.nan
        for syn in lookup.get("synonyms", pd.Series([None] * len(lookup)))
    ]
    return lookup.rename(
        columns={
            "molecule_fields.Combination synonym": "Combination synonyms",
            "molecule_fields.Target": "Target",
            "molecule_fields.Target protein synonym": "Target protein synonym",
            "molecule_fields.E3 ligase synonym": "E3 ligase synonym",
        }
    )


def get_matching_protocols(vault, label, pattern):
    protocols = vault.getProtocols(asDataFrame=True, **project_query_kwargs())
    if protocols.empty:
        return protocols

    name_col = first_existing_column(protocols, ["name", "protocol.name"])
    if name_col is None:
        raise RuntimeError(
            f"Could not find a protocol name column. Columns: {protocols.columns.tolist()}"
        )
    matches = protocols[
        protocols[name_col].astype(str).str.contains(pattern, case=False, regex=True, na=False)
    ].copy()
    matches["assay_group"] = label
    return matches


def fetch_protocol_data(vault, protocols, log=print):
    frames = []
    if protocols.empty:
        return pd.DataFrame()

    id_col = first_existing_column(protocols, ["id", "protocol.id"])
    name_col = first_existing_column(protocols, ["name", "protocol.name"])
    if id_col is None:
        raise RuntimeError(
            f"Could not find a protocol ID column. Columns: {protocols.columns.tolist()}"
        )

    for _, protocol in protocols.iterrows():
        protocol_id = int(protocol[id_col])
        protocol_name = protocol[name_col] if name_col is not None else str(protocol_id)
        log(f"Fetching {protocol['assay_group']}: {protocol_name} ({protocol_id})")

        df = normalize_records(
            vault.getProtocolData(
                id=protocol_id,
                asDataFrame=False,
                statusUpdates=False,
                **project_query_kwargs(),
            )
        )
        if df.empty:
            log(f"  no rows returned for protocol {protocol_id}")
            continue

        df = rename_readout_columns(df, protocol)
        df.insert(0, "assay_group", protocol["assay_group"])
        df.insert(1, "protocol_id", protocol_id)
        df.insert(2, "protocol_name", protocol_name)
        if "run" in df.columns:
            # The data rows carry only the run id; the date is on the protocol.
            dates = run_date_map(protocol)
            df.insert(3, "run_date", pd.to_numeric(df["run"], errors="coerce").map(
                lambda run: dates.get(int(run)) if pd.notna(run) else None))
        frames.append(add_molecule_id(df))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def fetch_from_cdd(log=print):
    """Download HiBiT + Jess protocol data joined to molecule structures/fields."""
    vault = connect()
    log(f"Connected to CDD Vault {vault.vaultNum}")

    protocol_frames = []
    for label, pattern in ASSAY_PROTOCOL_PATTERNS.items():
        matches = get_matching_protocols(vault, label, pattern)
        log(f"{label}: found {len(matches)} protocol(s)")
        if not matches.empty:
            protocol_frames.append(matches)
    if not protocol_frames:
        raise RuntimeError("No HiBiT or Jess protocols matched the configured patterns.")

    assay_data = fetch_protocol_data(
        vault, pd.concat(protocol_frames, ignore_index=True, sort=False), log=log
    )
    if assay_data.empty:
        raise RuntimeError("No protocol data rows were returned for the matched protocols.")
    if "molecule_id" not in assay_data.columns:
        raise RuntimeError(
            "Could not infer molecule_id from the protocol data. Columns: "
            f"{assay_data.columns.tolist()}"
        )

    lookup = build_smiles_lookup(vault)
    log(f"Molecules: {len(lookup)}; protocol rows: {len(assay_data)}")

    export = assay_data.merge(lookup, on="molecule_id", how="left")
    front = [
        "molecule_id", "molecule_name", "molecule_smiles", "assay_group",
        "protocol_id", "protocol_name", "run_date", "Synonyms", "Combination synonyms",
        "Target", "Target protein synonym", "E3 ligase synonym",
    ]
    front = [c for c in front if c in export.columns]
    export = export[front + [c for c in export.columns if c not in front]]
    log(f"Export rows: {len(export)}; without SMILES: "
        f"{int(export['molecule_smiles'].isna().sum())}")
    return export


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
def _read_meta():
    if not CACHE_META.exists():
        return {}
    try:
        return json.loads(CACHE_META.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_meta(**fields):
    meta = _read_meta()
    meta.update(fields)
    CACHE_META.write_text(json.dumps(meta, indent=2))


def _stamp(key):
    stamp = _read_meta().get(key)
    if not stamp:
        return None
    try:
        return dt.datetime.fromisoformat(stamp)
    except ValueError:
        return None


def downloaded_at():
    """When the cached CSV was pulled from CDD, or None if there is no cache.

    Falls back to the file's mtime when the sidecar is missing, so a CSV copied
    in by hand still counts as a download.
    """
    if not CACHE_CSV.exists():
        return None
    return _stamp("downloaded_at") or dt.datetime.fromtimestamp(CACHE_CSV.stat().st_mtime)


def cache_age():
    when = downloaded_at()
    return None if when is None else dt.datetime.now() - when


def _cache_is_usable():
    """True when the cached CSV exists and still has the columns we build on."""
    if not CACHE_CSV.exists():
        return False
    try:
        header = pd.read_csv(CACHE_CSV, nrows=0).columns
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    return all(col in header for col in REQUIRED_COLUMNS)


def cache_is_fresh():
    """True when the local copy exists and is less than MAX_CACHE_AGE old.

    Reporting only. An older copy is still used as it stands; refreshing it is
    the user's call.
    """
    if not _cache_is_usable():
        return False
    age = cache_age()
    return age is not None and age < MAX_CACHE_AGE


def download_needed():
    """True when there is nothing usable on disk and the back-off has expired.

    This is the only condition under which the app downloads on its own. The
    attempt time is kept in the sidecar, not just in memory, so restarting the
    app cannot turn a failing vault into a download on every start-up.
    """
    if _cache_is_usable():
        return False
    last = _stamp("last_attempt_at")
    return last is None or dt.datetime.now() - last >= RETRY_AFTER


def cache_status():
    """(is_fresh, human readable description) for the status bar."""
    if not _cache_is_usable():
        return False, "no usable local copy of the CDD data"
    when = downloaded_at()
    age = cache_age()
    hours = age.total_seconds() / 3600
    ago = f"{hours:.1f} h ago" if hours < 48 else f"{age.days} days ago"
    fresh = cache_is_fresh()
    hint = "" if fresh else " — over a day old; press “Refresh from CDD” for a fresh copy"
    return fresh, f"CDD data downloaded {when:%Y-%m-%d %H:%M} ({ago}){hint}"


def load_data(force=False, allow_download=True, log=print):
    """The CDD export: the local copy unless a download is asked for or needed.

    Whatever is on disk is used as it stands, however old. A download happens
    only on `force=True`, or when there is no usable copy to fall back on.

    `allow_download=False` guarantees no network call: it returns whatever is on
    disk (an empty frame if there is nothing usable), which is what every part of
    the app except the refresh path wants. A failed download falls back to the
    cached CSV when there is one, so the app stays usable offline.

    Returns (DataFrame, status string).
    """
    _, status = cache_status()
    usable = _cache_is_usable()

    if not force and usable:
        return read_cache(), status + " — using the local copy"
    if not allow_download:
        return pd.DataFrame(), status
    if not force and not download_needed():
        # Nothing usable on disk and an attempt failed recently: don't retry yet.
        return pd.DataFrame(), status

    reason = "forced refresh" if force else "no usable local copy"
    log(f"Downloading from CDD ({reason})…")
    # Recorded before the call so a crash mid-download still counts as an attempt.
    _write_meta(last_attempt_at=dt.datetime.now().isoformat(timespec="seconds"))
    try:
        df = fetch_from_cdd(log=log)
    except Exception as exc:                     # noqa: BLE001 - report, don't crash the app
        log(f"CDD download failed: {exc}")
        if usable:
            return read_cache(), f"{status} — CDD download failed ({exc}), using the local copy"
        raise
    save_cache(df)
    return df, cache_status()[1] + " — freshly downloaded"


def read_cache():
    return pd.read_csv(CACHE_CSV, low_memory=False)


def save_cache(df):
    """Write the export, via a temporary file so a reader never sees half of it."""
    tmp = CACHE_CSV.with_name(CACHE_CSV.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, CACHE_CSV)
    _write_meta(
        downloaded_at=dt.datetime.now().isoformat(timespec="seconds"),
        rows=int(len(df)),
        columns=int(df.shape[1]),
    )
    return CACHE_CSV


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="download even when there is a usable local copy")
    parser.add_argument("--status", action="store_true",
                        help="report the cache age and exit")
    args = parser.parse_args()

    if args.status:
        print(cache_status()[1] if CACHE_CSV.exists() else "no local copy yet")
        print(f"download needed: {download_needed()}")
    else:
        data, note = load_data(force=args.force)
        print(note)
        print(f"{len(data)} rows x {data.shape[1]} columns -> {CACHE_CSV}")
