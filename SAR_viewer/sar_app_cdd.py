"""
SAR matrix web service, reading straight from CDD Vault.

Same matrix as sar_app.py — row warheads x column warheads (any target on either axis),
median assay value per concentration, compounds drawn as structures — but the
data comes from the CDD Vault download implemented in cdd_data.py
(a port of Export_HiBiT_Jess_SMILES.ipynb) instead of an Excel export.

The page is always built from the CSV already in this directory. If
cdd_sar_data.csv is there, the app never contacts CDD on its own — whatever its
age, every table and figure comes off disk. A download happens in exactly two
cases: there is no usable copy in the directory, or the user asks for one with
"Refresh from CDD" (or --refresh). Either way it runs in the background, so the
page is populated from the local copy straight away and swaps over by itself
once the new data lands.

Run:
    ~/dash_env/bin/python sar_app_cdd.py
    ~/dash_env/bin/python sar_app_cdd.py --offline   # never contact CDD
    ~/dash_env/bin/python sar_app_cdd.py --refresh   # download whatever the age
then open http://127.0.0.1:8050

Needs cddVaultId / cddAPIToken in the environment for the download itself; with
a local copy present it also runs offline.
"""

import base64
import datetime as dt
import io
import math
import re
import threading
import warnings
from functools import lru_cache
from html import escape as html_escape
from pathlib import Path

# openpyxl MUST be imported before rdkit's drawing module: rdkit ships its own
# expat and importing rdMolDraw2D first makes openpyxl's XML parsing segfault.
import openpyxl  # noqa: F401  (import order matters, see above)
import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

import cdd_data

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
RDLogger.DisableLog("rdApp.*")

# Columns of the CDD export that identify a compound / combination.
MOL_COL = "molecule_name"
SYNONYM_COL = "Synonyms"
TARGET_COL = "Target"
COMBO_COL = "Combination synonyms"
POI_SYN_COL = "Target protein synonym"
E3_SYN_COL = "E3 ligase synonym"
ASSAY_COL = "assay_group"

# Assays are never pooled: the matrix always shows one, opening on this one.
DEFAULT_ASSAY = "HiBiT"
# Targets left out of the POI / E3 menus, and combinations using them.
EXCLUDED_TARGETS = frozenset({"CRBN"})
# Assays whose matrix cells show no dose-response curve on hover.
NO_CURVE_ASSAYS = frozenset({"HiBiT"})
# Assays limited to some readouts in the "Value shown" menu; others offer all
# they measured. Jess shows only the fitted DC50, with its degradation curve on
# hover.
ASSAY_READOUTS = {"Jess": ("DC50",)}
# Readouts where a lower value is the better compound: the colour scale opens
# on "low = strong" for these (the user can still flip it).
LOW_IS_STRONG = frozenset({"DC50", "Average DC50"})

# Structure columns, best first: mixtures carry no structure of their own, the
# single warheads we draw come from cxsmiles.
SMILES_CANDIDATES = ("cxsmiles", "smiles", "molecule_smiles", "original_structure")

# Readouts worth plotting, in menu order, with the label shown in the UI.
VALUE_COLUMNS = (
    ("Degradation (% DMSO)", "Degradation (% DMSO)"),
    ("Cell Viability (% DMSO)", "Cell Viability (% DMSO)"),
    ("DC50", "DC50 (µM)"),
    ("Minimum measured", "Dmax / minimum measured (%)"),
    ("Average DC50", "Average DC50 (µM)"),
    ("Average Dmax", "Average Dmax (%)"),
)

# Readouts reported once per dose-response run rather than per concentration.
PER_RUN_READOUTS = frozenset(
    {"DC50", "Minimum measured", "Maximum measured", "Average DC50",
     "Average Dmax", "Area under the curve"}
)

# Column header used where a readout has no concentration of its own.
PER_RUN_LABEL = "per run"

# Concentrations the matrix opens on (0.1 µM added to the notebook's 0.3 / 3).
DEFAULT_CONCENTRATIONS = (0.1, 0.3, 3.0)

POI_IMG = (260, 170)
E3_IMG = (240, 160)
# Sticky-offset geometry, derived from the image sizes so the frozen row/column
# line up with the rendered structure cells.
POI_COL_W = POI_IMG[0] + 16
E3_HEAD_H = E3_IMG[1] + 34

# How often the page asks the server whether the data has changed. The check is
# a dictionary lookup, not a download, so it can be frequent: it is what lets a
# background refresh appear without the user reloading the page.
CHECK_INTERVAL_MS = 60 * 1000


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------
_LOAD_LOCK = threading.Lock()
# `version` is bumped whenever `df` is replaced; the page polls it and only
# redraws when it moves, so a background download shows up on its own.
_CACHE = {"df": None, "status": "", "version": 0}
_DOWNLOAD_THREAD = None
# Cleared by --offline, so a machine without vault credentials never even tries.
ALLOW_DOWNLOAD = True


def harmonise_readouts(df):
    """Merge readout columns that differ only in the spacing of "(% DMSO)".

    The HiBiT protocol names its readouts "Degradation (%DMSO)", the Jess one
    "Degradation (% DMSO)". Left apart, HiBiT's values sit in a column the app
    never reads and every HiBiT matrix comes out empty.
    """
    canonical = {col: re.sub(r"\(%\s*", "(% ", str(col)) for col in df.columns}
    if all(col == name for col, name in canonical.items()):
        return df
    merged = {}
    for col, name in canonical.items():
        merged[name] = df[col] if name not in merged else merged[name].combine_first(df[col])
    return pd.DataFrame(merged, index=df.index)


def load_local():
    """Fill the in-memory copy from the CSV in this directory. Never downloads.

    This is the only thing the page ever waits for. Whatever is on disk is what
    the tables and figures are built from, so a slow or failing vault cannot
    delay — or blank — a page that has a perfectly good local copy.
    """
    with _LOAD_LOCK:
        if _CACHE["df"] is None:
            df, status = cdd_data.load_data(allow_download=False)
            _CACHE.update(df=harmonise_readouts(df), status=status,
                          version=_CACHE["version"] + 1)
        return _CACHE["df"]


def current_data():
    """The export every callback builds on: the local copy, read off disk once."""
    return load_local()


def _download(force):
    """Download and swap the result in. Runs on a background thread only."""
    try:
        df, status = cdd_data.load_data(force=force)
    except Exception as exc:                 # noqa: BLE001 - report, don't kill the app
        with _LOAD_LOCK:
            _CACHE["status"] = f"{cdd_data.cache_status()[1]} — CDD download failed ({exc})"
        return
    with _LOAD_LOCK:
        _CACHE.update(df=harmonise_readouts(df), status=status,
                      version=_CACHE["version"] + 1)


def start_download(force=False):
    """Start a CDD download in the background if one is warranted.

    Returns True if a download was started. Nothing waits for it: the page keeps
    showing the local copy and picks the new data up on the next tick. Without
    `force` — i.e. anything but the user asking — this starts nothing unless
    there is no usable copy in the directory (and no attempt failed in the last
    hour). An existing copy is used as it stands, however old it is.
    """
    global _DOWNLOAD_THREAD

    if not ALLOW_DOWNLOAD:
        return False
    with _LOAD_LOCK:
        if _DOWNLOAD_THREAD is not None and _DOWNLOAD_THREAD.is_alive():
            return False
        if not force and not cdd_data.download_needed():
            return False
        _DOWNLOAD_THREAD = threading.Thread(
            target=_download, args=(force,), daemon=True, name="cdd-download"
        )
        _DOWNLOAD_THREAD.start()
    return True


def download_running():
    return _DOWNLOAD_THREAD is not None and _DOWNLOAD_THREAD.is_alive()


def data_status():
    status = _CACHE["status"] or cdd_data.cache_status()[1]
    if download_running():
        status += " — downloading fresh data in the background"
    return status


def find_column(df, name):
    """Column `name`, tolerating the CDD Excel export's "<protocol>: " prefix."""
    if name in df.columns:
        return name
    for col in df.columns:
        text = str(col)
        if text == name or text.endswith(f": {name}") or text.startswith(f"{name} ("):
            return col
    for col in df.columns:
        if str(col).endswith(name):
            return col
    return None


def concentration_column(df):
    return (
        find_column(df, "Concentration")
        or find_column(df, "Concentration (μM)")
        or find_column(df, "Concentration (uM)")
    )


def smiles_column(df):
    for name in SMILES_CANDIDATES:
        col = find_column(df, name)
        if col is not None:
            return col
    return None


def value_columns(df):
    """[(column, label)] for the readouts with at least one value in `df`.

    Filtered on values, not just column names: the export has one column set
    for all assays, and HiBiT, for one, has no DC50 / Dmax fits at all.
    """
    out = []
    for name, label in VALUE_COLUMNS:
        col = find_column(df, name)
        if col is not None and df[col].notna().any() and col not in {c for c, _ in out}:
            out.append((col, label))
    return out


def readout_name(col):
    """The bare readout name behind a column, prefix or no prefix."""
    return str(col).split(": ")[-1]


def is_per_run_column(col):
    """True for readouts reported once per dose-response run (DC50, Dmax).

    These have no concentration of their own, so sorting on them ignores the
    "at (µM)" selection.
    """
    if not col:
        return False
    name = readout_name(col)
    return any(name.startswith(r) for r in PER_RUN_READOUTS)


def to_numeric_qualified(series):
    """Numbers out of a column that may hold '< 0.100' or '(not calculated)'.

    Censored values ('< 0.100') are taken at their bound so they still sort;
    free text becomes NaN.
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[<>~≈\s]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


# --------------------------------------------------------------------------
# reshaping
# --------------------------------------------------------------------------
def filter_assay(df, assay):
    """Rows of one assay. Assays are never mixed, so no assay means no rows."""
    if ASSAY_COL not in df.columns:
        return df
    return df[df[ASSAY_COL] == assay]


def building_blocks():
    """Single-warhead rows from the whole export, one per synonym.

    Taken across assays on purpose: which target a warhead hits and what it
    looks like do not depend on the assay a combination was measured in.
    """
    df = current_data()
    return df.dropna(subset=[SYNONYM_COL, TARGET_COL]).drop_duplicates(
        subset=[SYNONYM_COL]
    )


def building_block_smiles(targets):
    """Synonym -> SMILES lookup for the single-warhead rows (notebook: df_synonyms)."""
    blocks = building_blocks()
    smiles_col = smiles_column(blocks)
    if smiles_col is None:
        return {}
    blocks = blocks[blocks[TARGET_COL].isin(targets)]
    return dict(zip(blocks[SYNONYM_COL], blocks[smiles_col]))


def target_combinations(df, row_target, col_target):
    """Combinations pairing a `row_target` warhead with a `col_target` one.

    Either target can go on either axis. From here on the "POI" column holds
    the row warhead and "E3" the column warhead, so with VHL on the rows and
    FAK on the columns the two are swapped relative to the CDD fields.

    Warheads with no building-block record have no known target and are left
    out, since there is no telling which ligase they recruit.
    """
    combos = combinations_table(df)
    if combos.empty:
        return combos
    blocks = building_blocks()
    target_of = dict(zip(blocks[SYNONYM_COL], blocks[TARGET_COL]))
    poi_t, e3_t = combos["POI"].map(target_of), combos["E3"].map(target_of)

    as_is = combos[(poi_t == row_target) & (e3_t == col_target)]
    if row_target == col_target:
        return as_is
    swapped = combos[(poi_t == col_target) & (e3_t == row_target)].rename(
        columns={"POI": "E3", "E3": "POI"}
    )
    return pd.concat([as_is, swapped])


ALL_RUNS = "all"
RECENT_RUNS = "recent"


def scoped_data(assay, poi_target, e3_target, scope):
    """The assay's rows, cut down to each pair's most recent runs if asked.

    "Most recent" is per row x column pair: the rows of its latest run date,
    all runs of that day (the plates of one experiment usually share it).
    Everything built from the result — values, per-run lists, sort, curves,
    download — then follows the same selection.
    """
    df = filter_assay(current_data(), assay)
    if scope != RECENT_RUNS or "run_date" not in df.columns:
        return df
    combos = target_combinations(df, poi_target, e3_target)
    if combos.empty:
        return combos
    dates = combos["run_date"].astype(str).where(combos["run_date"].notna(), "")
    latest = dates.groupby([combos["POI"], combos["E3"]]).transform("max")
    return df.loc[combos.index[(dates == latest).to_numpy()]]


def combinations_table(df):
    """Bifunctional rows with POI / E3 synonym stems (notebook: df_combination).

    'A231-002' / 'A095-001' name a batch of a warhead; the stem before the dash
    is the warhead itself, which is what the matrix is indexed on.
    """
    needed = [COMBO_COL, POI_SYN_COL, E3_SYN_COL]
    if any(col not in df.columns for col in needed):
        # No data yet (first start, download still running): nothing to show.
        return pd.DataFrame(columns=list(dict.fromkeys([*df.columns, *needed, "POI", "E3"])))
    combos = df.dropna(subset=needed).copy()
    combos["POI"] = combos[POI_SYN_COL].astype(str).str.split("-").str[0]
    combos["E3"] = combos[E3_SYN_COL].astype(str).str.split("-").str[0]
    return combos


def experiment_runs(combos):
    """Tag each row with the dose-response run it came from.

    The CDD export gives a run id and a batch id; one curve is one batch within
    one run. Without those columns fall back to consecutive blocks of rows for a
    molecule sharing the same fit, which is how the Excel export has to be read.
    """
    if combos.empty:
        return pd.Series(index=combos.index, dtype=str)
    keys = [c for c in ("run", "batch") if c in combos.columns]
    if keys:
        return combos[keys].astype(str).agg("|".join, axis=1)

    fit_cols = [
        c for c in (find_column(combos, "DC50"), find_column(combos, "Minimum measured"))
        if c is not None
    ]
    key = combos[fit_cols].astype(str)
    changed = (key != key.shift()).any(axis=1) | (
        combos[MOL_COL] != combos[MOL_COL].shift()
    )
    return changed.cumsum()


# Readouts that come from a degradation fit, where a positive Hill slope means
# CDD fitted the wrong side of the curve (see is_hook_fit).
HOOK_READOUTS = frozenset({"DC50"})


def is_hook_fit(rows):
    """True for runs whose CDD fit has a positive Hill slope.

    A degradation curve falls with dose, so its Hill slope is negative. A
    PROTAC with a hook effect recovers at high doses; when a run starts at a
    dose already past maximal degradation, CDD fits that recovery instead, and
    its "DC50" is where the signal comes back, not a potency (A313 + A367:
    2.3 and 4.8 µM from such runs against ~0.03-0.09 µM from full-range ones).
    """
    hill = find_column(rows, "Hill slope")
    if hill is None:
        return pd.Series(False, index=rows.index)
    return to_numeric_qualified(rows[hill]) > 0


def readout_rows(df, poi_target, e3_target, concentrations, value_col, keep_hook=False):
    """The measurements behind the matrix, one row each, tagged with their cell.

    `_v` is the value, `_col` the column within its E3 block (a concentration,
    or PER_RUN_LABEL). Both the cell medians and the per-run lists are built
    from these rows, so the two always agree.

    DC50s from hook-side fits (`_hook`) are dropped unless `keep_hook`, which
    the per-run lists use to show them, marked, without counting them.
    """
    combos = target_combinations(df, poi_target, e3_target)
    if combos.empty:
        return combos

    combos["_v"] = to_numeric_qualified(combos[value_col])
    combos["_conc"] = pd.to_numeric(combos[concentration_column(df)], errors="coerce")

    if is_per_run_column(value_col):
        # DC50 / Dmax belong to a whole dose-response curve and are repeated on
        # every concentration row of it: keep one row per curve, and give the
        # readout a single column per E3 instead of one per concentration.
        combos = combos.dropna(subset=["_v"])
        combos["_run"] = experiment_runs(combos)
        combos = combos.drop_duplicates(subset=["_run", "POI", "E3"])
        combos["_col"] = PER_RUN_LABEL
        combos["_hook"] = (is_hook_fit(combos) if readout_name(value_col) in HOOK_READOUTS
                           else False)
        if not keep_hook:
            combos = combos[~combos["_hook"]]
    else:
        combos = combos.dropna(subset=["_v", "_conc"])
        if concentrations:
            combos = combos[combos["_conc"].isin([float(c) for c in concentrations])]
        combos["_col"] = combos["_conc"]
        combos["_hook"] = False
    return combos


def build_pivot(df, poi_target, e3_target, concentrations, value_col, aggfunc="median"):
    """POI (rows) x E3 ligase / concentration (columns) matrix of median values.

    Returns (pivot, counts, poi_smiles, e3_smiles): `counts` has the same shape
    as `pivot` and holds the number of measurements behind each median; the
    SMILES maps are keyed by the synonym stems used as row / column labels.
    """
    smiles_map = building_block_smiles({poi_target, e3_target})
    combos = readout_rows(df, poi_target, e3_target, concentrations, value_col)
    if combos.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, {}

    agg = combos.groupby(["POI", "E3", "_col"], as_index=False).agg(
        _value=("_v", aggfunc), _n=("_v", "count")
    )

    pivot = agg.pivot_table(
        index="POI", columns=["E3", "_col"], values="_value", aggfunc="first"
    ).sort_index(axis=1)
    counts = agg.pivot_table(
        index="POI", columns=["E3", "_col"], values="_n", aggfunc="first"
    ).reindex(index=pivot.index, columns=pivot.columns)

    poi_smiles = {p: smiles_map.get(p) for p in pivot.index}
    e3_smiles = {e: smiles_map.get(e) for e in pivot.columns.get_level_values(0).unique()}
    return pivot, counts, poi_smiles, e3_smiles


def run_values(df, poi_target, e3_target, concentrations, value_col):
    """{(POI, E3, column): [(value, run_date, n, hook), …]} newest run first.

    One entry per run (per batch within a run): the mean of that run's
    replicates in the cell, or its DC50 / Dmax for a per-run readout. Runs on
    the same date keep CDD's run order, later runs first; runs without a date
    go last. Hook-side DC50s are listed with `hook` True; the median leaves
    them out.
    """
    combos = readout_rows(df, poi_target, e3_target, concentrations, value_col,
                          keep_hook=True)
    if combos.empty or "run" not in combos.columns:
        return {}
    combos["_date"] = (combos["run_date"].astype(str).where(combos["run_date"].notna(), "")
                       if "run_date" in combos.columns else "")
    combos["_run_id"] = pd.to_numeric(combos["run"], errors="coerce")
    runs = combos.groupby(["POI", "E3", "_col", "_run_id", "batch"], as_index=False).agg(
        value=("_v", "mean"), n=("_v", "count"), date=("_date", "first"),
        hook=("_hook", "any"),
    )
    runs = runs.sort_values(["date", "_run_id"], ascending=False)
    out = {}
    for poi, e3, col, value, n, date, hook in zip(
            runs["POI"], runs["E3"], runs["_col"], runs["value"], runs["n"],
            runs["date"], runs["hook"]):
        out.setdefault((poi, e3, col), []).append(
            (float(value), date or None, int(n), bool(hook)))
    return out


# --------------------------------------------------------------------------
# sorting
# --------------------------------------------------------------------------


def values_at(pivot, sort_conc):
    """POI x E3 table of the matrix values at `sort_conc` (empty if none).

    A per-run readout (DC50, Dmax) has a single column per E3 and no
    concentration, so its values are used whatever `sort_conc` says.
    """
    if pivot.empty or sort_conc is None:
        return pd.DataFrame()
    level = pivot.columns.get_level_values(1)
    if PER_RUN_LABEL in set(level):
        conc = PER_RUN_LABEL
    else:
        matches = [c for c in level.unique() if float(c) == float(sort_conc)]
        if not matches:
            return pd.DataFrame()
        conc = matches[0]
    return pivot.xs(conc, axis=1, level=1)


def sort_by_best_pairs(pivot, counts, sort_conc, sort_dir):
    """Order the matrix so the best pairs at `sort_conc` fill the top-left corner.

    Columns: E3 warheads in the order of their best pair, i.e. walk the pairs
    best-first and add each E3 the first time it turns up. The E3 of the single
    best pair is therefore the first column.

    Rows: POI warheads measured with the first E3, sorted on that column; then
    the POIs not placed yet that were measured with the second E3, sorted on
    that one; and so on. "Best" is the lowest value for low → high, the highest
    for high → low. Warheads with no value at `sort_conc` keep name order at the
    end.

    Returns (pivot, counts, sorted_on) with sorted_on = {POI: (E3, value)}, the
    cell each row was placed by.
    """
    values = values_at(pivot, sort_conc)
    if values.empty:
        return pivot, counts, {}
    ascending = sort_dir != "desc"

    best = values.min(axis=0) if ascending else values.max(axis=0)
    measured = list(best.dropna().sort_values(ascending=ascending, kind="stable").index)
    e3_labels = list(pivot.columns.get_level_values(0).unique())
    e3_order = measured + [e3 for e3 in e3_labels if e3 not in measured]

    poi_order, sorted_on = [], {}
    for e3 in measured:
        column = values.loc[~values.index.isin(poi_order), e3].dropna()
        for poi, value in column.sort_values(ascending=ascending, kind="stable").items():
            poi_order.append(poi)
            sorted_on[poi] = (e3, value)
    poi_order += [poi for poi in pivot.index if poi not in sorted_on]

    cols = [c for e3 in e3_order for c in pivot.columns if c[0] == e3]
    return (pivot.loc[poi_order, cols], counts.loc[poi_order, cols], sorted_on)


def sort_description(pivot, sort_conc, sort_dir, readout, poi_target, e3_target,
                     sorted_on):
    """One-line explanation of the ordering, for the status bar."""
    if sort_conc is None or not sorted_on:
        return "No values at this concentration to sort on; warheads in alphabetical order."
    at = "" if any(c[1] == PER_RUN_LABEL for c in pivot.columns) else f" at {float(sort_conc):g} µM"
    best = "lowest" if sort_dir != "desc" else "highest"
    poi = pivot.index[0]
    e3, value = sorted_on[poi]
    return (
        f"Sorted on {readout}{at}, {best} first. Top-left: {poi} + {e3} ({value:.1f}). "
        f"Each {e3_target} column is placed by its best pair; each {poi_target} row "
        f"sits under the first column it was measured in (shown under its name), "
        "sorted on that value. Warheads with no value there are listed last."
    )


# --------------------------------------------------------------------------
# structure rendering
# --------------------------------------------------------------------------
def smiles_to_svg_uri(smiles, width, height):
    """Draw a SMILES as an inline SVG data URI (no temp files, no cairo needed)."""
    svg = smiles_to_svg(smiles, width, height)
    if svg is None:
        return None
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


@lru_cache(maxsize=1024)
def smiles_to_svg(smiles, width, height):
    """Draw a SMILES as SVG text, or None if it cannot be parsed."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 1.4
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def structure_cell(name, smiles, size, css_class):
    img_uri = smiles_to_svg_uri(smiles, *size)
    body = (
        html.Img(src=img_uri, className="struct-img", title=smiles or "")
        if img_uri
        else html.Div("no structure", className="struct-missing")
    )
    return html.Div([body, html.Div(name, className="struct-label")], className=css_class)


# --------------------------------------------------------------------------
# dose-response curves (the hover pop-up)
# --------------------------------------------------------------------------
# Geometry of the little chart. The height includes the tick labels, the axis
# title and the caption, so the pop-up never grows a scrollbar of its own.
CURVE_W, CURVE_H = 420, 274
CURVE_L, CURVE_R, CURVE_T, CURVE_B = 46, 16, 42, 66
# Legend characters per line before it wraps onto a second one.
LEGEND_CHARS = 78

# One series, so one hue: the app's own accent. The band and the replicate dots
# are the same hue at lower opacity — spread, not a second identity.
CURVE_INK = "#0d7c80"
CURVE_GRID = "#e1e0d9"
CURVE_AXIS = "#c3c2b7"
CURVE_MUTED = "#6b7a85"
CURVE_TEXT = "#16232b"

# Below this many replicates quartiles are noise, so the band is the full
# min-max range; at or above it the interquartile range is drawn inside it.
IQR_MIN_N = 5

# How far the replicate strip sits to the right of its concentration.
DOT_OFFSET = 7.5

# The y axis never stretches past these (% of DMSO): one broken lane can read
# millions of percent, and scaling to it flattens everything else. Points
# beyond them sit on the edge as triangles.
Y_CAP, Y_FLOOR = 250.0, -50.0


# CDD's fit parameters, per run: (key, column name).
FIT_COLUMNS = (
    ("dc50", "DC50"),
    ("hill", "Hill slope"),
    ("bottom", "Baseline response"),
    ("top", "Maximum response"),
)


def four_pl(conc, dc50, hill, bottom, top):
    """CDD's 4-parameter logistic: `bottom` at high dose, `top` at low dose.

    Hill slopes are negative for a degradation curve, which is what makes the
    response fall from `top` to `bottom` as the concentration rises.
    """
    return bottom + (top - bottom) / (1 + 10 ** ((math.log10(dc50) - math.log10(conc)) * hill))


def curve_fits(df, row_target, col_target):
    """{(row, column): (dc50, hill, bottom, top, n_runs, r2)} from CDD's fits.

    One fit per dose-response run; a pair's curve takes the median of each
    parameter over its runs, so its DC50 is the median DC50 the matrix cell
    shows. Runs without a complete fit are left out; `r2` is None when CDD
    reports none.
    """
    cols = {key: find_column(df, name) for key, name in FIT_COLUMNS}
    if any(col is None for col in cols.values()):
        return {}
    combos = target_combinations(df, row_target, col_target)
    if combos.empty:
        return {}
    for key, col in cols.items():
        combos[f"_{key}"] = to_numeric_qualified(combos[col])
    r2_col = find_column(df, "R squared")
    combos["_r2"] = to_numeric_qualified(combos[r2_col]) if r2_col else float("nan")
    runs = combos.dropna(subset=[f"_{key}" for key in cols])
    # Only fits that fall with dose: a median over fits pointing opposite ways
    # describes none of them (hook-side fits, see is_hook_fit).
    runs = runs[(runs["_dc50"] > 0) & (runs["_hill"] < 0)]
    if runs.empty:
        return {}
    runs = runs.assign(_run=experiment_runs(runs)).drop_duplicates(subset=["_run", "POI", "E3"])

    fits = {}
    for (poi, e3), group in runs.groupby(["POI", "E3"]):
        med = group[["_dc50", "_hill", "_bottom", "_top", "_r2"]].median()
        fits[(poi, e3)] = (
            float(med["_dc50"]), float(med["_hill"]), float(med["_bottom"]),
            float(med["_top"]), int(len(group)),
            None if pd.isna(med["_r2"]) else float(med["_r2"]),
        )
    return fits


def curve_readout(df, value_col):
    """The readout the pop-up plots for `value_col`.

    DC50 and Dmax have no concentration axis of their own — they are fits over a
    degradation curve — so hovering one shows the curve it was fitted from.
    """
    if value_col and not is_per_run_column(value_col):
        return value_col
    return find_column(df, "Degradation (% DMSO)")


def curve_series(df, value_col, row_target, col_target):
    """{(row, column): ((conc, ((value, flagged), …)), …)} for every measurement.

    `flagged` is CDD's own outlier mark for that readout. Flagged points are
    drawn as open rings but are *not* dropped: the matrix medians include them,
    and a curve that quietly disagreed with the cell it popped out of would be
    worse than one that shows what was flagged.

    Deliberately not filtered by the concentration picker: the matrix shows the
    concentrations you chose, the curve shows the whole dose-response behind
    them. Tuples all the way down so the SVG builder can be memoised on it.
    """
    if not value_col or value_col not in df.columns:
        return {}
    combos = target_combinations(df, row_target, col_target)
    if combos.empty:
        return {}

    combos["_v"] = to_numeric_qualified(combos[value_col])
    combos["_conc"] = pd.to_numeric(combos[concentration_column(df)], errors="coerce")
    flag_col = find_column(df, f"{readout_name(value_col)} outlier")
    combos["_flag"] = (
        combos[flag_col].astype(str).str.lower().eq("true")
        if flag_col in combos.columns else False
    )
    combos = combos.dropna(subset=["_v", "_conc"])
    if combos.empty:
        return {}

    out = {}
    for (poi, e3), group in combos.groupby(["POI", "E3"], sort=False):
        points = tuple(
            (float(conc), tuple(sorted(
                (float(row["_v"]), bool(row["_flag"])) for _, row in sub.iterrows()
            )))
            for conc, sub in group.groupby("_conc")
        )
        if points:
            out[(poi, e3)] = points
    return out


def quantile(values, q):
    """Linear-interpolation quantile of a sorted tuple (no numpy round-trip)."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _nice_top(vmax):
    """Round the y axis up to a sensible top, always leaving room for 100%.

    25 % steps unless that would crowd the axis with more than seven lines —
    a curve that overshoots the control badly should not turn the gridlines
    into hatching.
    """
    top = max(105.0, vmax * 1.04)
    for step in (25.0, 50.0, 100.0):
        rounded = math.ceil(top / step) * step
        if rounded / step <= 7:
            return rounded, step
    return math.ceil(top / 100.0) * 100.0, 100.0


def _decade_ticks(lo, hi):
    """Powers of ten inside the concentration range; the ends if none fall in."""
    ticks = [10.0 ** k for k in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
             if lo <= 10.0 ** k <= hi]
    return ticks or sorted({lo, hi})


def _fmt_conc(conc):
    return f"{conc:g}"


@lru_cache(maxsize=512)
def dose_response_svg_uri(points, title, readout_label, fit=None):
    """Draw one dose-response curve as an inline SVG data URI.

    Median line through the per-concentration medians (the same statistic the
    matrix cells show), a band for the spread, and every replicate as its own
    dot — with n between 1 and a handful, the raw points are more honest than
    any summary of them.

    Points CDD flagged as outliers (its analysis excludes them) are drawn as
    open rings but take no part in the medians, the spread or the axis range.
    """
    if not points:
        return None

    def kept(values):
        return [v for v, flag in values if not flag]

    concs = [c for c, _ in points]
    every = [v for _, values in points for v in kept(values)] or [0.0, 100.0]
    lo_c, hi_c = min(concs), max(concs)
    fit_xs = []
    if fit:
        dc50, hill, bottom, top_resp, n_runs, r2 = fit
        # Keep an extrapolated DC50 on the chart if it is within a decade.
        if lo_c / 10 <= dc50 < lo_c:
            lo_c = dc50
        elif hi_c < dc50 <= hi_c * 10:
            hi_c = dc50
        if hi_c > lo_c:
            span = math.log10(hi_c) - math.log10(lo_c)
            fit_xs = [10 ** (math.log10(lo_c) + span * i / 80) for i in range(81)]
        fit_ys = [four_pl(x, dc50, hill, bottom, top_resp) for x in fit_xs]
        every = every + fit_ys
    top, step = _nice_top(min(max(every), Y_CAP))
    floor = min(0.0, math.floor(max(min(every), Y_FLOOR) / step) * step)

    x0, x1 = CURVE_L, CURVE_W - CURVE_R
    y0, y1 = CURVE_T, CURVE_H - CURVE_B

    def sx(conc):
        if hi_c == lo_c:
            return (x0 + x1) / 2
        lo, hi = math.log10(lo_c), math.log10(hi_c)
        pad = (hi - lo) * 0.06
        return x0 + (math.log10(conc) - lo + pad) / (hi - lo + 2 * pad) * (x1 - x0)

    def sy(value):
        value = min(max(value, floor), top)
        return y1 - (value - floor) / (top - floor) * (y1 - y0)

    def off_scale(value):
        return value > top or value < floor

    e = html_escape
    ns = [len(kept(values)) for _, values in points]
    flagged_n = sum(1 for _, values in points for _, flag in values if flag)
    n_text = f"n {min(ns)}" if min(ns) == max(ns) else f"n {min(ns)}–{max(ns)}"

    if fit_xs:
        r2_text = "" if r2 is None else f" · R² {r2:.2f}"
        subtitle = (f"DC50 {dc50:.3g} µM · Hill {hill:.2f}{r2_text} · "
                    f"{n_runs} run{'s' if n_runs != 1 else ''} · {n_text} per point")
    else:
        subtitle = f"{readout_label} · {len(points)} concentrations · {n_text} per point"
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CURVE_W}" height="{CURVE_H}" '
        f'viewBox="0 0 {CURVE_W} {CURVE_H}" font-family="Inter, Segoe UI, system-ui, sans-serif">',
        f'<rect width="{CURVE_W}" height="{CURVE_H}" fill="#ffffff"/>',
        f'<text x="{CURVE_L}" y="18" font-size="12.5" font-weight="700" '
        f'fill="{CURVE_TEXT}">{e(title)}</text>',
        f'<text x="{CURVE_L}" y="31" font-size="10" fill="{CURVE_MUTED}">{e(subtitle)}</text>',
    ]

    # recessive chrome: solid hairlines, one shade off the surface
    tick = floor
    while tick <= top + 1e-9:
        y = sy(tick)
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                   f'stroke="{CURVE_GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{x0 - 7}" y="{y + 3.5:.1f}" font-size="9.5" text-anchor="end" '
                   f'fill="{CURVE_MUTED}">{tick:g}</text>')
        tick += step

    # the DMSO control: 100% is "no degradation at all", worth a line of its own
    if floor < 100 < top:
        y = sy(100)
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                   f'stroke="{CURVE_AXIS}" stroke-width="1"/>')
        svg.append(f'<text x="{x1 - 2}" y="{y - 4:.1f}" font-size="8.5" text-anchor="end" '
                   f'fill="{CURVE_MUTED}">DMSO control</text>')

    svg.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" '
               f'stroke="{CURVE_AXIS}" stroke-width="1"/>')

    # x ticks: a mark at every concentration measured, labels only on decades
    for conc in concs:
        x = sx(conc)
        svg.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y1 + 3}" '
                   f'stroke="{CURVE_AXIS}" stroke-width="1"/>')
    for conc in _decade_ticks(lo_c, hi_c):
        svg.append(f'<text x="{sx(conc):.1f}" y="{y1 + 15}" font-size="9.5" '
                   f'text-anchor="middle" fill="{CURVE_MUTED}">{_fmt_conc(conc)}</text>')

    svg.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y1 + 29}" font-size="10" '
               f'text-anchor="middle" fill="{CURVE_MUTED}">Concentration (µM)</text>')
    svg.append(f'<text transform="translate(13,{(y0 + y1) / 2:.1f}) rotate(-90)" '
               f'font-size="10" text-anchor="middle" fill="{CURVE_MUTED}">% of DMSO</text>')

    # spread first, so the median line and its markers sit on top of it
    spread_n = 0
    for conc, values in points:
        plain = kept(values)
        if len(plain) < 2:
            continue
        spread_n = max(spread_n, len(plain))
        x = sx(conc)
        svg.append(f'<line x1="{x:.1f}" y1="{sy(plain[0]):.1f}" x2="{x:.1f}" '
                   f'y2="{sy(plain[-1]):.1f}" stroke="{CURVE_INK}" stroke-width="2" '
                   f'stroke-opacity="0.45" stroke-linecap="round"/>')
        if len(plain) >= IQR_MIN_N:
            q1, q3 = quantile(plain, 0.25), quantile(plain, 0.75)
            svg.append(f'<rect x="{x - 4.5:.1f}" y="{sy(q3):.1f}" width="9" '
                       f'height="{max(2.0, sy(q1) - sy(q3)):.1f}" rx="2" '
                       f'fill="{CURVE_INK}" fill-opacity="0.28"/>')

    off_n = 0
    # Replicates as a strip set beside the range bar rather than on top of it:
    # centred, they hide under the median marker and the IQR box exactly where
    # the reader is looking.
    for conc, values in points:
        if len(values) < 2 and not any(flag for _, flag in values):
            continue
        x = sx(conc) + DOT_OFFSET
        for i, (value, flag) in enumerate(values):
            dx = (i % 3 - 1) * 2.0
            if off_scale(value):
                off_n += 1
                tip, base = (sy(top), sy(top) + 6) if value > top else (sy(floor), sy(floor) - 6)
                svg.append(f'<path d="M{x + dx:.1f},{tip:.1f} L{x + dx - 3.5:.1f},{base:.1f} '
                           f'L{x + dx + 3.5:.1f},{base:.1f} Z" '
                           + (f'fill="none" stroke="{CURVE_INK}" stroke-width="1.2"/>' if flag
                              else f'fill="{CURVE_INK}" fill-opacity="0.7"/>'))
            elif flag:
                svg.append(f'<circle cx="{x + dx:.1f}" cy="{sy(value):.1f}" r="3" '
                           f'fill="none" stroke="{CURVE_INK}" stroke-width="1.4"/>')
            else:
                svg.append(f'<circle cx="{x + dx:.1f}" cy="{sy(value):.1f}" r="1.8" '
                           f'fill="{CURVE_INK}" fill-opacity="0.55"/>')

    medians = [(sx(conc), sy(quantile(sorted(kept(values)), 0.5)))
               for conc, values in points if kept(values)]
    if fit_xs:
        # The fitted curve replaces the dot-to-dot line: it is what CDD's DC50
        # was read from.
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(fit_xs, fit_ys))
        svg.append(f'<polyline points="{path}" fill="none" stroke="{CURVE_INK}" '
                   f'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>')
        if lo_c <= dc50 <= hi_c:
            xd, yd = sx(dc50), sy(four_pl(dc50, dc50, hill, bottom, top_resp))
            svg.append(f'<line x1="{xd:.1f}" y1="{y1}" x2="{xd:.1f}" y2="{yd:.1f}" '
                       f'stroke="{CURVE_TEXT}" stroke-width="1" stroke-dasharray="3 3"/>')
            svg.append(f'<circle cx="{xd:.1f}" cy="{yd:.1f}" r="3.5" fill="#ffffff" '
                       f'stroke="{CURVE_TEXT}" stroke-width="1.5"/>')
            # Just above the axis, left of the drop line: under a falling
            # curve's high plateau, the one corner the data leaves empty.
            svg.append(f'<text x="{xd - 5:.1f}" y="{y1 - 6:.1f}" font-size="10" '
                       f'font-weight="600" text-anchor="end" fill="{CURVE_TEXT}">'
                       f'DC50 {dc50:.3g} µM</text>')
    elif len(medians) > 1:
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in medians)
        svg.append(f'<polyline points="{path}" fill="none" stroke="{CURVE_INK}" '
                   f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for x, y in medians:
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{CURVE_INK}" '
                   f'stroke="#ffffff" stroke-width="1.5"/>')

    parts = ["line: CDD fit (median parameters)", "● median"] if fit_xs else ["● median"]
    if spread_n:
        parts.append("bar: range")
        if spread_n >= IQR_MIN_N:
            parts.append("box: IQR")
        parts.append("dots: replicates")
    if flagged_n:
        parts.append("○ CDD outlier, not counted")
    if off_n:
        parts.append("▲ off scale")
    lines = [[]]
    for part in parts:
        if lines[-1] and len(" · ".join(lines[-1] + [part])) > LEGEND_CHARS:
            lines.append([])
        lines[-1].append(part)
    for i, line in enumerate(reversed(lines)):
        svg.append(f'<text x="8" y="{CURVE_H - 6 - 11 * i}" font-size="8.5" '
                   f'fill="{CURVE_MUTED}">{e(" · ".join(line))}</text>')
    svg.append("</svg>")

    return "data:image/svg+xml;base64," + base64.b64encode("".join(svg).encode()).decode()


def curve_key(poi, e3):
    """Stable id tying a matrix cell to its curve in the hidden gallery."""
    return f"{poi}::{e3}"


def curve_gallery(df, pivot, value_col, row_target, col_target):
    """Curve image per POI x E3 pair on show, as {key: [image]}.

    Each image is {"src", "title", "caption"}. Where CDD fitted the runs (Jess)
    the curve is its 4-parameter fit, drawn over the measured points.

    Rendered once per pair rather than once per cell: a row block can be eleven
    concentrations wide, and eleven copies of the same chart is eleven times the
    markup for nothing.
    """
    if pivot.empty:
        return {}
    readout = curve_readout(df, value_col)
    if readout is None:
        return {}
    label = dict(value_columns(df)).get(readout, readout_name(readout))

    series = curve_series(df, readout, row_target, col_target)
    fits = curve_fits(df, row_target, col_target)
    e3_labels = list(pivot.columns.get_level_values(0).unique())
    gallery = {}
    for poi in pivot.index:
        for e3 in e3_labels:
            key = curve_key(poi, e3)
            points = series.get((poi, e3))
            if not points:
                continue
            uri = dose_response_svg_uri(points, f"{poi} × {e3}", label, fits.get((poi, e3)))
            if uri:
                gallery[key] = [{"src": uri, "title": "", "caption": ""}]
    return gallery


# --------------------------------------------------------------------------
# table rendering
# --------------------------------------------------------------------------
def value_style(value, vmin, vmax, invert):
    """Sequential shading of the numeric cells."""
    if pd.isna(value) or vmax is None or vmax == vmin:
        return {}
    frac = (value - vmin) / (vmax - vmin)
    if invert:
        frac = 1 - frac
    # pale -> saturated teal
    alpha = 0.08 + 0.72 * frac
    return {
        "backgroundColor": f"rgba(13, 124, 128, {alpha:.3f})",
        "color": "#08262b" if frac < 0.55 else "#ffffff",
        "fontWeight": "600" if frac > 0.66 else "500",
    }


def conc_label(conc):
    """Header for a column level that is a concentration, or PER_RUN_LABEL."""
    return f"{conc:g} µM" if isinstance(conc, (int, float)) else str(conc)


# Runs listed in a cell before the rest are folded into "+N older" (all of
# them are in the cell's tooltip).
MAX_RUNS_LISTED = 8


def run_list(runs, value_fmt):
    """The per-run values under a cell's median, newest first."""
    lines = [
        html.Div(
            [html.Span(value_fmt(value) + (" hook fit" if hook else "")),
             html.Span(date or "no date", className="run-date")],
            className="run-line hook" if hook else "run-line",
            title=("Positive Hill slope: CDD fitted the high-dose recovery (hook), "
                   "so this is not a potency. Left out of the median and the curve.")
            if hook else None,
        )
        for value, date, _, hook in runs[:MAX_RUNS_LISTED]
    ]
    if len(runs) > MAX_RUNS_LISTED:
        # The whole list on hover of this line only, so it does not compete
        # with the dose-response pop-up over the rest of the cell.
        lines.append(html.Div(f"+{len(runs) - MAX_RUNS_LISTED} older", className="run-more",
                              title=run_tooltip(runs, value_fmt)))
    return html.Div(lines, className="run-list")


def run_tooltip(runs, value_fmt):
    return "\n".join(
        f"{date or 'no date'}   {value_fmt(value)}" + (f"  (mean of {n})" if n > 1 else "")
        + ("  hook fit, not counted" if hook else "")
        for value, date, n, hook in runs
    )


def render_matrix(pivot, counts, poi_smiles, e3_smiles, value_label, invert,
                  gallery=None, sorted_on=None, runs=None):
    if pivot.empty:
        return html.Div(
            "No data for this combination of target / concentration / assay column.",
            className="empty",
        )

    numeric = pivot.to_numpy(dtype=float)
    finite = numeric[~pd.isna(numeric)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (None, None)

    e3_labels = list(pivot.columns.get_level_values(0).unique())
    span = {e3: sum(1 for c in pivot.columns if c[0] == e3) for e3 in e3_labels}

    # header row 1: E3 ligase structures (the two corner cells span both rows)
    head_struct = [
        html.Th("", className="corner", rowSpan=2,
                style={"width": f"{POI_COL_W}px", "minWidth": f"{POI_COL_W}px"}),
        html.Th(value_label, className="corner value-name", rowSpan=2,
                style={"left": f"{POI_COL_W}px"}),
    ]
    for e3 in e3_labels:
        head_struct.append(
            html.Th(
                structure_cell(e3, e3_smiles.get(e3), E3_IMG, "e3-struct"),
                colSpan=span[e3],
                className="e3-head",
                **{"data-e3": e3},
            )
        )

    # header row 2: concentrations, frozen just below the structure row
    head_conc = [
        html.Th(conc_label(conc), className="conc-head", style={"top": f"{E3_HEAD_H}px"})
        for _, conc in pivot.columns
    ]

    gallery = gallery or {}
    runs = runs or {}
    # DC50s span decades, so significant figures; percentages one decimal.
    per_run = PER_RUN_LABEL in set(pivot.columns.get_level_values(1))
    value_fmt = (lambda v: f"{v:.3g}") if per_run else (lambda v: f"{v:.1f}")
    body_rows = []
    for poi, row in pivot.iterrows():
        cells = [
            html.Td(
                structure_cell(poi, poi_smiles.get(poi), POI_IMG, "poi-struct"),
                className="poi-cell",
                style={"width": f"{POI_COL_W}px", "minWidth": f"{POI_COL_W}px"},
            ),
            html.Td(
                [poi] + (
                    [html.Div(f"on {sorted_on[poi][0]}: {sorted_on[poi][1]:.1f}",
                              className="poi-score")]
                    if sorted_on and poi in sorted_on
                    else []
                ),
                className="poi-name", style={"left": f"{POI_COL_W}px"},
            ),
        ]
        seen_e3 = set()
        for col in pivot.columns:
            val = row[col]
            n = counts.at[poi, col] if not counts.empty else None
            if pd.isna(val):
                content = ""
            else:
                content = [html.Div(f"{val:.2f}", className="val-median")]
                if not pd.isna(n):
                    content.append(html.Div(f"median · n={int(n)}", className="val-n"))
                cell_runs = runs.get((poi, col[0], col[1]))
                if cell_runs:
                    content.append(run_list(cell_runs, value_fmt))

            e3 = col[0]
            key = curve_key(poi, e3)
            # data-e3 lets assets/sar_find.js jump to a compound's column.
            extra = {"data-e3": e3}
            if key in gallery:
                extra["data-curve"] = key
                # Only the first cell of each POI x E3 block is a tab stop, so the
                # curves stay keyboard-reachable without turning a wide matrix
                # into hundreds of tab presses.
                if e3 not in seen_e3:
                    extra["tabIndex"] = 0
                    seen_e3.add(e3)
            cells.append(
                html.Td(content,
                        className="val-cell has-curve" if "data-curve" in extra else "val-cell",
                        style=value_style(val, vmin, vmax, invert), **extra)
            )
        body_rows.append(html.Tr(cells, **{"data-poi": poi}))

    table = html.Div(
        html.Table(
            [
                html.Thead([html.Tr(head_struct), html.Tr(head_conc)]),
                html.Tbody(body_rows),
            ],
            className="sar-table",
        ),
        className="table-scroll",
    )
    # Each curve once, hidden; assets/sar_hover.js copies the right one into a
    # floating panel on hover. Keeping them out of the cells keeps the matrix
    # markup small however wide the table gets.
    curves = html.Div(
        [
            html.Img(src=image["src"], **{"data-key": key, "data-title": image["title"],
                                          "data-caption": image["caption"]})
            for key, images in gallery.items()
            for image in images
        ],
        id="curve-gallery",
    )
    return html.Div([table, curves])


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------
app = Dash(__name__, title="PROTAC SAR matrix (CDD)")
server = app.server


app.layout = html.Div(
    [
        # One compact bar: title, CDD refresh and every control share it, so the
        # matrix below gets as much of the window as possible.
        html.Div(
            [
                html.Div(
                    [
                        html.H1("PROTAC SAR matrix"),
                        html.Div(
                            [
                                html.Button("Refresh from CDD", id="refresh",
                                            className="btn secondary"),
                                html.Span(id="data-status", className="data-note"),
                            ],
                            className="title-data",
                        ),
                    ],
                    className="title-block",
                ),
                # Jump to a compound pair; handled in the browser by
                # assets/sar_find.js, so it never redraws the matrix.
                html.Div(
                    [
                        html.Label("Find compound (row / column)"),
                        html.Div(
                            [
                                dcc.Input(id="find-poi", type="text", placeholder="row, e.g. A645",
                                          list="find-poi-list", autoComplete="off",
                                          className="find-input"),
                                dcc.Input(id="find-e3", type="text", placeholder="col, e.g. A095",
                                          list="find-e3-list", autoComplete="off",
                                          className="find-input"),
                                html.Button("Go", id="find-go", className="btn find-btn"),
                            ],
                            className="find-row",
                        ),
                        html.Div(id="find-msg", className="find-msg"),
                    ],
                    className="ctrl find",
                ),
                html.Div(
                    [
                        html.Label("Assay"),
                        dcc.Dropdown(id="assay", clearable=False),
                    ],
                    className="ctrl",
                ),
                html.Div(
                    [
                        html.Label("Rows"),
                        dcc.Dropdown(id="poi-target", clearable=False),
                    ],
                    className="ctrl",
                ),
                html.Div(
                    [
                        html.Label("Columns"),
                        dcc.Dropdown(id="e3-target", clearable=False),
                    ],
                    className="ctrl",
                ),
                html.Div(
                    [
                        html.Label("Concentration (µM)"),
                        dcc.Dropdown(id="conc", multi=True),
                    ],
                    className="ctrl wide",
                ),
                html.Div(
                    [
                        html.Label("Value shown"),
                        dcc.Dropdown(id="value-col", clearable=False),
                    ],
                    className="ctrl wide",
                ),
                html.Div(
                    [
                        html.Label("Runs used"),
                        dcc.RadioItems(
                            id="scope",
                            options=[
                                {"label": "all runs", "value": ALL_RUNS},
                                {"label": "most recent run", "value": RECENT_RUNS},
                            ],
                            value=ALL_RUNS,
                            className="radio",
                        ),
                    ],
                    className="ctrl",
                ),
                html.Div(
                    [
                        html.Label("Colour scale"),
                        dcc.RadioItems(
                            id="invert",
                            options=[
                                {"label": "high = strong", "value": "no"},
                                {"label": "low = strong", "value": "yes"},
                            ],
                            value="no",
                            className="radio",
                        ),
                    ],
                    className="ctrl",
                ),
                html.Div(
                    [
                        html.Label("Sort at (µM)"),
                        dcc.Dropdown(id="sort-conc", clearable=False),
                    ],
                    className="ctrl",
                ),
                html.Div(
                    [
                        html.Label("Sort order"),
                        dcc.RadioItems(
                            id="sort-dir",
                            options=[
                                {"label": "low → high", "value": "asc"},
                                {"label": "high → low", "value": "desc"},
                            ],
                            value="asc",
                            className="radio",
                        ),
                    ],
                    className="ctrl",
                ),
                html.Div(
                    [
                        html.Button("Download CSV", id="dl-btn", className="btn"),
                        dcc.Download(id="dl"),
                    ],
                    className="ctrl btn-wrap",
                ),
            ],
            className="controls compact",
        ),
        dcc.Loading(html.Div(id="matrix"), type="dot"),
        html.Div(id="status", className="status"),
        # Bumped whenever the data is (re)loaded; every data-driven callback
        # listens to it instead of holding the DataFrame in the browser.
        dcc.Store(id="data-version", data=None),
        # The row / column targets last shown, so picking one axis's target for
        # the other can swap them.
        dcc.Store(id="axes", data=None),
        # The readout the colour scale was last defaulted for.
        dcc.Store(id="colour-readout", data=None),
        dcc.Interval(id="tick", interval=CHECK_INTERVAL_MS, n_intervals=0),
    ],
    className="page compact",
)


@app.callback(
    Output("data-version", "data"),
    Output("data-status", "children"),
    Input("refresh", "n_clicks"),
    Input("tick", "n_intervals"),
    State("data-version", "data"),
)
def refresh_data(_n_clicks, _tick, version):
    """Publish the local copy, and only ever download in the background.

    The page is populated from the CSV in this directory the moment it loads.
    Pressing "Refresh from CDD" is what asks for fresh data; otherwise a
    download only starts when there is nothing usable on disk. Either way it
    runs on its own thread and is picked up by a later tick, so nothing here
    waits on the vault.

    The trigger matters, not `n_clicks`: n_clicks stays at its last value for the
    life of the page, so testing it would make every later interval tick behave
    like another button press and re-download.
    """
    load_local()
    start_download(force=ctx.triggered_id == "refresh")

    current = _CACHE["version"]
    if current == version:
        return no_update, data_status()
    return current, data_status()


def keep_value(current, options, default):
    """Hold on to what the user picked, as long as it still exists."""
    values = [o["value"] for o in options]
    return current if current in values else default


@app.callback(
    Output("assay", "options"),
    Output("assay", "value"),
    Output("poi-target", "options"),
    Output("poi-target", "value"),
    Output("e3-target", "options"),
    Output("e3-target", "value"),
    Output("axes", "data"),
    Input("data-version", "data"),
    Input("poi-target", "value"),
    Input("e3-target", "value"),
    State("assay", "value"),
    State("axes", "data"),
)
def populate_controls(_version, poi, e3, assay, axes):
    """Fill the dropdowns, keeping whatever the user had selected.

    This runs again after every reload of the data, so it must not reset the
    controls: doing that throws away the view the user built and blanks the
    matrix while the whole chain of callbacks re-runs.

    Picking for one axis the target the other axis shows swaps the two. Without
    that, turning FAK x VHL into VHL x FAK passes through an empty VHL x VHL
    matrix, and everything built on the concentrations shown (the sort among
    them) is lost on the way.
    """
    df = current_data()
    if df.empty:
        return [], None, [], None, [], None, no_update

    assays = sorted(df[ASSAY_COL].dropna().astype(str).unique()) if ASSAY_COL in df else []
    a_opts = [{"label": a, "value": a} for a in assays]
    a_default = DEFAULT_ASSAY if DEFAULT_ASSAY in assays else (assays[0] if assays else None)

    targets = sorted(set(df[TARGET_COL].dropna().astype(str)) - EXCLUDED_TARGETS)
    t_opts = [{"label": t, "value": t} for t in targets]
    poi_default = "FAK" if "FAK" in targets else (targets[0] if targets else None)
    e3_default = "VHL" if "VHL" in targets else (targets[-1] if targets else None)

    poi = keep_value(poi, t_opts, poi_default)
    e3 = keep_value(e3, t_opts, e3_default)
    if poi == e3 and axes:
        if ctx.triggered_id == "poi-target":
            e3 = axes.get("rows")
        elif ctx.triggered_id == "e3-target":
            poi = axes.get("cols")
    return (
        a_opts, keep_value(assay, a_opts, a_default),
        t_opts, poi,
        t_opts, e3,
        {"rows": poi, "cols": e3},
    )


@app.callback(
    Output("value-col", "options"),
    Output("value-col", "value"),
    Input("data-version", "data"),
    Input("assay", "value"),
    State("value-col", "value"),
)
def populate_readouts(_version, assay, value_col):
    """Readouts the chosen assay actually measured, keeping the user's pick."""
    df = filter_assay(current_data(), assay)
    allowed = ASSAY_READOUTS.get(assay)
    v_opts = [{"label": label, "value": col} for col, label in value_columns(df)
              if allowed is None or readout_name(col) in allowed]
    v_default = v_opts[0]["value"] if v_opts else None
    return v_opts, keep_value(value_col, v_opts, v_default)


@app.callback(
    Output("invert", "value"),
    Output("colour-readout", "data"),
    Input("value-col", "value"),
    State("colour-readout", "data"),
)
def default_colour_scale(value_col, last_readout):
    """Pick the colour direction that suits a newly chosen readout.

    Only on a change of readout, so a data reload does not undo a flip the
    user made by hand.
    """
    if not value_col or value_col == last_readout:
        return no_update, no_update
    return ("yes" if readout_name(value_col) in LOW_IS_STRONG else "no"), value_col


@app.callback(
    Output("conc", "options"),
    Output("conc", "value"),
    Input("data-version", "data"),
    Input("assay", "value"),
    Input("poi-target", "value"),
    Input("e3-target", "value"),
    Input("scope", "value"),
    State("conc", "value"),
)
def populate_concentrations(_version, assay, poi_target, e3_target, scope, current):
    """Concentrations present in the data; open on 0.1 / 0.3 / 3 µM.

    Keeps the user's selection over a data reload, dropping only concentrations
    that are no longer in the data.
    """
    df = scoped_data(assay, poi_target, e3_target, scope)
    combos = target_combinations(df, poi_target, e3_target)
    if combos.empty:
        # Keep the picks: they come back into play with the next target or
        # assay that has data, and clearing them would reset the sort.
        return [], no_update
    concs = sorted(
        pd.to_numeric(combos[concentration_column(df)], errors="coerce")
        .dropna().unique().tolist()
    )
    opts = [{"label": f"{c:g}", "value": c} for c in concs]
    kept = [c for c in (current or []) if c in concs]
    return opts, kept or [c for c in DEFAULT_CONCENTRATIONS if c in concs] or concs


@app.callback(
    Output("sort-conc", "options"),
    Output("sort-conc", "value"),
    Input("conc", "value"),
    Input("value-col", "value"),
    State("sort-conc", "value"),
)
def populate_sort_conc(concs, value_col, current):
    """The concentrations on show to sort at; the matrix is always sorted.

    DC50 / Dmax have no concentration of their own: they sort on their single
    per-run column.
    """
    if is_per_run_column(value_col):
        values = [PER_RUN_LABEL]
    else:
        values = [float(c) for c in sorted(concs or [])]
    opts = [{"label": v if v == PER_RUN_LABEL else f"{v:g}", "value": v} for v in values]
    if not values:
        # Nothing to sort at right now: keep the choice for when data is back.
        return opts, no_update
    if current in values:
        return opts, current
    return opts, closest_sort_value(current, values)


def closest_sort_value(current, values):
    """Stand-in for a sort choice that is not on offer (or none made yet).

    A per-run readout sorts per run. A concentration that is gone gives way to
    the nearest one on show (on a log scale); with no usable previous choice,
    3 µM or else the highest on show.
    """
    if PER_RUN_LABEL in values:
        return PER_RUN_LABEL
    if current in (None, PER_RUN_LABEL):
        return 3.0 if 3.0 in values else values[-1]
    target = math.log10(float(current))
    return min(values, key=lambda c: abs(math.log10(c) - target))


@app.callback(
    Output("matrix", "children"),
    Output("status", "children"),
    Input("data-version", "data"),
    Input("assay", "value"),
    Input("poi-target", "value"),
    Input("e3-target", "value"),
    Input("conc", "value"),
    Input("value-col", "value"),
    Input("invert", "value"),
    Input("sort-conc", "value"),
    Input("sort-dir", "value"),
    Input("scope", "value"),
)
def update_matrix(_version, assay, poi_target, e3_target, concs, value_col, invert,
                  sort_conc, sort_dir, scope):
    # Say why the matrix is not there rather than returning no_update: on a fresh
    # page the div starts empty, so a silent no_update leaves a blank screen.
    if current_data().empty:
        return html.Div(
            "Downloading the CDD data… this page fills in by itself when it lands."
            if download_running() else
            "No CDD data available in this directory. Press “Refresh from CDD” to "
            "download it (cddVaultId and cddAPIToken must be set in the environment).",
            className="empty",
        ), data_status()
    if not (poi_target and e3_target and value_col):
        return html.Div("Choose a target for the rows, one for the columns and a readout.",
                        className="empty"), ""

    df = scoped_data(assay, poi_target, e3_target, scope)
    labels = dict(value_columns(df))
    pivot, counts, poi_smiles, e3_smiles = build_pivot(
        df, poi_target, e3_target, concs, value_col
    )
    readout = labels.get(value_col, readout_name(value_col))
    pivot, counts, sorted_on = sort_by_best_pairs(pivot, counts, sort_conc, sort_dir)
    cell_runs = run_values(df, poi_target, e3_target, concs, value_col)
    table = render_matrix(
        pivot, counts, poi_smiles, e3_smiles, readout, invert == "yes",
        gallery=None if assay in NO_CURVE_ASSAYS else curve_gallery(
            df, pivot, value_col, poi_target, e3_target),
        sorted_on=sorted_on,
        runs=cell_runs,
    )
    if pivot.empty:
        return table, ""

    n_vals = counts.to_numpy(dtype=float)
    n_vals = n_vals[~pd.isna(n_vals)]
    unit = "dose-response runs" if is_per_run_column(value_col) else "measurements"
    note = (
        " This readout is fitted per dose-response curve, so it has one column "
        "per E3 warhead and the concentration filter does not apply to it."
        if is_per_run_column(value_col) else ""
    )
    scope_note = ("Most recent run date only, per pair. " if scope == RECENT_RUNS else "")
    shown = {(poi, e3) for poi in pivot.index for e3 in pivot.columns.get_level_values(0)}
    hooks = sum(hook for (poi, e3, _), lst in cell_runs.items() if (poi, e3) in shown
                for *_, hook in lst)
    if hooks:
        note += (f" {hooks} DC50{'s' if hooks != 1 else ''} from hook-side fits (positive Hill "
                 "slope) left out of the medians and curves; marked “hook fit” in the cells.")
    status = (
        f"{scope_note}{len(pivot)} {poi_target} warheads × "
        f"{len(pivot.columns.get_level_values(0).unique())} {e3_target} warheads — "
        f"{int(pivot.notna().to_numpy().sum())} measured combinations, "
        f"{int(n_vals.sum())} {unit} (n per cell "
        f"{int(n_vals.min())}–{int(n_vals.max())}).{note} "
        f"{'' if assay in NO_CURVE_ASSAYS else 'Hover a cell for its dose-response curve. '}"
        f"{sort_description(pivot, sort_conc, sort_dir, readout, poi_target, e3_target, sorted_on)}"
    )
    return table, status


@app.callback(
    Output("dl", "data"),
    Input("dl-btn", "n_clicks"),
    State("assay", "value"),
    State("poi-target", "value"),
    State("e3-target", "value"),
    State("conc", "value"),
    State("value-col", "value"),
    State("sort-conc", "value"),
    State("sort-dir", "value"),
    State("scope", "value"),
    prevent_initial_call=True,
)
def download_csv(_n, assay, poi_target, e3_target, concs, value_col,
                 sort_conc, sort_dir, scope):
    df = scoped_data(assay, poi_target, e3_target, scope)
    pivot, counts, poi_smiles, e3_smiles = build_pivot(
        df, poi_target, e3_target, concs, value_col
    )
    # export in the order shown on screen
    stamp = dt.date.today().isoformat()
    pivot, counts, _ = sort_by_best_pairs(pivot, counts, sort_conc, sort_dir)
    if pivot.empty:
        return no_update

    # Mirror the notebook layout: E3 name / E3 SMILES / concentration header rows,
    # POI name + SMILES columns.
    out = pivot.copy()
    out.columns = pd.MultiIndex.from_tuples(
        [(e3, e3_smiles.get(e3), conc_label(conc)) for e3, conc in pivot.columns],
        names=["E3", "SMILES_e3", "concentration"],
    )
    # ...then the same grid of n values, appended on the right so the value
    # block keeps the notebook's shape.
    n_out = counts.copy()
    n_out.columns = pd.MultiIndex.from_tuples(
        [(e3, e3_smiles.get(e3), f"{conc_label(conc)} (n)") for e3, conc in counts.columns],
        names=out.columns.names,
    )
    out = pd.concat([out, n_out], axis=1)
    out.insert(0, ("SMILES_poi", "", ""), [poi_smiles.get(p) for p in out.index])
    out = out.reset_index()

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return dcc.send_string(
        buf.getvalue(),
        f"cdd_{poi_target}_x_{e3_target}_matrix_{'most_recent_' if scope == RECENT_RUNS else ''}{stamp}.csv"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        help="use 0.0.0.0 to expose the app on the network")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--refresh", action="store_true",
                        help="download from CDD on start-up regardless of cache age")
    parser.add_argument("--offline", action="store_true",
                        help="never contact CDD, not even when there is no local copy")
    args = parser.parse_args()

    ALLOW_DOWNLOAD = not args.offline

    # Serve the local copy first, always. A download — forced with --refresh, or
    # because there is nothing usable in the directory — runs in the background
    # while the page is already up, so start-up never waits on the vault.
    load_local()
    print(data_status())
    if start_download(force=args.refresh):
        print("Downloading fresh CDD data in the background; "
              "the page will update when it lands.")

    app.run(debug=args.debug, host=args.host, port=args.port)
