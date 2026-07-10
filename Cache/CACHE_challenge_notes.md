# DREAM x CACHE Target 2035 Drug Discovery Challenge — Notes

Source: https://www.synapse.org/Synapse:syn75349604/wiki/
Notes compiled: 2026-07-05

## What the challenge is

Second **Target 2035 DEL-ML Challenge**. Participants build DNA-encoded
library (DEL) + ML models to discover hits against **PGK2**
(phosphoglycerate kinase 2), following the DEL-ML approach introduced by
Google/X-Chem in 2020. Part of the broader [Target 2035](https://target2035.net/)
open-science initiative; data hosted on [AIRCHECK](https://aircheck.ai/).

Overview of the inaugural (WDR91) challenge: https://chemrxiv.org/doi/full/10.26434/chemrxiv.15004205
Sample baseline code: https://github.com/StructuralGenomicsConsortium/Target2035_Aircheck_Utils
Q&A webinar: July 7th, 9 AM EDT / 1 PM UTC / 3 PM CET.

## Data

Files (download from https://aircheck.ai/challenges, requires AIRCHECK
registration; academic emails auto-approved):

| File | Description | Size |
|---|---|---|
| `OpenDEL_libraries.zip` | Fully enumerated screening library (898,311,048 compounds across 12 sub-libraries) + building blocks | ~5 GB zipped (confirmed on wiki); likely 15–25 GB unzipped |
| `PGK2_selection.parquet` | DEL selection read counts / z-scores for 3 conditions (PGK2, PGK2+inhibitor, NTC) | not published — likely low single-digit GB |
| `PGK2_CACHE_Val_Test_Set` | 400K-compound ASMS screen, split into validation/test, 431 confirmed hits | small, likely <100 MB |

**Download limit**: 10 total dataset downloads per rolling 30-day period,
shared across all files — don't re-download casually.

**Important data notes**:
- DEL data is noisy, both training and test sets are highly imbalanced (few positives).
- Data is pocket-agnostic (ATP site vs phosphoglycerate site).
- Little chemical-space overlap between training (DEL) and test (ASMS) sets — predictions are mostly out-of-distribution.
- Files contain duplicate compounds (same SMILES, different compound IDs) by design — dedupe by structure, sum counts, combine z-scores via Stouffer's method before modeling.
- Baseline: LightGBM on ECFP4, trained on all DEL-hits + 2x negatives, found 5 hits (3 series) in top-50 on validation, 1 hit in top-50 on test.
- IBM's MAMMAL/MMELON models showed preliminary positive results.

## Challenge structure & 2026 key dates (deadlines 23:59 UTC)

| Dates | Phase | Details |
|---|---|---|
| June 16 | Registration opens | |
| June 16 – Sept 15 | **Blind test step** (retrospective) | Train on DEL data, predict true positives in ASMS validation split (up to 100 submissions of 50 hits, unscored/exploratory) and test split (max 2 submissions of 50 hits — this is scored) |
| Sept 16 – Oct 15 | **Active learning step** | Test-split labels released; fine-tune and resubmit (100 more validation submissions, 2 more test submissions, credits reset) |
| Oct 16 – Nov 30 | **Prospective phase** | Top 5 teams from Blind test step + top 5 from Active learning step only. Predict 50 novel hits from 2M-compound Enamine screening catalog; organizers synthesize & test experimentally |
| June 16 – Oct 15 | Writeup submissions | Accepted throughout |

## Evaluation

- **Validation/test splits**: ranked by (1) number of chemically distinct hit series, (2) number of hits, then for test split also (3) PR-AUC and (4) ROC-AUC.
- **Prospective phase**: ranked by number of hits, chemical novelty, chemical diversity.

## Submission process

1. Register on Synapse, become **Synapse Certified** (short quiz), join challenge team (Team 3595357).
2. Create one Synapse Project per team; share with teammates.
3. Submit prediction files:
   - Validation split: `.txt`, 50 CatalogIDs per line, named `Team_YOURTEAMNAME_UNIQUEID.txt`.
   - Test split: 3-column CSV (`CatalogID`, `Sel_50` binary flag for 50 chosen molecules, `Score` float), named `Team_YOURTEAMNAME_UNIQUEID.csv`.
   - Prospective phase: `Team_YOURTEAMNAME.csv`, 50 compounds from Enamine collection.
4. Submit a writeup (template linked on Submit page) describing the workflow.

## Conditions / incentives

- Retrospective phase is free; Prospective-phase compound procurement cost is on participants (waived for top-2 MAINFRAME teams, funded by SGC).
- Open science: no patents; publication embargo until organizers' overview paper is out.
- Top performers (better-than-random in Blind test/Active learning) invited as co-authors on the overview paper.
- Canadian academic/SME participants may qualify for extra funding via Conscience (form on main page).

## Practical / compute planning discussion

- Raw downloads: ~30 GB free disk space is comfortable (5 GB zip + unzip + working copies). Home disk here has 475 GB free — plenty.
- **Do not** run `kepler_ai`'s `calc_all_descriptors()` (PhysChem + RDK/AtomPair/Torsion/Morgan fingerprints as one flat DataFrame) on the full 898M-compound library as-is — would balloon into the hundreds of GB to low TB range. If full-library descriptors are ever needed, compute in chunks/streamed batches (HDF5 or memory-mapped arrays), not one in-memory DataFrame.
- **The 898M figure is training-pool size, not a prediction target.** Actual prediction targets are much smaller and easy: the 400K ASMS validation/test set (retrospective phase) and the 2M Enamine library (prospective phase).
- **Chemprop is feasible** for this challenge:
  - Don't train on all 898M raw DEL rows — follow the baseline's approach: dedupe first (sum counts, Stouffer-combine z-scores), then subsample to all DEL-hits + ~2x negatives (nets a training set in the hundreds of thousands to low millions of rows — well within Chemprop's normal operating range).
  - Inference on 400K / 2M compounds is fast (minutes to low hours on one GPU).
  - Chemprop v2 supports adding precomputed descriptors/fingerprints alongside its learned graph representation, so it can be combined with `kepler_ai`-style features if desired.

## ML Modelling Plan

### Data dictionary for `PGK2_selection.parquet`

(From the Synapse wiki's "Hit Selection" / "Deduplicate Selection" pages —
fetched via the Synapse REST API since the wiki UI is JS-rendered.)

| Column | Aggregation | Meaning |
|---|---|---|
| `compound` | first | BCM compound ID, `<library>-<BB1_UID>-<BB2_UID>-<BB3_UID>` |
| `SMILES` | key | Canonical SMILES — the dedup key |
| `count_PGK2` | sum | Read counts, PGK2 standard selection |
| `count_PGK2_with_inhibitor` | sum | Read counts, PGK2 + known ATP-site inhibitor (competition assay) |
| `count_NTC` | sum | Read counts, no-target control (matrix only) |
| `zscore_PGK2` / `_with_inhibitor` / `_NTC` | Stouffer: `sum(z)/sqrt(n)` | Enrichment Z-score (Faver et al., ACS Comb. Sci. 2019) |
| `historic_hits` | max | Times this exact structure was a hit in *prior, unrelated* DEL campaigns — a promiscuity/artifact flag, not PGK2-specific |

**Binder-mechanism decision:** we're labeling **both** orthosteric and
allosteric binders as hits (not orthosteric-only), because the challenge
is explicitly pocket-agnostic (ATP site vs. phosphoglycerate site) and the
true mechanism of the held-out ASMS hits is unknown — broader recall was
preferred over mechanism purity.

**Hit criteria used** (BCM's orthosteric recipe, with the
inhibitor-competition condition dropped so allosteric binders aren't
excluded):

| Criterion | Threshold |
|---|---|
| `count_PGK2` | `>= 3` |
| `count_NTC` | `== 0` |
| `historic_hits` | `< 5` |

### Status so far (completed 2026-07-06)

1. **Downloaded & extracted** all three AIRCHECK files into `~/Cache/`
   (`OpenDEL_libraries.zip` → `OpeDELLibrary/`, `Val-Test-set.zip` →
   `Val-Test-set/`). `OpenDEL_libraries.zip` + extracted `OpeDELLibrary/`
   (~10.5GB, the full 898M-compound enumeration — not needed for training
   since `PGK2_selection.parquet` already carries SMILES directly) are
   being archived to GCS cold storage
   (`gs://sandeep-linux-20260313-908b3fcbc575434b956480787faf485b/cache-challenge/`)
   and then removed locally to reclaim disk space.
2. **Deduped + labeled** `PGK2_selection.parquet` via
   `~/Cache/pipeline/dedup_and_label.py`:
   - Cleaning: 7,703,070 → 7,674,854 rows (drops invalid compound-ID/SMILES rows)
   - Dedup by SMILES: 7,674,854 → 7,487,567 unique structures (matches official wiki figures exactly)
   - Hits under the combined criteria above: **27,642** (0.37%)
   - Training set: 27,642 hits + 55,284 subsampled negatives (2:1 ratio) = 82,926 rows
   - Saved to `~/Cache/pipeline/PGK2_train_labeled.parquet`

### Chemical matter among the 27,642 hits (analysis: `~/Cache/pipeline/hit_series_analysis.py`)

Hits are predominantly organized into series, not isolated singletons/matched pairs:

- **Synthon-level series** (2 of 3 building blocks fixed, 3rd varies — the
  DEL-native analog-series definition): BB1+BB2 fixed → 2,280 series
  covering 14,281 hits (52%, largest group 541); BB1+BB3 fixed → 2,107
  series covering 13,252 hits (48%, largest 578); BB2+BB3 fixed → 3,510
  series covering 16,467 hits (60%, largest 295).
- **Single-BB sharing** is even more extreme: 93% of hits (25,766/27,642)
  share their BB3 with at least one other hit, and one BB3 value alone
  connects 5,329 hits. Flagged as worth checking before modeling — could
  be a genuinely important pharmacophore feature, or a
  promiscuity/artifact riding along with that synthon.
- **Bemis-Murcko scaffold grouping** (classic med-chem series
  definition): 16,341 distinct scaffolds among 27,642 hits; 12,893 (47%)
  are singleton chemotypes; 3,448 scaffolds form real series covering
  53.4% of hits; largest single series is 395 hits around one scaffold
  (pyrrolopyrimidine-carboxamide-piperidine-biphenyl core).

**Implication for modelling:** internal train/val splits must be
scaffold- or BB-aware (not random), or near-identical analogs leak across
the split and inflate metrics. Also worth remembering the organizers score
on number of distinct hit *series* found in the top-N picks, not raw hit
count — a model that only nails the dominant 395-member series would look
good on naive hit-count metrics while actually representing one chemotype,
not many.

### Investigation: the dominant BB3 (library `qDOS30`, BB3 ID `805`, shared by 5,329 hits)

Analysis script: `~/Cache/pipeline/bb3_investigation.py`. This building
block is `OC(=O)c1cnc2[nH]cnc2c1` — a pyrrolo[2,3-d]pyrimidine
carboxylic acid (7-deazapurine / adenine mimetic, MW 163, no PAINS
alerts). It is the core of the dominant Murcko-scaffold series found
above.

**Evidence this is real signal, not a promiscuity/assay artifact:**
- 101.6x hit-rate enrichment vs. background (37.5% hit rate among the
  14,209 screened compounds carrying this BB3, vs. 0.37% overall).
- Not a bead-binder: `count_NTC` for this group is ~0.
- Orthosteric mechanism confirmed: `count_PGK2_with_inhibitor` is
  essentially zero even among the hits (mean 0.0077 vs. mean
  `count_PGK2` of 19.5) — the known ATP-site inhibitor nearly completely
  blocks this fragment, consistent with a genuine adenine-mimetic
  hinge-binder occupying the ATP pocket directly.
- Low historic promiscuity: mean `historic_hits` in this group is only
  0.13 (0.34 restricted to hits) — not a serial hitter across unrelated
  past campaigns.
- Broad partner tolerance: pairs with 49 distinct BB1s and 610 distinct
  BB2s among its 5,329 hits and still confers hit status — consistent
  with an anchor fragment making the core binding contact while BB1/BB2
  explore solvent-exposed space, which is hard to explain as artifact.

**Modelling caution:** don't let a model shortcut to "contains this BB3 →
predict hit" without learning the surrounding SAR (would generalize
poorly to the structurally distinct ASMS test set). Final submissions
also shouldn't be dominated entirely by this one chemotype even though
it's the strongest series, since scoring rewards distinct hit series
found, not raw hit count.

### Remaining steps

1. **Featurization** — compute a molecular representation for every SMILES
   in the training set, `Val-Test-set/PGK2_Validation_split.csv`
   (244,329 compounds), and `PGK2_Test_split.csv` (184,633 compounds):
   - Baseline path: ECFP4 (Morgan, radius 2) fingerprints via RDKit —
     matches the organizers' own baseline and `Target2035_Aircheck_Utils`
     example notebook.
   - Chemprop path: no manual featurization needed — it learns a graph
     representation directly from SMILES; can optionally bolt on
     precomputed fingerprints/descriptors alongside.

2. **Split strategy for internal validation** — must NOT use a random
   split. DEL libraries are combinatorial (shared building blocks across
   many compounds), so a random split leaks near-duplicate structures
   between train/val and gives overoptimistic metrics. Use a
   scaffold-based or cluster-based split instead. This matters doubly here
   because the real test set (ASMS) has **little chemical-space overlap**
   with the DEL training data — an internal validation split should try to
   mimic that out-of-distribution gap, not just guard against leakage.

3. **Train models**:
   - Baseline: LightGBM classifier on ECFP4 (fast, matches organizers'
     published baseline: found 5 hits/3 series in top-50 on their
     validation split, 1 hit in top-50 on test).
   - Chemprop: message-passing neural net; feasible at this scale
     (82,926 training rows is well within normal operating range);
     inference on 244K/185K compounds is fast (minutes on one GPU).
   - Handle class imbalance (label ratio is 1:2 in the training set, but
     true prevalence is ~0.37% — keep this in mind when interpreting
     probabilities/calibration).

4. **Cross-validate / tune** — k-fold (scaffold-aware) or
   `optuna`-based hyperparameter search (kepler_ai's `model_builders.py`
   already uses this pattern for its global ADME/Tox models, reusable
   style-wise even though this is a different target/library).

5. **Evaluate** using the organizers' own metric definitions
   (`EvaluationCode/evaluation_function.py` from
   `Target2035_Aircheck_Utils`): ROC-AUC / PR-AUC over all scored
   compounds, plus a cluster-aware hit count — the real scoring rewards
   number of **chemically distinct hit series** found in the top-N
   picks, not just raw hit count. Clustering (e.g. Agglomerative on
   Tanimoto/Jaccard distance) should be applied to top-ranked candidates
   before final selection, not to all 400K compounds.

6. **Predict + select final compounds**:
   - Validation split: rank all 244,329 compounds by predicted score,
     cluster the top candidates, pick 50 diverse ones per submission
     (up to 100 submissions allowed, unscored — good place to
     experiment with different models/thresholds).
   - Test split: same idea, but only 2 submissions allowed and it's
     scored — should be the most-validated model/configuration.

7. **Format submissions** per the Synapse submission spec:
   - Validation: `.txt`, 50 CatalogIDs per line, named
     `Team_YOURTEAMNAME_UNIQUEID.txt`.
   - Test: 3-column CSV (`CatalogID`, `Sel_50` binary flag, `Score`
     float), named `Team_YOURTEAMNAME_UNIQUEID.csv`.
   - Submit through the team's Synapse project (Team 3595357).

8. **Active learning step** (Sept 16 – Oct 15 2026) — once test-split
   true labels are released, fold them back into training data and
   retrain/fine-tune, then resubmit (100 more validation submissions, 2
   more scored test submissions).

9. **Prospective phase** (Oct 16 – Nov 30 2026, top 5 teams from each
   step only) — apply the final model to a 2M-compound Enamine
   screening catalog (not yet downloaded) to pick 50 novel compounds for
   organizer-funded synthesis and experimental testing.
