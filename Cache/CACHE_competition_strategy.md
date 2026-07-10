# CACHE 2035 PGK2 Challenge: Strategy Advice

Prepared: 2026-07-10

## MMELON github repository for installation in VM

https://github.com/BiomedSciAI/biomed-multi-view

## Executive recommendation

The best chance of doing well is not to chase a single highest-AUC model. The scoring rewards finding chemically distinct hit series in the top 50, and the local data confirms that the ASMS validation/test compounds are mostly out of distribution from the DEL training data. The winning strategy should therefore be:

1. Build several complementary ligand-based rankers.
2. Validate them with scaffold/series-aware splits only.
3. Use validation submissions aggressively to learn which model families and selection rules transfer.
4. Submit test picks only after diversity-filtered validation performance is convincing.
5. Select final 50 compounds by score plus chemical-series diversity, not raw score alone.

The current notes and pipeline have made the right early calls: deduplication, Stouffer-combined z-scores, broad PGK2 hit labeling, and recognition that random train/validation splits are misleading. The main risk now is overfitting to DEL-specific series, especially the dominant qDOS30 BB3 adenine-like series, which appears to be real biology but is absent from the ASMS target splits.

## Local evidence

The current processed data are:

| Dataset | Rows |
|---|---:|
| Deduped DEL selection | 7,487,567 |
| Current training subset | 82,926 |
| Labeled DEL hits | 27,642 |
| Sampled DEL negatives | 55,284 |
| ASMS validation split | 244,328 |
| ASMS test split | 184,632 |

Overlap diagnostics from `openadmet-models`:

| Comparison | Exact SMILES overlap |
|---|---:|
| Training subset vs validation | 0 |
| Training subset vs test | 0 |
| Training hits vs validation | 0 |
| Training hits vs test | 0 |
| Full deduped DEL vs validation | 2 |
| Full deduped DEL vs test | 0 |
| Validation vs test | 0 |

Scaffold overlap is also limited:

| Target split | Rows sharing a DEL-hit scaffold | Unique overlapping DEL-hit scaffolds |
|---|---:|---:|
| Validation | 6,403 / 244,328 | 99 |
| Test | 8,378 / 184,632 | 81 |

The dominant DEL hit scaffold/motif is not directly present in the target splits:

| Query | Validation matches | Test matches |
|---|---:|---:|
| Dominant qDOS30 BB3 acid motif | 0 | 0 |
| Broader pyrrolopyrimidine core query | 0 | 0 |

Interpretation: the dominant DEL series is still useful as a training signal, but it cannot be used as a direct selection shortcut. The challenge is almost entirely transfer from noisy DEL enrichment to ASMS chemical space.

## What I would do next

### 1. Keep the current label, but add alternate targets

The current hit label is sensible:

- `count_PGK2 >= 3`
- `count_NTC == 0`
- `historic_hits < 5`
- no inhibitor-competition exclusion, so both orthosteric and allosteric binders remain eligible

Do not rely on this one binary label only. Train models against multiple related targets and ensemble their ranks:

- Binary broad-hit label: current label.
- Stricter high-confidence hit label: require higher `count_PGK2` or `zscore_PGK2`.
- Orthosteric-enriched label: require PGK2 signal reduced by inhibitor.
- Continuous enrichment target: predict `zscore_PGK2`, with NTC and historic hits used as penalties.
- Artifact/promiscuity model: predict NTC or historic-hit behavior and subtract it from binding score.

This should improve robustness because the held-out ASMS labels are not guaranteed to match one DEL threshold exactly.

### 2. Improve negative sampling

The current 2:1 random negative sample is a good first baseline, but it throws away most of the 7.46M non-hit structures and may make the task too easy internally. Add several negative pools:

- Random negatives, stratified by DEL library.
- Hard negatives near hit scaffolds or hit fingerprints.
- High-NTC negatives.
- Historic-hit/promiscuous negatives.
- Inhibitor-insensitive PGK2-enriched compounds as possible non-orthosteric/artifact contrasts, depending on the target variant.

For LightGBM this is cheap. Train several models with different negative definitions, then ensemble by rank. The goal is not calibrated probability; it is stable top-50 ranking under distribution shift.

### 3. Use model diversity, not just hyperparameter tuning

Recommended first model stack:

| Model | Features | Why |
|---|---|---|
| LightGBM baseline | ECFP4/ECFP6 bits or counts | Fast, strong baseline, easy to ensemble |
| Logistic/linear model | ECFP counts | Useful sanity check; often transfers better than complex models |
| Chemprop | SMILES graph | Captures patterns not explicit in Morgan bits |
| MMELON | IBM multi-view small-molecule embeddings or fine-tuned classifier | Strong fit for ligand-only transfer because it combines graph, SMILES-text, and 2D image views |
| kNN/similarity ranker | max/mean similarity to high-confidence hits minus artifacts | Useful for scaffold-near target compounds |
| Structure/pharmacophore screen | PGK2 ATP-site docking or shape/electrostatics for top few thousand | Adds target-structure signal missing from DEL-only learning |

The structure-based layer should be used as a reranker, not as the main screen. Docking all 400K is possible but unnecessary; docking or shape-screening the top 2K to 10K from ligand models is a better use of time.

### 3a. Use IBM MMELON as a ligand-transfer model

IBM's MMELON implementation is a better immediate fit for this challenge than a general biomedical prompt model because it directly produces small-molecule representations from SMILES. The public checkpoint `ibm/biomed.sm.mv-te-84m` combines three ligand views: molecular graph, SMILES text, and RDKit-generated 2D image. For CACHE, use it in two stages:

1. Generate pretrained MMELON embeddings for the DEL training subset, ASMS validation split, and ASMS test split.
2. Train lightweight heads on top of those embeddings: logistic regression, LightGBM, or a small MLP.

Only fine-tune the full MMELON model after the embedding baseline works. Full fine-tuning is more expensive and easier to overfit to DEL series; frozen embeddings plus simple supervised heads are a cleaner first experiment.

Recommended MMELON targets:

- Current broad binary PGK2 label.
- Strict high-confidence hit label.
- Continuous `zscore_PGK2` regression.
- Artifact/promiscuity penalty head using `count_NTC` and `historic_hits`.

Then add MMELON-derived ranks to the ensemble. Do not submit raw MMELON top 50 without clustering; its output still needs the same diversity-aware selector as LightGBM and Chemprop.

### 4. Validate like the competition scores

Do not optimize random-split ROC-AUC. Use at least three validation views:

- Scaffold split: Murcko or fingerprint clusters held out.
- DEL-series split: hold out building-block pair series where possible.
- Target-like split: train on DEL compounds far from a held-out subset and test transfer.

For every model, report:

- PR-AUC, not only ROC-AUC.
- Hits in top 50.
- Chemically distinct hit scaffolds/series in top 50.
- Diversity-adjusted top 50 after clustering.
- Performance excluding the dominant qDOS30 BB3 family.

The last item is important. A model that works only because it identifies the dominant DEL ATP-like family is unlikely to transfer, since that motif is absent from validation/test.

### 5. Selection rule for submissions

For validation submissions, use the 100 allowed attempts as experiments. Submit distinct strategies, not minor threshold variants:

- ECFP LightGBM raw top 50.
- ECFP LightGBM diversity-filtered top 50.
- Chemprop top 50.
- Ensemble rank top 50.
- Ensemble rank with max one to three molecules per Murcko scaffold.
- Orthosteric-biased model.
- Broad binder model.
- Similarity-to-hit-scaffold model.
- Structure-reranked ligand model.

For scored test submissions, be conservative. Use only two:

1. Best validation-proven ensemble, diversity-filtered.
2. A deliberately different backup: either more structurally diverse or more structure-reranked, depending on what validation feedback shows.

A practical selector:

1. Rank all target molecules by ensemble score.
2. Remove compounds with obvious liabilities if allowed by the challenge goals: reactive groups, frequent hitters, very high lipophilicity, extreme size, unstable groups.
3. Cluster the top 2,000 to 10,000 by ECFP Tanimoto.
4. Select one molecule from each high-scoring cluster first.
5. Add second representatives only for clusters with very strong scores and good analog support.
6. Manually inspect the final 50 for redundant chemotypes.

This matches the scoring better than taking the top 50 raw scores.

## Highest-risk assumptions

The biggest uncertainty is that the validation split is exploratory/unscored, but it likely still reflects the same ASMS source as test. Use it heavily, but do not overfit to it by repeatedly hand-tuning tiny chemical rules unless those rules are chemically defensible.

The second risk is treating all unlabeled DEL non-hits as true inactive compounds. DEL non-enrichment does not prove no binding; it can reflect synthesis, display, amplification, or selection noise. This is why rank ensembling, hard-negative definitions, and continuous enrichment targets are better than one binary classifier.

The third risk is ignoring the inhibitor condition entirely. Keeping allosteric binders in the broad label is sensible for recall, but an orthosteric-specific model should still be trained as a separate ensemble member because the strongest DEL evidence points to a real ATP-site series.

## Immediate action plan

1. Implement ECFP4/ECFP6 featurization for training, validation, and test.
2. Train LightGBM models for broad-hit, strict-hit, orthosteric-hit, and z-score targets.
3. Add scaffold/cluster split evaluation and top-50 distinct-series reporting.
4. Train Chemprop in parallel once the baseline pipeline is working.
5. Generate validation submissions from deliberately different selectors.
6. Use validation feedback to choose the two test submissions.
7. Prepare the writeup continuously, including failed approaches and evidence that the final selector optimizes distinct series.

## Bottom line

The best path is a pragmatic ensemble-and-diversity workflow. The data are too out-of-distribution for a single DEL-trained model to be trusted, and the scoring makes duplicate chemotypes expensive. Use the DEL data to learn broad PGK2 enrichment signals, use ASMS validation submissions to identify what transfers, and use diversity-aware selection so the final 50 compounds maximize independent chances of landing multiple hit series.
