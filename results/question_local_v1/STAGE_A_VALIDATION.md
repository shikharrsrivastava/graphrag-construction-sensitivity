# Stage A final validation

**Validation date:** 2026-11-11
**Dataset:** `data/pilot_500_v2`
**Status:** Evaluation executed; output integrity checks passed; full test suite passed.

## Scope and implementation findings

Stage A evaluates TF-IDF evidence retrieval on the fixed 500-question v2 sample. There are 498 questions with ten contexts and two questions with two contexts. The ten-context questions are the primary analysis. The two exceptions are retained and reported separately; no passages were added.

**Track A — question-local TF-IDF.** For every question, a new `TfidfVectorizer` is fitted using only that question's context passage title and text. It uses English stop words, unigram and bigram features, and sublinear term frequency. The query is transformed by that question-specific vectorizer and cosine scores rank only those contexts. Ties use passage ID as a stable secondary sort. This is genuinely question-local corpus fitting and ranking. It does not construct a graph per question; Track A is the lexical retrieval baseline.

**Track B — shared global graph with scoped candidates.** The existing `data/pilot_500/entity_graph.graphml` is a pooled, same-sentence entity co-occurrence graph built across the 4,947-passage corpus. For each question, Track A's top five lexical results seed graph expansion through the global `passage_entities`, graph neighbors, and `entity_passages` mappings. The seed and expanded passage IDs are then intersected with that question's context IDs before TF-IDF ranking. Reranking uses one TF-IDF index fitted across all 4,947 v2 passages. Therefore candidates ranked are context-filtered, but graph structure, entity mappings, and IDF statistics carry cross-question information. This track is not question-local graph construction. **Question-local graph construction was not implemented in Stage A.**

The global entity mapping contains entries for 4,880/4,947 passages. The remaining 67 have no extracted entities and are treated as empty graph mappings; their passages can still enter via lexical seeds. Existing datasets and historical results were not modified.

## Evidence matching and metric definitions

- Inputs are checked for unique question and passage IDs, agreement between `questions.json` and `question_contexts.json`, known context IDs, agreement between gold evidence mappings, and presence of every gold support in its question's contexts.
- Gold support is matched by exact `passage_id`, not title or text similarity. Each question's denominator is the count of its unique `gold_supporting_passage_ids`.
- `Recall@k` is the count of exact gold support IDs among the top `k` ranked passage IDs divided by that question's gold support count. The per-question recalls are averaged across questions.
- `complete_support_at_k` is 1 only when every gold support ID occurs in the top `k` results.
- Track A candidate-generation recall is gold support present in its full eligible context candidate universe. It is 1.0 by construction after validated v2 evidence coverage.
- Track B candidate-generation recall is gold support present in the graph-expanded candidates after filtering to the question's contexts, divided by gold support count. The saved candidate count and candidate-generation recall were independently reconstructed from the global graph, mappings, and Track A top-five seeds.
- Bootstrap confidence intervals resample questions with replacement (10,000 replicates; percentile 2.5th and 97.5th percentiles). The question is the resampling unit; there are no seed replicates in this experiment.
- Paired comparisons use one Track A and one Track B row per question. The statistic is the mean paired difference (Track A minus Track B); two-sided sign-flip randomization uses 20,000 Monte Carlo permutations and the plus-one p-value correction. Holm correction is applied across the six retrieval outcomes separately within each context-count stratum. The n=2 variable-context tests are descriptive and have little inferential power.

## Complete summary results

The values below reproduce `summary.csv`; CIs are question-level 95% bootstrap intervals.

| Track | Contexts | Questions | Metric | Mean | 95% CI |
|---|---:|---:|---|---:|---:|
| question_local | 2 | 2 | candidate_generation_recall | 1.0 | [1.0, 1.0] |
| question_local | 2 | 2 | recall_at_2 | 1.0 | [1.0, 1.0] |
| question_local | 2 | 2 | recall_at_5 | 1.0 | [1.0, 1.0] |
| question_local | 2 | 2 | recall_at_10 | 1.0 | [1.0, 1.0] |
| question_local | 2 | 2 | complete_support_at_2 | 1.0 | [1.0, 1.0] |
| question_local | 2 | 2 | complete_support_at_5 | 1.0 | [1.0, 1.0] |
| question_local | 2 | 2 | complete_support_at_10 | 1.0 | [1.0, 1.0] |
| question_local | 10 | 498 | candidate_generation_recall | 1.0 | [1.0, 1.0] |
| question_local | 10 | 498 | recall_at_2 | 0.6576305220883534 | [0.6315261044176707, 0.6827309236947792] |
| question_local | 10 | 498 | recall_at_5 | 0.8363453815261044 | [0.8142570281124498, 0.858433734939759] |
| question_local | 10 | 498 | recall_at_10 | 1.0 | [1.0, 1.0] |
| question_local | 10 | 498 | complete_support_at_2 | 0.3795180722891566 | [0.3373493975903614, 0.42369477911646586] |
| question_local | 10 | 498 | complete_support_at_5 | 0.6887550200803213 | [0.6485943775100401, 0.7289156626506024] |
| question_local | 10 | 498 | complete_support_at_10 | 1.0 | [1.0, 1.0] |
| shared_global_graph | 2 | 2 | candidate_generation_recall | 1.0 | [1.0, 1.0] |
| shared_global_graph | 2 | 2 | recall_at_2 | 1.0 | [1.0, 1.0] |
| shared_global_graph | 2 | 2 | recall_at_5 | 1.0 | [1.0, 1.0] |
| shared_global_graph | 2 | 2 | recall_at_10 | 1.0 | [1.0, 1.0] |
| shared_global_graph | 2 | 2 | complete_support_at_2 | 1.0 | [1.0, 1.0] |
| shared_global_graph | 2 | 2 | complete_support_at_5 | 1.0 | [1.0, 1.0] |
| shared_global_graph | 2 | 2 | complete_support_at_10 | 1.0 | [1.0, 1.0] |
| shared_global_graph | 10 | 498 | candidate_generation_recall | 0.9899598393574297 | [0.9829317269076305, 0.9959839357429718] |
| shared_global_graph | 10 | 498 | recall_at_2 | 0.5652610441767069 | [0.5381526104417671, 0.5913654618473896] |
| shared_global_graph | 10 | 498 | recall_at_5 | 0.7329317269076305 | [0.7078313253012049, 0.7580321285140562] |
| shared_global_graph | 10 | 498 | recall_at_10 | 0.9899598393574297 | [0.9839357429718876, 0.9959839357429718] |
| shared_global_graph | 10 | 498 | complete_support_at_2 | 0.2570281124497992 | [0.21887550200803213, 0.29518072289156627] |
| shared_global_graph | 10 | 498 | complete_support_at_5 | 0.5060240963855421 | [0.46184738955823296, 0.5502008032128514] |
| shared_global_graph | 10 | 498 | complete_support_at_10 | 0.9799196787148594 | [0.9658634538152611, 0.9919678714859438] |

## Paired comparisons

The 498-question primary stratum has mean paired differences (question-local minus shared-global graph) of 0.09236947791164658 for Recall@2, 0.10341365461847389 for Recall@5, and 0.010040160642570281 for Recall@10. Corresponding Holm-adjusted p-values are 0.0002999850007499625, 0.0002999850007499625, and 0.004599770011499425. Complete-support differences are 0.12248995983935743 (@2), 0.1827309236947791 (@5), and 0.020080321285140562 (@10); adjusted p-values are 0.0002999850007499625, 0.0002999850007499625, and 0.004599770011499425. These comparisons describe this fixed sample and this Track B setup.

For the two variable-context questions, all six paired mean differences are 0.0 and all raw and adjusted p-values are 1.0. Treat these as descriptive only.

`paired_comparisons.csv` is the exact per-outcome record. The full per-question JSONL files preserve all ranked passage IDs, gold IDs, counts, metrics, and question IDs.

## Context integrity and independent recomputation

The final audit read all three JSONL outputs and checked:

- 498 unique primary question IDs in each track and the two variable-context question IDs in their separate file.
- Every retrieved passage ID in every stored row belongs to that question's own context IDs, including Track B rows.
- Every stored Recall@2/5/10 and complete-support indicator matches a direct recomputation from the stored ranking and source gold IDs, using the number of gold IDs as denominator.
- Every Track B candidate count and candidate-generation recall matches an independent reconstruction from the saved Track A top-five seed ranking, global graph, entity mappings, and question context filter.
- All 28 summary rows (means, bootstrap intervals, candidate counts, ranking-time averages) and all 12 paired-comparison rows reproduce from the JSONL data and configured seed.

## Timing and candidate-metric labels

The final regenerated evaluation command completed in **6.58 seconds wall time** (5.59 seconds user, 0.38 seconds system). This is total run time, including input loading, global index fitting, graph loading, per-question graph candidate expansion, bootstrapping, paired tests, and output writing.

The summary's `mean_ranking_seconds` is narrower and not a like-for-like end-to-end efficiency comparison: Track A includes per-question TF-IDF fit, query transform, and scoring; Track B includes query transform and candidate scoring only. It excludes one-time global index fitting, graph loading, and graph expansion. These boundaries are recorded in `run_config.json`; do not interpret Track B's smaller per-question ranking time as total pipeline speed.

Candidate-generation recall is named in the outputs and defined separately for each track. Track A's value is necessarily 1.0 because its full context universe is its candidate set. Track B's value measures graph-expanded candidate recall after question-context filtering.

## Commands and validation outcomes

Final evaluation (the output directory already contained this run's artifacts, so the `run()` entry point was used directly to regenerate them without deleting the directory):

```bash
/usr/bin/time -p python -c "from pathlib import Path; from src.evaluate_question_local import run; r=run(Path('data/pilot_500_v2').resolve(), Path('results/question_local_v1').resolve()); print('questions={} primary={} variable={}'.format(r['metadata']['question_count'], r['metadata']['primary_question_count'], r['metadata']['variable_context_question_count']))"
```

Outcome: `questions=500 primary=498 variable=2`; real 6.58s, user 5.59s, sys 0.38s.

Final full test suite:

```bash
python -m unittest discover -v
```

Outcome: **15 tests passed**, 0.008 seconds reported by unittest; 17.2 seconds process wall time including Matplotlib font-cache initialization. Matplotlib emitted a non-fatal warning that `[USER_HOME]/.matplotlib` is not writable and used a temporary cache.

Independent artifact audit: all output counts, context memberships, gold denominators, per-question recall/completeness, reconstructed Track B candidate metrics, all 28 summary rows, and all 12 paired comparisons passed. `git diff --check` passed before creation of this report.

## Limitations and deferred work

- The sample is all hard; findings are limited to these sampled questions.
- No graph was constructed separately from each question's contexts. Track A evaluates question-local TF-IDF only; Track B intentionally exposes shared graph and global IDF information.
- The global graph's 67 missing passage-entity mappings are treated as empty; this is documented and does not remove those passages from Track A.
- No seed-based retrieval stability analysis was run because this Stage A TF-IDF/graph setup is deterministic and has no seed replicates. The experiment also does not estimate multi-seed graph construction variation.
- Per-question timing is not end-to-end comparable across tracks; use total run time for this execution and the timing definitions above for any component-level interpretation.
- BGE, answer generation, answer EM/F1/faithfulness, other baselines, and graph-construction ablations remain unexecuted. Do not begin them as part of this validation checkpoint.
