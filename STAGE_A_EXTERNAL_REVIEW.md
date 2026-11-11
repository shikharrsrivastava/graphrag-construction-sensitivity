# Stage A independent verification report

**Review date:** 2026-11-11
**Repository checkpoint:** `9f13cbef6a952e776d4aa4a3299239fde821b4cd`
**Inputs reviewed:** `data/pilot_500_v2`, `src/evaluate_question_local.py`, `tests/test_question_local.py`, and all files in `results/question_local_v1/`.

**Public release scope:** This independent review used local raw inputs and per-question JSONL outputs. The current aggregate-only release candidate excludes those files; they remain on the research machine and can be regenerated after independently obtaining the dataset as described in `data/README.md`.

This report independently derives counts and retrieval metrics from the saved per-question JSONL files, checks candidate generation against the saved global graph and mappings, and compares independently generated summaries and paired tests to the saved CSVs. It does not treat older pooled-corpus results as findings for this run. No experiment output or evaluator source was changed for this review.

## Dataset counts and evaluated rows

The source question file contains **500 questions**: **498 with ten contexts** and **2 with two contexts**. Every question has exactly two unique gold supporting passage IDs, so there are 996 primary gold-support IDs and 4 variable-context gold-support IDs.

| Scope | Questions | Question-local TF-IDF rows | Shared-global-graph rows | Total method rows |
|---|---:|---:|---:|---:|
| Primary, ten contexts | 498 | 498 | 498 | 996 |
| Variable-context, two contexts | 2 | 2 | 2 | 4 |
| All questions | 500 | 500 | 500 | 1,000 |

The canonical rows are the union of `primary_question_level.jsonl` (996 rows) and `variable_context_question_level.jsonl` (4 rows). `shared_global_question_level.jsonl` contains 500 additional copies of the shared-global-graph rows for convenient track-specific use; it is not another 500 questions and is not added to the 1,000-row count above.

## Context-membership audit

For every ranked passage ID in each canonical per-question row, I compared the ID with that question's `context_passage_ids` from `data/pilot_500_v2/questions.json`. The audit covered all **1,000 method rows** and **9,668 ranked passage ID occurrences**.

**Out-of-context retrieved passage IDs: 0; violating rows: 0 of 1,000.** The separate 500-row shared-global file was also checked and had zero violations. Track B's candidates are filtered before scoring; its saved full ranking is contained in the question's contexts.

## TF-IDF scoping and source verification

The primary ranking is question-local in both corpus fitting and ranking. The evaluator gets IDs directly from the current question, maps only those passages to documents, fits a fresh vectorizer to those documents, transforms that question, and scores only that matrix. The relevant source excerpt is [src/evaluate_question_local.py](src/evaluate_question_local.py#L68):

```python
ids = list(map(str, question["context_passage_ids"]))
docs = [passage_text(passage_by_id[pid]) for pid in ids]
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
matrix = vectorizer.fit_transform(docs)
query_vector = vectorizer.transform([str(question["question"])])
scores = cosine_similarity(query_vector, matrix).ravel()
ranked = sorted(zip(ids, scores), key=lambda item: (-float(item[1]), item[0]))
```

`rank_local()` is called inside the loop for each question ([lines 217–219](src/evaluate_question_local.py#L217)). Thus the Track A TF-IDF vocabulary and IDF values are fitted independently for each question. Rank candidates are exactly that question's context IDs; ties are broken by passage ID.

This establishes question-local **lexical retrieval**, not question-local graph construction. The evaluator also fits a global vectorizer for the secondary Track B ([lines 205–208](src/evaluate_question_local.py#L205)); this separate fit does not feed Track A scoring.

## Independent metric recomputation

I recomputed Recall@2, @5 and @10 from each saved `ranked_passage_ids` array and its question's exact gold IDs. For every row:

\[
\mathrm{Recall@k}(q)=\frac{|G_q \cap R_{q,k}|}{|G_q|}
\]

where `G_q` is the set of that question's unique `gold_supporting_passage_ids` and `R_{q,k}` is the first `k` IDs of its saved ranking. Every question has exactly two gold supports, so the per-question denominator is **2** in both strata. The reported mean is an unweighted average over questions, not a pooled average over retrieved passages.

For the primary 498 questions, the recomputed support-hit totals are:

| Track | Gold support IDs | Hits @2 | Hits @5 | Hits @10 | Questions with all supports @2 / @5 / @10 |
|---|---:|---:|---:|---:|---:|
| Question-local TF-IDF | 996 | 655 | 833 | 996 | 189 / 343 / 498 |
| Shared global graph | 996 | 563 | 730 | 986 | 128 / 252 / 488 |

Each saved `complete_support_at_k` flag was independently verified as 1 exactly when **both** gold passage IDs were present in the top `k`; its denominator when summarized is the number of questions in the stratum (498 primary or 2 variable), not the number of support IDs. All per-row Recall@k and complete-support values matched the recomputation.

## Candidate-generation recall

The source code computes candidate-generation recall as the proportion of gold support IDs in the method's candidate set ([lines 96–110](src/evaluate_question_local.py#L96)). For Track A the candidate set is the full context set, so candidate-generation recall is 1.0 by construction because input validation requires all gold supports to occur in those contexts.

For Track B, I independently rebuilt candidates from each saved Track A ranking's first five IDs, `passage_entities.json`, the pooled entity graph's neighbors, and `entity_passages.json`; then I intersected that graph expansion plus seeds with the question's context IDs. This matched all saved Track B candidate counts and candidate-generation recalls. On the 498 primary questions, Track B generated **4,680 candidates** over all questions and captured **986 of 996** gold-support IDs, giving mean candidate-generation recall `0.9899598393574297`. Track A's candidate universe totals 4,980 context candidates and includes 996/996 supports.

The 67 passages without any global extracted-entity mapping are treated as empty graph mappings. They can still be retained as Track B lexical seeds and remain part of Track A's context corpus.

## Confidence intervals and paired comparisons

The source implementation resamples **questions** with replacement, computes 10,000 bootstrap means, and reports the percentile 2.5th and 97.5th percentiles ([bootstrap code](src/evaluate_question_local.py#L120)). The mean point estimate is the arithmetic mean of the question-level values. The confidence intervals below were recomputed directly from the JSONL values using the run seed 2027 and the code's deterministic metric/stratum seed offsets; every mean and interval matched `summary.csv`.

### All saved summary estimates and 95% CIs

| Track | Contexts | n questions | Metric | Mean | 95% CI low | 95% CI high |
|---|---:|---:|---|---:|---:|---:|
| question_local | 2 | 2 | candidate_generation_recall | 1.0 | 1.0 | 1.0 |
| question_local | 2 | 2 | recall_at_2 | 1.0 | 1.0 | 1.0 |
| question_local | 2 | 2 | recall_at_5 | 1.0 | 1.0 | 1.0 |
| question_local | 2 | 2 | recall_at_10 | 1.0 | 1.0 | 1.0 |
| question_local | 2 | 2 | complete_support_at_2 | 1.0 | 1.0 | 1.0 |
| question_local | 2 | 2 | complete_support_at_5 | 1.0 | 1.0 | 1.0 |
| question_local | 2 | 2 | complete_support_at_10 | 1.0 | 1.0 | 1.0 |
| question_local | 10 | 498 | candidate_generation_recall | 1.0 | 1.0 | 1.0 |
| question_local | 10 | 498 | recall_at_2 | 0.6576305220883534 | 0.6315261044176707 | 0.6827309236947792 |
| question_local | 10 | 498 | recall_at_5 | 0.8363453815261044 | 0.8142570281124498 | 0.858433734939759 |
| question_local | 10 | 498 | recall_at_10 | 1.0 | 1.0 | 1.0 |
| question_local | 10 | 498 | complete_support_at_2 | 0.3795180722891566 | 0.3373493975903614 | 0.42369477911646586 |
| question_local | 10 | 498 | complete_support_at_5 | 0.6887550200803213 | 0.6485943775100401 | 0.7289156626506024 |
| question_local | 10 | 498 | complete_support_at_10 | 1.0 | 1.0 | 1.0 |
| shared_global_graph | 2 | 2 | candidate_generation_recall | 1.0 | 1.0 | 1.0 |
| shared_global_graph | 2 | 2 | recall_at_2 | 1.0 | 1.0 | 1.0 |
| shared_global_graph | 2 | 2 | recall_at_5 | 1.0 | 1.0 | 1.0 |
| shared_global_graph | 2 | 2 | recall_at_10 | 1.0 | 1.0 | 1.0 |
| shared_global_graph | 2 | 2 | complete_support_at_2 | 1.0 | 1.0 | 1.0 |
| shared_global_graph | 2 | 2 | complete_support_at_5 | 1.0 | 1.0 | 1.0 |
| shared_global_graph | 2 | 2 | complete_support_at_10 | 1.0 | 1.0 | 1.0 |
| shared_global_graph | 10 | 498 | candidate_generation_recall | 0.9899598393574297 | 0.9829317269076305 | 0.9959839357429718 |
| shared_global_graph | 10 | 498 | recall_at_2 | 0.5652610441767069 | 0.5381526104417671 | 0.5913654618473896 |
| shared_global_graph | 10 | 498 | recall_at_5 | 0.7329317269076305 | 0.7078313253012049 | 0.7580321285140562 |
| shared_global_graph | 10 | 498 | recall_at_10 | 0.9899598393574297 | 0.9839357429718876 | 0.9959839357429718 |
| shared_global_graph | 10 | 498 | complete_support_at_2 | 0.2570281124497992 | 0.21887550200803213 | 0.29518072289156627 |
| shared_global_graph | 10 | 498 | complete_support_at_5 | 0.5060240963855421 | 0.46184738955823296 | 0.5502008032128514 |
| shared_global_graph | 10 | 498 | complete_support_at_10 | 0.9799196787148594 | 0.9658634538152611 | 0.9919678714859438 |

The paired comparison pairs the two methods by question, calculates the mean question-level difference (question-local minus shared-global graph), and uses a two-sided sign-flip randomization test with 20,000 Monte Carlo permutations and a plus-one p-value correction ([test code](src/evaluate_question_local.py#L129)). Holm adjustment is applied across the six outcomes separately within each context-count stratum ([adjustment code](src/evaluate_question_local.py#L141)). The independent unit is the question; no seed repeats are treated as independent rows.

### All paired comparison outputs

| Contexts | n questions | Outcome | Mean paired difference | Raw p | Holm-adjusted p |
|---:|---:|---|---:|---:|---:|
| 10 | 498 | recall_at_2 | 0.09236947791164658 | 4.999750012499375e-05 | 0.0002999850007499625 |
| 10 | 498 | recall_at_5 | 0.10341365461847389 | 4.999750012499375e-05 | 0.0002999850007499625 |
| 10 | 498 | recall_at_10 | 0.010040160642570281 | 0.0022998850057497125 | 0.004599770011499425 |
| 10 | 498 | complete_support_at_2 | 0.12248995983935743 | 4.999750012499375e-05 | 0.0002999850007499625 |
| 10 | 498 | complete_support_at_5 | 0.1827309236947791 | 4.999750012499375e-05 | 0.0002999850007499625 |
| 10 | 498 | complete_support_at_10 | 0.020080321285140562 | 0.0022998850057497125 | 0.004599770011499425 |
| 2 | 2 | recall_at_2 | 0.0 | 1.0 | 1.0 |
| 2 | 2 | recall_at_5 | 0.0 | 1.0 | 1.0 |
| 2 | 2 | recall_at_10 | 0.0 | 1.0 | 1.0 |
| 2 | 2 | complete_support_at_2 | 0.0 | 1.0 | 1.0 |
| 2 | 2 | complete_support_at_5 | 0.0 | 1.0 | 1.0 |
| 2 | 2 | complete_support_at_10 | 0.0 | 1.0 | 1.0 |

All **28 summary rows** (means and both interval bounds) and **12 paired-test rows** matched independent recomputation. The two-question stratum is separated as requested but its CIs and tests are not reliable population inference.

## Shared graph versus question-local graph construction

The shared graph was built before Stage A from the pooled 4,947-passage `data/pilot_500` corpus. It has entity surface-form nodes and same-sentence co-occurrence edges. Track B seeds it with each question's local TF-IDF top five, expands seed entities by graph neighbors, collects passages associated with those entities, retains the lexical seeds, and finally intersects that set with the question's context IDs ([candidate expansion and filtering](src/evaluate_question_local.py#L220)). The existing global `passage_entities` map has entries for 4,880/4,947 passages; absent entries are treated as empty. The graph and its edge structure can encode co-occurrences from other questions; the shared TF-IDF index is fitted across all 4,947 passages.

**Question-local graph construction is not implemented.** The only graph loaded by `run()` is the prebuilt pooled graph ([lines 209–215](src/evaluate_question_local.py#L209)). Track A does not build or use any graph; it fits a local TF-IDF model. Track B filters graph-generated passage candidates after using shared graph structure. Therefore Track B must not be interpreted as a question-local graph experiment, and it does not remove cross-question graph information merely because final candidate IDs are scoped.

## Test outcomes and experiment command

The repository full-suite command was `python -m unittest discover -v`. The logged final run passed **15 tests, 0 failures**. Test methods reported as passing were:

- Graph construction variants: `test_passage_graph_preserves_nodes_and_counts_shared_passages`; `test_passage_graph_ignores_unknown_entity_ids`.
- Question-local evaluation: `test_context_and_evidence_ids_are_validated`; `test_evidence_matching_is_by_exact_passage_id`; `test_candidate_generation_recall_uses_generated_set`; `test_graph_candidates_are_filtered_to_question_contexts`; `test_question_level_bootstrap_and_paired_test`; `test_holm_adjustment_is_monotone_in_sorted_p_values`.
- Seed stability: `test_jaccard_for_partial_overlap`; `test_jaccard_for_empty_sets_is_one`; `test_partition_similarity_is_label_invariant`.
- Structural metrics: `test_remove_count_is_at_least_one_when_fraction_positive`; `test_random_perturbation_is_reproducible`; `test_critical_perturbation_removes_expected_number`; `test_lcc_fraction_uses_original_node_denominator`.

The Stage A focused suite passed six tests. An earlier five-test focused run preceded addition of the explicit graph-context filter test. Full-suite logs show the latest full run had 15 passes; unittest reported 0.009 seconds in the last output capture. A standalone full-suite process previously took about 17.2 seconds including Matplotlib font-cache setup and emitted a non-fatal warning that its preferred cache directory was not writable; a temporary cache was used.

Successful Stage A timed command recorded in `logs/progress.md`:

```bash
/usr/bin/time -p python -c "from pathlib import Path; from src.evaluate_question_local import run; r=run(Path('data/pilot_500_v2').resolve(), Path('results/question_local_v1').resolve()); print('questions={} primary={} variable={}'.format(r['metadata']['question_count'], r['metadata']['primary_question_count'], r['metadata']['variable_context_question_count']))"
```

It completed with `questions=500 primary=498 variable=2`; recorded wall runtime **5.75 seconds** (user 5.27s, system 0.39s). A later final-verification regeneration recorded 6.58 seconds. The subsequent CSV line-ending-only regeneration was not separately timed. These are measured end-to-end command times, not estimates copied from the earlier pooled-corpus experiments.

## What the 5.75-second runtime includes

The `/usr/bin/time` measurement surrounds Python startup/imports and the call to `run()`. Inside `run()`, it includes reading the three v2 JSON files, input validation, fitting the 4,947-passage global TF-IDF index for Track B, reading the graph and entity mappings, processing all 500 questions in both tracks (including per-question Track A TF-IDF fitting and Track B graph candidate expansion/filtering/reranking), calculating 10,000 bootstrap replicates for each summary metric, running the 20,000-permutation paired tests, and writing JSONL/CSV/config outputs.

The 5.75-second command does **not** include the separately run unittest suite or manual post-run audit. The report does not claim a per-phase timing breakdown. The saved `ranking_seconds` is not the total pipeline runtime: for Track A it covers per-question fit/query/scoring, while Track B times only global-index query transformation and candidate scoring. Track B's `ranking_seconds` excludes graph expansion and one-time global index fitting, so those per-question means are not an apples-to-apples speed comparison.

## Unresolved methodological limitations

1. Track A is question-local TF-IDF retrieval only; no graph is constructed from each question's contexts. This leaves the primary graph-construction question unanswered.
2. Track B's graph, entity-to-passage mappings, and IDF values share information from the full 4,947-passage corpus. Context filtering constrains returned candidates but does not make graph construction question-local.
3. The Track A candidate-generation recall of 1.0 is tautological under the chosen definition because every eligible context is a candidate. The informative candidate-generation measurement is Track B's graph-expanded recall.
4. Track B differs from Track A in both candidate-generation behavior and TF-IDF corpus statistics. Their paired difference does not isolate the causal effect of graph use alone.
5. The entity mapping is sparse for 67 passages. Those passages have no graph entity expansion path except when retained as lexical seeds.
6. Every sampled question is hard. Conclusions apply to this fixed 498-question primary sample and its two separately analyzed exceptions, not to an unqualified HotpotQA population.
7. The variable-context analysis has only two questions. Its point estimates are descriptive; bootstrap intervals and significance tests have very low inferential value.
8. TF-IDF and the reused graph expansion are deterministic in this run. No multi-seed retrieval stability estimate was produced.
9. No answer generation, answer EM/F1, faithfulness review, BGE retrieval, or external GraphRAG baseline is included. Those results remain unexecuted.
10. The output records full rankings of eligible candidates, but does not persist a separate generated-candidate ID list. Track B candidate counts and recall are reproducible from the saved local seed rankings plus the source graph/mappings; provenance depends on those source files remaining available.

## Independent-review conclusion

The saved outputs are internally consistent for the implemented methods: row counts, context membership (zero violations), ID-based gold matching, Recall@k, complete-support flags, candidate-generation recall, bootstrap intervals, and paired comparisons were independently reproduced. The implementation does **not** establish a question-local graph result. It establishes a question-local TF-IDF baseline and a separate shared-global-graph retrieval track whose final passage candidates are filtered to each question's contexts.
