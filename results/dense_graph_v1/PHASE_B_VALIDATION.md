# Phase B validation report

**Public release scope:** This report retains aggregate findings and validation evidence. Raw per-question records, local HotpotQA inputs, graph partitions, and extraction/embedding caches used in the original independent recomputation are excluded from the aggregate-only release candidate and remain in the maintainer's local research workspace. A fresh audit requires obtaining the data and regenerating matching artifacts as described in the reproduction guide.

**Date:** 2026-11-11
**Dataset:** `data/pilot_500_v2`
**Validated scopes:** 50-question feasibility pilot and 498-question primary evaluation
**Status:** retrieval and local graph construction executed; raw-output and statistical recomputation passed.

## Validation conclusion

The canonical 50-question pilot (`pilot50_final`) and 498-question primary run (`primary498_final`) each passed an independent audit that reconstructs local graph edges, graph candidates, BGE rankings, passage/evidence metrics, graph structure, bootstrap intervals, paired tests, Holm adjustment, and retrieval-overlap summaries from the saved raw rows and caches. The independent implementation is `src/phase_b_independent_audit.py`; its audit records are `phase_b_validation_audits/pilot50_independent.json` and `phase_b_validation_audits/primary498_independent.json`.

| Check | Pilot | Primary |
|---|---:|---:|
| Questions | 50 | 498 |
| Methods | 7 | 7 |
| Raw per-question rows | 350 | 3,486 |
| Rows per method | 50 | 498 |
| Candidate IDs outside the question's ten contexts | 0 | 0 |
| Ranked IDs outside the question's ten contexts | 0 | 0 |
| Graph-node scope violations | 0 | 0 |
| Graph-edge reconstruction mismatches | 0 | 0 |
| Candidate/ranking reconstruction mismatches | 0 | 0 |
| Recall / complete-support / candidate-recall mismatches | 0 | 0 |
| Gold denominator or context-record mismatches | 0 | 0 |
| Graph statistic mismatches | 0 | 0 |
| Saved summary/statistics CSVs matching independent recomputation | 5/5 | 5/5 |

The graph audit reconstructs each entity graph from only the ten passage IDs assigned to that question and compares the complete node and weighted-edge sets. It independently constructs the chunk kNN graph from only those ten passage vectors. It then rebuilds the candidate pools from the question's top-five BGE seeds and the locally reconstructed graph. No other question's candidate IDs, graph nodes, or edges enter these computations. Entity names may naturally occur in more than one question; that is shared vocabulary, not imported graph structure.

## Inputs, split, and evidence rules

The v2 dataset contains 500 questions and 4,947 distinct corpus passages. There are 498 ten-context primary questions and two two-context exceptions; all questions have gold support passage IDs in their own contexts. The primary evaluation has 404 bridge and 94 comparison questions. The two variable-context questions are excluded from both Phase B runs and were not supplemented with unrelated passages. The pilot sampled 41 bridge and 9 comparison questions using proportional, deterministic stratification (seed 2027).

Each raw record stores exact question-local context IDs, gold support IDs, generated candidate IDs, ranked IDs, and a gold support count. Passage evidence is matched by exact ID. Per-question Recall@k is `number of exact gold passage IDs in the first k / number of gold support IDs`. Complete support retrieval is a per-question indicator that every gold support ID appears in the first k. Candidate-generation recall uses the same gold support denominator and the graph-generated candidate set before final ranking. The vanilla methods have all ten contexts as candidates, so their candidate-generation recall is 1.0 by definition after input validation.

## Seven evaluated methods

The seven saved configurations are:

1. **Vanilla TF-IDF** (`vanilla_tfidf`): reuse the Stage A local TF-IDF function, fitting a separate vectorizer on each question's own ten passage titles/texts and ranking only those passages.
2. **Vanilla BGE** (`vanilla_bge`): rank all ten contexts by query-to-passage dense similarity.
3. **Sentence co-occurrence, frequency weighted** (`entity_sentence_frequency_weighted`): entity surface-form nodes; add an edge for entities co-occurring in one sentence; edge weight counts sentence co-occurrences across that question's ten passages.
4. **Passage co-occurrence, frequency weighted** (`entity_passage_frequency_weighted`): entity surface-form nodes; add an edge when entities occur in the same passage; edge weight counts co-occurring passages.
5. **Sentence co-occurrence, binary** (`entity_sentence_binary`): the sentence-frequency edge set with present-edge weight set to one.
6. **Sentence co-occurrence, strongest-half sparsification** (`entity_sentence_top50_sparsified`): retain the strongest ceiling(50%) of the sentence-frequency edges. Equal weights are resolved by canonical lexical endpoint order. Nodes are retained, so edge pruning can create isolates.
7. **BGE chunk similarity graph** (`chunk_bge_knn2`): one node per context passage/chunk; an undirected union of each node's two nearest neighbors by BGE cosine similarity.

For the four entity methods, the top five vanilla BGE passages seed expansion. Entities in the seed passages are expanded through their three highest-weight neighboring entities, and passages containing those entities are added. For the chunk graph, the top five BGE seed nodes expand to direct kNN neighbors. Seeds remain candidates. BGE scores rerank the resulting candidate pool. Therefore, these graph-assisted methods test graph-based candidate generation with a fixed BGE scoring order; they do not test graph-aware score fusion or a graph traversal reranker.

### Requested configurations that remain missing

- Binary weighting is tested only for sentence-level edges. A passage-level binary graph was not included.
- Only one entity sparsification rule was tested: strongest 50% of sentence-frequency edges. There is no threshold/top-k sweep, random-pruning control, or alternate retained fraction.
- Sentence *co-occurrence* is a construction unit, but there is no sentence-node graph. Entity and chunk are the only graph node types.
- There is no entity semantic-similarity graph, entity-resolution/alias ablation, bipartite entity-passage graph, or validated typed-relation extraction.
- There is no question-local community detection condition in Phase B. The graph methods here perform candidate expansion, not community-based retrieval.
- Contriever and an established GraphRAG package (LightRAG or Microsoft GraphRAG) are not evaluated.
- Answer generation, exact match, token F1, evidence faithfulness, and manual faithfulness review are not run.

Typed relationships were deferred because the available spaCy NER output does not provide validated relation types; no typed extraction method was developed and manually validated for this study.

## BGE identity and similarity verification

| Property | Verified value |
|---|---|
| Model | `BAAI/bge-small-en-v1.5` |
| Pinned Hugging Face revision | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Embedding dimension | 384 |
| Query prefix | `Represent this sentence for searching relevant passages: ` |
| Passage prefix | None |
| Normalization | `normalize_embeddings=True` for queries and passages; saved vectors are L2 normalized |
| Measured maximum norm error | `1.1921e-7` from 1.0, in both audits |
| Similarity | Dot product of L2-normalized vectors, equivalent to cosine similarity |
| Device | CPU; run config records CUDA and MPS unavailable |

The pinned revision appears in both the primary run configuration and the cache manifests. The primary cache has 4,943 passage vectors and the pilot has 500 unique passage vectors; the query cache covers the selected query IDs. The chunk graph computes its kNN edges from cosine-equivalent dot products of those same normalized BGE passage embeddings. BGE embeds passage text independently; it does not fit a corpus-wide statistic or graph.

## Question-local extraction and graph leakage

Each unique passage is processed as its own sequence of sentences by spaCy `en_core_web_sm` 3.8.0. Normalized entity surface forms are extracted sentence-by-sentence; no other passage is passed into the entity extractor. The legacy `passage_entities.json` has no recorded extractor/model revision, so it was not used to build these graphs. As a provenance diagnostic, all 4,876 legacy mapping entries overlapping the primary context corpus matched the current extraction; 67 context passages had no legacy mapping. The pilot overlap check was 498/498. A match to a prior automatic mapping does not establish entity precision or recall.

For each question, the audit reconstructed entity nodes as exactly the entities in that question's own context passages, reconstructed edge counts from either sentences or passages, applied binary weights or the strongest-half pruning rule as applicable, then rebuilt candidate passage IDs. For chunk graphs it reconstructed the ten context nodes, cosine similarities, union-kNN edges, and seed-neighbor candidates. Across all 498 questions, raw graph construction scope was `question_contexts_only`; all node, edge, and candidate reconstructions matched. Every candidate and every ranked passage belonged to its question's own ten-context set. **Observed cross-question passage/graph leakage violations: zero.**

## Independently recomputed retrieval results

Each cell is the question-level mean with a 95% percentile bootstrap interval in brackets. The intervals resample questions, not passages or graph nodes. `Candidate` means candidate-generation recall; `Complete@k` is the rate of questions with all gold supports retrieved by k.

### 50-question pilot

| Method | Candidate | Recall@2 | Recall@5 | Recall@10 | Complete@2 | Complete@5 | Complete@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla TF-IDF | 1.000 [1.000, 1.000] | 0.640 [0.560, 0.720] | 0.870 [0.810, 0.930] | 1.000 [1.000, 1.000] | 0.340 [0.220, 0.480] | 0.740 [0.620, 0.860] | 1.000 [1.000, 1.000] |
| Vanilla BGE | 1.000 [1.000, 1.000] | 0.670 [0.580, 0.760] | 0.870 [0.800, 0.930] | 1.000 [1.000, 1.000] | 0.440 [0.300, 0.580] | 0.760 [0.640, 0.860] | 1.000 [1.000, 1.000] |
| Entity sentence frequency | 0.970 [0.930, 1.000] | 0.670 [0.580, 0.760] | 0.870 [0.800, 0.930] | 0.970 [0.930, 1.000] | 0.440 [0.300, 0.580] | 0.760 [0.640, 0.880] | 0.940 [0.860, 1.000] |
| Entity passage frequency | 0.970 [0.930, 1.000] | 0.670 [0.580, 0.760] | 0.870 [0.800, 0.930] | 0.970 [0.930, 1.000] | 0.440 [0.300, 0.580] | 0.760 [0.640, 0.880] | 0.940 [0.860, 1.000] |
| Entity sentence binary | 0.970 [0.930, 1.000] | 0.670 [0.580, 0.760] | 0.870 [0.800, 0.930] | 0.970 [0.930, 1.000] | 0.440 [0.300, 0.580] | 0.760 [0.640, 0.880] | 0.940 [0.860, 1.000] |
| Entity sentence strongest half | 0.970 [0.930, 1.000] | 0.670 [0.580, 0.760] | 0.870 [0.800, 0.930] | 0.970 [0.930, 1.000] | 0.440 [0.300, 0.580] | 0.760 [0.640, 0.880] | 0.940 [0.860, 1.000] |
| BGE chunk union-kNN k=2 | 1.000 [1.000, 1.000] | 0.670 [0.580, 0.760] | 0.870 [0.800, 0.930] | 1.000 [1.000, 1.000] | 0.440 [0.300, 0.580] | 0.760 [0.640, 0.880] | 1.000 [1.000, 1.000] |

### Full 498-question primary run

| Method | Candidate | Recall@2 | Recall@5 | Recall@10 | Complete@2 | Complete@5 | Complete@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla TF-IDF | 1.000 [1.000, 1.000] | 0.658 [0.633, 0.683] | 0.836 [0.814, 0.858] | 1.000 [1.000, 1.000] | 0.380 [0.337, 0.422] | 0.689 [0.647, 0.729] | 1.000 [1.000, 1.000] |
| Vanilla BGE | 1.000 [1.000, 1.000] | 0.748 [0.723, 0.773] | 0.906 [0.888, 0.923] | 1.000 [1.000, 1.000] | 0.528 [0.484, 0.570] | 0.815 [0.781, 0.847] | 1.000 [1.000, 1.000] |
| Entity sentence frequency | 0.984 [0.976, 0.991] | 0.748 [0.723, 0.772] | 0.906 [0.888, 0.923] | 0.984 [0.976, 0.991] | 0.528 [0.484, 0.572] | 0.815 [0.781, 0.847] | 0.968 [0.952, 0.982] |
| Entity passage frequency | 0.984 [0.976, 0.991] | 0.748 [0.723, 0.773] | 0.906 [0.888, 0.923] | 0.984 [0.976, 0.991] | 0.528 [0.484, 0.572] | 0.815 [0.779, 0.849] | 0.968 [0.952, 0.982] |
| Entity sentence binary | 0.984 [0.976, 0.991] | 0.748 [0.724, 0.772] | 0.906 [0.888, 0.923] | 0.984 [0.976, 0.991] | 0.528 [0.484, 0.570] | 0.815 [0.779, 0.849] | 0.968 [0.952, 0.982] |
| Entity sentence strongest half | 0.984 [0.976, 0.991] | 0.748 [0.723, 0.773] | 0.906 [0.888, 0.924] | 0.984 [0.976, 0.991] | 0.528 [0.484, 0.572] | 0.815 [0.779, 0.849] | 0.968 [0.952, 0.982] |
| BGE chunk union-kNN k=2 | 0.997 [0.993, 1.000] | 0.748 [0.724, 0.772] | 0.906 [0.888, 0.923] | 0.997 [0.993, 1.000] | 0.528 [0.484, 0.572] | 0.815 [0.781, 0.849] | 0.994 [0.986, 1.000] |

These values were recalculated from each row's saved gold IDs, candidate IDs, and ranked IDs. BGE rankings were also reconstructed from cached question/passage vectors and matched; TF-IDF metrics were independently recomputed from its saved local rankings. The full, unrounded means and intervals are independently verified against `primary498_final/retrieval_summary.csv`. Graph topology means and intervals are in `graph_summary.csv`.

### Pilot-to-primary consistency

The pilot and full sample have the same qualitative ordering: BGE exceeded local TF-IDF at the early cutoffs; all graph-assisted methods matched BGE's Recall@2 and @5 because they use BGE seed scores and BGE reranking; entity graph candidate filtering reduced Recall@10 and complete support@10; chunk kNN retained more gold evidence than entity expansion. The primary sample estimates are more precise. BGE Recall@2 was 0.670 in the pilot and 0.748 in the primary set; BGE Recall@5 was 0.870 and 0.906. Entity candidate recall was 0.970 and 0.984; chunk candidate recall was 1.000 and 0.997. The changes are compatible with sampling variation and do not indicate a pipeline contradiction. Both sets have zero independent-audit failures.

## Bootstrap intervals and paired comparisons

Confidence intervals were independently recomputed with 10,000 bootstrap resamples of questions and percentile 2.5/97.5 bounds. The audit recalculated every retrieval, graph-statistic, and pairwise retrieval-overlap summary row, and all matched. The saved rows retain one outcome per question and configuration; no seed pairs or passages are treated as independent units.

Paired differences were recomputed as per-question method differences, tested using a two-sided sign-flip Monte Carlo test (20,000 draws, plus-one p correction), and Holm adjusted across 69 predeclared method/outcome comparisons. The exact results are in `primary498_final/paired_comparisons.csv` and pilot results in `pilot50_final/paired_comparisons.csv`; both matched the independent implementation.

| Comparison, direction | Outcome | Mean difference | Holm p | Interpretation |
|---|---|---:|---:|---|
| BGE − TF-IDF | Recall@2 | +0.09036 | 0.00345 | BGE higher after correction |
| BGE − TF-IDF | Recall@5 | +0.06928 | 0.00345 | BGE higher after correction |
| BGE − TF-IDF | Complete@2 | +0.14859 | 0.00345 | BGE higher after correction |
| BGE − TF-IDF | Complete@5 | +0.12651 | 0.00345 | BGE higher after correction |
| Entity sentence frequency − vanilla BGE | Candidate recall / Recall@10 | −0.01606 | 0.00345 | Entity expansion misses more gold support |
| Entity sentence frequency − vanilla BGE | Complete@10 | −0.03213 | 0.00580 | Lower complete support after correction |
| Chunk graph − entity sentence frequency | Candidate recall / Recall@10 | +0.01305 | 0.23714 / 0.22524 | Positive raw tests, not significant after Holm |
| Chunk graph − entity sentence frequency | Complete@10 | +0.02610 | 0.23659 | Not significant after Holm |
| Passage frequency, binary, or strongest-half graph − sentence-frequency graph | All tested retrieval outcomes | 0.00000 | 1.00000 | No paired outcome differences in this saved run |

There were 16 Holm-significant rows in the 498-question 69-test family. The 50-question pilot had zero Holm-significant rows. The latter is a pilot feasibility result, not evidence of equivalence. At Recall@10, vanilla TF-IDF and BGE are both 1.0 because all ten passages are ranked and all gold evidence is in context; entity/chunk results differ due to candidate filtering.

## Graph structure and efficiency

Primary mean structures from `primary498_final/graph_summary.csv`: sentence-frequency entity graph, 74.41 nodes / 156.22 edges / density 0.05904 / 15.51 components; passage-frequency, 74.41 / 499.22 / 0.17273 / 3.17; strongest-half sentence graph, 74.41 / 78.35 / 0.02967 / 29.45; chunk graph, 10 / 14.50 / 0.32222 / 1.10. The independent audit recomputed connected components, isolates and density from serialized nodes and edges with no mismatches.

The 498 run reports 133.51 seconds and peak process RSS of 1,231.2 MB. The measured interval begins after the input JSON files have been loaded and input validation has run. It includes fresh extraction of 4,443 uncached passages (50.44 s), local BGE model loading (0.12 s), 4,443 new passage embeddings (71.25 s), 448 new query embeddings (1.20 s), all 3,486 question-method evaluations, graph construction, question-level bootstrap summaries, the initial paired/stability summaries, and output writes. Five hundred passage and 50 query vectors were reused from the pilot cache. After adding direct graph-construction paired contrasts, the final 69-row-family paired table was recomputed separately from saved raw rows in 3.57 seconds.

The 133.51 seconds exclude JSON input loading/validation, model download, the previous 50-passage extraction/model cache warm-up, and the separate 3.57-second paired-table recomputation. `query_retrieval_seconds` is not end-to-end: it begins with cached vectors and excludes embedding generation and graph construction. Graph build, candidate generation, dense query scoring, reranking, and local TF-IDF fit/rank have separate fields in raw rows. Peak RSS is the process high-water mark, not total machine memory. Timing is specific to this CPU run and warm local model files.

## Test suite

The actual command was `.venv/bin/python -m unittest discover -v`. Output showed **24 tests, 24 passed, 0 failures**, with unittest-reported runtime 0.043 seconds. All listed methods returned `ok`:

- Dense/graph pilot (9): `test_binary_weights_and_top_half_sparsification_are_controlled`; `test_chunk_graph_nodes_and_candidates_are_question_scoped`; `test_entity_graph_edges_are_question_local_and_window_changes_frequency`; `test_evidence_metrics_use_exact_ids_and_support_denominator`; `test_paired_comparisons_include_direct_graph_construction_contrasts`; `test_question_level_bootstrap_sign_flip_and_holm`; `test_sparsification_ties_do_not_depend_on_context_or_edge_iteration_order`; `test_stratified_pilot_selection_is_reproducible_and_primary_only`; `test_weighted_neighbor_expansion_and_context_guard`.
- Graph-construction variants (2): `test_passage_graph_ignores_unknown_entity_ids`; `test_passage_graph_preserves_nodes_and_counts_shared_passages`.
- Question-local Stage A (6): `test_candidate_generation_recall_uses_generated_set`; `test_context_and_evidence_ids_are_validated`; `test_evidence_matching_is_by_exact_passage_id`; `test_graph_candidates_are_filtered_to_question_contexts`; `test_holm_adjustment_is_monotone_in_sorted_p_values`; `test_question_level_bootstrap_and_paired_test`.
- Seed stability (3): `test_jaccard_for_empty_sets_is_one`; `test_jaccard_for_partial_overlap`; `test_partition_similarity_is_label_invariant`.
- Structural metrics (4): `test_critical_perturbation_removes_expected_number`; `test_lcc_fraction_uses_original_node_denominator`; `test_random_perturbation_is_reproducible`; `test_remove_count_is_at_least_one_when_fraction_positive`.

The test process printed a non-fatal Matplotlib cache permission warning and used a temporary cache. No test failed. The independent audit itself also completed for both final run scopes and is stored separately from the experiment's earlier audit records.

## Output inventory and status

Use only these validated result directories for findings:

- `pilot50_final/`: canonical stratified pilot, raw JSONL, retrieval/graph/paired/stability CSVs, config, validation checks, prior audit record.
- `primary498_final/`: canonical full primary, corresponding 3,486 raw rows and summaries/config.
- `phase_b_validation_audits/`: new independent audit records for both canonical runs.
- `cache/`: passage-local extraction and pinned BGE embedding/query caches.

Other existing files were preserved and inspected, but they are not canonical results. `pilot50/` has a stale `failure.json` from its first offline model-load attempt and its later run configuration says model revision `unknown`; do not use it. `pilot50_verified/` uses the pinned BGE revision but its old audit recorded 50 sparsified-edge mismatches because the tie-break was not deterministic; it was superseded by the fixed and audited pilot. `cache/embeddings/unknown/` is an unpinned earlier pilot cache. The fixed `pilot50_final` and `primary498_final` use the pinned-revision cache.

## Why repeated-seed stability and extraction quality are unmeasured

No same-configuration repeated-seed experiment was run. Passage-local spaCy extraction, BGE inference on CPU, graph construction, deterministic lexical tie-breaking, and ranking are fixed for a given input. The seed 2027 controls only the selected pilot sample and bootstrap resampling; it is not an algorithmic graph/retrieval seed. The saved top-k Jaccard table compares different configurations, not repeated seeds. Repeated-seed robustness remains an outstanding experiment and should be designed around a real stochastic choice rather than duplicated deterministic runs.

Entity extraction has not been evaluated against manual entity labels. Matching 4,876 legacy automatic mappings is only a consistency/provenance check; those mappings are not independent ground truth. No entity precision, recall, alias-resolution accuracy, or human review was measured. A labeled sample and a defined annotation protocol would be needed to quantify extraction quality.

## Reproduction commands

```text
.venv/bin/python -m src.evaluate_dense_graph --scope pilot50 --run-name pilot50_final --output-dir results/dense_graph_v1
.venv/bin/python -m src.evaluate_dense_graph --scope primary498 --run-name primary498_final --output-dir results/dense_graph_v1
.venv/bin/python -m src.phase_b_independent_audit results/dense_graph_v1/pilot50_final --output results/dense_graph_v1/phase_b_validation_audits/pilot50_independent.json
.venv/bin/python -m src.phase_b_independent_audit results/dense_graph_v1/primary498_final --output results/dense_graph_v1/phase_b_validation_audits/primary498_independent.json
.venv/bin/python -m unittest discover -v
```

The evaluator refuses to overwrite non-empty raw output directories. The report does not claim generated-answer quality, faithfulness, or a completed GraphRAG system.
