# Phase B pilot validation and primary evaluation

Date: 2026-11-11  
Dataset: `data/pilot_500_v2`  
Primary unit: question (498 ten-context questions); the two variable-context questions were not included in this phase.

## Validation decision

The corrected 50-question pilot passed the full saved-row reconstruction audit and is suitable to scale. The audit recomputed the TF-IDF and BGE rankings, every graph candidate set, graph edges and weights, gold denominators, Recall@k, complete-support retrieval, and candidate-generation recall. Across 350 rows it found zero context candidate violations, zero context ranking violations, zero graph-node scope violations, zero graph edge mismatches, zero ranking mismatches, zero metric mismatches, and zero gold-denominator mismatches. The audit is saved in `pilot50_final/independent_audit.json`.

The 498-question run was then completed and independently reconstructed using the same audit. It passed all checks across 3,486 rows with the same zero-violation results. Its audit is in `primary498_final/independent_audit.json`.

## Methods

The pilot selected 50 of the 498 ten-context questions by deterministic proportional stratified sampling over question type (seed 2027): 41 bridge and 9 comparison questions. Each question's ten contexts were its entire eligible passage universe. Gold support was matched by exact passage ID, with the count of gold supporting passage IDs as the denominator. The primary evaluation used these seven configurations:

1. `vanilla_tfidf`: Stage A TF-IDF refit and ranked within each question's contexts.
2. `vanilla_bge`: BGE-small-en-v1.5 query-to-passage cosine ranking of all ten contexts.
3. `entity_sentence_frequency_weighted`: entity graph from within-sentence co-occurrence counts.
4. `entity_passage_frequency_weighted`: entity graph from within-passage co-occurrence counts.
5. `entity_sentence_binary`: sentence-level co-occurrence edges set to one when present.
6. `entity_sentence_top50_sparsified`: sentence-frequency graph retaining the strongest ceiling(50%) of edges; ties use canonical lexical endpoint order.
7. `chunk_bge_knn2`: question-local undirected union-kNN graph over the ten context passage chunks using BGE cosine similarity, k=2.

For graph-assisted retrieval, the top five BGE-ranked passages seed graph expansion. Entity graph expansion adds passages containing seed entities and the three highest-weight neighboring entities per seed entity; the chunk graph adds direct kNN neighbors of seed chunks. The BGE scores then rerank only the generated candidate set. Vanilla methods rank all ten contexts. Thus, graph edges affect candidate generation; they do not change the BGE scoring function or within-candidate score order.

Entity extraction was performed one passage and sentence at a time with spaCy `en_core_web_sm` 3.8.0. The legacy passage entity map's extraction revision/provenance was not recorded, so it was not used to construct graphs. For the primary sample, all 4,876 legacy mapping entries that overlapped the 4,943 unique context passages matched the fresh extraction exactly; 67 context passages had no legacy mapping. The pilot's 498/498 overlapping mappings matched. Graphs were built separately for each question from its own ten contexts. Passage extraction records and BGE vectors were cached by source text hash and model revision.

BGE model: `BAAI/bge-small-en-v1.5`, Hugging Face revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, CPU, 384-dimensional normalized vectors, maximum sequence length 512. Queries used the model-card search prefix `Represent this sentence for searching relevant passages: `; passage text had no prefix. [Model card](https://huggingface.co/BAAI/bge-small-en-v1.5).

Each reported mean has a 95% percentile interval from 10,000 question-level bootstrap resamples. Paired tests are two-sided sign-flip Monte Carlo tests with 20,000 permutations and plus-one p-value correction. Holm correction covers all 69 predeclared baseline and direct graph-construction method-outcome contrasts. Confidence intervals and every paired comparison are in each run's CSV files. Retrieval stability is reported as per-question top-k Jaccard overlap between different configurations; this is cross-configuration overlap, not repeat-seed stability. No repeated retrieval seeds were run.

## Pilot results (50 questions)

Values below are means. The complete 95% question bootstrap intervals for all seven outcomes per configuration are in `pilot50_final/retrieval_summary.csv`.

| Configuration | Candidate recall | Recall@2 | Recall@5 | Recall@10 | Complete support@5 | Complete support@10 |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla TF-IDF | 1.000 | 0.640 | 0.870 | 1.000 | 0.740 | 1.000 |
| Vanilla BGE | 1.000 | 0.670 | 0.870 | 1.000 | 0.760 | 1.000 |
| Entity sentence, frequency | 0.970 | 0.670 | 0.870 | 0.970 | 0.760 | 0.940 |
| Entity passage, frequency | 0.970 | 0.670 | 0.870 | 0.970 | 0.760 | 0.940 |
| Entity sentence, binary | 0.970 | 0.670 | 0.870 | 0.970 | 0.760 | 0.940 |
| Entity sentence, top 50% | 0.970 | 0.670 | 0.870 | 0.970 | 0.760 | 0.940 |
| Chunk BGE kNN k=2 | 1.000 | 0.670 | 0.870 | 1.000 | 0.760 | 1.000 |

Mean graph structure was measured for every question and is saved with percentile intervals in `pilot50_final/graph_summary.csv`. Sentence-frequency entity graphs averaged 74.0 nodes, 156.78 edges, density 0.05996, and 15.24 components. Passage-frequency graphs averaged 74.0 nodes, 494.10 edges, density 0.17340, and 2.80 components. The 50%-sparsified sentence graph averaged 78.62 edges, density 0.03012, and 29.08 components. Chunk graphs had 10 nodes, 14.46 edges, density 0.32133, and 1.12 components.

The pilot shows the graph variants had the same reported retrieval means, while their structures differed. Entity candidate generation omitted some gold support (mean recall 0.97); chunk graph and vanilla BGE candidate recall were 1.00. Direct paired tests among the graph-construction choices are included in the 69-comparison family. After Holm correction, no pilot comparison was significant. With 50 questions, that should be treated as a feasibility check, not evidence of equivalence.

## 498-question primary results

| Configuration | Candidate recall | Recall@2 | Recall@5 | Recall@10 | Complete support@2 | Complete support@5 | Complete support@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla TF-IDF | 1.0000 | 0.6576 | 0.8363 | 1.0000 | 0.3795 | 0.6888 | 1.0000 |
| Vanilla BGE | 1.0000 | 0.7480 | 0.9056 | 1.0000 | 0.5281 | 0.8153 | 1.0000 |
| Entity sentence, frequency | 0.9839 | 0.7480 | 0.9056 | 0.9839 | 0.5281 | 0.8153 | 0.9679 |
| Entity passage, frequency | 0.9839 | 0.7480 | 0.9056 | 0.9839 | 0.5281 | 0.8153 | 0.9679 |
| Entity sentence, binary | 0.9839 | 0.7480 | 0.9056 | 0.9839 | 0.5281 | 0.8153 | 0.9679 |
| Entity sentence, top 50% | 0.9839 | 0.7480 | 0.9056 | 0.9839 | 0.5281 | 0.8153 | 0.9679 |
| Chunk BGE kNN k=2 | 0.9970 | 0.7480 | 0.9056 | 0.9970 | 0.5281 | 0.8153 | 0.9940 |

The 95% intervals for each reported measure are in `primary498_final/retrieval_summary.csv`. Vanilla BGE exceeded vanilla TF-IDF by 0.0904 Recall@2, 0.0693 Recall@5, 0.1486 complete-support@2, and 0.1265 complete-support@5; each difference remained significant after Holm correction (adjusted p=0.00345). There was no difference at @10 because all ten-context passages are ranked and every gold passage is present in the context.

All entity graph variants matched BGE at Recall@2 and @5, but their candidate generation recall was 0.9839, and complete support@10 was 0.9679. Their differences from vanilla BGE at candidate recall, Recall@10 and complete-support@10 were significant after Holm correction (adjusted p values 0.00345–0.00580). The chunk graph had 0.9970 candidate recall and 0.9940 complete-support@10; its comparisons with vanilla BGE were not significant after correction. Direct tests of passage-frequency, binary, and sparsified entity construction against the sentence-frequency reference yielded zero paired metric differences (adjusted p=1.0). Chunk graph candidate recall and complete-support@10 were higher than the sentence-frequency entity graph by 0.0131 and 0.0261, respectively, but neither survived Holm correction (adjusted p=0.237 and 0.237). Structural differences did not produce retrieval mean differences among the entity variants in this design.

Mean primary graph statistics (bootstrap intervals in `primary498_final/graph_summary.csv`): sentence-frequency entity graphs averaged 74.41 nodes, 156.22 edges, density 0.05904, and 15.51 components; passage-frequency entity graphs averaged 74.41 nodes, 499.22 edges, density 0.17273, and 3.17 components; the top-half sparsified graphs averaged 78.35 edges, density 0.02967, and 29.45 components; chunk graphs averaged 10 nodes, 14.50 edges, density 0.32222, and 1.10 components.

At k=10, mean cross-configuration Jaccard overlap was 0.9329 between vanilla BGE and vanilla TF-IDF, 0.8333 between vanilla BGE and sentence-frequency entity retrieval, 0.8351 between vanilla BGE and passage-frequency entity retrieval, and 0.9329 between vanilla BGE and chunk retrieval. These values measure ranking overlap, not stability across random seeds. All pairwise overlaps and intervals are in `primary498_final/retrieval_stability_summary.csv`.

## Runtime and resources

The 498-question retrieval run completed in 133.51 seconds after input JSON loading, with process peak RSS 1,231.2 MB. It included fresh extraction for 4,443 passages (50.44 seconds), BGE model loading (0.12 seconds), 4,443 new passage embeddings (71.25 seconds; 500 passage embeddings were cached from the pilot), 448 new query embeddings (1.20 seconds), graph construction, retrieval, bootstrap summaries and output writing. After expanding the paired-test family to include direct graph-construction contrasts, paired comparisons were recomputed from saved raw rows in an additional 3.57 seconds. The exact stage timings are in `primary498_final/efficiency.csv`. This cached-run measurement excludes model download and the earlier extraction/embedding work used to warm the 500-passage pilot cache. Per-question `query_retrieval_seconds` starts with cached query/passage vectors and excludes graph construction and embedding creation; TF-IDF timing includes each question's fit and rank. Graph construction, candidate generation, BGE scoring, and reranking times are separate per-row fields. Do not read a retrieval-only time as total pipeline time.

Exact experiment command:

```text
.venv/bin/python -m src.evaluate_dense_graph --scope primary498 --run-name primary498_final --output-dir results/dense_graph_v1
```

Pilot command:

```text
.venv/bin/python -m src.evaluate_dense_graph --scope pilot50 --run-name pilot50_final --output-dir results/dense_graph_v1
```

Correctness tests:

```text
.venv/bin/python -m unittest tests.test_dense_graph_pilot -v
.venv/bin/python -m unittest discover -v
```

The focused suite passed 9 tests. The full repository suite passed all 24 tests. The initial full suite was completed before the 498-question run; the final full suite was rerun after adding direct graph-construction comparisons and passed all 24 tests. The final test output is recorded in `logs/progress.md`.

Independent output audits:

```text
.venv/bin/python -m src.validate_dense_graph_run results/dense_graph_v1/pilot50_final --output results/dense_graph_v1/pilot50_final/independent_audit.json
.venv/bin/python -m src.validate_dense_graph_run results/dense_graph_v1/primary498_final --output results/dense_graph_v1/primary498_final/independent_audit.json
```

## Run history and limitations

- The first sandbox model load failed because network name resolution was unavailable. BGE was fetched to the local Hugging Face cache after the authorized network retry. The failed attempt is retained in `pilot50/failure.json`; it did not substitute another model.
- An initial sparsification implementation used edge insertion order to break equal-weight ties. Independent reconstruction exposed nondeterminism. The rule now canonicalizes endpoint IDs, sorts context IDs during construction, has a regression test, and the corrected final pilot passes exact graph-edge reconstruction. Earlier `pilot50` and `pilot50_verified` directories are retained as superseded attempts; use only `pilot50_final` and `primary498_final` for validated results.
- The full dataset is the fixed all-hard 498-question subset; these results do not establish performance on the broader HotpotQA population.
- Graph-assisted runs use BGE both for seed selection and final candidate scoring. This isolates candidate generation from lexical/dense score changes but does not test graph-aware score fusion or graph reranking.
- Entity extraction quality was not manually annotated. The old entity map's undocumented provenance prevented reusing it as graph evidence; the spaCy extraction can omit or normalize entities imperfectly.
- Only one deterministic configuration/sample seed was evaluated. No repeated-seed retrieval stability estimate is available. Cross-configuration Jaccard is descriptive, not a substitute for seed stability.
- Typed relation extraction was not attempted because no validated typed extraction pipeline was available in this phase. Answer generation and external GraphRAG systems were not run.
- The two variable-context questions were excluded from the primary 498-question analysis and have no Phase B results here.
- Runtime is from one Apple Silicon CPU execution with warm local model files and partially warm caches; timing is environment-specific. Peak RSS is process high-water memory, not a system-wide measurement.
