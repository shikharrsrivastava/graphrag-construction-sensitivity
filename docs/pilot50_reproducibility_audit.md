# Reproducibility Audit: 50-Question HotpotQA Pilot

**Audit date:** 9 November 2026  
**Scope:** Existing files in `data/pilot_50/` and `results/pilot_50/perturbed_retrieval/`.  
**Purpose:** Verify what the pilot actually demonstrates, expose methodological limits, and define the next experiments. This is an audit of the supplied artifacts, not a claim that the full research study is complete.

## 1. Dataset and graph inspected

The saved manifest identifies the sample as `hotpotqa/hotpot_qa`, `distractor` configuration, `validation` split, sampled with shuffle seed 42. It contains:

| Item | Observed value |
|---|---:|
| Questions | 50 unique |
| Unique context passages | 492 |
| Difficulty | 50 hard (100%) |
| Question type | 37 bridge (74%); 13 comparison (26%) |
| Gold supporting passages | Exactly 2 per question |

**Sampling limitation:** this is not a representative validation-set sample because all 50 questions are hard. The loader's shuffle buffer for `--limit 50` is 1,000 (`max(1,000, limit × 20)`). The published dataset card lists 7,405 examples in the distractor validation split. For a 500- or 1,000-question run, the current loader formula gives a buffer of 10,000 or 20,000 respectively, which is large enough to shuffle the complete 7,405-example split before taking the requested sample. The next run should use 500 or 1,000, a new seed, and a new output folder; report the resulting level/type distribution rather than assuming it is balanced.

The graph is an spaCy NER-derived, undirected entity co-occurrence graph. Named entities are normalized by case-folding and whitespace normalization. Two entities are connected when they occur in the same sentence; edge weight counts how many such co-occurrences were observed. It is **not** a typed semantic knowledge graph and has no extracted relation labels.

| Graph statistic | Value |
|---|---:|
| Entity nodes | 3,216 |
| Co-occurrence edges | 7,240 |
| Mean degree | 4.5025 |
| Density | 0.001400 |
| Connected components | 541 |
| Largest component | 1,982 nodes (61.63%) |
| Isolated nodes | 272 (8.46%) |
| Nodes with degree ≤ 2 | 1,220 (37.94%) |
| Edge-weight median / maximum | 1 / 8 |

This is a sparse, fragmented graph. That structure is relevant to the literature on modularity/Leiden instability in sparse GraphRAG graphs; it also means we should not assume that a community assignment is a reliable proxy for semantic relatedness.

## 2. What the existing retrieval experiment does

The retrieval code implements a **TF-IDF-seeded, TF-IDF-reranked candidate-filtering experiment**:

1. The question is scored against every passage using TF-IDF with English stop-word removal and unigram/bigram features.
2. The two highest-ranked passages (`seed_k=2` in the saved perturbation experiment) provide entity seeds.
3. One-hop, Leiden-community, or Louvain-community expansion produces a candidate passage pool.
4. The same TF-IDF scores rerank that candidate pool, and Recall@5 / Recall@10 are computed against HotpotQA gold supporting passages.

The full-corpus TF-IDF method has all 492 passages available. The graph methods restrict candidate generation before the same TF-IDF ranking. Thus, the experiment compares **candidate-generation constraints**, not an independent dense retriever against a semantic GraphRAG retriever.

No dense retriever, answer-generation model, answer EM/F1, or faithfulness evaluation is present in these saved results. The implementation is a useful retrieval pilot, not a complete end-to-end GraphRAG evaluation.

## 3. Re-analysed retrieval results

The raw file contains five clustering/perturbation seeds for each question. The analysis script first averages repeated runs within each `question_id`, then computes means over the 50 question-level observations. It does not treat the 250 question-run rows as 250 independent questions.

### Original graph

| Method | Recall@5 | Recall@10 | Candidate-pool size (mean) | Gold evidence in candidate pool |
|---|---:|---:|---:|---:|
| Full-corpus TF-IDF | 0.790 | **0.960** | 492.0 | 1.000 |
| One-hop graph | 0.830 | 0.930 | 96.7 | 0.950 |
| Leiden community | 0.834 | 0.926 | 72.6 | 0.946 |
| Louvain community | 0.836 | 0.926 | 77.6 | 0.946 |

The graph-filtered methods use much smaller candidate pools. On this pilot, their Recall@5 is numerically higher than full-corpus TF-IDF, while full-corpus TF-IDF has higher Recall@10. This is a candidate-pool/recall trade-off, not a demonstrated general performance improvement.

### Under 5% edge removal

| Condition | Method | Recall@5 | Recall@10 | Candidate-pool size |
|---|---|---:|---:|---:|
| Random 5% | TF-IDF | 0.790 | 0.960 | 492.0 |
| Random 5% | One-hop | 0.830 | 0.930 | 94.3 |
| Random 5% | Leiden | 0.834 | 0.926 | 73.5 |
| Random 5% | Louvain | 0.828 | 0.920 | 74.6 |
| Critical-edge 5% | TF-IDF | 0.790 | 0.960 | 492.0 |
| Critical-edge 5% | One-hop | 0.820 | 0.920 | 44.1 |
| Critical-edge 5% | Leiden | 0.838 | 0.928 | 69.0 |
| Critical-edge 5% | Louvain | 0.836 | 0.926 | 70.7 |

For each 5% perturbation, 362 of 7,240 edges are removed. “Critical-edge” means edges with high **unweighted** edge betweenness: the implementation explicitly calls `edge_betweenness_centrality(..., weight=None)`. It therefore does not use the co-occurrence edge weights when ranking critical edges.

### Paired uncertainty check

A paired percentile bootstrap was run on the 50 question-level observations, comparing each graph method to full-corpus TF-IDF. There are 50 independent question units; five seeds are averaged within each unit. All 95% percentile intervals for the mean Recall@k difference in this pilot include zero.

| Condition | Comparison | k | Mean paired difference | 95% bootstrap interval |
|---|---|---:|---:|---:|
| Original | Leiden − TF-IDF | 5 | +0.044 | [−0.010, +0.100] |
| Original | Leiden − TF-IDF | 10 | −0.034 | [−0.078, +0.006] |
| Random 5% | Leiden − TF-IDF | 5 | +0.044 | [−0.014, +0.102] |
| Random 5% | Leiden − TF-IDF | 10 | −0.034 | [−0.078, +0.008] |
| Critical-edge 5% | Leiden − TF-IDF | 5 | +0.048 | [−0.010, +0.106] |
| Critical-edge 5% | Leiden − TF-IDF | 10 | −0.032 | [−0.074, +0.008] |

These are exploratory, unadjusted percentile intervals from 10,000 bootstrap resamples. They do **not** establish a statistically reliable improvement. The full 24-comparison table is in `results/pilot_50/audit/paired_bootstrap_vs_tfidf.csv`.

## 4. Structural robustness: what is and is not measured

The existing code compares each perturbed partition against the original-graph partition generated with the same clustering seed. In the unperturbed condition, it compares each partition to itself, so ARI = 1.0 is true by construction. This **does not measure cross-seed reproducibility on an unchanged graph**.

For the perturbation comparisons, the saved same-seed ARI means are:

| Edge condition | Leiden ARI | Louvain ARI |
|---|---:|---:|
| Random 5% | 0.8605 | 0.8470 |
| High-betweenness 5% | 0.7894 | 0.8131 |

The largest-connected-component fraction changes from 0.6163 in the original graph to 0.6100 under random removal and 0.5622 under critical-edge removal. These results describe structural sensitivity to edge deletion. They do not establish that more stable communities produce better answers.

### Cross-seed stability on the unchanged graph (newly executed)

A separate experiment now saves and compares ten Louvain partitions of the same unchanged graph (seeds 42–51). There are 45 unordered seed pairs. The result is different from the existing same-seed perturbation ARI above and directly measures run-to-run partition agreement.

| Measure | Result |
|---|---:|
| Pairwise partition ARI, mean (SD) | 0.8799 (0.0395) |
| Pairwise partition ARI, range | 0.8031–0.9830 |
| Pairwise partition NMI, mean (SD) | 0.9771 (0.0073) |
| Retrieved-passage Jaccard@5, mean | 0.9486 |
| Retrieved-passage Jaccard@5, question-bootstrap 95% CI | [0.9212, 0.9730] |
| Retrieved-passage Jaccard@10, mean | 0.9181 |
| Retrieved-passage Jaccard@10, question-bootstrap 95% CI | [0.8818, 0.9508] |

The retrieval intervals resample 50 questions after averaging each question's overlap across its 45 seed pairs. The 45 seed pairs are not treated as independent questions. For both retrieval budgets, the median per-question mean Jaccard is 1.0; some questions nevertheless show lower overlap. TF-IDF and one-hop retrieval are invariant to the community seed in this implementation, so their Jaccard is 1.0 by construction and they are useful sanity checks rather than clustering-stability findings.

**Interpretation:** Louvain community partitions show measurable seed variation (mean ARI 0.88), while the final top-k passage sets are more consistent (mean Jaccard 0.95 at k=5 and 0.92 at k=10). This suggests partition similarity alone may not predict evidence-retrieval variation in this pipeline. It does not show answer stability because no answers were generated, and it is only a 50-question, all-hard sample using Louvain. The corresponding 10-seed Leiden experiment remains to be run in an environment with `python-igraph` and `leidenalg` installed.


### Controlled graph-construction ablation: same sentence vs same passage

A second graph was derived from the saved `passage_entities.json` without rerunning NER. It preserves the exact same 3,216 entity nodes and node attributes; the only changed construction choice is the co-occurrence window:

- **Same-sentence:** connect entities that occur in the same sentence (the original graph).
- **Same-passage:** connect entities that occur anywhere in the same passage; edge weight is the number of shared passages.

This is a controlled edge-granularity comparison. It does not test semantic-similarity edges, typed relations, or a different node type.

| Graph statistic | Same sentence | Same passage |
|---|---:|---:|
| Entity nodes | 3,216 | 3,216 |
| Edges | 7,240 | 22,479 |
| Mean degree | 4.502 | 13.979 |
| Connected components | 541 | 27 |
| Largest component fraction | 0.616 | 0.934 |
| Isolated nodes | 272 | 0 |

All 7,240 same-sentence edges are also present in the same-passage graph; the passage-level graph adds 15,239 edges. The edge-set Jaccard is 0.322. Thus the coarser co-occurrence window substantially densifies and connects the graph.

Both variants were clustered using Louvain for ten seeds (42–51), and retrieval was evaluated using the existing TF-IDF top-2 seed selection and TF-IDF reranking. The retrieval metrics are first averaged across seeds within each question, and uncertainty is bootstrapped across the 50 questions.

| Variant | Retrieval method | Recall@5 | Recall@10 | Mean candidate pool |
|---|---|---:|---:|---:|
| Same sentence | TF-IDF baseline | 0.790 | 0.960 | 492.0 |
| Same sentence | One-hop | 0.830 | 0.930 | 96.7 |
| Same sentence | Louvain community | 0.839 | 0.931 | 76.0 |
| Same passage | TF-IDF baseline | 0.790 | 0.960 | 492.0 |
| Same passage | One-hop | 0.850 | 0.960 | 138.6 |
| Same passage | Louvain community | 0.822 | 0.937 | 64.1 |

The paired question-level bootstrap for **same-passage minus same-sentence** Recall@k gives:

| Method | k | Mean difference | 95% percentile CI |
|---|---:|---:|---:|
| Louvain community | 5 | −0.017 | [−0.042, +0.002] |
| Louvain community | 10 | +0.006 | [−0.027, +0.041] |
| One-hop | 5 | +0.020 | [0.000, +0.050] |
| One-hop | 10 | +0.030 | [0.000, +0.070] |

The intervals include or touch zero; this pilot does not establish a reliable retrieval improvement from the broader co-occurrence window. TF-IDF remains unchanged because it does not use the graph.

The partition comparison also changes: mean pairwise Louvain ARI across the ten seeds is **0.8799** for the same-sentence graph and **0.7818** for the same-passage graph; mean NMI is **0.9771** and **0.9196**, respectively. In this sample, making the graph denser did not make Louvain partitions more reproducible.

The retrieval-set stability comparison is:

| Graph construction | Jaccard@5 (95% question-bootstrap CI) | Jaccard@10 (95% question-bootstrap CI) |
|---|---:|---:|
| Same sentence | 0.9486 [0.9212, 0.9730] | 0.9181 [0.8818, 0.9508] |
| Same passage | 0.9317 [0.8952, 0.9635] | 0.8701 [0.8277, 0.9094] |

The paired difference (same-passage minus same-sentence) in per-question Jaccard is −0.0169 at k=5 (95% CI [−0.0562, +0.0186]) and −0.0479 at k=10 (95% CI [−0.0965, −0.0014]). The k=10 interval narrowly excludes zero in this **unadjusted exploratory** analysis. It is a signal to replicate, not a final claim: there are only 50 all-hard questions, two retrieval budgets are compared without multiplicity correction, and retrieval overlap is not answer correctness.

**Interpretation:** graph-construction granularity strongly changes topology and community assignments. It also appears to reduce Louvain top-10 evidence-set consistency for the same-passage variant in this pilot, although retrieval Recall@k differences remain uncertain. This supports measuring graph topology, partition stability, evidence-set stability, evidence recall, and answer quality separately. It does not establish a better GraphRAG method or an answer-quality result.

## 5. Implementation and reproducibility issues

1. **README status has been updated.** It now distinguishes the implemented TF-IDF-seeded candidate-filtering pilot from the not-yet-implemented dense-retrieval and answer-generation stages.
2. **Different output folders contain different sample sizes.** `data/processed` and the top-level `results/perturbed_retrieval` are the 15-question pilot. The 50-question experiment is in `data/pilot_50` and `results/pilot_50/perturbed_retrieval`. Always pass explicit paths; do not accidentally report the 15-question outputs as the 50-question run.
3. **Original perturbation run versus cross-seed follow-up.** The original perturbation experiment saves seed-42 partitions only. The newly added `evaluate_seed_stability.py` separately saves all 10 Louvain partitions (seeds 42–51) for the unchanged graph. The 10-seed Leiden follow-up has not yet been run in this audit runtime.
4. **Critical-edge selection ignores edge weights.** The current betweenness computation is unweighted. Keep this as a named condition or add a separately defined weighted-distance variant; do not silently describe it as weighted.
5. **The graph is built using spaCy surface-form NER.** Entity alias resolution, extraction recall/precision, and typed relation correctness are not evaluated.
6. **The current retrieval methods share TF-IDF for seed selection and final reranking.** A dense retriever and an independent chunk-graph/GraphRAG baseline are needed to test whether the observations generalize.
7. **No generated answers are recorded.** Add deterministic answer generation and report HotpotQA answer EM/F1 plus a clearly specified faithfulness assessment before discussing answer reliability.
8. **Environment portability.** The uploaded archive included a local macOS virtual environment. It is machine-specific and is excluded from the cleaned project bundle along with `__pycache__` and caches. The audit runtime used here does not have `python-igraph`/`leidenalg`, so the new Leiden cross-seed run could not be executed here.

## 6. Literature implications for novelty

The recent literature means the initial broad claim must be narrowed:

- **Nishida et al., TACL 2026, “Dissecting GraphRAG”** varies triple extraction, community clustering, and report-generation modules and evaluates downstream factoid QA. Therefore, a generic claim that graph construction/clustering design affects QA is already covered in part. [Paper](https://aclanthology.org/2026.tacl-1.29/), DOI: 10.1162/tacl.a.615.
- **Hossain & Sarıyüce, KDD 2026, “Core-based Hierarchies for Efficient GraphRAG”** studies sparse-graph modularity degeneracy and Leiden instability; its empirical appendix reports ten Leiden seeds and retrieval-node Jaccard variation. Therefore, seed sensitivity of Leiden and its retrieval implications are not new by themselves. [Paper](https://arxiv.org/abs/2603.05207), DOI: 10.1145/3770855.3818007.
- **Liao et al., 2026, “GraphRAG Best Practices”** evaluates indexing, graph-guided retrieval, and generation across MSMARCO, HotpotQA, and a regulatory corpus; on HotpotQA it tests 200 validation entries and multiple embeddings/retrievers/rerankers/generation formats. A generic end-to-end component ablation on HotpotQA would overlap with this work. [Paper](https://journals.sagepub.com/doi/10.3233/SSW260013), DOI: 10.3233/SSW260013.
- **Giordano & Razniewski, Findings of EACL 2026, “Foundations of LLM Knowledge Materialization”** directly studies LLM knowledge-graph extraction reproducibility/robustness across seed, language, randomness, and model variations, although it does not present the full downstream RAG QA propagation study in its abstract. [Paper](https://aclanthology.org/2026.findings-eacl.113/), DOI: 10.18653/v1/2026.findings-eacl.113.

**Candidate gap to validate, not yet a novelty claim:** systematically trace controlled graph-construction variation (extraction granularity, node/edge representation, entity resolution, edge weighting/sparsification) through (i) graph topology and community assignments, (ii) retrieved evidence under fixed retrieval budgets and more than one retriever, and (iii) final answer correctness/faithfulness on multi-hop factoid QA. The study should explicitly compare graph-build variation with clustering-seed variation and edge perturbation. A full literature review and exact comparison matrix are still required before making a “first” or “novel” claim.

## 7. Reproducible audit outputs

Run from the project root:

```bash
python -m src.analyze_pilot50
python -m src.evaluate_seed_stability --methods louvain --seeds 42 43 44 45 46 47 48 49 50 51 --output-dir results/pilot_50/seed_stability_louvain_10seeds
python -m src.evaluate_graph_construction_variants --dataset-dir data/pilot_50 --seeds 42 43 44 45 46 47 48 49 50 51 --output-dir results/pilot_50/graph_construction_variants
```

The script produces:

- `results/pilot_50/audit/dataset_graph_audit.json`
- `results/pilot_50/audit/question_level_metrics.csv`
- `results/pilot_50/audit/retrieval_question_level_summary.csv`
- `results/pilot_50/audit/paired_bootstrap_vs_tfidf.csv`
- `results/pilot_50/audit/structural_summary.csv`
- `results/pilot_50/audit/retrieval_summary.png`
- `results/pilot_50/audit/structural_summary.png`
- `results/pilot_50/seed_stability_louvain_10seeds/partition_similarity_pairwise.csv`
- `results/pilot_50/seed_stability_louvain_10seeds/partition_similarity_summary.csv`
- `results/pilot_50/seed_stability_louvain_10seeds/retrieval_jaccard_pairwise.csv`
- `results/pilot_50/seed_stability_louvain_10seeds/retrieval_jaccard_summary.csv`
- `results/pilot_50/seed_stability_louvain_10seeds/retrieval_jaccard_stability.png`
- `results/pilot_50/seed_stability_louvain_10seeds/retrieval_jaccard_question_level.csv`
- `results/pilot_50/seed_stability_louvain_10seeds/retrieval_jaccard_bootstrap_ci.csv`
- `results/pilot_50/graph_construction_variants/graph_variant_statistics.csv`
- `results/pilot_50/graph_construction_variants/retrieval_raw.csv`
- `results/pilot_50/graph_construction_variants/retrieval_question_level.csv`
- `results/pilot_50/graph_construction_variants/community_counts_by_seed.csv`
- `results/pilot_50/graph_construction_variants/partition_similarity_pairwise.csv`
- `results/pilot_50/graph_construction_variants/retrieval_summary.csv`
- `results/pilot_50/graph_construction_variants/paired_bootstrap_passage_vs_sentence.csv`
- `results/pilot_50/graph_construction_variants/retrieval_jaccard_bootstrap_ci.csv`
- `results/pilot_50/graph_construction_variants/paired_bootstrap_jaccard_passage_vs_sentence.csv`
- `results/pilot_50/graph_construction_variants/retrieval_seed_stability_by_variant.png`
- `results/pilot_50/graph_construction_variants/partition_similarity_summary.csv`
- `results/pilot_50/graph_construction_variants/graph_construction_retrieval_comparison.png`

The project test suite now passes 9/9 tests, including tests for cross-seed stability and controlled graph construction. `analyze_pilot50.py` reanalyses the saved raw pilot results; `evaluate_seed_stability.py` reruns clustering and retrieval for the unchanged graph. Neither script generates answers. The cross-seed experiment and controlled edge-granularity ablation were executed with Louvain only in this runtime.
