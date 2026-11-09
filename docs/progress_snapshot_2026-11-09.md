# Research Progress Snapshot — 9 November 2026

## Main objective

Respond to Prof. Pasquale De Meo's research guidance by strengthening the GraphRAG robustness study with reproducible evidence before sending a results-backed progress email.

## Completed in this audit cycle

1. **Audited the supplied project archive.** Distinguished the 15-question pilot from the later 50-question pilot and retained the original data/results.
2. **Re-analysed 50-question retrieval results.** Repeated seed runs were averaged within question; paired bootstrap resampling used 50 independent question units. The pilot does not establish a statistically reliable Recall@k improvement over full-corpus TF-IDF.
3. **Ran a 10-seed Louvain stability experiment** (seeds 42–51) on the unchanged 50-question entity graph, saving all partitions and pairwise ARI/NMI plus retrieval-passage Jaccard.
4. **Ran a controlled edge-granularity ablation.** Built a same-passage co-occurrence graph by reusing the exact same saved entity extraction and node universe, then compared it with the original same-sentence graph across ten Louvain seeds and evaluated evidence retrieval.
5. **Prepared a focused literature review** covering TACL 2026 GraphRAG module analysis, KDD 2026 sparse-graph/Leiden stability, SAGE 2026 modular GraphRAG evaluation, and EACL 2026 knowledge-materialization reproducibility.
6. **Updated README and experimental protocol.** Added reproducibility commands, scope boundaries, limitations, and output locations.
7. **Validation:** all 9 project unit tests pass; `compileall` succeeds.

## Preliminary results to retain

- The 50-question sample contains 50 hard questions (37 bridge, 13 comparison), 492 unique passages, and exactly two supporting passages per question. It is not representative of all HotpotQA difficulty levels.
- Original same-sentence graph: 3,216 nodes, 7,240 edges, 541 components, 272 isolates, LCC fraction 0.616.
- 10-seed Louvain on the unchanged graph: mean pairwise ARI 0.8799; mean NMI 0.9771. Retrieved passage Jaccard averaged 0.9486 at k=5 and 0.9181 at k=10 (question-bootstrap 95% CIs [0.9212, 0.9730] and [0.8818, 0.9508]).
- Same-passage graph (same entity nodes): 22,479 edges, 27 components, no isolates, LCC fraction 0.934. Its ten-seed mean pairwise Louvain ARI is 0.7818 (vs 0.8799 for same-sentence).
- In the graph-granularity retrieval ablation, Louvain Recall@5 was 0.839 for same-sentence and 0.822 for same-passage; Recall@10 was 0.931 and 0.937. Paired question-bootstrap intervals for Recall differences include/touch zero.
- Louvain retrieved-set Jaccard was 0.9486/0.9181 (k=5/10) for same-sentence and 0.9317/0.8701 for same-passage. The paired same-passage-minus-same-sentence Jaccard difference at k=10 was −0.0479 with unadjusted 95% CI [−0.0965, −0.0014]; this narrow exploratory signal needs replication and multiplicity-aware confirmation.

## Important limitations / still pending

- The 50-question sample is small and all-hard.
- The retrieval pipeline is TF-IDF-seeded and TF-IDF-reranked. It is not an independent dense retrieval baseline.
- No answer generation, HotpotQA answer EM/F1, or faithfulness evaluation has been run.
- The 10-seed cross-seed and graph-granularity runs used Louvain only. The audit runtime lacked `python-igraph` and `leidenalg`, so the matching Leiden experiments remain pending.
- A representative 500- or 1,000-question sample remains to be generated; the audit runtime lacked the `datasets` package and outbound model/dataset downloads.
- Dense retrieval (e.g. BGE-small/Contriever), a matched graph-assisted dense retriever, chunk-graph baseline, and an established GraphRAG system remain pending.
- Recent literature already covers substantial parts of the broad idea. Novelty must be narrowed after reading full papers and constructing a detailed method-by-method comparison.

## Decision

Do **not** send Prof. De Meo a progress email yet. The current work is meaningful pilot-stage progress, but it does not satisfy his central recommendations for larger-scale evaluation, dense retrieval, answer generation, and faithfulness. Next, run the 500-question sample and Leiden follow-up locally, then add dense retrieval and answer evaluation with a fixed protocol.
