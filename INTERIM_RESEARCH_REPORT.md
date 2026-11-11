# Interim Research Report: Graph Construction Sensitivity and Evidence Retrieval in Multi-Hop QA

**Report date:** 11 November 2026
**Project:** GraphRAG Robustness and Sensitivity Analysis
**Status:** historical graph/community experiments, Stage A lexical evaluation, and Phase B dense/local-graph retrieval are documented. Answer generation, faithfulness, and external GraphRAG comparisons remain unexecuted. This is an interim empirical report, not a completed GraphRAG research study or a novelty claim.

## Executive summary

The project asks whether changes to graph construction and community structure affect retrieval of supporting evidence for multi-hop questions and, ultimately, answer reliability. The initial historical experiments built a pooled named-entity co-occurrence graph, measured Leiden/Louvain community behavior, and evaluated TF-IDF-seeded graph candidates. Those experiments rank against the full pooled passage corpus and are not question-local distractor evaluation.

Stage A then established a separate question-local TF-IDF baseline using the 498 complete ten-context questions from `data/pilot_500_v2`, with the two two-context examples reported separately. It also retained a secondary shared-global-graph track; that track uses pooled graph and TF-IDF information and is not a local graph experiment.

Phase B implemented BGE dense retrieval and five question-local graph-assisted methods. A stratified 50-question pilot passed independent reconstruction and was scaled to the full 498-question primary set. Dense and graph retrieval metrics were independently recomputed from saved rows; all 3,486 full-set records passed context, graph, candidate, ranking, metric, and evidence checks. BGE performed better than local TF-IDF at Recall@2 and @5 in the full sample. Graph candidate generation preserved BGE's Recall@2 and @5 but excluded some supporting evidence before the final ranking, particularly for entity graphs.

These phases measure evidence retrieval. They do not measure generated answer correctness or faithfulness. No claim is made that the entire GraphRAG research is complete.

## Research motivation and scope

The motivating question began as: under graph-edge perturbations, does community-partition stability correspond to preservation of evidence for multi-hop question answering? The project scope now studies a more direct sensitivity chain: graph construction choices → graph structure/community organization → retrieved supporting passages → generated answer quality and evidence faithfulness.

The empirical distinction matters. Stable partitions do not themselves prove stable evidence retrieval, and stable retrieved evidence does not establish answer correctness. GraphRAG here refers to the research area under study; the implemented systems so far are retrieval experiments, not an end-to-end GraphRAG architecture. The local [literature review matrix](docs/literature_review_matrix.md) identifies related work and states that the broad graph-design question is not by itself a validated novelty claim. A full method-by-method literature review remains outstanding.

## Data and historical pooled-corpus experiments

### Original 500-question corpus and graph

`data/pilot_500` contains 500 HotpotQA validation distractor questions, 4,947 unique passages, all-hard questions (404 bridge, 96 comparison), and two gold supporting passages per question. It predates the v2 context-ID mapping and has no per-question candidate contexts suitable for the later question-local primary evaluation. The graph is a pooled spaCy named-entity surface-form co-occurrence graph built across the full passage collection.

The graph was independently loaded from `data/pilot_500/entity_graph.graphml` for this report:

| Statistic | Pooled graph |
|---|---:|
| Entity nodes | 28,178 |
| Co-occurrence edges | 75,013 |
| Connected components | 3,753 |
| Isolates | 1,898 |
| Density | 0.00018896 |
| Mean degree | 5.3242 |

This is an NER-derived co-occurrence graph, not a typed knowledge graph. The historic TF-IDF retrieval evaluator fits TF-IDF over all 4,947 passages, takes two top lexical seed passages, expands through entity one-hop or communities, and reranks the generated candidates using the same global-corpus TF-IDF scores. Results are pooled-corpus retrieval; they are not restricted to each question's own ten contexts.

### Historical pooled retrieval and ten-seed community runs

The saved 500-question retrieval quality summaries average the ten graph-seed runs (seeds 42–51) within question for Leiden and Louvain. Candidate pool sizes average 4,947 for TF-IDF, 1,291 for Leiden, 1,371 for Louvain, and 2,710 for one-hop graph retrieval.

| Method | Recall@5 (95% CI) | Complete@5 | Recall@10 (95% CI) | Complete@10 | Candidate recall |
|---|---:|---:|---:|---:|---:|
| Full-pool TF-IDF | 0.7180 [0.6920, 0.7430] | 0.4782 | 0.8780 [0.8590, 0.8970] | 0.7580 | 1.0000 |
| Leiden community | 0.7252 [0.6997, 0.7505] | 0.4982 | 0.8571 [0.8364, 0.8778] | 0.7340 | 0.9247 |
| Louvain community | 0.7248 [0.6991, 0.7501] | 0.4986 | 0.8598 [0.8391, 0.8798] | 0.7384 | 0.9277 |
| One-hop graph | 0.7260 [0.7000, 0.7510] | 0.4940 | 0.8600 [0.8390, 0.8800] | 0.7300 | 0.9470 |

Historical question-paired bootstrap comparisons versus full-pool TF-IDF were exploratory and not multiplicity-adjusted. At Recall@5, graph differences were small and their intervals included zero: Leiden +0.0072 [−0.0051, +0.0196], Louvain +0.0068 [−0.0055, +0.0192], and one-hop +0.0080 [−0.0040, +0.0200]. At Recall@10, differences were negative: Leiden −0.0209 [−0.0384, −0.0033], Louvain −0.0182 [−0.0354, −0.0013], one-hop −0.0180 [−0.0330, −0.0030]. Those historical comparisons should not be read as final question-local findings.

The ten saved Leiden partitions averaged 3,831.3 communities (SD 6.5; range 3,821–3,840); Louvain averaged 3,830.1 (SD 6.9; range 3,820–3,844). Across the 45 unordered seed pairs, partition ARI/NMI averaged 0.5467/0.8791 for Leiden and 0.5147/0.8732 for Louvain. Question-bootstrap retrieval Jaccard was 0.9195 [0.9076, 0.9309] at k=5 and 0.8576 [0.8435, 0.8716] at k=10 for Leiden; for Louvain it was 0.9095 [0.8964, 0.9220] and 0.8482 [0.8335, 0.8627]. Seed pairs were aggregated within question before question bootstrap. These metrics concern stability of the global pooled-corpus process, not question-local graphs.

### Earlier 15- and 50-question exploratory work

The repository also preserves a 15-question structural perturbation run and separate 50-question graph-construction/community stability studies. The 15-question work perturbed a 994-node, 2,430-edge graph; its findings are exploratory and not 500-question results. In the 50-question construction ablation, the same 3,216 entity nodes were used in both variants: sentence co-occurrence had 7,240 edges and 541 components; passage co-occurrence had 22,479 edges and 27 components. Ten-seed Louvain retrieval differences were small: sentence vs passage Recall@5 was 0.839 vs 0.822 for community retrieval and 0.830 vs 0.850 for one-hop retrieval; at @10 it was 0.931 vs 0.937 and 0.930 vs 0.960. These were TF-IDF global-corpus candidate experiments, not dense or question-local evaluations. Full detail remains in [the pilot audit](docs/pilot50_reproducibility_audit.md) and `results/pilot_50/graph_construction_variants/`.

## Stage A — question-local TF-IDF and shared global graph

Stage A used `data/pilot_500_v2`, which preserves each question's context IDs. It contains 498 ten-context primary questions and two two-context questions. All 500 questions have their gold support passages among their own contexts. Stage A reports the two variable-context examples separately.

**Track A, question-local:** TF-IDF is fitted anew on each question's own ten passages; ranking is also restricted to those ten. No graph is built in Track A.

**Track B, shared global graph:** the previously pooled entity graph and mappings are reused, and its graph candidates are filtered to each question's contexts before ranking. TF-IDF statistics are shared across the 4,947-passage corpus. This track discloses cross-question graph and corpus-statistic information; it is not question-local graph construction.

| Stage A track, 498 questions | Candidate recall | Recall@2 | Recall@5 | Recall@10 | Complete@2 | Complete@5 | Complete@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Track A question-local TF-IDF | 1.0000 | 0.6576 | 0.8363 | 1.0000 | 0.3795 | 0.6888 | 1.0000 |
| Track B shared global graph + TF-IDF | 0.9900 | 0.5653 | 0.7329 | 0.9900 | 0.2570 | 0.5060 | 0.9799 |

For Track A, candidate recall is 1.0 by construction because its candidate universe includes all validated contexts. Track B's candidate recall is the graph-generated evidence coverage after local-context filtering. Stage A's paired tests found Track A higher than Track B across its six primary outcomes after Holm adjustment. Stage A did not implement local graph construction. See [Stage A validation](results/question_local_v1/STAGE_A_VALIDATION.md) and [independent review](STAGE_A_EXTERNAL_REVIEW.md).

## Phase B — BGE dense retrieval and question-local graphs

### Design and validation

The 50-question stratified pilot used seed 2027 and 41 bridge / 9 comparison questions. Seven methods were compared: local TF-IDF, vanilla BGE, sentence-frequency entity co-occurrence, passage-frequency entity co-occurrence, sentence-binary entity co-occurrence, strongest-half sparse sentence-frequency entity co-occurrence, and a BGE passage-chunk union-kNN graph (k=2). Each graph, graph node universe, and passage candidate pool was built only from the current question's ten contexts. Graph candidate generation used top-five vanilla BGE seeds, entity expansion to three weighted neighbors, or the chunk graph's direct neighbors. Candidates were reranked with the same BGE passage scores used by vanilla BGE.

The pilot passed independent reconstruction with zero context, graph-node, graph-edge, candidate, ranking, metric, graph-statistic, or evidence-denominator discrepancies. It was then scaled to all 498 primary questions (404 bridge, 94 comparison), yielding 3,486 raw rows. The full run independently passed those same checks. See [Phase B validation](results/dense_graph_v1/PHASE_B_VALIDATION.md) for all seven methods' means and CIs, model identity, test outcomes, audits, and runtime boundaries.

### Primary numerical findings

| Phase B method | Candidate recall | Recall@2 | Recall@5 | Recall@10 | Complete@10 |
|---|---:|---:|---:|---:|---:|
| Local TF-IDF | 1.0000 | 0.6576 | 0.8363 | 1.0000 | 1.0000 |
| Vanilla BGE | 1.0000 | 0.7480 | 0.9056 | 1.0000 | 1.0000 |
| Sentence-frequency entity graph | 0.9839 | 0.7480 | 0.9056 | 0.9839 | 0.9679 |
| Passage-frequency entity graph | 0.9839 | 0.7480 | 0.9056 | 0.9839 | 0.9679 |
| Sentence-binary entity graph | 0.9839 | 0.7480 | 0.9056 | 0.9839 | 0.9679 |
| Strongest-half sparse entity graph | 0.9839 | 0.7480 | 0.9056 | 0.9839 | 0.9679 |
| BGE chunk graph kNN=2 | 0.9970 | 0.7480 | 0.9056 | 0.9970 | 0.9940 |

All full-set per-question bootstrap intervals for candidate recall, Recall@2/5/10, and complete support@2/5/10 appear in the Phase B validation report and saved CSV. BGE improved over TF-IDF by 0.0904 Recall@2 and 0.0693 Recall@5 (both Holm-adjusted p=0.00345 in a 69-test family). Entity graphs had no paired differences from one another on the reported retrieval outcomes in this run, despite large differences in topology; entity methods had lower candidate recall and complete support@10 than vanilla BGE. The chunk graph retained more evidence than entity graphs, but direct chunk-vs-sentence-frequency differences did not survive Holm correction.

The pilot and full-set findings are directionally consistent: BGE outperformed TF-IDF early, graph candidate methods matched BGE at @2/@5, and candidate filtering affected @10 evidence completeness. Full-set point estimates are more precise and should be used for primary reporting. The pilot remains a feasibility check.

## Methodological cautions and unresolved work

1. **Historical pooled runs are not local-context evaluations.** The 500-question graph and retrieval experiments use the entire pooled corpus; Stage A Track B also shares global graph/index information. Only Stage A Track A and all Phase B primary graphs/indices/candidates are question-local in the ways stated above.
2. **Sample scope:** all 500 questions are hard; Phase B primary includes 404 bridge and 94 comparison questions. Estimates describe this fixed sample, not all HotpotQA.
3. **Variable contexts:** the two two-context examples are separated in Stage A and omitted from Phase B's ten-context primary analyses.
4. **Retrieval stability:** Phase B saved cross-configuration top-k Jaccard, not repeated-seed stability. The encoder/extraction/ranking pipeline is deterministic in its current CPU configuration; no repeated same-configuration seed experiment was run. Historical pooled runs do have community-seed variation but cannot substitute for local graph/retrieval stability.
5. **Entity quality:** NER output has not been manually labeled or reviewed. Agreement with an older automatic mapping is not a quality benchmark. Entity precision/recall, alias resolution, and relation extraction are unmeasured.
6. **Graph design coverage:** no sentence-node graph, passage-binary graph, alternate sparsification sweep, typed edge extraction, entity-semantic graph, bipartite graph, or local community method has been tested.
7. **Retrieval mechanism:** Phase B graph methods use BGE both to choose seeds and to rerank graph candidates. The experiment isolates graph-assisted candidate generation, not graph-score fusion or a complete GraphRAG pipeline.
8. **Answer quality:** no answer generation, exact match, token F1, evidence faithfulness, or human faithfulness audit has been conducted. No results should be inferred from evidence recall.
9. **Baselines:** no Contriever, LightRAG, or Microsoft GraphRAG integration is included.
10. **Novelty:** the literature matrix records related studies and the research question remains exploratory. No “first” or novelty claim is made.

## Outstanding work and recommended next gates

- Design repeated-seed experiments only for genuinely stochastic components and analyze seeds as repeated measurements nested within question.
- Create a manually annotated entity/relation quality sample before considering typed relation edges or entity-resolution claims.
- Expand only the most informative local graph variants: at minimum passage-level binary weighting, one additional justified sparsification condition, and sentence nodes if a reliable mapping is feasible. Keep comparison budgets fixed.
- Extend the matched retrieval evaluation to a graph-aware ranking/fusion condition only with a predeclared rule and local scope.
- Run a small answer-generation pilot with a fixed model, prompt, decoding settings, and evidence-token budget; evaluate EM and token F1 and manually review faithfulness on a stratified subset before scaling.
- Evaluate one established GraphRAG system only after confirming a reproducible local setup and keeping the same questions and evidence budget.
- Read the full related papers and complete the comparison matrix before stating a research contribution.

## Reproducibility and source files

- Phase B run tables, raw question records, pinned embedding cache, and extraction cache: `results/dense_graph_v1/`.
- Independent Phase B recomputation: `src/phase_b_independent_audit.py`; audits at `results/dense_graph_v1/phase_b_validation_audits/`.
- Phase B evaluator and tests: `src/evaluate_dense_graph.py`, `tests/test_dense_graph_pilot.py`.
- Stage A evaluator and validation: `src/evaluate_question_local.py`, `results/question_local_v1/`.
- Historical graph and clustering implementation: `src/build_graph.py`, `src/cluster_graph.py`, and the saved results beneath `results/pilot_500/` and `results/pilot_50/`.
- Environment, commands, failures, and runtimes: `logs/progress.md`.
