# Next Experiment Protocol — Graph Construction Sensitivity and Multi-Hop QA

> **Historical planning document:** written before Stage A and Phase B were run. Phase B’s completed design and results are documented in [`experimental_design.md`](experimental_design.md) and [`research/PHASE_B_REVIEW_SUMMARY.md`](research/PHASE_B_REVIEW_SUMMARY.md). Items below marked proposed/planned should not be read as current execution status.

**Status:** proposed protocol for discussion and implementation. Not yet executed.

**Design principle:** preserve the original 50-question pilot; create every expanded dataset, graph, index, and result under a new versioned directory.

## 1. Candidate research question

How do controlled graph-construction choices change graph structure and community organization, and how do those changes propagate to retrieved evidence and generated answer quality on multi-hop question answering?

A more specific testable version:

> Under a fixed corpus, query set, retriever, retrieval budget, and generator, how sensitive are evidence recall and answer quality to graph-construction choices (node unit, edge semantics, weighting, and sparsification), compared with clustering-seed variation and edge perturbations?

This is a candidate question, not a confirmed novelty claim. The related-work matrix must be completed first because recent work already studies graph module design, Leiden seed instability, and KG-extraction reproducibility.

## 2. Dataset protocol

- Dataset: HotpotQA `distractor` / `validation`.
- First expanded run: 500 questions; confirm sample distribution across `level` and `type`.
- If the distribution and pipeline are sound, expand to 1,000 questions.
- Use a full-split shuffle (the current loader's buffer formula gives 10,000 for 500 questions, exceeding the 7,405-row validation split); set and record a new seed such as 2027.
- Save to `data/pilot_500/` and `data/pilot_1000/`; never overwrite `data/processed/` or `data/pilot_50/`.
- Save the complete dataset manifest, question IDs, context passages, and supporting-fact labels. Keep gold labels out of graph construction and retrieval configuration.
- Report counts by difficulty and question type, context length, passage count, number of gold supporting passages, and any missing evidence mappings.

## 3. Graph-construction factors

Do not vary every factor at once. Begin with a small, controlled set and keep the underlying passages fixed.

| Factor | Initial levels to compare | Required record |
|---|---|---|
| Entity co-occurrence granularity | Same-sentence edges; same-passage edges | Exact extraction unit and edge count |
| Edge weighting | Binary; co-occurrence frequency | Weight definition and distribution |
| Sparsification | No pruning; documented threshold/top-k rule | Threshold, retained edges, coverage |
| Entity resolution | Case/whitespace normalization baseline; conservative alias merge as a separate ablation | Resolution rule and manual spot-check |
| Node unit | Entity graph; passage/chunk graph; entity–passage bipartite graph | Node type, node count, mapping to source passages |
| Edge semantics | Co-occurrence; semantic-similarity edges; typed relations if reliable extraction is available | Model/prompt/version, similarity threshold or schema |

**Completed pilot ablation:** the saved 50-question entity mappings were reused to compare same-sentence and same-passage co-occurrence while holding the entity node set fixed. This changed the graph from 7,240 to 22,479 edges and the number of connected components from 541 to 27. Ten-seed Louvain partition agreement and TF-IDF-seeded retrieval were evaluated for both variants. The pilot retrieval differences were small and their paired question-level confidence intervals included or touched zero. Full details are in `docs/pilot50_reproducibility_audit.md` and `results/pilot_50/graph_construction_variants/`. This does not yet cover dense retrieval or generated answers.

Because node types differ between some variants, do not compare raw ARI on incompatible node sets. Use common-node ARI/NMI only where node identity aligns. For different graph representations, compare coverage, component/degree statistics, edge overlap where meaningful, retrieval candidate overlap, and downstream evidence/answer metrics.

## 4. Clustering and graph stability

For each construction variant:

- Save original-graph Leiden and Louvain partitions for at least 10 seeds. The Louvain-only 10-seed version has now been run on the 50-question graph; its preliminary results are recorded in `docs/pilot50_reproducibility_audit.md`.
- Run the matching Leiden 10-seed experiment (not yet run in the audit runtime because `python-igraph` and `leidenalg` were unavailable).
- Compute pairwise cross-seed ARI and NMI on the same node universe.
- Compute retrieval-candidate Jaccard and retrieved-evidence Jaccard across seeds per question.
- Apply fixed random and high-betweenness edge-removal budgets (e.g. 1%, 5%, 10%) with multiple perturbation seeds.
- Record both unweighted and explicitly defined weighted critical-edge conditions if used. For weighted shortest-path betweenness, define edge distance (e.g. inverse strength) rather than passing co-occurrence strength directly as a distance.
- Preserve all exact seeds, package versions, configs, partitions, removed-edge IDs, and run logs.

Do not interpret ARI = 1.0 from a partition compared with itself as a stability result.

## 5. Retrieval experiments

Use matched queries, context corpus, and retrieval budgets across systems.

1. Lexical baseline: existing TF-IDF full-corpus retrieval.
2. Dense baseline: one compact dense encoder (e.g. BGE-small or Contriever) with fixed model revision and cached passage embeddings.
3. Graph-assisted retrieval: use graph expansion/candidate selection, then rerank with the same dense retriever used by the dense baseline.
4. Chunk-graph baseline: build a graph over passage/chunk nodes and retrieve connected or similar chunks.
5. One established GraphRAG implementation: evaluate LightRAG or Microsoft GraphRAG only after checking model/runtime compatibility and documenting its configuration. LightRAG documents Ollama support for both LLM and embeddings; Microsoft GraphRAG's current getting-started guide specifies Python 3.10–3.12 and warns that indexing can consume substantial LLM resources.

Primary metrics:

- Supporting-evidence Recall@k (e.g. k = 5 and 10).
- All-gold-supports-retrieved rate@k.
- Candidate-generation recall before final reranking.
- MRR/nDCG if the ranking target is specified and implementation is verified.
- Candidate-pool size, latency, and memory/cost as efficiency measures.

Keep the first-stage seed budget, final context budget, and reranker identical across compared methods unless the experiment explicitly studies those choices.

## 6. Generation evaluation

- Use a fixed local generator already available in the user's workflow (e.g. Ollama Gemma 3 4B), with temperature 0 and fixed prompts/version.
- Generate answers from identical top-k evidence budgets for all methods.
- Evaluate HotpotQA answer Exact Match and token-level F1 using a documented normalization implementation.
- Add an automated faithfulness measure that checks answer claims against retrieved evidence; document the judge/model/prompt and do not treat LLM-judge scores as ground truth.
- Manually audit a stratified subset of examples (including failures and disagreements) for unsupported claims and missing bridge evidence.
- Save raw generated answers, the exact retrieved passage IDs/text, prompts, model tags, decoding options, and evaluation outputs.

## 7. Statistical analysis

- Independent sampling unit: question. Repeated graph/cluster seeds are repeated measurements nested within each question.
- Average repeated seed outcomes within question for the primary question-level comparison, or use a hierarchical/mixed-effects model if seed-level effects are estimated separately.
- Report 95% paired bootstrap confidence intervals for differences against baselines, with bootstrap resampling over questions.
- Pre-register primary comparisons/metrics before large runs; use Holm correction or another stated correction for families of secondary comparisons.
- Report effect sizes, confidence intervals, exact question counts, and distributions; do not equate a small numerical improvement with significance.

## 8. Compute-aware implementation order

1. Correct the dataset sample and validate the 500-question manifest.
2. Run the matching 10-seed Leiden stability and graph-construction ablation once `python-igraph` and `leidenalg` are available; current audit outputs cover Louvain.
3. Re-run the existing TF-IDF baseline and graph-assisted retrieval at 500 questions with 10 seeds.
4. Add BGE-small or Contriever dense retrieval, cache embeddings once, and run matched retrieval comparisons.
5. Extend the graph-construction matrix beyond edge window granularity (e.g. binary vs frequency weighting, then chunk-node graph) one factor at a time.
6. Add fixed-generator evaluation and faithfulness checks on a pilot subset, then scale if runtime and output quality are acceptable.
7. Add one established GraphRAG baseline after verifying compatibility with the local Ollama/Python setup.
8. Finalize the full literature comparison and only then state the exact contribution.

## 9. Reproducibility command template

The first 500-question data load should be run from the project root:

```bash
python -m src.load_data --split validation --limit 500 --seed 2027 --output-dir data/pilot_500
```

Then build the graph without overwriting the earlier pilots:

```bash
python -m src.build_graph \
  --passages data/pilot_500/passages.json \
  --graph-out data/pilot_500/entity_graph.graphml \
  --entity-passages-out data/pilot_500/entity_passages.json \
  --passage-entities-out data/pilot_500/passage_entities.json
```

The perturbed retrieval script recomputes the community partitions for each supplied seed, so use explicit paths and a new results folder:

```bash
python -m src.evaluate_perturbed_retrieval \
  --questions data/pilot_500/questions.json \
  --passages data/pilot_500/passages.json \
  --graph data/pilot_500/entity_graph.graphml \
  --entity-passages data/pilot_500/entity_passages.json \
  --passage-entities data/pilot_500/passage_entities.json \
  --output-dir results/pilot_500/perturbed_retrieval \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --removal-fraction 0.05 \
  --seed-k 2 \
  --k-values 5 10
```

This command reruns only the current TF-IDF candidate-filtering experiment. It does not implement dense retrieval or generation. Those should be added as separate, independently logged stages rather than implied by this run.

## 10. Current stopping rule

Do not send a results-backed progress email to the professor on the basis of the existing 50-question pilot alone. It is useful preliminary evidence but has a hard-only sample, TF-IDF-only retrieval, no answer generation, no cross-seed baseline stability analysis, and a small question count. Send an update after the 500/1,000-question evaluation, dense retrieval and generation stages are complete enough to report with reproducible artifacts and limitations.
