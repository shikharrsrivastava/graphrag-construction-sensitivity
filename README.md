# GraphRAG Clustering Robustness — Starter Research Prototype

**Provisional working title:** *Graph Construction Sensitivity and Evidence Retrieval in Multi-Hop QA*

This repository is a starter for a small, reproducible pilot studying whether graph-community stability under edge removal is associated with preservation of evidence for multi-hop question answering.

## Research question

When relationships in an entity co-occurrence graph are removed, do community-detection methods that preserve more stable partitions also preserve the evidence needed for multi-hop question answering?

## Scope and honesty

This is an **early experimental scaffold**, not a completed GraphRAG system or a claim of research novelty. The initial scripts load a small HotpotQA sample, build a spaCy entity-co-occurrence graph, run Leiden and Louvain, perturb edges, and report structural measurements. **A preliminary TF-IDF-seeded, TF-IDF-reranked supporting-evidence retrieval pilot is implemented and saved for 15- and 50-question samples. Dense retrieval and generated-answer evaluation are not implemented in this prototype.** Do not describe this as a complete GraphRAG system or use graph metrics alone as proof of answer robustness.

The graph construction is intentionally lightweight: named entities recognized by `en_core_web_sm` are nodes; entities appearing in the same sentence are connected, with edge weight counting co-occurrences. This is a proxy graph, not a gold knowledge graph. NER omissions and entity-resolution errors will affect results.

## Status

| Component | Status |
|---|---|
| Load a fixed-size HotpotQA distractor validation sample | Implemented in starter script; requires dependencies and dataset access |
| Save passages and gold supporting-fact labels separately | Implemented |
| Build entity co-occurrence graph with spaCy | Implemented |
| Run Leiden and Louvain on the same graph | Implemented |
| Random and high-edge-betweenness edge removal | Implemented |
| ARI, largest connected component fraction, community connectivity | Implemented |
| CSV summaries and structural-stability plot | Implemented |
| k-core hierarchy comparison | Planned |
| TF-IDF-seeded graph candidate retrieval / supporting-fact Recall@k | Implemented; 15- and 50-question pilot outputs saved |
| Dense retrieval (Contriever/BGE) | Planned |
| Answer generation / HotpotQA EM and F1 / faithfulness | Planned |
| Literature-backed novelty claim | Not established |

## Setup (macOS / Python)

Python 3.11 or 3.12 is a conservative choice for a research environment. A compatible Python 3.13 setup may also work depending on the versions available for your platform.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the pilot

Run commands from the repository root. The `-m src...` form is intentional.

### 1. Load data

```bash
python -m src.load_data --split validation --limit 500 --seed 2027 --output-dir data/pilot_500
```

This writes:
- `data/pilot_500/passages.json` — deduplicated source passages.
- `data/pilot_500/questions.json` — questions, answers, and gold supporting facts for evaluation.
- `data/pilot_500/dataset_manifest.json` — sample/configuration metadata.

The script uses a fixed seed and Hugging Face streaming mode to avoid intentionally downloading the entire dataset. Streaming availability depends on the upstream dataset repository and network access. Do not put dataset contents with incompatible redistribution terms into a public repository without checking the dataset license and terms.

### 2. Build the entity graph

```bash
python -m src.build_graph \
  --passages data/pilot_500/passages.json \
  --graph-out data/pilot_500/entity_graph.graphml \
  --entity-passages-out data/pilot_500/entity_passages.json \
  --passage-entities-out data/pilot_500/passage_entities.json
```

### 3. Cluster the original graph

```bash
python -m src.cluster_graph \
  --graph data/pilot_500/entity_graph.graphml \
  --output-dir results/pilot_500/partitions \
  --methods leiden louvain \
  --seed 42
```

### 4. Perturb and measure structural robustness

```bash
python -m src.perturb_graph \
  --graph data/pilot_500/entity_graph.graphml \
  --partition-dir results/pilot_500/partitions \
  --output-dir results/pilot_500/structural \
  --baseline-seed 42 \
  --seeds 42 43 44 45 46 \
  --removal-fraction 0.05
```

Outputs:
- `results/pilot_500/structural/structural_robustness_raw.csv` — per-method, per-condition, per-seed measurements.
- `results/pilot_500/structural/structural_robustness_summary.csv` — means and standard deviations across runs.
- `results/pilot_500/structural/figures/structural_robustness.png` — ARI and LCC fraction plots.

### 5. Run the TF-IDF retrieval pilot

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

This repeats the existing TF-IDF-seeded candidate-filtering experiment on 500 questions. It does not yet add dense retrieval or answer generation. For interpretation and limitations, see `docs/next_experiment_protocol.md`.

## Measurements

- **Community count:** number of detected communities.
- **ARI:** adjusted Rand index between the original and perturbed community assignments for the *same node set*. ARI is a partition-similarity metric, not a retrieval metric.
- **LCC fraction:** largest connected component size divided by the original graph's node count.
- **Connected-community fraction:** among non-singleton communities, the fraction whose induced subgraph is connected. NaN means there were no non-singleton communities to score.
- **Edge-removal fraction:** actual fraction of original edges removed.

Random removal uses a different edge sample for each seed. Highest-betweenness removal selects the top-ranked edges from the original graph. If edge betweenness ties at the cutoff, seeded tie-breaking may vary the removed set; otherwise the removed set can be identical across seeds. Repeated runs also vary the clustering algorithm seed.

## Methodological cautions

1. The first sample is small and is not a representative benchmark by itself.
2. The graph is built using off-the-shelf named-entity recognition and sentence-level co-occurrence; it does not encode typed semantic relations.
3. The HotpotQA gold labels are stored separately and are not used for graph construction. Source passages naturally include the evidence being evaluated; labels must not be used to tune graph construction.
4. The current prototype also contains a TF-IDF-seeded graph candidate-filtering retrieval pilot with supporting-passage Recall@k. It does **not** yet establish a statistically reliable advantage, compare dense retrievers, or evaluate generated-answer accuracy/faithfulness.
5. The k-core approach is a hierarchy, not directly interchangeable with a standard flat community-detection algorithm. Add it only with a clearly documented retrieval and comparison rule.
6. Check related work before making novelty claims. The project is an exploratory pilot, not an established novel method.

## Run small unit tests

```bash
python -m unittest discover -s tests -v
```

These tests cover structural helper functions, Jaccard/partition comparisons, and controlled passage-graph construction on synthetic graphs; they do not validate the full HotpotQA-to-GraphRAG pipeline.

## Repository policy

Do not commit `.venv/`, model files, cache directories, or large raw dataset dumps. A reproducible manifest and instructions are preferable. Only publish data/results that you are permitted to redistribute.


## Current experiment status (audited 9 November 2026)

- `data/processed/` and the top-level `results/` pilot outputs contain the earlier 15-question sample.
- `data/pilot_50/` and `results/pilot_50/perturbed_retrieval/` contain the later 50-question sample and its raw retrieval/structural runs. Use explicit file paths when evaluating the 50-question run.
- The 50-question sample is all `hard` (37 bridge, 13 comparison), so it is not a representative validation sample. For the next 500-question run, the existing loader's shuffle-buffer formula yields a buffer of 10,000, larger than the 7,405-example validation split. Save to `data/pilot_500/` and use a new seed.
- The saved retrieval pilot uses TF-IDF for initial seed selection and final candidate reranking; it is not a dense-retriever or full GraphRAG comparison.
- Pilot-50 audit and cross-seed analyses are reproducible with `python -m src.analyze_pilot50` and `python -m src.evaluate_seed_stability`. A same-sentence vs same-passage graph-construction ablation is implemented in `python -m src.evaluate_graph_construction_variants`. See [`docs/pilot50_reproducibility_audit.md`](docs/pilot50_reproducibility_audit.md) and [`docs/next_experiment_protocol.md`](docs/next_experiment_protocol.md).
- The uploaded macOS `.venv` is intentionally not included in clean source bundles. Recreate a local environment from `requirements.txt`; do not copy a virtual environment between machines.


## Cross-seed stability analysis

To compare partitions on the same unchanged graph across multiple algorithm seeds and measure retrieved-passage set overlap, run:

```bash
python -m src.evaluate_seed_stability \
  --questions data/pilot_500/questions.json \
  --passages data/pilot_500/passages.json \
  --graph data/pilot_500/entity_graph.graphml \
  --entity-passages data/pilot_500/entity_passages.json \
  --passage-entities data/pilot_500/passage_entities.json \
  --methods leiden louvain \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --output-dir results/pilot_500/seed_stability
```

This requires `python-igraph` and `leidenalg` for Leiden. The script saves every partition, pairwise ARI/NMI, per-question retrieval-set Jaccard, question-level bootstrap confidence intervals, and summary CSVs. The 50-question audit bundle includes a completed 10-seed Louvain run; the matching Leiden run still requires `python-igraph` and `leidenalg` in the local environment.

To compare same-sentence with same-passage entity co-occurrence on the existing 50-question pilot while preserving the exact entity node set, run:

```bash
python -m src.evaluate_graph_construction_variants \
  --dataset-dir data/pilot_50 \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --output-dir results/pilot_50/graph_construction_variants
```

For a future 500-question sample, change both paths to `data/pilot_500` and `results/pilot_500/graph_construction_variants`. This ablation evaluates TF-IDF-seeded candidate filtering only, not dense retrieval or generated answers. See [`docs/literature_review_matrix.md`](docs/literature_review_matrix.md) for the focused related-work assessment and [`docs/next_local_run.md`](docs/next_local_run.md) for the next Mac run commands.
