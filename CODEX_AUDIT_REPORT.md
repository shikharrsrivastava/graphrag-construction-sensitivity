# Repository Audit — GraphRAG Robustness Study

**Audit date:** 10 November 2026  
**Scope:** read-only source, data, result, environment, and Git inspection. No Python source was modified and no new experiment or test suite was run during this audit.

## Executive summary

This is a substantial exploratory prototype, not yet the full study described in the research handoff. The repository contains a validated 500-question HotpotQA sample with context membership (`data/pilot_500_v2`), an entity co-occurrence graph and ten-seed Leiden/Louvain retrieval stability outputs, plus a 50-question sentence-vs-passage graph construction ablation. The central gap is that the current evaluator and saved 500-question retrieval results use a pooled-corpus candidate universe. They do not implement the required question-local primary track. Dense retrieval, generated-answer EM/F1/faithfulness, and an established GraphRAG baseline are also absent.

## Repository and environment

- Git branch: `main`; working tree was clean at audit start. HEAD is `949fae9` (`Initial GraphRAG research baseline and experiments`).
- Python: existing `.venv`, Python 3.13.7. Installed packages include NetworkX 3.7, spaCy 3.8.18 and `en_core_web_sm` 3.8.0, scikit-learn 1.9.1, pandas 3.1.1, sentence-transformers 6.2.0, torch 2.15.0, python-igraph 1.0.0, and leidenalg 0.12.0. Do not recreate the environment or reinstall packages without a concrete need.
- Host: Apple Silicon macOS (`arm64`). Disk has ample free space. No model weights, embedding arrays, vector indexes, pickle databases, or experiment cache files were found in the repository.
- `ollama` executable exists, but its local API at `127.0.0.1:11434` was not reachable in the restricted audit environment. Model availability and generation feasibility remain unverified.
- `requirements.txt` lists the original lexical/graph stack but does not pin versions or declare sentence-transformers, torch, or generation dependencies despite their presence in the current environment.
- No root `AGENTS.md` existed before this audit.

## Data inventory and validation

| Directory | Questions | Passages | Context membership | Audit finding |
|---|---:|---:|---|---|
| `data/processed` | 15 | 150 | absent | Early pilot; not a suitable current benchmark. |
| `data/pilot_50` | 50 | 492 | absent | All 50 are hard (37 bridge, 13 comparison); graph ablation is exploratory. |
| `data/pilot_500` | 500 | 4,947 | absent | 404 bridge, 96 comparison, all hard. Contains graph and extraction mappings. |
| `data/pilot_500_v2` | 500 | 4,947 | present in question JSON and `question_contexts.json` | Preferred retrieval benchmark input. 498 questions have 10 unique contexts; 2 have 2. All 500 have gold supports in their own contexts; no missing gold titles. |

The graph and passage inventory for `pilot_500` were checked without dumping source data: 28,178 entity nodes, 75,013 edges, 3,753 components, 1,898 isolates, density 0.00018896, mean degree 5.324. This is an NER-derived surface-form entity co-occurrence graph; it is not a typed semantic knowledge graph.

`pilot_500_v2/dataset_manifest.json` records seed 2027, the context-count distribution, 500/500 gold-support coverage, and hashes. The two low-context questions must remain intact and be evaluated in the separate variable-context analysis specified in the handoff. The older `pilot_500` questions lack `context_passage_ids`; that copy cannot support a faithful question-local evaluator without joining to v2 by question ID. Preserve both datasets and their outputs; do not overwrite or regenerate them before comparing provenance.

The audit confirmed unique question IDs, passage counts, and gold passage IDs in the pooled passage corpus for both 500-question datasets. Context membership and evidence coverage were explicitly checked for v2. The historical README statements about data acquisition and sampling should be treated as provenance notes, not independently rerun acquisition evidence.

## Code and reusable work

- `src/load_data.py`: streaming HotpotQA loader, deduplicated passages, question-specific context IDs, supporting-fact labels, and manifest hashes. It is the source of v2-style data. Sampling with streaming buffer size `max(1000, limit*20)` may not produce an exact uniform sample of the full split; document this limitation and preserve the fixed sample.
- `src/build_graph.py`: spaCy NER extraction; normalized entity surface forms; sentence-level co-occurrence count weights; stores entity-to-passage and passage-to-entity mappings. Entity mappings can be cached and reused.
- `src/cluster_graph.py`: Louvain via NetworkX and Leiden via igraph/leidenalg; deterministic seed parameter and weighted modularity partitioning.
- `src/evaluate_retrieval.py`: shared lexical ranking and candidate construction. It refits TF-IDF across the supplied corpus; graph methods use the same lexical top-k seeds and TF-IDF reranking. Candidate sets are formed from global passage mappings.
- `src/evaluate_seed_stability.py`: multi-seed partitions, ARI/NMI, retrieval-set Jaccard, and question-level bootstrap aggregation. Reuse its seed and uncertainty logic after checking candidate scoping.
- `src/evaluate_graph_construction_variants.py`: controlled same-sentence vs same-passage co-occurrence comparison that preserves entity node set; only available on 50 questions and only for entity nodes / TF-IDF retrieval. Does not cover chunk/sentence nodes or semantic edges.
- `src/perturb_graph.py`, `src/evaluate_perturbed_retrieval.py`, `src/analyze_pilot50.py`, and `summarize_seed_retrieval.py`: perturbation, exploratory audit, and question-level summary utilities. These are useful references but not the requested final evaluator/statistical pipeline.
- Existing tests cover small graph construction, structural helpers, Jaccard and partition comparison. The repository documentation says these passed historically; they were not rerun during this read-only audit.

## Existing result inventory and interpretation

- `results/pilot_500/seed_stability_raw_leiden/` and `_louvain/`: ten partitions (seeds 42–51), seed-level retrieval rows, pairwise partition metrics, retrieval Jaccard, and question-level bootstrap CI files. These are reusable historical pooled-corpus outputs.
- `results/pilot_500/retrieval_quality_summary/`: question-level aggregation, bootstrap intervals and paired bootstrap comparisons versus TF-IDF. The reported TF-IDF Recall@5/10 is 0.718/0.878; Leiden 0.7252/0.8571; Louvain 0.7248/0.8598; one-hop 0.726/0.860. Candidate pools average 4,947 for TF-IDF, 1,291 Leiden, 1,371 Louvain and 2,710 one-hop. These numbers are not question-local HotpotQA distractor-context metrics and should not be presented as final benchmark findings.
- Saved 500-question partition stability bootstrap files report mean retrieval Jaccard@5/@10 of 0.9195/0.8576 for Leiden and 0.9095/0.8482 for Louvain. Seed-to-seed repeats are not independent questions; existing scripts generally aggregate within question before bootstrap.
- `results/pilot_50/graph_construction_variants/`: ten-seed Louvain same-sentence vs same-passage graph statistics, retrieval and stability outputs. This is a completed small ablation, not evidence for 500-question performance. The report in `docs/pilot50_reproducibility_audit.md` describes it as exploratory.
- Other legacy outputs under `results/`, `results/perturbations/`, and `results/pilot_50/` include 15/50-question pilots and edge perturbations. Keep them as historical artifacts; do not overwrite them.
- No retrieval output currently contains a question-local-only evaluation as the defined primary track, and no stored dense embeddings/index were found.

## Confirmed methodological gaps / risks

1. **Question scoping:** `src/evaluate_retrieval.evaluate()` ranks against every passage passed to it. Existing 500-question raw files have 500 questions and 4,947-passage pools, not per-question ten-context candidate universes. Filtering final retrieved IDs after ranking would still leak global ranking information; question-local TF-IDF/BGE ranking and graph construction must use only each question's contexts for Track A.
2. **Separate tracks:** preserve two explicitly labelled tracks: (A) question-local graphs and candidate universes, with 498 complete ten-context questions primary and two variable-context cases separate; (B) shared global graph filtered to each question's contexts. The latter still has cross-question graph information and cannot be conflated with A.
3. **Retrieval metrics:** current evaluator records support-passage Recall@k and all-gold retrieval but no Recall@2 in its documented runs, formal candidate-generation recall in its base implementation, or context-filter correctness checks. Current question-specific evidence references in v2 permit these to be implemented and validated.
4. **Statistics:** current summaries include question-level bootstrap intervals and paired bootstrap contrasts in places. No Holm-adjusted family of paired significance tests was found. Seed-pair rows must remain nested within question and never count as independent observations.
5. **Scope:** current graph variation covers co-occurrence window (sentence vs passage) on an entity graph, not chunk/sentence node alternatives, semantic similarity, edge binarization/weighting controls, or sparsification. Typed relationships need precision/quality validation before inclusion.
6. **Generation and baselines:** no answer generation, EM, token F1, faithfulness, manually reviewed faithfulness subset, vanilla dense RAG, chunk-graph baseline, LightRAG, or Microsoft GraphRAG implementation/results were found.
7. **Sampling:** all 500 existing samples are hard-only. The v2 context mapping fixes the evaluation scope but not the difficulty coverage. Claims must be limited to this sample; expansion to 1,000 is a later step after evaluator validation.
8. **Environment reproducibility:** currently installed versions differ from broad minimum-only requirements. Record actual versions in experiment configs and avoid reinstalling the working environment.

## Reproduction commands for saved work

Run from the repository root with the existing `.venv` activated. Commands below are documented entry points; they were not executed in this audit. Avoid commands that overwrite existing output directories unless writing to a new destination.

```bash
# Reanalyse existing 50-question perturbation outputs
python -m src.analyze_pilot50 \
  --dataset-dir data/pilot_50 \
  --input-dir results/pilot_50/perturbed_retrieval \
  --output-dir results/pilot_50/audit

# Recompute multi-seed stability (write to a new directory to preserve saved run)
python -m src.evaluate_seed_stability \
  --questions data/pilot_500/questions.json \
  --passages data/pilot_500/passages.json \
  --graph data/pilot_500/entity_graph.graphml \
  --entity-passages data/pilot_500/entity_passages.json \
  --passage-entities data/pilot_500/passage_entities.json \
  --methods leiden louvain --seeds 42 43 44 45 46 47 48 49 50 51 \
  --output-dir results/pilot_500/seed_stability_reproduced

# Recompute existing 50-question graph-construction ablation (new output path recommended)
python -m src.evaluate_graph_construction_variants \
  --dataset-dir data/pilot_50 --seeds 42 43 44 45 46 47 48 49 50 51 \
  --output-dir results/pilot_50/graph_construction_variants_reproduced

# Rebuild seed-retrieval summary from saved 500-question raw seed files
python summarize_seed_retrieval.py \
  --leiden-csv results/pilot_500/seed_stability_raw_leiden/seed_retrieval_raw.csv \
  --louvain-csv results/pilot_500/seed_stability_raw_louvain/seed_retrieval_raw.csv \
  --output-dir results/pilot_500/retrieval_quality_summary_reproduced
```

The README also documents original end-to-end commands for loading, graph building, clustering, perturbation, and retrieval. They target output paths already populated, so first redirect outputs or back up artifacts if rerunning. The legacy retrieval CLI does not accept `pilot_500_v2` context metadata and is not suitable as the Stage A primary evaluator.

## Recommended next actions

1. Implement and test a new evaluator on `data/pilot_500_v2` with strict passage-ID membership checks, separate local/global tracks, Recall@2/5/10, complete evidence retrieval, candidate-generation recall, and question-level bootstrap CIs. Do not alter saved historical outputs.
2. Run correctness tests on evidence matching and context filtering, then audit the resulting question-level output. Use the 498 ten-context set as the primary analysis and report the two exceptions separately.
3. Implement cached BGE-small-en-v1.5 embeddings and matched TF-IDF/BGE graph-assisted and chunk-graph retrieval. Measure indexing, encoding, query, and retrieval time/memory.
4. Select a limited, justified graph-construction matrix, reuse entity extraction, cache sentence/chunk embeddings, and report topology plus retrieval and stability. Separate construction effects from seed effects.
5. Run a small generation pilot only after checking Ollama service/model availability and resource cost; then fix prompt/model/decoding/evidence budget and validate faithfulness judgments manually.
6. Select one established baseline (LightRAG or Microsoft GraphRAG) based on reproducibility and local resource feasibility. Add question-level paired tests with Holm correction and bridge/comparison analyses.
7. Expand toward 1,000 only after Stage A validation. Keep every run's configuration, raw rows, logs, environment information, and completion status with the final report.

## Audit completion statement

This audit verifies file presence, source behavior by inspection, selected JSON manifests and schemas, graph statistics, result headers, installed package versions, and Git state. It does not verify upstream data redownload reproducibility, rerun historical metric calculations, rerun unit tests, validate literature claims, check Ollama model state, or run any new experiment.
