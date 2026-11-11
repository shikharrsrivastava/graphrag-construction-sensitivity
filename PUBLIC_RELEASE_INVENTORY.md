# Proposed public-release inventory

**Candidate:** code-first, aggregate-results-only. This document describes the staged public-release proposal; excluded research inputs and outputs remain on the local disk. No project license has been selected. This inventory is not a legal clearance.

The proposed index currently contains **123 files**. **167 previously tracked files** are staged for index removal only; all 167 remain present in the local working tree. The local ignored ZIP bundles and virtual environment were never part of the candidate index.

## Include in the proposed release

### Code, tests, and documentation

- `README.md`, `requirements.txt`, `summarize_seed_retrieval.py`
- All Python modules under `src/` and all five test modules under `tests/`
- Project research reports and methods documentation, including `INTERIM_RESEARCH_REPORT.md`, `STAGE_A_EXTERNAL_REVIEW.md`, `docs/`, and `docs/research/`
- `data/README.md` and the tracked `dataset_manifest.json` files. Manifests contain sample metadata and hashes, not raw benchmark text.
- `results/README.md`, Stage A and Phase B validation reports, and the canonical aggregate result files below

### Canonical aggregate evidence

| Path | Files retained |
|---|---|
| `results/question_local_v1/` | `summary.csv`, `paired_comparisons.csv`, `run_config.json`, `STAGE_A_VALIDATION.md` |
| `results/dense_graph_v1/pilot50_final/` | `retrieval_summary.csv`, `graph_summary.csv`, `efficiency.csv`, `paired_comparisons.csv`, `retrieval_stability_summary.csv`, `run_config.json`, `validation_checks.json`, `independent_audit.json` |
| `results/dense_graph_v1/primary498_final/` | The same eight aggregate artifact types as `pilot50_final/` |
| `results/dense_graph_v1/phase_b_validation_audits/` | `pilot50_independent.json`, `primary498_independent.json` |
| `results/dense_graph_v1/` | `PHASE_B_VALIDATION.md`, `PILOT_VALIDATION.md` |
| `results/` historical folders | Aggregate `*_summary.csv`, aggregate bootstrap/paired comparison tables, and figures (`*.png`) only; all remain explicitly historical/exploratory in the reports. |

The included summary tables and reports state that the exact saved results came from local per-question outputs. The raw files used for recomputation are excluded; an aggregate audit summary is not itself a substitute for rerunning an independent audit.

## Exclude from the proposed release; retain locally

| Local-only files | Reason |
|---|---|
| `data/**/questions.json`, `passages.json`, `question_contexts.json` | Original HotpotQA questions, answers, supporting labels, passage text, and allowed-context mapping. Redistribution remains subject to dataset/source rights review. |
| `data/**/*.graphml`, `entity_passages.json`, `passage_entities.json` | Dataset-derived graph, entity labels, and passage mappings; separate derivative and licensing review required. |
| All `results/**/*.jsonl` | Raw question-level evaluation records or entity extraction artifacts; contain benchmark-linked IDs, metrics, or extracted entities. |
| `results/dense_graph_v1/cache/` | Embedding tensors/metadata and extracted entities; non-aggregate derived caches. Includes the stale `embeddings/unknown/` cache. |
| `results/dense_graph_v1/pilot50/`, `pilot50_verified/` | Failed or superseded Phase B runs; preserved locally and described in the validation report. |
| `results/**/failure.json` | Failed-run traceback contains workstation and virtual-environment paths. |
| `results/**/partitions/`, `same_passage/`, `same_sentence/` | Unreviewed partition/community assignments and graph-construction artifacts. |
| `results/**/*_raw.csv`, named question-level CSVs, `*pairwise.csv`, and `selected_questions.json` | Raw or question/run-level data, rather than aggregate-only evidence. |
| `logs/` | Local execution log includes a workstation cache path. The research reports retain experiment outcomes and commands. |
| `.venv/`, model weight folders, local review ZIPs | Environment/model cache or convenience bundles; not necessary in the code-first release. ZIPs remain ignored and local. |

All excluded tracked files are removed from the proposed Git index only. Their working-tree copies are retained for research; no dataset, output, or cache is deleted. `.gitignore` prevents accidental re-addition.

## Files retained locally for full research reproduction

- All five historical/canonical data directories: `data/pilot_15/`, `pilot_50/`, `pilot_500/`, `pilot_500_v2/`, and `processed/`, including raw JSON, graph exports, and entity mappings.
- Stage A raw files: `primary_question_level.jsonl`, `shared_global_question_level.jsonl`, and `variable_context_question_level.jsonl`.
- Phase B raw `question_results.jsonl`, selected question lists, pairwise stability tables, extraction/embedding caches, and superseded/failed run directories.
- Historical raw question-level and retrieval CSVs, partition assignments, and `logs/progress.md`.
- Both local review ZIPs and the verified pre-publication Git bundle backup.

These local files are useful for review and future audit. Their exclusion from this candidate means the published aggregate results cannot be independently reconstructed from the public checkout alone.

## Licensing and publication questions still open

- The upstream `hotpotqa/hotpot_qa` dataset card identifies CC BY-SA 4.0. Confirm whether the license and notices cover the sampled question/passage files and the Wikipedia-derived source text, and whether any database or institutional requirements apply.
- Confirm redistribution terms for derived entity graphs/mappings, entity-extraction output, and BGE embedding caches before publishing any of them. The aggregate-only candidate excludes these derived artifacts.
- BGE model weights are not included; the upstream BGE model card lists MIT. The source code repository has no selected license; the project owner must choose it.
- Preserve HotpotQA attribution/citation and clearly separate upstream materials from original code and results.

## Git and privacy boundary

The current index proposal removes workstation-specific logs and failed outputs, and sanitizes the local Matplotlib path mention in the retained Stage A validation report. Reachable prior commits still contain workstation paths and the author's email. This candidate does not rewrite history; a GitHub publication would expose those historical values unless the owner separately authorizes a history rewrite. The earlier Phase B ZIP is in an unreachable local commit, not current `main` history.
