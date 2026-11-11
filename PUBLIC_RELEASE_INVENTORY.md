# Public repository inventory

This inventory describes the published `main` tree. The live GitHub tree contains **120 files**. The repository is code-first and provides aggregate experimental evidence; it does not redistribute raw HotpotQA questions or passages, raw question-level retrieval records, or dataset-derived caches. No project license has been selected, and this inventory is not legal clearance.

## Published contents

### Code, tests, and research documentation

- `src/`, `tests/`, `requirements.txt`, and `summarize_seed_retrieval.py`
- `README.md`, `AGENTS.md`, `CODEX_AUDIT_REPORT.md`, `INTERIM_RESEARCH_REPORT.md`, and `STAGE_A_EXTERNAL_REVIEW.md`
- Methodology, experimental design, reproduction guidance, historical notes, and the consolidated Phase B review under `docs/`
- `data/README.md` and five `dataset_manifest.json` provenance files. These manifests contain sampling metadata and hashes, not question or passage text.
- `results/README.md`, Stage A and Phase B validation reports, aggregate result tables, audit summaries, run configurations, and figures

### Canonical aggregate results

| Published path | Contents |
|---|---|
| `results/question_local_v1/` | Stage A `summary.csv`, `paired_comparisons.csv`, `run_config.json`, and `STAGE_A_VALIDATION.md` |
| `results/dense_graph_v1/pilot50_final/` | Eight aggregate/configuration files: retrieval, graph, efficiency, paired-comparison, stability, config, validation, and independent-audit summaries |
| `results/dense_graph_v1/primary498_final/` | The same eight aggregate/configuration file types for the 498-question run |
| `results/dense_graph_v1/phase_b_validation_audits/` | Independent aggregate audit summaries for the pilot and primary run |
| `results/dense_graph_v1/` | `PHASE_B_VALIDATION.md` and `PILOT_VALIDATION.md` |
| Historical `results/` directories | Aggregate summaries and figures, identified as historical or exploratory in the documentation |

The aggregate results were calculated from local per-question records. Those raw records are not in this repository, so independent raw-row metric recomputation requires separately obtaining and preparing HotpotQA and rerunning the evaluations.

## Not published

The public tree excludes:

- Raw HotpotQA questions, answers, supporting-fact labels, passages, and context mappings.
- Dataset-derived entity graphs, entity-to-passage mappings, and passage-to-entity mappings.
- Raw question-level JSONL/CSV outputs, selected-question lists, pairwise per-question comparisons, and unreviewed partition assignments.
- Entity-extraction and embedding caches, including superseded caches.
- Failed-run outputs and workstation-specific logs.
- Virtual environments, model weights, and local review ZIP bundles.

Copies of research inputs and outputs may remain in the separate local research workspace or backup. Their local retention does not grant redistribution rights.

## Reproduction boundary

The dataset instructions in `data/README.md` explain how to obtain HotpotQA independently and prepare a local sample. A seed does not guarantee byte-identical sampling if upstream data or library behavior changes. Phase B additionally requires the pinned BGE model snapshot and extraction dependencies. The reproduction guide documents these inputs and commands, but a fresh clean-environment rerun is not claimed here. Because raw question-level records are absent, the published checkout alone cannot reproduce every aggregate directly from saved per-question outputs.

## Licensing and attribution questions

- The upstream HotpotQA dataset card identifies CC BY-SA 4.0. Confirm applicable attribution, ShareAlike, underlying-source, database, and institutional requirements before redistributing benchmark text or derived artifacts.
- The BGE model card lists MIT; model weights are not included. Review the upstream model terms independently.
- No project license has been selected for original source code. Public visibility does not itself grant a reuse license.
- Cite HotpotQA and distinguish upstream data/model materials from the project’s code and experimental results.

## Privacy and Git history

The published `main` history has been sanitized. A live audit of its seven reachable commits found no workstation-specific paths, the superseded personal email, raw dataset files, excluded question-level outputs, review ZIPs, or unwanted caches. Local review bundles and the original research workspace are outside the published repository.
