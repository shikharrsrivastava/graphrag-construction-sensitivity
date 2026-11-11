# Experimental design and result provenance

## Sampling and analysis unit

The primary Phase A/B cohort is the 498-question subset of `data/pilot_500_v2` with exactly ten question-specific context passages. The two questions with two contexts are analyzed separately in Stage A and excluded from Phase B. All are HotpotQA `hard`; the set is not a probability sample of all HotpotQA or multi-hop questions. The question is the statistical unit. Two gold supporting passage IDs are matched by exact passage ID for each question.

## Current primary tracks

| Stage | Canonical inputs and outputs | Design |
|---|---|---|
| Stage A primary | `data/pilot_500_v2`; `results/question_local_v1/` | 498-question local TF-IDF, two variable-context questions separately; secondary shared-global-graph comparison is retained as a distinct track. |
| Phase B pilot | `results/dense_graph_v1/pilot50_final/` | Seed-2027 stratified sample of 50 primary questions, seven methods. |
| Phase B primary | `results/dense_graph_v1/primary498_final/` | All 498 primary questions, seven methods, same retrieval budgets and metrics as pilot. |

Phase B’s seven methods are local TF-IDF, vanilla BGE, sentence-frequency entity graph, passage-frequency entity graph, binary sentence entity graph, strongest-half sparsified sentence graph, and BGE chunk union-kNN with k=2. Graph methods use top-five BGE seed passages and fixed BGE reranking. See [methodology](methodology.md) and [Phase B validation](../results/dense_graph_v1/PHASE_B_VALIDATION.md) for the precise implementation and all results.

## Statistical family

Phase B bootstrap intervals use 10,000 resamples of question rows and percentile limits. Paired differences use two-sided sign-flip Monte Carlo tests with 20,000 draws and plus-one correction. Holm adjustment is across 69 predeclared method/outcome contrasts. The pilot is a feasibility sample, not a second independent confirmation of the full run; its questions are included in the 498-question cohort.

Stage A has its own bootstrap and paired comparison family; its configuration and timing semantics are saved in `results/question_local_v1/run_config.json`. Do not combine Stage A Track A and Track B estimates as if they used the same graph information scope.

## Historical and exploratory outputs

These paths remain to preserve the project’s research record. Their presence does not make them canonical or directly comparable to Stage A/B.

| Path | Role and status |
|---|---|
| Top-level `results/` retrieval/perturbation/partition outputs | Earliest 15-question pooled-corpus exploration using `data/processed` defaults; exploratory. |
| `results/pilot_50/audit/` | Early 50-question audit and retrieval summary. |
| `results/pilot_50/graph_construction_variants/` | Historical passage-vs-sentence graph/community ablation on pooled 50-question data; not Phase B question-local graph construction. |
| `results/pilot_50/seed_stability_*` | Historical Leiden/Louvain seed and top-k overlap analyses. |
| `results/pilot_50/perturbed_retrieval/` | Earlier perturbation retrieval study. |
| `results/pilot_500/retrieval_quality_summary/` | Historical pooled-corpus TF-IDF-seeded retrieval summaries. |
| `results/pilot_500/seed_stability_raw_leiden/`, `seed_stability_raw_louvain/` | Historical 500-question partition/retrieval seed analyses. |
| `results/dense_graph_v1/pilot50_final/` | Canonical audited Phase B pilot. |
| `results/dense_graph_v1/primary498_final/` | Canonical audited Phase B full run. |
| `results/dense_graph_v1/pilot50/` | Preserved failed initial attempt; model download failed offline and revision was recorded as unknown. Do not use as successful run output. |
| `results/dense_graph_v1/pilot50_verified/` | Preserved superseded attempt; earlier audit found 50 sparsified-edge mismatches caused by nondeterministic tie handling. Replaced by deterministic `pilot50_final/`. |

The corresponding failures and revisions are explained in `results/dense_graph_v1/PHASE_B_VALIDATION.md`; they were retained rather than deleted. `results/dense_graph_v1/cache/embeddings/unknown/` is an old unpinned cache and is not used by canonical results. The revision-keyed embedding cache and passage-local entity cache are retained because the independent audit reconstructs rankings and graphs from them.

## Data duplicates and path compatibility

Byte-for-byte checks found:

- `data/pilot_500/passages.json` and `data/pilot_500_v2/passages.json` are identical. The v2 package adds preserved question-context mappings and stronger provenance metadata; the v2 path is canonical for Stage A/B.
- Every file in `data/processed/` is an exact duplicate of its counterpart in `data/pilot_15/`. Existing historical scripts default to `data/processed`, so removing or renaming it would break those entry points.
- The `data/pilot_15/` sample is not the same sampling event as `data/pilot_50/` or `data/pilot_500/`.

Do not remove these copies as a size optimization without updating documented legacy paths and verifying their historical role. Current experiment scripts rely on repository-root paths defined in `src/common.py` and defaults under `data/` and `results/`; those trees have therefore been retained in place.

## Licensing and publication boundary

Dataset manifests identify `hotpotqa/hotpot_qa`, `distractor`, validation. The upstream dataset card identifies CC BY-SA 4.0. The BGE model is separately identified as MIT by its model card. The repository contains no project-level code license. This release-preparation pass does not select one on the author’s behalf. Check attribution, ShareAlike and source-content obligations before publishing the bundled benchmark-derived artifacts.

The current branch has no remote. The historical Phase B review ZIP is present in an unreachable local commit, not in the current `main` history. The ignored ZIP also remains in the local working directory. No history rewriting is authorized. Historical JSON outputs and logs also contain absolute workstation paths in reachable commits. These are publication review items, not grounds for silently rewriting prior research outputs or Git history.
