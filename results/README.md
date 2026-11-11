# Results directory guide

The public release candidate is aggregate-results-only. Canonical aggregate tables, configurations, independent audit summaries, validation reports, and figures are retained in the proposed index. Raw question-level records, local caches, partition files, and failed/superseded run artifacts remain in the maintainer's working tree but are ignored and excluded from the proposed index. They are not deleted. See [`PUBLIC_RELEASE_INVENTORY.md`](../PUBLIC_RELEASE_INVENTORY.md) for the file-level policy.

## Canonical results

| Path | Contents retained for release |
|---|---|
| `question_local_v1/` | Stage A summary metrics, paired comparisons, run configuration, and validation report. The three raw question-level JSONL files remain local. |
| `dense_graph_v1/pilot50_final/` | Canonical audited 50-question Phase B pilot: aggregate retrieval, graph, efficiency, paired-comparison and stability summaries; run config; checks and audit summary. |
| `dense_graph_v1/primary498_final/` | Canonical audited 498-question Phase B primary run, with the same aggregate artifact classes. |
| `dense_graph_v1/phase_b_validation_audits/` | Independent aggregate audit summaries for both canonical Phase B runs. |
| `dense_graph_v1/PHASE_B_VALIDATION.md` | Phase B methods, findings, audit and limitations report. Raw files referenced by the report are local-only in this candidate. |
| `dense_graph_v1/PILOT_VALIDATION.md` | Pilot validation report. |

The separate consolidated research review is [here](../docs/research/PHASE_B_REVIEW_SUMMARY.md). The saved metrics were calculated from local per-question records. Those records and their embedding/entity caches are not included in this release candidate, so an independent raw-row recomputation requires locally obtaining the data and rerunning the experiment.

## Historical and exploratory aggregate results

Historical pooled-corpus and perturbation studies remain explicitly exploratory. Aggregate summary CSVs, aggregate paired/bootstrap tables, and figures are retained where present under `retrieval/`, `retrieval_debug/`, `retrieval_seed2/`, `perturbations/`, `perturbed_retrieval/`, `pilot_50/`, and `pilot_500/`. They are not interchangeable with the Stage A/B question-local results. Raw question-level files, partition assignments, and pairwise per-run outputs remain local-only.

`dense_graph_v1/pilot50/` and `pilot50_verified/` are failed or superseded runs and are excluded from the release index. Their status and reasons are documented in the Phase B validation report. Canonical results are `pilot50_final/` and `primary498_final/` only.

## Output safety

Do not rerun over a canonical output directory. Stage A and Phase B refuse nonempty output directories; historical scripts may reuse defaults. Use a fresh local output path and keep raw inputs/outputs outside the public index. See [the reproduction guide](../docs/reproduction_guide.md).
