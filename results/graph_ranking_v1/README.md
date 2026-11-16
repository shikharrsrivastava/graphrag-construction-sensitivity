# Phase C outputs (graph-aware ranking)

| Path | What it is |
|---|---|
| `hotpot498/`, `musique500/` | Pre-registered full runs: `paired_headline_contrasts.csv` (confirmatory), `retrieval_summary.csv`, `run_config.json`, `validation_checks.json`. |
| `musique500_rep/` | Registered replication on a disjoint MuSiQue sample; `replication_contrasts.csv` holds the 4 registered tests. |
| `analysis_final/`, `figures/final/` | **Final** analysis tables and figures (all three samples). Use these. |
| `analysis/`, `figures/*.png` | Earlier snapshot made before the replication existed (two samples only). Kept for provenance, superseded by `*_final`. |
| `smoke_hotpot/` | 20-question smoke test of the pre-registered configs only. |
| `smoke_hotpot_v2/` | 20-question smoke test after the exploratory leave-one-out variants were added. |

Per-question raw records (`question_results.jsonl`) and the embedding/extraction cache are kept locally and not committed, following the repository's data policy. Every file except the headline and replication contrasts is EXPLORATORY.
