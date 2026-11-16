"""Registered replication tests for Phase C (see PREREGISTRATION_graph_ranking.md)."""
from __future__ import annotations

import argparse
import json

from src.common import ROOT
from src.evaluate_dense_graph import dump_csv, holm_adjust
from src.evaluate_graph_ranking import paired_diff

TESTS = (("ppr_entity_loo", "complete_support_at_5", "primary"), ("ppr_entity_loo", "recall_at_5", "secondary"),
         ("ppr_chunk", "complete_support_at_5", "secondary"), ("ppr_chunk", "recall_at_5", "secondary"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/graph_ranking_v1/musique500_rep")
    args = parser.parse_args()
    run_dir = ROOT / args.run_dir
    out = run_dir / "replication_contrasts.csv"
    if out.exists():
        raise FileExistsError(out)
    with (run_dir / "question_results.jsonl").open(encoding="utf-8") as h:
        rows = [json.loads(line) for line in h]
    tests = [{**paired_diff(rows, cfg, "vanilla_bge", m), "role": role} for cfg, m, role in TESTS]
    for t, adj in zip(tests, holm_adjust([t["p_value"] for t in tests])):
        t.update(holm_adjusted_p=adj, holm_family="4 registered replication tests",
                 replicated=bool(adj < 0.05 and t["mean_paired_difference"] > 0), status="REGISTERED_REPLICATION")
    dump_csv(out, tests)
    for t in tests:
        print(f"{t['comparison']:28s} {t['metric']:22s} {t['mean_b']:.3f}->{t['mean_a']:.3f} d={t['mean_paired_difference']:+.4f} "
              f"[{t['diff_ci95_low']:+.4f},{t['diff_ci95_high']:+.4f}] +{t['questions_a_better']}/-{t['questions_b_better']} "
              f"p={t['p_value']:.5f} holm={t['holm_adjusted_p']:.5f} replicated={t['replicated']}")


if __name__ == "__main__":
    main()
