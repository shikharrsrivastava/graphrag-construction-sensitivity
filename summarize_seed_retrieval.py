#!/usr/bin/env python3
"""Summarize 500-question retrieval quality from separate Leiden/Louvain seed runs.

Reads the raw per-seed retrieval CSVs created by src.evaluate_seed_stability.py.
Keeps TF-IDF and one-hop controls once, averages graph-based retrieval over seeds
within each question, then bootstraps over questions (not seed runs).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REQUIRED = {
    "question_id", "method", "k", "seed", "candidate_pool_size",
    "supporting_passage_recall_at_k", "all_gold_supports_retrieved_at_k",
    "gold_supporting_passage_ids", "candidate_gold_passage_ids",
}


def id_set(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {x for x in str(value).split(";") if x}


def ci_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        return float(values.mean()), float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[indices].mean(axis=1)
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def paired_ci(diff: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if not len(diff):
        return float("nan"), float("nan"), float("nan")
    if len(diff) == 1:
        return float(diff.mean()), float(diff[0]), float(diff[0])
    indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = diff[indices].mean(axis=1)
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leiden-csv",
        default="results/pilot_500/seed_stability_raw_leiden/seed_retrieval_raw.csv",
    )
    parser.add_argument(
        "--louvain-csv",
        default="results/pilot_500/seed_stability_raw_louvain/seed_retrieval_raw.csv",
    )
    parser.add_argument("--output-dir", default="results/pilot_500/retrieval_quality_summary")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    args = parser.parse_args()

    leiden_path = Path(args.leiden_csv).expanduser()
    louvain_path = Path(args.louvain_csv).expanduser()
    for path in (leiden_path, louvain_path):
        if not path.exists():
            print(f"ERROR: Required CSV not found: {path}", file=sys.stderr)
            print("Check that both seed-stability runs completed and saved their output files.", file=sys.stderr)
            return 2

    leiden_all = pd.read_csv(leiden_path)
    louvain_all = pd.read_csv(louvain_path)
    for path, frame in ((leiden_path, leiden_all), (louvain_path, louvain_all)):
        missing = REQUIRED - set(frame.columns)
        if missing:
            print(f"ERROR: {path} is missing columns: {sorted(missing)}", file=sys.stderr)
            return 2

    # The first run contains TF-IDF/one-hop plus Leiden rows. Keep controls once
    # (seed 42); average the clustering-based methods across all 10 seeds later.
    controls = leiden_all[
        leiden_all["method"].isin(["tfidf_baseline", "one_hop_graph"])
        & (leiden_all["seed"] == leiden_all["seed"].min())
    ].copy()
    leiden = leiden_all[leiden_all["method"] == "leiden_community"].copy()
    louvain = louvain_all[louvain_all["method"] == "louvain_community"].copy()
    if controls.empty or leiden.empty or louvain.empty:
        print("ERROR: Expected TF-IDF, one-hop, Leiden and Louvain rows were not all found.", file=sys.stderr)
        print("Methods in Leiden CSV:", sorted(leiden_all["method"].unique()), file=sys.stderr)
        print("Methods in Louvain CSV:", sorted(louvain_all["method"].unique()), file=sys.stderr)
        return 2

    raw = pd.concat([controls, leiden, louvain], ignore_index=True)
    raw["candidate_generation_recall"] = raw.apply(
        lambda r: len(id_set(r["candidate_gold_passage_ids"]))
        / max(len(id_set(r["gold_supporting_passage_ids"])), 1),
        axis=1,
    )

    # Independent unit is the question. Repeated clustering seeds are averaged
    # within question before uncertainty is estimated across questions.
    qlevel = raw.groupby(["method", "k", "question_id"], as_index=False).agg(
        recall_at_k=("supporting_passage_recall_at_k", "mean"),
        all_gold_supports_rate=("all_gold_supports_retrieved_at_k", "mean"),
        candidate_generation_recall=("candidate_generation_recall", "mean"),
        candidate_pool_size=("candidate_pool_size", "mean"),
        seed_runs=("seed", "nunique"),
    )

    rng = np.random.default_rng(args.bootstrap_seed)
    summary_rows = []
    for (method, k), block in qlevel.groupby(["method", "k"], sort=True):
        recall, recall_lo, recall_hi = ci_mean(block["recall_at_k"].to_numpy(), rng, args.bootstrap)
        all_gold, all_lo, all_hi = ci_mean(block["all_gold_supports_rate"].to_numpy(), rng, args.bootstrap)
        candidate, cand_lo, cand_hi = ci_mean(block["candidate_generation_recall"].to_numpy(), rng, args.bootstrap)
        pool = float(block["candidate_pool_size"].mean())
        summary_rows.append({
            "method": method,
            "k": int(k),
            "n_questions": int(block["question_id"].nunique()),
            "mean_supporting_passage_recall": recall,
            "recall_ci95_low": recall_lo,
            "recall_ci95_high": recall_hi,
            "all_gold_supports_rate": all_gold,
            "all_gold_ci95_low": all_lo,
            "all_gold_ci95_high": all_hi,
            "candidate_generation_recall": candidate,
            "candidate_generation_ci95_low": cand_lo,
            "candidate_generation_ci95_high": cand_hi,
            "mean_candidate_pool_size": pool,
            "mean_seed_runs_per_question": float(block["seed_runs"].mean()),
        })
    summary = pd.DataFrame(summary_rows)

    # Paired question-level differences against the full-corpus TF-IDF baseline.
    paired_rows = []
    for k in sorted(qlevel["k"].unique()):
        block = qlevel[qlevel["k"] == k]
        for metric in ("recall_at_k", "all_gold_supports_rate"):
            pivot = block.pivot(index="question_id", columns="method", values=metric)
            if "tfidf_baseline" not in pivot.columns:
                continue
            for method in ("one_hop_graph", "leiden_community", "louvain_community"):
                if method not in pivot.columns:
                    continue
                paired = pivot[["tfidf_baseline", method]].dropna()
                diff = (paired[method] - paired["tfidf_baseline"]).to_numpy(dtype=float)
                mean, lo, hi = paired_ci(diff, rng, args.bootstrap)
                paired_rows.append({
                    "metric": metric,
                    "k": int(k),
                    "comparison": f"{method} - tfidf_baseline",
                    "n_paired_questions": int(len(diff)),
                    "mean_paired_difference": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "questions_positive": int((diff > 0).sum()),
                    "questions_negative": int((diff < 0).sum()),
                    "questions_tied": int((diff == 0).sum()),
                    "interpretation": "question-paired bootstrap; exploratory, not multiplicity-adjusted",
                })
    paired_summary = pd.DataFrame(paired_rows)

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    qlevel.to_csv(out / "retrieval_quality_question_level.csv", index=False)
    summary.to_csv(out / "retrieval_quality_summary.csv", index=False)
    paired_summary.to_csv(out / "retrieval_quality_paired_vs_tfidf.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("Questions represented:", qlevel["question_id"].nunique())
    print("Methods:", ", ".join(sorted(qlevel["method"].unique())))
    print("\nQuestion-level retrieval quality summary (95% bootstrap CIs):")
    print(summary.round(4).to_string(index=False))
    print("\nPaired question-level differences vs TF-IDF (95% bootstrap CIs):")
    print(paired_summary.round(4).to_string(index=False))
    print(f"\nSaved outputs to: {out.resolve()}")
    print("Note: TF-IDF-seeded retrieval and TF-IDF reranking only; no dense encoder or answer generation is included.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
