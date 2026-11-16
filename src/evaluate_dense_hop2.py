"""Phase C-3: dense-only second-hop control (see PREREGISTRATION_controls_v2.md).

Reuses the Phase C caches, including the cached second-hop query embeddings, and
writes only to a new directory.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from src.common import ROOT, read_json
from src.evaluate_controls import GROUPS, RUNS, hop2_query, load_jsonl
from src.evaluate_dense_graph import (
    MODEL_ID, MODEL_REVISION, bootstrap_mean_ci, dump_csv, encode_cached, encode_queries_cached, holm_adjust,
    resolve_model_revision,
)
from src.evaluate_graph_ranking import HEADLINE_LAMBDA, SEED_K, STAT_SEED, combine, minmax, paired_diff, rank_contexts, row_for
from src.evaluate_question_local import passage_text

CONFIG = "dense_hop2"
TESTS = (("dense_hop2", "vanilla_bge"), ("ppr_entity_loo", "dense_hop2"))


def dense_hop2_scores(contexts: list[str], seeds: list[str], dense_by_seed: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean over seeds s != p of minmax(BGE(q2_s, p)); a passage never gets its own seed query."""
    h = {s: minmax(dense_by_seed[s]) for s in seeds}
    return {pid: float(np.mean([h[s][pid] for s in seeds if s != pid])) for pid in contexts}


def evaluate_question(q, passage_emb, query_emb, hop2_emb) -> tuple[dict[str, Any], list[str]]:
    qid = str(q["question_id"])
    contexts = list(map(str, q["context_passage_ids"]))
    matrix = np.vstack([passage_emb[pid] for pid in contexts]).astype(np.float32, copy=False)
    bge = {pid: float(s) for pid, s in zip(contexts, matrix @ query_emb)}
    bge_rank = rank_contexts(contexts, bge)
    seeds = bge_rank[:SEED_K]
    dense_by_seed = {s: {pid: float(x) for pid, x in zip(contexts, matrix @ hop2_emb[f"{qid}::hop2::{s}"])} for s in seeds}
    final = combine(bge, dense_hop2_scores(contexts, seeds, dense_by_seed), HEADLINE_LAMBDA)
    return row_for(q, CONFIG, rank_contexts(contexts, final, bge), {"seeds": seeds}), bge_rank


def run_dataset(name, results_root, out_root, cache_dir, model):
    data_dir = ROOT / RUNS[name]
    stored = load_jsonl(results_root / name / "question_results.jsonl")
    stored_rank = {(r["question_id"], r["config"]): r["ranked_passage_ids"] for r in stored}
    qids = list(dict.fromkeys(r["question_id"] for r in stored))
    questions_all = {str(q["question_id"]): q for q in read_json(data_dir / "questions.json")}
    questions = [questions_all[qid] for qid in qids]
    passage_by_id = {str(p["passage_id"]): p for p in read_json(data_dir / "passages.json")}
    needed = sorted({str(pid) for q in questions for pid in q["context_passage_ids"]})
    revision = resolve_model_revision(model, MODEL_REVISION)
    if revision != MODEL_REVISION:
        raise RuntimeError(f"Resolved BGE revision {revision} != pinned {MODEL_REVISION}")
    emb_dir = cache_dir / "embeddings" / revision
    passage_emb, _ = encode_cached(model, {pid: passage_text(passage_by_id[pid]) for pid in needed}, emb_dir / "passages.npz", revision)
    query_emb, _ = encode_queries_cached(model, questions, emb_dir / "queries.npz", revision)
    hop_questions = []
    for q in questions:
        contexts = list(map(str, q["context_passage_ids"]))
        m = np.vstack([passage_emb[pid] for pid in contexts]).astype(np.float32, copy=False)
        scores = {pid: float(s) for pid, s in zip(contexts, m @ query_emb[str(q["question_id"])])}
        for s in rank_contexts(contexts, scores)[:SEED_K]:
            hop_questions.append({"question_id": f"{q['question_id']}::hop2::{s}", "question": hop2_query(str(q["question"]), passage_by_id[s])})
    # One cache file per dataset: the Phase C-2 run shared a single file, so each dataset overwrote the previous one.
    hop2_emb, info = encode_queries_cached(model, hop_questions, emb_dir / f"hop2_queries_{name}.npz", revision)
    print(f"{name}: second-hop queries encoded={info['new_query_count']} cached={info['cache_hit_count']}", flush=True)
    rows, bad = [], []
    for q in questions:
        row, bge_rank = evaluate_question(q, passage_emb, query_emb[str(q["question_id"])], hop2_emb)
        if bge_rank != stored_rank[(str(q["question_id"]), "vanilla_bge")]:
            bad.append(str(q["question_id"]))
        rows.append(row)
    if bad:
        raise AssertionError(f"{name}: recomputed BGE ranking differs from stored on {len(bad)} questions")
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "dense_hop2_results.jsonl").open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "validation_checks.json").write_text(json.dumps({"question_count": len(questions), "rows": len(rows),
        "bge_ranking_recomputation_mismatches": 0, "second_hop_queries_encoded": info["new_query_count"], "second_hop_queries_cached": info["cache_hit_count"]}, indent=2) + "\n")
    return rows + [r for r in stored if r["config"] in {"vanilla_bge", "ppr_entity_loo"}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/graph_ranking_v1")
    parser.add_argument("--output-dir", default="results/graph_ranking_v1/controls_v2_dense")
    args = parser.parse_args()
    results_root, out_root = ROOT / args.results_root, ROOT / args.output_dir
    if (out_root / "dense_hop2_contrasts.csv").exists():
        raise FileExistsError(f"Refusing to overwrite {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu")
        rows_by_run = {n: run_dataset(n, results_root, out_root, results_root / "cache", model) for n in RUNS}
    except Exception as exc:
        (out_root / "failure.json").write_text(json.dumps({"error_type": type(exc).__name__, "error": str(exc),
                                                           "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise
    confirm, explore, summ = [], [], []
    for group, members in GROUPS.items():
        rows = [r for m in members for r in rows_by_run[m]]
        tests = [{**paired_diff(rows, a, b, "complete_support_at_5"), "group": group, "status": "PREREGISTERED_CONFIRMATORY"} for a, b in TESTS]
        for t, adj in zip(tests, holm_adjust([t["p_value"] for t in tests])):
            t.update(holm_adjusted_p=adj, holm_family="2 tests within group", significant_holm_0_05=bool(adj < 0.05))
        confirm += tests
        subsets = [("pooled", rows)] + ([(m, rows_by_run[m]) for m in members] if len(members) > 1 else [])
        for label, sub in subsets:
            for a, b in TESTS:
                explore.append({**paired_diff(sub, a, b, "recall_at_5"), "group": group, "subset": label, "status": "EXPLORATORY (unadjusted p)"})
        for cfg in ("vanilla_bge", CONFIG, "ppr_entity_loo"):
            for m in ("complete_support_at_5", "recall_at_5"):
                mean, lo, hi = bootstrap_mean_ci([r[m] for r in rows if r["config"] == cfg], STAT_SEED + sum(map(ord, f"d2:{group}:{cfg}:{m}")))
                summ.append({"group": group, "config": cfg, "metric": m, "mean": mean, "ci95_low": lo, "ci95_high": hi})
    dump_csv(out_root / "dense_hop2_contrasts.csv", confirm)
    dump_csv(out_root / "exploratory_recall_and_per_sample.csv", explore)
    dump_csv(out_root / "dense_hop2_summary.csv", summ)
    (out_root / "run_config.json").write_text(json.dumps({"preregistration": "PREREGISTRATION_controls_v2.md", "lambda": HEADLINE_LAMBDA,
                                                          "seed_k": SEED_K, "total_seconds": time.perf_counter() - started}, indent=2) + "\n")
    for t in confirm:
        print(f"{t['group']:15s} {t['comparison']:34s} {t['mean_b']:.3f}->{t['mean_a']:.3f} d={t['mean_paired_difference']:+.4f} "
              f"[{t['diff_ci95_low']:+.4f},{t['diff_ci95_high']:+.4f}] +{t['questions_a_better']}/-{t['questions_b_better']} "
              f"p={t['p_value']:.5f} holm={t['holm_adjusted_p']:.5f}")


if __name__ == "__main__":
    main()
