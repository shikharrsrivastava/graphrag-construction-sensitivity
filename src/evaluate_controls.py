"""Phase C-2 controls (see PREREGISTRATION_controls.md).

Adds hybrid BGE+BM25, a leave-one-out second-hop hybrid, and an entity graph without
title nodes. Reuses Phase C caches and code unchanged. Writes only to a new directory.
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
from src.evaluate_dense_graph import (
    MODEL_ID, MODEL_REVISION, bootstrap_mean_ci, dump_csv, encode_cached, encode_queries_cached,
    ensure_extraction_cache, holm_adjust, resolve_model_revision,
)
from src.evaluate_graph_ranking import (
    HEADLINE_LAMBDA, SEED_K, STAT_SEED, bm25_scores, build_passage_entity_graph, combine,
    graph_passage_scores_loo, paired_diff, rank_contexts, row_for,
)
from src.evaluate_question_local import passage_text

RUNS = {"hotpot498": "data/pilot_500_v2", "musique500": "data/musique_500", "musique500_rep": "data/musique_500_rep"}
GROUPS = {"hotpot498": ["hotpot498"], "musique_pooled": ["musique500", "musique500_rep"]}
CONTROL_CONFIGS = ["hybrid_bge_bm25", "hybrid_hop2", "ppr_entity_loo_notitle"]
TESTS = (("hybrid_bge_bm25", "vanilla_bge"), ("hybrid_hop2", "vanilla_bge"), ("ppr_entity_loo", "hybrid_hop2"),
         ("ppr_entity_loo_notitle", "vanilla_bge"), ("ppr_entity_loo", "ppr_entity_loo_notitle"))


def hop2_query(question: str, seed: dict[str, Any]) -> str:
    return f"{question} {passage_text(seed)}"


def hop2_scores(contexts: list[str], seeds: list[str], h_by_seed: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean over seeds s != p of the hybrid score of p under the query built from s."""
    out = {}
    for pid in contexts:
        others = [s for s in seeds if s != pid]
        out[pid] = float(np.mean([h_by_seed[s][pid] for s in others]))
    return out


def strip_titles(passage_by_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Blank every title so build_passage_entity_graph adds no title entity nodes."""
    return {pid: {**p, "title": ""} for pid, p in passage_by_id.items()}


def evaluate_question(q, passage_by_id, notitle_by_id, extraction_by_pid, passage_emb, query_emb, hop2_emb):
    qid = str(q["question_id"])
    contexts = list(map(str, q["context_passage_ids"]))
    matrix = np.vstack([passage_emb[pid] for pid in contexts]).astype(np.float32, copy=False)
    bge = {pid: float(s) for pid, s in zip(contexts, matrix @ query_emb)}
    bge_rank = rank_contexts(contexts, bge)
    seeds = bge_rank[:SEED_K]
    bm = bm25_scores(str(q["question"]), contexts, passage_by_id)
    rows = []

    hybrid1 = combine(bge, bm, 0.5)
    rows.append(row_for(q, "hybrid_bge_bm25", rank_contexts(contexts, hybrid1, bge), {"seeds": seeds}))

    h_by_seed = {}
    for s in seeds:
        e = hop2_emb[f"{qid}::hop2::{s}"]
        bge_s = {pid: float(x) for pid, x in zip(contexts, matrix @ e)}
        bm_s = bm25_scores(hop2_query(str(q["question"]), passage_by_id[s]), contexts, passage_by_id)
        h_by_seed[s] = combine(bge_s, bm_s, 0.5)
    final = combine(hybrid1, hop2_scores(contexts, seeds, h_by_seed), 0.5)
    rows.append(row_for(q, "hybrid_hop2", rank_contexts(contexts, final, bge), {"seeds": seeds}))

    node_of_pid = {pid: f"p::{pid}" for pid in contexts}
    g_title = build_passage_entity_graph(contexts, extraction_by_pid, passage_by_id, qid)
    ranked_title = rank_contexts(contexts, combine(bge, graph_passage_scores_loo(g_title, node_of_pid, seeds), HEADLINE_LAMBDA), bge)
    g_plain = build_passage_entity_graph(contexts, extraction_by_pid, notitle_by_id, qid)
    ranked_plain = rank_contexts(contexts, combine(bge, graph_passage_scores_loo(g_plain, node_of_pid, seeds), HEADLINE_LAMBDA), bge)
    rows.append(row_for(q, "ppr_entity_loo_notitle", ranked_plain,
                        {"seeds": seeds, "graph_node_count": g_plain.number_of_nodes(), "graph_edge_count": g_plain.number_of_edges(),
                         "title_graph_node_count": g_title.number_of_nodes(), "title_graph_edge_count": g_title.number_of_edges()}))
    return rows, ranked_title


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_dataset(name: str, results_root: Path, out_root: Path, cache_dir: Path, model) -> list[dict[str, Any]]:
    data_dir = ROOT / RUNS[name]
    stored = load_jsonl(results_root / name / "question_results.jsonl")
    qids = list(dict.fromkeys(r["question_id"] for r in stored))
    stored_rank = {(r["question_id"], r["config"]): r["ranked_passage_ids"] for r in stored}
    questions_all = {str(q["question_id"]): q for q in read_json(data_dir / "questions.json")}
    questions = [questions_all[qid] for qid in qids]
    passage_by_id = {str(p["passage_id"]): p for p in read_json(data_dir / "passages.json")}
    needed = sorted({str(pid) for q in questions for pid in q["context_passage_ids"]})
    extraction, _ = ensure_extraction_cache(passage_by_id, needed, cache_dir / "entity_extractions.jsonl")
    revision = resolve_model_revision(model, MODEL_REVISION)
    if revision != MODEL_REVISION:
        raise RuntimeError(f"Resolved BGE revision {revision} != pinned {MODEL_REVISION}")
    emb_dir = cache_dir / "embeddings" / revision
    passage_emb, _ = encode_cached(model, {pid: passage_text(passage_by_id[pid]) for pid in needed}, emb_dir / "passages.npz", revision)
    query_emb, _ = encode_queries_cached(model, questions, emb_dir / "queries.npz", revision)

    # Second-hop queries need BGE seeds, which come from the question-only scores.
    hop_questions = []
    for q in questions:
        contexts = list(map(str, q["context_passage_ids"]))
        m = np.vstack([passage_emb[pid] for pid in contexts]).astype(np.float32, copy=False)
        scores = {pid: float(s) for pid, s in zip(contexts, m @ query_emb[str(q["question_id"])])}
        for s in rank_contexts(contexts, scores)[:SEED_K]:
            hop_questions.append({"question_id": f"{q['question_id']}::hop2::{s}",
                                  "question": hop2_query(str(q["question"]), passage_by_id[s])})
    hop2_emb, hop_info = encode_queries_cached(model, hop_questions, emb_dir / "hop2_queries.npz", revision)
    print(f"{name}: hop2 queries new={hop_info['new_query_count']} cached={hop_info['cache_hit_count']}", flush=True)

    notitle_by_id = strip_titles(passage_by_id)
    rows, mismatches = [], []
    for q in questions:
        new_rows, ranked_title = evaluate_question(q, passage_by_id, notitle_by_id, extraction, passage_emb,
                                                   query_emb[str(q["question_id"])], hop2_emb)
        if ranked_title != stored_rank[(str(q["question_id"]), "ppr_entity_loo")]:
            mismatches.append(str(q["question_id"]))
        rows += new_rows
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if mismatches:
        (out_dir / "failure.json").write_text(json.dumps({"error": "ppr_entity_loo recomputation differs from stored Phase C ranking",
                                                          "question_ids": mismatches[:50], "count": len(mismatches)}, indent=2) + "\n")
        raise AssertionError(f"{name}: reproducibility gate failed for {len(mismatches)} questions")
    with (out_dir / "control_results.jsonl").open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "validation_checks.json").write_text(json.dumps({
        "question_count": len(questions), "control_rows": len(rows), "expected_rows": len(questions) * len(CONTROL_CONFIGS),
        "ppr_entity_loo_recomputation_mismatches": 0}, indent=2) + "\n")
    return rows + [r for r in stored if r["config"] in {"vanilla_bge", "bm25", "ppr_entity_loo"}]


def confirmatory(rows, group):
    tests = [{**paired_diff(rows, a, b, "complete_support_at_5"), "group": group, "status": "PREREGISTERED_CONFIRMATORY"} for a, b in TESTS]
    for t, adj in zip(tests, holm_adjust([t["p_value"] for t in tests])):
        t.update(holm_adjusted_p=adj, holm_family="5 tests within group", significant_holm_0_05=bool(adj < 0.05))
    return tests


def exploratory(rows, group, label):
    out = []
    for a, b in TESTS:
        d = paired_diff(rows, a, b, "recall_at_5")
        out.append({**d, "group": group, "subset": label, "status": "EXPLORATORY (unadjusted p)"})
    return out


def summary(rows, group):
    out = []
    for cfg in ["bm25", "vanilla_bge", *CONTROL_CONFIGS, "ppr_entity_loo"]:
        for m in ("complete_support_at_5", "recall_at_5"):
            vals = [r[m] for r in rows if r["config"] == cfg]
            mean, lo, hi = bootstrap_mean_ci(vals, STAT_SEED + sum(map(ord, f"ctl:{group}:{cfg}:{m}")))
            out.append({"group": group, "config": cfg, "metric": m, "n_questions": len(vals), "mean": mean, "ci95_low": lo, "ci95_high": hi})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/graph_ranking_v1")
    parser.add_argument("--output-dir", default="results/graph_ranking_v1/controls_v1")
    args = parser.parse_args()
    results_root, out_root = ROOT / args.results_root, ROOT / args.output_dir
    if (out_root / "control_contrasts.csv").exists():
        raise FileExistsError(f"Refusing to overwrite {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu")
        rows_by_run = {name: run_dataset(name, results_root, out_root, results_root / "cache", model) for name in RUNS}
    except Exception as exc:
        (out_root / "failure.json").write_text(json.dumps({"error_type": type(exc).__name__, "error": str(exc),
                                                           "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise
    confirm, explore, summ = [], [], []
    for group, members in GROUPS.items():
        rows = [r for m in members for r in rows_by_run[m]]
        confirm += confirmatory(rows, group)
        explore += exploratory(rows, group, "pooled")
        summ += summary(rows, group)
        if len(members) > 1:
            for m in members:
                explore += exploratory(rows_by_run[m], group, m)
    dump_csv(out_root / "control_contrasts.csv", confirm)
    dump_csv(out_root / "exploratory_recall_and_per_sample.csv", explore)
    dump_csv(out_root / "control_summary.csv", summ)
    (out_root / "run_config.json").write_text(json.dumps({"preregistration": "PREREGISTRATION_controls.md", "lambda": HEADLINE_LAMBDA,
                                                          "seed_k": SEED_K, "total_seconds": time.perf_counter() - started}, indent=2) + "\n")
    for t in confirm:
        print(f"{t['group']:15s} {t['comparison']:44s} {t['mean_b']:.3f}->{t['mean_a']:.3f} d={t['mean_paired_difference']:+.4f} "
              f"[{t['diff_ci95_low']:+.4f},{t['diff_ci95_high']:+.4f}] +{t['questions_a_better']}/-{t['questions_b_better']} "
              f"p={t['p_value']:.5f} holm={t['holm_adjusted_p']:.5f}")


if __name__ == "__main__":
    main()
