"""Independent integrity audit for a saved dense-graph retrieval run."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from src.common import ROOT, read_json
from src.evaluate_dense_graph import (
    CONFIGS, CHUNK_K, GRAPH_CONFIGS, SEED_K, SPARSIFY_RETAIN_FRACTION,
    build_chunk_similarity_graph, build_entity_graph_local, chunk_graph_candidates,
    local_graph_candidates, paired_comparisons, rank_by_scores, retrieval_metrics,
    stability_rows, summarize_rows,
)
from src.evaluate_question_local import passage_text, rank_local


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compare_saved_csv(path: Path, expected: list[dict]) -> bool:
    with path.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    if len(saved) != len(expected):
        return False
    for left, right in zip(saved, expected):
        if set(left) != set(right):
            return False
        for key, value in right.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    if not math.isclose(float(left[key]), float(value), rel_tol=1e-12, abs_tol=1e-12):
                        return False
                except ValueError:
                    return False
            elif str(left[key]) != str(value):
                return False
    return True


def audit(run_dir: Path, data_dir: Path) -> dict:
    config = json.loads((run_dir / "run_config.json").read_text())
    rows = read_jsonl(run_dir / "question_results.jsonl")
    selected = json.loads((run_dir / "selected_questions.json").read_text())
    all_questions = read_json(data_dir / "questions.json")
    all_passages = read_json(data_dir / "passages.json")
    qmap = {str(q["question_id"]): q for q in all_questions}
    pmap = {str(p["passage_id"]): p for p in all_passages}
    qids = selected["question_ids"]
    questions = {qid: qmap[qid] for qid in qids}
    extraction = {}
    with (run_dir.parent / "cache/entity_extractions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            extraction[record["passage_id"]] = record
    revision = config["dense_model"]["resolved_revision"]
    emb_dir = run_dir.parent / "cache/embeddings" / revision
    with np.load(emb_dir / "passages.npz", allow_pickle=False) as z:
        passage_embeddings = {pid: v for pid, v in zip(z["passage_ids"].astype(str), z["embeddings"])}
    with np.load(emb_dir / "queries.npz", allow_pickle=False) as z:
        query_embeddings = {qid: v for qid, v in zip(z["question_ids"].astype(str), z["embeddings"])}
    by_key = {(r["question_id"], r["config"]): r for r in rows}
    failures: list[str] = []
    checks = {"raw_rows": len(rows), "unique_questions": len(set(r["question_id"] for r in rows)),
              "rows_per_configuration": {}, "context_candidate_violations": 0,
              "context_rank_violations": 0, "graph_node_scope_violations": 0,
              "graph_edge_reconstruction_mismatches": 0, "ranking_reconstruction_mismatches": 0,
              "metric_reconstruction_mismatches": 0, "gold_denominator_mismatches": 0,
              "graph_structure_stat_mismatches": 0}
    for cfg in CONFIGS:
        checks["rows_per_configuration"][cfg] = sum(r["config"] == cfg for r in rows)
    for qid, q in questions.items():
        contexts = list(map(str, q["context_passage_ids"]))
        allowed, gold = set(contexts), set(map(str, q["gold_supporting_passage_ids"]))
        dense_scores = {pid: float(passage_embeddings[pid] @ query_embeddings[qid]) for pid in contexts}
        dense_rank = rank_by_scores(contexts, allowed, dense_scores)
        seeds = dense_rank[:min(SEED_K, len(dense_rank))]
        expected_by_config = {}
        tfidf_rank, _ = rank_local(q, pmap)
        expected_by_config["vanilla_tfidf"] = (set(contexts), tfidf_rank, None)
        expected_by_config["vanilla_bge"] = (set(contexts), dense_rank, None)
        specs = (
            ("entity_sentence_frequency_weighted", "sentence", "frequency", None),
            ("entity_passage_frequency_weighted", "passage", "frequency", None),
            ("entity_sentence_binary", "sentence", "binary", None),
            ("entity_sentence_top50_sparsified", "sentence", "frequency", SPARSIFY_RETAIN_FRACTION),
        )
        for name, window, weighting, retain in specs:
            graph, e2p, _ = build_entity_graph_local(contexts, extraction, qid, window, weighting, retain)
            candidates = local_graph_candidates(contexts, seeds, graph, e2p)
            expected_by_config[name] = (candidates, rank_by_scores(candidates, allowed, dense_scores), graph)
        chunk = build_chunk_similarity_graph(contexts, passage_embeddings, qid, CHUNK_K)
        chunk_candidates = chunk_graph_candidates(contexts, seeds, chunk)
        expected_by_config["chunk_bge_knn2"] = (chunk_candidates, rank_by_scores(chunk_candidates, allowed, dense_scores), chunk)
        for cfg in CONFIGS:
            row = by_key.get((qid, cfg))
            if row is None:
                failures.append(f"missing raw row {qid}/{cfg}")
                continue
            candidates, ranked, graph = expected_by_config[cfg]
            raw_candidates, raw_ranked = set(row["candidate_passage_ids"]), row["ranked_passage_ids"]
            if not raw_candidates <= allowed:
                checks["context_candidate_violations"] += 1
            if not set(raw_ranked) <= allowed:
                checks["context_rank_violations"] += 1
            if raw_candidates != candidates or raw_ranked != ranked:
                checks["ranking_reconstruction_mismatches"] += 1
                failures.append(f"candidate/rank reconstruction mismatch {qid}/{cfg}")
            if len(gold) != row["gold_support_count"] or sorted(gold) != row["gold_supporting_passage_ids"]:
                checks["gold_denominator_mismatches"] += 1
                failures.append(f"gold denominator mismatch {qid}/{cfg}")
            for k in (2, 5, 10):
                rec = len(gold & set(ranked[:k])) / len(gold)
                complete = int(gold <= set(ranked[:k]))
                if row[f"recall_at_{k}"] != rec or row[f"complete_support_at_{k}"] != complete:
                    checks["metric_reconstruction_mismatches"] += 1
                    failures.append(f"recall@{k} or complete-support mismatch {qid}/{cfg}")
            gen_recall = len(gold & candidates) / len(gold)
            if row["candidate_generation_recall"] != gen_recall:
                checks["metric_reconstruction_mismatches"] += 1
                failures.append(f"candidate-generation recall mismatch {qid}/{cfg}")
            if graph is not None:
                local_entities = {e for pid in allowed for e in extraction[pid]["entities"]}
                if cfg == "chunk_bge_knn2":
                    nodes_ok = set(row["graph_nodes"]) == {f"chunk::{pid}" for pid in allowed}
                else:
                    nodes_ok = set(row["graph_nodes"]) <= local_entities
                if not nodes_ok:
                    checks["graph_node_scope_violations"] += 1
                    failures.append(f"graph node leakage {qid}/{cfg}")
                expected_edges = sorted((min(str(u), str(v)), max(str(u), str(v)), float(d["weight"]))
                                        for u, v, d in graph.edges(data=True))
                actual_edges = sorted((min(e["source"], e["target"]), max(e["source"], e["target"]), float(e["weight"]))
                                      for e in row["graph_edges"])
                if expected_edges != actual_edges:
                    checks["graph_edge_reconstruction_mismatches"] += 1
                    failures.append(f"graph edge reconstruction mismatch {qid}/{cfg}")
                import networkx as nx
                rebuilt = nx.Graph()
                rebuilt.add_nodes_from(row["graph_nodes"])
                rebuilt.add_weighted_edges_from((e["source"], e["target"], float(e["weight"])) for e in row["graph_edges"])
                expected_stats = (rebuilt.number_of_nodes(), rebuilt.number_of_edges(),
                                  nx.density(rebuilt) if rebuilt.number_of_nodes() > 1 else 0.0,
                                  nx.number_connected_components(rebuilt) if rebuilt.number_of_nodes() else 0,
                                  len(list(nx.isolates(rebuilt))))
                actual_stats = (row["graph_node_count"], row["graph_edge_count"], row["graph_density"],
                                row["graph_component_count"], row["graph_isolate_count"])
                if any(not math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
                       for a, b in zip(actual_stats, expected_stats)):
                    checks["graph_structure_stat_mismatches"] += 1
                    failures.append(f"graph structure statistic mismatch {qid}/{cfg}")
    seed = int(config["sample_seed"])
    retrieval_expected, graph_expected = summarize_rows(rows, seed)
    checks["retrieval_summary_csv_matches_raw"] = compare_saved_csv(run_dir / "retrieval_summary.csv", retrieval_expected)
    checks["graph_summary_csv_matches_raw"] = compare_saved_csv(run_dir / "graph_summary.csv", graph_expected)
    checks["paired_comparisons_csv_matches_raw"] = compare_saved_csv(run_dir / "paired_comparisons.csv", paired_comparisons(rows, seed))
    stability_expected, stability_summary_expected = stability_rows(rows, seed)
    checks["stability_pairwise_csv_matches_raw"] = compare_saved_csv(run_dir / "retrieval_stability_pairwise.csv", stability_expected)
    checks["stability_summary_csv_matches_raw"] = compare_saved_csv(run_dir / "retrieval_stability_summary.csv", stability_summary_expected)
    for field in ("retrieval_summary_csv_matches_raw", "graph_summary_csv_matches_raw",
                  "paired_comparisons_csv_matches_raw", "stability_pairwise_csv_matches_raw",
                  "stability_summary_csv_matches_raw"):
        if not checks[field]:
            failures.append(f"{field} failed")
    checks["selected_question_count"] = len(qids)
    checks["expected_rows"] = len(qids) * len(CONFIGS)
    checks["model_revision"] = revision
    checks["failures"] = failures
    checks["passed"] = not failures and len(rows) == len(qids) * len(CONFIGS)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/pilot_500_v2")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.run_dir, args.data_dir)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
