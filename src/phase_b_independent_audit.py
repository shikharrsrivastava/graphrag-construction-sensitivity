"""Independent recomputation of Phase B rows and statistical CSVs.

This audit intentionally does not import the Phase B graph builders, metric
aggregators, or statistical test helpers. It reconstructs expected quantities
from the selected questions, extraction cache, embedding cache, and raw rows.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.common import ROOT, read_json

CONFIGS = (
    "vanilla_tfidf", "vanilla_bge", "entity_sentence_frequency_weighted",
    "entity_passage_frequency_weighted", "entity_sentence_binary",
    "entity_sentence_top50_sparsified", "chunk_bge_knn2",
)
GRAPH_CONFIGS = CONFIGS[2:]
KS = (2, 5, 10)
METRICS = tuple([f"recall_at_{k}" for k in KS] + [f"complete_support_at_{k}" for k in KS])
BOOT_N = 10_000
SIGN_N = 20_000


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ci_mean(values: list[float], seed: int) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(arr, size=(BOOT_N, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def paired_sign_flip(left: list[float], right: list[float], seed: int) -> tuple[float, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(SIGN_N, len(delta)))
    observed = abs(float(delta.mean()))
    p = (1 + np.count_nonzero(np.abs((signs * delta).mean(axis=1)) >= observed)) / (SIGN_N + 1)
    return float(delta.mean()), float(p)


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def same_csv(path: Path, expected: list[dict]) -> bool:
    with path.open(encoding="utf-8", newline="") as stream:
        actual = list(csv.DictReader(stream))
    if len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected):
        if set(actual_row) != set(expected_row):
            return False
        for name, value in expected_row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isclose(float(actual_row[name]), float(value), rel_tol=1e-12, abs_tol=1e-12):
                    return False
            elif str(actual_row[name]) != str(value):
                return False
    return True


def _pair_edges(entities: set[str]) -> list[tuple[str, str]]:
    ordered = sorted(entities)
    return [(ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i + 1, len(ordered))]


def local_entity_graph(context_ids: list[str], extraction: dict[str, dict], method: str) -> tuple[set[str], dict[tuple[str, str], float], dict[str, set[str]]]:
    """Recreate entity nodes/edges using only the current question's context IDs."""
    allowed = set(context_ids)
    by_passage: dict[str, set[str]] = {}
    nodes: set[str] = set()
    for pid in sorted(allowed):
        sentence_lists = extraction[pid]["sentences"]
        entities = set(e for sentence in sentence_lists for e in sentence)
        by_passage[pid] = entities
        nodes.update(entities)
    frequencies: Counter[tuple[str, str]] = Counter()
    if "passage" in method:
        for pid in sorted(allowed):
            frequencies.update(_pair_edges(by_passage[pid]))
    else:
        for pid in sorted(allowed):
            for sentence in extraction[pid]["sentences"]:
                frequencies.update(_pair_edges(set(sentence)))
    weights = {edge: (1.0 if method.endswith("binary") else float(count)) for edge, count in frequencies.items()}
    if "top50_sparsified" in method and weights:
        ranked = sorted(weights, key=lambda edge: (-weights[edge], edge[0], edge[1]))
        retain = max(1, math.ceil(0.5 * len(ranked)))
        weights = {edge: weights[edge] for edge in ranked[:retain]}
    entity_passages: dict[str, set[str]] = defaultdict(set)
    for pid, entities in by_passage.items():
        for entity in entities:
            entity_passages[entity].add(pid)
    # This is an explicit isolation condition, not a post-ranking filter.
    mapped_ids = set().union(*entity_passages.values()) if entity_passages else set()
    assert mapped_ids <= allowed
    return nodes, weights, entity_passages


def _adjacency(edges: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for (left, right), weight in edges.items():
        adjacency[left][right] = weight
        adjacency[right][left] = weight
    return adjacency


def _entity_candidates(context_ids: list[str], seeds: list[str], nodes: set[str],
                       edges: dict[tuple[str, str], float], entity_passages: dict[str, set[str]]) -> set[str]:
    allowed = set(context_ids)
    seed_entities = {entity for entity, pids in entity_passages.items() if pids.intersection(seeds)}
    expanded = set(seed_entities)
    adj = _adjacency(edges)
    for entity in seed_entities:
        neighbors = sorted(adj.get(entity, {}).items(), key=lambda item: (-item[1], item[0]))
        expanded.update(node for node, _ in neighbors[:3])
    candidates = set(seeds)
    for entity in expanded:
        candidates.update(entity_passages.get(entity, set()))
    if not candidates <= allowed:
        raise AssertionError("Independent entity candidate builder escaped context set")
    if not nodes <= set(entity_passages):
        raise AssertionError("Independent local graph contains unmapped entity nodes")
    return candidates


def _component_count(nodes: set[str], edges: list[tuple[str, str]]) -> tuple[int, int]:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(nodes)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node] & unseen:
                unseen.remove(neighbor)
                stack.append(neighbor)
    isolates = sum(not neighbors for neighbors in adjacency.values())
    return components, isolates


def _graph_stats(nodes: set[str], edges: list[tuple[str, str]]) -> tuple[int, int, float, int, int]:
    components, isolates = _component_count(nodes, edges) if nodes else (0, 0)
    possible = len(nodes) * (len(nodes) - 1) / 2
    density = len(edges) / possible if possible else 0.0
    return len(nodes), len(edges), density, components, isolates


def _retrieval_summaries(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    retrieval_fields = ["candidate_generation_recall", *METRICS]
    retrieval: list[dict] = []
    for config in CONFIGS:
        for group_name in ("all", "bridge", "comparison"):
            group = [r for r in rows if r["config"] == config and (group_name == "all" or r["question_type"] == group_name)]
            if not group:
                continue
            for metric in retrieval_fields:
                point, low, high = ci_mean([float(r[metric]) for r in group], seed + sum(map(ord, f"{config}:{group_name}:{metric}")))
                mean = lambda field: float(np.mean([float(r.get(field, 0.0)) for r in group]))
                retrieval.append({"config": config, "question_group": group_name, "question_count": len(group), "metric": metric,
                                  "mean": point, "ci95_low": low, "ci95_high": high,
                                  "mean_candidate_count": mean("candidate_count"),
                                  "mean_local_tfidf_fit_and_retrieval_seconds": mean("local_tfidf_fit_and_retrieval_seconds"),
                                  "mean_dense_query_scoring_seconds": mean("dense_query_scoring_seconds"),
                                  "mean_candidate_generation_seconds": mean("candidate_generation_seconds"),
                                  "mean_candidate_reranking_seconds": mean("candidate_reranking_seconds"),
                                  "mean_query_retrieval_seconds": mean("query_retrieval_seconds"),
                                  "mean_graph_construction_seconds": mean("graph_construction_seconds")})
    graph_fields = ("graph_node_count", "graph_edge_count", "graph_density", "graph_component_count", "graph_isolate_count")
    graph_summary: list[dict] = []
    for config in CONFIGS:
        for group_name in ("all", "bridge", "comparison"):
            group = [r for r in rows if r["config"] == config and r["graph_applicable"] and
                     (group_name == "all" or r["question_type"] == group_name)]
            if not group:
                continue
            for metric in graph_fields:
                point, low, high = ci_mean([float(r[metric]) for r in group], seed + sum(map(ord, f"graph:{config}:{group_name}:{metric}")))
                graph_summary.append({"config": config, "question_group": group_name, "question_count": len(group),
                                      "metric": metric, "mean": point, "ci95_low": low, "ci95_high": high,
                                      "mean_graph_construction_seconds": float(np.mean([r["graph_construction_seconds"] for r in group]))})
    return retrieval, graph_summary


def _paired_table(rows: list[dict], seed: int) -> list[dict]:
    by_key = {(r["question_id"], r["config"]): r for r in rows}
    contrasts = [("vanilla_bge", "vanilla_tfidf", METRICS)]
    contrasts += [("vanilla_bge", cfg, ("candidate_generation_recall", *METRICS)) for cfg in GRAPH_CONFIGS]
    contrasts += [("entity_sentence_frequency_weighted", cfg, ("candidate_generation_recall", *METRICS))
                  for cfg in GRAPH_CONFIGS if cfg != "entity_sentence_frequency_weighted"]
    output = []
    for ref, other, metrics in contrasts:
        ids = sorted(qid for qid, cfg in by_key if cfg == ref and (qid, other) in by_key)
        for metric in metrics:
            delta, p = paired_sign_flip([by_key[(qid, other)][metric] for qid in ids],
                                        [by_key[(qid, ref)][metric] for qid in ids],
                                        seed + sum(map(ord, f"{ref}:{other}:{metric}")))
            output.append({"n_questions": len(ids), "comparison": f"{other} - {ref}", "metric": metric,
                           "mean_paired_difference": delta, "p_value": p})
    adjustments = holm([row["p_value"] for row in output])
    for row, adjusted in zip(output, adjustments):
        row["holm_adjusted_p"] = adjusted
    return output


def _stability_tables(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    by_question: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(row)
    raw = []
    for qid, group in by_question.items():
        ordered = sorted(group, key=lambda r: r["config"])
        for left, right in itertools.combinations(ordered, 2):
            for k in KS:
                a, b = set(left["ranked_passage_ids"][:k]), set(right["ranked_passage_ids"][:k])
                overlap = len(a & b) / len(a | b) if (a or b) else 1.0
                raw.append({"question_id": qid, "config_a": left["config"], "config_b": right["config"], "k": k,
                            "top_k_jaccard": overlap,
                            "interpretation": "cross_configuration_retrieval_overlap_not_seed_stability"})
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in raw:
        groups[(row["config_a"], row["config_b"], row["k"])].append(row)
    summary = []
    for (a, b, k), group in sorted(groups.items()):
        point, low, high = ci_mean([r["top_k_jaccard"] for r in group], seed + k + sum(map(ord, a + b)))
        summary.append({"config_a": a, "config_b": b, "k": k, "question_count": len(group),
                        "mean_jaccard": point, "ci95_low": low, "ci95_high": high,
                        "interpretation": "cross_configuration_retrieval_overlap_not_seed_stability"})
    return raw, summary


def audit(run_dir: Path, data_dir: Path) -> dict:
    config = json.loads((run_dir / "run_config.json").read_text())
    rows = read_jsonl(run_dir / "question_results.jsonl")
    selection = json.loads((run_dir / "selected_questions.json").read_text())
    all_q = read_json(data_dir / "questions.json")
    question_map = {str(q["question_id"]): q for q in all_q}
    questions = {qid: question_map[qid] for qid in selection["question_ids"]}
    extraction = {r["passage_id"]: r for r in read_jsonl(run_dir.parent / "cache/entity_extractions.jsonl")}
    revision = config["dense_model"]["resolved_revision"]
    embed_dir = run_dir.parent / "cache/embeddings" / revision
    manifest_p = json.loads((embed_dir / "passages.json").read_text())
    manifest_q = json.loads((embed_dir / "queries.json").read_text())
    with np.load(embed_dir / "passages.npz", allow_pickle=False) as bundle:
        pids = bundle["passage_ids"].astype(str)
        pvecs = bundle["embeddings"]
        passage_embeddings = {pid: vec for pid, vec in zip(pids, pvecs)}
    with np.load(embed_dir / "queries.npz", allow_pickle=False) as bundle:
        qids_cache = bundle["question_ids"].astype(str)
        qvecs = bundle["embeddings"]
        query_embeddings = {qid: vec for qid, vec in zip(qids_cache, qvecs)}
    by_key = {(r["question_id"], r["config"]): r for r in rows}
    counts = Counter()
    failures = []
    if len(rows) != len(questions) * len(CONFIGS):
        failures.append("raw row total does not equal questions times seven methods")
    for name in CONFIGS:
        counts[f"rows_{name}"] = sum(r["config"] == name for r in rows)
        if counts[f"rows_{name}"] != len(questions):
            failures.append(f"wrong row count for {name}")
    dim = config["dense_model"]["embedding_dimension"]
    if manifest_p.get("revision") != revision or manifest_q.get("revision") != revision:
        failures.append("embedding cache manifest revision mismatch")
    if manifest_p.get("model_id") != config["dense_model"]["model_id"] or manifest_q.get("model_id") != config["dense_model"]["model_id"]:
        failures.append("embedding cache model id mismatch")
    if manifest_p.get("normalized") is not True or manifest_q.get("normalized") is not True:
        failures.append("embedding cache manifest does not assert normalized vectors")
    if pvecs.shape[1] != dim or qvecs.shape[1] != dim:
        failures.append("embedding dimension mismatch")
    vector_norm_errors = max(float(np.max(np.abs(np.linalg.norm(pvecs, axis=1) - 1))),
                             float(np.max(np.abs(np.linalg.norm(qvecs, axis=1) - 1))))
    if vector_norm_errors > 1e-5:
        failures.append("embedding vectors are not L2-normalized")
    if config["dense_model"].get("model_id") != "BAAI/bge-small-en-v1.5":
        failures.append("unexpected dense model")
    checks = {"rows": len(rows), "questions": len(questions), "methods": len(CONFIGS),
              "rows_per_method": {name: counts[f"rows_{name}"] for name in CONFIGS},
              "context_candidate_violations": 0, "context_rank_violations": 0,
              "context_graph_node_violations": 0, "graph_edge_reconstruction_mismatches": 0,
              "candidate_reconstruction_mismatches": 0, "ranking_reconstruction_mismatches": 0,
              "recall_reconstruction_mismatches": 0, "gold_denominator_mismatches": 0,
              "graph_statistic_mismatches": 0, "max_embedding_norm_error": vector_norm_errors,
              "model_id": config["dense_model"]["model_id"], "model_revision": revision,
              "embedding_dimension": dim, "similarity": "dot product over L2-normalized vectors (cosine)"}
    for qid, q in questions.items():
        context_ids = list(map(str, q["context_passage_ids"]))
        allowed, gold = set(context_ids), set(map(str, q["gold_supporting_passage_ids"]))
        qvec = query_embeddings[qid]
        scores = {pid: float(passage_embeddings[pid] @ qvec) for pid in context_ids}
        dense_rank = sorted(context_ids, key=lambda pid: (-scores[pid], pid))
        seeds = dense_rank[:5]
        local_graph_specs = {
            "entity_sentence_frequency_weighted": "sentence_frequency",
            "entity_passage_frequency_weighted": "passage_frequency",
            "entity_sentence_binary": "sentence_binary",
            "entity_sentence_top50_sparsified": "sentence_top50_sparsified",
        }
        expected_graphs: dict[str, tuple[set[str], dict[tuple[str, str], float], set[str]]] = {}
        for method, graph_kind in local_graph_specs.items():
            nodes, weights, entity_passages = local_entity_graph(context_ids, extraction, graph_kind)
            candidates = _entity_candidates(context_ids, seeds, nodes, weights, entity_passages)
            expected_graphs[method] = (nodes, weights, candidates)
        # Independent chunk cosine union-kNN graph over this question's ten passage vectors.
        ordered_ids = sorted(context_ids)
        vectors = np.vstack([passage_embeddings[pid] for pid in ordered_ids]).astype(np.float32, copy=False)
        cos = vectors @ vectors.T
        chunk_nodes = {f"chunk::{pid}" for pid in ordered_ids}
        chunk_edges: set[tuple[str, str]] = set()
        chunk_weights: dict[tuple[str, str], float] = {}
        chunk_adjacency: dict[str, set[str]] = {node: set() for node in chunk_nodes}
        for i, pid in enumerate(ordered_ids):
            neighbors = sorted((j for j in range(len(ordered_ids)) if j != i),
                               key=lambda j: (-float(cos[i, j]), ordered_ids[j]))[:2]
            for j in neighbors:
                a, b = f"chunk::{pid}", f"chunk::{ordered_ids[j]}"
                edge = tuple(sorted((a, b)))
                chunk_edges.add(edge)
                chunk_weights[edge] = float(cos[i, j])
                chunk_adjacency[a].add(b)
                chunk_adjacency[b].add(a)
        chunk_seed_nodes = {f"chunk::{pid}" for pid in seeds}
        chunk_expanded = set(chunk_seed_nodes)
        for node in chunk_seed_nodes:
            chunk_expanded.update(chunk_adjacency[node])
        chunk_candidates = set(seeds) | {node.removeprefix("chunk::") for node in chunk_expanded}

        for method in CONFIGS:
            row = by_key.get((qid, method))
            if row is None:
                failures.append(f"missing row {qid}/{method}")
                continue
            candidates = allowed if method in ("vanilla_tfidf", "vanilla_bge") else (
                expected_graphs[method][2] if method in expected_graphs else chunk_candidates)
            if not set(row["candidate_passage_ids"]) <= allowed:
                checks["context_candidate_violations"] += 1
            if not set(row["ranked_passage_ids"]) <= allowed:
                checks["context_rank_violations"] += 1
            if set(row["context_passage_ids"]) != allowed or set(row["gold_supporting_passage_ids"]) != gold:
                checks["gold_denominator_mismatches"] += 1
            if set(row["candidate_passage_ids"]) != candidates:
                checks["candidate_reconstruction_mismatches"] += 1
            if method == "vanilla_tfidf":
                expected_rank = row["ranked_passage_ids"]  # ranking is scored; this audit recomputes its metrics from the saved ranking.
            else:
                expected_rank = sorted(candidates, key=lambda pid: (-scores[pid], pid))
                if row["ranked_passage_ids"] != expected_rank:
                    checks["ranking_reconstruction_mismatches"] += 1
            if method in expected_graphs:
                local_nodes, edge_weights, _ = expected_graphs[method]
                saved_nodes = set(row["graph_nodes"])
                if saved_nodes != local_nodes:
                    checks["context_graph_node_violations"] += 1
                expected_edges = sorted((a, b, float(w)) for (a, b), w in edge_weights.items())
                actual_edges = sorted((min(e["source"], e["target"]), max(e["source"], e["target"]), float(e["weight"]))
                                      for e in row["graph_edges"])
                if expected_edges != actual_edges:
                    checks["graph_edge_reconstruction_mismatches"] += 1
            elif method == "chunk_bge_knn2":
                if set(row["graph_nodes"]) != chunk_nodes:
                    checks["context_graph_node_violations"] += 1
                actual_edges = {(min(e["source"], e["target"]), max(e["source"], e["target"])) for e in row["graph_edges"]}
                if actual_edges != chunk_edges:
                    checks["graph_edge_reconstruction_mismatches"] += 1
                saved_weights = {(min(e["source"], e["target"]), max(e["source"], e["target"])): float(e["weight"])
                                 for e in row["graph_edges"]}
                if set(saved_weights) != set(chunk_weights) or any(
                    not math.isclose(saved_weights[edge], weight, rel_tol=1e-7, abs_tol=1e-7)
                    for edge, weight in chunk_weights.items()
                ):
                    checks["graph_edge_reconstruction_mismatches"] += 1
            if method in GRAPH_CONFIGS and row.get("graph_construction_scope") != "question_contexts_only":
                checks["context_graph_node_violations"] += 1
            if sorted(gold) != row["gold_supporting_passage_ids"] or len(gold) != row["gold_support_count"]:
                checks["gold_denominator_mismatches"] += 1
            ranking = row["ranked_passage_ids"]
            for k in KS:
                recall = len(gold & set(ranking[:k])) / len(gold)
                complete = int(gold <= set(ranking[:k]))
                if row[f"recall_at_{k}"] != recall or row[f"complete_support_at_{k}"] != complete:
                    checks["recall_reconstruction_mismatches"] += 1
            candidate_recall = len(gold & candidates) / len(gold)
            if row["candidate_generation_recall"] != candidate_recall:
                checks["recall_reconstruction_mismatches"] += 1

    # Rebuild graph summary statistics directly from the serialized local graphs.
    for row in rows:
        if not row["graph_applicable"]:
            continue
        nodes = set(row["graph_nodes"])
        edges = [(e["source"], e["target"]) for e in row["graph_edges"]]
        expected = _graph_stats(nodes, edges)
        actual = (row["graph_node_count"], row["graph_edge_count"], row["graph_density"],
                  row["graph_component_count"], row["graph_isolate_count"])
        if any(not math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(actual, expected)):
            checks["graph_statistic_mismatches"] += 1

    sample_seed = int(config["sample_seed"])
    retrieval, graph = _retrieval_summaries(rows, sample_seed)
    paired = _paired_table(rows, sample_seed)
    stability, stability_summary = _stability_tables(rows, sample_seed)
    checks["retrieval_summary_matches_independent_raw_recompute"] = same_csv(run_dir / "retrieval_summary.csv", retrieval)
    checks["graph_summary_matches_independent_raw_recompute"] = same_csv(run_dir / "graph_summary.csv", graph)
    checks["paired_comparisons_match_independent_raw_recompute"] = same_csv(run_dir / "paired_comparisons.csv", paired)
    checks["stability_pairwise_matches_independent_raw_recompute"] = same_csv(run_dir / "retrieval_stability_pairwise.csv", stability)
    checks["stability_summary_matches_independent_raw_recompute"] = same_csv(run_dir / "retrieval_stability_summary.csv", stability_summary)
    for key, passed in checks.items():
        if key.endswith("independent_raw_recompute") or key.endswith("recompute"):
            if not passed:
                failures.append(key)
    check_fields = ("context_candidate_violations", "context_rank_violations", "context_graph_node_violations",
                    "graph_edge_reconstruction_mismatches", "candidate_reconstruction_mismatches",
                    "ranking_reconstruction_mismatches", "recall_reconstruction_mismatches",
                    "gold_denominator_mismatches", "graph_statistic_mismatches")
    for key in check_fields:
        if checks[key] != 0:
            failures.append(f"{key}={checks[key]}")
    checks["checks"] = dict(counts)
    checks["selected_question_count"] = len(questions)
    checks["expected_raw_rows"] = len(questions) * len(CONFIGS)
    checks["failures"] = failures
    checks["passed"] = not failures
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/pilot_500_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir, args.data_dir)
    encoded = json.dumps(result, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
