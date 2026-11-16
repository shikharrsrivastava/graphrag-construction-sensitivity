"""Phase C: question-local graph-aware ranking (see PREREGISTRATION_graph_ranking.md).

Every configuration ranks exactly the current question's contexts. Graphs, BM25
and TF-IDF indexes are built from those contexts only (Track A). BGE vectors and
spaCy extractions are cached per passage text, never as graphs.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import time
import traceback
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi

from src.build_graph import normalize_entity
from src.common import ROOT, read_json
from src.evaluate_dense_graph import (
    MODEL_ID, MODEL_REVISION, QUERY_PREFIX, bootstrap_mean_ci, build_chunk_similarity_graph, dump_csv,
    encode_cached, encode_queries_cached, ensure_extraction_cache, holm_adjust, max_rss_mb,
    resolve_model_revision, sign_flip_test,
)
from src.evaluate_question_local import passage_text, rank_local, validate_inputs

K_VALUES = (2, 5, 10)
SEED_K = 5
PPR_ALPHA = 0.85
CHUNK_K = 2
HEADLINE_LAMBDA = 0.5
LAMBDAS = (0.25, 0.5, 0.75)
STAT_SEED = 2027
CONFIRMATORY_METRICS = ("complete_support_at_5", "recall_at_5")
CONFIRMATORY_CONTRASTS = (("ppr_entity", "vanilla_bge"), ("ppr_chunk", "vanilla_bge"), ("vanilla_bge", "bm25"))
METRICS = [f"recall_at_{k}" for k in K_VALUES] + [f"complete_support_at_{k}" for k in K_VALUES]


def config_name(graph: str, lam: float) -> str:
    return f"ppr_{graph}" if lam == HEADLINE_LAMBDA else f"ppr_{graph}_lambda{lam}"


CONFIGS = ["bm25", "vanilla_tfidf", "vanilla_bge"] + [config_name(g, lam) for g in ("entity", "chunk") for lam in LAMBDAS]
PREREGISTERED_CONFIGS = {"bm25", "vanilla_bge", "ppr_entity", "ppr_chunk"}


# ---------------------------------------------------------------- scoring

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower())


def bm25_scores(question: str, context_ids: list[str], passage_by_id: dict[str, dict[str, Any]]) -> dict[str, float]:
    corpus = [tokenize(passage_text(passage_by_id[pid])) for pid in context_ids]
    scores = BM25Okapi(corpus, k1=1.5, b=0.75).get_scores(tokenize(question))
    return {pid: float(s) for pid, s in zip(context_ids, scores)}


def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo <= 1e-15:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def rank_contexts(context_ids: Iterable[str], primary: dict[str, float], tiebreak: dict[str, float] | None = None) -> list[str]:
    ids = list(map(str, context_ids))
    if set(primary) != set(ids):
        raise AssertionError("Score vector does not cover exactly the question contexts")
    tiebreak = tiebreak or {pid: 0.0 for pid in ids}
    return sorted(ids, key=lambda pid: (-primary[pid], -tiebreak[pid], pid))


def normalize_title(title: str) -> str:
    """Strip one trailing parenthetical disambiguator, then apply normalize_entity (Amendment 1)."""
    return normalize_entity(re.sub(r"\s*\([^()]*\)\s*$", "", str(title)))


def build_passage_entity_graph(context_ids: Iterable[str], extraction_by_pid: dict[str, dict[str, Any]],
                               passage_by_id: dict[str, dict[str, Any]], question_id: str) -> nx.Graph:
    """Bipartite passage-entity graph built from this question's contexts only."""
    ids = sorted(set(map(str, context_ids)))
    graph = nx.Graph(question_id=str(question_id), construction_scope="question_contexts_only",
                     node_type="passage+entity", context_passage_ids=json.dumps(ids))
    for pid in ids:
        graph.add_node(f"p::{pid}", kind="passage", passage_id=pid)
        entities = set(map(str, extraction_by_pid[pid]["entities"]))
        title = normalize_title(passage_by_id[pid].get("title", ""))
        if len(title) >= 2:
            entities.add(title)
        for ent in sorted(entities):
            graph.add_node(f"e::{ent}", kind="entity")
            graph.add_edge(f"p::{pid}", f"e::{ent}", weight=1.0)
    return graph


def personalized_pagerank(graph: nx.Graph, seed_nodes: list[str]) -> dict[str, float]:
    if not seed_nodes:
        raise ValueError("PPR needs at least one seed")
    if any(float(d.get("weight", 1.0)) < 0 for _, _, d in graph.edges(data=True)):
        raise ValueError("PPR requires non-negative edge weights")
    personalization = {node: 1.0 / len(seed_nodes) for node in seed_nodes}
    # Start the power iteration at the restart vector so nodes unreachable from a seed stay exactly 0.
    nstart = {node: personalization.get(node, 0.0) for node in graph}
    return nx.pagerank(graph, alpha=PPR_ALPHA, personalization=personalization, nstart=nstart,
                       weight="weight", max_iter=1000, tol=1e-10)


def graph_passage_scores(graph: nx.Graph, node_of_pid: dict[str, str], seeds: list[str]) -> dict[str, float]:
    pr = personalized_pagerank(graph, [node_of_pid[pid] for pid in seeds])
    return {pid: float(pr[node]) for pid, node in node_of_pid.items()}


def combine(bge: dict[str, float], graph: dict[str, float], lam: float) -> dict[str, float]:
    nb, ng = minmax(bge), minmax(graph)
    return {pid: (1.0 - lam) * nb[pid] + lam * ng[pid] for pid in bge}


# ---------------------------------------------------------------- metrics

def gold_ranks(ranked: list[str], gold: set[str]) -> dict[str, int]:
    return {pid: ranked.index(pid) + 1 for pid in sorted(gold)}


def row_for(q: dict[str, Any], config: str, ranked: list[str], extra: dict[str, Any]) -> dict[str, Any]:
    contexts = list(map(str, q["context_passage_ids"]))
    if sorted(ranked) != sorted(contexts) or len(ranked) != len(set(ranked)):
        raise AssertionError(f"{q['question_id']}/{config}: ranking is not a permutation of the question contexts")
    gold = set(map(str, q["gold_supporting_passage_ids"]))
    row = {"question_id": str(q["question_id"]), "question_type": str(q.get("type", "")),
           "hop_count": q.get("hop_count"), "context_count": len(contexts), "config": config,
           "gold_support_count": len(gold), "gold_supporting_passage_ids": sorted(gold),
           "ranked_passage_ids": ranked, "gold_ranks": gold_ranks(ranked, gold), **extra}
    for k in K_VALUES:
        top = set(ranked[:k])
        row[f"recall_at_{k}"] = len(gold & top) / len(gold)
        row[f"complete_support_at_{k}"] = int(gold <= top)
    return row


def evaluate_question(q: dict[str, Any], passage_by_id: dict[str, dict[str, Any]], extraction_by_pid: dict[str, dict[str, Any]],
                      passage_emb: dict[str, np.ndarray], query_emb: np.ndarray) -> list[dict[str, Any]]:
    qid = str(q["question_id"])
    contexts = list(map(str, q["context_passage_ids"]))
    rows = []

    t0 = time.perf_counter()
    bm = bm25_scores(str(q["question"]), contexts, passage_by_id)
    rows.append(row_for(q, "bm25", rank_contexts(contexts, bm), {"seconds": time.perf_counter() - t0}))

    ranked_tfidf, tfidf_s = rank_local(q, passage_by_id)
    rows.append(row_for(q, "vanilla_tfidf", ranked_tfidf, {"seconds": tfidf_s}))

    t0 = time.perf_counter()
    matrix = np.vstack([passage_emb[pid] for pid in contexts]).astype(np.float32, copy=False)
    bge = {pid: float(s) for pid, s in zip(contexts, matrix @ query_emb)}
    bge_rank = rank_contexts(contexts, bge)
    rows.append(row_for(q, "vanilla_bge", bge_rank, {"seconds": time.perf_counter() - t0}))
    seeds = bge_rank[:SEED_K]

    graphs = {}
    t0 = time.perf_counter()
    eg = build_passage_entity_graph(contexts, extraction_by_pid, passage_by_id, qid)
    graphs["entity"] = (eg, {pid: f"p::{pid}" for pid in contexts}, time.perf_counter() - t0)
    t0 = time.perf_counter()
    cg = build_chunk_similarity_graph(contexts, passage_emb, qid, CHUNK_K)
    graphs["chunk"] = (cg, {pid: f"chunk::{pid}" for pid in contexts}, time.perf_counter() - t0)

    for gname, (graph, node_of_pid, build_s) in graphs.items():
        if graph.graph.get("construction_scope") != "question_contexts_only":
            raise AssertionError("Graph lacks question-local scope marker")
        passage_nodes = {n for n, d in graph.nodes(data=True) if "passage_id" in d}
        if {graph.nodes[n]["passage_id"] for n in passage_nodes} != set(contexts):
            raise AssertionError(f"{qid}/{gname}: graph passage nodes differ from the question contexts")
        t0 = time.perf_counter()
        gscore = graph_passage_scores(graph, node_of_pid, seeds)
        ppr_s = time.perf_counter() - t0
        passage_degree = {pid: graph.degree(node_of_pid[pid]) for pid in contexts}
        stats = {"graph_node_count": graph.number_of_nodes(), "graph_edge_count": graph.number_of_edges(),
                 "graph_component_count": nx.number_connected_components(graph),
                 "isolated_passage_count": sum(1 for pid in contexts if passage_degree[pid] == 0),
                 "seed_passage_ids": seeds}
        if gname == "entity":
            # passages reachable from a seed through one shared entity (two hops in the bipartite graph)
            linked = set()
            for pid in seeds:
                for ent in graph.neighbors(node_of_pid[pid]):
                    linked.update(graph.nodes[n]["passage_id"] for n in graph.neighbors(ent) if graph.nodes[n].get("kind") == "passage")
            stats["passages_sharing_entity_with_seed"] = len(linked - set(seeds))
        for lam in LAMBDAS:
            final = combine(bge, gscore, lam)
            ranked = rank_contexts(contexts, final, bge)
            rows.append(row_for(q, config_name(gname, lam), ranked,
                                {"seconds": build_s + ppr_s, "lambda": lam, "graph": gname, **stats}))
    return rows


# ---------------------------------------------------------------- statistics

def summarize(rows: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    out = []
    groups = ["all"] + sorted({str(r[group_field]) for r in rows})
    for config in CONFIGS:
        for g in groups:
            sub = [r for r in rows if r["config"] == config and (g == "all" or str(r[group_field]) == g)]
            if not sub:
                continue
            for metric in METRICS:
                mean, lo, hi = bootstrap_mean_ci([r[metric] for r in sub], STAT_SEED + sum(map(ord, f"{config}:{g}:{metric}")))
                out.append({"config": config, "question_group": g, "n_questions": len(sub), "metric": metric,
                            "mean": mean, "ci95_low": lo, "ci95_high": hi,
                            "status": "preregistered_method" if config in PREREGISTERED_CONFIGS else "EXPLORATORY"})
    return out


def paired_diff(rows: list[dict[str, Any]], a: str, b: str, metric: str) -> dict[str, Any]:
    by = {(r["question_id"], r["config"]): r[metric] for r in rows}
    qids = sorted({r["question_id"] for r in rows if r["config"] == a} & {r["question_id"] for r in rows if r["config"] == b})
    xa = np.array([by[(q, a)] for q in qids], dtype=float)
    xb = np.array([by[(q, b)] for q in qids], dtype=float)
    seed = STAT_SEED + sum(map(ord, f"{a}:{b}:{metric}"))
    delta, p = sign_flip_test(xa, xb, seed)
    _, lo, hi = bootstrap_mean_ci(xa - xb, seed + 1)
    return {"comparison": f"{a} - {b}", "metric": metric, "n_questions": len(qids),
            "mean_a": float(xa.mean()), "mean_b": float(xb.mean()), "mean_paired_difference": delta,
            "diff_ci95_low": lo, "diff_ci95_high": hi, "questions_a_better": int((xa > xb).sum()),
            "questions_b_better": int((xa < xb).sum()), "p_value": p}


def headline_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tests = [paired_diff(rows, a, b, m) for a, b in CONFIRMATORY_CONTRASTS for m in CONFIRMATORY_METRICS]
    for t, adj in zip(tests, holm_adjust([t["p_value"] for t in tests])):
        t["holm_adjusted_p"] = adj
        t["holm_family"] = "6 tests within this dataset"
        t["significant_holm_0.05"] = bool(adj < 0.05)
        t["status"] = "PREREGISTERED_CONFIRMATORY"
    return tests


# ---------------------------------------------------------------- runner

def package_versions() -> dict[str, str]:
    out = {}
    for name in ("numpy", "networkx", "rank-bm25", "scikit-learn", "sentence-transformers", "torch", "transformers", "spacy", "en-core-web-sm"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    out["python"] = platform.python_version()
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # interrupted final line; that question is recomputed
    return rows


def run(data_dir: Path, out_dir: Path, cache_dir: Path, limit: int | None, batch_size: int) -> dict[str, Any]:
    started = time.perf_counter()
    questions_all = read_json(data_dir / "questions.json")
    passages_all = read_json(data_dir / "passages.json")
    validate_inputs(questions_all, passages_all, read_json(data_dir / "question_contexts.json"))
    counts = Counter(len(q["context_passage_ids"]) for q in questions_all)
    expected = counts.most_common(1)[0][0]
    questions = [q for q in questions_all if len(q["context_passage_ids"]) == expected]
    excluded = [str(q["question_id"]) for q in questions_all if len(q["context_passage_ids"]) != expected]
    if limit:
        questions = questions[:limit]
    passage_by_id = {str(p["passage_id"]): p for p in passages_all}
    needed = sorted({str(pid) for q in questions for pid in q["context_passage_ids"]})

    final_path = out_dir / "question_results.jsonl"
    if final_path.exists() and final_path.stat().st_size:
        raise FileExistsError(f"Refusing to overwrite existing results in {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "question_results.partial.jsonl"
    done_rows = load_jsonl(partial_path)
    done_by_q: dict[str, list[dict[str, Any]]] = {}
    for r in done_rows:
        done_by_q.setdefault(r["question_id"], []).append(r)
    done = {qid for qid, rs in done_by_q.items() if {r["config"] for r in rs} == set(CONFIGS)}

    extraction_by_pid, extraction_info = ensure_extraction_cache(passage_by_id, needed, cache_dir / "entity_extractions.jsonl")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu")
    revision = resolve_model_revision(model, MODEL_REVISION)
    if revision != MODEL_REVISION:
        raise RuntimeError(f"Resolved BGE revision {revision} != pinned {MODEL_REVISION}")
    emb_dir = cache_dir / "embeddings" / revision
    passage_emb, passage_info = encode_cached(model, {pid: passage_text(passage_by_id[pid]) for pid in needed},
                                              emb_dir / "passages.npz", revision, batch_size)
    query_emb, query_info = encode_queries_cached(model, questions, emb_dir / "queries.npz", revision, batch_size)

    with partial_path.open("w", encoding="utf-8") as handle:  # rewrite only complete questions, then append
        for qid in sorted(done):
            for r in done_by_q[qid]:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    with partial_path.open("a", encoding="utf-8") as handle:
        for q in questions:
            if str(q["question_id"]) in done:
                continue
            for r in evaluate_question(q, passage_by_id, extraction_by_pid, passage_emb, query_emb[str(q["question_id"])]):
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
            handle.flush()

    rows = load_jsonl(partial_path)
    order = {str(q["question_id"]): i for i, q in enumerate(questions)}
    rows = sorted((r for r in rows if r["question_id"] in order), key=lambda r: (order[r["question_id"]], CONFIGS.index(r["config"])))
    checks = validation_checks(rows, questions)
    if not checks["all_passed"]:
        (out_dir / "validation_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        raise AssertionError(f"Validation failed: {checks}")
    with final_path.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    partial_path.unlink()

    group_field = "question_type"
    dump_csv(out_dir / "retrieval_summary.csv", summarize(rows, group_field))
    dump_csv(out_dir / "paired_headline_contrasts.csv", headline_contrasts(rows))
    (out_dir / "validation_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    config = {
        "stage": "Phase C graph-aware ranking", "preregistration": "PREREGISTRATION_graph_ranking.md",
        "data_dir": str(data_dir.relative_to(ROOT)) if data_dir.is_relative_to(ROOT) else str(data_dir),
        "dataset_manifest": read_json(data_dir / "dataset_manifest.json") if (data_dir / "dataset_manifest.json").exists() else None,
        "question_count": len(questions), "expected_context_count": expected, "excluded_variable_context_question_ids": excluded,
        "limit": limit, "question_type_counts": dict(Counter(str(q.get("type")) for q in questions)),
        "configs": CONFIGS, "preregistered_configs": sorted(PREREGISTERED_CONFIGS), "headline_lambda": HEADLINE_LAMBDA,
        "lambdas": list(LAMBDAS), "seed_k": SEED_K, "ppr_alpha": PPR_ALPHA, "chunk_knn_k": CHUNK_K,
        "bm25": {"impl": "rank_bm25.BM25Okapi", "k1": 1.5, "b": 0.75, "tokenizer": "lowercase \\w+"},
        "dense_model": {"model_id": MODEL_ID, "revision": revision, "query_prefix": QUERY_PREFIX, "device": "cpu",
                        "passage_cache": passage_info, "query_cache": query_info},
        "entity_extraction": extraction_info, "stat_seed": STAT_SEED, "bootstrap_replicates": 10000,
        "sign_flip_draws": 20000, "packages": package_versions(), "platform": platform.platform(),
        "peak_rss_mb": max_rss_mb(), "total_seconds": time.perf_counter() - started,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    return {"rows": rows, "checks": checks, "config": config}


def validation_checks(rows: list[dict[str, Any]], questions: list[dict[str, Any]]) -> dict[str, Any]:
    by_q = {str(q["question_id"]): q for q in questions}
    permutation_failures = [f"{r['question_id']}/{r['config']}" for r in rows
                            if sorted(r["ranked_passage_ids"]) != sorted(map(str, by_q[r["question_id"]]["context_passage_ids"]))]
    per_q = Counter(r["question_id"] for r in rows)
    missing = [qid for qid in by_q if per_q.get(qid) != len(CONFIGS)]
    gold_rank_failures = [f"{r['question_id']}/{r['config']}" for r in rows
                          if any(r["ranked_passage_ids"][rank - 1] != pid for pid, rank in r["gold_ranks"].items())]
    checks = {"question_count": len(by_q), "row_count": len(rows), "expected_rows": len(by_q) * len(CONFIGS),
              "ranking_permutation_failures": permutation_failures, "questions_with_missing_configs": missing,
              "gold_rank_failures": gold_rank_failures}
    checks["all_passed"] = not permutation_failures and not missing and not gold_rank_failures and len(rows) == len(by_q) * len(CONFIGS)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="results/graph_ranking_v1/cache")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    resolve = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p  # noqa: E731
    out_dir = resolve(args.output_dir)
    try:
        result = run(resolve(args.data_dir), out_dir, resolve(args.cache_dir), args.limit, args.batch_size)
    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "failure.json").write_text(json.dumps({"error_type": type(exc).__name__, "error": str(exc),
                                                          "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise
    c = result["config"]
    print(f"questions={c['question_count']} rows={len(result['rows'])} checks_passed={result['checks']['all_passed']} "
          f"seconds={c['total_seconds']:.1f} peak_rss_mb={c['peak_rss_mb']:.0f}")


if __name__ == "__main__":
    main()
