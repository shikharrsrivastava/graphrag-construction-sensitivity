"""Question-local dense retrieval and graph-construction sensitivity pilot.

Each graph is constructed from one question's ten contexts only. BGE passage
and query embeddings are cached as text/model-keyed features, never as graphs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import random
import resource
import statistics
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import sklearn
import spacy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from huggingface_hub.constants import HF_HUB_CACHE

from src.build_graph import ENTITY_LABELS, normalize_entity
from src.common import ROOT, read_json
from src.evaluate_question_local import passage_text, rank_local, validate_inputs

MODEL_ID = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
K_VALUES = (2, 5, 10)
BOOTSTRAP_REPLICATES = 10000
PERMUTATIONS = 20000
PILOT_SEED = 2027
SEED_K = 5
CHUNK_K = 2
SPARSIFY_RETAIN_FRACTION = 0.5

CONFIGS = (
    "vanilla_tfidf",
    "vanilla_bge",
    "entity_sentence_frequency_weighted",
    "entity_passage_frequency_weighted",
    "entity_sentence_binary",
    "entity_sentence_top50_sparsified",
    "chunk_bge_knn2",
)
GRAPH_CONFIGS = CONFIGS[2:]
RETRIEVAL_METRICS = [f"recall_at_{k}" for k in K_VALUES] + [f"complete_support_at_{k}" for k in K_VALUES]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def context_text(passage: dict[str, Any]) -> str:
    return passage_text(passage)


def select_pilot_questions(questions: list[dict[str, Any]], seed: int = PILOT_SEED, n: int = 50) -> list[dict[str, Any]]:
    """Select a deterministic, proportionally stratified pilot from ten-context questions."""
    primary = [q for q in questions if len(set(map(str, q.get("context_passage_ids", [])))) == 10]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in primary:
        groups[str(q.get("type", "unknown"))].append(q)
    if sum(map(len, groups.values())) < n:
        raise ValueError(f"Need at least {n} complete ten-context questions")
    quotas = {name: int(n * len(group) / len(primary)) for name, group in groups.items()}
    left = n - sum(quotas.values())
    remainders = sorted(groups, key=lambda name: (-(n * len(groups[name]) / len(primary) - quotas[name]), name))
    for name in remainders[:left]:
        quotas[name] += 1
    rng = np.random.default_rng(seed)
    selected_ids: set[str] = set()
    for name in sorted(groups):
        group = groups[name]
        ix = rng.choice(len(group), size=quotas[name], replace=False)
        selected_ids.update(str(group[int(i)]["question_id"]) for i in ix)
    return [q for q in primary if str(q["question_id"]) in selected_ids]


def passage_source_hash(passage: dict[str, Any]) -> str:
    sentences = passage.get("sentences") or str(passage.get("text", "")).splitlines()
    return sha256_text(json.dumps([str(s) for s in sentences], ensure_ascii=False, separators=(",", ":")))


def load_extraction_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # ignore an interrupted final append; it will be regenerated
            cache[str(item["passage_id"])] = item
    return cache


def extract_passage_entities(passage: dict[str, Any], nlp) -> dict[str, Any]:
    """Extract normalized entities per sentence from this passage alone."""
    sentences = passage.get("sentences") or str(passage.get("text", "")).splitlines()
    per_sentence: list[list[str]] = []
    for sentence in sentences:
        sentence = str(sentence).strip()
        entities: set[str] = set()
        if sentence:
            doc = nlp(sentence)
            for ent in doc.ents:
                if ent.label_ not in ENTITY_LABELS:
                    continue
                normalized = normalize_entity(ent.text)
                if len(normalized) >= 2:
                    entities.add(normalized)
        per_sentence.append(sorted(entities))
    return {
        "passage_id": str(passage["passage_id"]),
        "source_sha256": passage_source_hash(passage),
        "sentences": per_sentence,
        "entities": sorted({e for sentence_entities in per_sentence for e in sentence_entities}),
    }


def ensure_extraction_cache(passages: dict[str, dict[str, Any]], passage_ids: Iterable[str], cache_path: Path, model_name: str = "en_core_web_sm") -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Reuse only text-matching passage-local cache entries; otherwise extract and append."""
    start_all = time.perf_counter()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = load_extraction_cache(cache_path)
    nlp = spacy.load(model_name)
    model_version = str(nlp.meta.get("version", "unknown"))
    result: dict[str, dict[str, Any]] = {}
    hit_count = 0
    extracted_count = 0
    extraction_seconds = 0.0
    for pid in sorted(set(map(str, passage_ids))):
        passage = passages[pid]
        expected_hash = passage_source_hash(passage)
        item = cached.get(pid)
        if item and item.get("source_sha256") == expected_hash and item.get("spacy_model") == model_name and item.get("spacy_model_version") == model_version:
            result[pid] = item
            hit_count += 1
            continue
        t0 = time.perf_counter()
        item = extract_passage_entities(passage, nlp)
        extraction_seconds += time.perf_counter() - t0
        item.update({"spacy_model": model_name, "spacy_model_version": model_version, "extraction_scope": "single_passage_sentences"})
        result[pid] = item
        with cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        extracted_count += 1
    summary = {
        "model": model_name, "model_version": model_version,
        "requested_unique_passages": len(set(map(str, passage_ids))),
        "cache_hits": hit_count, "newly_extracted": extracted_count,
        "new_extraction_seconds": extraction_seconds,
        "cache_load_and_prepare_seconds": time.perf_counter() - start_all,
    }
    return result, summary


def build_entity_graph_local(
    context_ids: Iterable[str],
    extraction_by_pid: dict[str, dict[str, Any]],
    question_id: str,
    window: str = "sentence",
    weighting: str = "frequency",
    retain_fraction: float | None = None,
) -> tuple[nx.Graph, dict[str, set[str]], dict[str, Any]]:
    """Build an entity graph from only the supplied context passages."""
    allowed = set(map(str, context_ids))
    if window not in {"sentence", "passage"} or weighting not in {"frequency", "binary"}:
        raise ValueError("Unsupported entity graph window or weighting")
    if retain_fraction is not None and not 0 < retain_fraction <= 1:
        raise ValueError("retain_fraction must be in (0, 1]")
    entities_by_pid: dict[str, set[str]] = {}
    graph = nx.Graph(question_id=str(question_id), context_passage_ids=json.dumps(sorted(allowed)),
                     construction_scope="question_contexts_only", window=window, weighting=weighting)
    for pid in sorted(allowed):
        record = extraction_by_pid[pid]
        per_sentence = [set(map(str, xs)) for xs in record.get("sentences", [])]
        entities_by_pid[pid] = set().union(*per_sentence) if per_sentence else set()
        graph.add_nodes_from(entities_by_pid[pid])
    pair_counts: Counter[tuple[str, str]] = Counter()
    if window == "sentence":
        for pid in sorted(allowed):
            for sentence_entities in extraction_by_pid[pid].get("sentences", []):
                pair_counts.update(itertools.combinations(sorted(set(map(str, sentence_entities))), 2))
    else:
        for pid in sorted(allowed):
            pair_counts.update(itertools.combinations(sorted(entities_by_pid[pid]), 2))
    for (left, right), count in pair_counts.items():
        graph.add_edge(left, right, weight=float(count if weighting == "frequency" else 1.0), frequency=int(count))
    original_edge_count = graph.number_of_edges()
    removed_edge_count = 0
    if retain_fraction is not None and graph.number_of_edges() > 0:
        ranked = sorted(graph.edges(data=True), key=lambda edge: (-float(edge[2]["weight"]),
                                                                  min(str(edge[0]), str(edge[1])),
                                                                  max(str(edge[0]), str(edge[1]))))
        keep_count = max(1, int(math.ceil(retain_fraction * len(ranked))))
        keep = {frozenset((str(u), str(v))) for u, v, _ in ranked[:keep_count]}
        to_remove = [(u, v) for u, v in graph.edges if frozenset((str(u), str(v))) not in keep]
        graph.remove_edges_from(to_remove)
        removed_edge_count = len(to_remove)
    entity_to_passages: dict[str, set[str]] = defaultdict(set)
    for pid, entities in entities_by_pid.items():
        for entity in entities:
            entity_to_passages[entity].add(pid)
    mapped_passages = {pid for passage_set in entity_to_passages.values() for pid in passage_set}
    if not mapped_passages <= allowed:
        raise AssertionError("Local entity map contains an out-of-question passage")
    graph.graph["original_edge_count"] = original_edge_count
    graph.graph["removed_edge_count"] = removed_edge_count
    graph.graph["retained_edge_fraction_requested"] = retain_fraction
    return graph, entity_to_passages, {"original_edge_count": original_edge_count, "removed_edge_count": removed_edge_count}


def build_chunk_similarity_graph(context_ids: Iterable[str], embeddings: dict[str, np.ndarray], question_id: str, k_neighbors: int = CHUNK_K) -> nx.Graph:
    """Build a question-local undirected union-kNN semantic graph over passage chunks."""
    ids = sorted(set(map(str, context_ids)))
    graph = nx.Graph(question_id=str(question_id), context_passage_ids=json.dumps(ids),
                     construction_scope="question_contexts_only", node_type="passage_chunk",
                     similarity="cosine", symmetrization="union_knn", k_neighbors=int(k_neighbors))
    node_by_pid = {pid: f"chunk::{pid}" for pid in ids}
    graph.add_nodes_from(node_by_pid[pid] for pid in ids)
    if len(ids) < 2:
        return graph
    matrix = np.vstack([embeddings[pid] for pid in ids]).astype(np.float32, copy=False)
    similarities = matrix @ matrix.T
    for i, pid in enumerate(ids):
        neighbors = sorted((j for j in range(len(ids)) if j != i), key=lambda j: (-float(similarities[i, j]), ids[j]))[:min(k_neighbors, len(ids) - 1)]
        for j in neighbors:
            left, right = node_by_pid[pid], node_by_pid[ids[j]]
            graph.add_edge(left, right, weight=float(similarities[i, j]), cosine_similarity=float(similarities[i, j]))
    nx.set_node_attributes(graph, {node_by_pid[pid]: pid for pid in ids}, "passage_id")
    return graph


def local_graph_candidates(context_ids: Iterable[str], seed_passage_ids: Iterable[str], graph: nx.Graph,
                           entity_to_passages: dict[str, set[str]], neighbor_k: int = 3) -> set[str]:
    allowed = set(map(str, context_ids))
    seeds = set(map(str, seed_passage_ids))
    seed_entities = {str(e) for pid in seeds for e in entity_to_passages if pid in entity_to_passages[e]}
    expanded = set(seed_entities)
    for entity in seed_entities:
        if entity in graph:
            ranked_neighbors = sorted(graph[entity].items(), key=lambda item: (-float(item[1].get("weight", 1.0)), str(item[0])))
            expanded.update(str(neighbor) for neighbor, _ in ranked_neighbors[:neighbor_k])
    candidates = seeds | {pid for entity in expanded for pid in entity_to_passages.get(entity, set())}
    if not candidates <= allowed:
        raise AssertionError("Entity graph generated a passage outside its question contexts")
    if not set(map(str, graph.nodes)) <= set(entity_to_passages):
        raise AssertionError("Question-local graph contains an entity outside its question's passage map")
    return candidates


def chunk_graph_candidates(context_ids: Iterable[str], seed_passage_ids: Iterable[str], graph: nx.Graph) -> set[str]:
    allowed = set(map(str, context_ids))
    seeds = set(map(str, seed_passage_ids))
    chunk_by_pid = {str(data.get("passage_id")): node for node, data in graph.nodes(data=True)}
    seed_nodes = {chunk_by_pid[pid] for pid in seeds if pid in chunk_by_pid}
    expanded = set(seed_nodes)
    for node in seed_nodes:
        expanded.update(graph.neighbors(node))
    candidates = seeds | {str(graph.nodes[node]["passage_id"]) for node in expanded if "passage_id" in graph.nodes[node]}
    if not candidates <= allowed:
        raise AssertionError("Chunk graph generated a passage outside its question contexts")
    return candidates


def graph_statistics(graph: nx.Graph | None) -> dict[str, Any]:
    if graph is None:
        return {"graph_applicable": False, "graph_node_count": None, "graph_edge_count": None,
                "graph_density": None, "graph_component_count": None, "graph_isolate_count": None}
    return {
        "graph_applicable": True, "graph_node_count": graph.number_of_nodes(), "graph_edge_count": graph.number_of_edges(),
        "graph_density": float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0,
        "graph_component_count": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
        "graph_isolate_count": len(list(nx.isolates(graph))),
    }


def rank_by_scores(candidate_ids: Iterable[str], context_ids: Iterable[str], score_by_pid: dict[str, float]) -> list[str]:
    allowed = set(map(str, context_ids))
    candidates = set(map(str, candidate_ids))
    if not candidates <= allowed:
        raise AssertionError("Ranking candidates contain a passage outside the question contexts")
    return sorted(candidates, key=lambda pid: (-float(score_by_pid[pid]), pid))


def retrieval_metrics(question: dict[str, Any], config: str, candidate_ids: Iterable[str], ranked_ids: list[str], graph: nx.Graph | None,
                      timings: dict[str, float], seed_passage_ids: list[str] | None = None) -> dict[str, Any]:
    contexts = set(map(str, question["context_passage_ids"]))
    gold = set(map(str, question["gold_supporting_passage_ids"]))
    candidates = set(map(str, candidate_ids))
    if not candidates <= contexts or not set(ranked_ids) <= contexts:
        raise AssertionError(f"Context leakage in {question['question_id']} / {config}")
    row = {
        "question_id": str(question["question_id"]), "question_type": str(question.get("type", "")),
        "difficulty": str(question.get("level", "")), "context_count": len(contexts), "config": config,
        "candidate_count": len(candidates), "gold_support_count": len(gold),
        "candidate_generation_recall": len(gold & candidates) / len(gold),
        "gold_supporting_passage_ids": sorted(gold), "context_passage_ids": sorted(contexts),
        "candidate_passage_ids": sorted(candidates), "ranked_passage_ids": ranked_ids,
        "seed_passage_ids": seed_passage_ids or [],
        "graph_construction_scope": graph.graph.get("construction_scope") if graph is not None else None,
        **graph_statistics(graph), **timings,
    }
    if graph is None:
        row["graph_nodes"] = []
        row["graph_edges"] = []
    else:
        row["graph_nodes"] = sorted(map(str, graph.nodes))
        row["graph_edges"] = [
            {"source": str(u), "target": str(v), "weight": float(d.get("weight", 1.0))}
            for u, v, d in sorted(graph.edges(data=True), key=lambda x: (str(x[0]), str(x[1])))
        ]
        row["graph_original_edge_count"] = int(graph.graph.get("original_edge_count", graph.number_of_edges()))
        row["graph_removed_edge_count"] = int(graph.graph.get("removed_edge_count", 0))
        if graph.graph.get("construction_scope") != "question_contexts_only":
            raise AssertionError("Primary graph row lacks question-local graph scope marker")
    for k in K_VALUES:
        retrieved = set(ranked_ids[:k])
        row[f"recall_at_{k}"] = len(gold & retrieved) / len(gold)
        row[f"complete_support_at_{k}"] = int(gold <= retrieved)
    return row


def bootstrap_mean_ci(values: Iterable[float], seed: int, n_boot: int = BOOTSTRAP_REPLICATES) -> tuple[float, float, float]:
    vals = np.asarray(list(values), dtype=float)
    if not len(vals):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def sign_flip_test(a: Iterable[float], b: Iterable[float], seed: int, n_perm: int = PERMUTATIONS) -> tuple[float, float]:
    diff = np.asarray(list(a), dtype=float) - np.asarray(list(b), dtype=float)
    if not len(diff):
        return math.nan, math.nan
    observed = abs(float(diff.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(diff)))
    p = (1 + np.count_nonzero(np.abs((signs * diff).mean(axis=1)) >= observed)) / (n_perm + 1)
    return float(diff.mean()), float(p)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted, running = [1.0] * len(p_values), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def summarize_rows(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retrieval_fields = ["candidate_generation_recall"] + RETRIEVAL_METRICS
    retrieval = []
    for config in CONFIGS:
        for group_name in ("all", "bridge", "comparison"):
            group = [r for r in rows if r["config"] == config and (group_name == "all" or r["question_type"] == group_name)]
            if not group:
                continue
            for metric in retrieval_fields:
                mean, low, high = bootstrap_mean_ci([float(r[metric]) for r in group], seed + sum(map(ord, f"{config}:{group_name}:{metric}")))
                retrieval.append({"config": config, "question_group": group_name, "question_count": len(group), "metric": metric,
                                 "mean": mean, "ci95_low": low, "ci95_high": high,
                                 "mean_candidate_count": statistics.mean(r["candidate_count"] for r in group),
                                 "mean_local_tfidf_fit_and_retrieval_seconds": statistics.mean(r.get("local_tfidf_fit_and_retrieval_seconds", 0.0) for r in group),
                                 "mean_dense_query_scoring_seconds": statistics.mean(r.get("dense_query_scoring_seconds", 0.0) for r in group),
                                 "mean_candidate_generation_seconds": statistics.mean(r.get("candidate_generation_seconds", 0.0) for r in group),
                                 "mean_candidate_reranking_seconds": statistics.mean(r.get("candidate_reranking_seconds", 0.0) for r in group),
                                 "mean_query_retrieval_seconds": statistics.mean(r.get("query_retrieval_seconds", 0.0) for r in group),
                                 "mean_graph_construction_seconds": statistics.mean(r.get("graph_construction_seconds", 0.0) for r in group)})
    graph_fields = ["graph_node_count", "graph_edge_count", "graph_density", "graph_component_count", "graph_isolate_count"]
    graph_summary = []
    for config in CONFIGS:
        for group_name in ("all", "bridge", "comparison"):
            group = [r for r in rows if r["config"] == config and r["graph_applicable"] and (group_name == "all" or r["question_type"] == group_name)]
            if not group:
                continue
            for metric in graph_fields:
                vals = [float(r[metric]) for r in group]
                mean, low, high = bootstrap_mean_ci(vals, seed + sum(map(ord, f"graph:{config}:{group_name}:{metric}")))
                graph_summary.append({"config": config, "question_group": group_name, "question_count": len(group), "metric": metric,
                                      "mean": mean, "ci95_low": low, "ci95_high": high,
                                      "mean_graph_construction_seconds": statistics.mean(r["graph_construction_seconds"] for r in group)})
    return retrieval, graph_summary


def paired_comparisons(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_key = {(r["question_id"], r["config"]): r for r in rows}
    contrasts = [("vanilla_bge", "vanilla_tfidf", RETRIEVAL_METRICS)]
    contrasts += [("vanilla_bge", config, ["candidate_generation_recall"] + RETRIEVAL_METRICS) for config in GRAPH_CONFIGS]
    # Directly test the construction choices against the sentence-frequency reference.
    contrasts += [("entity_sentence_frequency_weighted", config,
                   ["candidate_generation_recall"] + RETRIEVAL_METRICS)
                  for config in GRAPH_CONFIGS if config != "entity_sentence_frequency_weighted"]
    tests = []
    for ref, other, metrics in contrasts:
        qids = sorted(qid for qid, cfg in by_key if cfg == ref and (qid, other) in by_key)
        for metric in metrics:
            delta, p = sign_flip_test([by_key[(qid, other)][metric] for qid in qids],
                                      [by_key[(qid, ref)][metric] for qid in qids],
                                      seed + sum(map(ord, f"{ref}:{other}:{metric}")))
            tests.append({"n_questions": len(qids), "comparison": f"{other} - {ref}", "metric": metric,
                          "mean_paired_difference": delta, "p_value": p})
    adj = holm_adjust([float(row["p_value"]) for row in tests])
    for row, p_adjusted in zip(tests, adj):
        row["holm_adjusted_p"] = p_adjusted
    return tests


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left, right = set(a), set(b)
    return 1.0 if not left and not right else len(left & right) / len(left | right)


def stability_rows(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_q[row["question_id"]].append(row)
    raw = []
    for qid, qrows in by_q.items():
        for left, right in itertools.combinations(sorted(qrows, key=lambda r: r["config"]), 2):
            for k in K_VALUES:
                raw.append({"question_id": qid, "config_a": left["config"], "config_b": right["config"], "k": k,
                            "top_k_jaccard": jaccard(left["ranked_passage_ids"][:k], right["ranked_passage_ids"][:k]),
                            "interpretation": "cross_configuration_retrieval_overlap_not_seed_stability"})
    summary = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        groups[(row["config_a"], row["config_b"], row["k"])].append(row)
    for (a, b, k), group in sorted(groups.items()):
        mean, low, high = bootstrap_mean_ci([r["top_k_jaccard"] for r in group], seed + k + sum(map(ord, a + b)))
        summary.append({"config_a": a, "config_b": b, "k": k, "question_count": len(group), "mean_jaccard": mean,
                        "ci95_low": low, "ci95_high": high,
                        "interpretation": "cross_configuration_retrieval_overlap_not_seed_stability"})
    return raw, summary


def dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def max_rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024


def cached_model_snapshot() -> tuple[Path | None, str | None]:
    """Return an installed BGE snapshot and Hub commit without making network calls."""
    repo_dir = Path(HF_HUB_CACHE) / "models--BAAI--bge-small-en-v1.5"
    ref = repo_dir / "refs" / "main"
    if not ref.exists():
        return None, None
    revision = ref.read_text(encoding="utf-8").strip()
    snapshot = repo_dir / "snapshots" / revision
    if revision and snapshot.is_dir() and (snapshot / "config.json").exists():
        return snapshot, revision
    return None, None


def resolve_model_revision(model: SentenceTransformer, fallback: str | None = None) -> str:
    try:
        revision = model._first_module().auto_model.config._commit_hash
        if revision:
            return str(revision)
    except Exception:
        pass
    return str(fallback or "unknown")


def encode_cached(model: SentenceTransformer, texts_by_id: dict[str, str], cache_path: Path, revision: str,
                  batch_size: int = 16) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load matching normalized passage vectors and encode only cache misses."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    hash_by_id = {pid: sha256_text(text) for pid, text in texts_by_id.items()}
    existing: dict[str, np.ndarray] = {}
    manifest_path = cache_path.with_suffix(".json")
    if cache_path.exists() and manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        if meta.get("model_id") == MODEL_ID and meta.get("revision") == revision and meta.get("normalized") is True:
            with np.load(cache_path, allow_pickle=False) as bundle:
                for pid, text_hash, embedding in zip(bundle["passage_ids"].astype(str), bundle["text_hashes"].astype(str), bundle["embeddings"]):
                    if pid in hash_by_id and hash_by_id[pid] == text_hash:
                        existing[pid] = embedding.astype(np.float32, copy=False)
    missing = [pid for pid in sorted(texts_by_id) if pid not in existing]
    t0 = time.perf_counter()
    if missing:
        encoded = model.encode([texts_by_id[pid] for pid in missing], batch_size=batch_size, show_progress_bar=False,
                               convert_to_numpy=True, normalize_embeddings=True)
        for pid, vector in zip(missing, encoded):
            existing[pid] = np.asarray(vector, dtype=np.float32)
        tmp = cache_path.with_suffix(".tmp.npz")
        ids = sorted(existing)
        np.savez_compressed(tmp, passage_ids=np.asarray(ids), text_hashes=np.asarray([hash_by_id[pid] for pid in ids]),
                            embeddings=np.vstack([existing[pid] for pid in ids]).astype(np.float32))
        tmp.replace(cache_path)
        manifest_path.write_text(json.dumps({"model_id": MODEL_ID, "revision": revision, "normalized": True,
                                             "embedding_dim": int(existing[ids[0]].shape[0]) if ids else 0,
                                             "passage_count": len(ids), "text_hashes": hash_by_id}, indent=2) + "\n")
    elapsed = time.perf_counter() - t0
    return existing, {"new_embedding_count": len(missing), "cache_hit_count": len(texts_by_id) - len(missing),
                      "embedding_compute_seconds": elapsed, "cache_path": str(cache_path.relative_to(ROOT))}


def encode_queries_cached(model: SentenceTransformer, questions: list[dict[str, Any]], cache_path: Path, revision: str,
                          batch_size: int = 16) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = {str(q["question_id"]): QUERY_PREFIX + str(q["question"]) for q in questions}
    hashes = {qid: sha256_text(txt) for qid, txt in inputs.items()}
    existing: dict[str, np.ndarray] = {}
    manifest_path = cache_path.with_suffix(".json")
    if cache_path.exists() and manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        if meta.get("model_id") == MODEL_ID and meta.get("revision") == revision:
            with np.load(cache_path, allow_pickle=False) as bundle:
                for qid, text_hash, embedding in zip(bundle["question_ids"].astype(str), bundle["text_hashes"].astype(str), bundle["embeddings"]):
                    if qid in hashes and hashes[qid] == text_hash:
                        existing[qid] = embedding.astype(np.float32, copy=False)
    missing = [qid for qid in sorted(inputs) if qid not in existing]
    t0 = time.perf_counter()
    if missing:
        encoded = model.encode([inputs[qid] for qid in missing], batch_size=batch_size, show_progress_bar=False,
                               convert_to_numpy=True, normalize_embeddings=True)
        for qid, vector in zip(missing, encoded):
            existing[qid] = np.asarray(vector, dtype=np.float32)
        tmp = cache_path.with_suffix(".tmp.npz")
        ids = sorted(existing)
        np.savez_compressed(tmp, question_ids=np.asarray(ids), text_hashes=np.asarray([hashes[qid] for qid in ids]),
                            embeddings=np.vstack([existing[qid] for qid in ids]).astype(np.float32))
        tmp.replace(cache_path)
        manifest_path.write_text(json.dumps({"model_id": MODEL_ID, "revision": revision, "normalized": True,
                                             "query_prefix": QUERY_PREFIX, "question_count": len(ids),
                                             "text_hashes": hashes}, indent=2) + "\n")
    elapsed = time.perf_counter() - t0
    return existing, {"new_query_count": len(missing), "cache_hit_count": len(inputs) - len(missing), "query_compute_seconds": elapsed}


def make_configs_for_question(q: dict[str, Any], passage_by_id: dict[str, dict[str, Any]],
                              extractions: dict[str, dict[str, Any]], passage_embeddings: dict[str, np.ndarray],
                              query_embedding: np.ndarray) -> list[dict[str, Any]]:
    qid = str(q["question_id"])
    contexts = list(map(str, q["context_passage_ids"]))
    context_set = set(contexts)
    gold = set(map(str, q["gold_supporting_passage_ids"]))
    outputs: list[dict[str, Any]] = []

    # Existing Stage A function: refit TF-IDF on this question's context passages.
    ranked_tfidf, tfidf_seconds = rank_local(q, passage_by_id)
    outputs.append(retrieval_metrics(q, "vanilla_tfidf", contexts, ranked_tfidf, None,
                                     {"graph_construction_seconds": 0.0, "query_embedding_seconds": 0.0,
                                      "local_tfidf_fit_and_retrieval_seconds": tfidf_seconds,
                                      "dense_query_scoring_seconds": 0.0, "candidate_generation_seconds": 0.0,
                                      "candidate_reranking_seconds": 0.0, "query_retrieval_seconds": tfidf_seconds}))

    # Query vectors are precomputed and normalized; all comparison methods rank these same contexts.
    passage_matrix = np.vstack([passage_embeddings[pid] for pid in contexts]).astype(np.float32, copy=False)
    q_score_start = time.perf_counter()
    dense_scores = passage_matrix @ query_embedding
    score_time = time.perf_counter() - q_score_start
    score_by_pid = {pid: float(score) for pid, score in zip(contexts, dense_scores)}
    rank_start = time.perf_counter()
    dense_rank = rank_by_scores(contexts, context_set, score_by_pid)
    vanilla_rank_seconds = time.perf_counter() - rank_start
    outputs.append(retrieval_metrics(q, "vanilla_bge", contexts, dense_rank, None,
                                     {"graph_construction_seconds": 0.0, "query_embedding_seconds": 0.0,
                                      "local_tfidf_fit_and_retrieval_seconds": 0.0,
                                      "dense_query_scoring_seconds": score_time, "candidate_generation_seconds": 0.0,
                                      "candidate_reranking_seconds": vanilla_rank_seconds,
                                      "query_retrieval_seconds": score_time + vanilla_rank_seconds}))
    seeds = dense_rank[:min(SEED_K, len(dense_rank))]

    entity_specs = (
        ("entity_sentence_frequency_weighted", "sentence", "frequency", None),
        ("entity_passage_frequency_weighted", "passage", "frequency", None),
        ("entity_sentence_binary", "sentence", "binary", None),
        ("entity_sentence_top50_sparsified", "sentence", "frequency", SPARSIFY_RETAIN_FRACTION),
    )
    for config, window, weighting, retain in entity_specs:
        graph_start = time.perf_counter()
        graph, entity_to_passages, pruning = build_entity_graph_local(contexts, extractions, qid, window, weighting, retain)
        graph_build_seconds = time.perf_counter() - graph_start
        candidate_start = time.perf_counter()
        candidates = local_graph_candidates(contexts, seeds, graph, entity_to_passages)
        candidate_seconds = time.perf_counter() - candidate_start
        rank_start = time.perf_counter()
        ranked = rank_by_scores(candidates, context_set, score_by_pid)
        rerank_seconds = time.perf_counter() - rank_start
        # Charge seed scoring to each config's full query path; it is reused computationally here.
        graph.graph.update(pruning)
        outputs.append(retrieval_metrics(q, config, candidates, ranked, graph,
                                         {"graph_construction_seconds": graph_build_seconds,
                                          "query_embedding_seconds": 0.0, "local_tfidf_fit_and_retrieval_seconds": 0.0,
                                          "dense_query_scoring_seconds": score_time,
                                          "candidate_generation_seconds": candidate_seconds,
                                          "candidate_reranking_seconds": rerank_seconds,
                                          "query_retrieval_seconds": score_time + candidate_seconds + rerank_seconds}))

    graph_start = time.perf_counter()
    chunk_graph = build_chunk_similarity_graph(contexts, passage_embeddings, qid, CHUNK_K)
    graph_build_seconds = time.perf_counter() - graph_start
    candidate_start = time.perf_counter()
    candidates = chunk_graph_candidates(contexts, seeds, chunk_graph)
    candidate_seconds = time.perf_counter() - candidate_start
    rank_start = time.perf_counter()
    ranked = rank_by_scores(candidates, context_set, score_by_pid)
    rerank_seconds = time.perf_counter() - rank_start
    outputs.append(retrieval_metrics(q, "chunk_bge_knn2", candidates, ranked, chunk_graph,
                                     {"graph_construction_seconds": graph_build_seconds,
                                      "query_embedding_seconds": 0.0, "local_tfidf_fit_and_retrieval_seconds": 0.0,
                                      "dense_query_scoring_seconds": score_time,
                                      "candidate_generation_seconds": candidate_seconds,
                                      "candidate_reranking_seconds": rerank_seconds,
                                      "query_retrieval_seconds": score_time + candidate_seconds + rerank_seconds}))
    for row in outputs:
        if row["config"] in GRAPH_CONFIGS and not set(row["candidate_passage_ids"]) <= context_set:
            raise AssertionError(f"Candidate leakage for {qid} / {row['config']}")
    return outputs


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_scope(data_dir: Path, output_root: Path, scope: str = "pilot50", sample_seed: int = PILOT_SEED,
              batch_size: int = 16, run_name: str | None = None) -> dict[str, Any]:
    questions_all = read_json(data_dir / "questions.json")
    passages_all = read_json(data_dir / "passages.json")
    contexts = read_json(data_dir / "question_contexts.json")
    validate_inputs(questions_all, passages_all, contexts)
    primary = [q for q in questions_all if len(set(map(str, q["context_passage_ids"]))) == 10]
    if scope == "pilot50":
        questions = select_pilot_questions(questions_all, sample_seed, 50)
    elif scope == "primary498":
        questions = primary
    else:
        raise ValueError(f"Unknown scope {scope}")
    passage_by_id = {str(p["passage_id"]): p for p in passages_all}
    selected_context_ids = sorted({str(pid) for q in questions for pid in q["context_passage_ids"]})
    out_dir = output_root / (run_name or scope)
    cache_dir = output_root / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_raw = out_dir / "question_results.jsonl"
    if final_raw.exists() and final_raw.stat().st_size:
        raise FileExistsError(f"Refusing to overwrite existing evaluation outputs: {out_dir}")
    started = time.perf_counter()
    run_status: dict[str, Any] = {"scope": scope, "status": "started", "sample_seed": sample_seed}
    (out_dir / "run_config.json").write_text(json.dumps(run_status, indent=2) + "\n")
    (out_dir / "selected_questions.json").write_text(json.dumps({"scope": scope, "sample_seed": sample_seed,
        "question_ids": [str(q["question_id"]) for q in questions],
        "question_type_counts": dict(Counter(str(q.get("type", "")) for q in questions)),
        "context_count": 10}, indent=2) + "\n", encoding="utf-8")
    extract_cache_path = cache_dir / "entity_extractions.jsonl"
    extraction_by_pid, extraction_info = ensure_extraction_cache(passage_by_id, selected_context_ids, extract_cache_path)
    # Cross-check the older aggregate mapping as a provenance diagnostic only; graph construction
    # below uses the fresh/cacheable sentence-level passage-local extraction records.
    old_mapping = read_json(ROOT / "data/pilot_500/passage_entities.json")
    mapping_compared = 0
    mapping_exact_match = 0
    for pid in selected_context_ids:
        if pid in old_mapping:
            mapping_compared += 1
            if set(old_mapping[pid]) == set(extraction_by_pid[pid]["entities"]):
                mapping_exact_match += 1

    torch.set_num_threads(2)
    device = "cpu"  # MPS and CUDA were unavailable during environment inspection.
    load_start = time.perf_counter()
    snapshot_path, snapshot_revision = cached_model_snapshot()
    model_source = str(snapshot_path) if snapshot_path else MODEL_ID
    model = SentenceTransformer(model_source, device=device)
    model.max_seq_length = 512
    model_load_seconds = time.perf_counter() - load_start
    model_revision = resolve_model_revision(model, snapshot_revision)
    model_cache_dir = cache_dir / "embeddings" / model_revision
    texts = {pid: context_text(passage_by_id[pid]) for pid in selected_context_ids}
    passage_embeddings, passage_cache_info = encode_cached(model, texts, model_cache_dir / "passages.npz", model_revision, batch_size)
    query_embeddings, query_cache_info = encode_queries_cached(model, questions, model_cache_dir / "queries.npz", model_revision, batch_size)

    rows: list[dict[str, Any]] = []
    for q in questions:
        qid = str(q["question_id"])
        rows.extend(make_configs_for_question(q, passage_by_id, extraction_by_pid, passage_embeddings, query_embeddings[qid]))
    if len(rows) != len(questions) * len(CONFIGS):
        raise AssertionError("Unexpected question/configuration row count")
    retrieval_summary, graph_summary = summarize_rows(rows, sample_seed)
    paired = paired_comparisons(rows, sample_seed)
    stability_raw, stability_summary = stability_rows(rows, sample_seed)

    # Independent scope guards on serialized question rows.
    question_by_id = {str(q["question_id"]): q for q in questions}
    checks = {"question_count": len(questions), "config_count": len(CONFIGS), "row_count": len(rows),
              "expected_row_count": len(questions) * len(CONFIGS), "context_violation_count": 0,
              "graph_scope_violation_count": 0, "candidate_scope_violation_count": 0,
              "cross_question_entity_or_chunk_node_leakage_count": 0}
    for row in rows:
        q = question_by_id[row["question_id"]]
        allowed = set(map(str, q["context_passage_ids"]))
        if not set(row["ranked_passage_ids"]) <= allowed:
            checks["context_violation_count"] += 1
        if not set(row["candidate_passage_ids"]) <= allowed:
            checks["candidate_scope_violation_count"] += 1
        if row["graph_applicable"]:
            if row.get("graph_construction_scope") != "question_contexts_only":
                checks["graph_scope_violation_count"] += 1
            graph_nodes = set(row.get("graph_nodes", []))
            if row["config"] == "chunk_bge_knn2":
                expected_nodes = {f"chunk::{pid}" for pid in allowed}
                if graph_nodes != expected_nodes:
                    checks["graph_scope_violation_count"] += 1
                    checks["cross_question_entity_or_chunk_node_leakage_count"] += len(graph_nodes - expected_nodes)
            else:
                local_entities = {e for pid in allowed for e in extraction_by_pid[pid]["entities"]}
                if not graph_nodes <= local_entities:
                    checks["graph_scope_violation_count"] += 1
                    checks["cross_question_entity_or_chunk_node_leakage_count"] += len(graph_nodes - local_entities)
            for edge in row.get("graph_edges", []):
                if edge["source"] not in graph_nodes or edge["target"] not in graph_nodes:
                    checks["graph_scope_violation_count"] += 1
    if any(checks[k] for k in ("context_violation_count", "graph_scope_violation_count", "candidate_scope_violation_count")):
        raise AssertionError(f"Scope validation failed: {checks}")
    checks["every_graph_serialized"] = all(not r["graph_applicable"] or isinstance(r["graph_nodes"], list) for r in rows)
    checks["all_question_contexts_ten"] = all(r["context_count"] == 10 for r in rows)
    checks["gold_supports_present"] = all(set(r["gold_supporting_passage_ids"]) <= set(r["context_passage_ids"]) for r in rows)
    write_jsonl(final_raw, rows)
    dump_csv(out_dir / "retrieval_summary.csv", retrieval_summary)
    dump_csv(out_dir / "graph_summary.csv", graph_summary)
    dump_csv(out_dir / "paired_comparisons.csv", paired)
    dump_csv(out_dir / "retrieval_stability_pairwise.csv", stability_raw)
    dump_csv(out_dir / "retrieval_stability_summary.csv", stability_summary)
    checks["entity_graph_node_scope_checked"] = True
    checks["chunk_graph_node_scope_checked"] = True
    (out_dir / "validation_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    elapsed_total = time.perf_counter() - started
    efficiency = [
        {"stage": "spacy_extraction_new_passages", "seconds": extraction_info["new_extraction_seconds"], "items": extraction_info["newly_extracted"], "peak_process_rss_mb": max_rss_mb()},
        {"stage": "bge_model_load", "seconds": model_load_seconds, "items": 1, "peak_process_rss_mb": max_rss_mb()},
        {"stage": "bge_passage_embedding_compute_or_cache", "seconds": passage_cache_info["embedding_compute_seconds"], "items": len(selected_context_ids), "peak_process_rss_mb": max_rss_mb()},
        {"stage": "bge_query_embedding_compute_or_cache", "seconds": query_cache_info["query_compute_seconds"], "items": len(questions), "peak_process_rss_mb": max_rss_mb()},
        {"stage": "total_scope_run_after_input_load", "seconds": elapsed_total, "items": len(rows), "peak_process_rss_mb": max_rss_mb()},
    ]
    dump_csv(out_dir / "efficiency.csv", efficiency)
    metadata = {
        "stage": "Phase B dense retrieval / local graph construction",
        "scope": scope, "dataset": "data/pilot_500_v2", "question_count": len(questions),
        "question_type_counts": dict(Counter(str(q.get("type", "")) for q in questions)),
        "context_count_per_question": 10, "sample_seed": sample_seed, "selection": "stratified random by question type from 498 ten-context questions" if scope == "pilot50" else "all 498 ten-context questions",
        "configs": list(CONFIGS), "seed_k": SEED_K, "entity_neighbor_k": 3, "chunk_knn_k": CHUNK_K,
        "sparsification": {"method": "retain strongest ceiling(50% of sentence-frequency edges)", "tie_break": "descending weight then lexical endpoint IDs", "edges_retained": "see per-question original/removed graph edge counts"},
        "entity_extraction": {**extraction_info, "existing_mapping_used_for_graphs": False,
                              "existing_mapping_crosscheck_exact": f"{mapping_exact_match}/{mapping_compared}",
                              "existing_mapping_provenance": "mapping manifest does not record extractor/model revision; fresh passage-local sentence extraction is used for all graph construction"},
        "dense_model": {"model_id": MODEL_ID, "resolved_revision": model_revision, "device": device,
                        "max_seq_length": int(model.max_seq_length), "embedding_dimension": int(next(iter(passage_embeddings.values())).shape[0]),
                        "query_prefix": QUERY_PREFIX, "passages_prefixed": False, "normalized_embeddings": True,
                        "passage_cache": passage_cache_info, "query_cache": query_cache_info,
                        "model_load_seconds": model_load_seconds},
        "global_model_data_sharing": "passage embeddings are independently encoded from text; each graph, seed set, candidate pool, and ranking is built per question from its ten contexts",
        "statistics": {"bootstrap_unit": "question", "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                       "confidence_interval": "percentile 2.5/97.5 percentiles of bootstrap means",
                       "paired_test": "two-sided paired sign-flip Monte Carlo randomization, plus-one correction",
                       "permutations": PERMUTATIONS,
                       "holm_family": "all 69 prespecified baseline and direct graph-construction method/outcome contrasts in paired_comparisons.csv",
                       "paired_comparison_count": 69},
        "runtime_semantics": "efficiency.csv records one-time model load, passage/query encoding and total scope runtime; raw rows record graph build plus query retrieval split into scoring, candidate generation, reranking, and query_retrieval_seconds. Query retrieval starts from cached embeddings and excludes embedding and graph construction; local TF-IDF timing includes its per-question fit and ranking.",
        "resource": {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available(),
                     "python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                     "networkx": nx.__version__, "spacy": spacy.__version__, "spacy_model": extraction_info["model"],
                     "spacy_model_version": extraction_info["model_version"], "torch_threads": torch.get_num_threads(),
                     "process_peak_rss_mb": max_rss_mb()},
        "validation": checks, "total_scope_runtime_seconds": elapsed_total,
    }
    (out_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "metadata": metadata, "checks": checks, "out_dir": out_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/pilot_500_v2")
    parser.add_argument("--output-dir", default="results/dense_graph_v1")
    parser.add_argument("--scope", choices=("pilot50", "primary498"), default="pilot50")
    parser.add_argument("--seed", type=int, default=PILOT_SEED)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--run-name", default=None, help="Optional result subdirectory name; cache remains shared under output-dir/cache")
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    try:
        result = run_scope((ROOT / args.data_dir).resolve(), output_root, args.scope, args.seed, args.batch_size, args.run_name)
    except Exception as exc:
        out_dir = output_root / (args.run_name or args.scope)
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {"scope": args.scope, "status": "failed", "error_type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc()}
        (out_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
    print(f"scope={args.scope} questions={result['metadata']['question_count']} configs={len(CONFIGS)} rows={len(result['rows'])}")
    print(f"output={result['out_dir']}")
    print(f"peak_rss_mb={result['metadata']['resource']['process_peak_rss_mb']:.1f} total_seconds={result['metadata']['total_scope_runtime_seconds']:.2f}")


if __name__ == "__main__":
    main()
