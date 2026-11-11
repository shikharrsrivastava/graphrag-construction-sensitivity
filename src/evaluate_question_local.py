"""Stage A TF-IDF retrieval evaluation with strict HotpotQA context scoping.

Track A fits TF-IDF separately for each question's own contexts. Track B uses
the pooled entity graph to generate candidates from Track A's lexical seeds,
filters them to the question's contexts, then reranks with a shared-corpus
TF-IDF index. Track B shares graph and corpus information across questions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import networkx as nx
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.common import ROOT, read_json

K_VALUES = (2, 5, 10)


def passage_text(passage: dict[str, Any]) -> str:
    return f"{passage.get('title', '')}. {passage.get('text', '')}".strip()


def validate_inputs(questions: list[dict[str, Any]], passages: list[dict[str, Any]], contexts: dict[str, Any]) -> None:
    passage_ids = [str(p["passage_id"]) for p in passages]
    if len(passage_ids) != len(set(passage_ids)):
        raise ValueError("Passage IDs are not unique")
    known = set(passage_ids)
    qids = [str(q["question_id"]) for q in questions]
    if len(qids) != len(set(qids)):
        raise ValueError("Question IDs are not unique")
    by_id = {str(p["passage_id"]): p for p in passages}
    for q in questions:
        qid = str(q["question_id"])
        if qid not in contexts:
            raise ValueError(f"Missing context mapping for {qid}")
        cids = list(map(str, q.get("context_passage_ids", [])))
        mapped = list(map(str, contexts[qid].get("context_passage_ids", [])))
        if cids != mapped:
            raise ValueError(f"Question/context mapping mismatch for {qid}")
        if len(cids) != len(set(cids)):
            raise ValueError(f"Duplicate context passage IDs for {qid}")
        if not set(cids) <= known:
            raise ValueError(f"Unknown context passage ID for {qid}")
        gold = set(map(str, q.get("gold_supporting_passage_ids", [])))
        map_gold = set(map(str, contexts[qid].get("gold_supporting_passage_ids", [])))
        if gold != map_gold:
            raise ValueError(f"Question/context evidence mismatch for {qid}")
        if not gold or not gold <= set(cids):
            raise ValueError(f"Gold evidence missing from contexts for {qid}")
        # Ensure passage records referenced by IDs are directly addressable.
        assert all(pid in by_id for pid in cids)


def rank_local(question: dict[str, Any], passage_by_id: dict[str, dict[str, Any]]) -> tuple[list[str], float]:
    """Fit and rank only the current question's context corpus."""
    ids = list(map(str, question["context_passage_ids"]))
    docs = [passage_text(passage_by_id[pid]) for pid in ids]
    start = time.perf_counter()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs)
    query_vector = vectorizer.transform([str(question["question"])])
    scores = cosine_similarity(query_vector, matrix).ravel()
    elapsed = time.perf_counter() - start
    # Stable tie-breaking by ID makes results reproducible.
    ranked = sorted(zip(ids, scores), key=lambda item: (-float(item[1]), item[0]))
    return [pid for pid, _ in ranked], elapsed


def rank_shared(question: dict[str, Any], vectorizer, matrix, index: dict[str, int], eligible_ids: set[str]) -> tuple[list[str], float]:
    """Rank graph-generated candidates using shared corpus TF-IDF statistics."""
    ids = list(map(str, question["context_passage_ids"]))
    start = time.perf_counter()
    qv = vectorizer.transform([str(question["question"])])
    ids = [pid for pid in ids if pid in eligible_ids]
    candidate_rows = [index[pid] for pid in ids]
    scores = cosine_similarity(qv, matrix[candidate_rows]).ravel()
    elapsed = time.perf_counter() - start
    ranked = sorted(zip(ids, scores), key=lambda item: (-float(item[1]), item[0]))
    return [pid for pid, _ in ranked], elapsed


def metrics_for(question: dict[str, Any], ranked: list[str], track: str, elapsed: float, candidate_ids: set[str] | None = None) -> dict[str, Any]:
    gold = set(map(str, question["gold_supporting_passage_ids"]))
    contexts = set(map(str, question["context_passage_ids"]))
    result: dict[str, Any] = {
        "question_id": str(question["question_id"]), "question_type": str(question.get("type", "")),
        "difficulty": str(question.get("level", "")), "context_count": len(contexts), "track": track,
        "candidate_count": len(contexts if candidate_ids is None else candidate_ids), "gold_support_count": len(gold),
        "candidate_generation_recall": len(gold & (contexts if candidate_ids is None else candidate_ids)) / len(gold),
        "ranking_seconds": elapsed, "gold_supporting_passage_ids": sorted(gold),
        "ranked_passage_ids": ranked,
    }
    for k in K_VALUES:
        retrieved = set(ranked[:k])
        result[f"recall_at_{k}"] = len(gold & retrieved) / len(gold)
        result[f"complete_support_at_{k}"] = int(gold <= retrieved)
    return result


def filter_graph_candidates(context_ids: Iterable[str], graph_candidates: Iterable[str], seed_ids: Iterable[str]) -> set[str]:
    """Keep graph expansions inside the question's contexts while retaining valid seeds."""
    allowed = set(map(str, context_ids))
    return (set(map(str, graph_candidates)) | set(map(str, seed_ids))) & allowed


def bootstrap_ci(values: Iterable[float], seed: int, n_boot: int = 10000) -> tuple[float, float, float]:
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_sign_flip_p(a: Iterable[float], b: Iterable[float], seed: int, n_perm: int = 20000) -> tuple[float, float]:
    """Two-sided paired randomization test on independent question-level differences."""
    d = np.asarray(list(a), dtype=float) - np.asarray(list(b), dtype=float)
    if d.size == 0:
        return math.nan, math.nan
    observed = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    p = (1 + int(np.count_nonzero(np.abs((signs * d).mean(axis=1)) >= observed))) / (n_perm + 1)
    return float(d.mean()), p


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def summarize(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["track"], row["context_count"])].append(row)
    summary = []
    metric_names = ["candidate_generation_recall"] + [f"recall_at_{k}" for k in K_VALUES] + [f"complete_support_at_{k}" for k in K_VALUES]
    for (track, nctx), group in sorted(groups.items()):
        for metric in metric_names:
            mean, lo, hi = bootstrap_ci((float(r[metric]) for r in group), seed + nctx + sum(map(ord, metric)))
            summary.append({"track": track, "context_count": nctx, "question_count": len(group), "metric": metric,
                            "mean": mean, "ci95_low": lo, "ci95_high": hi,
                            "mean_candidate_count": statistics.mean(r["candidate_count"] for r in group),
                            "mean_ranking_seconds": statistics.mean(r["ranking_seconds"] for r in group)})
    return summary


def paired_comparisons(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_key = {(r["question_id"], r["track"]): r for r in rows}
    metrics = [f"recall_at_{k}" for k in K_VALUES] + [f"complete_support_at_{k}" for k in K_VALUES]
    tests = []
    for nctx in (10, 2):
        qids = sorted(qid for qid, track in by_key if by_key[(qid, track)]["context_count"] == nctx and track == "question_local")
        for metric in metrics:
            local = [float(by_key[(qid, "question_local")][metric]) for qid in qids]
            shared = [float(by_key[(qid, "shared_global_graph")][metric]) for qid in qids]
            delta, p = paired_sign_flip_p(local, shared, seed + nctx + sum(map(ord, metric)))
            tests.append({"context_count": nctx, "n_questions": len(qids), "comparison": "question_local - shared_global_graph",
                          "metric": metric, "mean_paired_difference": delta, "p_value": p})
    for nctx in (10, 2):
        stratum = [row for row in tests if row["context_count"] == nctx]
        adjusted = holm_adjust([row["p_value"] for row in stratum])
        for row, p_adj in zip(stratum, adjusted):
            row["holm_adjusted_p"] = p_adj
    return tests


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def run(data_dir: Path, output_dir: Path, seed: int = 2027, graph_dir: Path | None = None) -> dict[str, Any]:
    questions = read_json(data_dir / "questions.json")
    passages = read_json(data_dir / "passages.json")
    contexts = read_json(data_dir / "question_contexts.json")
    validate_inputs(questions, passages, contexts)
    passage_by_id = {str(p["passage_id"]): p for p in passages}
    global_ids = list(passage_by_id)
    global_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    global_matrix = global_vectorizer.fit_transform([passage_text(passage_by_id[pid]) for pid in global_ids])
    global_index = {pid: i for i, pid in enumerate(global_ids)}
    graph_dir = graph_dir or (ROOT / "data/pilot_500")
    graph_path = graph_dir / "entity_graph.graphml"
    graph = nx.Graph(nx.read_graphml(graph_path))
    entity_passages = read_json(graph_dir / "entity_passages.json")
    passage_entities = read_json(graph_dir / "passage_entities.json")
    if not set(passage_entities) <= set(passage_by_id):
        raise ValueError("Global graph passage-entity mapping contains IDs outside the v2 passage corpus")
    rows: list[dict[str, Any]] = []
    for q in questions:
        ranked, elapsed = rank_local(q, passage_by_id)
        rows.append(metrics_for(q, ranked, "question_local", elapsed))
        # The secondary track starts with question-local lexical seeds, expands via
        # the existing pooled entity graph, and filters graph candidates to contexts.
        contexts_for_q = set(map(str, q["context_passage_ids"]))
        seed_ids = set(ranked[:5])
        seed_entities = {str(e) for pid in seed_ids for e in passage_entities.get(pid, [])}
        graph_entities = seed_entities & set(map(str, graph.nodes))
        expanded_entities = set(graph_entities)
        for entity in graph_entities:
            expanded_entities.update(map(str, graph.neighbors(entity)))
        generated = {pid for entity in expanded_entities for pid in entity_passages.get(entity, [])}
        candidate_ids = filter_graph_candidates(contexts_for_q, generated, seed_ids)
        ranked, elapsed_global = rank_shared(q, global_vectorizer, global_matrix, global_index, candidate_ids)
        rows.append(metrics_for(q, ranked, "shared_global_graph", elapsed_global, candidate_ids))
    primary = [r for r in rows if r["context_count"] == 10]
    variable = [r for r in rows if r["context_count"] != 10]
    summary = summarize(rows, seed)
    comparisons = paired_comparisons(rows, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("primary_question_level.jsonl", primary), ("variable_context_question_level.jsonl", variable), ("shared_global_question_level.jsonl", [r for r in rows if r["track"] == "shared_global_graph"])):
        with (output_dir / name).open("w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "paired_comparisons.csv", comparisons)
    metadata = {
        "stage": "A", "dataset": "data/pilot_500_v2", "seed": seed,
        "primary": "498 ten-context questions, question-local corpus fit and ranking",
        "variable_context": "2 questions, analyzed separately",
        "track_A": "TF-IDF is fit independently on each question's contexts; candidate universe contains only those contexts",
        "track_B": "the historical 4,947-passage entity co-occurrence graph generates candidates from local TF-IDF seeds; candidates are filtered to the question's contexts before ranking with shared-corpus TF-IDF statistics; cross-question graph and IDF information are shared",
        "metrics": ["Recall@2", "Recall@5", "Recall@10", "complete supporting-evidence retrieval", "candidate-generation recall"],
        "candidate_generation_definition": "for Track A, gold evidence present in the entire context universe; for Track B, gold evidence present in graph-expanded context-filtered candidates",
        "bootstrap": {"unit": "question", "replicates": 10000, "confidence": 0.95},
        "paired_test": "two-sided paired sign-flip randomization test with 20,000 Monte Carlo permutations; Holm correction across six retrieval metrics within each context-count stratum",
        "timing_definition": "ranking_seconds includes per-question local vectorizer fit/query/scoring on Track A; Track B records per-query global-index transform and candidate scoring only. One-time global-index fit, graph load, and graph candidate expansion are excluded; total wall time is recorded in logs/progress.md",
        "software": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "global_graph": str(graph_path.relative_to(ROOT)), "global_graph_construction": "NER entity nodes; same-sentence co-occurrence edges",
        "global_graph_mapping_coverage": f"{len(passage_entities)}/{len(passage_by_id)} passages have at least one extracted entity; absent mappings are treated as empty",
        "question_count": len(questions), "primary_question_count": len(primary) // 2,
        "variable_context_question_count": len(variable) // 2, "passage_count": len(passages),
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary, "comparisons": comparisons, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/pilot_500_v2")
    parser.add_argument("--output-dir", default="results/question_local_v1")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    out = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output directory is non-empty; refusing to overwrite: {out}")
    result = run((ROOT / args.data_dir).resolve(), out, args.seed)
    print(f"questions={result['metadata']['question_count']} primary={result['metadata']['primary_question_count']} variable={result['metadata']['variable_context_question_count']}")
    print(f"outputs={out}")


if __name__ == "__main__":
    main()
