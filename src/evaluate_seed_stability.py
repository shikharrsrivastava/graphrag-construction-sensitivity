"""Measure clustering-seed and retrieved-evidence stability on an unchanged graph.

Unlike ARI against a same-seed baseline (which equals 1 for the original graph
by construction), this script compares partitions produced on the *same* graph
under different algorithm seeds. It also compares the final retrieved passage
sets across those seeds for every question.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.cluster_graph import partition_graph, partition_payload
from src.common import ROOT, read_json, write_json
from src.evaluate_retrieval import evaluate

METHOD_TO_RETRIEVAL = {
    "leiden": "leiden_community",
    "louvain": "louvain_community",
}
RETRIEVAL_METHODS = ["tfidf_baseline", "one_hop_graph", "leiden_community", "louvain_community"]


def community_labels(communities: list[set[str]], nodes: list[str]) -> list[int]:
    membership: dict[str, int] = {}
    for label, group in enumerate(communities):
        for node in group:
            membership[str(node)] = label
    return [membership.get(node, -1) for node in nodes]


def parse_ids(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {item for item in str(value).split(";") if item}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def compare_partitions(
    partitions: dict[int, list[set[str]]],
    node_order: list[str],
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_a, seed_b in combinations(sorted(partitions), 2):
        labels_a = community_labels(partitions[seed_a], node_order)
        labels_b = community_labels(partitions[seed_b], node_order)
        rows.append({
            "method": method,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "ari": float(adjusted_rand_score(labels_a, labels_b)),
            "nmi": float(normalized_mutual_info_score(labels_a, labels_b)),
            "community_count_a": len(partitions[seed_a]),
            "community_count_b": len(partitions[seed_b]),
        })
    return rows


def compare_retrieval_sets(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, k, question_id), group in raw.groupby(["method", "k", "question_id"]):
        seed_to_ids = {
            int(row.seed): parse_ids(row.retrieved_passage_ids)
            for row in group.itertuples(index=False)
        }
        for seed_a, seed_b in combinations(sorted(seed_to_ids), 2):
            rows.append({
                "method": method,
                "k": int(k),
                "question_id": str(question_id),
                "seed_a": seed_a,
                "seed_b": seed_b,
                "retrieval_jaccard": jaccard(seed_to_ids[seed_a], seed_to_ids[seed_b]),
            })
    return pd.DataFrame(rows)


def bootstrap_question_level_summary(
    pairwise: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 2027,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize seed-pair overlaps, resampling questions as independent units.

    Each question contributes one value: its mean Jaccard across all unordered
    seed pairs. This prevents the 45 seed pairs per question from being counted
    as 45 independent questions.
    """
    if pairwise.empty:
        return pd.DataFrame(), pd.DataFrame()

    question_level = pairwise.groupby(
        ["method", "k", "question_id"], as_index=False
    ).agg(question_mean_jaccard=("retrieval_jaccard", "mean"))

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (method, k), block in question_level.groupby(["method", "k"], sort=True):
        values = block["question_mean_jaccard"].to_numpy(dtype=float)
        if len(values):
            indices = rng.integers(0, len(values), size=(n_boot, len(values)))
            bootstrap_means = values[indices].mean(axis=1)
            low, high = np.quantile(bootstrap_means, [0.025, 0.975])
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        else:
            low = high = mean = sd = float("nan")
        rows.append({
            "method": method,
            "k": int(k),
            "n_independent_questions": int(len(values)),
            "mean_question_level_jaccard": mean,
            "sd_question_level_jaccard": sd,
            "ci95_percentile_low": float(low),
            "ci95_percentile_high": float(high),
            "bootstrap_resamples": int(n_boot),
            "bootstrap_seed": int(seed),
        })
    return question_level, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="data/pilot_50/questions.json")
    parser.add_argument("--passages", default="data/pilot_50/passages.json")
    parser.add_argument("--graph", default="data/pilot_50/entity_graph.graphml")
    parser.add_argument("--entity-passages", default="data/pilot_50/entity_passages.json")
    parser.add_argument("--passage-entities", default="data/pilot_50/passage_entities.json")
    parser.add_argument("--output-dir", default="results/pilot_50/seed_stability")
    parser.add_argument("--methods", nargs="+", choices=["leiden", "louvain"], default=["leiden", "louvain"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    parser.add_argument("--seed-k", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error("provide at least two distinct seeds")

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path

    questions = read_json(resolve(args.questions))
    passages = read_json(resolve(args.passages))
    entity_passages = read_json(resolve(args.entity_passages))
    passage_entities = read_json(resolve(args.passage_entities))
    graph_path = resolve(args.graph)
    graph = nx.Graph(nx.read_graphml(graph_path))
    node_order = sorted(map(str, graph.nodes))
    output_dir = resolve(args.output_dir)
    partition_dir = output_dir / "partitions"
    partition_dir.mkdir(parents=True, exist_ok=True)

    partition_rows: list[dict[str, Any]] = []
    retrieval_frames: list[pd.DataFrame] = []
    partitions_by_method: dict[str, dict[int, list[set[str]]]] = {method: {} for method in args.methods}

    # First persist each method's partitions on the unchanged graph.
    for method in args.methods:
        for seed in args.seeds:
            communities = partition_graph(graph, method, seed)
            partitions_by_method[method][seed] = communities
            payload = partition_payload(str(graph_path), method, seed, communities)
            write_json(payload, partition_dir / f"{method}_seed{seed}.json")
        partition_rows.extend(compare_partitions(partitions_by_method[method], node_order, method))

    # Evaluate once per seed with all requested partitions, avoiding duplicate
    # TF-IDF and one-hop rows when both Leiden and Louvain are selected.
    for seed in args.seeds:
        leiden_map = {}
        louvain_map = {}
        if "leiden" in partitions_by_method:
            leiden_map = {
                node: cid
                for cid, group in enumerate(partitions_by_method["leiden"][seed])
                for node in group
            }
        if "louvain" in partitions_by_method:
            louvain_map = {
                node: cid
                for cid, group in enumerate(partitions_by_method["louvain"][seed])
                for node in group
            }
        frame = evaluate(
            questions, passages, graph, entity_passages, passage_entities,
            leiden_map, louvain_map, args.seed_k, args.k_values,
        )
        keep = ["tfidf_baseline", "one_hop_graph"]
        if "leiden" in args.methods:
            keep.append("leiden_community")
        if "louvain" in args.methods:
            keep.append("louvain_community")
        frame = frame[frame["method"].isin(keep)].copy()
        frame["seed"] = int(seed)
        retrieval_frames.append(frame)

    retrieval_raw = pd.concat(retrieval_frames, ignore_index=True)
    retrieval_pairwise = compare_retrieval_sets(retrieval_raw)
    partition_pairs = pd.DataFrame(partition_rows)
    retrieval_summary = retrieval_pairwise.groupby(["method", "k"], as_index=False).agg(
        question_seed_pairs=("retrieval_jaccard", "count"),
        mean_retrieval_jaccard=("retrieval_jaccard", "mean"),
        sd_retrieval_jaccard=("retrieval_jaccard", "std"),
        median_retrieval_jaccard=("retrieval_jaccard", "median"),
    )
    question_level_jaccard, jaccard_ci_summary = bootstrap_question_level_summary(
        retrieval_pairwise, n_boot=args.bootstrap, seed=args.bootstrap_seed
    )
    partition_summary = partition_pairs.groupby("method", as_index=False).agg(
        seed_pairs=("ari", "count"),
        mean_pairwise_ari=("ari", "mean"),
        sd_pairwise_ari=("ari", "std"),
        mean_pairwise_nmi=("nmi", "mean"),
        sd_pairwise_nmi=("nmi", "std"),
        min_pairwise_ari=("ari", "min"),
        max_pairwise_ari=("ari", "max"),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_raw.to_csv(output_dir / "seed_retrieval_raw.csv", index=False)
    retrieval_pairwise.to_csv(output_dir / "retrieval_jaccard_pairwise.csv", index=False)
    retrieval_summary.to_csv(output_dir / "retrieval_jaccard_summary.csv", index=False)
    question_level_jaccard.to_csv(output_dir / "retrieval_jaccard_question_level.csv", index=False)
    jaccard_ci_summary.to_csv(output_dir / "retrieval_jaccard_bootstrap_ci.csv", index=False)
    partition_pairs.to_csv(output_dir / "partition_similarity_pairwise.csv", index=False)
    partition_summary.to_csv(output_dir / "partition_similarity_summary.csv", index=False)

    # Plot the graph-method retrieval overlap (full-corpus TF-IDF is seed invariant).
    plot = retrieval_summary[retrieval_summary["method"].isin(["leiden_community", "louvain_community"])]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in ["leiden_community", "louvain_community"]:
        block = plot[plot["method"] == method].sort_values("k")
        if not block.empty:
            ax.plot(block["k"], block["mean_retrieval_jaccard"], marker="o", label=method)
    ax.set_xlabel("Final retrieval budget k")
    ax.set_ylabel("Mean pairwise Jaccard of retrieved passage IDs")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval-set stability across clustering seeds on the unchanged graph")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "retrieval_jaccard_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Questions: {len(questions)} | Passages: {len(passages)} | Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print("\nPairwise partition similarity across seeds:")
    print(partition_summary.to_string(index=False))
    print("\nPairwise retrieved-passage Jaccard across seeds (descriptive):")
    print(retrieval_summary.to_string(index=False))
    print("\nQuestion-level bootstrap intervals (questions are the independent units):")
    print(jaccard_ci_summary.to_string(index=False))
    print(f"\nSaved stability artifacts to {output_dir}")


if __name__ == "__main__":
    main()
