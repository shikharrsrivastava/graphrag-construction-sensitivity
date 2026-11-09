"""Perturb graph edges, re-cluster, and measure structural stability (not QA quality)."""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.cluster_graph import partition_graph
from src.common import ROOT, read_json


def normalized_edge(u: Any, v: Any) -> tuple[str, str]:
    a, b = str(u), str(v)
    return (a, b) if a <= b else (b, a)


def n_edges_to_remove(edge_count: int, fraction: float) -> int:
    return 0 if edge_count == 0 or fraction <= 0 else min(edge_count, max(1, math.ceil(edge_count * fraction)))


def perturb_random(graph: nx.Graph, fraction: float, seed: int) -> tuple[nx.Graph, list[tuple[str, str]]]:
    edges = sorted(normalized_edge(u, v) for u, v in graph.edges())
    count = n_edges_to_remove(len(edges), fraction)
    rng = random.Random(seed)
    removed = sorted(rng.sample(edges, count)) if count else []
    perturbed = graph.copy()
    perturbed.remove_edges_from(removed)
    return perturbed, removed


def perturb_critical(graph: nx.Graph, fraction: float, seed: int) -> tuple[nx.Graph, list[tuple[str, str]]]:
    """Remove highest edge-betweenness edges; use seed only to break cutoff ties."""
    edges = sorted(normalized_edge(u, v) for u, v in graph.edges())
    count = n_edges_to_remove(len(edges), fraction)
    if count == 0:
        return graph.copy(), []
    scores = nx.edge_betweenness_centrality(graph, weight=None)
    edge_scores = [(normalized_edge(u, v), float(score)) for (u, v), score in scores.items()]
    edge_scores.sort(key=lambda item: (-item[1], item[0]))
    if count >= len(edge_scores):
        removed = [edge for edge, _ in edge_scores]
    else:
        cutoff = edge_scores[count - 1][1]
        tied = sorted(edge for edge, score in edge_scores if math.isclose(score, cutoff, rel_tol=1e-12, abs_tol=1e-15))
        tied_set = set(tied)
        higher = [edge for edge, score in edge_scores if score > cutoff and edge not in tied_set]
        lower = [edge for edge, score in edge_scores if score < cutoff and edge not in tied_set]
        needed = max(0, count - len(higher))
        rng = random.Random(seed)
        chosen_ties = rng.sample(tied, min(needed, len(tied)))
        removed = higher + chosen_ties
        if len(removed) < count:
            removed.extend(lower[: count - len(removed)])
        removed = removed[:count]
    removed = sorted(removed)
    perturbed = graph.copy()
    perturbed.remove_edges_from(removed)
    return perturbed, removed


def community_labels(communities: list[set[str]], nodes: list[str]) -> list[int]:
    membership: dict[str, int] = {}
    for idx, community in enumerate(communities):
        for node in community:
            membership[str(node)] = idx
    return [membership.get(node, -1) for node in nodes]


def structural_metrics(graph: nx.Graph, communities: list[set[str]], original_node_count: int) -> dict[str, Any]:
    node_count = graph.number_of_nodes()
    if node_count == 0:
        lcc_fraction = float("nan")
    else:
        largest = max((len(component) for component in nx.connected_components(graph)), default=0)
        denominator = max(original_node_count, 1)
        lcc_fraction = largest / denominator

    non_singletons = [community for community in communities if len(community) > 1]
    if non_singletons:
        connected_count = sum(
            1 for community in non_singletons
            if nx.is_connected(graph.subgraph(community))
        )
        connected_fraction = connected_count / len(non_singletons)
    else:
        connected_fraction = float("nan")

    return {
        "node_count": node_count,
        "edge_count": graph.number_of_edges(),
        "community_count": len(communities),
        "lcc_fraction": lcc_fraction,
        "connected_non_singleton_community_fraction": connected_fraction,
    }


def build_result_row(
    graph: nx.Graph,
    original_graph: nx.Graph,
    method: str,
    condition: str,
    seed: int,
    baseline_labels: list[int],
    original_node_order: list[str],
    communities: list[set[str]],
    removed_edge_count: int,
) -> dict[str, Any]:
    current_labels = community_labels(communities, original_node_order)
    ari = adjusted_rand_score(baseline_labels, current_labels) if original_node_order else float("nan")
    metrics = structural_metrics(graph, communities, original_graph.number_of_nodes())
    return {
        "method": method,
        "condition": condition,
        "seed": int(seed),
        "ari_vs_original_partition": float(ari),
        "removed_edges": int(removed_edge_count),
        "removed_edge_fraction": removed_edge_count / original_graph.number_of_edges() if original_graph.number_of_edges() else 0.0,
        **metrics,
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "ari_vs_original_partition", "removed_edge_fraction", "community_count",
        "lcc_fraction", "connected_non_singleton_community_fraction", "edge_count",
    ]
    grouped = raw.groupby(["method", "condition"], dropna=False)[value_columns].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def save_plot(summary: pd.DataFrame, plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    conditions = list(dict.fromkeys(summary["condition"].tolist()))
    methods = list(dict.fromkeys(summary["method"].tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for method in methods:
        subset = summary[summary["method"] == method].set_index("condition").reindex(conditions)
        x = list(range(len(conditions)))
        axes[0].errorbar(
            x,
            subset["ari_vs_original_partition_mean"],
            yerr=subset["ari_vs_original_partition_std"].fillna(0),
            marker="o", capsize=3, label=method,
        )
        axes[1].errorbar(
            x,
            subset["lcc_fraction_mean"],
            yerr=subset["lcc_fraction_std"].fillna(0),
            marker="o", capsize=3, label=method,
        )
    for axis in axes:
        axis.set_xticks(range(len(conditions)), conditions, rotation=20, ha="right")
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[0].set_title("Community partition similarity")
    axes[0].set_ylabel("ARI vs. original partition")
    axes[0].set_ylim(-1.05, 1.05)
    axes[1].set_title("Graph connectivity")
    axes[1].set_ylabel("Largest connected component / original nodes")
    axes[1].set_ylim(0, 1.05)
    fig.suptitle("Structural robustness under edge removal (not answer-quality evaluation)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="data/processed/entity_graph.graphml")
    parser.add_argument("--partition-dir", default="results/partitions")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--baseline-seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--methods", nargs="+", choices=["leiden", "louvain"], default=["leiden", "louvain"])
    parser.add_argument("--removal-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.removal_fraction <= 1:
        parser.error("--removal-fraction must be between 0 and 1")

    graph_path = Path(args.graph)
    if not graph_path.is_absolute():
        graph_path = ROOT / graph_path
    partition_dir = Path(args.partition_dir)
    if not partition_dir.is_absolute():
        partition_dir = ROOT / partition_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}. Run src.build_graph first.")

    original_graph = nx.read_graphml(graph_path)
    original_graph = nx.Graph(original_graph)  # graphml may return a Graph/DiGraph type
    original_nodes = sorted(map(str, original_graph.nodes))
    rows: list[dict[str, Any]] = []

    for method in args.methods:
        partition_file = partition_dir / f"{method}_seed{args.baseline_seed}.json"
        if not partition_file.exists():
            raise FileNotFoundError(f"Missing baseline partition {partition_file}. Run src.cluster_graph first.")
        payload = read_json(partition_file)
        baseline_communities = [set(map(str, item["entities"])) for item in payload["communities"]]
        baseline_communities.sort(key=lambda group: tuple(sorted(group)))
        baseline_labels = community_labels(baseline_communities, original_nodes)

        # Original graph baseline: ARI is 1 by definition against its own partition.
        rows.append(build_result_row(
            original_graph, original_graph, method, "original", args.baseline_seed,
            baseline_labels, original_nodes, baseline_communities, 0,
        ))

        for seed in args.seeds:
            perturbations = [
                ("random_5pct", *perturb_random(original_graph, args.removal_fraction, seed)),
                ("critical_edge_betweenness", *perturb_critical(original_graph, args.removal_fraction, seed)),
            ]
            for condition, perturbed_graph, removed_edges in perturbations:
                communities = partition_graph(perturbed_graph, method, seed)
                rows.append(build_result_row(
                    perturbed_graph, original_graph, method, condition, seed,
                    baseline_labels, original_nodes, communities, len(removed_edges),
                ))

    raw = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "structural_robustness_raw.csv"
    summary_path = out_dir / "structural_robustness_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary = summarize(raw)
    summary.to_csv(summary_path, index=False)
    plot_path = out_dir / "figures" / "structural_robustness.png"
    save_plot(summary, plot_path)
    print(f"Saved raw results: {raw_path}")
    print(f"Saved summary:     {summary_path}")
    print(f"Saved figure:      {plot_path}")
    print("Interpretation guardrail: these outputs measure graph/partition structure only; retrieval and answer quality are not evaluated by this script.")


if __name__ == "__main__":
    main()
