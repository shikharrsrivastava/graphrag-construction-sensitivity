"""Evaluate evidence retrieval on original and edge-perturbed graphs.

This is a TF-IDF retrieval pilot. It compares the candidate-generation effect of
one-hop, Leiden-community and Louvain-community retrieval before any answer
 generation. ARI is paired by clustering seed: each perturbed partition is
compared with an original-graph partition generated using the same seed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.cluster_graph import partition_graph
from src.common import ROOT, read_json
from src.evaluate_retrieval import evaluate
from src.perturb_graph import perturb_critical, perturb_random

METHODS = ("leiden", "louvain")
RETRIEVAL_METHODS = (
    "tfidf_baseline",
    "one_hop_graph",
    "leiden_community",
    "louvain_community",
)


def partition_to_map(communities: list[set[str]]) -> dict[str, int]:
    """Convert a community list into stable entity -> integer labels."""
    ordered = sorted(
        (set(map(str, community)) for community in communities),
        key=lambda group: tuple(sorted(group)),
    )
    return {
        node: community_id
        for community_id, community in enumerate(ordered)
        for node in community
    }


def partition_ari(
    original_communities: list[set[str]],
    current_communities: list[set[str]],
    node_order: list[str],
) -> float:
    """Compare partitions over the same fixed node universe."""
    original_map = partition_to_map(original_communities)
    current_map = partition_to_map(current_communities)
    original_labels = [original_map.get(node, -1) for node in node_order]
    current_labels = [current_map.get(node, -1) for node in node_order]
    return float(adjusted_rand_score(original_labels, current_labels))


def graph_structure(graph: nx.Graph, original_node_count: int) -> dict[str, Any]:
    """Return graph-level measurements that do not depend on clustering method."""
    if graph.number_of_nodes() == 0:
        lcc_fraction = float("nan")
    else:
        largest_component = max(
            (len(component) for component in nx.connected_components(graph)),
            default=0,
        )
        lcc_fraction = largest_component / max(original_node_count, 1)
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "component_count": nx.number_connected_components(graph),
        "lcc_fraction": lcc_fraction,
    }


def ids_to_set(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    value = str(value).strip()
    return set(item for item in value.split(";") if item)


def add_candidate_recall(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["candidate_generation_recall"] = frame.apply(
        lambda row: len(ids_to_set(row["candidate_gold_passage_ids"]))
        / max(len(ids_to_set(row["gold_supporting_passage_ids"])), 1),
        axis=1,
    )
    return frame


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conditions = ["original", "random_5pct", "critical_edge_betweenness_5pct"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

    for axis, k in zip(axes, (5, 10)):
        subset_k = summary[summary["k"] == k]
        for method in RETRIEVAL_METHODS:
            method_rows = subset_k[subset_k["method"] == method].set_index("condition")
            method_rows = method_rows.reindex(conditions)
            axis.plot(
                conditions,
                method_rows["retrieval_recall_mean"],
                marker="o",
                label=method,
            )
        axis.set_title(f"Supporting-passage Recall@{k}")
        axis.set_ylabel("Mean recall")
        axis.set_ylim(0, 1.05)
        axis.tick_params(axis="x", rotation=22)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    fig.suptitle("Evidence retrieval under graph-edge perturbations (TF-IDF pilot)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="data/processed/questions.json")
    parser.add_argument("--passages", default="data/processed/passages.json")
    parser.add_argument("--graph", default="data/processed/entity_graph.graphml")
    parser.add_argument("--entity-passages", default="data/processed/entity_passages.json")
    parser.add_argument("--passage-entities", default="data/processed/passage_entities.json")
    parser.add_argument("--output-dir", default="results/perturbed_retrieval")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--removal-fraction", type=float, default=0.05)
    parser.add_argument("--seed-k", type=int, default=2)
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10])
    args = parser.parse_args()

    if not 0 <= args.removal_fraction <= 1:
        parser.error("--removal-fraction must be between 0 and 1")
    if args.seed_k < 1 or any(k < 1 for k in args.k_values):
        parser.error("--seed-k and all --k-values must be positive integers")
    if not args.seeds:
        parser.error("provide at least one seed")

    questions = read_json(args.questions)
    passages = read_json(args.passages)
    entity_passages = read_json(args.entity_passages)
    passage_entities = read_json(args.passage_entities)

    graph_path = Path(args.graph).expanduser()
    if not graph_path.is_absolute():
        graph_path = ROOT / graph_path
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")
    original_graph = nx.Graph(nx.read_graphml(graph_path))
    node_order = sorted(map(str, original_graph.nodes))
    original_node_count = original_graph.number_of_nodes()

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_frames: list[pd.DataFrame] = []
    structural_rows: list[dict[str, Any]] = []

    random_condition = f"random_{int(round(args.removal_fraction * 100))}pct"
    critical_condition = f"critical_edge_betweenness_{int(round(args.removal_fraction * 100))}pct"

    for seed in args.seeds:
        # Paired baseline: each perturbed partition is compared with the
        # original-graph partition produced by the same algorithm seed.
        baseline_communities = {
            method: partition_graph(original_graph, method, seed)
            for method in METHODS
        }
        baseline_maps = {
            method: partition_to_map(baseline_communities[method])
            for method in METHODS
        }

        original_eval = evaluate(
            questions, passages, original_graph, entity_passages, passage_entities,
            baseline_maps["leiden"], baseline_maps["louvain"], args.seed_k, args.k_values,
        )
        original_eval["condition"] = "original"
        original_eval["perturbation_seed"] = seed
        original_eval["partition_ari_vs_same_seed_baseline"] = original_eval["method"].map(
            {"leiden_community": 1.0, "louvain_community": 1.0}
        )
        retrieval_frames.append(original_eval)

        original_struct = graph_structure(original_graph, original_node_count)
        for method in METHODS:
            structural_rows.append({
                "method": method,
                "condition": "original",
                "seed": seed,
                "removed_edges": 0,
                "removed_edge_fraction": 0.0,
                "community_count": len(baseline_communities[method]),
                "ari_vs_same_seed_baseline": 1.0,
                **original_struct,
            })

        perturbations = [
            (random_condition, *perturb_random(original_graph, args.removal_fraction, seed)),
            (critical_condition, *perturb_critical(original_graph, args.removal_fraction, seed)),
        ]

        for condition, perturbed_graph, removed_edges in perturbations:
            perturbed_communities = {
                method: partition_graph(perturbed_graph, method, seed)
                for method in METHODS
            }
            perturbed_maps = {
                method: partition_to_map(perturbed_communities[method])
                for method in METHODS
            }
            ari_by_method = {
                method: partition_ari(
                    baseline_communities[method],
                    perturbed_communities[method],
                    node_order,
                )
                for method in METHODS
            }

            run_eval = evaluate(
                questions, passages, perturbed_graph, entity_passages, passage_entities,
                perturbed_maps["leiden"], perturbed_maps["louvain"],
                args.seed_k, args.k_values,
            )
            run_eval["condition"] = condition
            run_eval["perturbation_seed"] = seed
            run_eval["partition_ari_vs_same_seed_baseline"] = run_eval["method"].map({
                "leiden_community": ari_by_method["leiden"],
                "louvain_community": ari_by_method["louvain"],
            })
            retrieval_frames.append(run_eval)

            structure = graph_structure(perturbed_graph, original_node_count)
            for method in METHODS:
                structural_rows.append({
                    "method": method,
                    "condition": condition,
                    "seed": seed,
                    "removed_edges": len(removed_edges),
                    "removed_edge_fraction": len(removed_edges) / max(original_graph.number_of_edges(), 1),
                    "community_count": len(perturbed_communities[method]),
                    "ari_vs_same_seed_baseline": ari_by_method[method],
                    **structure,
                })

    raw = add_candidate_recall(pd.concat(retrieval_frames, ignore_index=True))
    structural = pd.DataFrame(structural_rows)

    raw_path = output_dir / "perturbed_retrieval_raw.csv"
    structural_path = output_dir / "perturbed_structural_raw.csv"
    raw.to_csv(raw_path, index=False)
    structural.to_csv(structural_path, index=False)

    summary = raw.groupby(["condition", "method", "k"], as_index=False).agg(
        question_run_count=("question_id", "count"),
        unique_question_count=("question_id", "nunique"),
        replicate_count=("perturbation_seed", "nunique"),
        candidate_pool_size_mean=("candidate_pool_size", "mean"),
        candidate_generation_recall_mean=("candidate_generation_recall", "mean"),
        retrieval_recall_mean=("supporting_passage_recall_at_k", "mean"),
        retrieval_recall_std=("supporting_passage_recall_at_k", "std"),
        all_gold_supports_retrieved_rate=("all_gold_supports_retrieved_at_k", "mean"),
        partition_ari_mean=("partition_ari_vs_same_seed_baseline", "mean"),
    )
    summary_path = output_dir / "perturbed_retrieval_summary.csv"
    summary.to_csv(summary_path, index=False)

    structural_summary = structural.groupby(["condition", "method"], as_index=False).agg(
        replicate_count=("seed", "nunique"),
        ari_mean=("ari_vs_same_seed_baseline", "mean"),
        ari_std=("ari_vs_same_seed_baseline", "std"),
        lcc_fraction_mean=("lcc_fraction", "mean"),
        lcc_fraction_std=("lcc_fraction", "std"),
        community_count_mean=("community_count", "mean"),
        edge_count_mean=("edge_count", "mean"),
    )
    structural_summary_path = output_dir / "perturbed_structural_summary.csv"
    structural_summary.to_csv(structural_summary_path, index=False)

    figure_path = output_dir / "figures" / "perturbed_retrieval_recall.png"
    plot_summary(summary, figure_path)

    print(f"Questions: {len(questions)} | Passages: {len(passages)}")
    print(f"Original graph: {original_graph.number_of_nodes()} entities, {original_graph.number_of_edges()} edges")
    print(f"Seeds: {args.seeds} | Edge-removal fraction: {args.removal_fraction:.1%}")
    print(f"Saved retrieval rows:       {raw_path}")
    print(f"Saved structural rows:      {structural_path}")
    print(f"Saved retrieval summary:   {summary_path}")
    print(f"Saved structural summary:  {structural_summary_path}")
    print(f"Saved figure:              {figure_path}")
    print("\nRetrieval summary:")
    print(summary.to_string(index=False))
    print("\nStructural summary (ARI is paired to the same seed on the original graph):")
    print(structural_summary.to_string(index=False))
    print("\nGuardrail: TF-IDF retrieval only; this script does not generate answers or evaluate answer EM/F1.")
    print("Guardrail: graph perturbations remove entity co-occurrence edges; they do not edit source passages or gold labels.")


if __name__ == "__main__":
    main()
