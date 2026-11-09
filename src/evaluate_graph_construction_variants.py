"""Compare sentence-level and passage-level entity co-occurrence graphs.

This reuses the *same* saved spaCy entity extraction and passage mappings, so
only edge granularity changes. It is a controlled pilot ablation, not a full
semantic GraphRAG or dense-retrieval comparison.
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

from src.cluster_graph import partition_graph
from src.common import ROOT, read_json, write_json
from src.evaluate_retrieval import evaluate
from src.evaluate_seed_stability import bootstrap_question_level_summary, compare_retrieval_sets, community_labels


def build_passage_cooccurrence_graph(
    sentence_graph: nx.Graph,
    passage_entities: dict[str, list[str]],
) -> nx.Graph:
    """Connect entities appearing anywhere in the same passage.

    Node IDs and node attributes are copied from the sentence-level baseline.
    Edge weight is the number of passages in which an entity pair co-occurs.
    """
    graph = nx.Graph()
    graph.add_nodes_from((str(node), dict(attrs)) for node, attrs in sentence_graph.nodes(data=True))
    for passage_id, entities in passage_entities.items():
        present = sorted(set(map(str, entities)) & set(graph.nodes))
        for left, right in combinations(present, 2):
            if graph.has_edge(left, right):
                graph[left][right]["weight"] += 1
            else:
                graph.add_edge(left, right, weight=1)
    graph.graph.update(sentence_graph.graph)
    graph.graph["construction_rule"] = (
        "NER entities as nodes; entities appearing anywhere in the same passage are connected; "
        "edge weight is the number of shared passages. Entity extraction is reused unchanged."
    )
    graph.graph["edge_granularity"] = "same_passage"
    return graph


def graph_stats(graph: nx.Graph, name: str) -> dict[str, Any]:
    degree = [d for _, d in graph.degree()]
    components = list(nx.connected_components(graph))
    weights = [float(d.get("weight", 1.0)) for _, _, d in graph.edges(data=True)]
    return {
        "variant": name,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0,
        "mean_degree": float(np.mean(degree)) if degree else 0.0,
        "components": len(components),
        "largest_component_nodes": max(map(len, components), default=0),
        "largest_component_fraction": max(map(len, components), default=0) / max(graph.number_of_nodes(), 1),
        "isolates": sum(d == 0 for d in degree),
        "mean_edge_weight": float(np.mean(weights)) if weights else None,
        "max_degree": max(degree, default=0),
    }


def pairwise_partition_rows(partitions: dict[int, list[set[str]]], nodes: list[str], variant: str) -> list[dict[str, Any]]:
    rows = []
    for seed_a, seed_b in combinations(sorted(partitions), 2):
        a = community_labels(partitions[seed_a], nodes)
        b = community_labels(partitions[seed_b], nodes)
        rows.append({
            "variant": variant,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "ari": float(adjusted_rand_score(a, b)),
            "nmi": float(normalized_mutual_info_score(a, b)),
            "community_count_a": len(partitions[seed_a]),
            "community_count_b": len(partitions[seed_b]),
        })
    return rows


def bootstrap_paired_graph_difference(qlevel: pd.DataFrame, baseline: str, comparison: str, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (method, k), block in qlevel.groupby(["method", "k"], sort=True):
        pivot = block.pivot(index="question_id", columns="variant", values="recall_at_k")
        if baseline not in pivot or comparison not in pivot:
            continue
        diff = (pivot[comparison] - pivot[baseline]).dropna().to_numpy(dtype=float)
        if not len(diff):
            continue
        indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        boot = diff[indices].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append({
            "method": method,
            "k": int(k),
            "contrast": f"{comparison} - {baseline}",
            "n_independent_questions": int(len(diff)),
            "mean_paired_recall_difference": float(diff.mean()),
            "ci95_percentile_low": float(lo),
            "ci95_percentile_high": float(hi),
            "questions_improved": int((diff > 0).sum()),
            "questions_worsened": int((diff < 0).sum()),
            "questions_tied": int((diff == 0).sum()),
            "bootstrap_resamples": int(n_boot),
            "bootstrap_seed": int(seed),
        })
    return pd.DataFrame(rows)


def bootstrap_paired_jaccard_difference(
    question_level: pd.DataFrame,
    baseline: str,
    comparison: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Compare retrieval-set stability, resampling independent questions."""
    rng = np.random.default_rng(seed)
    rows = []
    for (method, k), block in question_level.groupby(["method", "k"], sort=True):
        pivot = block.pivot(index="question_id", columns="variant", values="question_mean_jaccard")
        if baseline not in pivot or comparison not in pivot:
            continue
        difference = (pivot[comparison] - pivot[baseline]).dropna().to_numpy(dtype=float)
        if not len(difference):
            continue
        indices = rng.integers(0, len(difference), size=(n_boot, len(difference)))
        boot = difference[indices].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append({
            "method": method,
            "k": int(k),
            "contrast": f"{comparison} - {baseline}",
            "n_independent_questions": int(len(difference)),
            "mean_paired_jaccard_difference": float(difference.mean()),
            "ci95_percentile_low": float(low),
            "ci95_percentile_high": float(high),
            "questions_more_stable": int((difference > 0).sum()),
            "questions_less_stable": int((difference < 0).sum()),
            "questions_tied": int((difference == 0).sum()),
            "bootstrap_resamples": int(n_boot),
            "bootstrap_seed": int(seed),
            "interpretation": "exploratory; unadjusted for multiple k comparisons",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/pilot_50")
    parser.add_argument("--output-dir", default="results/pilot_50/graph_construction_variants")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    parser.add_argument("--seed-k", type=int, default=2)
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error("provide at least two distinct seeds")

    dataset_dir = Path(args.dataset_dir).expanduser()
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = read_json(dataset_dir / "questions.json")
    passages = read_json(dataset_dir / "passages.json")
    entity_passages = read_json(dataset_dir / "entity_passages.json")
    passage_entities = read_json(dataset_dir / "passage_entities.json")
    sentence_graph = nx.Graph(nx.read_graphml(dataset_dir / "entity_graph.graphml"))
    passage_graph = build_passage_cooccurrence_graph(sentence_graph, passage_entities)

    node_order = sorted(map(str, sentence_graph.nodes))
    if set(node_order) != set(map(str, passage_graph.nodes)):
        raise ValueError("Graph variants must use exactly the same node set for paired partition comparison")

    graphs = {
        "same_sentence": sentence_graph,
        "same_passage": passage_graph,
    }
    base_edges = {frozenset(map(str, edge)) for edge in sentence_graph.edges}
    passage_edges = {frozenset(map(str, edge)) for edge in passage_graph.edges}
    stats = [graph_stats(graph, name) for name, graph in graphs.items()]
    edge_union = base_edges | passage_edges
    stats.append({
        "variant": "edge_comparison",
        "nodes": len(node_order),
        "edges": len(edge_union),
        "density": None,
        "mean_degree": None,
        "components": None,
        "largest_component_nodes": None,
        "largest_component_fraction": None,
        "isolates": None,
        "mean_edge_weight": None,
        "max_degree": None,
        "sentence_edges_also_in_passage_graph": len(base_edges & passage_edges),
        "edge_jaccard": len(base_edges & passage_edges) / len(edge_union) if edge_union else 1.0,
        "passage_only_edges": len(passage_edges - base_edges),
    })
    pd.DataFrame(stats).to_csv(output_dir / "graph_variant_statistics.csv", index=False)

    all_raw = []
    partition_pairs = []
    community_counts = []
    for variant, graph in graphs.items():
        partitions_by_seed: dict[int, list[set[str]]] = {}
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            communities = partition_graph(graph, "louvain", seed)
            partitions_by_seed[seed] = communities
            payload = {
                "graph_variant": variant,
                "graph_path": str(dataset_dir / "entity_graph.graphml") if variant == "same_sentence" else "derived from passage_entities.json",
                "method": "louvain",
                "seed": seed,
                "node_count": len(node_order),
                "community_count": len(communities),
                "communities": [
                    {"community_id": idx, "entities": sorted(group)}
                    for idx, group in enumerate(sorted(communities, key=lambda group: tuple(sorted(group))))
                ],
            }
            write_json(payload, variant_dir / f"louvain_seed{seed}.json")
            mapping = {
                entity: community_id
                for community_id, group in enumerate(communities)
                for entity in group
            }
            evaluated = evaluate(
                questions, passages, graph, entity_passages, passage_entities,
                {}, mapping, args.seed_k, args.k_values,
            )
            evaluated = evaluated[evaluated["method"].isin(["tfidf_baseline", "one_hop_graph", "louvain_community"])].copy()
            evaluated["candidate_generation_recall"] = evaluated.apply(
                lambda row: (
                    len([item for item in str(row["candidate_gold_passage_ids"]).split(";") if item])
                    / int(row["gold_support_count"])
                    if int(row["gold_support_count"]) > 0 else float("nan")
                ),
                axis=1,
            )
            evaluated["variant"] = variant
            evaluated["seed"] = seed
            all_raw.append(evaluated)
            community_counts.append({"variant": variant, "seed": seed, "community_count": len(communities)})
        partition_pairs.extend(pairwise_partition_rows(partitions_by_seed, node_order, variant))

    raw = pd.concat(all_raw, ignore_index=True)
    raw.to_csv(output_dir / "retrieval_raw.csv", index=False)
    partition_pair_df = pd.DataFrame(partition_pairs)
    partition_pair_df.to_csv(output_dir / "partition_similarity_pairwise.csv", index=False)
    pd.DataFrame(community_counts).to_csv(output_dir / "community_counts_by_seed.csv", index=False)

    retrieval_pairwise_frames = []
    retrieval_jaccard_q_frames = []
    retrieval_jaccard_ci_frames = []
    for variant, variant_raw in raw.groupby("variant", sort=True):
        pairwise = compare_retrieval_sets(variant_raw)
        pairwise["variant"] = variant
        retrieval_pairwise_frames.append(pairwise)
        question_block, ci_block = bootstrap_question_level_summary(
            pairwise, n_boot=args.bootstrap, seed=args.bootstrap_seed
        )
        question_block["variant"] = variant
        ci_block["variant"] = variant
        retrieval_jaccard_q_frames.append(question_block)
        retrieval_jaccard_ci_frames.append(ci_block)
    retrieval_pairwise = pd.concat(retrieval_pairwise_frames, ignore_index=True)
    retrieval_jaccard_q = pd.concat(retrieval_jaccard_q_frames, ignore_index=True)
    retrieval_jaccard_ci = pd.concat(retrieval_jaccard_ci_frames, ignore_index=True)
    jaccard_difference = bootstrap_paired_jaccard_difference(
        retrieval_jaccard_q, "same_sentence", "same_passage",
        args.bootstrap, args.bootstrap_seed,
    )
    retrieval_pairwise.to_csv(output_dir / "retrieval_jaccard_pairwise.csv", index=False)
    retrieval_jaccard_q.to_csv(output_dir / "retrieval_jaccard_question_level.csv", index=False)
    retrieval_jaccard_ci.to_csv(output_dir / "retrieval_jaccard_bootstrap_ci.csv", index=False)
    jaccard_difference.to_csv(output_dir / "paired_bootstrap_jaccard_passage_vs_sentence.csv", index=False)

    qlevel = raw.groupby(["variant", "method", "k", "question_id"], as_index=False).agg(
        recall_at_k=("supporting_passage_recall_at_k", "mean"),
        candidate_pool_size=("candidate_pool_size", "mean"),
        candidate_generation_recall=("candidate_generation_recall", "mean"),
        all_gold_retrieved=("all_gold_supports_retrieved_at_k", "mean"),
        n_seed_runs=("seed", "nunique"),
    )
    qlevel.to_csv(output_dir / "retrieval_question_level.csv", index=False)
    summary = qlevel.groupby(["variant", "method", "k"], as_index=False).agg(
        n_questions=("question_id", "nunique"),
        recall_at_k_mean=("recall_at_k", "mean"),
        recall_at_k_sd=("recall_at_k", "std"),
        candidate_pool_size_mean=("candidate_pool_size", "mean"),
        candidate_generation_recall_mean=("candidate_generation_recall", "mean"),
        all_gold_retrieved_rate=("all_gold_retrieved", "mean"),
    )
    summary.to_csv(output_dir / "retrieval_summary.csv", index=False)

    bootstrap = bootstrap_paired_graph_difference(qlevel, "same_sentence", "same_passage", args.bootstrap, args.bootstrap_seed)
    bootstrap.to_csv(output_dir / "paired_bootstrap_passage_vs_sentence.csv", index=False)
    partition_summary = partition_pair_df.groupby("variant", as_index=False).agg(
        seed_pairs=("ari", "count"), mean_pairwise_ari=("ari", "mean"),
        sd_pairwise_ari=("ari", "std"), mean_pairwise_nmi=("nmi", "mean"),
    )
    partition_summary.to_csv(output_dir / "partition_similarity_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, k in zip(axes, sorted(set(args.k_values))):
        block = summary[summary["k"] == k]
        methods = ["tfidf_baseline", "one_hop_graph", "louvain_community"]
        x = np.arange(len(methods))
        width = 0.34
        for idx, variant in enumerate(["same_sentence", "same_passage"]):
            vals = [float(block[(block["variant"] == variant) & (block["method"] == method)]["recall_at_k_mean"].iloc[0]) for method in methods]
            ax.bar(x + (idx - 0.5) * width, vals, width, label=variant.replace("_", " "))
        ax.set_xticks(x, methods, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Supporting-evidence Recall@{k}")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean question-level recall")
    axes[1].legend()
    fig.suptitle("Sentence-level vs passage-level entity co-occurrence (TF-IDF seed/rerank)")
    fig.tight_layout()
    fig.savefig(output_dir / "graph_construction_retrieval_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    stability = retrieval_jaccard_ci[
        retrieval_jaccard_ci["method"] == "louvain_community"
    ].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for variant in ["same_sentence", "same_passage"]:
        block = stability[stability["variant"] == variant].sort_values("k")
        if block.empty:
            continue
        x = block["k"].to_numpy(dtype=float)
        mean = block["mean_question_level_jaccard"].to_numpy(dtype=float)
        low = block["ci95_percentile_low"].to_numpy(dtype=float)
        high = block["ci95_percentile_high"].to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", label=variant.replace("_", " "))
        ax.fill_between(x, low, high, alpha=0.15)
    ax.set_xlabel("Final retrieval budget k")
    ax.set_ylabel("Mean pairwise Jaccard of retrieved passage IDs")
    ax.set_ylim(0, 1.05)
    ax.set_title("Louvain evidence-set stability across seeds by graph construction")
    ax.set_xticks(sorted(set(args.k_values)))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "retrieval_seed_stability_by_variant.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Graph statistics:")
    print(pd.DataFrame(stats).to_string(index=False))
    print("\nRetrieval summary:")
    print(summary.to_string(index=False))
    print("\nPaired question-level bootstrap: same-passage minus same-sentence")
    print(bootstrap.to_string(index=False))
    print("\nPartition similarity across 10 Louvain seeds:")
    print(partition_summary.to_string(index=False))
    print("\nRetrieved-passage Jaccard across seeds by graph construction:")
    print(retrieval_jaccard_ci.to_string(index=False))
    print("\nPaired bootstrap: same-passage minus same-sentence retrieved-set Jaccard:")
    print(jaccard_difference.to_string(index=False))
    print(f"\nSaved outputs under {output_dir}")


if __name__ == "__main__":
    main()
