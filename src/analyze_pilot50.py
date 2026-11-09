"""Reanalyse the saved 50-question HotpotQA pilot at the question level.

The raw file contains repeated clustering/perturbation seeds for each question.
This script first averages those repeated runs within each question, then computes
paired bootstrap confidence intervals across independent questions. It avoids
reporting seed x question rows as independent questions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.common import ROOT, read_json, write_json

CONDITIONS = ["original", "random_5pct", "critical_edge_betweenness_5pct"]
METHODS = ["tfidf_baseline", "one_hop_graph", "leiden_community", "louvain_community"]
GRAPH_METHODS = ["one_hop_graph", "leiden_community", "louvain_community"]


def summarize_dataset_and_graph(dataset_dir: Path) -> dict[str, Any]:
    questions = read_json(dataset_dir / "questions.json")
    passages = read_json(dataset_dir / "passages.json")
    manifest = read_json(dataset_dir / "dataset_manifest.json")
    graph = nx.Graph(nx.read_graphml(dataset_dir / "entity_graph.graphml"))

    degrees = [degree for _, degree in graph.degree()]
    weights = [float(data.get("weight", 1.0)) for _, _, data in graph.edges(data=True)]
    components = list(nx.connected_components(graph))
    largest_component = max((len(component) for component in components), default=0)
    level_counts = pd.Series([str(row.get("level", "unknown")) for row in questions]).value_counts().to_dict()
    type_counts = pd.Series([str(row.get("type", "unknown")) for row in questions]).value_counts().to_dict()
    gold_count_dist = pd.Series([len(row.get("gold_supporting_passage_ids", [])) for row in questions]).value_counts().to_dict()

    return {
        "manifest": manifest,
        "question_count": len(questions),
        "unique_question_ids": len({str(row["question_id"]) for row in questions}),
        "passage_count": len(passages),
        "difficulty_distribution": level_counts,
        "question_type_distribution": type_counts,
        "gold_support_count_distribution": {str(k): int(v) for k, v in gold_count_dist.items()},
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
            "density": float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0,
            "components": len(components),
            "largest_component_nodes": largest_component,
            "largest_component_fraction": largest_component / max(graph.number_of_nodes(), 1),
            "isolates": sum(degree == 0 for degree in degrees),
            "degree_le_2": sum(degree <= 2 for degree in degrees),
            "edge_weight_min": min(weights) if weights else None,
            "edge_weight_median": float(np.median(weights)) if weights else None,
            "edge_weight_mean": float(np.mean(weights)) if weights else None,
            "edge_weight_max": max(weights) if weights else None,
            "construction_rule": graph.graph.get("construction_rule", "not recorded"),
        },
    }


def question_level_frame(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "question_id", "condition", "method", "k", "perturbation_seed",
        "supporting_passage_recall_at_k", "candidate_pool_size",
        "candidate_generation_recall", "all_gold_supports_retrieved_at_k",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Raw CSV is missing columns: {missing}")

    return raw.groupby(
        ["condition", "method", "k", "question_id"], as_index=False
    ).agg(
        recall_at_k=("supporting_passage_recall_at_k", "mean"),
        candidate_pool_size=("candidate_pool_size", "mean"),
        candidate_generation_recall=("candidate_generation_recall", "mean"),
        all_gold_retrieved_rate=("all_gold_supports_retrieved_at_k", "mean"),
        seed_replicates=("perturbation_seed", "nunique"),
    )


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[indices].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return float(low), float(high)


def paired_bootstrap(question_level: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for condition in CONDITIONS:
        for k in sorted(question_level["k"].unique()):
            block = question_level[
                (question_level["condition"] == condition) & (question_level["k"] == k)
            ]
            pivot = block.pivot(index="question_id", columns="method", values="recall_at_k")
            if "tfidf_baseline" not in pivot.columns:
                continue
            for method in GRAPH_METHODS:
                if method not in pivot.columns:
                    continue
                paired = pivot[["tfidf_baseline", method]].dropna()
                difference = (paired[method] - paired["tfidf_baseline"]).to_numpy(dtype=float)
                low, high = bootstrap_mean_ci(difference, rng, n_boot)
                rows.append({
                    "condition": condition,
                    "k": int(k),
                    "comparison": f"{method} - tfidf_baseline",
                    "n_independent_questions": int(len(difference)),
                    "mean_paired_recall_difference": float(difference.mean()),
                    "ci95_percentile_low": low,
                    "ci95_percentile_high": high,
                    "questions_improved": int((difference > 0).sum()),
                    "questions_worsened": int((difference < 0).sum()),
                    "questions_tied": int((difference == 0).sum()),
                    "interpretation": "CI includes zero" if low <= 0 <= high else "CI excludes zero; exploratory, unadjusted",
                })
    return pd.DataFrame(rows)


def plot_retrieval(question_level: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = {
        "tfidf_baseline": "TF-IDF full-corpus",
        "one_hop_graph": "One-hop graph",
        "leiden_community": "Leiden community",
        "louvain_community": "Louvain community",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, k in zip(axes, (5, 10)):
        block = question_level[question_level["k"] == k]
        x = np.arange(len(CONDITIONS))
        for method in METHODS:
            means = []
            lows = []
            highs = []
            for condition in CONDITIONS:
                values = block[(block["condition"] == condition) & (block["method"] == method)]["recall_at_k"].to_numpy()
                if len(values):
                    means.append(float(values.mean()))
                    lows.append(float(values.mean() - values.std(ddof=1) / np.sqrt(len(values))))
                    highs.append(float(values.mean() + values.std(ddof=1) / np.sqrt(len(values))))
                else:
                    means.append(np.nan); lows.append(np.nan); highs.append(np.nan)
            axis.plot(x, means, marker="o", label=labels[method])
        axis.set_xticks(x, ["Original", "Random 5%", "Critical 5%"], rotation=15)
        axis.set_title(f"Supporting-passage Recall@{k}")
        axis.set_ylabel("Mean recall across 50 questions")
        axis.set_ylim(0, 1.05)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("50-question pilot — TF-IDF-seeded graph candidate filtering")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_structure(structural: pd.DataFrame, output_path: Path) -> None:
    summary = structural.groupby(["condition", "method"], as_index=False).agg(
        ari_mean=("ari_vs_same_seed_baseline", "mean"),
        ari_std=("ari_vs_same_seed_baseline", "std"),
        lcc_mean=("lcc_fraction", "mean"),
    )
    conditions = ["original", "random_5pct", "critical_edge_betweenness_5pct"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method in ["leiden", "louvain"]:
        block = summary[summary["method"] == method].set_index("condition").reindex(conditions)
        x = np.arange(len(conditions))
        axes[0].plot(x, block["ari_mean"], marker="o", label=method)
        axes[1].plot(x, block["lcc_mean"], marker="o", label=method)
    for axis in axes:
        axis.set_xticks(np.arange(len(conditions)), ["Original*", "Random 5%", "Critical 5%"], rotation=15)
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("ARI vs. same-seed original partition")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Partition retention under perturbation")
    axes[1].set_ylabel("Largest component / original nodes")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Graph connectivity")
    fig.suptitle("Structural response to edge removal (*original ARI = 1 by construction)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/pilot_50")
    parser.add_argument("--input-dir", default="results/pilot_50/perturbed_retrieval")
    parser.add_argument("--output-dir", default="results/pilot_50/audit")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path

    dataset_dir, input_dir, output_dir = map(resolve, (args.dataset_dir, args.input_dir, args.output_dir))
    raw = pd.read_csv(input_dir / "perturbed_retrieval_raw.csv")
    structural = pd.read_csv(input_dir / "perturbed_structural_raw.csv")
    dataset_graph = summarize_dataset_and_graph(dataset_dir)
    qlevel = question_level_frame(raw)
    bootstrap = paired_bootstrap(qlevel, args.bootstrap, args.seed)
    retrieval_summary = qlevel.groupby(["condition", "method", "k"], as_index=False).agg(
        n_questions=("question_id", "nunique"),
        recall_at_k_mean=("recall_at_k", "mean"),
        recall_at_k_sd=("recall_at_k", "std"),
        candidate_pool_mean=("candidate_pool_size", "mean"),
        candidate_generation_recall_mean=("candidate_generation_recall", "mean"),
        all_gold_retrieved_rate=("all_gold_retrieved_rate", "mean"),
    )
    structural_summary = structural.groupby(["condition", "method"], as_index=False).agg(
        seed_runs=("seed", "nunique"),
        ari_same_seed_mean=("ari_vs_same_seed_baseline", "mean"),
        ari_same_seed_sd=("ari_vs_same_seed_baseline", "std"),
        lcc_fraction_mean=("lcc_fraction", "mean"),
        community_count_mean=("community_count", "mean"),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    qlevel.to_csv(output_dir / "question_level_metrics.csv", index=False)
    retrieval_summary.to_csv(output_dir / "retrieval_question_level_summary.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_bootstrap_vs_tfidf.csv", index=False)
    structural_summary.to_csv(output_dir / "structural_summary.csv", index=False)
    write_json(dataset_graph, output_dir / "dataset_graph_audit.json")
    plot_retrieval(qlevel, output_dir / "retrieval_summary.png")
    plot_structure(structural, output_dir / "structural_summary.png")

    print(json.dumps(dataset_graph, indent=2))
    print("\nQuestion-level retrieval summary:")
    print(retrieval_summary.to_string(index=False))
    print("\nPaired bootstrap comparisons vs. TF-IDF:")
    print(bootstrap.to_string(index=False))
    print(f"\nWrote audit outputs to {output_dir}")


if __name__ == "__main__":
    main()
