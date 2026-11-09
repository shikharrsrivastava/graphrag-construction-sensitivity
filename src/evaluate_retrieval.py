"""Evaluate lexical and graph-assisted evidence retrieval on the HotpotQA pilot.

This is an initial retrieval pilot, not a complete GraphRAG answer-generation system.
It uses TF-IDF for the baseline, seed retrieval, and candidate reranking so the
candidate-generation strategy is the main difference between methods.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.common import ROOT, read_json


def load_partition(path: str | Path) -> dict[str, int]:
    payload = read_json(path)
    mapping = payload.get("entity_to_community", {})
    return {str(entity): int(community_id) for entity, community_id in mapping.items()}


def passage_text(passage: dict[str, Any]) -> str:
    title = str(passage.get("title", ""))
    text = str(passage.get("text", ""))
    return f"{title}. {text}".strip()


def rank_candidate_ids(
    candidate_ids: set[str],
    passage_index: dict[str, int],
    scores,
) -> list[str]:
    """Rank candidate passage IDs by the current question's TF-IDF cosine score."""
    valid_ids = [pid for pid in candidate_ids if pid in passage_index]
    return sorted(valid_ids, key=lambda pid: (-float(scores[passage_index[pid]]), pid))


def collect_passage_candidates(entities: set[str], entity_passages: dict[str, list[str]]) -> set[str]:
    candidates: set[str] = set()
    for entity in entities:
        candidates.update(map(str, entity_passages.get(entity, [])))
    return candidates


def graph_candidate_sets(
    seed_passage_ids: list[str],
    passage_entities: dict[str, list[str]],
    entity_passages: dict[str, list[str]],
    graph: nx.Graph,
    entity_to_community: dict[str, int],
) -> dict[str, set[str]]:
    """Create one-hop and community candidate pools from the same lexical seeds."""
    seed_ids = set(map(str, seed_passage_ids))
    seed_entities: set[str] = set()
    for passage_id in seed_ids:
        seed_entities.update(map(str, passage_entities.get(passage_id, [])))

    graph_nodes = set(map(str, graph.nodes))
    graph_seed_entities = seed_entities & graph_nodes

    # One-hop: retain seeds and add passages containing seed entities or direct neighbours.
    one_hop_entities = set(graph_seed_entities)
    for entity in graph_seed_entities:
        one_hop_entities.update(map(str, graph.neighbors(entity)))
    one_hop_candidates = seed_ids | collect_passage_candidates(one_hop_entities, entity_passages)

    candidates: dict[str, set[str]] = {"one_hop_graph": one_hop_candidates}

    # Community-based retrieval: use the communities containing the same seed entities.
    for method in ("leiden_community", "louvain_community"):
        # The actual entity_to_community map for each method is inserted by the caller.
        candidates[method] = set(seed_ids)

    return candidates


def community_candidate_ids(
    seed_passage_ids: list[str],
    passage_entities: dict[str, list[str]],
    entity_passages: dict[str, list[str]],
    entity_to_community: dict[str, int],
) -> set[str]:
    seed_ids = set(map(str, seed_passage_ids))
    seed_entities: set[str] = set()
    for passage_id in seed_ids:
        seed_entities.update(map(str, passage_entities.get(passage_id, [])))

    seed_communities = {
        entity_to_community[entity]
        for entity in seed_entities
        if entity in entity_to_community
    }
    community_entities = {
        entity for entity, community_id in entity_to_community.items()
        if community_id in seed_communities
    }
    return seed_ids | collect_passage_candidates(community_entities, entity_passages)


def evaluate(
    questions: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    graph: nx.Graph,
    entity_passages: dict[str, list[str]],
    passage_entities: dict[str, list[str]],
    leiden_map: dict[str, int],
    louvain_map: dict[str, int],
    seed_k: int,
    k_values: list[int],
) -> pd.DataFrame:
    passage_ids = [str(item["passage_id"]) for item in passages]
    passage_index = {pid: idx for idx, pid in enumerate(passage_ids)}
    corpus = [passage_text(item) for item in passages]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    passage_matrix = vectorizer.fit_transform(corpus)
    rows: list[dict[str, Any]] = []

    for question in questions:
        question_id = str(question["question_id"])
        query = str(question.get("question", ""))
        gold_ids = set(map(str, question.get("gold_supporting_passage_ids", [])))
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, passage_matrix).ravel()
        all_candidates = set(passage_ids)
        all_ranked = rank_candidate_ids(all_candidates, passage_index, scores)
        seed_ids = all_ranked[: min(seed_k, len(all_ranked))]

        candidate_pools: dict[str, set[str]] = {
            "tfidf_baseline": all_candidates,
            "one_hop_graph": all_candidates,
        }
        graph_candidates = graph_candidate_sets(
            seed_ids, passage_entities, entity_passages, graph, leiden_map
        )
        candidate_pools["one_hop_graph"] = graph_candidates["one_hop_graph"]
        candidate_pools["leiden_community"] = community_candidate_ids(
            seed_ids, passage_entities, entity_passages, leiden_map
        )
        candidate_pools["louvain_community"] = community_candidate_ids(
            seed_ids, passage_entities, entity_passages, louvain_map
        )

        for method, candidate_ids in candidate_pools.items():
            ranked = rank_candidate_ids(candidate_ids, passage_index, scores)
            for k in sorted(set(k_values)):
                retrieved = set(ranked[:k])
                hits = len(retrieved & gold_ids)
                recall = hits / len(gold_ids) if gold_ids else float("nan")
                rows.append({
                    "question_id": question_id,
                    "question_type": str(question.get("type", "")),
                    "difficulty": str(question.get("level", "")),
                    "method": method,
                    "k": k,
                    "candidate_pool_size": len(candidate_ids),
                    "gold_support_count": len(gold_ids),
                    "gold_supports_retrieved": hits,
                    "supporting_passage_recall_at_k": recall,
                    "all_gold_supports_retrieved_at_k": int(bool(gold_ids) and gold_ids.issubset(retrieved)),
                    "gold_supporting_passage_ids": ";".join(sorted(gold_ids)),
                    "candidate_gold_passage_ids": ";".join(sorted(candidate_ids & gold_ids)),
                    "retrieved_passage_ids": ";".join(ranked[:k]),
                    "retrieved_gold_passage_ids": ";".join(sorted(retrieved & gold_ids)),
                    "seed_k": seed_k,
                })
    return pd.DataFrame(rows)


def save_plot(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in summary["method"].drop_duplicates():
        subset = summary[summary["method"] == method].sort_values("k")
        ax.plot(subset["k"], subset["supporting_passage_recall_at_k_mean"], marker="o", label=method)
    ax.set_xlabel("Final retrieval budget k (passages)")
    ax.set_ylabel("Mean supporting-passage Recall@k")
    ax.set_title("Pilot evidence retrieval (TF-IDF reranking; not answer-quality evaluation)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="data/processed/questions.json")
    parser.add_argument("--passages", default="data/processed/passages.json")
    parser.add_argument("--graph", default="data/processed/entity_graph.graphml")
    parser.add_argument("--entity-passages", default="data/processed/entity_passages.json")
    parser.add_argument("--passage-entities", default="data/processed/passage_entities.json")
    parser.add_argument("--partition-dir", default="results/partitions")
    parser.add_argument("--seed-k", type=int, default=5, help="Number of initial TF-IDF seeds used by graph methods")
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--output-dir", default="results/retrieval")
    args = parser.parse_args()
    if args.seed_k < 1 or any(k < 1 for k in args.k_values):
        parser.error("--seed-k and all --k-values must be positive integers")

    questions = read_json(args.questions)
    passages = read_json(args.passages)
    entity_passages = read_json(args.entity_passages)
    passage_entities = read_json(args.passage_entities)
    graph_path = Path(args.graph)
    if not graph_path.is_absolute():
        graph_path = ROOT / graph_path
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}; build it first.")
    graph = nx.Graph(nx.read_graphml(graph_path))

    partition_dir = Path(args.partition_dir)
    if not partition_dir.is_absolute():
        partition_dir = ROOT / partition_dir
    leiden_path = partition_dir / "leiden_seed42.json"
    louvain_path = partition_dir / "louvain_seed42.json"
    if not leiden_path.exists() or not louvain_path.exists():
        raise FileNotFoundError(f"Expected both {leiden_path} and {louvain_path}; run src.cluster_graph first.")
    leiden_map = load_partition(leiden_path)
    louvain_map = load_partition(louvain_path)

    results = evaluate(
        questions, passages, graph, entity_passages, passage_entities,
        leiden_map, louvain_map, args.seed_k, args.k_values,
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "retrieval_raw.csv"
    summary_path = output_dir / "retrieval_summary.csv"
    plot_path = output_dir / "figures" / "retrieval_recall.png"
    results.to_csv(raw_path, index=False)
    summary = results.groupby(["method", "k"], as_index=False).agg(
        question_count=("question_id", "nunique"),
        candidate_pool_size_mean=("candidate_pool_size", "mean"),
        supporting_passage_recall_at_k_mean=("supporting_passage_recall_at_k", "mean"),
        supporting_passage_recall_at_k_std=("supporting_passage_recall_at_k", "std"),
        all_gold_supports_retrieved_rate=("all_gold_supports_retrieved_at_k", "mean"),
    )
    summary.to_csv(summary_path, index=False)
    save_plot(summary, plot_path)
    print(f"Questions: {len(questions)} | Passages: {len(passages)} | Graph: {graph.number_of_nodes()} entities, {graph.number_of_edges()} edges")
    print(f"Saved per-question results: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {plot_path}")
    print("Guardrail: this is a TF-IDF lexical retrieval pilot. Graph methods use the same TF-IDF seeds and reranking; dense embeddings and answer generation are not included.")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
