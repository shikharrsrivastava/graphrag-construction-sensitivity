"""Run Leiden and/or Louvain on the same entity graph and save partitions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import networkx as nx

from src.common import ROOT, write_json


def partition_graph(graph: nx.Graph, method: str, seed: int = 42) -> list[set[str]]:
    """Return a deterministic ordering of communities for a given graph/method/seed."""
    nodes = sorted(str(node) for node in graph.nodes)
    if not nodes:
        return []
    if graph.number_of_edges() == 0:
        communities = [{node} for node in nodes]
        return communities

    if method == "louvain":
        groups = nx.algorithms.community.louvain_communities(graph, weight="weight", seed=seed)
        communities = [set(map(str, group)) for group in groups]
    elif method == "leiden":
        try:
            import igraph as ig
            import leidenalg
        except ImportError as exc:
            raise RuntimeError("Leiden requires `python-igraph` and `leidenalg`; install requirements.txt.") from exc

        index = {node: idx for idx, node in enumerate(nodes)}
        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        for left, right, data in graph.edges(data=True):
            left_key, right_key = str(left), str(right)
            edges.append((index[left_key], index[right_key]))
            weights.append(float(data.get("weight", 1.0)))
        ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
        ig_graph.vs["name"] = nodes
        if weights:
            ig_graph.es["weight"] = weights
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            weights="weight" if weights else None,
            seed=int(seed),
            n_iterations=2,
        )
        grouped: dict[int, set[str]] = {}
        for node, community_id in zip(nodes, partition.membership):
            grouped.setdefault(int(community_id), set()).add(node)
        communities = list(grouped.values())
    else:
        raise ValueError(f"Unsupported method: {method!r}; choose 'leiden' or 'louvain'.")

    communities = [set(group) for group in communities if group]
    communities.sort(key=lambda group: tuple(sorted(group)))
    # Preserve any nodes not represented by a partition implementation as singleton communities.
    covered = set().union(*communities) if communities else set()
    communities.extend({node} for node in nodes if node not in covered)
    communities.sort(key=lambda group: tuple(sorted(group)))
    return communities


def partition_payload(graph_path: str, method: str, seed: int, communities: Iterable[set[str]]) -> dict:
    ordered = sorted((sorted(map(str, group)) for group in communities), key=lambda group: tuple(group))
    entity_to_community = {
        node: community_id
        for community_id, nodes in enumerate(ordered)
        for node in nodes
    }
    return {
        "graph_path": graph_path,
        "method": method,
        "seed": int(seed),
        "node_count": len(entity_to_community),
        "community_count": len(ordered),
        "communities": [{"community_id": idx, "entities": nodes} for idx, nodes in enumerate(ordered)],
        "entity_to_community": entity_to_community,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="data/processed/entity_graph.graphml")
    parser.add_argument("--output-dir", default="results/partitions")
    parser.add_argument("--methods", nargs="+", choices=["leiden", "louvain"], default=["leiden", "louvain"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    graph_path = Path(args.graph)
    if not graph_path.is_absolute():
        graph_path = ROOT / graph_path
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}. Run src.build_graph first.")
    graph = nx.read_graphml(graph_path)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        communities = partition_graph(graph, method, args.seed)
        payload = partition_payload(str(graph_path), method, args.seed, communities)
        out_path = out_dir / f"{method}_seed{args.seed}.json"
        write_json(payload, out_path)
        print(f"{method}: {payload['community_count']} communities across {payload['node_count']} nodes -> {out_path}")


if __name__ == "__main__":
    main()
