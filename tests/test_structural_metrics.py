"""Small unit tests for graph perturbation helpers (run with `python -m unittest`)."""
import unittest

import networkx as nx

from src.perturb_graph import n_edges_to_remove, perturb_random, perturb_critical, structural_metrics
from src.cluster_graph import partition_graph


class StructuralMetricsTests(unittest.TestCase):
    def test_remove_count_is_at_least_one_when_fraction_positive(self):
        self.assertEqual(n_edges_to_remove(10, 0.05), 1)
        self.assertEqual(n_edges_to_remove(0, 0.05), 0)

    def test_random_perturbation_is_reproducible(self):
        graph = nx.cycle_graph(10, create_using=nx.Graph)
        graph = nx.relabel_nodes(graph, {node: str(node) for node in graph.nodes})
        g1, r1 = perturb_random(graph, 0.2, seed=12)
        g2, r2 = perturb_random(graph, 0.2, seed=12)
        self.assertEqual(r1, r2)
        self.assertEqual(g1.number_of_edges(), g2.number_of_edges())

    def test_critical_perturbation_removes_expected_number(self):
        graph = nx.path_graph(10)
        graph = nx.relabel_nodes(graph, {node: str(node) for node in graph.nodes})
        perturbed, removed = perturb_critical(graph, 0.2, seed=7)
        self.assertEqual(len(removed), 2)
        self.assertEqual(perturbed.number_of_edges(), graph.number_of_edges() - 2)

    def test_lcc_fraction_uses_original_node_denominator(self):
        graph = nx.Graph([("a", "b"), ("b", "c")])
        graph.add_nodes_from(["d", "e"])
        communities = partition_graph(graph, "louvain", seed=42)
        metrics = structural_metrics(graph, communities, original_node_count=5)
        self.assertAlmostEqual(metrics["lcc_fraction"], 3 / 5)


if __name__ == "__main__":
    unittest.main()
