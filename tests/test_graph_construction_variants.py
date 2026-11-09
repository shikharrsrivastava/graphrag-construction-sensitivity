"""Tests for controlled graph-construction ablations."""
import unittest

import networkx as nx

from src.evaluate_graph_construction_variants import build_passage_cooccurrence_graph


class PassageGraphTests(unittest.TestCase):
    def test_passage_graph_preserves_nodes_and_counts_shared_passages(self):
        sentence_graph = nx.Graph()
        sentence_graph.add_node("a", display_name="A")
        sentence_graph.add_node("b", display_name="B")
        sentence_graph.add_node("c", display_name="C")
        sentence_graph.add_edge("a", "b", weight=1)
        passage_entities = {
            "p1": ["a", "b", "c"],
            "p2": ["a", "b"],
        }

        passage_graph = build_passage_cooccurrence_graph(sentence_graph, passage_entities)

        self.assertEqual(set(passage_graph.nodes), set(sentence_graph.nodes))
        self.assertEqual(passage_graph["a"]["b"]["weight"], 2)
        self.assertEqual(passage_graph["a"]["c"]["weight"], 1)
        self.assertEqual(passage_graph["b"]["c"]["weight"], 1)
        self.assertEqual(passage_graph.nodes["a"]["display_name"], "A")
        self.assertEqual(passage_graph.graph["edge_granularity"], "same_passage")

    def test_passage_graph_ignores_unknown_entity_ids(self):
        sentence_graph = nx.Graph()
        sentence_graph.add_nodes_from(["a", "b"])
        passage_graph = build_passage_cooccurrence_graph(sentence_graph, {"p1": ["a", "unknown"]})
        self.assertEqual(passage_graph.number_of_nodes(), 2)
        self.assertEqual(passage_graph.number_of_edges(), 0)


if __name__ == "__main__":
    unittest.main()
