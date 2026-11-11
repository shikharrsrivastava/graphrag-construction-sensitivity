"""Correctness and leakage guards for question-local dense/graph pilot logic."""
import unittest

import networkx as nx
import numpy as np

from src.evaluate_dense_graph import (
    MODEL_ID,
    MODEL_REVISION,
    bootstrap_mean_ci,
    build_chunk_similarity_graph,
    build_entity_graph_local,
    chunk_graph_candidates,
    holm_adjust,
    local_graph_candidates,
    paired_comparisons,
    retrieval_metrics,
    select_pilot_questions,
    sign_flip_test,
)


class DenseGraphPilotTests(unittest.TestCase):
    def setUp(self):
        self.contexts = ["p1", "p2"]
        self.extractions = {
            "p1": {"sentences": [["alice", "bob"], ["alice", "bob"]], "entities": ["alice", "bob"]},
            "p2": {"sentences": [["alice", "bob", "carol"]], "entities": ["alice", "bob", "carol"]},
            "foreign": {"sentences": [["outsider", "secret"]], "entities": ["outsider", "secret"]},
        }

    def test_dense_model_revision_is_pinned_to_validated_run(self):
        self.assertEqual(MODEL_ID, "BAAI/bge-small-en-v1.5")
        self.assertEqual(MODEL_REVISION, "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")

    def test_stratified_pilot_selection_is_reproducible_and_primary_only(self):
        qs = []
        for i in range(20):
            qs.append({"question_id": f"b{i}", "type": "bridge", "context_passage_ids": [f"p{i}_{j}" for j in range(10)]})
        for i in range(10):
            qs.append({"question_id": f"c{i}", "type": "comparison", "context_passage_ids": [f"c{i}_{j}" for j in range(10)]})
        qs.append({"question_id": "variable", "type": "bridge", "context_passage_ids": ["v1", "v2"]})
        a = select_pilot_questions(qs, seed=13, n=10)
        b = select_pilot_questions(qs, seed=13, n=10)
        self.assertEqual([q["question_id"] for q in a], [q["question_id"] for q in b])
        self.assertEqual(len(a), 10)
        self.assertEqual(sum(q["type"] == "bridge" for q in a), 7)
        self.assertEqual(sum(q["type"] == "comparison" for q in a), 3)
        self.assertNotIn("variable", {q["question_id"] for q in a})

    def test_entity_graph_edges_are_question_local_and_window_changes_frequency(self):
        sentence, local, _ = build_entity_graph_local(self.contexts, self.extractions, "q1", "sentence", "frequency")
        passage, _, _ = build_entity_graph_local(self.contexts, self.extractions, "q1", "passage", "frequency")
        self.assertEqual(sentence["alice"]["bob"]["weight"], 3.0)
        self.assertEqual(passage["alice"]["bob"]["weight"], 2.0)
        self.assertEqual(set(sentence.nodes), {"alice", "bob", "carol"})
        self.assertNotIn("secret", sentence)
        self.assertNotIn("foreign", set().union(*local.values()))
        self.assertEqual(set(json_contexts(sentence)), {"p1", "p2"})

    def test_binary_weights_and_top_half_sparsification_are_controlled(self):
        binary, _, _ = build_entity_graph_local(self.contexts, self.extractions, "q1", "sentence", "binary")
        sparse, _, details = build_entity_graph_local(self.contexts, self.extractions, "q1", "sentence", "frequency", .5)
        self.assertEqual(binary["alice"]["bob"]["weight"], 1.0)
        self.assertEqual(binary["alice"]["bob"]["frequency"], 3)
        self.assertEqual(sparse.number_of_edges(), 2)
        self.assertEqual(details, {"original_edge_count": 3, "removed_edge_count": 1})
        self.assertTrue(sparse.has_edge("alice", "bob"))

    def test_sparsification_ties_do_not_depend_on_context_or_edge_iteration_order(self):
        first, _, _ = build_entity_graph_local(["p1", "p2"], self.extractions, "q1", "sentence", "frequency", .5)
        second, _, _ = build_entity_graph_local(["p2", "p1"], self.extractions, "q1", "sentence", "frequency", .5)
        canonical = lambda graph: {frozenset((u, v)) for u, v in graph.edges}
        self.assertEqual(canonical(first), canonical(second))

    def test_weighted_neighbor_expansion_and_context_guard(self):
        graph = nx.Graph(construction_scope="question_contexts_only")
        graph.add_edge("alice", "bob", weight=1.0)
        graph.add_edge("alice", "carol", weight=4.0)
        entity_to_passages = {"alice": {"p1"}, "bob": {"p2"}, "carol": {"p3"}}
        candidates = local_graph_candidates(["p1", "p2", "p3"], ["p1"], graph, entity_to_passages, neighbor_k=1)
        self.assertEqual(candidates, {"p1", "p3"})
        with self.assertRaises(AssertionError):
            local_graph_candidates(["p1", "p2"], ["p1"], graph, entity_to_passages, neighbor_k=2)

    def test_chunk_graph_nodes_and_candidates_are_question_scoped(self):
        embeddings = {"p1": np.array([1.0, 0.0]), "p2": np.array([0.9, 0.1]), "foreign": np.array([0.0, 1.0])}
        graph = build_chunk_similarity_graph(["p1", "p2"], embeddings, "q1", k_neighbors=1)
        self.assertEqual(set(graph.nodes), {"chunk::p1", "chunk::p2"})
        self.assertEqual(set(chunk_graph_candidates(["p1", "p2"], ["p1"], graph)), {"p1", "p2"})
        self.assertTrue(all("foreign" not in node for node in graph.nodes))

    def test_evidence_metrics_use_exact_ids_and_support_denominator(self):
        q = {"question_id": "q", "question_type": "bridge", "type": "bridge", "level": "hard",
             "context_passage_ids": ["p1", "p2", "p3"], "gold_supporting_passage_ids": ["p1", "p3"]}
        row = retrieval_metrics(q, "test", ["p1", "p2", "p3"], ["p2", "p1", "p3"], None, {})
        self.assertEqual(row["recall_at_2"], .5)
        self.assertEqual(row["recall_at_5"], 1.0)
        self.assertEqual(row["complete_support_at_2"], 0)
        self.assertEqual(row["complete_support_at_5"], 1)
        self.assertEqual(row["candidate_generation_recall"], 1.0)

    def test_question_level_bootstrap_sign_flip_and_holm(self):
        mean, low, high = bootstrap_mean_ci([0.0, 1.0, 1.0], seed=3, n_boot=1000)
        self.assertAlmostEqual(mean, 2 / 3)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)
        delta, p = sign_flip_test([1.0, 1.0, 0.0], [0.0, 0.0, 0.0], seed=3, n_perm=2000)
        self.assertAlmostEqual(delta, 2 / 3)
        self.assertTrue(0 <= p <= 1)
        adjusted = holm_adjust([.04, .01, .03])
        self.assertAlmostEqual(adjusted[1], .03)
        self.assertGreaterEqual(adjusted[0], adjusted[1])

    def test_paired_comparisons_include_direct_graph_construction_contrasts(self):
        rows = []
        for qid in ("q1", "q2"):
            for config in ("vanilla_tfidf", "vanilla_bge", "entity_sentence_frequency_weighted",
                           "entity_passage_frequency_weighted", "entity_sentence_binary",
                           "entity_sentence_top50_sparsified", "chunk_bge_knn2"):
                rows.append({"question_id": qid, "config": config, "candidate_generation_recall": 1.0,
                             **{f"recall_at_{k}": 1.0 for k in (2, 5, 10)},
                             **{f"complete_support_at_{k}": 1.0 for k in (2, 5, 10)}})
        results = paired_comparisons(rows, seed=1)
        names = {row["comparison"] for row in results}
        self.assertIn("entity_passage_frequency_weighted - entity_sentence_frequency_weighted", names)
        self.assertIn("entity_sentence_binary - entity_sentence_frequency_weighted", names)
        self.assertIn("entity_sentence_top50_sparsified - entity_sentence_frequency_weighted", names)
        self.assertIn("chunk_bge_knn2 - entity_sentence_frequency_weighted", names)
        self.assertEqual(len(results), 69)


def json_contexts(graph):
    import json
    return json.loads(graph.graph["context_passage_ids"])


if __name__ == "__main__":
    unittest.main()
