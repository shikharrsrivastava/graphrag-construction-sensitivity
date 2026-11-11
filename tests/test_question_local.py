"""Correctness checks for Stage A context-scoped retrieval and statistics."""
import unittest

from src.evaluate_question_local import (
    bootstrap_ci,
    filter_graph_candidates,
    holm_adjust,
    metrics_for,
    paired_sign_flip_p,
    validate_inputs,
)


class QuestionLocalTests(unittest.TestCase):
    def setUp(self):
        self.q = {
            "question_id": "q1", "question": "which person?", "type": "bridge", "level": "hard",
            "context_passage_ids": ["p1", "p2", "p3"], "gold_supporting_passage_ids": ["p1", "p3"],
        }
        self.passages = [{"passage_id": p, "title": p, "text": p} for p in ("p1", "p2", "p3", "outside")]
        self.contexts = {"q1": {"context_passage_ids": ["p1", "p2", "p3"], "gold_supporting_passage_ids": ["p1", "p3"]}}

    def test_context_and_evidence_ids_are_validated(self):
        validate_inputs([self.q], self.passages, self.contexts)
        bad = dict(self.q, context_passage_ids=["p1", "outside"])
        with self.assertRaises(ValueError):
            validate_inputs([bad], self.passages, self.contexts)

    def test_evidence_matching_is_by_exact_passage_id(self):
        row = metrics_for(self.q, ["p2", "p1", "p3"], "question_local", 0.01)
        self.assertEqual(row["recall_at_2"], 0.5)
        self.assertEqual(row["recall_at_5"], 1.0)
        self.assertEqual(row["complete_support_at_2"], 0)
        self.assertEqual(row["complete_support_at_5"], 1)
        self.assertEqual(row["candidate_generation_recall"], 1.0)

    def test_candidate_generation_recall_uses_generated_set(self):
        row = metrics_for(self.q, ["p1"], "shared_global_graph", 0.01, {"p1", "p2"})
        self.assertEqual(row["candidate_generation_recall"], 0.5)
        self.assertEqual(row["candidate_count"], 2)

    def test_graph_candidates_are_filtered_to_question_contexts(self):
        candidates = filter_graph_candidates(["p1", "p2"], ["p2", "outside"], ["p1", "outside"])
        self.assertEqual(candidates, {"p1", "p2"})

    def test_question_level_bootstrap_and_paired_test(self):
        mean, lo, hi = bootstrap_ci([0.0, 1.0, 1.0], seed=7, n_boot=1000)
        self.assertAlmostEqual(mean, 2 / 3)
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)
        delta, p = paired_sign_flip_p([1, 1, 0], [0, 0, 0], seed=7, n_perm=2000)
        self.assertAlmostEqual(delta, 2 / 3)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self):
        adjusted = holm_adjust([0.04, 0.01, 0.03])
        self.assertAlmostEqual(adjusted[1], 0.03)
        self.assertGreaterEqual(adjusted[0], adjusted[1])
        self.assertGreaterEqual(adjusted[2], adjusted[1])


if __name__ == "__main__":
    unittest.main()
