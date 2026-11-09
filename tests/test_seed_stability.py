"""Tests for cross-seed partition and retrieval stability helpers."""
import unittest

from src.evaluate_seed_stability import compare_partitions, jaccard


class SeedStabilityTests(unittest.TestCase):
    def test_jaccard_for_partial_overlap(self):
        self.assertAlmostEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)

    def test_jaccard_for_empty_sets_is_one(self):
        self.assertEqual(jaccard(set(), set()), 1.0)

    def test_partition_similarity_is_label_invariant(self):
        partitions = {
            42: [{"a", "b"}, {"c", "d"}],
            43: [{"c", "d"}, {"a", "b"}],
        }
        rows = compare_partitions(partitions, ["a", "b", "c", "d"], "louvain")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["ari"], 1.0)
        self.assertAlmostEqual(rows[0]["nmi"], 1.0)


if __name__ == "__main__":
    unittest.main()
