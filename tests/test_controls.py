import unittest

import numpy as np

from src import evaluate_controls as ec
from src import evaluate_graph_ranking as gr
from tests.test_graph_ranking import extractions, passages


class ControlTests(unittest.TestCase):
    def test_hop2_excludes_own_seed_query(self):
        h = {"a": {"a": 1.0, "b": 0.0, "c": 0.0}, "b": {"a": 0.0, "b": 1.0, "c": 0.5}}
        out = ec.hop2_scores(["a", "b", "c"], ["a", "b"], h)
        self.assertEqual(out["a"], 0.0)   # only seed b's query counts for a; a's self-match is ignored
        self.assertEqual(out["b"], 0.0)   # only seed a's query counts for b
        self.assertEqual(out["c"], 0.25)  # mean of both seeds' scores for a non-seed

    def test_hop2_query_contains_question_and_seed_text(self):
        q = ec.hop2_query("Where?", {"title": "Paris", "text": "Capital of France."})
        self.assertTrue(q.startswith("Where?") and "Paris" in q and "Capital of France." in q)

    def test_strip_titles_removes_title_entity_nodes(self):
        ps = passages()
        with_title = gr.build_passage_entity_graph(["a", "b"], extractions(), ps, "q")
        without = gr.build_passage_entity_graph(["a", "b"], extractions(), ec.strip_titles(ps), "q")
        self.assertIn("e::alpha band", with_title)            # title of passage a
        ents = lambda g: {n for n in g if n.startswith("e::")}  # noqa: E731
        self.assertEqual(ents(without), {"e::alpha band", "e::john smith", "e::paris"})  # NER entities only
        self.assertEqual(ents(with_title) - ents(without), set())  # titles here coincide with entities...
        ps2 = ec.strip_titles({"x": {"passage_id": "x", "title": "Only Title", "text": "zzz"}})
        g = gr.build_passage_entity_graph(["x"], {"x": {"entities": []}}, ps2, "q")
        self.assertEqual({n for n in g if n.startswith("e::")}, set())
        g2 = gr.build_passage_entity_graph(["x"], {"x": {"entities": []}}, {"x": {"passage_id": "x", "title": "Only Title", "text": "z"}}, "q")
        self.assertEqual({n for n in g2 if n.startswith("e::")}, {"e::only title"})

    def test_control_rows_are_permutations(self):
        ps = passages()
        rng = np.random.default_rng(1)
        emb = {pid: (v := np.abs(rng.normal(size=8)).astype(np.float32)) / np.linalg.norm(v) for pid in ps}
        q = {"question_id": "q", "question": "Where was the Alpha Band founder born?", "type": "bridge",
             "context_passage_ids": list(ps), "gold_supporting_passage_ids": ["a", "b"]}
        probe, _ = ec.evaluate_question(q, ps, ec.strip_titles(ps), extractions(), emb, emb["a"],
                                        {f"q::hop2::{s}": emb[s] for s in ps})
        self.assertEqual([r["config"] for r in probe], ec.CONTROL_CONFIGS)
        for r in probe:
            self.assertEqual(sorted(r["ranked_passage_ids"]), sorted(ps))


if __name__ == "__main__":
    unittest.main()
