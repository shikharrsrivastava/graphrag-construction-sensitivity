import unittest

import numpy as np

from src import evaluate_graph_ranking as gr
from src.evaluate_dense_graph import holm_adjust
from src.load_musique import convert_record, hop_count, is_eligible, sample_questions


def passages():
    return {
        "a": {"passage_id": "a", "title": "Alpha Band", "text": "Alpha Band was formed by John Smith."},
        "b": {"passage_id": "b", "title": "John Smith (musician)", "text": "John Smith was born in Paris."},
        "c": {"passage_id": "c", "title": "Paris", "text": "Paris is the capital of France."},
        "d": {"passage_id": "d", "title": "Unrelated", "text": "Bananas are yellow."},
    }


def extractions():
    return {"a": {"entities": ["alpha band", "john smith"]}, "b": {"entities": ["john smith", "paris"]},
            "c": {"entities": ["paris", "france"]}, "d": {"entities": []}}


class ScoringTests(unittest.TestCase):
    def test_minmax_constant_maps_to_zero(self):
        self.assertEqual(gr.minmax({"x": 3.0, "y": 3.0}), {"x": 0.0, "y": 0.0})
        self.assertEqual(gr.minmax({"x": 1.0, "y": 3.0}), {"x": 0.0, "y": 1.0})

    def test_combine_lambda_endpoints(self):
        bge, graph = {"x": 0.9, "y": 0.1}, {"x": 0.0, "y": 1.0}
        self.assertEqual(gr.combine(bge, graph, 0.0), {"x": 1.0, "y": 0.0})
        self.assertEqual(gr.combine(bge, graph, 1.0), {"x": 0.0, "y": 1.0})
        self.assertEqual(gr.combine(bge, graph, 0.5), {"x": 0.5, "y": 0.5})

    def test_rank_ties_break_by_bge_then_id(self):
        ranked = gr.rank_contexts(["b", "a", "c"], {"a": 1.0, "b": 1.0, "c": 0.0}, {"a": 0.1, "b": 0.2, "c": 0.0})
        self.assertEqual(ranked, ["b", "a", "c"])
        self.assertEqual(gr.rank_contexts(["b", "a"], {"a": 1.0, "b": 1.0}), ["a", "b"])

    def test_rank_requires_exact_context_coverage(self):
        with self.assertRaises(AssertionError):
            gr.rank_contexts(["a", "b"], {"a": 1.0})
        with self.assertRaises(AssertionError):
            gr.rank_contexts(["a"], {"a": 1.0, "z": 2.0})

    def test_bm25_ranks_lexical_match_first(self):
        scores = gr.bm25_scores("capital of France", list(passages()), passages())
        self.assertEqual(gr.rank_contexts(list(passages()), scores)[0], "c")

    def test_title_normalization_strips_disambiguator(self):
        self.assertEqual(gr.normalize_title("John Smith (musician)"), "john smith")
        self.assertEqual(gr.normalize_title("Green (album)"), "green")
        self.assertEqual(gr.normalize_title("Paris"), "paris")


class GraphTests(unittest.TestCase):
    def test_entity_graph_is_question_local_and_bipartite(self):
        g = gr.build_passage_entity_graph(["a", "b", "c"], extractions(), passages(), "q")
        self.assertEqual({d["passage_id"] for _, d in g.nodes(data=True) if d.get("kind") == "passage"}, {"a", "b", "c"})
        self.assertNotIn("p::d", g)
        for u, v in g.edges():
            self.assertNotEqual(u[:3], v[:3])  # passage-entity edges only
        self.assertIn("e::john smith", g)  # from title "John Smith (musician)" and from entity mentions
        self.assertEqual(g.graph["construction_scope"], "question_contexts_only")

    def test_ppr_boosts_passage_linked_to_seed(self):
        ctx = ["a", "b", "c", "d"]
        g = gr.build_passage_entity_graph(ctx, extractions(), passages(), "q")
        scores = gr.graph_passage_scores(g, {pid: f"p::{pid}" for pid in ctx}, ["a"])
        self.assertGreater(scores["b"], scores["c"])  # one shared entity away vs two
        self.assertGreater(scores["c"], scores["d"])
        self.assertAlmostEqual(scores["d"], 0.0, places=12)

    def test_evaluate_question_rows_are_permutations(self):
        ps = passages()
        rng = np.random.default_rng(0)
        emb = {pid: (v := np.abs(rng.normal(size=8)).astype(np.float32)) / np.linalg.norm(v) for pid in ps}
        q = {"question_id": "q", "question": "Where was the Alpha Band founder born?", "type": "bridge",
             "context_passage_ids": list(ps), "gold_supporting_passage_ids": ["a", "b"]}
        rows = gr.evaluate_question(q, ps, extractions(), emb, emb["a"])
        self.assertEqual([r["config"] for r in rows], gr.CONFIGS)
        for r in rows:
            self.assertEqual(sorted(r["ranked_passage_ids"]), sorted(ps))
            for pid, rank in r["gold_ranks"].items():
                self.assertEqual(r["ranked_passage_ids"][rank - 1], pid)
        self.assertTrue(gr.validation_checks(rows, [q])["all_passed"])


class MetricAndStatsTests(unittest.TestCase):
    def test_row_metrics(self):
        q = {"question_id": "q", "context_passage_ids": list("abcdefghij"), "gold_supporting_passage_ids": ["a", "f"]}
        row = gr.row_for(q, "x", list("abcdefghij"), {})
        self.assertEqual(row["recall_at_2"], 0.5)
        self.assertEqual(row["complete_support_at_5"], 0)
        self.assertEqual(row["complete_support_at_10"], 1)
        self.assertEqual(row["gold_ranks"], {"a": 1, "f": 6})
        with self.assertRaises(AssertionError):
            gr.row_for(q, "x", list("abcdefghi") + ["z"], {})

    def test_holm_and_paired_diff(self):
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        rows = []
        for i in range(40):
            rows.append({"question_id": str(i), "config": "A", "m": 1})
            rows.append({"question_id": str(i), "config": "B", "m": 1 if i < 10 else 0})
        d = gr.paired_diff(rows, "A", "B", "m")
        self.assertAlmostEqual(d["mean_paired_difference"], 0.75)
        self.assertEqual((d["questions_a_better"], d["questions_b_better"]), (30, 0))
        self.assertLess(d["p_value"], 0.001)
        same = gr.paired_diff(rows, "A", "A", "m")
        self.assertEqual(same["mean_paired_difference"], 0.0)
        self.assertEqual(same["p_value"], 1.0)


def raw_musique(qid="2hop__1_2", n=20, gold=(3, 7), dup=False):
    paras = [{"idx": i, "title": f"T{i}", "paragraph_text": f"text {i}", "is_supporting": i in gold} for i in range(n)]
    if dup:
        paras[1] = dict(paras[0], idx=1)
    return {"id": qid, "question": "Q?", "answer": "A", "answer_aliases": ["a"], "answerable": True,
            "paragraphs": paras, "question_decomposition": [{"id": 1, "question": "x", "answer": "y", "paragraph_support_idx": 3}]}


class MusiqueLoaderTests(unittest.TestCase):
    def test_real_field_names_convert(self):
        q, ps = convert_record(raw_musique())
        self.assertEqual(q["type"], "2hop")
        self.assertEqual(len(q["context_passage_ids"]), 20)
        self.assertEqual(len(q["gold_supporting_passage_ids"]), 2)
        self.assertTrue(set(q["gold_supporting_passage_ids"]) <= set(q["context_passage_ids"]))
        self.assertEqual({p["title"] for p in ps if p["passage_id"] in q["gold_supporting_passage_ids"]}, {"T3", "T7"})
        self.assertTrue(is_eligible(q))

    def test_hop_count_and_duplicates(self):
        self.assertEqual(hop_count("3hop1__1_2_3"), 3)
        self.assertEqual(hop_count("4hop3__1"), 4)
        q, _ = convert_record(raw_musique(dup=True))
        self.assertEqual(q["unique_context_passage_count"], 19)
        self.assertFalse(is_eligible(q))

    def test_sampling_is_deterministic_and_filters(self):
        recs = [raw_musique(f"2hop__{i}_0") for i in range(30)] + [raw_musique("2hop__short", n=18)]
        a, _, eligible = sample_questions(recs, 10, 2027)
        b, _, _ = sample_questions(recs, 10, 2027)
        self.assertEqual(eligible, 30)
        self.assertEqual([q["question_id"] for q in a], [q["question_id"] for q in b])
        self.assertNotIn("2hop__short", {q["question_id"] for q in a})


if __name__ == "__main__":
    unittest.main()
