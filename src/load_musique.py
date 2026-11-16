"""Prepare a fixed MuSiQue (answerable v1.0) sample in the project's data format.

Output files mirror ``src.load_data``: ``questions.json``, ``passages.json``,
``question_contexts.json`` and ``dataset_manifest.json``. Sampling follows
PREREGISTRATION_graph_ranking.md: questions with exactly 20 unique paragraphs,
IDs sorted lexicographically, 500 drawn with ``numpy.random.default_rng(seed)``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.common import ROOT, write_json
from src.load_data import make_passage_id

REPO_ID = "bdsaglam/musique"
REVISION = "22873a405dd809893b22ada0b499299fb612d2df"
FILES = {"validation": "musique_ans_v1.0_dev.jsonl", "train": "musique_ans_v1.0_train.jsonl"}
EXPECTED_CONTEXTS = 20


def hop_count(question_id: str) -> int:
    """MuSiQue IDs start with '2hop__', '3hop1__', '4hop3__', ..."""
    prefix = str(question_id).split("__", 1)[0]
    if not prefix[:1].isdigit() or "hop" not in prefix:
        raise ValueError(f"Unrecognized MuSiQue ID prefix: {question_id}")
    return int(prefix[0])


def convert_record(record: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert one raw MuSiQue record (real field names) to question + passage records."""
    passages: list[dict[str, Any]] = []
    context_ids: list[str] = []
    gold_ids: list[str] = []
    titles: list[str] = []
    for para in sorted(record["paragraphs"], key=lambda p: int(p["idx"])):
        title = str(para["title"]).strip()
        text = str(para["paragraph_text"]).strip()
        pid = make_passage_id(title, [text])
        passages.append({"passage_id": pid, "title": title, "text": text, "sentences": [text]})
        context_ids.append(pid)
        titles.append(title)
        if bool(para["is_supporting"]):
            gold_ids.append(pid)
    unique_ids = list(dict.fromkeys(context_ids))
    qid = str(record["id"])
    question = {
        "question_id": qid,
        "question": str(record["question"]),
        "answer": str(record["answer"]),
        "answer_aliases": list(map(str, record.get("answer_aliases") or [])),
        "type": f"{hop_count(qid)}hop",
        "musique_type": qid.split("__", 1)[0],
        "hop_count": hop_count(qid),
        "level": "musique_ans",
        "context_titles": titles,
        "context_passage_ids": unique_ids,
        "context_passage_count": len(context_ids),
        "unique_context_passage_count": len(unique_ids),
        "gold_supporting_passage_ids": sorted(set(gold_ids)),
        "question_decomposition": record.get("question_decomposition"),
    }
    return question, passages


def is_eligible(question: dict[str, Any]) -> bool:
    return (question["unique_context_passage_count"] == EXPECTED_CONTEXTS
            and question["context_passage_count"] == EXPECTED_CONTEXTS
            and len(question["gold_supporting_passage_ids"]) >= 2
            and set(question["gold_supporting_passage_ids"]) <= set(question["context_passage_ids"]))


def sample_questions(records: Iterable[dict[str, Any]], limit: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    converted = [convert_record(r) for r in records]
    eligible = sorted((c for c in converted if is_eligible(c[0])), key=lambda c: c[0]["question_id"])
    if len(eligible) < limit:
        raise ValueError(f"Only {len(eligible)} eligible questions for limit {limit}")
    idx = np.random.default_rng(seed).choice(len(eligible), size=limit, replace=False)
    chosen = sorted((eligible[int(i)] for i in idx), key=lambda c: c[0]["question_id"])
    passages: dict[str, dict[str, Any]] = {}
    for _, ps in chosen:
        for p in ps:
            if p["passage_id"] in passages and passages[p["passage_id"]]["text"] != p["text"]:
                raise ValueError(f"Passage ID collision for {p['passage_id']}")
            passages.setdefault(p["passage_id"], p)
    return [c[0] for c in chosen], sorted(passages.values(), key=lambda p: p["passage_id"]), len(eligible)


def sha256_list(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=sorted(FILES))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output-dir", default="data/musique_500")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out = out if out.is_absolute() else ROOT / out
    if (out / "questions.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing sample in {out}")

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO_ID, FILES[args.split], repo_type="dataset", revision=args.revision)
    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    questions, passages, eligible_n = sample_questions(records, args.limit, args.seed)
    contexts = {q["question_id"]: {"context_passage_ids": q["context_passage_ids"],
                                   "gold_supporting_passage_ids": q["gold_supporting_passage_ids"]} for q in questions}
    manifest = {
        "dataset": REPO_ID, "file": FILES[args.split], "revision": args.revision, "split": args.split,
        "source_question_count": len(records), "eligible_question_count": eligible_n,
        "eligibility": f"exactly {EXPECTED_CONTEXTS} unique paragraphs, >=2 gold paragraphs",
        "sampling": "IDs sorted lexicographically; numpy.random.default_rng(seed).choice without replacement",
        "seed": args.seed, "loaded_questions": len(questions), "unique_passages": len(passages),
        "hop_distribution": dict(sorted(Counter(q["type"] for q in questions).items())),
        "musique_type_distribution": dict(sorted(Counter(q["musique_type"] for q in questions).items())),
        "gold_support_count_distribution": {str(k): v for k, v in sorted(Counter(len(q["gold_supporting_passage_ids"]) for q in questions).items())},
        "question_ids_sha256": sha256_list(q["question_id"] for q in questions),
        "sorted_passage_ids_sha256": sha256_list(sorted(p["passage_id"] for p in passages)),
        "license_note": "MuSiQue is released under CC BY 4.0; paragraphs are Wikipedia-derived. Do not redistribute without review.",
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(passages, out / "passages.json")
    write_json(questions, out / "questions.json")
    write_json(contexts, out / "question_contexts.json")
    write_json(manifest, out / "dataset_manifest.json")
    print(json.dumps({k: manifest[k] for k in ("eligible_question_count", "loaded_questions", "unique_passages", "hop_distribution")}))


if __name__ == "__main__":
    main()
