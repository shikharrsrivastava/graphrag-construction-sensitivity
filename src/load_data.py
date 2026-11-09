
"""Load a reproducible HotpotQA sample and preserve question-specific contexts."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any

from datasets import load_dataset
from src.common import ROOT, write_json

DATASET_ID = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
EXPECTED_CONTEXT_PASSAGES = 10


def shuffle_buffer_size_for_limit(limit: int) -> int:
    return max(1000, int(limit) * 20)


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title).strip()).casefold()


def normalize_passage_text(sentences: list[str]) -> str:
    return " ".join(
        str(s).strip() for s in sentences
    ).strip().casefold()


def extract_pairs(value: Any, first_key: str, second_key: str):
    if isinstance(value, dict):
        first = value.get(first_key, []) or []
        second = value.get(second_key, []) or []
        return list(zip(first, second))

    if isinstance(value, (list, tuple)):
        return [
            (item[0], item[1])
            for item in value
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ]

    return []


def clean_sentences(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = []

    return [
        str(sentence).strip()
        for sentence in values
        if sentence is not None and str(sentence).strip()
    ]


def context_to_docs(context: Any) -> list[dict[str, Any]]:
    docs = []

    if isinstance(context, dict):
        titles = context.get("title", []) or []
        sentence_lists = context.get("sentences", []) or []

        for title, sentences in zip(titles, sentence_lists):
            docs.append({
                "title": str(title).strip(),
                "sentences": clean_sentences(sentences),
            })

    elif isinstance(context, (list, tuple)):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                docs.append({
                    "title": str(item[0]).strip(),
                    "sentences": clean_sentences(item[1]),
                })

    return docs


def make_passage_id(title: str, sentences: list[str]) -> str:
    payload = (
        normalize_title(title)
        + "\n"
        + normalize_passage_text(sentences)
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"p_{digest}"


def safe_sentence_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_sample(
    split: str,
    limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    dataset = load_dataset(
        DATASET_ID,
        CONFIG,
        split=split,
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=shuffle_buffer_size_for_limit(limit),
    )

    rows = list(islice(iter(dataset), limit))

    if not rows:
        raise RuntimeError("No examples returned from HotpotQA.")

    # Shared, deduplicated passage corpus.
    passage_by_key = {}
    passage_key_by_id = {}

    questions = []
    seen_question_ids = set()

    for row_number, row in enumerate(rows):

        contexts = context_to_docs(row.get("context", {}))

        # Exact passages belonging to THIS question.
        context_passage_ids = []

        title_to_ids = defaultdict(list)

        for context_doc in contexts:

            title = context_doc["title"]
            sentences = context_doc["sentences"]

            if not title or not sentences:
                continue

            passage_id = make_passage_id(title, sentences)

            title_key = normalize_title(title)

            key = (
                title_key,
                normalize_passage_text(sentences),
            )

            # Detect shortened-hash collisions.
            previous_key = passage_key_by_id.get(passage_id)

            if previous_key is not None and previous_key != key:
                raise RuntimeError(
                    f"Passage ID collision: {passage_id}"
                )

            passage_key_by_id[passage_id] = key

            # Deduplicate the global corpus.
            if key not in passage_by_key:
                passage_by_key[key] = {
                    "passage_id": passage_id,
                    "title": title,
                    "text": "\n".join(sentences),
                    "sentences": sentences,
                }

            # Preserve original question-context membership.
            context_passage_ids.append(passage_id)

            if passage_id not in title_to_ids[title_key]:
                title_to_ids[title_key].append(passage_id)

        # Gold labels are kept separate from graph construction.
        gold_pairs = extract_pairs(
            row.get("supporting_facts", {}),
            "title",
            "sent_id",
        )

        gold_facts = []

        for title, sentence_id in gold_pairs:

            parsed_id = safe_sentence_id(sentence_id)

            if title is not None and parsed_id is not None:
                gold_facts.append({
                    "title": str(title),
                    "sent_id": parsed_id,
                })

        gold_title_keys = {
            normalize_title(item["title"])
            for item in gold_facts
        }

        missing_gold_titles = [
            title
            for title in gold_title_keys
            if title not in title_to_ids
        ]

        gold_passage_ids = sorted({
            passage_id
            for title in gold_title_keys
            for passage_id in title_to_ids.get(title, [])
        })

        question_id = str(row.get("id") or f"row-{row_number}")

        if question_id in seen_question_ids:
            raise RuntimeError(
                f"Duplicate question ID: {question_id}"
            )

        seen_question_ids.add(question_id)

        questions.append({
            "question_id": question_id,
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
            "type": str(row.get("type", "")),
            "level": str(row.get("level", "")),

            "context_titles": [
                item["title"] for item in contexts
            ],

            "context_passage_ids": context_passage_ids,
            "context_passage_count": len(context_passage_ids),
            "unique_context_passage_count": len(set(context_passage_ids)),

            "gold_supporting_facts": gold_facts,
            "gold_supporting_passage_ids": gold_passage_ids,
            "missing_gold_support_titles": missing_gold_titles,
        })

    passages = sorted(
        passage_by_key.values(),
        key=lambda item: item["passage_id"],
    )

    return passages, questions


def build_manifest(passages, questions, split, limit, seed):

    levels = Counter(q["level"] for q in questions)
    types = Counter(q["type"] for q in questions)

    gold_counts = Counter(
        str(len(q["gold_supporting_passage_ids"]))
        for q in questions
    )

    context_counts = Counter(
        str(q["context_passage_count"])
        for q in questions
    )

    unique_context_counts = Counter(
        str(q["unique_context_passage_count"])
        for q in questions
    )

    expected_count = sum(
        q["context_passage_count"] == EXPECTED_CONTEXT_PASSAGES
        for q in questions
    )

    all_gold_in_context = sum(
        set(q["gold_supporting_passage_ids"]).issubset(
            set(q["context_passage_ids"])
        )
        for q in questions
    )

    missing_gold = sum(
        bool(q["missing_gold_support_titles"])
        for q in questions
    )

    question_hash = "\n".join(
        q["question_id"] for q in questions
    )

    passage_hash = "\n".join(sorted(
        p["passage_id"] for p in passages
    ))

    return {
        "dataset": DATASET_ID,
        "config": CONFIG,
        "split": split,

        "requested_question_limit": limit,
        "loaded_questions": len(questions),
        "unique_question_ids": len({
            q["question_id"] for q in questions
        }),
        "unique_passages": len(passages),

        "stream_shuffle_seed": seed,
        "shuffle_buffer_size": shuffle_buffer_size_for_limit(limit),

        "difficulty_distribution": dict(levels),
        "question_type_distribution": dict(types),
        "gold_support_count_distribution": dict(gold_counts),

        "context_passage_count_distribution": dict(context_counts),
        "unique_context_passage_count_distribution": dict(
            unique_context_counts
        ),

        "expected_context_passages_per_question": EXPECTED_CONTEXT_PASSAGES,
        "questions_with_expected_context_count": expected_count,

        "questions_with_all_gold_supports_in_context": all_gold_in_context,
        "questions_with_missing_gold_support_titles": missing_gold,

        "total_question_context_slots": sum(
            q["context_passage_count"] for q in questions
        ),

        "question_ids_sha256": hashlib.sha256(
            question_hash.encode("utf-8")
        ).hexdigest(),

        "sorted_passage_ids_sha256": hashlib.sha256(
            passage_hash.encode("utf-8")
        ).hexdigest(),

        "note": (
            "question_contexts are preserved in original order. "
            "passages.json is the deduplicated global corpus. "
            "Gold labels are used only for evaluation."
        ),
    }


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
    )

    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/processed")

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")

    passages, questions = load_sample(
        args.split,
        args.limit,
        args.seed,
    )

    output_dir = Path(args.output_dir).expanduser()

    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(passages, output_dir / "passages.json")
    write_json(questions, output_dir / "questions.json")

    # Compact question-to-context mapping for retrieval evaluation.
    question_contexts = {
        q["question_id"]: {
            "context_passage_ids": q["context_passage_ids"],
            "gold_supporting_passage_ids": q["gold_supporting_passage_ids"],
        }
        for q in questions
    }

    write_json(
        question_contexts,
        output_dir / "question_contexts.json",
    )

    manifest = build_manifest(
        passages,
        questions,
        args.split,
        args.limit,
        args.seed,
    )

    write_json(
        manifest,
        output_dir / "dataset_manifest.json",
    )

    print(f"Loaded {len(questions)} questions.")
    print(f"Unique passages: {len(passages)}")
    print(f"Saved to: {output_dir.resolve()}")

    print(
        "Questions with 10 contexts:",
        f"{manifest['questions_with_expected_context_count']}/{len(questions)}",
    )

    print(
        "Questions with all gold passages present:",
        f"{manifest['questions_with_all_gold_supports_in_context']}/{len(questions)}",
    )

    print(
        "Questions with missing gold titles:",
        manifest["questions_with_missing_gold_support_titles"],
    )

    if manifest["questions_with_all_gold_supports_in_context"] != len(questions):
        print("WARNING: Some gold supporting passages are missing from their contexts.")

    if manifest["questions_with_expected_context_count"] != len(questions):
        print("WARNING: Some questions do not have exactly 10 usable context passages.")


if __name__ == "__main__":
    main()
