"""Build a lightweight entity co-occurrence graph from passage sentences."""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import spacy

from src.common import ROOT, read_json, write_json

ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "WORK_OF_ART", "PRODUCT", "NORP"}


def normalize_entity(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def build_entity_graph(passages: list[dict[str, Any]], model_name: str = "en_core_web_sm") -> tuple[nx.Graph, dict[str, list[str]], dict[str, list[str]]]:
    nlp = spacy.load(model_name)
    graph = nx.Graph()
    entity_passages: dict[str, set[str]] = defaultdict(set)
    passage_entities: dict[str, set[str]] = defaultdict(set)
    entity_types: dict[str, set[str]] = defaultdict(set)
    display_names: dict[str, str] = {}

    for passage in passages:
        passage_id = str(passage["passage_id"])
        sentences = passage.get("sentences") or str(passage.get("text", "")).splitlines()
        for sentence in sentences:
            sentence = str(sentence).strip()
            if not sentence:
                continue
            doc = nlp(sentence)
            entities_in_sentence: dict[str, str] = {}
            for ent in doc.ents:
                if ent.label_ not in ENTITY_LABELS:
                    continue
                normalized = normalize_entity(ent.text)
                if len(normalized) < 2:
                    continue
                entities_in_sentence.setdefault(normalized, ent.text.strip())
                entity_types[normalized].add(ent.label_)
                display_names.setdefault(normalized, ent.text.strip())
                entity_passages[normalized].add(passage_id)
                passage_entities[passage_id].add(normalized)

            for entity in entities_in_sentence:
                if entity not in graph:
                    graph.add_node(entity, display_name=display_names[entity], entity_types=",".join(sorted(entity_types[entity])))
            for left, right in combinations(sorted(entities_in_sentence), 2):
                if graph.has_edge(left, right):
                    graph[left][right]["weight"] += 1
                else:
                    graph.add_edge(left, right, weight=1)

    # Populate final node metadata after all mentions have been seen.
    for entity in graph.nodes:
        graph.nodes[entity]["display_name"] = display_names.get(entity, entity)
        graph.nodes[entity]["entity_types"] = ",".join(sorted(entity_types.get(entity, set())))

    graph.graph["construction_rule"] = "NER entities as nodes; entities co-occurring in the same sentence are connected; edge weight is co-occurrence count."
    graph.graph["spacy_model"] = model_name
    graph.graph["node_unit"] = "normalized entity surface form"

    entity_passages_out = {key: sorted(values) for key, values in sorted(entity_passages.items())}
    passage_entities_out = {key: sorted(values) for key, values in sorted(passage_entities.items())}
    return graph, entity_passages_out, passage_entities_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", default="data/processed/passages.json")
    parser.add_argument("--graph-out", default="data/processed/entity_graph.graphml")
    parser.add_argument("--entity-passages-out", default="data/processed/entity_passages.json")
    parser.add_argument("--passage-entities-out", default="data/processed/passage_entities.json")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    args = parser.parse_args()

    passages = read_json(args.passages)
    graph, entity_passages, passage_entities = build_entity_graph(passages, args.spacy_model)
    graph_path = Path(args.graph_out)
    if not graph_path.is_absolute():
        graph_path = ROOT / graph_path
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, graph_path)
    write_json(entity_passages, args.entity_passages_out)
    write_json(passage_entities, args.passage_entities_out)
    print(f"Built graph: {graph.number_of_nodes()} entities, {graph.number_of_edges()} co-occurrence edges.")
    print(f"Wrote graph to {graph_path.resolve()}")
    print("Note: this is an NER-derived co-occurrence graph, not a semantically typed knowledge graph.")


if __name__ == "__main__":
    main()
