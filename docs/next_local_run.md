# Historical Local Run Plan — 500-question sample and Leiden validation

> **Historical instructions — not executable from the public aggregate-only checkout.** This handoff predates validated Stage A and Phase B. Its commands depend on raw HotpotQA files and exploratory outputs that are not published; several proposed steps have since been superseded. Keep this page as research history, not as a current run plan. Use [`reproduction_guide.md`](reproduction_guide.md) for current data preparation and canonical study reproduction, and [`research/PHASE_B_REVIEW_SUMMARY.md`](research/PHASE_B_REVIEW_SUMMARY.md) for current status.

The commands below document the historical handoff only. They will not run from this public checkout without separately obtaining and preparing the required dataset and other local inputs.

## 1. Set up the environment

From the extracted project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If Python 3.13 cannot resolve a compatible package build, use Python 3.12 for this environment. The project requires the `datasets`, spaCy model, `python-igraph`, and `leidenalg` packages for the dataset loader, graph construction, and Leiden experiments.

## 2. First run the missing Leiden cross-seed experiment on the existing 50-question graph

This uses saved data and does not download HotpotQA again:

```bash
python -m src.evaluate_seed_stability \
  --questions data/pilot_50/questions.json \
  --passages data/pilot_50/passages.json \
  --graph data/pilot_50/entity_graph.graphml \
  --entity-passages data/pilot_50/entity_passages.json \
  --passage-entities data/pilot_50/passage_entities.json \
  --methods leiden \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --output-dir results/pilot_50/seed_stability_leiden_10seeds
```

This fills the one remaining clustering-method comparison for the existing 50-question graph. Keep the current Louvain results intact.

## 3. Create a new 500-question HotpotQA sample

```bash
python -m src.load_data \
  --split validation \
  --limit 500 \
  --seed 2027 \
  --output-dir data/pilot_500
```

The loader now records the shuffle buffer size and the actual difficulty/type/gold-support distributions in `data/pilot_500/dataset_manifest.json`. Inspect this manifest before graph construction. Do not assume the sample is balanced; report the actual distribution.

HotpotQA's `distractor` validation split contains 7,405 examples. The 500-question shuffle buffer is 10,000, which is larger than that split. The dataset card lists CC BY-SA 4.0 licensing; retain attribution and check share-alike requirements before publicly redistributing modified dataset material.

## 4. Build the original same-sentence entity graph

```bash
python -m src.build_graph \
  --passages data/pilot_500/passages.json \
  --graph-out data/pilot_500/entity_graph.graphml \
  --entity-passages-out data/pilot_500/entity_passages.json \
  --passage-entities-out data/pilot_500/passage_entities.json
```

## 5. Run the graph-construction ablation on 500 questions

```bash
python -m src.evaluate_graph_construction_variants \
  --dataset-dir data/pilot_500 \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --output-dir results/pilot_500/graph_construction_variants
```

This compares same-sentence and same-passage entity co-occurrence, preserving the entity node set and the saved NER mapping.

## 6. Run the existing TF-IDF perturbation/evidence-retrieval experiment on 500 questions

```bash
python -m src.evaluate_perturbed_retrieval \
  --questions data/pilot_500/questions.json \
  --passages data/pilot_500/passages.json \
  --graph data/pilot_500/entity_graph.graphml \
  --entity-passages data/pilot_500/entity_passages.json \
  --passage-entities data/pilot_500/passage_entities.json \
  --output-dir results/pilot_500/perturbed_retrieval \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --removal-fraction 0.05 \
  --seed-k 2 \
  --k-values 5 10
```

This remains the TF-IDF-seeded / TF-IDF-reranked pilot. It is not the dense-retrieval or generation stage.

## 7. What to bring back for the next iteration

Send the following output files (or zip the `data/pilot_500/` and `results/pilot_500/` folders):

- `data/pilot_500/dataset_manifest.json`
- `results/pilot_50/seed_stability_leiden_10seeds/partition_similarity_summary.csv`
- `results/pilot_50/seed_stability_leiden_10seeds/retrieval_jaccard_bootstrap_ci.csv`
- `results/pilot_500/graph_construction_variants/retrieval_summary.csv`
- `results/pilot_500/graph_construction_variants/paired_bootstrap_passage_vs_sentence.csv`
- `results/pilot_500/perturbed_retrieval/perturbed_retrieval_summary.csv`

Once the larger sample and Leiden outputs are validated, the next code addition should be the matched dense-retrieval baseline (BGE-small or Contriever). Then add local generation and answer EM/F1/faithfulness on a controlled subset before scaling generation runs.
