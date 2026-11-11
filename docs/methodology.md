# Methodology

This document describes the executed retrieval experiments and the boundaries on their interpretation. They are empirical comparisons, not a proposed GraphRAG architecture.

**Public release scope:** Raw HotpotQA input files and per-question outputs/caches are excluded from the proposed Git index. The counts and results below describe the locally executed experiments and the retained aggregate manifests/reports; they do not imply that a fresh checkout contains all inputs needed to rerun them.

## Dataset and sample

The source is the HotpotQA `distractor` validation split, sampled and stored in this repository. `data/pilot_500_v2/dataset_manifest.json` records 500 questions, 4,947 unique passages, 404 bridge and 96 comparison questions overall, and two context-count strata: 498 questions with ten contexts and two with two contexts. All 500 have their gold support passages in the provided contexts. The primary analysis uses only the 498 complete ten-context questions; the two exceptions are separate Stage A outputs. Every question has two gold support passage IDs. All are from one hard-only sample.

Question context passage IDs are treated as the allowed candidate universe. Evidence matching is exact passage-ID matching, not title matching or semantic similarity. For gold support set `G_q`, candidate set `C_q`, ranked IDs `R_q`, and cutoff k:

- Candidate-generation recall = `|G_q ∩ C_q| / |G_q|`.
- Recall@k = `|G_q ∩ R_q[:k]| / |G_q|`.
- Complete support@k = 1 if `G_q ⊆ R_q[:k]`, otherwise 0.

The reported complete support value is the question-level mean of this indicator. Vanilla TF-IDF and BGE have all ten contexts in their candidate pool, so their candidate recall is 1. Graph methods’ candidate recall is computed on the pre-reranking generated set.

## Stage A: local lexical baseline and shared-graph comparison

`src/evaluate_question_local.py` has two tracks:

**Question-local track (primary):** It fits `TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)` separately on each question’s own context passage title and text. It transforms that question only, ranks those same passages by cosine similarity, and resolves ties by passage ID. No other question’s passages participate in that local fit or candidate universe.

**Shared-global-graph track (secondary):** It loads the pre-existing graph in `data/pilot_500`, whose graph and entity mappings were built across the pooled corpus. It takes five local TF-IDF seed passages, expands through neighboring entities, filters candidates to the current question’s contexts, and ranks them using a TF-IDF index fitted over the shared passage corpus. This track is context-filtered but has cross-question graph and corpus-statistic information; it is not question-local graph construction.

Stage A uses 10,000 question-level bootstrap resamples and percentile 95% intervals. Paired sign-flip tests use 20,000 Monte Carlo draws; Holm correction is applied across the six retrieval outcomes within each context-count stratum. One-time global TF-IDF index construction and graph expansion are excluded from Track B’s per-query ranking timer; see the Stage A report/log for runtime semantics.

## Phase B: dense retrieval and question-local graphs

The Phase B implementation is `src/evaluate_dense_graph.py`. Its dataset is `data/pilot_500_v2`. The 50-question pilot is stratified proportionally by question type from the 498 complete questions, with selection seed 2027 (41 bridge, 9 comparison). The full run evaluates all 498. Each configuration has one raw record per question.

### Dense model

- Model: `BAAI/bge-small-en-v1.5`, pinned to Hugging Face commit `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
- Query prefix: `Represent this sentence for searching relevant passages: `; passage texts have no prefix.
- Embeddings: 384 dimensions, float32 cache, normalized at encoding (`normalize_embeddings=True`).
- Similarity: dot product of normalized vectors, equivalent to cosine similarity.
- Ranking uses passage-ID tie-breaking.

Passage vectors are computed from passage text and cached; the dense model has no pooled corpus statistic. For each question, scoring matrices and chunk graph construction use only its ten context vectors. Model weights are downloaded from Hugging Face on first use and are not stored in this repository.

### Seven retrieval configurations

| ID | Graph or retrieval construction | Candidate-generation and ranking behavior |
|---|---|---|
| `vanilla_tfidf` | A fresh local TF-IDF fit for every question, reusing the Stage A function. | All ten context passages are ranked with local TF-IDF. |
| `vanilla_bge` | No graph. | All ten contexts ranked by query/passage BGE similarity. |
| `entity_sentence_frequency_weighted` | Normalized entity surface-form nodes; add an undirected edge for each pair found in the same sentence; weight is total sentence co-occurrence count in the question’s contexts. | BGE’s top five context passages seed expansion. Entities in those passages are expanded to their top three weighted entity neighbors; passages mapped to the expanded entities are added. BGE scores rerank candidates. |
| `entity_passage_frequency_weighted` | Same nodes and passage-local entity extraction; each passage adds each pair at most once; edge weight counts co-occurring contexts. | Same entity-neighbor candidate expansion and BGE reranking. |
| `entity_sentence_binary` | Sentence co-occurrence edge set; every present edge has weight 1. | Same entity-neighbor candidate expansion and BGE reranking. |
| `entity_sentence_top50_sparsified` | Sentence-frequency graph; retain `ceil(0.5 × edge count)` highest-weight edges, breaking ties by lexical endpoint IDs; retain nodes that become isolated. | Same entity-neighbor candidate expansion and BGE reranking. |
| `chunk_bge_knn2` | Ten passage/chunk nodes; each selects its two nearest question-local BGE passage vectors; symmetrize by union-kNN. | BGE top five passage nodes seed direct-neighbor expansion; BGE scores rerank the resulting candidates. |

The entity extractor processes passages independently with spaCy `en_core_web_sm` 3.8.0, storing recognized normalized entities by sentence. Legacy `passage_entities.json` provenance is undocumented and was not used to construct Phase B graphs. Graph expansion is a candidate filter/expander; there is no graph-score fusion, generated answer stage, or local community retrieval.

### Inference and validation

For Phase B, the independent unit is the question. Confidence intervals use 10,000 bootstrap samples of question rows, percentile 2.5% and 97.5% limits. Paired differences are tested with two-sided sign-flip Monte Carlo randomization (20,000 draws; plus-one p correction). Holm correction covers the 69 pre-specified baseline and graph-construction outcome contrasts in each run’s `paired_comparisons.csv`.

`src/phase_b_independent_audit.py` is separate from experiment graph builders, metric aggregators, and statistical helpers. It reconstructs graph structures, candidates, rankings, metric denominators, structural measures, and summary statistics from raw saved question rows, the dataset, entity cache, and embedding cache. The saved independent audit reports zero out-of-context candidates/rankings, zero graph node/edge reconstruction mismatches, zero metric or denominator mismatches, and matching summary files for both final scopes.

## Findings and limits

On the 498 questions, BGE Recall@2 was 0.748 (95% CI 0.723–0.773) versus local TF-IDF 0.658 (0.633–0.683); Recall@5 was 0.906 (0.888–0.923) versus 0.836 (0.814–0.858). Entity graphs preserved the BGE ranking among candidates that survived expansion but reduced candidate recall to 0.984 and complete support@10 to 0.968. Chunk kNN candidate recall was 0.997 and complete support@10 0.994. No chunk/BGE paired difference survived Holm correction. These are retrieval results only.

The graph sensitivity matrix remains incomplete: passage-level binary weighting, alternative sparsification controls, sentence nodes, entity semantic edges, bipartite entity-passage graphs, alias resolution, validated typed relations, local community-based retrieval, and repeated-seed stability remain outstanding. Entity extraction has no manually reviewed quality estimate. Answer EM/F1, faithfulness, and an established GraphRAG system have not been evaluated.

## Data and model licensing notes

The upstream HotpotQA dataset card labels the dataset CC BY-SA 4.0. The BGE model card labels `BAAI/bge-small-en-v1.5` MIT. These are upstream licenses; the repository has not selected a license for its original code. Review [data/README.md](../data/README.md) and the upstream terms before redistribution.

- [HotpotQA dataset card and license](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [BGE-small model card and license](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [Creative Commons BY-SA 4.0 legal text](https://creativecommons.org/licenses/by-sa/4.0/legalcode)
