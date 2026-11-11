# Phase B: Consolidated Review Summary

**Prepared:** 2026-11-11
**Dataset:** `data/pilot_500_v2`
**Evaluation scope:** 50-question feasibility pilot and 498-question primary evaluation
**Purpose:** Independent researcher review of the executed Phase B dense retrieval and question-local graph experiment.

## 1. Scope and interpretation

Phase B compares seven retrieval configurations on the same primary set of 498 HotpotQA questions, each with ten allowed context passages. The two exceptional two-context questions are excluded, consistent with the separate variable-context analysis in Stage A. Each method has one saved row per question (3,486 rows total). Questions are the inferential unit. The dataset contains 404 bridge and 94 comparison questions, all from the existing hard sample; claims should not be generalized beyond this sample without additional sampling.

The experiment is a retrieval and graph-construction study. It does not evaluate answer generation, Exact Match, token F1, or evidence faithfulness. The graph methods construct separate graphs from each question’s ten passages. There is no pooled global graph in the Phase B primary track.

## 2. Data, evidence matching, and metric definitions

The run configuration records all 498 ten-context questions and all seven configurations. Exact passage IDs are used throughout. For question *q*, with gold support set `Gq`, candidates `Cq`, and ranked passage IDs `Rq`, candidate-generation recall is `|Gq ∩ Cq| / |Gq|`; Recall@k is `|Gq ∩ Rq[:k]| / |Gq|`; complete-support@k is the indicator `Gq ⊆ Rq[:k]`. The denominator is the count of gold support passage IDs for that question, not the number of contexts, candidate passages, or retrieved passages. In this dataset, both gold supports occur in each question’s own contexts. The vanilla methods use all ten contexts as candidates, so their candidate-generation recall is 1 by construction. Graph candidate recall is measured before reranking and is distinct from Recall@k after BGE reranking.

The saved full raw file `results/dense_graph_v1/primary498_final/question_results.jsonl` contains each question’s context IDs, gold IDs, candidate IDs, ranked IDs, candidate recall and denominator, graph nodes and edges, graph scope, graph structure, and timing fields for every configuration. The audit matched every gold denominator and metric to an independent calculation.

## 3. Exact seven configurations

All listed graphs, seeds, candidate pools, and rankings are constructed question by question. For the graph-assisted configurations, BGE’s top five passages for that question provide seed passages. The seed passages are retained. Graph expansion adds locally mapped passages, after which the candidates are sorted by the same vanilla BGE query-to-passage score. The graph does not contribute an additional retrieval score, weighted fusion, community retrieval, or global index.

| Saved configuration | Implementation |
|---|---|
| `vanilla_tfidf` | Reuses Stage A’s `rank_local`: fits a fresh scikit-learn TF-IDF vectorizer on that question’s ten passage title/text strings, with unigram and bigram features, English stop words, sublinear term frequency; transforms that question only and ranks its ten passages. |
| `vanilla_bge` | Scores and ranks all ten allowed passages by query-to-passage BGE dot product (cosine-equivalent for normalized vectors). No graph or candidate pruning. |
| `entity_sentence_frequency_weighted` | Nodes are normalized entity surface forms. Each sentence in each allowed passage contributes one count for every pair of entities found in that same sentence. An undirected edge’s weight is the total pair co-occurrence count across the ten passages. |
| `entity_passage_frequency_weighted` | Same entity nodes and extractor, but each passage contributes a pair once if both entities occur anywhere in that passage. Edge weight is the number of the question’s passages in which that pair co-occurs. |
| `entity_sentence_binary` | Uses the same sentence-level edge existence as sentence frequency; every present edge is assigned weight 1. It tests binary versus frequency edge weights at the sentence level only. |
| `entity_sentence_top50_sparsified` | Starts from the sentence-level frequency graph and retains `ceil(0.5 × edge_count)` strongest edges. Ordering is descending frequency, then lexical endpoint IDs for deterministic ties. All entity nodes are retained, so pruning can create isolates. |
| `chunk_bge_knn2` | One node per context passage/chunk. Computes pairwise cosine-equivalent BGE similarity using the ten question-local passage vectors, finds each node’s two nearest neighbors (lexical passage-ID tie break), and makes an undirected union-kNN graph. Candidate expansion adds direct neighbors of BGE’s top-five seed nodes. |

The entity extractor is spaCy `en_core_web_sm` 3.8.0. It processes each passage on its own, splits it into sentences, filters to the configured spaCy entity label set, normalizes surface forms, and records per-sentence entities. A legacy passage-to-entity mapping was not used to build graphs because its manifest does not establish the extractor/model provenance. Its overlap was checked against the new output as a consistency diagnostic only.

### BGE model and vector scoring

| Property | Verified value |
|---|---|
| Model identifier | `BAAI/bge-small-en-v1.5` |
| Resolved, pinned revision | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Embedding dimension | 384 |
| Device | CPU; run configuration says CUDA and MPS unavailable |
| Maximum sequence length | 512 |
| Query prefix | `Represent this sentence for searching relevant passages: ` |
| Passage prefix | None |
| Normalization | `normalize_embeddings=True` for query and passage vectors; cached embeddings are float32 and L2-normalized |
| Similarity | Dot product of normalized vectors, mathematically cosine similarity |
| Independently measured max unit-norm error | `1.1920928955078125e-07` |

The pinned revision is in the primary run config and cache manifest. The 4,943 distinct passage vectors and question query vectors are cached under the revision-specific cache directory. Passage embeddings are generated independently from passage text; they are shared vector representations, not a pooled graph/index statistic. For every question, the evaluator slices the vectors to that question’s ten context IDs before scoring and graph construction.

## 4. Full 498-question retrieval results

Cells show the question-level mean and 95% percentile bootstrap CI in brackets. Candidate is candidate-generation recall. Complete@k is the proportion of questions for which all gold supporting passages occur in the top k. These results are for the complete 498-question primary sample and are independently reproduced from raw per-question IDs.

| Configuration | Candidate recall | Recall@2 | Recall@5 | Recall@10 | Complete@2 | Complete@5 | Complete@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla TF-IDF | 1.000 [1.000, 1.000] | 0.658 [0.633, 0.683] | 0.836 [0.814, 0.858] | 1.000 [1.000, 1.000] | 0.380 [0.337, 0.422] | 0.689 [0.647, 0.729] | 1.000 [1.000, 1.000] |
| Vanilla BGE | 1.000 [1.000, 1.000] | 0.748 [0.723, 0.773] | 0.906 [0.888, 0.923] | 1.000 [1.000, 1.000] | 0.528 [0.484, 0.570] | 0.815 [0.781, 0.847] | 1.000 [1.000, 1.000] |
| Entity sentence frequency | 0.984 [0.976, 0.991] | 0.748 [0.723, 0.772] | 0.906 [0.888, 0.923] | 0.984 [0.976, 0.991] | 0.528 [0.484, 0.572] | 0.815 [0.781, 0.847] | 0.968 [0.952, 0.982] |
| Entity passage frequency | 0.984 [0.976, 0.991] | 0.748 [0.723, 0.773] | 0.906 [0.888, 0.923] | 0.984 [0.976, 0.991] | 0.528 [0.484, 0.572] | 0.815 [0.779, 0.849] | 0.968 [0.952, 0.982] |
| Entity sentence binary | 0.984 [0.976, 0.991] | 0.748 [0.724, 0.772] | 0.906 [0.888, 0.923] | 0.984 [0.976, 0.991] | 0.528 [0.484, 0.570] | 0.815 [0.779, 0.849] | 0.968 [0.952, 0.982] |
| Entity sentence strongest half | 0.984 [0.976, 0.991] | 0.748 [0.723, 0.773] | 0.906 [0.888, 0.924] | 0.984 [0.976, 0.991] | 0.528 [0.484, 0.572] | 0.815 [0.779, 0.849] | 0.968 [0.952, 0.982] |
| BGE chunk union-kNN k=2 | 0.997 [0.993, 1.000] | 0.748 [0.724, 0.772] | 0.906 [0.888, 0.923] | 0.997 [0.993, 1.000] | 0.528 [0.484, 0.572] | 0.815 [0.781, 0.849] | 0.994 [0.986, 1.000] |

All 95% intervals above use 10,000 question bootstrap replicates, percentile bounds at 2.5% and 97.5%. Full-precision means and bounds are recorded in `primary498_final/retrieval_summary.csv`. The displayed table rounds to three decimals; it does not replace the full-precision CSV.

## 5. Paired comparisons and multiplicity correction

Each paired test uses the per-question difference for the two named methods and metric. The test is a two-sided sign-flip Monte Carlo randomization test with 20,000 draws and the plus-one p-value correction. Holm’s step-down correction is applied over the 69 pre-specified full-run method/outcome contrasts saved in `paired_comparisons.csv`; the independent audit reproduced this file. The direction is **left method minus right method**. A negative value therefore means the right-hand method has the higher mean.

| Left minus right | Metric | Mean difference | Raw p | Holm p |
|---|---|---:|---:|---:|
| vanilla_tfidf − vanilla_bge | Recall@2 | -0.090361 | 0.000050 | 0.003450 |
| vanilla_tfidf − vanilla_bge | Recall@5 | -0.069277 | 0.000050 | 0.003450 |
| vanilla_tfidf − vanilla_bge | Recall@10 | 0.000000 | 1.000000 | 1.000000 |
| vanilla_tfidf − vanilla_bge | Complete@2 | -0.148594 | 0.000050 | 0.003450 |
| vanilla_tfidf − vanilla_bge | Complete@5 | -0.126506 | 0.000050 | 0.003450 |
| vanilla_tfidf − vanilla_bge | Complete@10 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence frequency − vanilla BGE | Candidate recall | -0.016064 | 0.000050 | 0.003450 |
| Entity sentence frequency − vanilla BGE | Recall@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence frequency − vanilla BGE | Recall@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence frequency − vanilla BGE | Recall@10 | -0.016064 | 0.000050 | 0.003450 |
| Entity sentence frequency − vanilla BGE | Complete@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence frequency − vanilla BGE | Complete@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence frequency − vanilla BGE | Complete@10 | -0.032129 | 0.000100 | 0.005800 |
| Entity passage frequency − vanilla BGE | Candidate recall | -0.016064 | 0.000100 | 0.005800 |
| Entity passage frequency − vanilla BGE | Recall@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity passage frequency − vanilla BGE | Recall@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity passage frequency − vanilla BGE | Recall@10 | -0.016064 | 0.000100 | 0.005800 |
| Entity passage frequency − vanilla BGE | Complete@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity passage frequency − vanilla BGE | Complete@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity passage frequency − vanilla BGE | Complete@10 | -0.032129 | 0.000050 | 0.003450 |
| Entity sentence binary − vanilla BGE | Candidate recall | -0.016064 | 0.000050 | 0.003450 |
| Entity sentence binary − vanilla BGE | Recall@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence binary − vanilla BGE | Recall@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence binary − vanilla BGE | Recall@10 | -0.016064 | 0.000050 | 0.003450 |
| Entity sentence binary − vanilla BGE | Complete@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence binary − vanilla BGE | Complete@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence binary − vanilla BGE | Complete@10 | -0.032129 | 0.000100 | 0.005800 |
| Entity sentence strongest half − vanilla BGE | Candidate recall | -0.016064 | 0.000100 | 0.005800 |
| Entity sentence strongest half − vanilla BGE | Recall@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence strongest half − vanilla BGE | Recall@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence strongest half − vanilla BGE | Recall@10 | -0.016064 | 0.000050 | 0.003450 |
| Entity sentence strongest half − vanilla BGE | Complete@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence strongest half − vanilla BGE | Complete@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence strongest half − vanilla BGE | Complete@10 | -0.032129 | 0.000050 | 0.003450 |
| Chunk kNN − vanilla BGE | Candidate recall | -0.003012 | 0.255287 | 1.000000 |
| Chunk kNN − vanilla BGE | Recall@2 | 0.000000 | 1.000000 | 1.000000 |
| Chunk kNN − vanilla BGE | Recall@5 | 0.000000 | 1.000000 | 1.000000 |
| Chunk kNN − vanilla BGE | Recall@10 | -0.003012 | 0.247138 | 1.000000 |
| Chunk kNN − vanilla BGE | Complete@2 | 0.000000 | 1.000000 | 1.000000 |
| Chunk kNN − vanilla BGE | Complete@5 | 0.000000 | 1.000000 | 1.000000 |
| Chunk kNN − vanilla BGE | Complete@10 | -0.006024 | 0.252637 | 1.000000 |
| Entity passage frequency − entity sentence frequency | Candidate recall | 0.000000 | 1.000000 | 1.000000 |
| Entity passage frequency − entity sentence frequency | Recall@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity passage frequency − entity sentence frequency | Recall@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity passage frequency − entity sentence frequency | Recall@10 | 0.000000 | 1.000000 | 1.000000 |
| Entity passage frequency − entity sentence frequency | Complete@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity passage frequency − entity sentence frequency | Complete@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity passage frequency − entity sentence frequency | Complete@10 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence binary − entity sentence frequency | Candidate recall | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence binary − entity sentence frequency | Recall@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence binary − entity sentence frequency | Recall@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence binary − entity sentence frequency | Recall@10 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence binary − entity sentence frequency | Complete@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence binary − entity sentence frequency | Complete@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence binary − entity sentence frequency | Complete@10 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence strongest half − entity sentence frequency | Candidate recall | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence strongest half − entity sentence frequency | Recall@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence strongest half − entity sentence frequency | Recall@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence strongest half − entity sentence frequency | Recall@10 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence strongest half − entity sentence frequency | Complete@2 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence strongest half − entity sentence frequency | Complete@5 | 0.000000 | 1.000000 | 1.000000 |
| Entity sentence strongest half − entity sentence frequency | Complete@10 | 0.000000 | 1.000000 | 1.000000 |
| Chunk kNN − entity sentence frequency | Candidate recall | +0.013052 | 0.004650 | 0.237138 |
| Chunk kNN − entity sentence frequency | Recall@2 | 0.000000 | 1.000000 | 1.000000 |
| Chunk kNN − entity sentence frequency | Recall@5 | 0.000000 | 1.000000 | 1.000000 |
| Chunk kNN − entity sentence frequency | Recall@10 | +0.013052 | 0.004250 | 0.225239 |
| Chunk kNN − entity sentence frequency | Complete@2 | 0.000000 | 1.000000 | 1.000000 |
| Chunk kNN − entity sentence frequency | Complete@5 | 0.000000 | 1.000000 | 1.000000 |
| Chunk kNN − entity sentence frequency | Complete@10 | +0.026104 | 0.004550 | 0.236588 |

There are 16 Holm-adjusted p-values below .05 in the primary family; they are the four early-cutoff TF-IDF/BGE differences and the candidate-recall, Recall@10, and Complete@10 losses for each of the four entity graph variants relative to vanilla BGE. There are no significant chunk-versus-BGE or graph-construction-variant differences after correction. The pilot’s 69-comparison family had zero Holm-significant results. Non-significance in this 50-question pilot is not evidence of equivalence. The complete underlying comparison CSVs retain exact precision.

## 6. Graph structure

The table reports primary-run means and 95% question-bootstrap intervals, computed from the per-question serialized graphs. Density is undirected NetworkX density; components and isolates are graph counts. Entity node counts are entity surface-form nodes; chunk graphs have ten passage nodes by design. Vanilla methods have no graph and are not applicable.

| Graph configuration | Nodes | Edges | Density | Connected components | Isolates | Mean graph build seconds/question |
|---|---:|---:|---:|---:|---:|---:|
| Entity sentence frequency | 74.412 [72.281, 76.502] | 156.221 [148.261, 164.349] | 0.05904 [0.05698, 0.06116] | 15.510 [14.930, 16.112] | 5.060 [4.803, 5.333] | 0.000219 |
| Entity passage frequency | 74.412 [72.313, 76.576] | 499.217 [470.511, 530.950] | 0.17273 [0.16894, 0.17668] | 3.169 [2.974, 3.368] | 0.110 [0.078, 0.149] | 0.000401 |
| Entity sentence binary | 74.412 [72.289, 76.548] | 156.221 [148.779, 164.282] | 0.05904 [0.05701, 0.06114] | 15.510 [14.920, 16.122] | 5.060 [4.805, 5.325] | 0.000220 |
| Entity sentence strongest half | 74.412 [72.283, 76.574] | 78.351 [74.387, 82.502] | 0.02967 [0.02866, 0.03071] | 29.452 [28.510, 30.414] | 21.904 [21.155, 22.655] | 0.000383 |
| BGE chunk union-kNN k=2 | 10.000 [10.000, 10.000] | 14.500 [14.394, 14.608] | 0.32222 [0.31981, 0.32459] | 1.100 [1.074, 1.129] | 0.000 [0, 0] | 0.000072 |

The density of the chunk graph is not directly comparable to the entity graph density because their node universes differ. Passage-level co-occurrence creates more edges and fewer components than sentence-level co-occurrence in this implementation. Strongest-half sparsification retains all nodes but roughly halves sentence-level edges, increasing fragmentation and isolates.

## 7. Pilot versus full run

The pilot is a deterministic stratified sample of 50 primary questions (41 bridge, 9 comparison; selection seed 2027), with 350 rows. The full run includes all 498 primary questions, with 3,486 rows. Both have seven rows per selected question and independently audited zero leakage/reconstruction violations. The table compares question-level means; pilot 95% CIs are included to show pilot precision. Full-run point estimates and CIs are in Section 4.

| Configuration | Pilot Candidate | Pilot R@2 | Pilot R@5 | Pilot R@10 | Pilot Complete@2 | Pilot Complete@5 | Pilot Complete@10 | Full candidate | Full R@2 | Full R@5 | Full R@10 | Full complete@2 | Full complete@5 | Full complete@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla TF-IDF | 1.000 | 0.640 [0.560, 0.720] | 0.870 [0.810, 0.930] | 1.000 | 0.340 [0.220, 0.480] | 0.740 [0.620, 0.860] | 1.000 | 1.000 | 0.658 | 0.836 | 1.000 | 0.380 | 0.689 | 1.000 |
| Vanilla BGE | 1.000 | 0.670 [0.580, 0.760] | 0.870 [0.800, 0.930] | 1.000 | 0.440 [0.300, 0.580] | 0.760 [0.640, 0.860] | 1.000 | 1.000 | 0.748 | 0.906 | 1.000 | 0.528 | 0.815 | 1.000 |
| Entity sentence frequency | 0.970 [0.930, 1.000] | 0.670 [0.580, 0.760] | 0.870 [0.800, 0.930] | 0.970 [0.930, 1.000] | 0.440 [0.300, 0.580] | 0.760 [0.640, 0.880] | 0.940 [0.860, 1.000] | 0.984 | 0.748 | 0.906 | 0.984 | 0.528 | 0.815 | 0.968 |
| Entity passage frequency | 0.970 [0.930, 1.000] | 0.670 [0.580, 0.760] | 0.870 [0.800, 0.930] | 0.970 [0.930, 1.000] | 0.440 [0.300, 0.580] | 0.760 [0.640, 0.880] | 0.940 [0.860, 1.000] | 0.984 | 0.748 | 0.906 | 0.984 | 0.528 | 0.815 | 0.968 |
| Entity sentence binary | 0.970 [0.930, 1.000] | 0.670 [0.580, 0.760] | 0.870 [0.800, 0.930] | 0.970 [0.930, 1.000] | 0.440 [0.300, 0.580] | 0.760 [0.640, 0.880] | 0.940 [0.860, 1.000] | 0.984 | 0.748 | 0.906 | 0.984 | 0.528 | 0.815 | 0.968 |
| Entity sentence strongest half | 0.970 [0.930, 1.000] | 0.670 [0.580, 0.760] | 0.870 [0.800, 0.930] | 0.970 [0.930, 1.000] | 0.440 [0.300, 0.580] | 0.760 [0.640, 0.880] | 0.940 [0.860, 1.000] | 0.984 | 0.748 | 0.906 | 0.984 | 0.528 | 0.815 | 0.968 |
| BGE chunk union-kNN k=2 | 1.000 | 0.670 [0.580, 0.760] | 0.870 [0.800, 0.930] | 1.000 | 0.440 [0.300, 0.580] | 0.760 [0.640, 0.880] | 1.000 | 0.997 | 0.748 | 0.906 | 0.997 | 0.528 | 0.815 | 0.994 |

Pilot and full data have the same qualitative pattern: BGE exceeds TF-IDF at the early cutoffs; graph-assisted systems preserve BGE’s top-five ranking where candidates survive, because they use the same dense ranking; entity expansion loses some candidate recall and complete support at ten; the chunk graph retains more candidate evidence. Pilot estimates are noisier, and the full-run changes are plausible under sampling variability. The full pilot/full overlap comparison is not an independent replication because the pilot questions are included in the full 498.

## 8. Runtime and memory

Primary-run `efficiency.csv` reports **133.513017917 seconds** total scope runtime and process peak RSS **1,231.203125 MB**. The canonical pilot run reports **1.521263958 seconds** and peak RSS **672.890625 MB**; it used all 500 passage and 50 query vectors from cache and ran after local model loading. The primary timer starts after input JSON loading and input validation. It includes extraction of 4,443 uncached passages (50.4401 s), BGE model load (0.1171 s), computation of 4,443 new passage embeddings (71.2491 s), 448 new query embeddings (1.2048 s), per-question graph construction/evaluation for 3,486 rows, summaries, and output writing. It reused 500 passage vectors and 50 query vectors from the pilot cache. An additional paired-comparison family recomputation of 3.568 seconds is reported in the validation record and occurred separately after the run; it is not part of 133.513 seconds.

The total excludes loading and validation of the input JSON, model download, the earlier pilot cache warm-up, and the later independent audit. The row field `query_retrieval_seconds` starts with cached vectors and excludes embedding and graph building; graph construction, candidate generation, dense scoring, reranking, and local TF-IDF fit/ranking have separate timing fields. It is therefore not an end-to-end timing. The run is a warm local model CPU measurement from the recorded machine/software environment. Peak RSS is the process high-water mark, not system-wide memory. The mean graph build times are shown in Section 6; extremely short per-question graph-build time does not include the entity extraction prerequisite or dense passage embedding.

## 9. Independent audit and observed violations

The independent audit is implemented in `src/phase_b_independent_audit.py`. The audit intentionally does not import the experiment’s graph construction, metric aggregation, or statistical helper implementations. It reads the selected question data, extraction cache, pinned BGE embedding cache, and saved raw JSONL; rebuilds local entity/chunk structures, candidates, rankings, denominators, retrieval metrics, graph statistics, bootstrap summaries, paired tests/Holm adjustment, and cross-configuration overlap summaries; and compares rebuilt outputs to saved CSVs.

| Audit check | Pilot 50 | Primary 498 |
|---|---:|---:|
| Raw rows | 350 | 3,486 |
| Rows per method | 50 each | 498 each |
| Candidate passage IDs outside question contexts | 0 | 0 |
| Ranked passage IDs outside question contexts | 0 | 0 |
| Graph node scope violations | 0 | 0 |
| Entity/chunk graph reconstruction mismatches | 0 | 0 |
| Candidate reconstruction mismatches | 0 | 0 |
| Ranking reconstruction mismatches | 0 | 0 |
| Retrieval metric / gold-denominator mismatches | 0 | 0 |
| Graph-statistic mismatches | 0 | 0 |
| Recomputed summary CSVs matching | 5 of 5 | 5 of 5 |
| Audit status | passed | passed |

For the primary dataset, each graph’s node/edge set was reconstructed only from that question’s own ten contexts, and BGE rankings were reconstructed from the cached vectors. Every primary passage candidate and ranked passage belongs to that question’s own context. The precise observed cross-question graph-node/edge/passage-candidate leakage count is **zero**. Shared entity surface forms across different question graphs do not imply shared edges or imported passage mappings. The independent maximum vector-norm deviation was `1.1920928955078125e-07`. The full audit JSON has no failures and says retrieval, graph, paired comparison, and stability CSVs match recomputation.

There were earlier preserved noncanonical artifacts. `pilot50/` contains a stale failure record from the initial offline model-load attempt and an unpinned revision; `pilot50_verified/` contains a superseded run whose earlier audit found 50 tie-dependent sparsified-edge mismatches. The deterministic lexical tie rule was applied in `pilot50_final/`, which passed independent audit. Those old records are not used for results and were not removed or rewritten.

## 10. Tests and execution commands

The exact full-suite command was run during this review:

```text
.venv/bin/python -m unittest discover -v
```

Actual result: **Ran 24 tests in 0.042s — OK** (24 passed, 0 failures, 0 errors). The run printed a non-fatal PyTorch macOS redirect note and a Matplotlib cache permission warning; Matplotlib used a temporary cache. The test names and groups are recorded in `results/dense_graph_v1/PHASE_B_VALIDATION.md`; they cover question-local graph edges/candidates and context guards, exact-ID evidence metrics and denominators, bootstrap/sign-flip/Holm logic, deterministic sparsification, pilot sampling, Stage A context/evidence validation, graph variants, and structural perturbation metrics. The validation script commands recorded in the run report are:

```text
.venv/bin/python -m src.evaluate_dense_graph --scope pilot50 --run-name pilot50_final --output-dir results/dense_graph_v1
.venv/bin/python -m src.evaluate_dense_graph --scope primary498 --run-name primary498_final --output-dir results/dense_graph_v1
.venv/bin/python -m src.phase_b_independent_audit results/dense_graph_v1/pilot50_final --output results/dense_graph_v1/phase_b_validation_audits/pilot50_independent.json
.venv/bin/python -m src.phase_b_independent_audit results/dense_graph_v1/primary498_final --output results/dense_graph_v1/phase_b_validation_audits/primary498_independent.json
.venv/bin/python -m unittest discover -v
```

## 11. Stability and extraction-quality limitations

No same-configuration repeated-seed runs were performed. The sample-selection seed (`2027`) selects the pilot; the bootstrap RNG resamples questions. Neither creates repeated graph/retrieval realizations. The pipeline uses passage-local spaCy extraction, deterministic BGE CPU embeddings, deterministic graph rules, and lexical tie-breaking. The saved Jaccard table compares top-k results across different configurations; it is explicitly labeled cross-configuration overlap, not repeated-seed stability. Thus retrieval stability under seeds or stochastic graph construction remains unmeasured.

Entity extraction quality was not measured against manually labeled entities. A diagnostic found exact agreement for 4,876 of 4,876 legacy mapping entries overlapping the primary context corpus, while 67 of the 4,943 context passages had no legacy mapping. That mapping’s extractor provenance is not established, and agreement between automatic outputs is not human ground truth. No entity precision, recall, alias-resolution accuracy, or manual error analysis can be claimed.

## 12. Requirements still uncompleted

The following Professor De Meo requirements remain outside the executed Phase B package or only partially covered:

| Requirement | Status after Phase B |
|---|---|
| Compare TF-IDF and BGE-small-en-v1.5 | Completed for question-local retrieval on the 498 primary questions. |
| Graph construction sensitivity: entity, chunk, semantics, weighting, sparsification | Partially completed by the four entity variants and one chunk semantic kNN graph. Missing passage-level binary weighting; other sparsification fractions/strategies and controls; sentence-node graph; entity semantic-similarity graph; entity-passage bipartite graph; alias/entity-resolution ablation. |
| Typed relation graphs | Deferred; no suitable relation extraction was implemented and manually validated. |
| Local community retrieval | Not implemented in Phase B. The methods use seed-based one-hop candidate expansion, not community retrieval. |
| Multiple seeds / stability | Not completed. No repeated same-configuration seed runs; question bootstrap is uncertainty estimation, not run stability. |
| Question-level intervals and paired inference | Completed for this single retrieval set with bootstrap, sign-flip tests and Holm correction. This does not replace repeated-run stability analysis. |
| Answer Exact Match and token F1 | Not run; no answer generation. |
| Evidence faithfulness and manually reviewed subset | Not run. |
| Vanilla dense RAG and an established GraphRAG system (LightRAG or Microsoft GraphRAG) | Vanilla BGE retrieval is present; generation pipeline and established GraphRAG integration are not. |
| Fixed generation model/prompt/decoding/evidence-token budget | Not applicable yet; must be fixed before a future answer evaluation. |
| Broader 500–1,000 question sample | Current validated set is 498 primary questions from one fixed all-hard sample; no expanded sample or generalization claim. |
| Contriever | Not evaluated; it remains conditional on justification after BGE. |
| Entity extractor human quality validation | Not done. |
| Final research package requirements | Final dedicated results collection, academic figures, updated setup guidance, seven-slide faculty presentation, and formal progress-email draft remain outstanding. |

The current evidence supports a retrieval-only finding on this fixed sample: BGE improved early-cutoff retrieval over local TF-IDF; the entity candidate-expansion methods retained the same BGE ordering among surviving candidates but reduced candidate recall and complete-support retrieval at ten; chunk expansion was closer to vanilla BGE, with no Holm-significant difference from BGE in this comparison family. This is not evidence about generated-answer quality or a completed GraphRAG system.

## 13. Output inventory

The table inventories every file under `results/dense_graph_v1/` as present when this report was prepared. `pilot50_final/` and `primary498_final/` are the canonical, audited runs. Other directories and cache revisions are retained historical/intermediate artifacts and must not be pooled into canonical findings.

| Directory / filename(s) | Purpose |
|---|---|
| `results/dense_graph_v1/PHASE_B_VALIDATION.md` | Detailed phase validation, result tables, methodology, runtime, limitations, and reproduction information. |
| `results/dense_graph_v1/PILOT_VALIDATION.md` | Earlier pilot validation report; use final pilot outputs for numbers. |
| `results/dense_graph_v1/phase_b_validation_audits/pilot50_independent.json` | Independent recomputation audit result for canonical pilot. |
| `results/dense_graph_v1/phase_b_validation_audits/primary498_independent.json` | Independent recomputation audit result for canonical primary run. |
| `results/dense_graph_v1/cache/entity_extractions.jsonl` | Per-passage cached spaCy sentence/entity extractions with source hashes and model provenance. |
| `results/dense_graph_v1/cache/embeddings/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/passages.json`, `passages.npz` | Pinned-revision passage embedding cache and manifest. |
| `results/dense_graph_v1/cache/embeddings/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/queries.json`, `queries.npz` | Pinned-revision query embedding cache and manifest. |
| `results/dense_graph_v1/cache/embeddings/unknown/passages.json`, `passages.npz` | Retained old unpinned pilot passage cache; not used for canonical findings. |
| `results/dense_graph_v1/cache/embeddings/unknown/queries.json`, `queries.npz` | Retained old unpinned pilot query cache; not used for canonical findings. |
| `results/dense_graph_v1/pilot50_final/question_results.jsonl` | Canonical raw pilot rows: 350 question-method records, including graph details and timings. |
| `results/dense_graph_v1/primary498_final/question_results.jsonl` | Canonical raw primary rows: 3,486 question-method records. |
| `pilot50_final/retrieval_summary.csv`, `primary498_final/retrieval_summary.csv` | Per-method means and bootstrap CIs for candidate recall, Recall@k, complete support, candidate counts, and timing summaries. |
| `pilot50_final/paired_comparisons.csv`, `primary498_final/paired_comparisons.csv` | All paired tests and Holm-adjusted p-values (69 primary contrasts). |
| `pilot50_final/graph_summary.csv`, `primary498_final/graph_summary.csv` | Bootstrap summary for node/edge counts, density, connected components, isolates, and graph build time. |
| `pilot50_final/efficiency.csv`, `primary498_final/efficiency.csv` | Stage runtime, item counts, and peak process RSS. |
| `pilot50_final/retrieval_stability_summary.csv`, `primary498_final/retrieval_stability_summary.csv` | Aggregated cross-configuration top-k overlap, not repeated-seed stability. |
| `pilot50_final/retrieval_stability_pairwise.csv`, `primary498_final/retrieval_stability_pairwise.csv` | Per-question cross-configuration Jaccard rows. |
| `pilot50_final/run_config.json`, `primary498_final/run_config.json` | Data scope, configurations, model revision, seeds, software/resource settings, timings, and validation metadata. |
| `pilot50_final/selected_questions.json`, `primary498_final/selected_questions.json` | Exact question IDs used in each run. |
| `pilot50_final/validation_checks.json`, `primary498_final/validation_checks.json` | Evaluator-side counts and scope guards for each canonical run. |
| `pilot50_final/independent_audit.json`, `primary498_final/independent_audit.json` | Audit records saved with canonical run directories; the later dedicated audit JSONs above are the independent review records cited here. |
| `results/dense_graph_v1/pilot50/` (11 files: `efficiency.csv`, `failure.json`, `graph_summary.csv`, `paired_comparisons.csv`, `question_results.jsonl`, `retrieval_stability_pairwise.csv`, `retrieval_stability_summary.csv`, `retrieval_summary.csv`, `run_config.json`, `selected_questions.json`, `validation_checks.json`) | Preserved early run artifacts, including offline-load failure and unpinned revision metadata; excluded from findings. |
| `results/dense_graph_v1/pilot50_verified/` (11 files: `efficiency.csv`, `graph_summary.csv`, `independent_audit.json`, `paired_comparisons.csv`, `question_results.jsonl`, `retrieval_stability_pairwise.csv`, `retrieval_stability_summary.csv`, `retrieval_summary.csv`, `run_config.json`, `selected_questions.json`, `validation_checks.json`) | Preserved superseded pilot; earlier nondeterministic sparsification tie handling yielded 50 edge mismatches; excluded from findings. |

Source and tests relevant to review are `src/evaluate_dense_graph.py`, `src/phase_b_independent_audit.py`, `src/evaluate_question_local.py`, `tests/test_dense_graph_pilot.py`, `tests/test_graph_construction_variants.py`, `tests/test_question_local.py`, `tests/test_seed_stability.py`, and `tests/test_structural_metrics.py`. The exact file paths and source text remain in the repository; this summary does not modify them.

## 14. Git checkpoint and working tree

At final verification, `HEAD` was **`a37a1811c7484d481d60a0c8b606af1362def5d6`** on branch `main`. The exact `git status --short --branch` output after creating this report is:

```text
## main
?? PHASE_B_REVIEW_SUMMARY.md
?? stage_a_review_bundle.zip
```

The only untracked paths are this new review summary and the pre-existing Stage A review bundle. This request did not ask for a commit; the hash refers to the existing HEAD, so it does not include this document.
