# Graph Construction Sensitivity and Evidence Retrieval

An empirical research repository studying how graph construction choices affect evidence retrieval for multi-hop question answering. The repository contains historical pooled-corpus experiments, a validated question-local TF-IDF baseline (Stage A), an independently audited BGE and question-local graph comparison (Phase B), and a pre-registered study of graph-aware ranking on HotpotQA and MuSiQue, with controls (Phase C).

> **Research status:** Retrieval experiments are complete for the saved 498-question HotpotQA sample (Stages A–B) and for the Phase C graph-aware ranking study on HotpotQA and two disjoint 500-question MuSiQue samples ([results](RESULTS_graph_ranking.md)). Generated answers, answer accuracy, faithfulness, and an established GraphRAG system have not been evaluated. This is not a completed GraphRAG architecture and does not establish broad generalization beyond the sampled questions.

## Research motivation and question

Multi-hop QA requires retrieving evidence from more than one document. Graph-based retrieval depends on choices such as node type, edge definition, weighting, and sparsification. This project measures how those choices affect whether gold supporting passages remain in a question’s retrieval candidate set and top-k results.

**Central question:** How do graph construction choices affect evidence retrieval for multi-hop questions, and what additional work is needed to connect retrieval robustness to generated-answer reliability?

Objectives:

1. Establish lexical and dense passage-retrieval references under controlled context limits.
2. Compare question-local entity co-occurrence and chunk similarity graph candidate generation.
3. Report passage recall, complete supporting-evidence retrieval, candidate recall, graph structure, uncertainty, paired inference, and efficiency.
4. Preserve historical pooled-corpus findings separately from question-local findings.
5. Identify the evidence required before drawing conclusions about answer quality or faithfulness.

## Experimental architecture

The repository has four distinct experiment families:

| Family | Data and graph scope | Purpose | Status |
|---|---|---|---|
| Historical pooled-corpus studies | Earlier 15-, 50-, and 500-question data; a single graph/partition may pool passages across questions | Leiden/Louvain communities, perturbations, construction variants, and TF-IDF-seeded retrieval | Exploratory/historical; not question-local distractor evaluation. |
| Stage A | 498 ten-context primary questions from `data/pilot_500_v2`; separately reports two variable-context questions | Local TF-IDF retrieval; secondary shared-global-graph comparison | Completed and validated. |
| Phase B | The same 498 ten-context primary questions; graph and candidates rebuilt independently from each question’s ten contexts | TF-IDF, BGE, four entity graph variants, and BGE chunk kNN graph | Pilot of 50 and full 498 completed; raw-output audit passed. |
| Phase C | HotpotQA 498 + MuSiQue 500 + disjoint MuSiQue replication 500; question-local graphs | BM25, TF-IDF, BGE, personalized-PageRank re-ranking over entity and chunk graphs | Pre-registered; completed; one exploratory finding replicated. |

Graphs and indexes in the primary Phase B question-local track are created from only the current question’s allowed passages. The secondary Stage A shared-global-graph track reuses a pooled graph and corpus TF-IDF statistics, then filters candidates to each question’s contexts; it is explicitly not question-local graph construction.

## Dataset

The experiments use HotpotQA `distractor` validation. The public repository does not redistribute raw questions, benchmark passages, context mappings, or dataset-derived graphs; see [data preparation and licensing](data/README.md). The canonical local dataset had 500 questions, 4,947 deduplicated passages, 498 questions with ten contexts, and two with two contexts. All 500 had gold support in their supplied contexts. The primary 498 are all `hard` (404 bridge, 94 comparison); conclusions are limited to this fixed sample.

Raw question-level retrieval records are also not redistributed. The published aggregates can be inspected, but complete metric recomputation requires independently obtaining and preparing HotpotQA as described in `data/README.md`, then rerunning the documented evaluation. The public checkout alone is not sufficient to recompute every reported metric from raw question-level records.

Phase C also uses MuSiQue (answerable v1.0 dev, from `bdsaglam/musique` at the pinned revision recorded in `data/musique_500/dataset_manifest.json`): two disjoint 500-question samples (seeds 2027 and 2028) drawn from questions with exactly 20 paragraphs, 2–4 hops. MuSiQue is released under CC BY 4.0 and its paragraphs are Wikipedia-derived; raw MuSiQue questions and passages are not redistributed here either, and `python -m src.load_musique` regenerates them (the second sample uses `--seed 2028 --exclude-questions-from data/musique_500/questions.json`).

The upstream HotpotQA dataset card identifies the dataset as CC BY-SA 4.0. Review the dataset and source-content terms before distributing data or derived caches. The code repository itself currently has **no project-level `LICENSE`**; no license has been selected for original code. Model weights are not included; Phase B fetches the pinned BGE model snapshot when needed.

## Methods and configurations

### Stage A

The primary Stage A track fits a fresh TF-IDF vectorizer per question over its ten allowed passage title/text fields and ranks only those ten passages. The separate secondary track starts from local TF-IDF seeds, expands them using the historical pooled entity graph, filters candidates to the question’s contexts, then reranks with corpus-wide TF-IDF statistics. Its graph and inverse-document-frequency information are shared across questions.

For the 498 primary questions, local TF-IDF achieved Recall@2 **0.658** (95% question-bootstrap CI 0.632–0.683), Recall@5 **0.836** (0.814–0.858), Recall@10 **1.000**, and complete gold-support retrieval at 5 **0.689** (0.649–0.729). The shared-global-graph track achieved Recall@2 **0.565** (0.538–0.591), Recall@5 **0.733** (0.708–0.758), and complete support@5 **0.506** (0.462–0.550). The two two-context questions are reported separately in the Stage A outputs. These summary CIs are question-level percentile bootstrap intervals.

### Phase B

The dense model is `BAAI/bge-small-en-v1.5`, pinned in code to revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, with 384-dimensional L2-normalized embeddings. Similarity is dot product of normalized vectors (cosine-equivalent). Entity graphs use sentence-level or passage-level co-occurrence; sentence edges are tested with frequency weights, binary weights, and a deterministic strongest-half sparsification. The chunk graph connects each passage node to its two nearest BGE neighbors using an undirected union-kNN rule.

| Configuration | Recall@2 | Recall@5 | Candidate-generation recall | Complete support@10 |
|---|---:|---:|---:|---:|
| Local TF-IDF | 0.658 | 0.836 | 1.000 | 1.000 |
| Vanilla BGE | 0.748 | 0.906 | 1.000 | 1.000 |
| Entity sentence, frequency weighted | 0.748 | 0.906 | 0.984 | 0.968 |
| Entity passage, frequency weighted | 0.748 | 0.906 | 0.984 | 0.968 |
| Entity sentence, binary | 0.748 | 0.906 | 0.984 | 0.968 |
| Entity sentence, strongest half | 0.748 | 0.906 | 0.984 | 0.968 |
| BGE chunk union-kNN, k=2 | 0.748 | 0.906 | 0.997 | 0.994 |

The means are over 498 questions. Graph methods use BGE’s top five passages as seeds and BGE scores to rerank graph-generated candidates. Thus, the graphs affect candidate generation; they do not add a graph score or community-ranking stage. Detailed results and confidence intervals are in [the Phase B validation report](results/dense_graph_v1/PHASE_B_VALIDATION.md) and the [consolidated research review](docs/research/PHASE_B_REVIEW_SUMMARY.md).

## Phase C: graph-aware ranking (pre-registered)

Phase B's graphs could only add candidates that BGE then re-ranked, so BGE's top 5 could not change. Phase C lets the graph change the score: `final = 0.5·minmax(BGE) + 0.5·minmax(PPR)`, where PPR is personalized PageRank (α = 0.85) seeded on BGE's top 5, over a question-local passage–entity graph or the Phase B chunk 2-NN graph. The design is fixed in [PREREGISTRATION_graph_ranking.md](PREREGISTRATION_graph_ranking.md), and the full write-up is [RESULTS_graph_ranking.md](RESULTS_graph_ranking.md). On HotpotQA, BGE reproduces Phase B exactly.

Complete support@5 (all gold passages in top 5), mean over questions; CIs and all contrasts are in RESULTS_graph_ranking.md:

| Method | HotpotQA (n=498) | MuSiQue (n=500) | MuSiQue replication (n=500) |
|---|---:|---:|---:|
| BM25 | 0.612 | 0.170 | 0.170 |
| BGE (dense) | 0.815 | 0.350 | 0.358 |
| BGE + entity-graph PPR (pre-registered) | 0.815 (Holm p = 1.00) | 0.350 (Holm p = 1.00) | 0.358 |
| BGE + chunk-graph PPR (pre-registered) | 0.807 (Holm p = 1.00) | 0.376 (Holm p = 0.040) | 0.374 (not replicated, Holm p = 0.34) |
| BGE + entity-graph leave-one-out PPR (exploratory → replicated) | 0.787 | 0.468 | **0.482** (Δ +0.124 [+0.082, +0.168], Holm p = 0.0002) |

Pre-registered result: neither graph method improves on BGE by the decision rule, apart from a small MuSiQue chunk-graph gain that did not replicate. Entity-graph PPR never changed BGE's top 5 on the 998 headline questions. A leave-one-out entity-graph variant, which scores each passage by the support it receives from the *other* top passages, was added before the full runs after a smoke check of top-5 change counts. It improved MuSiQue complete support@5 by about 12 points, and this replicated on a disjoint, pre-registered sample. The same variant did not help on HotpotQA (−0.028, n.s.) and hurt comparison questions (−0.202). Pre-registered controls (PREREGISTRATION_controls.md): a hybrid BGE+BM25 retriever with a second-hop query did not reproduce the MuSiQue gain (0.324 vs BGE 0.354), nor did a dense-only second-hop control (0.369; PREREGISTRATION_controls_v2.md), and the gain survived removing title nodes (0.470 vs 0.475 with titles). BGE headroom explains part of the difference: Recall@10 is 1.000 on HotpotQA and 0.843 on MuSiQue. Figures are in `results/graph_ranking_v1/figures/final/`.

Reproduce (writes to new directories; the evaluator refuses to overwrite):

```bash
python -m src.load_musique --split validation --limit 500 --seed 2027 --output-dir data/musique_500
python -m src.evaluate_graph_ranking --data-dir data/pilot_500_v2 --output-dir results/reproduction/hotpot498
python -m src.evaluate_graph_ranking --data-dir data/musique_500 --output-dir results/reproduction/musique500
```

## Statistics and validation

Question is the independent unit. Stage A and Phase B use 10,000 question bootstrap replicates and percentile 95% intervals. Phase B paired differences use a two-sided sign-flip Monte Carlo test (20,000 draws, plus-one p correction) and Holm correction across 69 pre-specified method/outcome comparisons. Stage A reports paired comparisons separately by context stratum.

The independent Phase B audit was run against local raw per-question records and caches, reconstructing question-local graphs, candidates, rankings, metrics, structural statistics, and inference. It found zero out-of-context candidates/rankings, graph-scope violations, or metric/denominator mismatches in either run. Aggregate audit reports are included; the raw inputs used to recompute them are excluded from this release candidate. The full suite contains 46 tests (25 from Stages A–B, 16 Phase C tests for context scoping, PPR, metrics, paired statistics and the MuSiQue loader, and 5 for the Phase C-2/C-3 controls), including a guard for the pinned model revision.

## Limitations and future work

- The sample is fixed and all-hard; it does not support population-wide claims.
- The Stage A shared-global-graph comparison shares graph and TF-IDF corpus statistics across questions; it is not a local graph condition.
- Phases B and C use one dense model (BGE-small) and deterministic configurations. It does not measure same-configuration repeated-seed stability.
- spaCy entity quality has not been measured against human labels; agreement with a legacy automatic mapping is not a gold evaluation.
- Passage-level binary weighting, alternate sparsification controls, sentence-node graphs, entity-semantic graphs, alias-resolution ablations, bipartite-graph variants other than Phase C's passage–entity graph, local community retrieval, and validated typed relations remain untested.
- No answer generation, Exact Match, token F1, evidence faithfulness, or manual faithfulness review has been performed.
- LightRAG/Microsoft GraphRAG, Contriever and 2WikiMultiHopQA have not been run.
- The project has no selected code license or published maintainer email. The published Git history has been sanitized; local review bundles are not included in the repository. Dataset and model terms remain separate from any future project code license.
- The dataset and BGE model carry their own upstream terms; do not apply a future code license to them without checking compatibility.

Future research should add controlled repeated runs where a real source of stochasticity exists, validate entity extraction with an annotation protocol, complete a small justified graph matrix, and only then evaluate fixed-model/prompt/decoding answer generation and manually reviewed evidence faithfulness. Evaluate an established GraphRAG baseline after the retrieval pipeline and answer protocol are ready.

## Installation and quick checks

Use Python 3.13 or another version supported by the listed packages. The verified environment used Python 3.13.7 on CPU. The repository’s existing `.venv` is local and ignored; do not copy it between machines.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt   # includes rank-bm25 and datasets, used by Phase C
python -m spacy download en_core_web_sm
python -m unittest discover -v
```

For exact extractor reproduction, install `en_core_web_sm` 3.8.0 compatible with spaCy 3.8.18. Phase B downloads the exact BGE revision on first use. This requires access to Hugging Face and sufficient local disk for the model and generated caches. See [docs/reproduction_guide.md](docs/reproduction_guide.md) for commands that write to fresh output directories.

## Repository map

```text
data/                         Dataset instructions and provenance manifests
docs/                         Methodology, design, reproduction, historical notes
docs/research/                Consolidated Phase B review
results/question_local_v1/    Stage A aggregate outputs and validation
results/dense_graph_v1/       Phase B aggregate outputs, audits, validation
results/graph_ranking_v1/     Phase C aggregate outputs, analysis tables, figures
results/pilot_50/             Historical 50-question graph experiments
results/pilot_500/            Historical pooled 500-question experiments
src/                          Experiment and validation modules
tests/                        Unit tests
```

The public repository keeps aggregate canonical findings while excluding raw per-question outputs and local research inputs. See [the release inventory](PUBLIC_RELEASE_INVENTORY.md) for the published scope. Reproduction commands require a locally obtained HotpotQA sample and the pinned BGE model; the included loader's seed does not guarantee the original sample if upstream data or library behavior has changed.

Do not interpret every file under `results/` as a final result. Canonical runs and preserved failed/superseded artifacts are labelled in [docs/experimental_design.md](docs/experimental_design.md) and the validation reports.

## Citation and contact

If you use the benchmark, cite HotpotQA:

```bibtex
@inproceedings{yang2018hotpotqa,
  title     = {HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author    = {Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle = {Proceedings of EMNLP},
  year      = {2018}
}
```

For this research code, cite the repository and the relevant release commit. No maintainer email is published; use the repository’s [Issues page](https://github.com/shikharrsrivastava/graphrag-construction-sensitivity/issues) for project questions.
