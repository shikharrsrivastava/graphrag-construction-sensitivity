# Graph Construction Sensitivity and Evidence Retrieval

An empirical research repository studying how graph construction choices affect evidence retrieval for multi-hop question answering. The repository contains historical pooled-corpus experiments, a validated question-local TF-IDF baseline (Stage A), and an independently audited BGE and question-local graph comparison (Phase B).

> **Research status:** Retrieval experiments are complete for the saved 498-question primary sample. Generated answers, answer accuracy, faithfulness, and an established GraphRAG system have not been evaluated. This is not a completed GraphRAG architecture and does not establish broad generalization beyond the sampled HotpotQA questions.

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

The repository has three distinct experiment families:

| Family | Data and graph scope | Purpose | Status |
|---|---|---|---|
| Historical pooled-corpus studies | Earlier 15-, 50-, and 500-question data; a single graph/partition may pool passages across questions | Leiden/Louvain communities, perturbations, construction variants, and TF-IDF-seeded retrieval | Exploratory/historical; not question-local distractor evaluation. |
| Stage A | 498 ten-context primary questions from `data/pilot_500_v2`; separately reports two variable-context questions | Local TF-IDF retrieval; secondary shared-global-graph comparison | Completed and validated. |
| Phase B | The same 498 ten-context primary questions; graph and candidates rebuilt independently from each question’s ten contexts | TF-IDF, BGE, four entity graph variants, and BGE chunk kNN graph | Pilot of 50 and full 498 completed; raw-output audit passed. |

Graphs and indexes in the primary Phase B question-local track are created from only the current question’s allowed passages. The secondary Stage A shared-global-graph track reuses a pooled graph and corpus TF-IDF statistics, then filters candidates to each question’s contexts; it is explicitly not question-local graph construction.

## Dataset

The experiments use HotpotQA `distractor` validation. The public release candidate excludes raw questions, passages, context mappings, and dataset-derived graphs; see [data preparation and licensing](data/README.md). The canonical local dataset had 500 questions, 4,947 deduplicated passages, 498 questions with ten contexts, and two with two contexts. All 500 had gold support in their supplied contexts. The primary 498 are all `hard` (404 bridge, 94 comparison); conclusions are limited to this fixed sample.

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

## Statistics and validation

Question is the independent unit. Stage A and Phase B use 10,000 question bootstrap replicates and percentile 95% intervals. Phase B paired differences use a two-sided sign-flip Monte Carlo test (20,000 draws, plus-one p correction) and Holm correction across 69 pre-specified method/outcome comparisons. Stage A reports paired comparisons separately by context stratum.

The independent Phase B audit was run against local raw per-question records and caches, reconstructing question-local graphs, candidates, rankings, metrics, structural statistics, and inference. It found zero out-of-context candidates/rankings, graph-scope violations, or metric/denominator mismatches in either run. Aggregate audit reports are included; the raw inputs used to recompute them are excluded from this release candidate. The full suite contains 25 tests, including a guard for the pinned model revision.

## Limitations and future work

- The sample is fixed and all-hard; it does not support population-wide claims.
- The Stage A shared-global-graph comparison shares graph and TF-IDF corpus statistics across questions; it is not a local graph condition.
- Phase B uses one dense model and deterministic configurations. It does not measure same-configuration repeated-seed stability.
- spaCy entity quality has not been measured against human labels; agreement with a legacy automatic mapping is not a gold evaluation.
- Passage-level binary weighting, alternate sparsification controls, sentence-node graphs, entity-semantic graphs, entity-passage bipartite graphs, alias-resolution ablations, local community retrieval, and validated typed relations remain untested.
- No answer generation, Exact Match, token F1, evidence faithfulness, or manual faithfulness review has been performed.
- LightRAG/Microsoft GraphRAG and Contriever have not been run.
- The project has no selected code license or maintainer email. Reachable historical commits contain workstation paths. An earlier 29 MB review bundle exists in an unreachable local commit and is not part of current `main` history. Review the historical-data/licensing and history-exposure issues before publication.
- The dataset and BGE model carry their own upstream terms; do not apply a future code license to them without checking compatibility.

Future research should add controlled repeated runs where a real source of stochasticity exists, validate entity extraction with an annotation protocol, complete a small justified graph matrix, and only then evaluate fixed-model/prompt/decoding answer generation and manually reviewed evidence faithfulness. Evaluate an established GraphRAG baseline after the retrieval pipeline and answer protocol are ready.

## Installation and quick checks

Use Python 3.13 or another version supported by the listed packages. The verified environment used Python 3.13.7 on CPU. The repository’s existing `.venv` is local and ignored; do not copy it between machines.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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
results/pilot_50/             Historical 50-question graph experiments
results/pilot_500/            Historical pooled 500-question experiments
src/                          Experiment and validation modules
tests/                        Unit tests
```

The public release candidate keeps aggregate canonical findings while excluding raw per-question outputs and local research inputs. See [the release inventory](PUBLIC_RELEASE_INVENTORY.md) for the proposed scope. Reproduction commands require a locally obtained HotpotQA sample and the pinned BGE model; the included loader's seed does not guarantee the original sample if upstream data or library behavior has changed.

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

For this research code, cite the repository and its release commit after a public repository and citation metadata have been established. No public GitHub remote or contact email is configured in this checkout; use the repository’s Issues page after publication for project questions.
