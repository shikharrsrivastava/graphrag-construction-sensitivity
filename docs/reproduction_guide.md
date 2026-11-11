# Reproduction guide

Run commands from the repository root. The project keeps its current `src/`, `data/`, and `results/` paths because scripts resolve defaults relative to the repository root and historical results refer to the existing directory layout.

## What is included and what must be downloaded

- The public release candidate includes dataset provenance manifests and instructions only. Raw sampled HotpotQA questions/passages, context maps, and derived graphs are excluded from its Git index; see [data preparation and licensing](../data/README.md).
- Phase B’s pinned BGE model weights are **not** included. A first Phase B run downloads revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` from Hugging Face and requires network access. The model cache can be sizable.
- `en_core_web_sm` model files are not included in requirements and must be installed separately. The validated extraction used spaCy 3.8.18 and model version 3.8.0.
- The existing `.venv` and local model weights should not be copied to another computer.

Aggregate saved results can be inspected directly. Raw per-question records and extraction/embedding caches are retained only in the maintainer's local research workspace and are not part of the aggregate-only release candidate. To recreate inputs, embeddings, and extraction, you need the upstream dataset, dependencies, spaCy model, pinned BGE snapshot, and enough local disk. The reported run time is machine-specific. A fresh clean-environment installation was not tested in this release-preparation pass; the existing environment was inspected and used for tests/entry-point checks.

## Install

The verified machine used Python 3.13.7 on macOS, CPU execution, and these direct package versions:

| Package | Verified version |
|---|---:|
| datasets | 5.1.0 |
| huggingface-hub | 2.4.0 |
| leidenalg | 0.12.0 |
| matplotlib | 3.11.2 |
| networkx | 3.7 |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| python-igraph | 1.0.0 |
| scikit-learn | 1.9.1 |
| sentence-transformers | 6.2.0 |
| spaCy | 3.8.18 |
| torch | 2.15.0 |

`requirements.txt` lists direct dependencies and compatible lower bounds; it is not a complete lock file. Install into a fresh environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For the validated spaCy model version, verify after installation:

```bash
python -c 'import spacy; nlp=spacy.load("en_core_web_sm"); print(spacy.__version__, nlp.meta["version"])'
```

The recorded validated values are `3.8.18 3.8.0`. If you install another version, record it in any new run config and treat it as a changed extractor configuration.

## Test and import checks

Run the full existing unit suite:

```bash
python -m unittest discover -v
```

The checked-in suite covers question-local candidate guards, exact evidence IDs and denominators, graph variants and leakage, bootstrap/sign-flip/Holm calculations, pilot sampling, structural measures, and the pinned Phase B model revision.

Lightweight entry-point checks:

```bash
python -m src.evaluate_question_local --help
python -m src.evaluate_dense_graph --help
python -m src.phase_b_independent_audit --help
python -m src.validate_dense_graph_run --help
```

## Reproduce Stage A into a new directory

The local canonical output was `results/question_local_v1/`; aggregate summaries and configuration are included, while its raw per-question JSONL files are excluded from this release candidate. Stage A refuses nonempty output directories. Use a new empty local output directory:

```bash
python -m src.evaluate_question_local \
  --data-dir /path/to/local-data/pilot_500_v2 \
  --output-dir results/reproduction/question_local_v1
```

This reads the v2 question/context mappings and the legacy pooled graph/mappings from the repository-relative `data/pilot_500/` path for the explicitly secondary shared-global-graph track. The local TF-IDF primary track does not use that graph. If using an external dataset directory, Stage A's shared-graph inputs must also be prepared at the expected path. The command writes raw question-level files, a summary CSV, paired-comparison CSV, and run config. Keep raw outputs local unless their data-derived content is cleared for redistribution.

## Reproduce Phase B pilot and full run into a new directory

Use a fresh output root. Phase B refuses to overwrite a nonempty raw run directory. The verified source pins BGE to the same model revision as the saved canonical run.

```bash
python -m src.evaluate_dense_graph \
  --data-dir /path/to/local-data/pilot_500_v2 \
  --output-dir results/reproduction/dense_graph_v1 \
  --scope pilot50 --run-name pilot50_reproduced --seed 2027

python -m src.evaluate_dense_graph \
  --data-dir /path/to/local-data/pilot_500_v2 \
  --output-dir results/reproduction/dense_graph_v1 \
  --scope primary498 --run-name primary498_reproduced --seed 2027
```

The second command reuses any exact text/model-matched passage vectors cached by the first. It also computes new vectors for uncached passage and query IDs. One run directory contains raw `question_results.jsonl`, retrieval/graph/statistical summaries, efficiency data, selected question IDs, run config, and validation checks; embedding and entity extraction caches are under the output root's `cache/`. Raw results and caches are local-only in this release candidate.

## Run validation and independent audit

Evaluator-side guards run as part of each Phase B evaluation. To independently audit the freshly reproduced runs, save audit records to a separate directory:

```bash
python -m src.phase_b_independent_audit \
  results/reproduction/dense_graph_v1/pilot50_reproduced \
  --data-dir /path/to/local-data/pilot_500_v2 \
  --output results/reproduction/dense_graph_v1/audits/pilot50.json

python -m src.phase_b_independent_audit \
  results/reproduction/dense_graph_v1/primary498_reproduced \
  --data-dir /path/to/local-data/pilot_500_v2 \
  --output results/reproduction/dense_graph_v1/audits/primary498.json
```

The audit needs the matching BGE embedding cache and entity extraction cache created by evaluation. `src.validate_dense_graph_run` is an evaluator-side validation utility; it imports evaluator implementation helpers and is not a substitute for the separate independent audit.

The audit needs matching BGE embedding and entity extraction caches created by evaluation. The canonical audit execution was previously run and its aggregate records are under `results/dense_graph_v1/phase_b_validation_audits/`; raw caches are not included in this release candidate. The audit commands were verified via `--help`, but this documentation pass has not rerun the costly 498-question experiment.

## Historical experiment commands

Legacy commands are documented in [the original README history](../INTERIM_RESEARCH_REPORT.md), `docs/next_experiment_protocol.md`, `docs/next_local_run.md`, and their script `--help` output. Their defaults may point at historical `results/` paths and their inputs use pooled graphs. Always pass a new output directory before rerunning. Do not describe their pooled graph runs as question-local evaluations.

The data-generation script `src.load_data` fetches from Hugging Face and performs a streaming sample. Re-running the sampler can yield changed examples if upstream data or library behavior changes. Compare the generated hashes with the reference manifest described in `data/README.md`; do not claim a byte-identical or complete experiment reproduction unless the input hashes, package/model revisions, and required graph inputs match.
