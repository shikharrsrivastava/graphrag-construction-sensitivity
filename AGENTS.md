# GraphRAG Robustness Study — Project Instructions

## Research scope

Working title: **GraphRAG Robustness and Sensitivity Analysis: Investigating the Impact of Graph Construction Choices on Retrieval Performance and Answer Reliability.** The research asks how graph construction choices affect retrieval and generated answer reliability. This is an empirical evaluation; do not claim a new GraphRAG architecture.

Follow Professor Pasquale De Meo's scope:

- Evaluate 500–1,000 HotpotQA questions, validating the existing 500-question benchmark and methodology before expansion.
- Compare TF-IDF with BGE-small-en-v1.5 dense retrieval; consider Contriever only after BGE is working and justified.
- Evaluate answer Exact Match, token F1 and evidence faithfulness.
- Study node type (entity, chunk, sentence where feasible), edge semantics (co-occurrence, semantic similarity, typed relations only when validated), weighting, and controlled sparsification. Use a manageable design, not the full Cartesian product.
- Include vanilla dense RAG, a chunk-graph alternative, and one established GraphRAG system (LightRAG or Microsoft GraphRAG initially).
- Use multiple seeds, question-level confidence intervals, paired significance tests and a justified multiple-comparison correction. The question is the independent unit; repeated seeds are nested within question.
- Defer multimodal GraphRAG and knowledge-graph learning.

## Data and evaluation rules

- Preserve all existing source data and result artifacts. Never overwrite, delete, or rebuild an existing result before verifying redundancy and provenance. Write new runs to new, named directories.
- Prefer `data/pilot_500_v2`: it preserves `context_passage_ids`, has 498 ten-context questions and two two-context questions, and records that all 500 have their gold supporting passages in their contexts. Keep the two exceptional examples in a separate variable-context analysis; do not add unrelated passages.
- Preserve two explicit retrieval tracks:
  - **Track A, question-local:** graph, index and candidate universe use only that question's contexts. Primary result is the 498 complete ten-context questions; report the other two separately.
  - **Track B, shared global graph:** reuse global graph/index information but filter candidates to that question's contexts. Clearly disclose cross-question graph information.
- Never label pooled-corpus retrieval as question-local distractor evaluation. Do not filter only after global ranking and call it local; ranking must be scoped to the correct corpus.
- Report Recall@2, @5, @10, complete gold-support retrieval, candidate-generation recall, retrieval stability and efficiency. Validate passage IDs and matching rules.
- Report bridge and comparison results separately. The existing 500-question sample is all hard; avoid claims beyond the sampled population.
- For answer evaluation, keep the model, prompt, decoding and evidence-token budget fixed. Validate faithfulness on a manually reviewed subset. Report failed, preliminary, and unexecuted work honestly.

## Engineering and execution

- Audit existing scripts, data, caches and output schemas before implementation. Reuse existing entity extraction and Leiden/Louvain code where sound.
- Keep runs sequential on the documented Apple Silicon / 16 GB class machine unless actual environment inspection supports safe parallelism. Do not assume CUDA. Cache extraction and embeddings, checkpoint work, and make long jobs resumable.
- Inspect current environment before installing anything. Do not recreate `.venv`, reinstall dependencies, or delete model/package caches without need and approval.
- Preserve reproducibility: save run configuration, seed, package versions, raw per-question outputs, logs, and output checks. Every reported metric must trace to saved data and configuration.
- Add correctness tests for context filtering, evidence matching, metric calculations, and statistical aggregation as those components are implemented. Do not run expensive experiments before validating the evaluator.
- Keep source data immutable. Use new output paths for reruns. Use Git commits as recovery checkpoints for significant changes.
- Document failures and blockers exactly. Never fabricate results or imply a stage is complete when it is not.

## Research deliverables

Create a dedicated final results directory with raw outputs, consolidated CSV/JSON tables, graph-construction/retrieval/answer/statistical comparisons, academic figures, reproducible configurations, updated setup guidance, a comprehensive report, a seven-slide faculty presentation, and a formal progress email draft for Professor De Meo and involved faculty. Clearly distinguish completed, preliminary, failed, and unexecuted experiments.

See [CODEX_AUDIT_REPORT.md](CODEX_AUDIT_REPORT.md) for the verified repository baseline and next recommended actions.
