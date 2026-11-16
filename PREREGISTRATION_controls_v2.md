# Pre-registration: dense-only second-hop control (Phase C-3)

**Status:** fixed before any dense-only second-hop score was computed. The commit adding this file precedes the code and its outputs.

**Disclosure.** This control was added **after** the Phase C-2 controls (`PREREGISTRATION_controls.md`) had been run and their results seen. An external reviewer pointed out that the C-2 non-graph control mixes BGE and BM25 50/50, and BM25 is far weaker than BGE on MuSiQue (complete-support@5 0.170 vs 0.354), so that control was handicapped and scored below plain BGE. This control removes the lexical half. No result of the dense-only variant existed when this file was written. Its design is fixed here, with no tuning, and the C-2 results are not edited.

## Design (reuses all caches, including the cached second-hop query embeddings)

`dense_hop2`: for each seed `s` in BGE's top 5, take the same second-hop query `q2_s = question + " " + "{title}. {text}" of s` and the same BGE embeddings as in C-2. Let `h_s(p) = minmax(BGE(q2_s, p))` over the question's contexts (no BM25), and `hop2d(p) = mean of h_s(p) over seeds s ≠ p` (leave-one-out, as in C-2 and in `ppr_entity_loo`). Final: `0.5·minmax(BGE(q, p)) + 0.5·minmax(hop2d(p))`, tie-break BGE then passage ID, λ = 0.5. Every ranking is a permutation of the question's own contexts.

Data: `hotpot498`, `musique500`, `musique500_rep` as before. Groups: HotpotQA (n = 498) and MuSiQue pooled (n = 1,000).

## Confirmatory tests (complete-support@5; paired sign-flip; Holm across these 2 tests within each group)

1. `dense_hop2 − vanilla_bge`
2. `ppr_entity_loo − dense_hop2`

## Interpretation rules (fixed in advance; MuSiQue pooled decides)

* **Graph-specific against the fair non-graph control:** test 2 is positive with Holm p < 0.05.
* **Explained without a graph:** `dense_hop2` recovers at least half of the `ppr_entity_loo` gain over BGE (Δ test 1 ≥ 0.5 × Δ of `ppr_entity_loo − vanilla_bge`) and test 2 is not significantly positive. The finding is then reported as "second-hop evidence helps; a graph is not necessary here", and the earlier "graph-specific" wording in `RESULTS_graph_ranking.md` is corrected.
* Anything in between is reported as inconclusive. Recall@5 and per-sample results are EXPLORATORY with unadjusted p-values.

## Out of scope

Other non-graph second-hop methods (a trained reranker, multi-query fusion, tuning λ or the query format), a second retriever, answer generation.

## Amendments

1. **2026-11-16, before any dense-only score was computed.** The first attempt stopped with a guard error: the C-2 run wrote all three datasets' second-hop embeddings to one cache file, so each dataset overwrote the previous one and only the last dataset's embeddings survived. "Reuses all caches, including the second-hop query embeddings" above was therefore partly wrong. The missing second-hop queries are re-encoded from the identical inputs (same text, model revision and prefix) into one cache file per dataset. The C-2 results are unaffected, since they used the in-memory embeddings. The failed attempt's `failure.json` is kept as `failure_attempt1.json`. No design element changes.
