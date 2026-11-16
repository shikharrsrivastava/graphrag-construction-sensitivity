# Pre-registration: non-graph and no-title controls for leave-one-out PPR (Phase C-2)

**Status:** fixed before any control result was computed. The commit that adds this file precedes the controls code and its outputs; the commit order is the evidence. These are **diagnostic controls on a finding that was already replicated**; none of them changes the Phase C pre-registered results, and the Phase C documents are not edited except for the wording correction in section 7.

## 1. Why

`ppr_entity_loo` (leave-one-out personalized PageRank over a passage–entity graph, λ = 0.5) improved MuSiQue complete-support@5 by about 12 points and replicated on a disjoint sample. Two questions decide whether "graph" is the right word for that effect:

1. **Non-graph control.** Is the gain just "reward passages that match the question *plus* the best other passage", which a hybrid dense+lexical retriever with a second-hop query can do without any graph?
2. **Title-node ablation.** The entity graph adds each passage's own (normalized, disambiguator-stripped) title as an entity node, which mimics Wikipedia hyperlink structure. Does the gain survive when the graph uses only spaCy entities?

(These two controls were proposed by an external reviewer of the repository and accepted by the project owner. The reviewer's numbered points were not available to the assistant that wrote this file; the controls below are the assistant's reading of them.)

## 2. Data and code reused unchanged

Samples and per-question results: `hotpot498`, `musique500`, `musique500_rep` (Phase C). BGE-small at the pinned revision, the entity-extraction cache, BM25 (k1 = 1.5, b = 0.75, lowercase `\w+`), min–max normalization, seeds = BGE top 5, λ = 0.5 for every combination, tie-break BGE then passage ID. Every method ranks exactly the question's own contexts. No parameter below is tuned.

## 3. New configurations

| Config | Definition |
|---|---|
| `hybrid_bge_bm25` | `0.5·minmax(BGE) + 0.5·minmax(BM25)` with the original question as query. |
| `hybrid_hop2` | Second-hop hybrid. For each seed `s` in BGE's top 5, form `q2_s = question + " " + "{title}. {text}" of s`. Let `h_s(p) = 0.5·minmax(BGE(q2_s, p)) + 0.5·minmax(BM25(q2_s, p))`, with min–max over the question's contexts, and `hop2(p) = mean of h_s(p) over seeds s ≠ p`. This leave-one-out step mirrors `ppr_entity_loo` (a passage never gets credit for its own seed query). Final: `0.5·minmax(hybrid_bge_bm25) + 0.5·minmax(hop2)`. BGE embeds `q2_s` with the same query prefix as in Phase B. |
| `ppr_entity_loo_notitle` | Exactly `ppr_entity_loo`, except title entity nodes are omitted: the graph has passage nodes and spaCy-entity nodes only. |

`ppr_entity_loo` is also recomputed inside the control script as a **reproducibility gate**: its recomputed rankings must equal the stored Phase C rankings on every question, otherwise the control run is invalid.

## 4. Confirmatory tests (complete-support@5, the headline metric; paired sign-flip, question is the unit)

Groups: **HotpotQA** (n = 498) and **MuSiQue pooled** (the two disjoint samples, n = 1,000). Holm correction across these **5 tests within each group**:

1. `hybrid_bge_bm25 − vanilla_bge`
2. `hybrid_hop2 − vanilla_bge`
3. `ppr_entity_loo − hybrid_hop2`
4. `ppr_entity_loo_notitle − vanilla_bge`
5. `ppr_entity_loo − ppr_entity_loo_notitle`

Recall@5, per-sample (non-pooled) results, hop-count and question-type breakdowns are reported as **EXPLORATORY** with unadjusted p-values.

## 5. Interpretation rules (fixed in advance; MuSiQue pooled is the deciding group)

* **Graph-specific.** The graph is supported as the cause only if test 3 is positive with Holm p < 0.05, meaning leave-one-out entity PPR beats the best non-graph second-hop control.
* **Explained without a graph.** If `hybrid_hop2` recovers at least half of the `ppr_entity_loo` gain over BGE (Δ test 2 ≥ 0.5 × Δ of `ppr_entity_loo − vanilla_bge`) and test 3 is not significantly positive, the finding is reported as "second-hop evidence helps; a graph is not necessary here".
* **Title dependence.** Title nodes are reported as necessary if test 5 is positive with Holm p < 0.05, and as not necessary otherwise. If test 4 is not significantly positive, the finding is reported as "does not survive without title nodes".
* Anything in between is reported as inconclusive, not rounded toward either reading.
* The HotpotQA group is reported in the same way, but the rules above are decided on MuSiQue.

## 6. Out of scope

Second retriever, 2WikiMultiHopQA, tuning λ/α/seed count/query construction, answer generation. All come after these controls.

## 7. Wording correction to the Phase C record (no result changes)

`PREREGISTRATION_graph_ranking.md` and `RESULTS_graph_ranking.md` say the smoke test was checked "for mechanics only; no accuracy metric was inspected". The smoke-test output files (`smoke_hotpot/retrieval_summary.csv`) contain 20-question accuracy and are committed, so that claim cannot be verified from the repository. The accurate statement, which replaces it in `RESULTS_graph_ranking.md`: *before adding the leave-one-out variant, only top-5 change counts and graph statistics were examined, and the accuracy columns of the smoke output were not read; this is the author's account and is not independently verifiable.* `PREREGISTRATION_graph_ranking.md` is left as committed, with this correction recorded here.

## Amendments

None.
