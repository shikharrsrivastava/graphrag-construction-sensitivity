# Phase C results: when does a question-local graph change dense retrieval?

*Retrieval-only. Design fixed in [PREREGISTRATION_graph_ranking.md](PREREGISTRATION_graph_ranking.md) (committed before any run). All numbers come from `results/graph_ranking_v1/`. 95% CIs are question-level percentile bootstrap (10,000 replicates); p-values come from a paired two-sided sign-flip test (20,000 draws).*

## Research question and what changed since Phase B

Phase B used graphs only to generate candidates, which BGE then re-ranked. BGE's own top 5 was therefore always kept, so **the top 5 could not change by construction**. Phase C lets the graph change the ranking score:
`final = 0.5·minmax(BGE) + 0.5·minmax(PPR)`, where PPR is personalized PageRank (α = 0.85) seeded on BGE's top 5. It runs over a question-local passage–entity graph (`ppr_entity`) or the Phase B BGE 2-NN chunk graph (`ppr_chunk`). Every method ranks exactly the question's own contexts: 10 for HotpotQA, 20 for MuSiQue. We also add MuSiQue (answerable dev), which is harder and less saturated than HotpotQA.

**Validity gates passed.** All 16,478 question×method rankings are permutations of the question's contexts. On HotpotQA, `vanilla_bge` reproduces Phase B exactly (Recall@5 0.9056, complete-support@5 0.8153; maximum absolute difference 0). The regenerated HotpotQA sample matches the recorded hashes.

## Pre-registered headline (complete-support@5 = all gold passages in the top 5)

Holm correction is applied within each dataset over 6 tests (3 contrasts × complete-support@5 / Recall@5).

| Dataset | Contrast | complete-support@5 | Δ [95% CI] | Holm p | Recall@5 Δ [95% CI] | Holm p |
|---|---|---|---|---|---|---|
| HotpotQA (n = 498) | ppr_entity − BGE | 0.815 → 0.815 | +0.000 [0, 0] | 1.00 | +0.000 [0, 0] | 1.00 |
| | ppr_chunk − BGE | 0.815 → 0.807 | −0.008 [−0.024, +0.008] | 1.00 | −0.005 [−0.013, +0.003] | 1.00 |
| | BGE − BM25 | 0.612 → 0.815 | +0.203 [+0.153, +0.253] | 0.0003 | +0.105 [+0.079, +0.132] | 0.0003 |
| MuSiQue (n = 500) | ppr_entity − BGE | 0.350 → 0.350 | +0.000 [0, 0] | 1.00 | +0.000 [0, 0] | 1.00 |
| | ppr_chunk − BGE | 0.350 → 0.376 | +0.026 [+0.008, +0.046] | 0.040 | +0.003 [−0.008, +0.014] | 1.00 |
| | BGE − BM25 | 0.170 → 0.350 | +0.180 [+0.136, +0.224] | 0.0003 | +0.148 [+0.120, +0.176] | 0.0003 |

**By the pre-registered decision rule:**
- `ppr_entity` does not help on either dataset. It changed BGE's top 5 on **0 of 998** questions (and on 1 of 500 in the replication sample, with no metric change), because uniform restart on the seeds keeps the seeds' PageRank mass dominant.
- `ppr_chunk` does not help on HotpotQA.
- `ppr_chunk` passes the MuSiQue threshold narrowly (+2.6 points, Holm p = 0.040). It **did not replicate** on a fresh, disjoint MuSiQue sample (+1.6 points [−0.4, +3.6], Holm p = 0.34 in the registered replication family), so it should be treated as a probable false positive.
- BGE beats BM25 by a large margin on both datasets.

## Exploratory finding, then replicated

The smoke test showed that uniform-seed PPR rarely moves non-seed passages. This was a mechanics check, and no accuracy was inspected. Before the full runs I logged one variant, `ppr_entity_loo`: each passage's graph score is the PPR mass it receives from the *other* seeds (leave-one-out). On `musique500` it raised complete-support@5 from 0.350 to 0.468 (+0.118, 90 questions better and 31 worse).

Because this was exploratory, a **replication was pre-registered and committed before it ran**. It used 500 new MuSiQue questions (seed 2028, no overlap) with unchanged code.

| Replication (MuSiQue, n = 500, Holm over 4) | BGE → method | Δ [95% CI] | better / worse | Holm p | Replicated |
|---|---|---|---|---|---|
| ppr_entity_loo, complete-support@5 (primary) | 0.358 → 0.482 | **+0.124 [+0.082, +0.168]** | 93 / 31 | **0.0002** | **yes** |
| ppr_entity_loo, Recall@5 | 0.678 → 0.720 | +0.042 [+0.017, +0.066] | 136 / 75 | 0.003 | yes |
| ppr_chunk, complete-support@5 | 0.358 → 0.374 | +0.016 [−0.004, +0.036] | 17 / 9 | 0.34 | no |
| ppr_chunk, Recall@5 | 0.678 → 0.679 | +0.002 [−0.009, +0.012] | 30 / 24 | 0.72 | no |

## Other exploratory results (unadjusted; descriptive)

* **The same method does not help on HotpotQA.** `ppr_entity_loo` gives complete-support@5 of 0.787 vs 0.815 (−0.028 [−0.064, +0.008]). On bridge questions it is +0.012 (34 better / 29 worse). On comparison questions it is **−0.202** (0 better / 19 worse): those questions ask about two unconnected entities, so rewarding graph connectivity pushes out a gold passage. `ppr_chunk_loo` is negative or null on every sample (HotpotQA −0.042, MuSiQue −0.024, replication −0.004).
* **By hop count** (`ppr_entity_loo`, MuSiQue replication): 2-hop 0.504 → 0.585 (+0.081); 3-hop 0.230 → 0.461 (+0.230); 4-hop 0.092 → 0.158 (+0.066, n = 76, CI includes small values). The first sample shows the same pattern (+0.115, +0.157, +0.056).
* **Top-5 changes and gold movement.** On the MuSiQue replication, `ppr_entity_loo` changed the top-5 set on 440/500 questions. Complete-support@5 improved on 93 and worsened on 31; the rest were unchanged. Gold passages entering the top 5 outnumber those leaving it (+186 / −125). On HotpotQA the balance is negative (+37 / −58).
* **λ sensitivity** (pre-registered graphs only). `ppr_entity` never changes the top 5 at λ = 0.25, 0.5 or 0.75 (at most 1 question). `ppr_chunk` on HotpotQA degrades as λ grows (λ = 0.75: −0.044 [−0.068, −0.022]). On MuSiQue it peaks at λ = 0.5.
* **Where gains occur.** All improvements come from questions where BGE failed complete-support@5, and all losses from questions where BGE succeeded. This is partly mechanical: a binary metric can only move in one direction in each stratum. The net effect above is what matters.
* **Headroom.** HotpotQA saturates: BGE Recall@10 = 1.000 and complete-support@10 = 1.000 (with 10 contexts, any full ranking is "complete" at 10). MuSiQue has headroom: Recall@10 0.843 / 0.849 and complete-support@10 0.606 / 0.632 (two samples).

## Limitations

* **One small embedding model** (BGE-small, pinned) and one PPR setting (α = 0.85, top-5 seeds, λ = 0.5). The LOO effect may differ with stronger retrievers.
* **spaCy entities are not validated** against human labels. The entity graph is co-occurrence plus title matching, with no relation typing.
* **Retrieval only.** No answer generation, EM/F1 or faithfulness, so a retrieval gain is not shown to improve answers.
* **Two datasets, both Wikipedia-based**, with fixed samples: the HotpotQA questions are all "hard" and are not a simple random sample, and the MuSiQue questions are answerable, 20-paragraph ones. No 2WikiMultiHopQA run.
* The LOO variant was conceived after seeing HotpotQA mechanics (not accuracy). Its MuSiQue effect is replicated, but its λ, α and seed count were never varied.
* This design (method, pre-registration and analysis) was drafted with an AI coding assistant and accepted by the project owner, because the originally intended patch was unavailable.

## Conclusion

Letting a simple graph influence ranking scores does not by itself help a strong dense retriever. Under the pre-registered design, entity-graph PPR left BGE's top 5 unchanged on all 998 questions. Chunk-graph PPR gave a small MuSiQue gain that did not replicate, and neither helped on HotpotQA.

The graph matters only when the ranking rewards a passage for being connected to *other* strong evidence (leave-one-out PPR over an entity graph). In that case it improved complete supporting-evidence retrieval on MuSiQue by about 12 points, and the gain replicated on a disjoint sample (0.358 → 0.482, Holm p = 0.0002), with the largest gain on 3-hop questions. The same method did not help on HotpotQA and clearly hurt comparison questions.

The supported claim is therefore narrow. In this setting, graph-aware ranking helps when the questions are chain-structured multi-hop questions, the dense retriever has headroom, and the graph signal excludes self-support. It does not help on saturated or comparison-style questions. Whether this transfers to answer quality, other retrievers, validated entity graphs or other datasets remains untested.
