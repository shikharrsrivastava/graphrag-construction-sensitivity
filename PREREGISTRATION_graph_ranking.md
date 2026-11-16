# Pre-registration: Graph-aware ranking (Phase C)

**Status:** fixed before any Phase C result was computed. Committed to Git before the evaluator was written or run; the commit timestamp is the registration time. Any later change is listed under *Amendments* with its reason and date, and anything not listed here is reported as **EXPLORATORY**.

**Authorship disclosure.** The original Phase C patch and pre-registration could not be located. At the project owner's request, this design was drafted by an AI coding assistant (Claude) and accepted by the owner. It was written after Phase B results were known (Phase B: graph expansion never changed BGE's top-5) but before any graph-aware ranking or MuSiQue retrieval result existed. It is not a design supplied by Professor De Meo.

## 1. Research question

Phase B used graphs only to *generate candidates* that were then reranked by BGE similarity alone. Because BGE's top-5 seeds were always in the candidate set and the reranker was BGE, the top-5 was invariant by construction. Phase C asks:

> When a question-local graph is allowed to change the **ranking score**, does it improve complete supporting-evidence retrieval over a strong dense retriever, on HotpotQA and on the harder, less saturated MuSiQue?

This is an empirical retrieval evaluation, not a new architecture. No answers are generated.

## 2. Data

| Dataset | Source | Sample | Ranking universe |
|---|---|---|---|
| HotpotQA | `data/pilot_500_v2` (regenerated; must match the recorded question-ID and passage-ID SHA-256 hashes) | the 498 ten-context questions (404 bridge, 94 comparison); the two 2-context questions are excluded from Phase C | the question's 10 contexts |
| MuSiQue | `bdsaglam/musique`, file `musique_ans_v1.0_dev.jsonl`, revision `22873a405dd809893b22ada0b499299fb612d2df` (answerable v1.0 dev, 2,417 questions) | restrict to questions with exactly 20 unique paragraphs; sort IDs lexicographically; draw 500 without replacement with `numpy.random.default_rng(2027).choice` | the question's 20 paragraphs |

Track A only (question-local): every index, graph and score uses only the current question's contexts, and every configuration ranks exactly that set. MuSiQue gold = paragraphs with `is_supporting = true`; hop count is taken from the ID prefix (`2hop`, `3hop*`, `4hop*`).

## 3. Methods

Shared: BGE = `BAAI/bge-small-en-v1.5` at revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, normalized embeddings, the Phase B query prefix, passage text `"{title}. {text}"`, CPU.

| Config | Definition |
|---|---|
| `bm25` | `rank_bm25.BM25Okapi` (k1 = 1.5, b = 0.75) fit on the question's contexts; tokens = lowercase `\w+` over title + text. |
| `vanilla_tfidf` | Phase B local TF-IDF (reference only; not in the confirmatory family). |
| `vanilla_bge` | cosine(query, passage); identical to Phase B. |
| `ppr_entity` | Graph-aware ranking over a bipartite passage–entity graph (below). |
| `ppr_chunk` | Graph-aware ranking over the Phase B BGE union-kNN passage graph (k = 2, edge weight = cosine). |

**Entity graph.** The nodes are the question's passages plus entity nodes. An entity is (a) a spaCy `en_core_web_sm` entity of the Phase B labels, normalized with the existing `normalize_entity`, extracted passage-locally (reusing the Phase B cache rules), or (b) the passage's own normalized title. Edges run passage–entity with weight 1. Passages are therefore linked through shared entities, including a mention of another passage's title (the bridge pattern).

**Graph score.** This is personalized PageRank (`networkx.pagerank`, alpha = 0.85, `weight="weight"`, max_iter = 1000, tol = 1e-10). The personalization is uniform over BGE's top-5 passages. The graph score of a passage is its PageRank mass; entity-node mass is discarded.

**Final score.** `final = (1 − λ) · minmax(BGE) + λ · minmax(PPR)`, with min–max taken over the question's contexts (a constant vector maps to 0). Ties are broken by BGE score descending, then passage ID ascending. **λ = 0.5** is the confirmatory value. λ ∈ {0.25, 0.75} is exploratory sensitivity only.

## 4. Outcomes and confirmatory tests

* **Headline metric:** complete_support@5 (all gold passages in top 5). **Secondary:** Recall@5.
* **Contrasts (per dataset):**
  1. `ppr_entity − vanilla_bge`
  2. `ppr_chunk − vanilla_bge`
  3. `vanilla_bge − bm25`
* Each contrast × each of the 2 metrics gives **6 tests per dataset**, and Holm correction is applied within each dataset's family of 6.
* Test: a paired two-sided sign-flip test with question as the unit (20,000 Monte Carlo draws, plus-one correction). The 95% CI of the paired difference is a percentile bootstrap over questions (10,000 replicates). The method is deterministic, so there are no seeds to nest.
* **Decision rule:** graph-aware ranking "helps" on a dataset iff the complete_support@5 contrast is positive with Holm-adjusted p < 0.05. A non-significant result is reported as a null, and a significant negative result as harm.
* Output: `paired_headline_contrasts.csv` per dataset.

## 5. Validity gates (checked before analysis)

1. Every configuration's ranked list is a permutation of exactly the question's contexts.
2. HotpotQA `vanilla_bge` must reproduce `results/dense_graph_v1/primary498_final` (Recall@5 = 0.9056…, complete_support@5 = 0.8152…, to floating-point precision). If not, stop and debug.
3. The regenerated HotpotQA hashes must equal the recorded manifest.

## 6. Exploratory analyses (labelled EXPLORATORY in all outputs)

* Results by question type: HotpotQA bridge/comparison; MuSiQue 2/3/4 hops.
* Top-5 change counts: how many questions changed, and among those, how many improved or worsened on complete_support@5 and Recall@5.
* Gold-passage rank movement relative to BGE.
* λ ∈ {0.25, 0.75} sensitivity.
* Gains conditional on BGE complete_support@5 = 0 vs 1.
* Headroom: BGE Recall@10 and complete_support@10.
* Other metrics: Recall@2/@10 and complete_support@2/@10 for all methods.

## 7. Out of scope

Answer generation, EM/F1, faithfulness, LightRAG/MS GraphRAG, Contriever, 2WikiMultiHopQA, entity-extraction validation, and tuning λ, alpha or the seed count.

## Exploratory addendum (post-smoke, pre-full-run; NOT confirmatory)

**2026-11-16.** The 20-question HotpotQA smoke test was checked for mechanics only; no accuracy metric was inspected. `ppr_entity` changed BGE's top-5 on 0/20 questions and `ppr_chunk` on 7/20. With uniform restart on the top-5 seeds, the seeds hold most of the PPR mass, so non-seed passages linked through entities rarely overtake them.

One exploratory variant per graph was therefore added before any full run: `ppr_entity_loo` and `ppr_chunk_loo`. Each passage's graph score is the PPR mass it receives when it is itself removed from the seed set (leave-one-out), combined with λ = 0.5. These variants are not part of the Holm family and must always be labelled EXPLORATORY. Their paired tests against `vanilla_bge` are reported with their own Holm correction across the two variants × two metrics.

## Confirmatory replication (registered 2026-11-16, AFTER the Phase C results above; before the replication run)

**Motivation.** In the exploratory analysis of `musique500`, `ppr_entity_loo` raised complete_support@5 over `vanilla_bge` by +0.118 (Holm p = 0.0002 within the exploratory family). The pre-registered `ppr_chunk` result (+0.026, Holm p = 0.040) was borderline. Both observations are now tested on fresh data with no further changes to any method.

* **Sample:** `data/musique_500_rep`, drawn from the same pinned MuSiQue file with the same eligibility rule. The 500 `musique500` IDs are excluded first, and 500 are then drawn with `numpy.random.default_rng(2028)`. The two samples are disjoint.
* **Code:** `src/evaluate_graph_ranking.py` exactly as used for `musique500`, so `ppr_entity_loo` keeps λ = 0.5, alpha = 0.85 and the top-5 seeds.
* **Tests (Holm across these 4):**
  1. `ppr_entity_loo − vanilla_bge` on complete_support@5 (**primary**) and on Recall@5.
  2. `ppr_chunk − vanilla_bge` on complete_support@5 and on Recall@5.
* **Decision rule:** a finding is "replicated" iff it has the same sign and Holm p < 0.05 on the replication sample. Otherwise it is reported as not replicated, whatever the first sample showed.
* **Output:** `results/graph_ranking_v1/musique500_rep/replication_contrasts.csv`.

## Amendments

1. **2026-11-16, before the evaluator was run on any question.** I specified "normalized title" as follows: remove a single trailing parenthetical disambiguator (for example `Green (album)` becomes `green`), then apply `normalize_entity`. Without this, Wikipedia-style titles could never match a plain mention. No result had been computed when this was fixed.
