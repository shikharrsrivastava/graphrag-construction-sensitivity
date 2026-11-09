# Focused Literature Review: GraphRAG Design and Reproducibility

**Review date:** 9 November 2026  
**Purpose:** establish what the recent literature already covers before claiming a research gap. This is a focused initial review, not an exhaustive systematic review.

## 1. Closest and most relevant work

| Work | What it studies | Data / evaluation from the paper | Overlap with our project | Implication for our research |
|---|---|---|---|---|
| Nishida et al. (TACL 2026), [Dissecting GraphRAG: A Modular Analysis of Knowledge Structuring for Factoid Question Answering](https://aclanthology.org/2026.tacl-1.29/) | Module-level choices in triple extraction, community clustering, and report generation. The paper reports that extraction quality and the granularity of knowledge units affect downstream QA; template reports outperform LLM summaries in its setting. | Two knowledge-intensive benchmarks; factoid QA. | Directly overlaps with the broad premise that graph knowledge structuring and community design affect downstream QA. | We cannot claim that testing graph-construction or community choices against QA is novel by itself. Read its full methodology and identify any untested stability/replicability measures. |
| Hossain & Sarıyüce (KDD 2026), [Core-based Hierarchies for Efficient GraphRAG](https://arxiv.org/abs/2603.05207) | Studies modularity degeneracy in sparse knowledge graphs and the non-reproducibility of Leiden-based communities; proposes a deterministic k-core hierarchy. | Real-world earnings, news, and podcast datasets; community and GraphRAG generation comparisons. | Very close to our seed-stability work. Its central argument is that sparse graph structure can yield multiple near-optimal partitions, and it considers retrieval consequences. | Seed instability and graph-community reproducibility are not novel claims. We must compare against this work directly. A k-core baseline may be more defensible than adding another community detector, but only after we understand its protocol. |
| Liao et al. (2026), [GraphRAG Best Practices: A Modular Evaluation Framework and Regulatory Case Study](https://journals.sagepub.com/doi/10.3233/SSW260013) | Component-level study across graph indexing, graph-guided retrieval, and graph-enhanced generation. Tests chunking, extraction prompts, entity resolution, embedding choices, retrievers, traversal depth, rerankers, and graph serialization. | MSMARCO, HotpotQA (200 validation questions), and EU banking regulation QA; includes retrieval and answer-quality measures. | Broad overlap with graph design choices and end-to-end impact. In particular, it already compares sentence-level vs hierarchical chunking and measures faithfulness/relevancy. | A generic “component ablation” on HotpotQA will overlap. We need a more precise robustness/reproducibility question, matched baselines, and a clear distinction from their design-optimization focus. |
| Giordano & Razniewski (Findings of EACL 2026), [Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness](https://aclanthology.org/2026.findings-eacl.113/) | Studies reproducibility and robustness of LLM-extracted structured knowledge under changes in seed, language, randomness, and model. | miniGPTKBs across history, entertainment, and finance; yield, lexical similarity, and semantic similarity. | Related to variation in generated knowledge structures, but not directly a complete graph-to-retrieval-to-answer propagation evaluation in the paper's abstract. | Useful background for extraction reproducibility; distinguishes LLM-based graph construction variation from our current deterministic spaCy NER surface-form graph. |

## 2. What this means for the proposed contribution

The broad starting idea—“GraphRAG changes when its graph or clustering changes, and this can affect QA”—is already partly covered by the works above. We should not present that statement as the novelty.

A narrower candidate question worth testing is:

> **Under controlled changes to graph construction, how strongly do structural and community changes propagate to the evidence passages retrieved by matched lexical and dense retrievers, and then to answer correctness and faithfulness on multi-hop QA? How does this propagation compare with clustering-seed variation and edge perturbation?**

This candidate is still not a validated novelty claim. The full papers and their appendices must be compared before finalizing it.

## 3. Preliminary evidence from our own 50-question pilot

Our local pilot currently differs from the closest papers in several important ways:

- The existing graph is a spaCy named-entity surface-form co-occurrence graph, not an LLM-extracted typed knowledge graph.
- The current retrieval experiment is lexical TF-IDF seed selection and TF-IDF reranking, not dense retrieval or a full GraphRAG pipeline.
- We have measured retrieval evidence IDs and graph partition agreement, but no generated answers, EM/F1, or faithfulness evaluation.
- The saved 50-question sample happens to contain only hard questions, so it is too small and skewed for a general benchmark claim.
- We have now tested same-sentence vs same-passage co-occurrence while holding the node set and entity extraction fixed. That changes graph density and components substantially, while the retrieval differences in this pilot remain small and uncertain.

These differences are useful for defining the exact experimental question, but they do not prove novelty or superiority.

## 4. Practical design decisions

1. Read the full TACL, KDD, and SAGE papers—not only abstracts—and record their graph construction options, seed counts, data splits, graph statistics, retrieval metrics, answer-generation model, and significance procedures.
2. Explicitly position our current pilot as exploratory and replication-oriented.
3. At 500–1,000 HotpotQA questions, add a matched dense baseline and graph-assisted dense retrieval; keep question set, context corpus, seed retrieval budget, reranker, and final evidence budget aligned.
4. Evaluate generated answers separately from evidence recall. Report answer EM/F1 and evidence-grounded faithfulness with transparent judge limitations.
5. Consider k-core as a baseline because the KDD 2026 work proposes it specifically in response to sparse-graph Leiden instability.
6. Keep MMGraphRAG and multimodal extensions outside the current study scope, as advised by the mentor.

## 5. References

- Nishida, N., Munne, R. F., Liu, S., Tokunaga, N., Yamagata, Y., Cheng, F., Kozaki, K., & Matsumoto, Y. (2026). *Dissecting GraphRAG: A Modular Analysis of Knowledge Structuring for Factoid Question Answering*. Transactions of the Association for Computational Linguistics, 14, 627–655. https://doi.org/10.1162/tacl.a.615
- Hossain, J., & Sarıyüce, A. E. (2026). *Core-based Hierarchies for Efficient GraphRAG*. arXiv:2603.05207; accepted at KDD 2026. https://doi.org/10.1145/3770855.3818007
- Liao, Y.-Y., et al. (2026). *GraphRAG Best Practices: A Modular Evaluation Framework and Regulatory Case Study*. https://doi.org/10.3233/SSW260013
- Giordano, L., & Razniewski, S. (2026). *Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness*. Findings of EACL 2026, 2145–2164. https://doi.org/10.18653/v1/2026.findings-eacl.113
