# Research Brief — Outline

## Working title
Does More Stable Clustering Improve GraphRAG Answers?

## Research question
Under graph-edge perturbations, does community-partition stability correspond to preservation of supporting evidence for multi-hop question answering?

## Current implementation
Describe only components that are actually implemented and have been run. The starter currently supports a lightweight NER co-occurrence graph, Leiden/Louvain clustering, edge perturbation, ARI and graph-connectivity metrics.

## Planned evaluation
Connect entity communities to passages and implement a fixed-budget candidate retrieval rule; evaluate supporting-fact Recall@k on HotpotQA. Answer generation and Exact Match/F1 are later steps.

## Preliminary findings
Populate only from recorded, reproducible runs. Do not insert hypothetical numbers.

## Limitations
NER-derived co-occurrence is not a typed knowledge graph; a small sample is a pilot; graph structure alone does not establish retrieval or answer quality; novelty is not yet established.

## Related work
Review and accurately summarize From Local to Global (2024), KG²RAG (NAACL 2025), Dissecting GraphRAG (TACL 2026), and Core-based Hierarchies for Efficient GraphRAG (2026), as relevant. Verify full papers and distinguish prior work from the proposed question.
