# KnowledgeOS

A modular RAG platform built from scratch for learning and benchmarking.

## Architecture

The system uses a plugin-based design. Every component subclasses one of seven ABCs defined in ``core/interfaces.py``.

### Data Contracts

Two dataclasses flow through every stage: Document and Chunk. Both carry a tenant_id field for multi-tenant isolation.

## Evaluation

We use a hand-labeled gold set of 10 query-to-relevant-text pairs. Metrics include recall@k, MRR, and 4-gram faithfulness.

### Baseline Numbers

Our M0 baseline achieved recall@1 of 0.8, recall@3 of 1.0, and faithfulness of 0.53. The recall gap motivates adding a reranker in M6.

## Roadmap

Twelve milestones total, from M0 (spine) through M12 (research lab). Each milestone is stop-safe: if you halt after any milestone, you still have a coherent artifact.
"@
