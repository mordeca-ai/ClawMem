# Intent search pipeline (the shared causal pipeline)

The `intent_search` MCP tool classifies query intent and uses graph traversal to find causal chains across AI agent memory that keyword and vector search alone can't reach.

Since v0.32.0 this pipeline is ONE shared implementation (`src/causal-retrieval.ts`) behind every causal surface — `intent_search`, `memory_retrieve`'s causal mode, `query_plan`'s graph clauses, and REST `/retrieve`'s causal mode. Callers differ only in declared stages and visibility policy: `intent_search` runs unfiltered (system memory is its substrate); the default-filtered callers exclude `_clawmem` but gain the WHY observation lane described below. `enable_graph_traversal: false` disables every graph stage — adaptive traversal, MPFP, entity expansion, and the causal one-hop.

## Pipeline stages

```
User Query
  │
  ▼
Intent Classification (LLM)
  │ → WHY, WHEN, ENTITY, or WHAT
  │ (or force_intent override)
  │
  ▼
Anchor Search — named ranked lists
  │ BM25 + Vector in parallel (temporal window applied here AND in every graph leg)
  │ + WHY observation lane (default-filtered callers only): BM25 + Vector restricted to
  │   `_clawmem` observation documents in the search SQL, top-5 per channel
  │
  ▼
Intent-Weighted RRF over the named lists
  │ WHY  → boost vector [1.0, 1.5]
  │ WHEN → boost BM25   [1.5, 1.0]
  │ Other → balanced     [1.0, 1.0]
  │
  ▼
Bounded One-Hop Causal Step (WHY only)
  │ BOTH directions over relation_type='causal' — the only backward causal reach
  │ (adaptive/MPFP inbound is semantic+entity only)
  │ Caps: ≤3 per anchor, ≤10 unique endpoints, enforced before fusion
  │ Hits re-enter RRF as a fifth named list; results carry
  │ causal: [{anchorDocid, direction: "cause"|"effect"}]
  │
  ▼
Graph Traversal (WHY/ENTITY only; base-eligible seeds only)
  │ Multi-hop beam search over memory_relations
  │ Outbound: all edge types
  │ Inbound: semantic + entity only
  │ Budget: 30 nodes, depth: 2, beam: 5
  │ Candidate eligibility (active, non-invalidated, time window, collection policy)
  │ enforced IN the SQL — ineligible rows never consume beam slots
  │ Scores normalized to [0,1]
  │
  ▼
Cross-Encoder Reranking
  │ 200 chars/doc context
  │ File-keyed score join
  │
  ▼
Position-Aware Blending
  │ origWeight · upstream score + (1 − origWeight) · rerank
  │ origWeight = 0.75 (top 3), 0.60 (mid), 0.40 (tail)
  │ (intent_search keeps this curve; the query tool uses blendRerank instead)
  │
  ▼
Composite Scoring
  │
  ▼
Results (with intent + confidence metadata)
```

Entity co-occurrence expansion (ENTITY intent) is a legacy stage OUTSIDE the eligibility
contract — its aggregates carry no source-document provenance, so no visibility or time policy
can be honored through them. It runs only on direct `intent_search`.

## Intent types

| Intent | Signal | Graph traversal | RRF weighting |
|--------|--------|----------------|---------------|
| WHY | "why did", "what caused", "reason for", "decided to" | Yes | Boost vector |
| WHEN | "when did", "first/last occurrence", timeline | No | Boost BM25 |
| ENTITY | Named component/person/service needing cross-doc linkage | Yes | Balanced |
| WHAT | General factual | No | Balanced |

## Graph traversal

When intent is WHY or ENTITY, the pipeline runs `adaptiveTraversal()`:

1. Anchors on top 10 search results
2. Traverses `memory_relations` edges:
   - **Outbound** (source→target): semantic, supporting, contradicts, causal, temporal
   - **Inbound** (target→source): semantic and entity only
3. Beam search with query embedding as relevance signal
4. Discovered nodes are hydrated from the database and merged with search results
5. Scores are normalized to [0, 1] before merging

## Graph edge sources

| Source | Edge types | How populated |
|--------|-----------|--------------|
| A-MEM `generateMemoryLinks()` | semantic, supporting, contradicts | During indexing (new docs) |
| A-MEM `inferCausalLinks()` | causal | Post-response (decision-extractor) |
| Beads `syncBeadsIssues()` | causal, supporting, semantic | `beads_sync` or watcher |
| `buildTemporalBackbone()` | temporal | `build_graphs` (manual) |
| `buildSemanticGraph()` | semantic | `build_graphs` (manual) |

## Differences from query

| Aspect | `query` | `intent_search` |
|--------|---------|-----------------|
| Query expansion | Yes | No |
| Intent hint | Manual (`intent` param) | Auto-detected |
| Rerank context | 4000 chars/doc | 200 chars/doc |
| Graph traversal | No | Yes (WHY/ENTITY) |
| MMR diversity | Yes | No |
| `compact` param | Yes | No |
| `collection` filter | Yes | No |
| Best for | General recall | Causal chains spanning docs |

## When to use

Use `intent_search` **directly** (not as a fallback from query) when:
- The question starts with "why"
- You need to trace decision chains
- You're asking about entity relationships across documents
- You need temporal context ("when did this change")

For WHEN queries, start with `enable_graph_traversal=false` (BM25-biased). Fall back to `query()` if recall drifts.
