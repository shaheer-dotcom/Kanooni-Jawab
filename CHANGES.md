# ASKFIDES Retrieval — Changes Log

**Date:** 2026-08-20  
**Files modified:** `retrieval/retriever.py`  
**Files deleted:** `retrieval/raptor.py`

---

## 1. Removed RAPTOR Feature Entirely

RAPTOR (Recursive Abstractive Processing for Tree-Organised Retrieval) was a hierarchical
summarisation layer that built L1/L2/L3 summaries of documents and stored them in Qdrant.
It was removed because:

- It caused **~28s of blocking startup time** — on every server boot, `RaptorIndexer.__init__()`
  fired 42 sequential HTTP `PUT /collections/{name}/index` requests to Qdrant for 7 jurisdictions
  × 6 fields, most returning 404 (collections don't exist yet).
- Only the `canada` collection exists currently, so RAPTOR indexes were returning no results anyway.

### What was removed from `retriever.py`

| Location | What was removed |
|---|---|
| Docstring | Step 3 "RAPTOR retrieval" from pipeline list |
| Import | `from raptor import RaptorIndexer` |
| `__init__` | `self.raptor = RaptorIndexer(...)` instantiation + log line |
| `_build_filter` | `must_not` condition filtering out `is_raptor="true"` points |
| `_raptor_retrieve()` | Entire method deleted |
| `_ensure_filter_indexes` | `"is_raptor"` removed from fields list |
| `_build_grounded_context` | `raptor_note` variable and its inclusion in context blocks |
| `_build_eval_debug` | `"is_raptor"` field removed from chunk record dict |
| `_build_references` | `"raptor_level"` field removed from references dict |
| `_run_vector_pipeline` | `raptor_chunks` block removed; `all_chunks = fused_chunks + raptor_chunks` → `all_chunks = fused_chunks` |
| `_run_vector_pipeline_events` | Same as above for streaming path |
| CLI `main()` | `RAPTOR ->` removed from pipeline description string |
| CLI reference output | `[RAPTOR L{n}]` label removed from source printout |

### `raptor.py` deleted

The entire file (`RaptorConfig`, `RaptorNode`, `RaptorIndexer` classes) was deleted.
It is no longer imported or referenced anywhere.

---

## 2. Performance Fixes — Reduced Query Latency

**Problem:** A sample query took **~290 seconds** to return a response.  
Root cause breakdown from logs:
- `Inference Embeddings: 7/7 [03:07<00:00, 26.76s/it]` — BGE-M3 re-embedding 50 MMR candidates
- 5 sub-queries × 2 collection passes × sequential `embed_query()` calls = 20 Qdrant round trips
- `needs_cases=True` + `needs_legislation=True` triggered two separate search passes over the
  same `canada` collection

### Fix 2 — Batch embed all sub-queries in one call (`_hybrid_search_collection`)

**Before:** called `self.embeddings.embed_query(sub_q)` once per sub-query inside a `for` loop —
N sequential model inference calls.

**After:** calls `self.embeddings.embed_documents(sub_queries)` once before the loop to get all
vectors in a single batched model pass. The pre-computed `all_vectors` list is then zipped with
`sub_queries` in the loop. The fallback path also reuses `all_vectors[0]` instead of re-embedding.

### Fix 3 — Cap sub-queries at 3

**Before:** The intent prompt asked Gemini to generate 4 alternative phrasings (5 total including
the original). The pipeline received `sub_queries=5` in the logs.

**After (two-part):**
1. Prompt updated to ask for 2 alternative phrasings (3 total including original).
2. Hard cap added in `_normalize_intent_routing`: `intent.sub_queries = intent.sub_queries[:3]`
   as a safety net in case Gemini returns more.

Reduces embedding work by ~40%.

### Fix 4 — Single collection search for hybrid route

**Before:** When `needs_cases=True` and `needs_legislation=True`, the pipeline called
`_run_collection("case")` then `_run_collection("legislation")` — two full embed+search passes
over the same `canada` collection (since all doc types share one collection).

**After:** When both flags are true, calls `_run_collection("any")` once. The `_run_collection`
inner function maps `dt="any"` to `effective_dt=None`, which is passed to `_build_filter`.
`_build_filter` signature changed from `doc_type: str` to `doc_type: Optional[str]` — when
`None`, the `doc_type` filter condition is simply omitted from the Qdrant query, so both cases
and legislation are returned in one pass.

### Fix 5 — Score-based MMR, no re-embedding

**Before:** `_mmr_rerank` re-embedded the query and all 50 candidate chunk texts using
`embed_query()` + `embed_documents()` to compute cosine similarity for diversity selection.
This was the **3-minute block** visible in the logs: `7/7 [03:07<00:00, 26.76s/it]`.

**After:** MMR now uses scores already present on each chunk from the Qdrant retrieval:
- **Relevance** = `_dense_score` normalised to [0, 1] (already on every chunk payload)
- **Redundancy** = fraction of already-selected chunks sharing the same `doc_id` / `parent_id`,
  penalising multiple chunks from the same source document

Zero additional model inference calls. The `query` parameter is kept in the signature for
API compatibility but is no longer used.

`import numpy as np` was also removed from `retriever.py` since it was only used by the
old embedding-based MMR.

---

## Summary of Expected Latency Improvement

For the sample query (scenario_analysis, hybrid route):

| Step | Before | After |
|---|---|---|
| Sub-query embedding | ~5 × 26s sequential = ~130s | 1 batched call ≈ 26s |
| Collection passes | 2 (case + legislation) | 1 (any) |
| MMR re-ranking | ~3 min (embed 50 chunks) | <1ms (score arithmetic) |
| Startup (RAPTOR init) | ~28s blocking | 0s |
| **Total estimate** | **~290s** | **~35-40s (CPU) / ~5s (GPU)** |

> **Note:** The remaining bottleneck is BGE-M3 running on CPU (~26s per batch).
> Enabling CUDA on the retrieval server would bring total latency to ~5s.
> Check with: `python -c "import torch; print(torch.cuda.is_available())"`
