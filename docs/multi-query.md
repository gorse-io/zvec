# MultiQuery

`Collection.MultiQuery` evaluates two or more dense-vector, sparse-vector, or
full-text branches over one live collection snapshot. Every branch receives
the same SQL scalar filter and produces its own bounded candidate list. An
explicit `Reranker` then combines those lists into at most `TopK` documents.

```go
results, err := collection.MultiQuery(ctx, zvec.MultiQuery{
    Queries: []zvec.SubQuery{
        {
            Field: "embedding",
            DenseVector: zvec.VectorFP32{0.1, 0.2, 0.3},
            NumCandidates: 50,
        },
        {
            Field: "keywords",
            SparseVector: zvec.SparseVectorFP32{
                Indices: []uint32{3, 17},
                Values:  []float32{0.8, 0.5},
            },
            NumCandidates: 50,
        },
        {
            Field: "body",
            FTS: &zvec.FTSClause{Match: "portable vector search"},
            Params: zvec.FTSQueryParams{DefaultOperator: "AND"},
            NumCandidates: 50,
        },
    },
    TopK: 10,
    Filter: "published = true",
    Projection: zvec.Projection{OutputFields: []string{"title"}},
    Reranker: myReranker, // omit to use default RRF
})
```

Exactly one target must be set in each `SubQuery`. Dense and sparse branches
accept the same index-specific `QueryParams` as `Collection.Query`. An FTS
branch targets an FTS-indexed `STRING` field and uses exactly one of:

- `FTSClause.Query` for terms, quoted phrases, parentheses, `AND`, `OR`, `+`
  must, and `-` must-not syntax;
- `FTSClause.Match` for natural text analyzed without operator parsing.

`FTSQueryParams.DefaultOperator` is case-insensitive `OR` or `AND`; empty means
`OR`. Index text and query text use the field's configured whitespace,
standard, ngram, or Jieba tokenizer followed by lowercase, ASCII-folding, and
stemmer filters in declaration order. Full-text candidates use exact BM25 with
the pinned `k1=1.2` and `b=0.75` defaults. The shared scalar filter masks
eligible documents after corpus statistics are formed, so filtering does not
change IDF or average document length.

`TopK == 0` defaults to 10. `NumCandidates == 0` also defaults to 10. Both are
bounded by 100,000, and MultiQuery requires at least two branches.

The process [`RuntimeConfig`](runtime-config.md) admits the complete operation
as one query task. Its scratch estimate increases with branch count. A
selective shared filter may route vector branches to exact scans and FTS
branches to posting seeks according to the configured planner ratios; these
routes are exact and do not change the result set.

## Reranker contract

`Reranker.Rerank` receives batches in sub-query order. Each `RerankBatch`
contains an independent field schema and projected, score-ordered document
copies. Vector scores retain their metric semantics; FTS scores are descending
BM25 values. A reranker may reorder candidates and replace their scores, but
its output must:

- contain at most `topK` documents;
- contain no duplicate document;
- select every document from at least one input batch, preserving its DocID
  and primary key;
- use finite scores.

The collection validates this boundary and rematerializes the selected
documents from the immutable snapshot, so changes to candidate field maps do
not alter stored or returned data. Caller code runs after the collection read
lock is released and may safely call other collection methods. Context and
reranker errors are propagated through the structured zvec error model.

The generic `Reranker` abstraction and baseline-compatible
[`RRFReranker`](rrf-reranker.md) are executable now. A nil reranker selects RRF
with `rank_constant=60`. [`WeightedReranker`](weighted-reranker.md) provides the
pinned metric-specific score normalization formulas and explicit per-branch
weights. [`CallbackReranker`](callback-reranker.md) adapts context-aware Go
functions, propagates returned errors, and contains callback panics as
structured internal errors.

## Storage boundary

MultiQuery reuses the collection's per-segment runtime cache. `Flush` publishes
vector files and FTS/INVERT Pebble directories only for newly immutable segments; reopen loads
matching artifacts instead of retraining them. Segment-local branches are
merged globally, and deletion masks prevent superseded versions from becoming
candidates. The native Go format does not read C++ collection files.
