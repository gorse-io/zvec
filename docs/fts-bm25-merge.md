# BM25 scoring and FTS segment merging

The v0.5 FTS scoring layer implements the pinned zvec BM25 formula and compacts
native Go FTS dictionaries without CGO or a RocksDB intermediate. It consumes
the term frequencies, document lengths, positions, and deletion-aware corpus
statistics introduced by the preceding FTS units.

## Scoring

`DefaultBM25Params` returns the baseline parameters `K1=1.2` and `B=0.75`.
`NewBM25Scorer` validates these parameters and takes an owned snapshot of
`FTSCorpusStats`. Build the statistics once across every segment participating
in a query:

```go
stats, err := core.AggregateFTSCorpusStats(ctx, []core.FTSSegmentView{
    {Dictionary: segment0, DeletedDocuments: deleted0},
    {Dictionary: segment1, DeletedDocuments: deleted1},
})
scorer, err := core.NewBM25Scorer(core.DefaultBM25Params(), stats)
```

Every independently searched segment can then share that immutable scorer,
which gives it the same live document count, average document length, and term
document frequencies. Scores are therefore comparable before a later
collection-level result merge.

The implementation uses the baseline Robertson-Sparck Jones smoothing:

```text
idf(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

score(t, d) = idf(t) *
              tf(t,d) * (K1 + 1) /
              (tf(t,d) + K1 * (1 - B + B * |d| / avgdl))
```

`IDF`, `TermIDF`, `Score`, `ScoreWithIDF`,
`ScoreWithIDFAndBoost`, and `MaxScoreBound` expose the same mathematical
building blocks as the fixed C++ scorer. Term and phrase boosts are linear, so
the AST rewriter can merge duplicate query leaves without changing their
score.

`NewFTSScoredQueryIterator` extends the exact iterator with `Score`. Term
nodes score their current inline tf and document length; phrase nodes sum the
terms that passed exact position verification; OR sums every child matching
the document; AND sums all required children and any optional (`Should`)
children matching the same document. Excluded children never contribute.
The original `NewFTSQueryIterator` remains a score-free exact iterator and
returns zero from `Score`.

`SearchFTS` keeps a bounded top-k heap and remains exact. Once that heap is
full, a root disjunction uses MaxScore WAND pivots to bypass candidates whose
matching children cannot reach the minimum competitive score. An aligned
pivot then computes safe BM25 upper bounds lazily for the current 128-posting
blocks and seeks past a shared document range only when its bound is strictly
below the threshold. Results use descending score and ascending document ID
for deterministic ties. A zero `TopK` returns immediately after validating the
dictionary and scorer. Non-positive scores are omitted, matching the baseline
search boundary.

```go
results, err := core.SearchFTS(
    ctx,
    segment0,
    queryAST,
    scorer,
    core.FTSSearchOptions{
        TopK: 10,
        FTSQueryExecutionOptions: core.FTSQueryExecutionOptions{
            DeletedDocuments: deleted0,
        },
    },
)
```

Block maxima are derived from inline term frequency and document length using
the query's deletion-aware global scorer. They are cached by each query-local
term iterator, so no posting-format migration is required and separately
searched segments retain comparable bounds. AND, optional, OR, and phrase
trees add non-negative child bounds over the shortest shared range; negative
clauses never contribute. A strict comparison preserves score ties and their
ascending-document-ID order.

WAND uses conservatively rounded tf-to-infinity term bounds and caches the
result on the query-local disjunction. Non-finite potential disables pruning
instead of hiding the existing invalid-score error. Competitive scheduling is
separate from ordinary `Advance(target)`: nested disjunctions therefore retain
exact seek behavior when an AND parent requests an optional score at a chosen
document.

## Native segment merge

`MergeFTSTermDictionaries` consumes `FTSSegmentView` values in slice order.
It snapshots each deletion bitmap and makes a two-part streaming compaction:

1. It walks document-length arrays to calculate exact survivor statistics and
   the dense destination ranges.
2. It performs a byte-lexical multi-way term merge, holding only one term's
   decoded postings at a time. Each surviving posting is remapped, then its
   tf, document length, and position list are re-encoded into a checksummed
   native posting list.

The document-ID spaces are:

```text
source local ID  -> input dictionary document ID
scan order       -> source slices concatenated in caller order
output local ID  -> dense rank among non-deleted scan-order documents
```

Thus an output ID is the source's scan-order position minus the number of
earlier deletions. Empty documents are preserved even though they have no
postings. A term present only in deleted documents disappears, and maximum tf
is recomputed from survivors. The output dictionary can immediately be
queried, saved to Pebble, reopened, or fed into another merge; no source
dictionary or bitmap is modified.

```go
merged, err := core.MergeFTSTermDictionaries(ctx, []core.FTSSegmentView{
    {Dictionary: segment0, DeletedDocuments: deleted0},
    {Dictionary: segment1, DeletedDocuments: deleted1},
})
err = merged.Save(ctx, "segment-fts.pebble")
reopened, err := core.OpenFTSTermDictionary(ctx, "segment-fts.pebble")
```

An empty source slice produces a valid empty dictionary. Invalid deletion bits,
nil source dictionaries, count overflow, cancellation, and malformed internal
metadata fail explicitly with `ErrInvalidFTSMerge`. Persistence writes term
metadata and independently checksummed posting payloads as bounded,
ordinal-keyed Pebble chunks.

## Compatibility and tests

The fixture pins the baseline BM25 scorer and RocksDB reducer source hashes at
commit `58375ff`, plus representative IDF/score values and dense merge counts.
Tests cover formula values, parameter and statistics validation, ownership,
concurrent scorer use, boolean/phrase/boost score composition, stable top-k,
deletion-aware scoring, iterator seek, dense remapping, empty documents,
all-deleted inputs, maximum-tf recomputation, position preservation, Pebble
save and reopen, source immutability, cancellation, concurrent merges, fuzzing, and
benchmarks.

```sh
go test ./internal/core -run '^Test(BM25|SearchFTS|MergeFTS)'
go test ./internal/core -run '^$' -fuzz '^FuzzBM25Scorer$'
go test ./internal/core -run '^$' -fuzz '^FuzzMergeFTSTermDictionaries$'
go test ./internal/core -run '^$' -bench '^(BenchmarkSearchFTSBM25|BenchmarkMergeFTSTermDictionaries)$' -benchmem
```

Collection Query and MultiQuery use deletion-aware BM25 and persist one FTS
dictionary per immutable data segment. They aggregate global live-corpus stats
without merging the dictionaries. Mixed vector/sparse/FTS fusion and exact
query-local block-max WAND pruning are available through MultiQuery.
