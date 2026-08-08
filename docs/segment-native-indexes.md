# Segment-native collection indexes

Collection indexes are owned by immutable data segments. Each persisted
segment may publish vector, FTS, and INVERT artifacts together with
the segment ID, document bounds, schema hash, and artifact paths in the
manifest. Flat indexes remain reconstructed in memory because their source
records already provide the exact representation.

ANN artifacts are immutable `.zvi` files. FTS and INVERT artifacts are
immutable `.pebble` directories with chunked ordered keys.

`Flush` seals the current WAL-backed segment first, then builds artifacts only
for immutable segments that do not already have matching metadata and valid
artifacts. Existing segment metadata and paths are copied unchanged into the next
manifest generation. The artifact set is published atomically and obsolete
compacted-segment artifacts are pruned after publication.
If a process stops between the data-segment and index-manifest publications,
the data remains committed; a later `Flush` or `Optimize` completes the missing
index build.

Open and query maintain one runtime cache entry per segment. Immutable entries
are reused across inserts, updates, deletes, and later flushes. Only the
WAL-backed mutable entry changes as new document versions are appended.

Query, MultiQuery, and GroupByQuery search each segment independently and then
apply the metric-aware global merge. Segment artifacts deliberately retain
deleted and superseded document versions, so every local search receives a
live-document mask. Scalar filters use each segment's INVERT postings and are
then forward-evaluated. FTS computes deletion-aware corpus statistics across
all segment dictionaries before sharing one BM25 scorer with the local
searches; SQL filtering changes eligibility but not IDF or average document
length.

`Optimize` rewrites the live snapshot into replacement segments, clears index
metadata for the removed segment IDs, builds the replacement artifacts, and
prunes the old files after their manifest is no longer current.
