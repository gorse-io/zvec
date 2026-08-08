# Native Go disk format

The Go implementation uses its own disk format and does not open collection
files created by the C++ implementation. Format version 2 uses the manifest
protocol below and Pebble-backed FTS and INVERT persistence. Version 1 is
rejected outright: there is no compatibility reader, fallback, migration, or
dual-write path.

## Manifest publication

Each metadata snapshot is stored in an immutable file named
`MANIFEST-<20-digit generation>`. A binary header records the `ZVECMAN` magic,
disk-format version, header size, generation, JSON payload length, and CRC32C of
the payload. The JSON contains schema bytes, persisted segment capacity,
segment metadata, snapshot generations, the next segment ID, and the first
document ID reserved by the current writing segment. Readers require that
reserved document ID explicitly and reject manifests that omit it, preventing
ID reuse when a rewrite reclaims the highest deleted or superseded versions.

Optional segment index snapshots record the schema SHA-256, owning segment ID,
document count and bounds, and field/kind/path metadata for immutable artifacts
in `indexes/`. Every identity component must match before an artifact is opened.
When index metadata is absent, indexes rebuild from segment documents.

`CURRENT` is the commit point. It is itself framed and checksummed and names one
manifest. Publication writes and synchronizes the immutable manifest first,
then writes a synchronized temporary pointer and atomically installs or replaces
`CURRENT`. Directory metadata is synchronized where the operating system
supports it. Recovery reads only the manifest named by `CURRENT`; a higher
numbered orphan manifest is never treated as committed.

Manifest writers are serialized by `.version.lock`. A manager also verifies
that `CURRENT` still names the generation it opened, so stale writers fail with
a version conflict instead of losing a newer update. Published files are never
rewritten in place.

## Write-ahead log

A WAL begins with a 32-byte `ZVECWAL` header containing its codec version,
header size, maximum record size, reserved fields, and a CRC32C header checksum.
Each record has its own `ZREC` magic, codec version, header size, monotonically
increasing LSN, payload length, payload CRC32C, header CRC32C, and reserved
field. Payloads are non-empty and limited to 4 MiB.

Recovery validates the complete log before it can be appended. An incomplete
final record header or a header-valid but incomplete final payload is treated as
a crashed append and truncated back to the preceding record. Invalid magic,
versions, LSNs, lengths, reserved fields, header checksums, or the checksum of a
complete payload are reported as corruption and are never silently truncated.
Writers hold an exclusive sidecar advisory lock, and readers replay a stable
valid-prefix snapshot through an independent file handle. A read-only open
uses a shared lock and excludes an incomplete crash tail without truncating it;
the next writable open repairs that tail under the exclusive lock.

## Segments and collection snapshots

An immutable segment starts with a 64-byte `ZVECSEG` header containing the
codec version, segment and document-ID range, document count, payload length,
payload CRC32C, and header CRC32C. Records are stored in contiguous document-ID
order and contain the primary key plus an opaque schema-coded document payload.
Each record also checksums its key and payload. The first file listed for a
segment in the manifest is its data file. Index artifacts are referenced by the
top-level segment index snapshots so the segment data-file contract is stable.

The opaque document payload inside a segment is itself a versioned `ZVECDOC`
frame. Its header stores codec version, field count, payload length, payload
CRC32C, and header CRC32C. Fields are sorted by name and each entry records an
explicit public `DataType`, element count, and byte length. Scalar, array,
dense-vector, sparse-vector, and NULL encodings therefore round-trip without
JSON number coercion or ambiguous Go slices. Sparse coordinates are canonical
and corruption that changes their ordering is rejected.

Primary-key snapshots (`ZVECPK`) sort keys bytewise and map each key to a
segment/document location. Delete snapshots (`ZVECDEL`) store strictly sorted
global document IDs. Both use a common versioned header with item count,
payload length, payload CRC32C, and header CRC32C. Snapshots and segments are
written as immutable files and atomically installed without replacing an
existing generation.

## Collection recovery and flush

Every manifest names one empty writing segment and its WAL. Opening a
collection loads only the immutable segments and primary-key/delete snapshots
named by `CURRENT`, creates the writing segment at the next global document ID,
and replays the WAL's verified prefix in LSN order. Replayed operations must
match the writing segment, contiguous document IDs, and the recovered
primary-key state. A structurally valid but impossible operation is corruption,
not a request to repair metadata heuristically.

Flush first synchronizes the current WAL. For a non-empty writing segment it
then writes a new immutable segment, complete primary-key and deletion
snapshots, and the next empty WAL. Only after all those files are durable does
it publish a manifest that references them. `CURRENT` remains the sole commit
point: a crash before replacement recovers the old WAL, while a crash after
replacement has every file needed by the new version. An empty flush only
synchronizes the WAL and does not create a new manifest generation.

Artifact names are unique and immutable. ANN artifacts remain regular `.zvi`
files. FTS and INVERT artifacts are independent `.pebble` directories. After
segment publication, Flush builds vector, FTS, and INVERT artifacts only for
newly immutable segments and
publishes their metadata in another atomic manifest generation. Metadata and
files for unchanged segments are reused exactly. A failed retry never
overwrites an immutable file. Unreferenced artifacts and higher-numbered orphan
manifests are ignored during recovery.

Schema-changing data rewrites use the same commit protocol. The writer builds
new immutable segments from the complete live snapshot, writes fresh
primary-key and empty deletion snapshots, creates a fresh WAL, and publishes a
manifest that names all of them together with the new schema. Live document IDs
are preserved, including gaps, while the next writable ID remains monotonic.
Contiguous ID runs become independent segments. Before `CURRENT` changes, a
failure removes the new artifacts and the old schema, WAL, and segments remain
authoritative; after it changes, recovery sees only the complete rewritten
version. Superseded and deleted record versions are no longer referenced.

Optimize uses that rewrite protocol without changing the schema. Live
documents are split at document-ID gaps and at `MaxDocsPerSegment`; the new
primary-key snapshot maps the preserved IDs to their new segment IDs, the
delete snapshot is empty, and the writing segment starts at the same monotonic
next ID. After `CURRENT` commits, obsolete files matching the native segment,
WAL, WAL-lock, and snapshot naming schemes are removed and their directories
are synchronized. Unknown files and manifest generations are never selected
for pruning. A crash during pruning leaves only harmless unreferenced files;
even a no-op Optimize retries the cleanup.

HNSW, HNSW-RaBitQ, IVF, Vamana, DiskANN, and sparse HNSW retain their native
`.zvi` formats. FTS and INVERT use Pebble byte-ordered keys: document/row
bitmaps and posting lists are split into bounded chunks whose ordinal suffixes
must be contiguous. FTS keeps its independently checksummed compressed posting
payloads inside those chunks. Reopen validates metadata, chunk ordering,
posting domains, and dictionary statistics before publishing immutable memory.

Each Pebble artifact also has a wrapper-owned, synchronized `ZVEC-INDEX` marker;
the wrapper rejects a missing, corrupt, non-regular, or symlinked marker and
rejects a symlink in place of the artifact directory. Collection manifests
reference only completed directories. DiskANN retains its file or read-only
mapping for random access until the collection closes; the persisted collection
`EnableMmap` option selects that reader. Obsolete `indexes/*.zvi` files and
`.pebble` directories are pruned only after a newer manifest becomes
authoritative. Pruning refuses top-level symlinks and removes an obsolete
Pebble directory as one package-owned artifact.

## DDL and Optimize crash boundary

CreateIndex and DropIndex publish schema-only manifest generations after their
full live-snapshot validation completes. AddColumn, AlterColumn, DropColumn,
and Optimize publish a manifest only after every replacement segment,
primary-key/delete snapshot, and WAL has been installed and synchronized. All
six operations therefore share the same binary recovery rule: the old CURRENT
means the old schema and files remain authoritative; the new CURRENT means the
entire new version is authoritative. Recovery never combines a schema from one
generation with payloads or index parameters from another.

The crash suite holds the cross-process version lock while a child operation is
inside publication, verifies CURRENT is byte-for-byte unchanged, and kills the
child. Data-rewriting operations additionally prove their unreferenced new
segments are ignored. A complementary child is killed after CURRENT advances
but before its collection handle closes; reopen verifies the new schema,
payloads, document IDs, query results, and next write. Optimize also injects a
post-commit pruning failure and verifies that a no-op retry reclaims the
remaining obsolete artifacts without publishing another manifest.

`.collection.lock` controls handle ownership across processes. A writable
collection holds it exclusively for its lifetime. Read-only collections hold
shared locks, allowing multiple readers while preventing a concurrent writer.
Closing without Flush is safe because each successful mutation synchronizes
its WAL record before changing memory.

## WAL operations

Collection mutations inside WAL records use a separate `ZOP1` frame. It stores
the operation kind, target segment ID, assigned global document ID, primary-key
and document-payload lengths, and a CRC32C covering the header, key, and
payload. Insert reserves the next contiguous document ID, synchronizes this WAL
operation, and only then applies the document to the write segment and
primary-key map.
