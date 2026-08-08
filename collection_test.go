// Copyright 2026-present the zvec-go project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package zvec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/gofrs/flock"
	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/gorse-io/zvec/internal/core"
	"github.com/gorse-io/zvec/internal/db"
	dbsql "github.com/gorse-io/zvec/internal/db/sql"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCollectionCRUDFlushReopenAndReadOnly(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "books")
	schema := testPublicCollectionSchema()
	options := NewCollectionOptions()
	options.WALSyncEvery = 1
	collection, err := CreateAndOpen(ctx, path, schema, options)
	require.NoError(t, err)
	require.Equal(t, path, collection.Path())
	require.Equal(t, schema, collection.Schema())
	{
		got := collection.Options()
		require.True(t, got.EnableMmap)
		require.Equal(t, DefaultMaxBufferSize, got.MaxBufferSize)
	}

	documents := []Document{
		testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "bravo", "high", 2, 2, []float32{5, 0}),
	}
	inserted, err := collection.Insert(ctx, documents)
	require.NoError(t, err)
	require.True(t, inserted[0].DocID == 0)
	require.True(t, inserted[1].DocID == 1)
	{
		stats := collection.Stats()
		require.True(t, stats.DocumentCount == 2)
		require.True(t, stats.IndexCompleteness["embedding"] == 1)
		require.True(t, stats.IndexCompleteness["sparse"] == 1)
	}

	mixed, err := collection.Insert(ctx, []Document{
		documents[0],
		testPublicDocument("c", "charlie", "low", 3, 3, []float32{4, 0}),
	})
	var batchError *BatchWriteError
	require.ErrorAs(t, err, &batchError)
	require.True(t, batchError.Failed == 1)
	require.ErrorIs(t, err, ErrAlreadyExists)
	require.Error(t, mixed[0].Err)
	require.NoError(t, mixed[1].Err)
	require.True(t, mixed[1].DocID == 2)

	updated, err := collection.Update(ctx, []Document{{
		PrimaryKey: "a", Fields: map[string]any{"title": "alpha-updated", "category": nil},
	}})
	require.NoError(t, err)
	require.True(t, updated[0].DocID == 3)

	upserted, err := collection.Upsert(ctx, []Document{{
		PrimaryKey: "b", Fields: map[string]any{"rating": int32(20)},
	}})
	require.NoError(t, err)
	require.True(t, upserted[0].DocID == 4)

	invalidUpsert, err := collection.Upsert(ctx, []Document{{PrimaryKey: "new", Fields: map[string]any{"title": "new"}}})
	require.ErrorIs(t, err, ErrInvalidArgument)
	require.Error(t, invalidUpsert[0].Err)

	fetched, err := collection.Fetch(ctx, []string{"a", "b", "missing"}, Projection{IncludeVectors: true})
	require.NoError(t, err)
	require.Nil(t, fetched[2])
	require.True(t, fetched[0].Fields["title"] == "alpha-updated")
	require.Nil(t, fetched[0].Fields["category"])
	require.Equal(t, int32(1), fetched[0].Fields["rating"])
	{
		_, found := fetched[0].Fields["embedding"]
		require.True(t, found)
	}
	require.Equal(t, int32(20), fetched[1].Fields["rating"])
	require.True(t, fetched[1].Fields["title"] == "bravo")

	deleted, err := collection.Delete(ctx, []string{"c", "missing"})
	require.ErrorIs(t, err, ErrNotFound)
	require.NoError(t, deleted[0].Err)
	require.Error(t, deleted[1].Err)
	{
		got := collection.Stats().DocumentCount
		require.True(t, got == 2)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}
	{
		_, err := collection.Fetch(ctx, []string{"a"}, Projection{})
		require.ErrorIs(t, err, ErrFailedPrecondition)
	}

	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	readOnly, err := Open(ctx, path, readOnlyOptions)
	require.NoError(t, err)

	defer readOnly.Close()
	{
		got := readOnly.Stats().DocumentCount
		require.True(t, got == 2)
	}

	fetched, err = readOnly.Fetch(ctx, []string{"a"}, Projection{})
	require.NoError(t, err)
	require.True(t, fetched[0].DocID == 3)
	{
		_, found := fetched[0].Fields["embedding"]
		require.False(t, found)
	}
	{
		_, err := readOnly.Insert(ctx, []Document{testPublicDocument("x", "xray", "low", 1, 1, []float32{1, 0})})
		require.ErrorIs(t, err, ErrPermissionDenied)
	}
}

func TestCollectionDenseSparseRadiusProjectionAndGroupBy(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "query"), testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	documents := []Document{
		testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "bravo", "high", 2, 5, []float32{5, 0}),
		testPublicDocument("c", "charlie", "low", 3, 4, []float32{4, 0}),
		testPublicDocument("d", "delta", "high", 4, 3, []float32{3, 0}),
		testPublicDocument("e", "echo", "", 5, 6, []float32{6, 0}),
	}
	documents[4].Fields["category"] = nil
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	params := NewFlatQueryParams()
	params.Radius = 4
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
		Params: params, Projection: Projection{OutputFields: []string{"title"}},
	})
	require.NoError(t, err)
	{
		got := documentKeys(results)
		require.Equal(t, []string{"e", "b", "c"}, got)
	}
	require.Equal(t, map[string]any{"title": "echo"}, results[0].Fields)
	require.True(t, results[0].Score == 6)

	sparseResults, err := collection.Query(ctx, VectorQuery{
		Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}},
		TopK: 3, Projection: Projection{OutputFields: []string{}},
	})
	require.NoError(t, err)
	{
		got := documentKeys(sparseResults)
		require.Equal(t, []string{"e", "b", "c"}, got)
	}
	require.NotNil(t, sparseResults[0].Fields)
	require.Len(t, sparseResults[0].Fields, 0)

	groups, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0},
		GroupByField: "category", GroupCount: 3, TopKPerGroup: 2,
		Projection: Projection{OutputFields: []string{"title"}},
	})
	require.NoError(t, err)
	require.Len(t, groups, 3)
	require.True(t, groups[0].Value == "")
	require.True(t, groups[1].Value == "high")
	require.True(t, groups[2].Value == "low")
	{
		got := documentKeys(groups[1].Documents)
		require.Equal(t, []string{"b", "d"}, got)
	}
	{
		got := documentKeys(groups[2].Documents)
		require.Equal(t, []string{"c", "a"}, got)
	}

	filtered, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1, Filter: "rating > 1"})
	require.NoError(t, err)
	require.Equal(t, []string{"e"}, documentKeys(filtered))
	{
		_, err := collection.Query(ctx, VectorQuery{Field: "embedding", SparseVector: SparseVectorFP32{}, TopK: 1})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		_, err := collection.GroupByQuery(ctx, GroupByVectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, GroupByField: "sparse", GroupCount: 1, TopKPerGroup: 1})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
}

func TestCollectionUnifiedQueryTargets(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "unified"), testMultiQuerySchema(), NewCollectionOptions())
	require.NoError(t, err)
	defer collection.Close()
	_, err = collection.Insert(ctx, testMultiQueryDocuments())
	require.NoError(t, err)

	filtered, err := collection.Query(ctx, VectorQuery{
		TopK: 2, Filter: "rating >= 2", Projection: Projection{OutputFields: []string{"rating"}},
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, documentKeys(filtered))
	require.Zero(t, filtered[0].Score)
	require.Equal(t, map[string]any{"rating": int32(2)}, filtered[0].Fields)

	fts, err := collection.Query(ctx, VectorQuery{
		Field: "title", FTS: &FTSClause{Match: "go"}, TopK: 3, Filter: "category = 'keep'",
		Projection: Projection{OutputFields: []string{"title"}},
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "a"}, documentKeys(fts))
	require.Greater(t, fts[0].Score, fts[1].Score)

	byDenseID, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", PrimaryKey: "c", TopK: 2,
		Projection: Projection{OutputFields: []string{}},
	})
	require.NoError(t, err)
	require.Equal(t, []string{"c", "a"}, documentKeys(byDenseID))

	bySparseID, err := collection.Query(ctx, VectorQuery{
		Field: "sparse", PrimaryKey: "b", TopK: 2,
	})
	require.NoError(t, err)
	require.Equal(t, []string{"c", "b"}, documentKeys(bySparseID))

	groups, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
		Field: "embedding", PrimaryKey: "a", GroupByField: "category",
		GroupCount: 2, TopKPerGroup: 1,
	})
	require.NoError(t, err)
	require.Len(t, groups, 2)
	require.Equal(t, "drop", groups[0].Value)
	require.Equal(t, []string{"c"}, documentKeys(groups[0].Documents))

	_, err = collection.Query(ctx, VectorQuery{Field: "embedding", PrimaryKey: "missing", TopK: 1})
	require.ErrorIs(t, err, ErrNotFound)
	_, err = collection.Query(ctx, VectorQuery{
		Field: "embedding", PrimaryKey: "a", DenseVector: VectorFP32{1, 0}, TopK: 1,
	})
	require.ErrorIs(t, err, ErrInvalidArgument)
	_, err = collection.Query(ctx, VectorQuery{Field: "embedding", TopK: 1})
	require.ErrorIs(t, err, ErrInvalidArgument)
	_, err = collection.Query(ctx, VectorQuery{TopK: 1, Params: NewFlatQueryParams()})
	require.ErrorIs(t, err, ErrInvalidArgument)
}

func TestCollectionMultiQueryPrimaryKeyTarget(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "multi-id"), testMultiQuerySchema(), NewCollectionOptions())
	require.NoError(t, err)
	defer collection.Close()
	_, err = collection.Insert(ctx, testMultiQueryDocuments())
	require.NoError(t, err)

	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "embedding", PrimaryKey: "c", NumCandidates: 2},
			{Field: "title", FTS: &FTSClause{Match: "go"}, NumCandidates: 2},
		},
		TopK: 2,
		Reranker: testRerankerFunc(func(_ context.Context, batches []RerankBatch, _ int) ([]Document, error) {
			require.Equal(t, []string{"c", "a"}, documentKeys(batches[0].Documents))
			require.Equal(t, []string{"b", "a"}, documentKeys(batches[1].Documents))
			return []Document{batches[0].Documents[0], batches[1].Documents[0]}, nil
		}),
	}
	results, err := collection.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"c", "b"}, documentKeys(results))

	query.Queries[0] = SubQuery{Field: "embedding", PrimaryKey: "missing"}
	_, err = collection.MultiQuery(ctx, query)
	require.ErrorIs(t, err, ErrNotFound)
	query.Queries[0] = SubQuery{Field: "embedding", PrimaryKey: "a", DenseVector: VectorFP32{1, 0}}
	_, err = collection.MultiQuery(ctx, query)
	require.ErrorIs(t, err, ErrInvalidArgument)
}

func TestCollectionRuntimeIndexesAreReusedUntilSnapshotChanges(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "runtime-cache"), testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)
	defer collection.Close()
	_, err = collection.Insert(ctx, []Document{
		testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "bravo", "high", 2, 2, []float32{2, 0}),
	})
	require.NoError(t, err)

	query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 2}
	_, err = collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, uint64(1), collection.indexBuildCount)
	_, err = collection.Query(ctx, query)
	require.NoError(t, err)
	_, err = collection.Query(ctx, VectorQuery{TopK: 1, Filter: "rating >= 1"})
	require.NoError(t, err)
	require.Equal(t, uint64(1), collection.indexBuildCount)

	_, err = collection.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"rating": int32(3)}}})
	require.NoError(t, err)
	_, err = collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, uint64(2), collection.indexBuildCount)
	_, err = collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, uint64(2), collection.indexBuildCount)
}

func TestCollectionPersistsAndReopensSnapshotIndexes(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "persisted-indexes")
	hnsw := NewHNSWIndexParams(MetricTypeIP)
	hnsw.M, hnsw.EFConstruction = 4, 16
	sparseHNSW := NewHNSWIndexParams(MetricTypeIP)
	sparseHNSW.M, sparseHNSW.EFConstruction = 4, 16
	schema := NewCollectionSchema("persisted_indexes",
		FieldSchema{Name: "text", DataType: DataTypeString, Index: NewFTSIndexParams()},
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Index: NewInvertIndexParams()},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: hnsw},
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: sparseHNSW},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	_, err = collection.Insert(ctx, []Document{
		{PrimaryKey: "a", Fields: map[string]any{"text": "red apple", "rating": int32(1), "embedding": VectorFP32{1, 0}, "sparse": SparseVectorFP32{Indices: []uint32{1}, Values: []float32{1}}}},
		{PrimaryKey: "b", Fields: map[string]any{"text": "green apple", "rating": int32(2), "embedding": VectorFP32{0.8, 0.2}, "sparse": SparseVectorFP32{Indices: []uint32{1}, Values: []float32{0.8}}}},
		{PrimaryKey: "c", Fields: map[string]any{"text": "blue berry", "rating": int32(3), "embedding": VectorFP32{0, 1}, "sparse": SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}}}},
	})
	require.NoError(t, err)
	require.NoError(t, collection.Flush(ctx))

	manifest := collection.store.Manifest()
	require.Len(t, manifest.SegmentIndexSnapshots, 1)
	snapshot := manifest.SegmentIndexSnapshots[0]
	require.Equal(t, manifest.PersistedSegments[0].ID, snapshot.SegmentID)
	require.Equal(t, uint64(3), snapshot.DocumentCount)
	require.Len(t, snapshot.Artifacts, 4)
	artifactKinds := make(map[string]string)
	for _, artifact := range snapshot.Artifacts {
		artifactKinds[artifact.Field] = artifact.Kind
		info, statErr := os.Stat(filepath.Join(path, filepath.FromSlash(artifact.File)))
		require.NoError(t, statErr)
		if artifact.Kind == collectionFTSArtifactKind || artifact.Kind == collectionInvertArtifactKind {
			require.True(t, info.IsDir())
			require.Equal(t, ".pebble", filepath.Ext(artifact.File))
		} else {
			require.True(t, info.Mode().IsRegular())
			require.Equal(t, ".zvi", filepath.Ext(artifact.File))
		}
	}
	require.Equal(t, collectionFTSArtifactKind, artifactKinds["text"])
	require.Equal(t, collectionInvertArtifactKind, artifactKinds["rating"])
	require.Equal(t, collectionVectorArtifactKind(IndexTypeHNSW), artifactKinds["embedding"])
	require.Equal(t, collectionVectorArtifactKind(IndexTypeHNSW), artifactKinds["sparse"])
	require.NoError(t, collection.Close())

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)
	vectorResults, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 2, Filter: "rating >= 2"})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, documentKeys(vectorResults))
	ftsResults, err := collection.Query(ctx, VectorQuery{Field: "text", FTS: &FTSClause{Match: "apple"}, TopK: 10})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b"}, documentKeys(ftsResults))
	filterResults, err := collection.Query(ctx, VectorQuery{Filter: "rating >= 2", TopK: 10})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, documentKeys(filterResults))
	sparseResults, err := collection.Query(ctx, VectorQuery{Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{1}, Values: []float32{1}}, TopK: 2})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b"}, documentKeys(sparseResults))
	require.NoError(t, collection.Close())

	var invertPath string
	for _, artifact := range snapshot.Artifacts {
		if artifact.Kind == collectionInvertArtifactKind {
			invertPath = filepath.Join(path, filepath.FromSlash(artifact.File))
		}
	}
	require.NotEmpty(t, invertPath)
	require.NoError(t, os.WriteFile(filepath.Join(invertPath, "ZVEC-INDEX"), []byte("corrupt"), 0o600))
	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)
	defer collection.Close()
	_, err = collection.Query(ctx, VectorQuery{Filter: "rating >= 2", TopK: 10})
	require.ErrorIs(t, err, dbsql.ErrCorruptInvertedIndex)
}

func TestCollectionRebuildsMissingOptionalSegmentIndexSnapshots(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "missing-index-metadata")
	hnsw := NewHNSWIndexParams(MetricTypeIP)
	hnsw.M, hnsw.EFConstruction = 4, 16
	schema := NewCollectionSchema("missing_index_metadata",
		FieldSchema{Name: "text", DataType: DataTypeString, Index: NewFTSIndexParams()},
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Index: NewInvertIndexParams()},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: hnsw},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	_, err = collection.Insert(ctx, []Document{
		{PrimaryKey: "a", Fields: map[string]any{"text": "red apple", "rating": int32(1), "embedding": VectorFP32{1, 0}}},
		{PrimaryKey: "b", Fields: map[string]any{"text": "green apple", "rating": int32(2), "embedding": VectorFP32{0.8, 0.2}}},
	})
	require.NoError(t, err)
	require.NoError(t, collection.Flush(ctx))
	require.NotEmpty(t, collection.store.Manifest().SegmentIndexSnapshots)
	require.NoError(t, collection.Close())

	versions, err := db.OpenVersionManager(ctx, path)
	require.NoError(t, err)
	manifest, err := versions.Update(ctx, func(manifest *db.Manifest) error {
		manifest.SegmentIndexSnapshots = nil
		return nil
	})
	require.NoError(t, err)
	require.Equal(t, uint64(2), manifest.WritingSegmentStartDocID)
	require.Empty(t, manifest.SegmentIndexSnapshots)
	require.NoError(t, os.RemoveAll(filepath.Join(path, "indexes")))

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)
	defer collection.Close()
	vectorResults, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 2})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b"}, documentKeys(vectorResults))
	ftsResults, err := collection.Query(ctx, VectorQuery{Field: "text", FTS: &FTSClause{Match: "apple"}, TopK: 10})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b"}, documentKeys(ftsResults))
	filterResults, err := collection.Query(ctx, VectorQuery{Filter: "rating >= 2", TopK: 10})
	require.NoError(t, err)
	require.Equal(t, []string{"b"}, documentKeys(filterResults))
	require.NoError(t, collection.Flush(ctx))
	require.NotEmpty(t, collection.store.Manifest().SegmentIndexSnapshots)
}

func TestCollectionSegmentNativeIndexesAreIncremental(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "segment-native-indexes")
	hnsw := NewHNSWIndexParams(MetricTypeIP)
	hnsw.M, hnsw.EFConstruction = 4, 16
	schema := NewCollectionSchema("segment_native_indexes",
		FieldSchema{Name: "text", DataType: DataTypeString, Index: NewFTSIndexParams()},
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Index: NewInvertIndexParams()},
		FieldSchema{Name: "category", DataType: DataTypeString},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: hnsw},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	_, err = collection.Insert(ctx, []Document{{
		PrimaryKey: "a", Fields: map[string]any{
			"text": "red apple", "rating": int32(1), "category": "red", "embedding": VectorFP32{1, 0},
		},
	}})
	require.NoError(t, err)
	require.NoError(t, collection.Flush(ctx))
	firstSnapshot := collection.store.Manifest().SegmentIndexSnapshots[0]
	require.Len(t, firstSnapshot.Artifacts, 3)

	_, err = collection.Insert(ctx, []Document{{
		PrimaryKey: "b", Fields: map[string]any{
			"text": "green apple", "rating": int32(2), "category": "green", "embedding": VectorFP32{0.8, 0.2},
		},
	}})
	require.NoError(t, err)
	require.NoError(t, collection.Flush(ctx))
	manifest := collection.store.Manifest()
	require.Len(t, manifest.SegmentIndexSnapshots, 2)
	require.Equal(t, firstSnapshot, manifest.SegmentIndexSnapshots[0], "publishing a new segment rebuilt the old segment artifacts")
	require.NoError(t, collection.Close())

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)
	defer collection.Close()
	results, err := collection.Query(ctx, VectorQuery{Field: "text", FTS: &FTSClause{Match: "apple"}, TopK: 10})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b"}, documentKeys(results))
	require.Equal(t, uint64(2), collection.indexBuildCount)
	firstRuntime := collection.segmentIndexes[manifest.PersistedSegments[0].ID]
	secondRuntime := collection.segmentIndexes[manifest.PersistedSegments[1].ID]
	require.NotNil(t, firstRuntime)
	require.NotNil(t, secondRuntime)

	_, err = collection.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{
		"text": "red berry", "rating": int32(3),
	}}})
	require.NoError(t, err)
	results, err = collection.Query(ctx, VectorQuery{Field: "text", FTS: &FTSClause{Match: "apple"}, TopK: 10})
	require.NoError(t, err)
	require.Equal(t, []string{"b"}, documentKeys(results), "stale segment versions must be filtered before BM25 merge")
	require.Equal(t, uint64(3), collection.indexBuildCount, "only the new mutable segment should be built")
	require.Same(t, firstRuntime, collection.segmentIndexes[manifest.PersistedSegments[0].ID])
	require.Same(t, secondRuntime, collection.segmentIndexes[manifest.PersistedSegments[1].ID])

	results, err = collection.Query(ctx, VectorQuery{TopK: 10, Filter: "rating >= 3"})
	require.NoError(t, err)
	require.Equal(t, []string{"a"}, documentKeys(results))
	groups, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, GroupByField: "category",
		GroupCount: 2, TopKPerGroup: 1,
	})
	require.NoError(t, err)
	require.Len(t, groups, 2)
}

func TestCollectionScalarQuantizedDiskANNDirectSchema(t *testing.T) {
	ctx := context.Background()
	for _, quantize := range []QuantizeType{QuantizeTypeFP16, QuantizeTypeInt8, QuantizeTypeInt4} {
		t.Run(quantize.String(), func(t *testing.T) {
			diskANN := NewDiskANNIndexParams(MetricTypeL2)
			diskANN.MaxDegree, diskANN.ListSize, diskANN.PQChunks = 4, 8, 2
			diskANN.Quantize = quantize
			diskANN.Quantizer.EnableRotate = quantize != QuantizeTypeFP16
			schema := NewCollectionSchema("later", FieldSchema{
				Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: diskANN,
			})
			collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "later"), schema, NewCollectionOptions())
			require.NoError(t, err)

			defer collection.Close()
			{
				_, err := collection.Insert(ctx, []Document{
					{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP32{0, 0, 0, 0}}},
					{PrimaryKey: "b", Fields: map[string]any{"embedding": VectorFP32{1.013, .031, -.077, .125}}},
					{PrimaryKey: "c", Fields: map[string]any{"embedding": VectorFP32{3.117, -.271, .051, -.2}}},
				})
				require.NoError(t, err)
			}

			results, err := collection.Query(ctx, VectorQuery{
				Field: "embedding", DenseVector: VectorFP32{.9, .02, -.04, .1}, TopK: 3,
			})
			require.NoError(t, err)
			require.Equal(t, []string{"b", "a", "c"}, documentKeys(results))
			{
				got := collection.Stats().IndexCompleteness["embedding"]
				require.True(t, got == 1)
			}
		})
	}
}

func TestCollectionReplaysPublicDocumentPayloadWithoutFlush(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "recovery")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{testPublicDocument("a", "alpha", "low", 1, 2, []float32{2, 0})})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	reopened, err := Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer reopened.Close()
	results, err := reopened.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1,
		Projection: Projection{IncludeVectors: true},
	})
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.True(t, results[0].PrimaryKey == "a")
	require.True(t, results[0].Score == 2)
}

func TestCollectionDestroyAndArgumentErrors(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "destroy")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		err := collection.Destroy(ctx)
		require.NoError(t, err)
	}
	{
		_, err := os.Stat(path)
		require.ErrorIs(t, err, os.ErrNotExist)
	}
	{
		_, err := Open(ctx, path, NewCollectionOptions())
		require.ErrorIs(t, err, ErrNotFound)
	}
	{
		_, err := CreateAndOpen(nil, path, testPublicCollectionSchema(), NewCollectionOptions())
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	var nilCollection *Collection
	{
		_, err := nilCollection.Insert(ctx, []Document{{PrimaryKey: "a"}})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		_, err := nilCollection.Query(ctx, VectorQuery{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
}

func TestEncodeGroupValueMatchesPinnedFormatting(t *testing.T) {
	tests := []struct {
		value    any
		dataType DataType
		want     string
	}{
		{value: nil, dataType: DataTypeString, want: ""},
		{value: true, dataType: DataTypeBool, want: "true"},
		{value: int32(-2), dataType: DataTypeInt32, want: "-2"},
		{value: uint64(9), dataType: DataTypeUint64, want: "9"},
		{value: float32(1.25), dataType: DataTypeFloat, want: "1.250000"},
		{value: 2.5, dataType: DataTypeDouble, want: "2.500000"},
	}
	for _, testCase := range tests {
		got, err := encodeGroupValue(testCase.value, testCase.dataType)
		require.NoError(t, err)
		require.Equal(t, testCase.want, got)
	}
}

func testPublicCollectionSchema() CollectionSchema {
	schema := NewCollectionSchema("books",
		FieldSchema{Name: "title", DataType: DataTypeString},
		FieldSchema{Name: "category", DataType: DataTypeString, Nullable: true},
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Nullable: true},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func testPublicDocument(primaryKey, title, category string, rating int32, score float32, dense []float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"title": title, "category": category, "rating": rating,
		"embedding": VectorFP32(dense),
		"sparse":    SparseVectorFP32{Indices: []uint32{2}, Values: []float32{score}},
	}}
}

func documentKeys(documents []Document) []string {
	keys := make([]string, len(documents))
	for index := range documents {
		keys[index] = documents[index].PrimaryKey
	}
	return keys
}

func TestAddColumnBackfillsAtomicallyAndSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "add-column")
	schema := addColumnSchema()
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := []Document{
		addColumnDocument("a", 1, []float32{3, 0}),
		addColumnDocument("b", 2, []float32{2, 0}),
		addColumnDocument("c", 3, []float32{1, 0}),
	}
	inserted, err := collection.Insert(ctx, documents)
	require.NoError(t, err)

	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "b", Fields: map[string]any{"count": int32(4)}}})
	require.NoError(t, err)

	beforeIDs := map[string]uint64{"a": inserted[0].DocID, "c": inserted[2].DocID, "b": updated[0].DocID}
	initialGeneration := collection.store.Manifest().Generation
	index := NewInvertIndexParams()
	index.EnableRangeOptimization = true
	field := FieldSchema{Name: "derived", DataType: DataTypeInt64, Index: index}
	{
		err := collection.AddColumn(ctx, field, "(count * 2) + 1", AddColumnOptions{Concurrency: 3})
		require.NoError(t, err)
	}
	require.True(t, collection.store.Manifest().Generation > initialGeneration,
		"AddColumn did not publish a new manifest generation")
	require.True(t, collection.Stats().DocumentCount == 3)

	fetched, err := collection.Fetch(ctx, []string{"a", "b", "c"}, Projection{})
	require.NoError(t, err)

	wantDerived := []int64{3, 9, 7}
	for index, document := range fetched {
		require.NotNil(t, document)
		require.Equal(t, beforeIDs[document.PrimaryKey], document.DocID)
		require.Equal(t, wantDerived[index], document.Fields["derived"])
	}
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
		Filter: "derived >= 7", Projection: Projection{OutputFields: []string{"derived"}},
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, documentKeys(results))
	{
		err := collection.AddColumn(ctx, FieldSchema{Name: "optional", DataType: DataTypeFloat, Nullable: true}, "", AddColumnOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	fetched, err = collection.Fetch(ctx, []string{"a", "b", "c"}, Projection{OutputFields: []string{"optional"}})
	require.NoError(t, err)

	for _, document := range fetched {
		value, found := document.Fields["optional"]
		require.True(t, found)
		require.Nil(t, value)
		require.Equal(t, beforeIDs[document.PrimaryKey], document.DocID)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	fetched, err = collection.Fetch(ctx, []string{"a", "b", "c"}, Projection{})
	require.NoError(t, err)

	for index, document := range fetched {
		require.NotNil(t, document)
		require.Equal(t, wantDerived[index], document.Fields["derived"])
		{
			value, found := document.Fields["optional"]
			require.True(t, found)
			require.Nil(t, value)
		}
	}
	missing := addColumnDocument("missing", 5, []float32{1, 0})
	{
		_, err := collection.Insert(ctx, []Document{missing})
		require.Error(t, err,
			"insert without added non-nullable field succeeded")
	}

	missing.Fields["derived"] = int64(11)
	missing.Fields["optional"] = nil
	{
		_, err := collection.Insert(ctx, []Document{missing})
		require.NoError(t, err)
	}
}

func TestAddColumnValidationAndFailureRollback(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	{
		err := nilCollection.AddColumn(ctx, FieldSchema{}, "", AddColumnOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	path := filepath.Join(t.TempDir(), "add-column-errors")
	collection, err := CreateAndOpen(ctx, path, addColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		err := collection.AddColumn(nil, FieldSchema{Name: "nil_ctx", DataType: DataTypeInt32}, "1", AddColumnOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		_, err := collection.Insert(ctx, []Document{addColumnDocument("one", 2, []float32{1, 0})})
		require.NoError(t, err)
	}

	initialGeneration := collection.store.Manifest().Generation
	tests := []struct {
		name       string
		field      FieldSchema
		expression string
		options    AddColumnOptions
	}{
		{"unsupported type", FieldSchema{Name: "text", DataType: DataTypeString, Nullable: true}, "", AddColumnOptions{}},
		{"non-nullable without expression", FieldSchema{Name: "required", DataType: DataTypeInt32}, "", AddColumnOptions{}},
		{"duplicate", FieldSchema{Name: "count", DataType: DataTypeInt32}, "count", AddColumnOptions{}},
		{"missing reference", FieldSchema{Name: "missing_ref", DataType: DataTypeInt32}, "missing + 1", AddColumnOptions{}},
		{"syntax", FieldSchema{Name: "syntax", DataType: DataTypeInt32}, "count +", AddColumnOptions{}},
		{"evaluation", FieldSchema{Name: "divide", DataType: DataTypeInt32}, "count / 0", AddColumnOptions{}},
		{"negative concurrency", FieldSchema{Name: "workers", DataType: DataTypeInt32}, "1", AddColumnOptions{Concurrency: -1}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			{
				err := collection.AddColumn(ctx, test.field, test.expression, test.options)
				require.Error(t, err,
					"AddColumn succeeded")
			}
			{
				_, found := collection.Schema().Field(test.field.Name)
				require.False(t, found && test.field.Name != "count")
			}
			{
				got := collection.store.Manifest().Generation
				require.Equal(t, initialGeneration, got)
			}
		})
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.AddColumn(canceled, FieldSchema{Name: "canceled", DataType: DataTypeInt32}, "1", AddColumnOptions{})
		require.ErrorIs(t, err, context.Canceled)
	}

	fetched, err := collection.Fetch(ctx, []string{"one"}, Projection{IncludeVectors: true})
	require.NoError(t, err)
	require.NotNil(t, fetched[0])
	require.Equal(t, addColumnDocument("one", 2, []float32{1, 0}).Fields, fetched[0].Fields)
}

func TestAddColumnEmptyCollectionMatchesDeferredExpressionBehavior(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "add-column-empty")
	collection, err := CreateAndOpen(ctx, path, addColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		err := collection.AddColumn(ctx, FieldSchema{Name: "deferred", DataType: DataTypeInt32}, "CASE WHEN count > 0 THEN 1 END", AddColumnOptions{})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	document := addColumnDocument("one", 1, []float32{1, 0})
	document.Fields["deferred"] = int32(7)
	{
		_, err := collection.Insert(ctx, []Document{document})
		require.NoError(t, err)
	}
}

func TestAddColumnRejectsReadOnlyHandle(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "add-column-read-only")
	collection, err := CreateAndOpen(ctx, path, addColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	options := NewCollectionOptions()
	options.ReadOnly = true
	collection, err = Open(ctx, path, options)
	require.NoError(t, err)

	defer collection.Close()
	err = collection.AddColumn(ctx, FieldSchema{Name: "new", DataType: DataTypeInt32}, "1", AddColumnOptions{})
	require.ErrorIs(t, err, ErrPermissionDenied)
}

func addColumnSchema() CollectionSchema {
	schema := NewCollectionSchema("add_columns",
		FieldSchema{Name: "count", DataType: DataTypeInt32},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func addColumnDocument(primaryKey string, count int32, embedding []float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"count": count, "embedding": VectorFP32(embedding),
	}}
}

func TestAlterColumnMigratesNamesTypesIndexesAndReopens(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "alter-column")
	collection, err := CreateAndOpen(ctx, path, alterColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	inserted, err := collection.Insert(ctx, []Document{
		alterColumnDocument("a", 1, float32(1.75), true, math.MaxUint32, []float32{3, 0}),
		alterColumnDocument("b", 2, nil, true, 10, []float32{2, 0}),
		alterColumnDocument("c", 3, nil, false, 20, []float32{1, 0}),
	})
	require.NoError(t, err)

	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "b", Fields: map[string]any{"count": int32(4)}}})
	require.NoError(t, err)

	wantIDs := map[string]uint64{"a": inserted[0].DocID, "b": updated[0].DocID, "c": inserted[2].DocID}

	invert := NewInvertIndexParams()
	invert.EnableRangeOptimization = true
	total := FieldSchema{Name: "total", DataType: DataTypeInt64, Index: invert}
	{
		err := collection.AlterColumn(ctx, "count", "", &total, AlterColumnOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	invert.EnableRangeOptimization = false
	storedTotal, _ := collection.Schema().Field("total")
	require.True(t, storedTotal.Index.(InvertIndexParams).EnableRangeOptimization,
		"AlterColumn retained caller-owned index parameters")
	{
		err := collection.AlterColumn(ctx, "maybe", "cost", nil, AlterColumnOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	price := FieldSchema{Name: "price", DataType: DataTypeDouble, Nullable: true}
	{
		err := collection.AlterColumn(ctx, "cost", "", &price, AlterColumnOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	capField := FieldSchema{Name: "cap", DataType: DataTypeInt32}
	{
		err := collection.AlterColumn(ctx, "cap", "", &capField, AlterColumnOptions{Concurrency: 1})
		require.NoError(t, err)
	}

	schema := collection.Schema()
	for _, old := range []string{"count", "maybe", "cost"} {
		{
			_, found := schema.Field(old)
			require.False(t, found)
		}
	}
	for _, current := range []string{"total", "price", "cap"} {
		{
			_, found := schema.Field(current)
			require.True(t, found)
		}
	}
	fetched, err := collection.Fetch(ctx, []string{"a", "b", "c"}, Projection{})
	require.NoError(t, err)

	wantTotals := []int64{1, 4, 3}
	wantCaps := []int32{-1, 10, 20}
	for index, document := range fetched {
		require.NotNil(t, document)
		require.Equal(t, wantIDs[document.PrimaryKey], document.DocID)
		require.Equal(t, wantTotals[index], document.Fields["total"])
		require.Equal(t, wantCaps[index], document.Fields["cap"])
	}
	require.Equal(t, float64(1.75), fetched[0].Fields["price"])
	{
		value, found := fetched[1].Fields["price"]
		require.True(t, found)
		require.Nil(t, value)
	}
	{
		_, found := fetched[2].Fields["price"]
		require.False(t, found)
	}

	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
		Filter: "total >= 3", Projection: Projection{OutputFields: []string{"total"}},
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, documentKeys(results))
	require.True(t, collection.Stats().DocumentCount == 3)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	fetched, err = collection.Fetch(ctx, []string{"a", "b", "c"}, Projection{})
	require.NoError(t, err)
	require.Equal(t, int64(1), fetched[0].Fields["total"])
	require.Equal(t, float64(1.75), fetched[0].Fields["price"])

	document := Document{PrimaryKey: "d", Fields: map[string]any{
		"total": int64(5), "price": float64(2.5), "cap": int32(30), "text": "d",
		"embedding": VectorFP32{0.5, 0},
	}}
	{
		_, err := collection.Insert(ctx, []Document{document})
		require.NoError(t, err)
	}
}

func TestAlterColumnValidationAndPublicationRollback(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	{
		err := nilCollection.AlterColumn(ctx, "count", "renamed", nil, AlterColumnOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	path := filepath.Join(t.TempDir(), "alter-column-errors")
	collection, err := CreateAndOpen(ctx, path, alterColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{alterColumnDocument("one", 2, nil, true, 3, []float32{1, 0})})
		require.NoError(t, err)
	}
	{
		err := collection.AlterColumn(nil, "count", "renamed", nil, AlterColumnOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	initialSchema := collection.Schema()
	initialGeneration := collection.store.Manifest().Generation
	tests := []struct {
		name    string
		column  string
		rename  string
		field   *FieldSchema
		options AlterColumnOptions
	}{
		{"empty column", "", "renamed", nil, AlterColumnOptions{}},
		{"missing column", "missing", "renamed", nil, AlterColumnOptions{}},
		{"both forms", "count", "renamed", &FieldSchema{Name: "count", DataType: DataTypeInt64}, AlterColumnOptions{}},
		{"neither form", "count", "", nil, AlterColumnOptions{}},
		{"rename same", "count", "count", nil, AlterColumnOptions{}},
		{"rename duplicate", "count", "cap", nil, AlterColumnOptions{}},
		{"invalid rename", "count", "bad name", nil, AlterColumnOptions{}},
		{"old type unsupported", "text", "renamed_text", nil, AlterColumnOptions{}},
		{"new type unsupported", "count", "", &FieldSchema{Name: "count", DataType: DataTypeString}, AlterColumnOptions{}},
		{"nullable to required", "maybe", "", &FieldSchema{Name: "maybe", DataType: DataTypeFloat}, AlterColumnOptions{}},
		{"replacement duplicate", "count", "", &FieldSchema{Name: "cap", DataType: DataTypeInt64}, AlterColumnOptions{}},
		{"negative concurrency", "count", "renamed", nil, AlterColumnOptions{Concurrency: -1}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			{
				err := collection.AlterColumn(ctx, test.column, test.rename, test.field, test.options)
				require.Error(t, err,
					"AlterColumn succeeded")
			}
			require.Equal(t, initialSchema, collection.Schema())
			{
				got := collection.store.Manifest().Generation
				require.Equal(t, initialGeneration, got)
			}
		})
	}
	equal := initialSchema.Fields[0].Clone()
	{
		err := collection.AlterColumn(ctx, "count", "", &equal, AlterColumnOptions{})
		require.NoError(t, err)
	}
	{
		got := collection.store.Manifest().Generation
		require.Equal(t, initialGeneration, got)
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.AlterColumn(canceled, "count", "renamed", nil, AlterColumnOptions{})
		require.ErrorIs(t, err, context.Canceled)
	}

	versionLock := flock.New(filepath.Join(path, ".version.lock"))
	locked, err := versionLock.TryLock()
	require.NoError(t, err)
	require.True(t, locked)

	deadline, cancel := context.WithTimeout(ctx, 75*time.Millisecond)
	err = collection.AlterColumn(deadline, "count", "renamed", nil, AlterColumnOptions{Concurrency: 2})
	cancel()
	if !errors.Is(err, context.DeadlineExceeded) {
		_ = versionLock.Close()
	}
	require.ErrorIs(t, err, context.DeadlineExceeded)
	{
		err := versionLock.Close()
		require.NoError(t, err)
	}
	require.Equal(t, initialSchema, collection.Schema())
	require.Equal(t, initialGeneration, collection.store.Manifest().Generation)

	fetched, err := collection.Fetch(ctx, []string{"one"}, Projection{})
	require.NoError(t, err)
	require.NotNil(t, fetched[0])
	require.Equal(t, int32(2), fetched[0].Fields["count"])
}

func TestAlterColumnRejectsReadOnlyHandle(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "alter-column-read-only")
	collection, err := CreateAndOpen(ctx, path, alterColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	options := NewCollectionOptions()
	options.ReadOnly = true
	collection, err = Open(ctx, path, options)
	require.NoError(t, err)

	defer collection.Close()
	err = collection.AlterColumn(ctx, "count", "renamed", nil, AlterColumnOptions{})
	require.ErrorIs(t, err, ErrPermissionDenied)
}

func TestAlterColumnEmptyCollectionPublishesSchemaOnly(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "alter-column-empty")
	collection, err := CreateAndOpen(ctx, path, alterColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	initialGeneration := collection.store.Manifest().Generation
	replacement := FieldSchema{Name: "amount", DataType: DataTypeInt64}
	{
		err := collection.AlterColumn(ctx, "count", "", &replacement, AlterColumnOptions{Concurrency: 4})
		require.NoError(t, err)
	}
	require.True(t, collection.store.Manifest().Generation > initialGeneration)
	require.Nil(t, collection.store.Manifest().PersistedSegments)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, found := collection.Schema().Field("amount")
		require.True(t, found,
			"reopened schema does not contain replacement field")
	}
}

func alterColumnSchema() CollectionSchema {
	schema := NewCollectionSchema("alter_columns",
		FieldSchema{Name: "count", DataType: DataTypeInt32},
		FieldSchema{Name: "maybe", DataType: DataTypeFloat, Nullable: true},
		FieldSchema{Name: "cap", DataType: DataTypeUint32},
		FieldSchema{Name: "text", DataType: DataTypeString},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func alterColumnDocument(primaryKey string, count int32, maybe any, includeMaybe bool, cap uint32, embedding []float32) Document {
	fields := map[string]any{
		"count": count, "cap": cap, "text": primaryKey, "embedding": VectorFP32(embedding),
	}
	if includeMaybe {
		fields["maybe"] = maybe
	}
	return Document{PrimaryKey: primaryKey, Fields: fields}
}

func TestCollectionDenseHNSWQueryControlsAndRecall(t *testing.T) {
	ctx := context.Background()
	params := NewHNSWIndexParams(MetricTypeL2)
	params.M = 12
	params.EFConstruction = 80
	schema := NewCollectionSchema("dense_hnsw",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: params},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "hnsw"), schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	documents := annDenseDocuments(core.DefaultHNSWBruteForceThreshold + 200)
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := documents[713].Fields["embedding"].(VectorFP32)
	queryParams := NewHNSWQueryParams()
	queryParams.EF = 120
	queryParams.PrefetchOffset = math.MaxUint32
	queryParams.PrefetchLines = math.MaxUint32
	query := VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 20,
		Filter: "rating >= 1", Params: queryParams,
	}
	approximate, err := collection.Query(ctx, query)
	require.NoError(t, err)

	queryParams.Linear = true
	query.Params = queryParams
	exact, err := collection.Query(ctx, query)
	require.NoError(t, err)
	{
		recall := documentRecall(approximate, exact)
		require.True(t, recall >= .75)
	}

	for _, document := range approximate {
		require.True(t, document.Fields["rating"].(int32) >= 1)
	}
	queryParams.Linear = false
	queryParams.UseRefiner = true
	query.Params = queryParams
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, documentKeys(approximate), documentKeys(refined))
	{
		got := collection.Stats().IndexCompleteness["embedding"]
		require.True(t, got == 1)
	}
}

func TestCollectionHNSWRaBitQQueryCreateIndexOptimizeAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "rabitq")
	schema := NewCollectionSchema("rabitq_collection",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 64, Index: NewFlatIndexParams(MetricTypeL2)},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := annRaBitQDocuments(180)
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := documents[73].Fields["embedding"].(VectorFP32)
	exact := exactDenseDocumentResults(t, documents, queryVector, core.MetricL2, 15)

	indexParams := NewHNSWRaBitQIndexParams(MetricTypeL2)
	indexParams.TotalBits = 7
	indexParams.NumClusters = 8
	indexParams.M = 8
	indexParams.EFConstruction = 40
	{
		err := collection.CreateIndex(ctx, "embedding", indexParams, CreateIndexOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	defaulted, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: queryVector, TopK: 5})
	require.NoError(t, err)
	require.Len(t, defaulted, 5)

	queryParams := NewHNSWRaBitQQueryParams()
	queryParams.EF = 100
	query := VectorQuery{Field: "embedding", DenseVector: queryVector, TopK: 15, Params: queryParams}
	approximate, err := collection.Query(ctx, query)
	require.NoError(t, err)
	{
		recall := documentRecall(approximate, exact)
		require.True(t, recall >= .85)
	}

	queryParams.Linear = true
	queryParams.UseRefiner = true
	query.Params = queryParams
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, documentKeys(exact), documentKeys(refined))
	require.Equal(t, documentScores(exact), documentScores(refined))

	queryParams.Linear = false
	queryParams.UseRefiner = false
	queryParams.Radius = approximate[len(approximate)-1].Score
	query.Params = queryParams
	query.Filter = "rating >= 1"
	filtered, err := collection.Query(ctx, query)
	require.NoError(t, err)

	for _, document := range filtered {
		require.True(t, document.Fields["rating"].(int32) >= 1)
		require.True(t, document.Score <= queryParams.Radius)
	}
	{
		got := collection.Stats().IndexCompleteness["embedding"]
		require.True(t, got == 1)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	reopened, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, filtered, reopened,
		"reopened HNSW-RaBitQ query differs")

	field, _ := collection.Schema().Field("embedding")
	require.Equal(t, IndexTypeHNSWRaBitQ, field.IndexType())
	require.True(t, collection.Stats().IndexCompleteness["embedding"] == 1)
}

func TestCollectionVamanaQueryCreateIndexQuantizeOptimizeAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "vamana")
	schema := NewCollectionSchema("vamana_collection",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: NewFlatIndexParams(MetricTypeL2)},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := annDenseDocuments(320)
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := documents[217].Fields["embedding"].(VectorFP32)

	indexParams := NewVamanaIndexParams(MetricTypeL2)
	indexParams.MaxDegree = 12
	indexParams.SearchListSize = 60
	indexParams.MaxOcclusionSize = 120
	indexParams.SaturateGraph = true
	indexParams.UseContiguousMemory = true
	indexParams.UseIDMap = true
	indexParams.Quantize = QuantizeTypeInt8
	indexParams.Quantizer.EnableRotate = true
	{
		err := collection.CreateIndex(ctx, "embedding", indexParams, CreateIndexOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	defaulted, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: queryVector, TopK: 5})
	require.NoError(t, err)
	require.Len(t, defaulted, 5)

	queryParams := NewVamanaQueryParams()
	queryParams.EFSearch = 100
	queryParams.PrefetchOffset = math.MaxUint32
	queryParams.PrefetchLines = math.MaxUint32
	query := VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 15,
		Filter: "rating >= 1", Params: queryParams,
	}
	approximate, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Len(t, approximate, 15)

	for _, document := range approximate {
		require.True(t, document.Fields["rating"].(int32) >= 1)
	}
	queryParams.UseRefiner = true
	query.Params = queryParams
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)

	byKey := make(map[string]Document, len(documents))
	for _, document := range documents {
		byKey[document.PrimaryKey] = document
	}
	for _, document := range refined {
		original := byKey[document.PrimaryKey].Fields["embedding"].(VectorFP32)
		want, err := core.MetricL2.Compute(queryVector, original)
		require.NoError(t, err)
		require.Equal(t, want, document.Score)
	}
	queryParams.UseRefiner = false
	queryParams.Radius = approximate[len(approximate)-1].Score
	query.Params = queryParams
	bounded, err := collection.Query(ctx, query)
	require.NoError(t, err)

	for _, document := range bounded {
		require.True(t, document.Score <= queryParams.Radius)
	}
	queryParams.Linear = true
	queryParams.Radius = 0
	query.Params = queryParams
	linear, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Len(t, linear, 15)
	{
		got := collection.Stats().IndexCompleteness["embedding"]
		require.True(t, got == 1)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	reopened, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, linear, reopened,
		"reopened Vamana query differs")

	field, _ := collection.Schema().Field("embedding")
	require.Equal(t, IndexTypeVamana, field.IndexType())
	require.True(t, collection.Stats().IndexCompleteness["embedding"] == 1)
}

func TestCollectionDiskANNQueryCreateIndexRefineOptimizeAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "diskann")
	schema := NewCollectionSchema("diskann_collection",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: NewFlatIndexParams(MetricTypeL2)},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := annDenseDocuments(320)
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := documents[217].Fields["embedding"].(VectorFP32)

	indexParams := NewDiskANNIndexParams(MetricTypeL2)
	indexParams.MaxDegree = 12
	indexParams.ListSize = 60
	indexParams.PQChunks = 2
	{
		err := collection.CreateIndex(ctx, "embedding", indexParams, CreateIndexOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	defaulted, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: queryVector, TopK: 5})
	require.NoError(t, err)
	require.Len(t, defaulted, 5)

	params := NewDiskANNQueryParams()
	params.ListSize = 100
	query := VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 15,
		Filter: "rating >= 1", Params: params,
	}
	approximate, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Len(t, approximate, 15)

	for _, document := range approximate {
		require.True(t, document.Fields["rating"].(int32) >= 1)
	}

	params.UseRefiner = true
	query.Params = params
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)

	byKey := make(map[string]Document, len(documents))
	for _, document := range documents {
		byKey[document.PrimaryKey] = document
	}
	for _, document := range refined {
		original := byKey[document.PrimaryKey].Fields["embedding"].(VectorFP32)
		want, err := core.MetricL2.Compute(queryVector, original)
		require.NoError(t, err)
		require.Equal(t, want, document.Score)
	}

	params.UseRefiner = false
	params.Radius = approximate[len(approximate)-1].Score
	query.Params = params
	bounded, err := collection.Query(ctx, query)
	require.NoError(t, err)

	for _, document := range bounded {
		require.True(t, document.Score <= params.Radius)
	}
	params.Linear = true
	params.Radius = 0
	query.Params = params
	linear, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Len(t, linear, 15)
	{
		recall := documentRecall(approximate, linear)
		require.True(t, recall >= .80)
	}
	{
		got := collection.Stats().IndexCompleteness["embedding"]
		require.True(t, got == 1)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	reopened, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, linear, reopened,
		"reopened DiskANN query differs")

	field, _ := collection.Schema().Field("embedding")
	require.Equal(t, IndexTypeDiskANN, field.IndexType())
	require.True(t, collection.Stats().IndexCompleteness["embedding"] == 1)
}

func TestCollectionDiskANNDirectFP16SchemaDefaults(t *testing.T) {
	ctx := context.Background()
	params := NewDiskANNIndexParams(MetricTypeL2)
	params.MaxDegree, params.ListSize, params.PQChunks = 4, 8, 1
	schema := NewCollectionSchema("diskann_direct",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP16, Dimension: 2, Index: params},
	)
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "direct"), schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{
			{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP16{Float16FromFloat32(0), Float16FromFloat32(0)}}},
			{PrimaryKey: "b", Fields: map[string]any{"embedding": VectorFP16{Float16FromFloat32(1), Float16FromFloat32(0)}}},
			{PrimaryKey: "c", Fields: map[string]any{"embedding": VectorFP16{Float16FromFloat32(3), Float16FromFloat32(0)}}},
		})
		require.NoError(t, err)
	}

	results, err := collection.Query(ctx, VectorQuery{
		Field:       "embedding",
		DenseVector: VectorFP16{Float16FromFloat32(0.9), Float16FromFloat32(0)},
		TopK:        2,
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "a"}, documentKeys(results))
	require.True(t, collection.Stats().IndexCompleteness["embedding"] == 1)
}

func TestCollectionScalarQuantizedDiskANNBackfillRefineOptimizeAndReopen(t *testing.T) {
	ctx := context.Background()
	for _, quantize := range []QuantizeType{QuantizeTypeFP16, QuantizeTypeInt8, QuantizeTypeInt4} {
		t.Run(quantize.String(), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "diskann")
			schema := NewCollectionSchema("quantized_diskann",
				FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: NewFlatIndexParams(MetricTypeL2)},
				FieldSchema{Name: "rating", DataType: DataTypeInt32},
			)
			schema.MaxDocsPerSegment = MinMaxDocsPerSegment
			collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
			require.NoError(t, err)

			documents := annDenseDocuments(48)
			{
				_, err := collection.Insert(ctx, documents)
				require.NoError(t, err)
			}

			indexParams := NewDiskANNIndexParams(MetricTypeL2)
			indexParams.MaxDegree, indexParams.ListSize, indexParams.PQChunks = 8, len(documents), 2
			indexParams.Quantize = quantize
			indexParams.Quantizer.EnableRotate = quantize != QuantizeTypeFP16
			{
				err := collection.CreateIndex(ctx, "embedding", indexParams, CreateIndexOptions{Concurrency: 2})
				require.NoError(t, err)
			}

			queryVector := append(VectorFP32(nil), documents[17].Fields["embedding"].(VectorFP32)...)
			queryVector[1] += .137
			queryParams := NewDiskANNQueryParams()
			queryParams.ListSize = len(documents)
			query := VectorQuery{Field: "embedding", DenseVector: queryVector, TopK: 12, Params: queryParams}
			graph, err := collection.Query(ctx, query)
			require.NoError(t, err)

			queryParams.Linear = true
			query.Params = queryParams
			linear, err := collection.Query(ctx, query)
			require.NoError(t, err)
			require.Equal(t, linear, graph)

			byKey := make(map[string]Document, len(documents))
			for _, document := range documents {
				byKey[document.PrimaryKey] = document
			}
			quantizedDifference := false
			for _, document := range linear {
				original := byKey[document.PrimaryKey].Fields["embedding"].(VectorFP32)
				score, err := core.MetricL2.Compute(queryVector, original)
				require.NoError(t, err)

				if document.Score != score {
					quantizedDifference = true
					break
				}
			}
			require.True(t, quantizedDifference,
				"DiskANN scalar quantization did not affect any first-stage score")

			queryParams.Linear = false
			queryParams.UseRefiner = true
			query.Params = queryParams
			refined, err := collection.Query(ctx, query)
			require.NoError(t, err)

			exact := exactDenseDocumentResults(t, documents, queryVector, core.MetricL2, query.TopK)
			require.Equal(t, documentKeys(exact), documentKeys(refined))
			require.Equal(t, documentScores(exact), documentScores(refined))

			queryParams.UseRefiner = false
			queryParams.Radius = graph[7].Score
			query.Filter = "rating >= 1"
			query.Params = queryParams
			bounded, err := collection.Query(ctx, query)
			require.NoError(t, err)

			for _, document := range bounded {
				require.True(t, document.Fields["rating"].(int32) >= 1)
				require.True(t, document.Score <= queryParams.Radius)
			}
			require.False(t, len(bounded) == 0,
				"filter/radius removed every document")

			if quantize == QuantizeTypeFP16 {
				results, err := collection.Insert(ctx, []Document{{
					PrimaryKey: "overflow", Fields: map[string]any{"embedding": VectorFP32{70000, 0, 0, 0}, "rating": int32(1)},
				}})
				require.Error(t, err)
				require.Len(t, results, 1)
				require.ErrorIs(t, results[0].Err, ErrInvalidArgument)
			}
			{
				_, err := collection.Delete(ctx, []string{"d0003"})
				require.NoError(t, err)
			}
			{
				err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
				require.NoError(t, err)
			}
			{
				got := collection.Stats().IndexCompleteness["embedding"]
				require.True(t, got == 1)
			}

			beforeReopen, err := collection.Query(ctx, query)
			require.NoError(t, err)
			{
				err := collection.Close()
				require.NoError(t, err)
			}

			collection, err = Open(ctx, path, NewCollectionOptions())
			require.NoError(t, err)

			defer collection.Close()
			reopened, err := collection.Query(ctx, query)
			require.NoError(t, err)
			require.Equal(t, beforeReopen, reopened)

			field, _ := collection.Schema().Field("embedding")
			persisted := field.Index.(DiskANNIndexParams)
			require.Equal(t, quantize, persisted.Quantize)
			require.Equal(t, quantize != QuantizeTypeFP16, persisted.Quantizer.EnableRotate)
		})
	}
}

func TestCollectionQuantizedIVFSOARRefinementCreateIndexAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "ivf")
	schema := NewCollectionSchema("quantized_ivf",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: NewFlatIndexParams(MetricTypeL2)},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := annDenseDocuments(320)
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := documents[217].Fields["embedding"].(VectorFP32)
	exact, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 15, Filter: "rating >= 1",
	})
	require.NoError(t, err)

	indexParams := NewIVFIndexParams(MetricTypeL2)
	indexParams.NList = 16
	indexParams.NIterations = 12
	indexParams.UseSOAR = true
	indexParams.Quantize = QuantizeTypeInt8
	indexParams.Quantizer.EnableRotate = true
	standardParams := indexParams
	standardParams.UseSOAR = false
	{
		err := collection.CreateIndex(ctx, "embedding", standardParams, CreateIndexOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	compatibilityParams := NewIVFQueryParams()
	compatibilityParams.NProbe = 4
	compatibilityQuery := VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 15,
		Filter: "rating >= 1", Params: compatibilityParams,
	}
	standardResults, err := collection.Query(ctx, compatibilityQuery)
	require.NoError(t, err)
	{
		err := collection.CreateIndex(ctx, "embedding", indexParams, CreateIndexOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	soarResults, err := collection.Query(ctx, compatibilityQuery)
	require.NoError(t, err)
	require.Equal(t, standardResults, soarResults)

	queryParams := NewIVFQueryParams()
	queryParams.NProbe = 16
	queryParams.UseRefiner = true
	queryParams.ScaleFactor = 100
	queryParams.Radius = exact[len(exact)-1].Score
	query := VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 15,
		Filter: "rating >= 1", Params: queryParams,
	}
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, documentKeys(exact), documentKeys(refined))

	for position := range refined {
		require.Equal(t, exact[position].Score, refined[position].Score)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	reopened, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, refined, reopened,
		"reopened refined IVF differs")

	field, _ := collection.Schema().Field("embedding")
	persisted := field.Index.(IVFIndexParams)
	require.Equal(t, IndexTypeIVF, field.IndexType())
	require.True(t, persisted.UseSOAR)
	require.True(t, collection.Stats().IndexCompleteness["embedding"] == 1)
}

func TestCollectionQuantizedFlatRotationAndRefiner(t *testing.T) {
	ctx := context.Background()
	params := NewFlatIndexParams(MetricTypeL2)
	params.Quantize = QuantizeTypeInt4
	params.Quantizer.EnableRotate = true
	schema := NewCollectionSchema("quantized_flat",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: params},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "flat"), schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	documents := annDenseDocuments(80)
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := VectorFP32{3.25, -1.5, 7, 2}
	queryParams := NewFlatQueryParams()
	queryParams.UseRefiner = true
	queryParams.ScaleFactor = 100
	refined, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 12, Params: queryParams,
	})
	require.NoError(t, err)

	want := exactDenseDocumentResults(t, documents, queryVector, core.MetricL2, 12)
	require.Equal(t, documentKeys(want), documentKeys(refined))

	for position := range refined {
		require.Equal(t, want[position].Score, refined[position].Score)
	}
}

func TestCollectionSparseHNSWFP16Controls(t *testing.T) {
	ctx := context.Background()
	params := NewHNSWIndexParams(MetricTypeIP)
	params.M = 8
	params.EFConstruction = 40
	params.Quantize = QuantizeTypeFP16
	schema := NewCollectionSchema("sparse_hnsw",
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: params},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "sparse"), schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	documents := make([]Document, 240)
	for position := range documents {
		documents[position] = Document{PrimaryKey: fmt.Sprintf("s%03d", position), Fields: map[string]any{
			"sparse": SparseVectorFP32{
				Indices: []uint32{uint32(position % 31), uint32(100 + position%37), uint32(200 + position%43)},
				Values:  []float32{float32(position%7) + .12345, float32(position%11) + .33331, float32(position%13) + .77771},
			},
			"rating": int32(position % 3),
		}}
	}
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryParams := NewHNSWQueryParams()
	queryParams.EF = 24
	queryParams.PrefetchOffset = 8
	queryParams.PrefetchLines = 0
	queryParams.Radius = 1
	query := VectorQuery{
		Field: "sparse", SparseVector: documents[117].Fields["sparse"].(SparseVectorFP32),
		TopK: 20, Filter: "rating >= 1", Params: queryParams,
	}
	got, err := collection.Query(ctx, query)
	require.NoError(t, err)

	queryParams.Linear = true
	query.Params = queryParams
	want, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, want, got)

	queryParams.Linear = false
	queryParams.UseRefiner = true
	query.Params = queryParams
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)

	approximateByKey := make(map[string]float32, len(got))
	for _, document := range got {
		approximateByKey[document.PrimaryKey] = document.Score
	}
	originalByKey := make(map[string]SparseVectorFP32, len(documents))
	for _, document := range documents {
		originalByKey[document.PrimaryKey] = document.Fields["sparse"].(SparseVectorFP32)
	}
	changedScore := false
	querySparse := query.SparseVector.(SparseVectorFP32)
	for _, document := range refined {
		require.True(t, document.Fields["rating"].(int32) >= 1)
		require.True(t, document.Score >= queryParams.Radius)

		original := originalByKey[document.PrimaryKey]
		exact, err := ailego.SparseInnerProduct(querySparse.Indices, querySparse.Values, original.Indices, original.Values)
		require.NoError(t, err)
		require.Equal(t, exact, document.Score)

		if approximate, found := approximateByKey[document.PrimaryKey]; found && approximate != exact {
			changedScore = true
		}
	}
	require.False(t, len(refined) == 0)
	require.True(t, changedScore)
}

func TestCollectionSparseFlatFP16RefinementMultiQueryAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "sparse-flat-refine")
	params := NewFlatIndexParams(MetricTypeIP)
	params.Quantize = QuantizeTypeFP16
	schema := NewCollectionSchema("sparse_flat_refine",
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: params},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := []Document{
		{PrimaryKey: "a", Fields: map[string]any{"sparse": SparseVectorFP32{Indices: []uint32{0, 3}, Values: []float32{1.0003, 4.0007}}, "rating": int32(1)}},
		{PrimaryKey: "b", Fields: map[string]any{"sparse": SparseVectorFP32{Indices: []uint32{0, 2}, Values: []float32{2.0009, 5.0011}}, "rating": int32(2)}},
		{PrimaryKey: "c", Fields: map[string]any{"sparse": SparseVectorFP32{Indices: []uint32{1, 3}, Values: []float32{9.0021, 1.0004}}, "rating": int32(3)}},
		{PrimaryKey: "d", Fields: map[string]any{"sparse": SparseVectorFP32{Indices: []uint32{0, 1, 3}, Values: []float32{1.5006, 2.0013, 2.5008}}, "rating": int32(4)}},
		{PrimaryKey: "e", Fields: map[string]any{"sparse": SparseVectorFP32{Indices: []uint32{2, 3}, Values: []float32{8.0005, 3.0009}}, "rating": int32(5)}},
		{PrimaryKey: "f", Fields: map[string]any{"sparse": SparseVectorFP32{Indices: []uint32{0, 2}, Values: []float32{.5003, 1.0007}}, "rating": int32(6)}},
	}
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := SparseVectorFP32{Indices: []uint32{0, 3}, Values: []float32{2.0007, 1.0003}}
	queryParams := NewFlatQueryParams()
	queryParams.UseRefiner = true
	queryParams.ScaleFactor = 100
	exact := exactSparseDocumentResults(t, documents, queryVector, len(documents))
	queryParams.Radius = exact[4].Score
	query := VectorQuery{Field: "sparse", SparseVector: queryVector, TopK: 5, Params: queryParams}
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, documentKeys(exact[:5]), documentKeys(refined))
	require.Equal(t, documentScores(exact[:5]), documentScores(refined))

	unrefinedParams := queryParams
	unrefinedParams.UseRefiner = false
	unrefinedParams.Radius = 0
	unrefined, err := collection.Query(ctx, VectorQuery{
		Field: "sparse", SparseVector: queryVector, TopK: 5, Params: unrefinedParams,
	})
	require.NoError(t, err)
	require.NotEqual(t, documentScores(refined), documentScores(unrefined),
		"FP16 sparse first-stage scores unexpectedly equal original-vector scores")

	alternate := SparseVectorFP32{Indices: []uint32{1, 2}, Values: []float32{.7503, 1.2509}}
	alternateExact := exactSparseDocumentResults(t, documents, alternate, 4)
	alternateParams := queryParams
	alternateParams.Radius = 0
	multi := MultiQuery{
		Queries: []SubQuery{
			{Field: "sparse", SparseVector: queryVector, Params: queryParams, NumCandidates: 5},
			{Field: "sparse", SparseVector: alternate, Params: alternateParams, NumCandidates: 4},
		},
		TopK: 2,
		Reranker: testRerankerFunc(func(_ context.Context, batches []RerankBatch, _ int) ([]Document, error) {
			require.Equal(t, documentKeys(exact[:5]), documentKeys(batches[0].Documents))
			require.Equal(t, documentScores(exact[:5]), documentScores(batches[0].Documents))
			require.Equal(t, documentKeys(alternateExact), documentKeys(batches[1].Documents))
			require.Equal(t, documentScores(alternateExact), documentScores(batches[1].Documents))

			return []Document{batches[0].Documents[0], batches[1].Documents[0]}, nil
		}),
	}
	beforeReopen, err := collection.MultiQuery(ctx, multi)
	require.NoError(t, err)
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	reopened, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, refined, reopened)

	reopenedMulti, err := collection.MultiQuery(ctx, multi)
	require.NoError(t, err)
	require.Equal(t, beforeReopen, reopenedMulti)
}

func TestCollectionANNValidationAndBackfillRollback(t *testing.T) {
	ctx := context.Background()
	schema := NewCollectionSchema("ann_validation",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: NewFlatIndexParams(MetricTypeL2)},
		FieldSchema{Name: "group", DataType: DataTypeString},
	)
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "validation"), schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{{PrimaryKey: "huge", Fields: map[string]any{
			"embedding": VectorFP32{70000, 1, 2, 3}, "group": "g",
		}}})
		require.NoError(t, err)
	}

	before := collection.Schema()
	generation := collection.store.Manifest().Generation
	quantized := NewIVFIndexParams(MetricTypeL2)
	quantized.NList = 1
	quantized.Quantize = QuantizeTypeFP16
	{
		err := collection.CreateIndex(ctx, "embedding", quantized, CreateIndexOptions{})
		require.Error(t, err,
			"FP16 overflow backfill succeeded")
	}
	require.Equal(t, before, collection.Schema(),
		"failed ANN backfill changed schema generation")
	require.Equal(t, generation, collection.store.Manifest().Generation,
		"failed ANN backfill changed schema generation")

	vamana := NewVamanaIndexParams(MetricTypeL2)
	vamana.MaxDegree, vamana.SearchListSize, vamana.MaxOcclusionSize = 4, 8, 16
	vamana.Quantize = QuantizeTypeFP16
	{
		err := collection.CreateIndex(ctx, "embedding", vamana, CreateIndexOptions{})
		require.Error(t, err,
			"Vamana FP16 overflow backfill succeeded")
	}
	require.Equal(t, before, collection.Schema(),
		"failed Vamana backfill changed schema generation")
	require.Equal(t, generation, collection.store.Manifest().Generation,
		"failed Vamana backfill changed schema generation")

	diskANN := NewDiskANNIndexParams(MetricTypeL2)
	diskANN.MaxDegree, diskANN.ListSize, diskANN.PQChunks = 4, 8, 2
	diskANN.Quantize = QuantizeTypeFP16
	{
		err := collection.CreateIndex(ctx, "embedding", diskANN, CreateIndexOptions{})
		require.Error(t, err,
			"DiskANN FP16 overflow backfill succeeded")
	}
	require.Equal(t, before, collection.Schema(),
		"failed DiskANN backfill changed schema generation")
	require.Equal(t, generation, collection.store.Manifest().Generation,
		"failed DiskANN backfill changed schema generation")

	soar := NewIVFIndexParams(MetricTypeL2)
	soar.UseSOAR = true
	{
		err := collection.CreateIndex(ctx, "embedding", soar, CreateIndexOptions{})
		require.NoError(t, err)
	}

	soarField, _ := collection.Schema().Field("embedding")
	{
		persisted := soarField.Index.(IVFIndexParams)
		require.True(t, persisted.UseSOAR)
		require.Equal(t, generation+1, collection.store.Manifest().Generation)
	}

	hnswParams := NewHNSWQueryParams()
	hnswParams.EF = MaxGraphEFSearch + 1
	{
		_, err := collection.Query(ctx, VectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4}, TopK: 1, Params: hnswParams,
		})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	index := NewHNSWIndexParams(MetricTypeL2)
	{
		err := collection.CreateIndex(ctx, "embedding", index, CreateIndexOptions{})
		require.NoError(t, err)
	}
	{
		_, err := collection.Query(ctx, VectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4}, TopK: 1, Params: hnswParams,
		})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	groups, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4},
		GroupByField: "group", GroupCount: 1, TopKPerGroup: 1,
	})
	require.NoError(t, err)
	require.Len(t, groups, 1)
	require.True(t, groups[0].Value == "g")

	refined := NewHNSWQueryParams()
	refined.UseRefiner = true
	{
		_, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4}, Params: refined,
			GroupByField: "group", GroupCount: 1, TopKPerGroup: 1,
		})
		require.ErrorIs(t, err, ErrNotSupported)
	}

	linear := NewHNSWQueryParams()
	linear.Linear = true
	{
		_, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4}, Params: linear,
			GroupByField: "group", GroupCount: 1, TopKPerGroup: 1,
		})
		require.NoError(t, err)
	}
}

func TestCollectionGroupByPreservesUnsupportedANNBoundary(t *testing.T) {
	tests := []struct {
		name   string
		index  IndexParams
		params func(linear bool) QueryParams
	}{
		{
			name: "IVF", index: NewIVFIndexParams(MetricTypeL2),
			params: func(linear bool) QueryParams {
				params := NewIVFQueryParams()
				params.Linear = linear
				return params
			},
		},
		{
			name: "Vamana", index: NewVamanaIndexParams(MetricTypeL2),
			params: func(linear bool) QueryParams {
				params := NewVamanaQueryParams()
				params.Linear = linear
				return params
			},
		},
		{
			name: "DiskANN", index: NewDiskANNIndexParams(MetricTypeL2),
			params: func(linear bool) QueryParams {
				params := NewDiskANNQueryParams()
				params.Linear = linear
				return params
			},
		},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			ctx := context.Background()
			schema := NewCollectionSchema("unsupported_native_group",
				FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: testCase.index},
				FieldSchema{Name: "group", DataType: DataTypeString},
			)
			collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "collection"), schema, NewCollectionOptions())
			require.NoError(t, err)

			defer collection.Close()
			{
				_, err := collection.Insert(ctx, []Document{
					{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP32{0, 0, 0, 0}, "group": "a"}},
					{PrimaryKey: "b", Fields: map[string]any{"embedding": VectorFP32{1, 1, 1, 1}, "group": "b"}},
				})
				require.NoError(t, err)
			}

			query := GroupByVectorQuery{
				Field: "embedding", DenseVector: VectorFP32{0, 0, 0, 0},
				GroupByField: "group", GroupCount: 2, TopKPerGroup: 1,
				Params: testCase.params(false),
			}
			{
				_, err := collection.GroupByQuery(ctx, query)
				require.ErrorIs(t, err, ErrNotSupported)
			}

			query.Params = testCase.params(true)
			groups, err := collection.GroupByQuery(ctx, query)
			require.NoError(t, err)
			require.Len(t, groups, 2)
		})
	}
}

func TestCollectionQuantizedWriteRejectsUnrepresentableVector(t *testing.T) {
	ctx := context.Background()
	params := NewFlatIndexParams(MetricTypeL2)
	params.Quantize = QuantizeTypeFP16
	schema := NewCollectionSchema("quantized_write", FieldSchema{
		Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: params,
	})
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "write"), schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	results, err := collection.Insert(ctx, []Document{{
		PrimaryKey: "overflow", Fields: map[string]any{"embedding": VectorFP32{70000, 1}},
	}})
	require.Error(t, err)
	require.Len(t, results, 1)
	require.ErrorIs(t, results[0].Err, ErrInvalidArgument)
	require.True(t, collection.Stats().DocumentCount == 0,
		"failed quantized write changed document count")
}

func annDenseDocuments(count int) []Document {
	documents := make([]Document, count)
	for position := range documents {
		documents[position] = Document{PrimaryKey: fmt.Sprintf("d%04d", position), Fields: map[string]any{
			"embedding": VectorFP32{
				float32(position%23) - 7.25,
				float32((position*7)%31) - 11.5,
				float32((position*13)%37) + .125,
				float32((position*19)%41) - 3.75,
			},
			"rating": int32(position % 3),
		}}
	}
	return documents
}

func annRaBitQDocuments(count int) []Document {
	documents := make([]Document, count)
	for position := range documents {
		vector := make(VectorFP32, 64)
		for dimension := range vector {
			vector[dimension] = float32(int((position*37+dimension*17+position*dimension*3)%211)-105) / 19
		}
		documents[position] = Document{PrimaryKey: fmt.Sprintf("r%04d", position), Fields: map[string]any{
			"embedding": vector,
			"rating":    int32(position % 3),
		}}
	}
	return documents
}

func documentRecall(got, want []Document) float64 {
	keys := make(map[string]struct{}, len(want))
	for _, document := range want {
		keys[document.PrimaryKey] = struct{}{}
	}
	matched := 0
	for _, document := range got {
		if _, found := keys[document.PrimaryKey]; found {
			matched++
		}
	}
	if len(want) == 0 {
		return 1
	}
	return float64(matched) / float64(len(want))
}

func documentScores(documents []Document) []float32 {
	scores := make([]float32, len(documents))
	for index := range documents {
		scores[index] = documents[index].Score
	}
	return scores
}

func exactDenseDocumentResults(t testing.TB, documents []Document, query VectorFP32, metric core.Metric, topK int) []Document {
	t.Helper()
	candidates := make([]core.Candidate, len(documents))
	byID := make(map[uint64]Document, len(documents))
	for position, document := range documents {
		document.DocID = uint64(position)
		candidates[position] = core.Candidate{Key: uint64(position), Vector: []float32(document.Fields["embedding"].(VectorFP32))}
		byID[uint64(position)] = document
	}
	results, err := core.TopK(context.Background(), metric, []float32(query), candidates, topK)
	require.NoError(t, err)

	output := make([]Document, len(results))
	for position, result := range results {
		output[position] = byID[result.Key]
		output[position].Score = result.Score
	}
	return output
}

func exactSparseDocumentResults(t testing.TB, documents []Document, query SparseVectorFP32, topK int) []Document {
	t.Helper()
	index, err := core.NewSparseFlatIndex(core.MetricIP)
	require.NoError(t, err)

	byID := make(map[uint64]Document, len(documents))
	for position, document := range documents {
		vector, err := sparseValueToCore(document.Fields["sparse"])
		require.NoError(t, err)

		key := uint64(position)
		{
			err := index.AddSparse(context.Background(), key, vector)
			require.NoError(t, err)
		}

		document.DocID = key
		byID[key] = document
	}
	queryVector, err := sparseValueToCore(query)
	require.NoError(t, err)

	results, err := index.SearchSparseWithOptions(context.Background(), queryVector, core.SearchOptions{TopK: topK})
	require.NoError(t, err)

	output := make([]Document, len(results))
	for position, result := range results {
		output[position] = byID[result.Key]
		output[position].Score = result.Score
	}
	return output
}

const (
	atomicRecoveryPathEnv     = "ZVEC_ATOMIC_RECOVERY_PATH"
	atomicRecoveryMutationEnv = "ZVEC_ATOMIC_RECOVERY_MUTATION"
	atomicRecoveryPhaseEnv    = "ZVEC_ATOMIC_RECOVERY_PHASE"
)

type atomicRecoveryCase struct {
	name        string
	dataRewrite bool
}

func TestAtomicDDLAndOptimizeCrashRecovery(t *testing.T) {
	if path := os.Getenv(atomicRecoveryPathEnv); path != "" {
		runAtomicRecoveryChild(path, os.Getenv(atomicRecoveryMutationEnv), os.Getenv(atomicRecoveryPhaseEnv))
		return
	}

	tests := []atomicRecoveryCase{
		{name: "create_index"},
		{name: "drop_index"},
		{name: "add_column", dataRewrite: true},
		{name: "alter_column", dataRewrite: true},
		{name: "drop_column", dataRewrite: true},
		{name: "optimize", dataRewrite: true},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Run("before_current", func(t *testing.T) {
				path, generation, current, initialSegments := createAtomicRecoveryFixture(t)
				versionLock := flock.New(filepath.Join(path, ".version.lock"))
				locked, err := versionLock.TryLock()
				require.NoError(t, err)
				require.True(t, locked)

				lockClosed := false
				defer func() {
					if !lockClosed {
						_ = versionLock.Close()
					}
				}()

				command := startAtomicRecoveryChild(t, path, testCase.name, "before_current")
				waitForAtomicMarker(t, command, filepath.Join(path, ".atomic-blocked"))
				if testCase.dataRewrite {
					waitForAdditionalSegment(t, command, path, initialSegments)
				}
				after, err := os.ReadFile(filepath.Join(path, "CURRENT"))
				require.NoError(t, err)
				require.True(t, bytes.Equal(after, current),
					"CURRENT changed before the held publication boundary")

				killAtomicRecoveryChild(t, command)
				{
					err := versionLock.Close()
					require.NoError(t, err)
				}

				lockClosed = true

				collection, err := Open(context.Background(), path, NewCollectionOptions())
				require.NoError(t, err)

				defer collection.Close()
				{
					got := collection.store.Manifest().Generation
					require.Equal(t, generation, got)
				}

				assertAtomicInitialState(t, collection)
				assertAtomicContinuedWrite(t, collection, testCase.name, false)
			})

			t.Run("after_current", func(t *testing.T) {
				path, generation, _, _ := createAtomicRecoveryFixture(t)
				blocker := ""
				if testCase.name == "optimize" {
					// Force the post-publication prune to stop. The child still
					// observes a newer CURRENT and waits to be killed with an open
					// collection handle, leaving cleanup for recovery to retry.
					blocker = filepath.Join(path, "wal", "99999999999999999999-99999999999999999999.wal")
					{
						err := os.Mkdir(blocker, 0o755)
						require.NoError(t, err)
					}
				}
				command := startAtomicRecoveryChild(t, path, testCase.name, "after_current")
				waitForAtomicMarker(t, command, filepath.Join(path, ".atomic-committed"))
				killAtomicRecoveryChild(t, command)

				collection, err := Open(context.Background(), path, NewCollectionOptions())
				require.NoError(t, err)

				defer collection.Close()
				{
					got := collection.store.Manifest().Generation
					require.True(t, got > generation)
				}

				assertAtomicCommittedState(t, collection, testCase.name)
				if blocker != "" {
					{
						err := os.Remove(blocker)
						require.NoError(t, err)
					}

					committedGeneration := collection.store.Manifest().Generation
					{
						err := collection.Optimize(context.Background(), OptimizeOptions{})
						require.NoError(t, err)
					}
					{
						got := collection.store.Manifest().Generation
						require.Equal(t, committedGeneration, got)
					}

					assertOptimizeArtifacts(t, path, 1)
				}
				assertAtomicContinuedWrite(t, collection, testCase.name, true)
			})
		})
	}
}

func runAtomicRecoveryChild(path, mutation, phase string) {
	collection, err := Open(context.Background(), path, NewCollectionOptions())
	if err != nil {
		os.Exit(91)
	}
	initialGeneration := collection.store.Manifest().Generation
	if err := os.WriteFile(filepath.Join(path, ".atomic-started"), []byte(mutation), 0o600); err != nil {
		os.Exit(92)
	}
	if phase == "before_current" {
		result := make(chan error, 1)
		go func() {
			result <- runAtomicRecoveryMutation(collection, mutation)
		}()
		select {
		case <-result:
			os.Exit(93)
		case <-time.After(150 * time.Millisecond):
			if err := os.WriteFile(filepath.Join(path, ".atomic-blocked"), []byte(mutation), 0o600); err != nil {
				os.Exit(94)
			}
			for {
				time.Sleep(time.Second)
			}
		}
	}

	mutationErr := runAtomicRecoveryMutation(collection, mutation)
	if collection.store.Manifest().Generation <= initialGeneration {
		os.Exit(95)
	}
	// Optimize may report the deliberately injected post-commit prune error.
	// Every other successful publication must return nil.
	if mutationErr != nil && mutation != "optimize" {
		os.Exit(96)
	}
	if err := os.WriteFile(filepath.Join(path, ".atomic-committed"), []byte(mutation), 0o600); err != nil {
		os.Exit(97)
	}
	for {
		time.Sleep(time.Second)
	}
}

func runAtomicRecoveryMutation(collection *Collection, mutation string) error {
	ctx := context.Background()
	switch mutation {
	case "create_index":
		return collection.CreateIndex(ctx, "rating", NewInvertIndexParams(), CreateIndexOptions{Concurrency: 2})
	case "drop_index":
		return collection.DropIndex(ctx, "indexed")
	case "add_column":
		return collection.AddColumn(ctx, FieldSchema{Name: "added", DataType: DataTypeInt64}, "rating + 1", AddColumnOptions{Concurrency: 2})
	case "alter_column":
		replacement := FieldSchema{Name: "renamed", DataType: DataTypeInt64}
		return collection.AlterColumn(ctx, "alter_me", "", &replacement, AlterColumnOptions{Concurrency: 2})
	case "drop_column":
		return collection.DropColumn(ctx, "drop_me")
	case "optimize":
		return collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
	default:
		return fmt.Errorf("unknown atomic recovery mutation %q", mutation)
	}
}

func createAtomicRecoveryFixture(t *testing.T) (path string, generation uint64, current []byte, segments int) {
	t.Helper()
	ctx := context.Background()
	path = filepath.Join(t.TempDir(), "atomic-recovery")
	options := NewCollectionOptions()
	options.WALSyncEvery = 1
	collection, err := CreateAndOpen(ctx, path, atomicRecoverySchema(), options)
	require.NoError(t, err)

	documents := []Document{
		atomicRecoveryDocument("a", "alpha", 1, 1, 11, 21, 4),
		atomicRecoveryDocument("b", "bravo", 2, 2, 12, 22, 2),
		atomicRecoveryDocument("c", "charlie", 3, 3, 13, 23, 3),
	}
	{
		_, err := collection.Insert(ctx, documents[:1])
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		_, err := collection.Insert(ctx, documents[1:2])
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		_, err := collection.Insert(ctx, documents[2:])
		require.NoError(t, err)
	}

	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"rating": int32(10)}}})
	require.NoError(t, err)
	require.True(t, updated[0].DocID == 3)
	{
		_, err := collection.Delete(ctx, []string{"b"})
		require.NoError(t, err)
	}

	generation = collection.store.Manifest().Generation
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	current, err = os.ReadFile(filepath.Join(path, "CURRENT"))
	require.NoError(t, err)

	segments = countAtomicSegments(t, path)
	require.True(t, segments == 2)

	return path, generation, current, segments
}

func atomicRecoverySchema() CollectionSchema {
	schema := NewCollectionSchema("atomic_recovery",
		FieldSchema{Name: "title", DataType: DataTypeString},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
		FieldSchema{Name: "indexed", DataType: DataTypeInt32, Index: NewInvertIndexParams()},
		FieldSchema{Name: "alter_me", DataType: DataTypeInt32},
		FieldSchema{Name: "drop_me", DataType: DataTypeInt32},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func atomicRecoveryDocument(primaryKey, title string, rating, indexed, alter, drop int32, score float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"title": title, "rating": rating, "indexed": indexed,
		"alter_me": alter, "drop_me": drop,
		"embedding": VectorFP32{score, 0},
	}}
}

func assertAtomicInitialState(t *testing.T, collection *Collection) {
	t.Helper()
	require.Equal(t, atomicRecoverySchema(), collection.Schema())

	assertAtomicLiveDocuments(t, collection)
}

func assertAtomicCommittedState(t *testing.T, collection *Collection, mutation string) {
	t.Helper()
	schema := collection.Schema()
	switch mutation {
	case "create_index":
		field, _ := schema.Field("rating")
		require.Equal(t, IndexTypeInvert, field.IndexType())

	case "drop_index":
		field, _ := schema.Field("indexed")
		require.Equal(t, IndexTypeUndefined, field.IndexType())

	case "add_column":
		field, found := schema.Field("added")
		require.True(t, found)
		require.Equal(t, DataTypeInt64, field.DataType)

	case "alter_column":
		_, oldFound := schema.Field("alter_me")
		field, newFound := schema.Field("renamed")
		require.False(t, oldFound)
		require.True(t, newFound)
		require.Equal(t, DataTypeInt64, field.DataType)

	case "drop_column":
		{
			_, found := schema.Field("drop_me")
			require.False(t, found)
		}

	case "optimize":
		manifest := collection.store.Manifest()
		require.Len(t, manifest.PersistedSegments, 1)
		require.True(t, manifest.PersistedSegments[0].MinDocID == 2)
		require.True(t, manifest.PersistedSegments[0].MaxDocID == 3)
		require.True(t, manifest.PersistedSegments[0].DocCount == 2)
		require.True(t, manifest.WritingSegmentStartDocID == 4)
	}
	assertAtomicLiveDocuments(t, collection)
	fetched, err := collection.Fetch(context.Background(), []string{"a", "c"}, Projection{})
	require.NoError(t, err)

	switch mutation {
	case "add_column":
		require.Equal(t, int64(11), fetched[0].Fields["added"])
		require.Equal(t, int64(4), fetched[1].Fields["added"])

	case "alter_column":
		require.Equal(t, int64(11), fetched[0].Fields["renamed"])
		require.Equal(t, int64(13), fetched[1].Fields["renamed"])
		{
			_, found := fetched[0].Fields["alter_me"]
			require.False(t, found)
		}

	case "drop_column":
		{
			_, found := fetched[0].Fields["drop_me"]
			require.False(t, found)
		}
	}
}

func assertAtomicLiveDocuments(t *testing.T, collection *Collection) {
	t.Helper()
	fetched, err := collection.Fetch(context.Background(), []string{"a", "b", "c"}, Projection{IncludeVectors: true})
	require.NoError(t, err)
	require.NotNil(t, fetched[0])
	require.True(t, fetched[0].DocID == 3)
	require.Nil(t, fetched[1])
	require.NotNil(t, fetched[2])
	require.True(t, fetched[2].DocID == 2)
	require.Equal(t, int32(10), fetched[0].Fields["rating"])
	require.Equal(t, int32(3), fetched[2].Fields["rating"])

	results, err := collection.Query(context.Background(), VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "indexed >= 1",
	})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "c"}, documentKeys(results))
}

func assertAtomicContinuedWrite(t *testing.T, collection *Collection, mutation string, committed bool) {
	t.Helper()
	document := atomicRecoveryDocument("d", "durable", 4, 4, 14, 24, 1)
	if committed {
		switch mutation {
		case "add_column":
			document.Fields["added"] = int64(5)
		case "alter_column":
			delete(document.Fields, "alter_me")
			document.Fields["renamed"] = int64(14)
		case "drop_column":
			delete(document.Fields, "drop_me")
		}
	}
	inserted, err := collection.Insert(context.Background(), []Document{document})
	require.NoError(t, err)
	require.True(t, inserted[0].DocID == 4)
}

func startAtomicRecoveryChild(t *testing.T, path, mutation, phase string) *exec.Cmd {
	t.Helper()
	command := exec.Command(os.Args[0], "-test.run=^TestAtomicDDLAndOptimizeCrashRecovery$")
	command.Env = append(os.Environ(),
		atomicRecoveryPathEnv+"="+path,
		atomicRecoveryMutationEnv+"="+mutation,
		atomicRecoveryPhaseEnv+"="+phase,
	)
	{
		err := command.Start()
		require.NoError(t, err)
	}

	t.Cleanup(func() {
		if command.ProcessState == nil {
			_ = command.Process.Kill()
			_ = command.Wait()
		}
	})
	return command
}

func waitForAtomicMarker(t *testing.T, command *exec.Cmd, marker string) {
	t.Helper()
	deadline := time.NewTimer(10 * time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			if _, err := os.Stat(marker); err == nil {
				return
			} else if !errors.Is(err, os.ErrNotExist) {
				require.NoError(t, err)
			}
		case <-deadline.C:
			require.FailNowf(t, "atomic marker timeout", "child %d did not create marker %q", command.Process.Pid, marker)
		}
	}
}

func waitForAdditionalSegment(t *testing.T, command *exec.Cmd, path string, initial int) {
	t.Helper()
	deadline := time.NewTimer(10 * time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			if countAtomicSegments(t, path) > initial {
				return
			}
		case <-deadline.C:
			require.FailNowf(t, "segment artifact timeout", "child %d did not create pre-commit segment artifacts", command.Process.Pid)
		}
	}
}

func countAtomicSegments(t *testing.T, path string) int {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(path, "segments", "*", "data-*.seg"))
	require.NoError(t, err)

	return len(files)
}

func killAtomicRecoveryChild(t *testing.T, command *exec.Cmd) {
	t.Helper()
	{
		err := command.Process.Kill()
		require.NoError(t, err)
	}
	{
		err := command.Wait()
		require.Error(t, err,
			"killed atomic recovery child exited successfully")
	}
}

func TestCollectionBinaryInvertedDDLQueryOptimizeAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "binary-invert")
	schema := NewCollectionSchema("binary_invert",
		FieldSchema{Name: "payload", DataType: DataTypeBinary, Nullable: true},
		FieldSchema{Name: "blobs", DataType: DataTypeArrayBinary, Nullable: true, Index: NewInvertIndexParams()},
		FieldSchema{
			Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2,
			Index: NewFlatIndexParams(MetricTypeIP),
		},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{
			{PrimaryKey: "a", Fields: map[string]any{"payload": Binary("x"), "blobs": BinaryArray{Binary("x"), Binary("y")}, "embedding": VectorFP32{3, 0}}},
			{PrimaryKey: "b", Fields: map[string]any{"payload": Binary("y"), "blobs": BinaryArray{Binary("z")}, "embedding": VectorFP32{2, 0}}},
			{PrimaryKey: "c", Fields: map[string]any{"payload": nil, "blobs": nil, "embedding": VectorFP32{1, 0}}},
		})
		require.NoError(t, err)
	}
	{
		err := collection.CreateIndex(ctx, "payload", NewInvertIndexParams(), CreateIndexOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	plan, err := buildFilterPlan("payload = 'x'", collection.Schema())
	require.NoError(t, err)

	documents, err := collection.store.LiveDocuments(ctx)
	require.NoError(t, err)

	decoded := make([]Document, len(documents))
	for index := range documents {
		decoded[index], err = decodeStoredDocument(documents[index])
		require.NoError(t, err)
	}
	evaluated, err := evaluateFilterDocuments(ctx, plan, decoded, 1)
	require.NoError(t, err)
	require.True(t, evaluated.usedIndex)

	assertQuery := func(handle *Collection, filter string, want []string) {
		t.Helper()
		results, queryErr := handle.Query(ctx, VectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: filter,
		})
		require.NoError(t, queryErr)
		require.Equal(t, want, documentKeys(results))
	}
	assertQuery(collection, "payload IN ('x', 'z')", []string{"a"})
	assertQuery(collection, "payload >= 'y'", []string{"b"})
	assertQuery(collection, "payload IS NULL", []string{"c"})
	assertQuery(collection, "blobs CONTAIN_ANY ('y')", []string{"a"})
	assertQuery(collection, "array_length(blobs) = 1", []string{"b"})
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	options := NewCollectionOptions()
	options.ReadOnly = true
	reopened, err := Open(ctx, path, options)
	require.NoError(t, err)

	defer reopened.Close()
	assertQuery(reopened, "payload IN ('x', 'z') AND blobs CONTAIN_ALL ('x', 'y')", []string{"a"})
}

func TestCreateFTSIndexBackfillQueryAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "fts-backfill")
	schema := NewCollectionSchema("fts_backfill",
		FieldSchema{Name: "title", DataType: DataTypeString},
		FieldSchema{
			Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2,
			Index: NewFlatIndexParams(MetricTypeIP),
		},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{
			{PrimaryKey: "go", Fields: map[string]any{"title": "Go vector search", "embedding": VectorFP32{1, 0}}},
			{PrimaryKey: "db", Fields: map[string]any{"title": "Database internals", "embedding": VectorFP32{0.5, 0}}},
		})
		require.NoError(t, err)
	}

	params := FTSIndexParams{Tokenizer: "whitespace", Filters: []string{"lowercase", "stemmer"}, ExtraParams: `{"stemmer_lang":"english"}`}
	{
		err := collection.CreateIndex(ctx, "title", params, CreateIndexOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	field, found := collection.Schema().Field("title")
	require.True(t, found)
	require.True(t, equalIndexParams(field.Index, params))
	require.True(t, collection.Stats().IndexCompleteness["title"] == 1)

	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "title", FTS: &FTSClause{Match: "searching"}, NumCandidates: 2},
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 2},
		},
		TopK: 1, Projection: Projection{OutputFields: []string{"title"}},
	}
	want, err := collection.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Len(t, want, 1)
	require.True(t, want[0].PrimaryKey == "go")
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	options := NewCollectionOptions()
	options.ReadOnly = true
	reopened, err := Open(ctx, path, options)
	require.NoError(t, err)

	defer reopened.Close()
	got, err := reopened.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Equal(t, want, got)
}

func TestCreateFTSIndexBackfillFailureRollsBack(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "fts-rollback")
	schema := NewCollectionSchema("fts_rollback", FieldSchema{Name: "title", DataType: DataTypeString})
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"title": "中文"}}})
		require.NoError(t, err)
	}

	beforeSchema := collection.Schema()
	beforeGeneration := collection.store.Manifest().Generation
	params := FTSIndexParams{Tokenizer: "jieba", ExtraParams: `{"jieba_dict_dir":"missing-jieba-resources"}`}
	{
		err := collection.CreateIndex(ctx, "title", params, CreateIndexOptions{})
		require.Error(t, err,
			"missing Jieba resources unexpectedly succeeded")
	}
	require.Equal(t, beforeSchema, collection.Schema(),
		"failed FTS backfill changed schema or manifest")
	require.Equal(t, beforeGeneration, collection.store.Manifest().Generation,
		"failed FTS backfill changed schema or manifest")
}

func TestCreateScalarIndexPublishesSchemaAndSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "create-scalar-index")
	schema := createIndexSchema()
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{
			createIndexDocument("a", "alpha", 1, 3),
			createIndexDocument("b", "alphabet", 2, 2),
			createIndexDocument("c", "beta", 3, 1),
		})
		require.NoError(t, err)
	}

	initialGeneration := collection.store.Manifest().Generation
	params := NewInvertIndexParams()
	{
		err := collection.CreateIndex(ctx, "rating", &params, CreateIndexOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	createdGeneration := collection.store.Manifest().Generation
	require.True(t, createdGeneration > initialGeneration)

	rating, _ := collection.Schema().Field("rating")
	require.True(t, equalIndexParams(rating.Index, NewInvertIndexParams()))

	params.EnableExtendedWildcard = true
	rating, _ = collection.Schema().Field("rating")
	stored := rating.Index.(InvertIndexParams)
	require.False(t, stored.EnableExtendedWildcard,
		"CreateIndex retained caller-owned parameter pointer")

	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "rating>=2",
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, documentKeys(results))
	{
		err := collection.CreateIndex(ctx, "rating", NewInvertIndexParams(), CreateIndexOptions{})
		require.NoError(t, err)
	}
	{
		got := collection.store.Manifest().Generation
		require.Equal(t, createdGeneration, got)
	}

	changed := NewInvertIndexParams()
	changed.EnableRangeOptimization = false
	{
		err := collection.CreateIndex(ctx, "rating", changed, CreateIndexOptions{Concurrency: 1})
		require.NoError(t, err)
	}
	require.True(t, collection.store.Manifest().Generation > createdGeneration,
		"changed index parameters did not publish a generation")

	extended := NewInvertIndexParams()
	extended.EnableExtendedWildcard = true
	{
		err := collection.CreateIndex(ctx, "title", extended, CreateIndexOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	results, err = collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "title LIKE '%bet'",
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b"}, documentKeys(results))
	{
		// The schema manifest and existing write WAL are independently durable;
		// neither needs a Flush before reopening.
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	rating, _ = collection.Schema().Field("rating")
	title, _ := collection.Schema().Field("title")
	require.NotNil(t, rating.Index)
	require.False(t, rating.Index.(InvertIndexParams).EnableRangeOptimization)
	require.NotNil(t, title.Index)
	require.True(t, title.Index.(InvertIndexParams).EnableExtendedWildcard)
	require.True(t, collection.Stats().DocumentCount == 3)
}

func TestCreateFlatIndexChangesMetricAtomically(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "create-flat-index")
	schema := NewCollectionSchema("create_flat",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{
			{PrimaryKey: "far", Fields: map[string]any{"embedding": VectorFP32{10, 0}}},
			{PrimaryKey: "near", Fields: map[string]any{"embedding": VectorFP32{1, 0}}},
		})
		require.NoError(t, err)
	}

	query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 2}
	before, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"far", "near"}, documentKeys(before))
	{
		err := collection.CreateIndex(ctx, "embedding", NewFlatIndexParams(MetricTypeL2), CreateIndexOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	after, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"near", "far"}, documentKeys(after))
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	after, err = collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"near", "far"}, documentKeys(after))
}

func TestCreateIndexValidationAndRollback(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "create-index-errors")
	schema := NewCollectionSchema("index_errors",
		FieldSchema{Name: "text", DataType: DataTypeString},
		FieldSchema{Name: "already_fts", DataType: DataTypeString, Index: NewFTSIndexParams()},
		FieldSchema{Name: "binary", DataType: DataTypeBinary},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	var typedNil *InvertIndexParams
	invalidFlat := NewFlatIndexParams(MetricTypeUndefined)
	tests := []struct {
		name    string
		column  string
		index   IndexParams
		options CreateIndexOptions
		want    error
	}{
		{"empty-column", "", NewInvertIndexParams(), CreateIndexOptions{}, ErrInvalidArgument},
		{"nil-index", "text", nil, CreateIndexOptions{}, ErrInvalidArgument},
		{"typed-nil-index", "text", typedNil, CreateIndexOptions{}, ErrInvalidArgument},
		{"negative-concurrency", "text", NewInvertIndexParams(), CreateIndexOptions{Concurrency: -1}, ErrInvalidArgument},
		{"missing-field", "missing", NewInvertIndexParams(), CreateIndexOptions{}, ErrNotFound},
		{"invert-vector", "embedding", NewInvertIndexParams(), CreateIndexOptions{}, ErrInvalidArgument},
		{"flat-scalar", "text", NewFlatIndexParams(MetricTypeIP), CreateIndexOptions{}, ErrInvalidArgument},
		{"scalar-index-conflict", "already_fts", NewInvertIndexParams(), CreateIndexOptions{}, ErrNotSupported},
		{"invalid-index-params", "embedding", invalidFlat, CreateIndexOptions{}, ErrInvalidArgument},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			beforeSchema := collection.Schema()
			beforeGeneration := collection.store.Manifest().Generation
			err := collection.CreateIndex(ctx, testCase.column, testCase.index, testCase.options)
			require.ErrorIs(t, err, testCase.want)
			require.Equal(t, beforeSchema, collection.Schema(),
				"failed CreateIndex changed schema or manifest")
			require.Equal(t, beforeGeneration, collection.store.Manifest().Generation,
				"failed CreateIndex changed schema or manifest")
		})
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.CreateIndex(canceled, "text", NewInvertIndexParams(), CreateIndexOptions{})
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := collection.CreateIndex(nil, "text", NewInvertIndexParams(), CreateIndexOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	var nilCollection *Collection
	{
		err := nilCollection.CreateIndex(ctx, "text", NewInvertIndexParams(), CreateIndexOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}
	{
		err := collection.CreateIndex(ctx, "text", NewInvertIndexParams(), CreateIndexOptions{})
		require.ErrorIs(t, err, ErrFailedPrecondition)
	}

	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	readOnly, err := Open(ctx, path, readOnlyOptions)
	require.NoError(t, err)

	defer readOnly.Close()
	{
		err := readOnly.CreateIndex(ctx, "text", NewInvertIndexParams(), CreateIndexOptions{})
		require.ErrorIs(t, err, ErrPermissionDenied)
	}
}

func TestCreateIndexBackfillFailureLeavesSchemaUnchanged(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "create-index-rollback"), createIndexSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.store.Insert(ctx, []db.WriteInput{{PrimaryKey: "corrupt", Payload: []byte("not-a-document")}})
		require.NoError(t, err)
	}

	before := collection.Schema()
	generation := collection.store.Manifest().Generation
	{
		err := collection.CreateIndex(ctx, "rating", NewInvertIndexParams(), CreateIndexOptions{Concurrency: 2})
		require.Error(t, err,
			"CreateIndex unexpectedly accepted corrupt backfill data")
	}
	require.Equal(t, before, collection.Schema(),
		"failed backfill changed published schema")
	require.Equal(t, generation, collection.store.Manifest().Generation,
		"failed backfill changed published schema")
}

func createIndexSchema() CollectionSchema {
	schema := NewCollectionSchema("create_index",
		FieldSchema{Name: "title", DataType: DataTypeString},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func createIndexDocument(primaryKey, title string, rating int32, score float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"title": title, "rating": rating, "embedding": VectorFP32{score, 0},
	}}
}

func TestDeleteByFilterAcrossSegmentsAndWALRecovery(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "delete-filter")
	schema := deleteFilterSchema()
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	first := []Document{
		deleteFilterDocument("a", "alpha", int32(1), StringArray{"red"}, 5),
		deleteFilterDocument("b", "beta", int32(2), StringArray{"blue"}, 4),
		deleteFilterDocument("c", "gamma", nil, StringArray{"red", "blue"}, 3),
	}
	{
		_, err := collection.Insert(ctx, first)
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}

	second := []Document{
		deleteFilterDocument("d", "delta", int32(4), StringArray{}, 2),
		deleteFilterDocument("e", "omega", int32(5), nil, 1),
	}
	{
		_, err := collection.Insert(ctx, second)
		require.NoError(t, err)
	}
	{
		err := collection.DeleteByFilter(ctx, "rating >")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		got := collection.Stats().DocumentCount
		require.True(t, got == 5)
	}
	{
		err := collection.DeleteByFilter(ctx, "(rating>=2 AND title LIKE 'd%') OR tags CONTAIN_ALL ('red', 'blue')")
		require.NoError(t, err)
	}
	{
		got := collection.Stats().DocumentCount
		require.True(t, got == 3)
	}

	assertFetchedPresence(t, collection, []string{"a", "b", "c", "d", "e"}, []bool{true, true, false, false, true})
	{
		err := collection.DeleteByFilter(ctx, "rating>100")
		require.NoError(t, err)
	}
	{
		err := collection.DeleteByFilter(ctx, "title LIKE '%ta'")
		require.NoError(t, err)
	}
	{
		got := collection.Stats().DocumentCount
		require.True(t, got == 2)
	}
	{
		// Close without Flush: both immutable-segment and writing-segment deletes
		// must be reconstructed from the WAL.
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	assertFetchedPresence(t, collection, []string{"a", "b", "c", "d", "e"}, []bool{true, false, false, false, true})
	{
		got := collection.Stats().DocumentCount
		require.True(t, got == 2)
	}
	{
		err := collection.DeleteByFilter(ctx, "title LIKE '%'")
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	collection, err = Open(ctx, path, readOnlyOptions)
	require.NoError(t, err)

	defer collection.Close()
	{
		got := collection.Stats().DocumentCount
		require.True(t, got == 0)
	}
}

func TestDeleteByFilterUsesOnlyCurrentDocumentVersions(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "delete-current"), deleteFilterSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{deleteFilterDocument("versioned", "before", int32(1), StringArray{"old"}, 1)})
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		_, err := collection.Update(ctx, []Document{{PrimaryKey: "versioned", Fields: map[string]any{
			"title": "after", "rating": int32(3), "tags": StringArray{"new"},
		}}})
		require.NoError(t, err)
	}
	{
		err := collection.DeleteByFilter(ctx, "rating=1 OR tags CONTAIN_ANY ('old')")
		require.NoError(t, err)
	}

	assertFetchedPresence(t, collection, []string{"versioned"}, []bool{true})
	{
		err := collection.DeleteByFilter(ctx, "rating=3 AND tags CONTAIN_ANY ('new')")
		require.NoError(t, err)
	}

	assertFetchedPresence(t, collection, []string{"versioned"}, []bool{false})
}

func TestDeleteByFilterValidationCancellationAndLifecycle(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	{
		err := nilCollection.DeleteByFilter(ctx, "rating=1")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	path := filepath.Join(t.TempDir(), "delete-filter-errors")
	collection, err := CreateAndOpen(ctx, path, deleteFilterSchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		err := collection.DeleteByFilter(nil, "rating=1")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	for _, filter := range []string{"", "   ", "missing=1", "embedding=1"} {
		{
			err := collection.DeleteByFilter(ctx, filter)
			assert.ErrorIs(t, err, ErrInvalidArgument)
		}
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.DeleteByFilter(canceled, "rating=1")
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := collection.DeleteByFilter(ctx, "rating=1")
		require.NoError(t, err)
	}
	{
		_, err := collection.Insert(ctx, []Document{deleteFilterDocument("a", "alpha", int32(1), StringArray{}, 1)})
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}
	{
		err := collection.DeleteByFilter(ctx, "rating=1")
		require.ErrorIs(t, err, ErrFailedPrecondition)
	}

	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	readOnly, err := Open(ctx, path, readOnlyOptions)
	require.NoError(t, err)

	defer readOnly.Close()
	{
		err := readOnly.DeleteByFilter(ctx, "rating=1")
		require.ErrorIs(t, err, ErrPermissionDenied)
	}
	{
		got := readOnly.Stats().DocumentCount
		require.True(t, got == 1)
	}
}

func deleteFilterSchema() CollectionSchema {
	extended := NewInvertIndexParams()
	extended.EnableExtendedWildcard = true
	schema := NewCollectionSchema("delete_filter",
		FieldSchema{Name: "title", DataType: DataTypeString, Index: extended},
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Nullable: true, Index: NewInvertIndexParams()},
		FieldSchema{Name: "tags", DataType: DataTypeArrayString, Nullable: true, Index: NewInvertIndexParams()},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func deleteFilterDocument(primaryKey, title string, rating, tags any, score float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"title": title, "rating": rating, "tags": tags, "embedding": VectorFP32{score, 0},
	}}
}

func assertFetchedPresence(t *testing.T, collection *Collection, keys []string, present []bool) {
	t.Helper()
	fetched, err := collection.Fetch(context.Background(), keys, Projection{})
	require.NoError(t, err)
	require.Len(t, fetched, len(present))

	for index := range fetched {
		assert.Equal(t, present[index], fetched[index] != nil)
	}
}

func TestDropColumnRemovesPayloadsAndSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-column")
	collection, err := CreateAndOpen(ctx, path, dropColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	inserted, err := collection.Insert(ctx, []Document{
		dropColumnDocument("a", 5, float64(1.25), true, []float32{3, 0}),
		dropColumnDocument("b", 3, nil, true, []float32{2, 0}),
		dropColumnDocument("c", 1, nil, false, []float32{1, 0}),
	})
	require.NoError(t, err)

	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "b", Fields: map[string]any{"rating": int32(4)}}})
	require.NoError(t, err)

	wantIDs := map[string]uint64{"a": inserted[0].DocID, "b": updated[0].DocID, "c": inserted[2].DocID}
	{
		err := collection.DropColumn(ctx, "rating")
		require.NoError(t, err)
	}
	{
		_, found := collection.Schema().Field("rating")
		require.False(t, found,
			"dropped indexed field remains in schema")
	}

	assertStoredFieldAbsent(t, ctx, collection, "rating", wantIDs)
	{
		_, err := collection.Query(ctx, VectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 3, Filter: "rating >= 2",
		})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		err := collection.DropColumn(ctx, "optional")
		require.NoError(t, err)
	}

	assertStoredFieldAbsent(t, ctx, collection, "optional", wantIDs)
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 3,
	})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b", "c"}, documentKeys(results))

	for index := range results {
		{
			_, found := results[index].Fields["rating"]
			require.False(t, found)
		}
		{
			_, found := results[index].Fields["optional"]
			require.False(t, found)
		}
	}
	require.True(t, collection.Stats().DocumentCount == 3)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	assertStoredFieldAbsent(t, ctx, collection, "rating", wantIDs)
	assertStoredFieldAbsent(t, ctx, collection, "optional", wantIDs)
	withDropped := dropColumnDocument("bad", 2, nil, false, []float32{1, 0})
	{
		_, err := collection.Insert(ctx, []Document{withDropped})
		require.Error(t, err,
			"insert containing dropped fields succeeded")
	}

	valid := Document{PrimaryKey: "d", Fields: map[string]any{
		"text": "d", "embedding": VectorFP32{0.5, 0},
	}}
	{
		_, err := collection.Insert(ctx, []Document{valid})
		require.NoError(t, err)
	}
}

func TestDropColumnValidationAndPublicationRollback(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	{
		err := nilCollection.DropColumn(ctx, "rating")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	path := filepath.Join(t.TempDir(), "drop-column-errors")
	collection, err := CreateAndOpen(ctx, path, dropColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{dropColumnDocument("one", 2, nil, false, []float32{1, 0})})
		require.NoError(t, err)
	}
	{
		err := collection.DropColumn(nil, "rating")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	initialSchema := collection.Schema()
	initialGeneration := collection.store.Manifest().Generation
	for _, column := range []string{"", "missing", "text", "embedding"} {
		t.Run(column, func(t *testing.T) {
			{
				err := collection.DropColumn(ctx, column)
				require.Error(t, err,
					"DropColumn succeeded")
			}
			require.Equal(t, initialSchema, collection.Schema())
			require.Equal(t, initialGeneration, collection.store.Manifest().Generation)
		})
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.DropColumn(canceled, "rating")
		require.ErrorIs(t, err, context.Canceled)
	}

	versionLock := flock.New(filepath.Join(path, ".version.lock"))
	locked, err := versionLock.TryLock()
	require.NoError(t, err)
	require.True(t, locked)

	deadline, cancel := context.WithTimeout(ctx, 75*time.Millisecond)
	err = collection.DropColumn(deadline, "rating")
	cancel()
	if !errors.Is(err, context.DeadlineExceeded) {
		_ = versionLock.Close()
	}
	require.ErrorIs(t, err, context.DeadlineExceeded)
	{
		err := versionLock.Close()
		require.NoError(t, err)
	}
	require.Equal(t, initialSchema, collection.Schema())
	require.Equal(t, initialGeneration, collection.store.Manifest().Generation)

	fetched, err := collection.Fetch(ctx, []string{"one"}, Projection{})
	require.NoError(t, err)
	require.NotNil(t, fetched[0])
	require.Equal(t, int32(2), fetched[0].Fields["rating"])
}

func TestDropColumnRejectsLastFieldAndReadOnlyHandle(t *testing.T) {
	ctx := context.Background()
	t.Run("last field", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "drop-last-column")
		schema := NewCollectionSchema("drop_last", FieldSchema{Name: "only", DataType: DataTypeInt32})
		schema.MaxDocsPerSegment = MinMaxDocsPerSegment
		collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
		require.NoError(t, err)

		defer collection.Close()
		{
			err := collection.DropColumn(ctx, "only")
			require.ErrorIs(t, err, ErrInvalidArgument)
		}
	})
	t.Run("read only", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "drop-column-read-only")
		collection, err := CreateAndOpen(ctx, path, dropColumnSchema(), NewCollectionOptions())
		require.NoError(t, err)
		{
			err := collection.Close()
			require.NoError(t, err)
		}

		options := NewCollectionOptions()
		options.ReadOnly = true
		collection, err = Open(ctx, path, options)
		require.NoError(t, err)

		defer collection.Close()
		{
			err := collection.DropColumn(ctx, "rating")
			require.ErrorIs(t, err, ErrPermissionDenied)
		}
	})
}

func TestDropColumnEmptyCollectionPublishesSchemaOnly(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-column-empty")
	collection, err := CreateAndOpen(ctx, path, dropColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	initialGeneration := collection.store.Manifest().Generation
	{
		err := collection.DropColumn(ctx, "rating")
		require.NoError(t, err)
	}
	require.True(t, collection.store.Manifest().Generation > initialGeneration)
	require.Nil(t, collection.store.Manifest().PersistedSegments)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, found := collection.Schema().Field("rating")
		require.False(t, found,
			"reopened schema contains dropped field")
	}
}

func assertStoredFieldAbsent(t *testing.T, ctx context.Context, collection *Collection, field string, wantIDs map[string]uint64) {
	t.Helper()
	stored, err := collection.store.LiveDocuments(ctx)
	require.NoError(t, err)

	for _, item := range stored {
		require.Equal(t, wantIDs[item.PrimaryKey], item.DocID)

		fields, decodeErr := unmarshalDocumentPayload(item.Payload)
		require.NoError(t, decodeErr)
		{
			_, found := fields[field]
			require.False(t, found)
		}
	}
}

func dropColumnSchema() CollectionSchema {
	schema := NewCollectionSchema("drop_columns",
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Index: NewInvertIndexParams()},
		FieldSchema{Name: "optional", DataType: DataTypeDouble, Nullable: true},
		FieldSchema{Name: "text", DataType: DataTypeString},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func dropColumnDocument(primaryKey string, rating int32, optional any, includeOptional bool, embedding []float32) Document {
	fields := map[string]any{
		"rating": rating, "text": primaryKey, "embedding": VectorFP32(embedding),
	}
	if includeOptional {
		fields["optional"] = optional
	}
	return Document{PrimaryKey: primaryKey, Fields: fields}
}

func TestDropScalarIndexPublishesAndPreservesForwardResults(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-scalar-index")
	schema := createIndexSchema()
	schema.Fields[1].Index = NewInvertIndexParams()
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{
			createIndexDocument("a", "alpha", 1, 3),
			createIndexDocument("b", "beta", 2, 2),
			createIndexDocument("c", "gamma", 3, 1),
		})
		require.NoError(t, err)
	}

	query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "rating>=2"}
	before, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, documentKeys(before))

	generation := collection.store.Manifest().Generation
	{
		err := collection.DropIndex(ctx, "rating")
		require.NoError(t, err)
	}
	require.True(t, collection.store.Manifest().Generation > generation,
		"DropIndex did not publish a manifest generation")

	rating, _ := collection.Schema().Field("rating")
	require.Nil(t, rating.Index)
	require.Equal(t, IndexTypeUndefined, rating.IndexType())

	after, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, documentKeys(after))

	idempotentGeneration := collection.store.Manifest().Generation
	{
		err := collection.DropIndex(ctx, "rating")
		require.NoError(t, err)
	}
	require.Equal(t, idempotentGeneration, collection.store.Manifest().Generation,
		"idempotent scalar DropIndex advanced generation")
	{
		err := collection.CreateIndex(ctx, "rating", NewInvertIndexParams(), CreateIndexOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.DropIndex(ctx, "rating")
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	rating, _ = collection.Schema().Field("rating")
	require.Nil(t, rating.Index)
	require.True(t, collection.Stats().DocumentCount == 3)
}

func TestDropVectorIndexRestoresDefaultFlatIP(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-vector-index")
	schema := NewCollectionSchema("drop_vector",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeL2)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{
			{PrimaryKey: "far", Fields: map[string]any{"embedding": VectorFP32{10, 0}}},
			{PrimaryKey: "near", Fields: map[string]any{"embedding": VectorFP32{1, 0}}},
		})
		require.NoError(t, err)
	}

	query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 2}
	before, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"near", "far"}, documentKeys(before))
	{
		err := collection.DropIndex(ctx, "embedding")
		require.NoError(t, err)
	}

	field, _ := collection.Schema().Field("embedding")
	flat, ok := field.Index.(FlatIndexParams)
	require.True(t, ok)
	require.Equal(t, MetricTypeIP, flat.Metric)
	require.Equal(t, QuantizeTypeUndefined, flat.Quantize)

	after, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"far", "near"}, documentKeys(after))

	generation := collection.store.Manifest().Generation
	{
		err := collection.DropIndex(ctx, "embedding")
		require.NoError(t, err)
	}
	require.Equal(t, generation, collection.store.Manifest().Generation,
		"idempotent vector DropIndex advanced generation")
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	field, _ = collection.Schema().Field("embedding")
	{
		flat, ok = field.Index.(FlatIndexParams)
		require.True(t, ok)
		require.Equal(t, MetricTypeIP, flat.Metric)
	}
}

func TestDropUnsupportedOrFTSIndexRemovesMetadata(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-later-index")
	schema := NewCollectionSchema("drop_later",
		FieldSchema{Name: "text", DataType: DataTypeString, Index: NewFTSIndexParams()},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{
			"text": "hello", "embedding": VectorFP32{1, 0},
		}}})
		require.NoError(t, err)
	}
	{
		err := collection.DropIndex(ctx, "text")
		require.NoError(t, err)
	}
	{
		err := collection.DropIndex(ctx, "embedding")
		require.NoError(t, err)
	}

	text, _ := collection.Schema().Field("text")
	embedding, _ := collection.Schema().Field("embedding")
	require.Nil(t, text.Index)
	require.Equal(t, IndexTypeFlat, embedding.IndexType())

	results, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1})
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.True(t, results[0].PrimaryKey == "a")
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	text, _ = collection.Schema().Field("text")
	embedding, _ = collection.Schema().Field("embedding")
	require.Nil(t, text.Index)
	require.Equal(t, IndexTypeFlat, embedding.IndexType())
}

func TestDropIndexValidationLifecycleAndRollback(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-index-errors")
	collection, err := CreateAndOpen(ctx, path, createIndexSchema(), NewCollectionOptions())
	require.NoError(t, err)

	generation := collection.store.Manifest().Generation
	{
		err := collection.DropIndex(ctx, "title")
		require.NoError(t, err)
	}
	require.Equal(t, generation, collection.store.Manifest().Generation,
		"unindexed scalar no-op advanced generation")

	for _, testCase := range []struct {
		name   string
		column string
		want   error
	}{
		{"empty", "", ErrInvalidArgument},
		{"missing", "missing", ErrNotFound},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			before := collection.Schema()
			beforeGeneration := collection.store.Manifest().Generation
			{
				err := collection.DropIndex(ctx, testCase.column)
				require.ErrorIs(t, err, testCase.want)
			}
			require.Equal(t, before, collection.Schema(),
				"failed DropIndex changed schema")
			require.Equal(t, beforeGeneration, collection.store.Manifest().Generation,
				"failed DropIndex changed schema")
		})
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.DropIndex(canceled, "embedding")
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := collection.DropIndex(nil, "embedding")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	var nilCollection *Collection
	{
		err := nilCollection.DropIndex(ctx, "embedding")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}
	{
		err := collection.DropIndex(ctx, "embedding")
		require.ErrorIs(t, err, ErrFailedPrecondition)
	}

	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	readOnly, err := Open(ctx, path, readOnlyOptions)
	require.NoError(t, err)

	defer readOnly.Close()
	{
		err := readOnly.DropIndex(ctx, "embedding")
		require.ErrorIs(t, err, ErrPermissionDenied)
	}

	corrupt, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "drop-index-rollback"), NewCollectionSchema("drop_rollback",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeL2)},
	), NewCollectionOptions())
	require.NoError(t, err)

	defer corrupt.Close()
	{
		_, err := corrupt.store.Insert(ctx, []db.WriteInput{{PrimaryKey: "corrupt", Payload: []byte("bad")}})
		require.NoError(t, err)
	}

	before := corrupt.Schema()
	beforeGeneration := corrupt.store.Manifest().Generation
	{
		err := corrupt.DropIndex(ctx, "embedding")
		require.Error(t, err,
			"DropIndex accepted corrupt vector backfill")
	}
	require.Equal(t, before, corrupt.Schema(),
		"failed vector DropIndex changed schema")
	require.Equal(t, beforeGeneration, corrupt.store.Manifest().Generation,
		"failed vector DropIndex changed schema")
}

func TestCollectionSQLFilterQueryGroupByAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "filters")
	schema := NewCollectionSchema("filters",
		FieldSchema{Name: "title", DataType: DataTypeString, Nullable: true},
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Nullable: true},
		FieldSchema{Name: "tags", DataType: DataTypeArrayString, Nullable: true},
		FieldSchema{Name: "numbers", DataType: DataTypeArrayInt32},
		FieldSchema{Name: "active", DataType: DataTypeBool},
		FieldSchema{Name: "payload", DataType: DataTypeBinary},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := []Document{
		filterDocument("a", "user-22", int32(1), StringArray{"red", "blue"}, Int32Array{1, 2}, true, "x", 1),
		filterDocument("b", "user-%22", int32(2), StringArray{"blue"}, Int32Array{}, false, "y", 4),
		filterDocument("c", "user-_22", nil, nil, Int32Array{2, 3}, true, "x", 3),
		filterDocument("d", "other", int32(4), StringArray{}, Int32Array{4}, false, "z", 2),
	}
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	tests := []struct {
		filter string
		want   []string
	}{
		{"rating=2 OR rating=4", []string{"b", "d"}},
		{`title LIKE 'user-\%%'`, []string{"b"}},
		{`title LIKE 'user-\_%'`, []string{"c"}},
		{"tags CONTAIN_ALL ('red', 'blue')", []string{"a"}},
		{"tags CONTAIN_ANY ('blue')", []string{"b", "a"}},
		{"tags CONTAIN_ALL ()", []string{"b", "d", "a"}},
		{"tags NOT CONTAIN_ANY ()", []string{"b", "d", "a"}},
		{"tags CONTAIN_ANY ()", []string{}},
		{"array_length(tags) = 0", []string{"d"}},
		{"rating IS NULL", []string{"c"}},
		{"rating IS NOT NULL AND active = false", []string{"b", "d"}},
		{"payload = 'x'", []string{"c", "a"}},
	}
	for _, testCase := range tests {
		t.Run(testCase.filter, func(t *testing.T) {
			results, queryErr := collection.Query(ctx, VectorQuery{
				Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
				Filter: testCase.filter, Projection: Projection{OutputFields: []string{}},
			})
			require.NoError(t, queryErr)
			{
				got := documentKeys(results)
				require.Equal(t, testCase.want, got)
			}
		})
	}

	sparse, err := collection.Query(ctx, VectorQuery{
		Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{1}, Values: []float32{1}},
		TopK: 10, Filter: "numbers CONTAIN_ANY (2)",
	})
	require.NoError(t, err)
	require.Equal(t, []string{"c", "a"}, documentKeys(sparse))

	groups, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, Filter: "tags CONTAIN_ANY ('blue')",
		GroupByField: "active", GroupCount: 2, TopKPerGroup: 2,
	})
	require.NoError(t, err)
	require.Len(t, groups, 2)
	require.True(t, groups[0].Value == "false")
	require.True(t, groups[1].Value == "true")
	require.Equal(t, []string{"b"}, documentKeys(groups[0].Documents))
	require.Equal(t, []string{"a"}, documentKeys(groups[1].Documents))
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	readOnly := NewCollectionOptions()
	readOnly.ReadOnly = true
	collection, err = Open(ctx, path, readOnly)
	require.NoError(t, err)

	defer collection.Close()
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "rating >= 2",
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "d"}, documentKeys(results))
}

func TestCollectionSQLFilterValidationAndCancellation(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "filter-errors"), testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0})})
		require.NoError(t, err)
	}

	for _, filter := range []string{
		"rating >", "missing=1", "rating='bad'", "rating=2147483648",
		"embedding=1", "category CONTAIN_ANY ('low')", "rating LIKE '1%'",
	} {
		_, queryErr := collection.Query(ctx, VectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1, Filter: filter,
		})
		assert.ErrorIs(t, queryErr, ErrInvalidArgument)
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = collection.Query(canceled, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1, Filter: "rating=1",
	})
	require.ErrorIs(t, err, context.Canceled)
}

func TestCollectionScalarInvertedCandidatesMatchForwardSemantics(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "inverted-filters")
	extended := NewInvertIndexParams()
	extended.EnableExtendedWildcard = true
	schema := NewCollectionSchema("inverted_filters",
		FieldSchema{Name: "title", DataType: DataTypeString, Nullable: true, Index: extended},
		FieldSchema{Name: "code", DataType: DataTypeString, Index: NewInvertIndexParams()},
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Nullable: true, Index: NewInvertIndexParams()},
		FieldSchema{Name: "tags", DataType: DataTypeArrayString, Nullable: true, Index: NewInvertIndexParams()},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	plan, err := buildFilterPlan("title LIKE '%alpha' AND rating>=2", schema)
	require.NoError(t, err)

	configured := make(map[string]dbsql.Field)
	for _, field := range plan.Fields() {
		configured[field.Name] = field
	}
	require.True(t, configured["title"].Indexed)
	require.True(t, configured["title"].ExtendedWildcard)
	require.True(t, configured["title"].RangeOptimized)
	require.True(t, configured["rating"].Indexed)
	require.True(t, configured["rating"].RangeOptimized)

	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := []Document{
		invertedFilterDocument("a", "alpha", "alpha", int32(1), StringArray{"red", "blue"}, 5),
		invertedFilterDocument("b", "alphabet", "alphabet", int32(2), StringArray{"blue"}, 4),
		invertedFilterDocument("c", "beta", "beta", int32(3), nil, 3),
		invertedFilterDocument("d", "gamma-alpha", "gamma-alpha", int32(4), StringArray{}, 2),
		invertedFilterDocument("e", nil, "omega", nil, StringArray{"red"}, 1),
	}
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	assertQuery := func(filter string, want []string) {
		t.Helper()
		results, queryErr := collection.Query(ctx, VectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: filter,
		})
		require.NoError(t, queryErr)
		{
			got := documentKeys(results)
			require.Equal(t, want, got)
		}
	}
	assertQuery("title LIKE '%alpha'", []string{"a", "d"})
	assertQuery("code LIKE '%bet'", []string{"b"})
	assertQuery("rating>=2 AND code LIKE 'a%'", []string{"b"})
	assertQuery("rating=1 OR code LIKE '%pha%'", []string{"a", "b", "d"})
	assertQuery("tags CONTAIN_ANY ('red')", []string{"a", "e"})
	assertQuery("array_length(tags)=0", []string{"d"})
	assertQuery("title IS NULL", []string{"e"})
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	readOnly := NewCollectionOptions()
	readOnly.ReadOnly = true
	collection, err = Open(ctx, path, readOnly)
	require.NoError(t, err)

	defer collection.Close()
	assertQuery("rating>=2 AND tags NOT CONTAIN_ANY ('blue')", []string{"d"})
}

func TestFilterSchemaRejectsFTSAndValueAdapterCoversEveryScalarArray(t *testing.T) {
	fts := NewFTSIndexParams()
	schema := NewCollectionSchema("fts_filter",
		FieldSchema{Name: "body", DataType: DataTypeString, Index: fts},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	{
		_, err := buildFilterPlan("body='text'", schema)
		require.Error(t, err,
			"FTS field scalar filter succeeded")
	}

	tests := []struct {
		kind  dbsql.ValueKind
		array bool
		raw   any
		len   int
	}{
		{dbsql.ValueBinary, false, Binary("x"), 0},
		{dbsql.ValueString, false, "x", 0},
		{dbsql.ValueBool, false, true, 0},
		{dbsql.ValueInt32, false, int32(-1), 0},
		{dbsql.ValueInt64, false, int64(-1), 0},
		{dbsql.ValueUint32, false, uint32(1), 0},
		{dbsql.ValueUint64, false, uint64(1), 0},
		{dbsql.ValueFloat32, false, float32(1.5), 0},
		{dbsql.ValueFloat64, false, float64(1.5), 0},
		{dbsql.ValueBinary, true, BinaryArray{Binary("x")}, 1},
		{dbsql.ValueString, true, StringArray{"x"}, 1},
		{dbsql.ValueBool, true, BoolArray{true}, 1},
		{dbsql.ValueInt32, true, Int32Array{-1}, 1},
		{dbsql.ValueInt64, true, Int64Array{-1}, 1},
		{dbsql.ValueUint32, true, Uint32Array{1}, 1},
		{dbsql.ValueUint64, true, Uint64Array{1}, 1},
		{dbsql.ValueFloat32, true, Float32Array{1.5}, 1},
		{dbsql.ValueFloat64, true, Float64Array{1.5}, 1},
	}
	for _, testCase := range tests {
		field := dbsql.Field{Name: "value", Kind: testCase.kind, Array: testCase.array, Filterable: true}
		value, err := toFilterValue(field, testCase.raw, true)
		require.NoError(t, err)
		require.Equal(t, testCase.kind, value.Kind())
		require.Equal(t, testCase.array, value.IsArray())
		require.False(t, value.IsNull())

		if testCase.array {
			{
				length, ok := value.Len()
				require.True(t, ok)
				require.Equal(t, testCase.len, length)
			}
		}
	}
	null, err := toFilterValue(dbsql.Field{Name: "missing", Kind: dbsql.ValueString, Filterable: true}, nil, false)
	require.NoError(t, err)
	require.True(t, null.IsNull())
	{
		_, err := toFilterValue(dbsql.Field{Name: "bad", Kind: dbsql.ValueInt32, Filterable: true}, int64(1), true)
		require.Error(t, err,
			"mismatched adapter value succeeded")
	}
}

func filterDocument(primaryKey, title string, rating, tags any, numbers Int32Array, active bool, payload string, score float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"title": title, "rating": rating, "tags": tags, "numbers": numbers,
		"active": active, "payload": Binary(payload),
		"embedding": VectorFP32{score, 0},
		"sparse":    SparseVectorFP32{Indices: []uint32{1}, Values: []float32{score}},
	}}
}

func invertedFilterDocument(primaryKey string, title any, code string, rating any, tags any, score float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"title": title, "code": code, "rating": rating, "tags": tags,
		"embedding": VectorFP32{score, 0},
	}}
}

func TestCollectionDenseQuantizedLinearGroupByAndRefinement(t *testing.T) {
	flat := NewFlatIndexParams(MetricTypeL2)
	flat.Quantize = QuantizeTypeInt4
	flat.Quantizer.EnableRotate = true
	hnsw := NewHNSWIndexParams(MetricTypeL2)
	hnsw.M, hnsw.EFConstruction = 8, 32
	hnsw.Quantize = QuantizeTypeInt8
	hnsw.Quantizer.EnableRotate = true
	rabitq := NewHNSWRaBitQIndexParams(MetricTypeL2)
	rabitq.TotalBits, rabitq.NumClusters = 5, 4
	rabitq.M, rabitq.EFConstruction = 6, 24

	tests := []struct {
		name   string
		index  IndexParams
		params func(refine bool) QueryParams
	}{
		{
			name: "Flat INT4", index: flat,
			params: func(refine bool) QueryParams {
				value := NewFlatQueryParams()
				value.UseRefiner, value.ScaleFactor = refine, 100
				return value
			},
		},
		{
			name: "HNSW INT8", index: hnsw,
			params: func(refine bool) QueryParams {
				value := NewHNSWQueryParams()
				value.Linear, value.UseRefiner = true, refine
				return value
			},
		},
		{
			name: "HNSW RaBitQ", index: rabitq,
			params: func(refine bool) QueryParams {
				value := NewHNSWRaBitQQueryParams()
				value.Linear, value.UseRefiner = true, refine
				return value
			},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			ctx := context.Background()
			path := filepath.Join(t.TempDir(), "collection")
			schema := NewCollectionSchema("dense_group",
				FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 64, Index: testCase.index},
				FieldSchema{Name: "group", DataType: DataTypeString, Nullable: true},
				FieldSchema{Name: "rating", DataType: DataTypeInt32},
			)
			schema.MaxDocsPerSegment = MinMaxDocsPerSegment
			collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
			require.NoError(t, err)

			documents := annRaBitQDocuments(24)
			for position := range documents {
				if position%7 == 0 {
					documents[position].Fields["group"] = nil
				} else {
					documents[position].Fields["group"] = fmt.Sprintf("g%d", position%4)
				}
			}
			{
				_, err := collection.Insert(ctx, documents)
				require.NoError(t, err)
			}

			queryVector := append(VectorFP32(nil), documents[11].Fields["embedding"].(VectorFP32)...)
			queryVector[3] += .137
			filter := "rating >= 1"
			groupQuery := GroupByVectorQuery{
				Field: "embedding", DenseVector: queryVector, Filter: filter,
				GroupByField: "group", GroupCount: 4, TopKPerGroup: 2,
			}

			groupQuery.Params = testCase.params(false)
			firstStageGroups, err := collection.GroupByQuery(ctx, groupQuery)
			require.NoError(t, err)

			firstStage, err := collection.Query(ctx, VectorQuery{
				Field: "embedding", DenseVector: queryVector, TopK: len(documents),
				Filter: filter, Params: testCase.params(false),
			})
			require.NoError(t, err)

			assertCollectionGroupsMatchResults(t, firstStageGroups, firstStage, "group", core.MetricL2, 4, 2)

			groupQuery.Params = testCase.params(true)
			refinedGroups, err := collection.GroupByQuery(ctx, groupQuery)
			require.NoError(t, err)

			refined, err := collection.Query(ctx, VectorQuery{
				Field: "embedding", DenseVector: queryVector, TopK: len(documents),
				Filter: filter, Params: testCase.params(true),
			})
			require.NoError(t, err)

			assertCollectionGroupsMatchResults(t, refinedGroups, refined, "group", core.MetricL2, 4, 2)
			require.True(t, collectionGroupScoresDiffer(firstStageGroups, refinedGroups),
				"refinement did not change any quantized group score")
			{
				err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
				require.NoError(t, err)
			}
			{
				err := collection.Close()
				require.NoError(t, err)
			}

			collection, err = Open(ctx, path, NewCollectionOptions())
			require.NoError(t, err)

			defer collection.Close()
			reopened, err := collection.GroupByQuery(ctx, groupQuery)
			require.NoError(t, err)
			require.Equal(t, refinedGroups, reopened)
		})
	}
}

func TestCollectionSparseFP16LinearGroupByAndRefinement(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "collection")
	index := NewHNSWIndexParams(MetricTypeIP)
	index.M, index.EFConstruction = 8, 32
	index.Quantize = QuantizeTypeFP16
	schema := NewCollectionSchema("sparse_group",
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: index},
		FieldSchema{Name: "group", DataType: DataTypeString, Nullable: true},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := make([]Document, 30)
	for position := range documents {
		group := any(fmt.Sprintf("g%d", position%4))
		if position%9 == 0 {
			group = nil
		}
		documents[position] = Document{PrimaryKey: fmt.Sprintf("s%02d", position), Fields: map[string]any{
			"sparse": SparseVectorFP32{
				Indices: []uint32{uint32(position % 7), uint32(20 + position%11), uint32(50 + position%13)},
				Values:  []float32{float32(position%5) + .12345, float32(position%7) + .33331, float32(position%9) + .77771},
			},
			"group": group, "rating": int32(position % 3),
		}}
	}
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := SparseVectorFP32{
		Indices: []uint32{2, 24, 57},
		Values:  []float32{2.0007, 1.0003, 3.0009},
	}
	params := NewHNSWQueryParams()
	params.Linear = true
	filter := "rating >= 1"
	groupQuery := GroupByVectorQuery{
		Field: "sparse", SparseVector: queryVector, Filter: filter, Params: params,
		GroupByField: "group", GroupCount: 4, TopKPerGroup: 2,
	}
	firstStageGroups, err := collection.GroupByQuery(ctx, groupQuery)
	require.NoError(t, err)

	firstStage, err := collection.Query(ctx, VectorQuery{
		Field: "sparse", SparseVector: queryVector, TopK: len(documents), Filter: filter, Params: params,
	})
	require.NoError(t, err)

	assertCollectionGroupsMatchResults(t, firstStageGroups, firstStage, "group", core.MetricIP, 4, 2)

	params.UseRefiner = true
	groupQuery.Params = params
	refinedGroups, err := collection.GroupByQuery(ctx, groupQuery)
	require.NoError(t, err)

	refined, err := collection.Query(ctx, VectorQuery{
		Field: "sparse", SparseVector: queryVector, TopK: len(documents), Filter: filter, Params: params,
	})
	require.NoError(t, err)

	assertCollectionGroupsMatchResults(t, refinedGroups, refined, "group", core.MetricIP, 4, 2)
	require.True(t, collectionGroupScoresDiffer(firstStageGroups, refinedGroups),
		"sparse refinement did not change any FP16 group score")
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	reopened, err := collection.GroupByQuery(ctx, groupQuery)
	require.NoError(t, err)
	require.Equal(t, refinedGroups, reopened)
}

func TestCollectionNativeDenseHNSWGroupBy(t *testing.T) {
	hnsw := NewHNSWIndexParams(MetricTypeL2)
	hnsw.M, hnsw.EFConstruction = 4, 16
	quantized := hnsw
	quantized.Quantize = QuantizeTypeInt8
	quantized.Quantizer.EnableRotate = true
	rabitq := NewHNSWRaBitQIndexParams(MetricTypeL2)
	rabitq.TotalBits, rabitq.NumClusters, rabitq.SampleCount = 5, 1, 8
	rabitq.M, rabitq.EFConstruction = 4, 16

	tests := []struct {
		name   string
		index  IndexParams
		params func(linear bool) QueryParams
	}{
		{
			name: "HNSW", index: hnsw,
			params: func(linear bool) QueryParams {
				params := NewHNSWQueryParams()
				params.EF, params.Linear = 4, linear
				return params
			},
		},
		{
			name: "HNSW INT8", index: quantized,
			params: func(linear bool) QueryParams {
				params := NewHNSWQueryParams()
				params.EF, params.Linear = 4, linear
				return params
			},
		},
		{
			name: "HNSW RaBitQ", index: rabitq,
			params: func(linear bool) QueryParams {
				params := NewHNSWRaBitQQueryParams()
				params.EF, params.Linear = 4, linear
				return params
			},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			ctx := context.Background()
			path := filepath.Join(t.TempDir(), "collection")
			schema := NewCollectionSchema("native_dense_groups",
				FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 64, Index: testCase.index},
				FieldSchema{Name: "group", DataType: DataTypeString},
			)
			collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
			require.NoError(t, err)

			documents := make([]Document, 8)
			for position := range documents {
				value, group := float32(position)/10, "near"
				if position == 6 {
					value, group = 10, "middle"
				} else if position == 7 {
					value, group = 20, "far"
				}
				vector := make(VectorFP32, 64)
				for dimension := range vector {
					vector[dimension] = value + float32(dimension%3)/1000
				}
				documents[position] = Document{PrimaryKey: fmt.Sprintf("d%d", position), Fields: map[string]any{
					"embedding": vector, "group": group,
				}}
			}
			{
				_, err := collection.Insert(ctx, documents)
				require.NoError(t, err)
			}

			query := GroupByVectorQuery{
				Field: "embedding", DenseVector: make(VectorFP32, 64),
				GroupByField: "group", GroupCount: 3, TopKPerGroup: 1,
			}
			query.Params = testCase.params(false)
			native, err := collection.GroupByQuery(ctx, query)
			require.NoError(t, err)

			query.Params = testCase.params(true)
			linear, err := collection.GroupByQuery(ctx, query)
			require.NoError(t, err)
			require.Equal(t, linear, native)
			{
				err := collection.Close()
				require.NoError(t, err)
			}

			collection, err = Open(ctx, path, NewCollectionOptions())
			require.NoError(t, err)

			defer collection.Close()
			query.Params = testCase.params(false)
			reopened, err := collection.GroupByQuery(ctx, query)
			require.NoError(t, err)
			require.Equal(t, native, reopened)
		})
	}
}

func TestCollectionNativeSparseHNSWGroupBy(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "collection")
	index := NewHNSWIndexParams(MetricTypeIP)
	index.M, index.EFConstruction, index.Quantize = 4, 16, QuantizeTypeFP16
	schema := NewCollectionSchema("native_sparse_groups",
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: index},
		FieldSchema{Name: "group", DataType: DataTypeString},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := make([]Document, 8)
	for position := range documents {
		value, group := float32(10-position)+.1234, "hot"
		if position == 6 {
			value, group = 2.1234, "warm"
		} else if position == 7 {
			value, group = 1.1234, "cold"
		}
		documents[position] = Document{PrimaryKey: fmt.Sprintf("s%d", position), Fields: map[string]any{
			"sparse": SparseVectorFP32{Indices: []uint32{0, 3}, Values: []float32{value, value / 2}},
			"group":  group,
		}}
	}
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	query := GroupByVectorQuery{
		Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{0, 3}, Values: []float32{1, .5}},
		GroupByField: "group", GroupCount: 3, TopKPerGroup: 1,
	}
	params := NewHNSWQueryParams()
	params.EF = 4
	query.Params = params
	native, err := collection.GroupByQuery(ctx, query)
	require.NoError(t, err)

	params.Linear = true
	query.Params = params
	linear, err := collection.GroupByQuery(ctx, query)
	require.NoError(t, err)
	require.Equal(t, linear, native)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	params.Linear = false
	query.Params = params
	reopened, err := collection.GroupByQuery(ctx, query)
	require.NoError(t, err)
	require.Equal(t, native, reopened)
}

func assertCollectionGroupsMatchResults(
	t testing.TB,
	got []GroupResult,
	results []Document,
	groupField string,
	metric core.Metric,
	groupCount, topKPerGroup int,
) {
	t.Helper()
	byGroup := make(map[string][]core.Result)
	byID := make(map[uint64]Document, len(results))
	for _, document := range results {
		value := ""
		if raw := document.Fields[groupField]; raw != nil {
			value = raw.(string)
		}
		byGroup[value] = append(byGroup[value], core.Result{Key: document.DocID, Score: document.Score})
		byID[document.DocID] = document
	}
	batch := make([]core.GroupResult, 0, len(byGroup))
	for value, groupResults := range byGroup {
		batch = append(batch, core.GroupResult{Value: value, Results: groupResults})
	}
	want := core.MergeGroupResults(metric, groupCount, topKPerGroup, batch)
	require.Len(t, got, len(want))

	for groupIndex := range want {
		require.Equal(t, want[groupIndex].Value, got[groupIndex].Value)
		require.Len(t, got[groupIndex].Documents, len(want[groupIndex].Results))

		for documentIndex, result := range want[groupIndex].Results {
			document := got[groupIndex].Documents[documentIndex]
			expected := byID[result.Key]
			require.Equal(t, expected.PrimaryKey, document.PrimaryKey)
			require.Equal(t, result.Key, document.DocID)
			require.Equal(t, result.Score, document.Score)
		}
	}
}

func collectionGroupScoresDiffer(left, right []GroupResult) bool {
	leftScores := make(map[string]float32)
	for _, group := range left {
		for _, document := range group.Documents {
			leftScores[document.PrimaryKey] = document.Score
		}
	}
	for _, group := range right {
		for _, document := range group.Documents {
			if score, found := leftScores[document.PrimaryKey]; found && score != document.Score {
				return true
			}
		}
	}
	return false
}

type testRerankerFunc func(context.Context, []RerankBatch, int) ([]Document, error)

func (f testRerankerFunc) Rerank(ctx context.Context, batches []RerankBatch, topK int) ([]Document, error) {
	return f(ctx, batches, topK)
}

type testNilReranker struct{}

func (*testNilReranker) Rerank(context.Context, []RerankBatch, int) ([]Document, error) {
	return nil, nil
}

func TestCollectionMultiQueryDenseSparseFTSFilterProjectionAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "hybrid")
	collection, err := CreateAndOpen(ctx, path, testMultiQuerySchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, testMultiQueryDocuments())
		require.NoError(t, err)
	}
	{
		completeness := collection.Stats().IndexCompleteness["title"]
		require.True(t, completeness == 1)
	}

	assertBatches := func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
		require.True(t, topK == 2)
		require.Len(t, batches, 3)
		{
			got := []string{batches[0].Field.Name, batches[1].Field.Name, batches[2].Field.Name}
			require.Equal(t, []string{"embedding", "sparse", "title"}, got)
		}
		{
			got := documentKeys(batches[0].Documents)
			require.Equal(t, []string{"a", "b", "d"}, got)
		}
		{
			got := documentKeys(batches[1].Documents)
			require.Equal(t, []string{"b", "a"}, got)
		}
		{
			got := documentKeys(batches[2].Documents)
			require.Equal(t, []string{"b", "a"}, got)
		}

		for _, batch := range batches {
			for _, document := range batch.Documents {
				require.Len(t, document.Fields, 1)
			}
		}
		first := batches[1].Documents[0]
		second := batches[1].Documents[1]
		first.Score, second.Score = 42, 41
		first.Fields["title"] = "forged by reranker"
		return []Document{first, second}, nil
	}
	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 3},
			{Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}}, NumCandidates: 3},
			{Field: "title", FTS: &FTSClause{Match: "go"}, NumCandidates: 3},
		},
		TopK: 2, Filter: "category = 'keep'",
		Projection: Projection{OutputFields: []string{"title"}},
		Reranker:   testRerankerFunc(assertBatches),
	}
	results, err := collection.MultiQuery(ctx, query)
	require.NoError(t, err)

	assertHybridResults(t, results)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	reopenOptions := NewCollectionOptions()
	reopenOptions.ReadOnly = true
	reopened, err := Open(ctx, path, reopenOptions)
	require.NoError(t, err)

	defer reopened.Close()
	results, err = reopened.MultiQuery(ctx, query)
	require.NoError(t, err)

	assertHybridResults(t, results)
}

func TestCollectionMultiQueryFTSExpressionDefaultOperatorAndFilteredBM25(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "fts"), testMultiQuerySchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, testMultiQueryDocuments())
		require.NoError(t, err)
	}

	params := NewFTSQueryParams()
	params.DefaultOperator = "and"
	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "title", FTS: &FTSClause{Query: `"go search"`}, NumCandidates: 4},
			{Field: "title", FTS: &FTSClause{Match: "go database"}, Params: params, NumCandidates: 4},
		},
		TopK: 2,
		Reranker: testRerankerFunc(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
			require.True(t, topK == 2)
			require.Equal(t, []string{"b"}, documentKeys(batches[0].Documents))
			require.Equal(t, []string{"a", "c"}, documentKeys(batches[1].Documents))

			return []Document{batches[0].Documents[0], batches[1].Documents[0]}, nil
		}),
	}
	results, err := collection.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"b", "a"}, documentKeys(results))

	var unfilteredScore, filteredScore float32
	scoreReranker := func(destination *float32) Reranker {
		return testRerankerFunc(func(_ context.Context, batches []RerankBatch, _ int) ([]Document, error) {
			for _, document := range batches[0].Documents {
				if document.PrimaryKey == "a" {
					*destination = document.Score
					return []Document{document}, nil
				}
			}
			require.FailNow(t, "document a missing from FTS candidates")
			return nil, nil
		})
	}
	base := MultiQuery{
		Queries: []SubQuery{
			{Field: "title", FTS: &FTSClause{Match: "go"}, NumCandidates: 4},
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 4},
		},
		TopK: 1, Reranker: scoreReranker(&unfilteredScore),
	}
	{
		_, err := collection.MultiQuery(ctx, base)
		require.NoError(t, err)
	}

	base.Filter = "category = 'keep'"
	base.Reranker = scoreReranker(&filteredScore)
	{
		_, err := collection.MultiQuery(ctx, base)
		require.NoError(t, err)
	}
	require.False(t, unfilteredScore == 0)
	require.Equal(t, unfilteredScore, filteredScore)
}

func TestCollectionMultiQueryVectorParamsAndEmptySnapshot(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "params"), testMultiQuerySchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, testMultiQueryDocuments())
		require.NoError(t, err)
	}

	denseParams := NewFlatQueryParams()
	denseParams.Radius = 0.9
	sparseParams := NewFlatQueryParams()
	sparseParams.Radius = 2
	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, Params: denseParams},
			{Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}}, Params: sparseParams},
		},
		Filter: "category = 'keep'",
		Reranker: testRerankerFunc(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
			require.Equal(t, DefaultMultiQueryTopK, topK)
			{
				got := documentKeys(batches[0].Documents)
				require.Equal(t, []string{"a"}, got)
			}
			{
				got := documentKeys(batches[1].Documents)
				require.Equal(t, []string{"b"}, got)
			}

			return []Document{batches[0].Documents[0], batches[1].Documents[0]}, nil
		}),
	}
	{
		results, err := collection.MultiQuery(ctx, query)
		require.NoError(t, err)
		require.Equal(t, []string{"a", "b"}, documentKeys(results))
	}

	empty, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "empty"), testMultiQuerySchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer empty.Close()
	emptyQuery := query
	emptyQuery.Filter = ""
	emptyQuery.Queries[0].Params = nil
	emptyQuery.Queries[1] = SubQuery{Field: "title", FTS: &FTSClause{Match: "go"}}
	emptyQuery.Reranker = testRerankerFunc(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
		require.Equal(t, DefaultMultiQueryTopK, topK)
		require.Len(t, batches, 2)
		require.Len(t, batches[0].Documents, 0)
		require.Len(t, batches[1].Documents, 0)

		return nil, nil
	})
	{
		results, err := empty.MultiQuery(ctx, emptyQuery)
		require.NoError(t, err)
		require.Len(t, results, 0)
	}
}

func TestCollectionFTSAnalyzerConfiguration(t *testing.T) {
	ctx := context.Background()
	tests := []struct {
		name       string
		params     FTSIndexParams
		text       string
		wantTokens []string
	}{
		{
			name: "whitespace filters",
			params: FTSIndexParams{
				Tokenizer: "whitespace", Filters: []string{"lowercase", "ascii_folding"},
			},
			text: "CAFÉ Go", wantTokens: []string{"cafe", "go"},
		},
		{
			name: "standard max token length",
			params: FTSIndexParams{
				Tokenizer: "standard", ExtraParams: `{"max_token_length":2}`,
			},
			text: "abc", wantTokens: []string{"ab", "c"},
		},
		{
			name: "ngram options",
			params: FTSIndexParams{
				Tokenizer: "ngram", ExtraParams: `{"ngram_min":1,"ngram_max":2,"token_chars":["letter"]}`,
			},
			text: "a1b", wantTokens: []string{"a", "b"},
		},
		{
			name: "stemmer language",
			params: FTSIndexParams{
				Tokenizer: "whitespace", Filters: []string{"lowercase", "stemmer"}, ExtraParams: `{"stemmer_lang":"english"}`,
			},
			text: "RUNNING", wantTokens: []string{"run"},
		},
	}
	jiebaDirectory, err := filepath.Abs(filepath.Join("internal", "core", "testdata", "jieba"))
	require.NoError(t, err)

	jiebaExtra, _ := json.Marshal(map[string]string{
		"jieba_dict_dir": jiebaDirectory,
		"user_dict_path": filepath.Join(jiebaDirectory, "user.dict.utf8"),
		"cut_mode":       "search",
	})
	tests = append(tests, struct {
		name       string
		params     FTSIndexParams
		text       string
		wantTokens []string
	}{
		name: "jieba resources", params: FTSIndexParams{Tokenizer: "jieba", ExtraParams: string(jiebaExtra)},
		text: "中华人民共和国", wantTokens: []string{"中华", "人民", "共和", "共和国", "中华人民共和国"},
	})
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			{
				err := testCase.params.Validate()
				require.NoError(t, err)
			}

			analyzer, err := newCollectionFTSAnalyzer(ctx, testCase.params)
			require.NoError(t, err)

			tokens, err := analyzer.Analyze(ctx, testCase.text)
			require.NoError(t, err)

			got := make([]string, len(tokens))
			for index := range tokens {
				got[index] = tokens[index].Text
			}
			require.Equal(t, testCase.wantTokens, got)
		})
	}

	defaults, err := newCollectionFTSAnalyzer(ctx, NewFTSIndexParams())
	require.NoError(t, err)

	pipeline, ok := defaults.(*core.FTSTokenizerPipeline)
	require.True(t, ok)
	require.True(t, pipeline.TokenizerName() == "standard")
	require.Equal(t, []string{"lowercase"}, pipeline.FilterNames())
}

func TestCollectionMultiQueryValidationAndRerankerBoundaries(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "validation")
	collection, err := CreateAndOpen(ctx, path, testMultiQuerySchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, testMultiQueryDocuments())
		require.NoError(t, err)
	}

	validReranker := testRerankerFunc(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
		return firstDistinctDocuments(batches, topK), nil
	})
	valid := func() MultiQuery {
		return MultiQuery{
			Queries: []SubQuery{
				{Field: "embedding", DenseVector: VectorFP32{1, 0}},
				{Field: "title", FTS: &FTSClause{Match: "go"}},
			},
			Reranker: validReranker,
		}
	}
	tests := []struct {
		name   string
		mutate func(*MultiQuery)
	}{
		{name: "one sub-query", mutate: func(query *MultiQuery) { query.Queries = query.Queries[:1] }},
		{name: "negative top-k", mutate: func(query *MultiQuery) { query.TopK = -1 }},
		{name: "oversized top-k", mutate: func(query *MultiQuery) { query.TopK = MaxQueryTopK + 1 }},
		{name: "negative candidates", mutate: func(query *MultiQuery) { query.Queries[0].NumCandidates = -1 }},
		{name: "oversized candidates", mutate: func(query *MultiQuery) { query.Queries[0].NumCandidates = MaxQueryTopK + 1 }},
		{name: "missing field", mutate: func(query *MultiQuery) { query.Queries[0].Field = "missing" }},
		{name: "missing target", mutate: func(query *MultiQuery) { query.Queries[0].DenseVector = nil }},
		{name: "multiple targets", mutate: func(query *MultiQuery) { query.Queries[0].FTS = &FTSClause{Match: "go"} }},
		{name: "dense target on FTS", mutate: func(query *MultiQuery) { query.Queries[0].Field = "title" }},
		{name: "FTS target on vector", mutate: func(query *MultiQuery) { query.Queries[1].Field = "embedding" }},
		{name: "empty FTS clause", mutate: func(query *MultiQuery) { query.Queries[1].FTS = &FTSClause{} }},
		{name: "two FTS strings", mutate: func(query *MultiQuery) { query.Queries[1].FTS = &FTSClause{Query: "go", Match: "go"} }},
		{name: "wrong FTS params", mutate: func(query *MultiQuery) { value := NewFlatQueryParams(); query.Queries[1].Params = value }},
		{name: "wrong vector params", mutate: func(query *MultiQuery) { value := NewFTSQueryParams(); query.Queries[0].Params = value }},
		{name: "malformed FTS expression", mutate: func(query *MultiQuery) { query.Queries[1].FTS = &FTSClause{Query: "(go"} }},
		{name: "invalid filter", mutate: func(query *MultiQuery) { query.Filter = "rating >>> 1" }},
		{name: "invalid projection", mutate: func(query *MultiQuery) { query.Projection.OutputFields = []string{"missing"} }},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			query := valid()
			testCase.mutate(&query)
			{
				_, err := collection.MultiQuery(ctx, query)
				require.ErrorIs(t, err, ErrInvalidArgument)
			}
		})
	}
	{
		_, err := collection.MultiQuery(ctx, valid())
		require.NoError(t, err)
	}

	explicitRRF := valid()
	explicitRRF.Reranker = NewRRFReranker()
	wantRRF, err := collection.MultiQuery(ctx, explicitRRF)
	require.NoError(t, err)

	for name, reranker := range map[string]Reranker{"nil": nil, "typed nil": (*testNilReranker)(nil)} {
		t.Run(name+" reranker", func(t *testing.T) {
			query := valid()
			query.Reranker = reranker
			got, err := collection.MultiQuery(ctx, query)
			require.NoError(t, err)
			require.Equal(t, wantRRF, got)
		})
	}
	var nilCollection *Collection
	{
		_, err := nilCollection.MultiQuery(ctx, valid())
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		_, err := collection.MultiQuery(nil, valid())
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		_, err := collection.MultiQuery(canceled, valid())
		require.ErrorIs(t, err, context.Canceled)
	}

	foreign := Document{PrimaryKey: "foreign", DocID: math.MaxUint64, Score: 1}
	duplicateQuery := valid()
	duplicateQuery.TopK = 2
	tooManyQuery := valid()
	tooManyQuery.TopK = 1
	boundaryTests := []struct {
		name  string
		query MultiQuery
		make  func([]RerankBatch) []Document
	}{
		{name: "foreign document", query: valid(), make: func([]RerankBatch) []Document { return []Document{foreign} }},
		{name: "duplicate document", query: duplicateQuery, make: func(b []RerankBatch) []Document { return []Document{b[0].Documents[0], b[0].Documents[0]} }},
		{name: "non-finite score", query: valid(), make: func(b []RerankBatch) []Document {
			value := b[0].Documents[0]
			value.Score = float32(math.NaN())
			return []Document{value}
		}},
		{name: "too many documents", query: tooManyQuery, make: func(b []RerankBatch) []Document { return []Document{b[0].Documents[0], b[0].Documents[1]} }},
		{name: "wrong primary key", query: valid(), make: func(b []RerankBatch) []Document {
			value := b[0].Documents[0]
			value.PrimaryKey = "forged"
			return []Document{value}
		}},
	}
	for _, testCase := range boundaryTests {
		t.Run(testCase.name, func(t *testing.T) {
			query := testCase.query
			query.Reranker = testRerankerFunc(func(_ context.Context, batches []RerankBatch, _ int) ([]Document, error) {
				return testCase.make(batches), nil
			})
			{
				_, err := collection.MultiQuery(ctx, query)
				require.ErrorIs(t, err, ErrInvalidArgument)
			}
		})
	}

	sentinel := errors.New("reranker failed")
	errorQuery := valid()
	errorQuery.Reranker = testRerankerFunc(func(context.Context, []RerankBatch, int) ([]Document, error) {
		return nil, sentinel
	})
	{
		_, err := collection.MultiQuery(ctx, errorQuery)
		require.ErrorIs(t, err, sentinel)
	}

	cancelContext, cancelDuringRerank := context.WithCancel(ctx)
	cancelQuery := valid()
	cancelQuery.Reranker = testRerankerFunc(func(_ context.Context, batches []RerankBatch, _ int) ([]Document, error) {
		cancelDuringRerank()
		return batches[0].Documents[:1], nil
	})
	{
		_, err := collection.MultiQuery(cancelContext, cancelQuery)
		require.ErrorIs(t, err, context.Canceled)
	}

	// Caller code must run without the Collection read lock: a write from the
	// reranker completes and does not enter the current candidate snapshot.
	writeQuery := valid()
	writeQuery.TopK = 1
	writeQuery.Reranker = testRerankerFunc(func(callbackContext context.Context, batches []RerankBatch, _ int) ([]Document, error) {
		_, err := collection.Insert(callbackContext, []Document{{PrimaryKey: "later", Fields: map[string]any{
			"title": "later", "category": "keep", "rating": int32(9),
			"embedding": VectorFP32{9, 0}, "sparse": SparseVectorFP32{Indices: []uint32{2}, Values: []float32{9}},
		}}})
		if err != nil {
			return nil, err
		}
		return batches[0].Documents[:1], nil
	})
	results, err := collection.MultiQuery(ctx, writeQuery)
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.False(t, results[0].PrimaryKey == "later")
	require.True(t, collection.Stats().DocumentCount == 5)
	{
		err := collection.Close()
		require.NoError(t, err)
	}
	{
		_, err := collection.MultiQuery(ctx, valid())
		require.ErrorIs(t, err, ErrFailedPrecondition)
	}
}

func TestCollectionMultiQueryConcurrentSnapshotSearch(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "concurrent"), testMultiQuerySchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, testMultiQueryDocuments())
		require.NoError(t, err)
	}

	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 4},
			{Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}}, NumCandidates: 4},
			{Field: "title", FTS: &FTSClause{Match: "go"}, NumCandidates: 4},
		},
		TopK: 3,
		Reranker: testRerankerFunc(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
			return firstDistinctDocuments(batches, topK), nil
		}),
	}
	var wait sync.WaitGroup
	errorsFound := make(chan error, 8)
	for worker := 0; worker < 8; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 12; iteration++ {
				results, err := collection.MultiQuery(ctx, query)
				if err != nil {
					errorsFound <- err
					return
				}
				if got := documentKeys(results); !assert.Equal(t, []string{"c", "a", "b"}, got) {
					errorsFound <- errors.New("non-deterministic MultiQuery result")
					return
				}
			}
		}()
	}
	wait.Wait()
	close(errorsFound)
	for err := range errorsFound {
		require.NoError(t, err)
	}
}

func TestMultiQueryPinnedCompatibilityFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/multi_query_58375ff.json")
	require.NoError(t, err)

	var fixture struct {
		BaselineCommit     string `json:"baseline_commit"`
		QueryHeaderHash    string `json:"query_header_sha256"`
		CollectionHash     string `json:"collection_source_sha256"`
		RerankerHeaderHash string `json:"reranker_header_sha256"`
		MaxTopK            int    `json:"max_topk"`
		MinimumSubQueries  int    `json:"minimum_sub_queries"`
		DefaultTopK        int    `json:"default_topk"`
		DefaultCandidates  int    `json:"default_num_candidates"`
		DefaultFTSOperator string `json:"default_fts_operator"`
	}
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b")
	require.True(t, fixture.QueryHeaderHash == "2c482b4c9832ffb07086e9789c88a4f7de6bc278c3f7ae901b4091e7acbdd193")
	require.True(t, fixture.CollectionHash == "cf4145fa9cbed9bf8975c440f024ce98359b2c6792f008e630db5da7f6422493")
	require.True(t, fixture.RerankerHeaderHash == "bc1949536968bc27f0cb11026d0ab8633dbb46641365455c20b433367837c7d6")
	require.Equal(t, MaxQueryTopK, fixture.MaxTopK)
	require.True(t, fixture.MinimumSubQueries == 2)
	require.Equal(t, DefaultMultiQueryTopK, fixture.DefaultTopK)
	require.Equal(t, DefaultSubQueryCandidates, fixture.DefaultCandidates)
	require.Equal(t, NewFTSQueryParams().DefaultOperator, fixture.DefaultFTSOperator)
}

func FuzzMultiQueryTargetKind(f *testing.F) {
	f.Add(uint8(0))
	f.Add(uint8(1))
	f.Add(uint8(2))
	f.Add(uint8(4))
	f.Add(uint8(7))
	f.Fuzz(func(t *testing.T, flags uint8) {
		flags &= 7
		query := SubQuery{}
		if flags&1 != 0 {
			query.DenseVector = VectorFP32{1}
		}
		if flags&2 != 0 {
			query.SparseVector = SparseVectorFP32{Indices: []uint32{1}, Values: []float32{1}}
		}
		if flags&4 != 0 {
			query.FTS = &FTSClause{Match: "x"}
		}
		_, err := multiQueryTargetKind(query)
		valid := flags == 1 || flags == 2 || flags == 4
		require.Equal(t, valid, err == nil)
	})
}

func BenchmarkV05HybridMultiQuery(b *testing.B) {
	ctx := context.Background()
	schema := testMultiQuerySchema()
	collection, err := CreateAndOpen(ctx, filepath.Join(b.TempDir(), "benchmark"), schema, NewCollectionOptions())
	if err != nil {
		require.NoError(b, err)
	}

	defer collection.Close()
	documents := make([]Document, 256)
	for index := range documents {
		documents[index] = Document{PrimaryKey: "doc-" + benchmarkNumber(index), Fields: map[string]any{
			"title": "go vector search", "category": "keep", "rating": int32(index),
			"embedding": VectorFP32{float32(index) / 256, 1},
			"sparse":    SparseVectorFP32{Indices: []uint32{2}, Values: []float32{float32(index) / 256}},
		}}
	}
	{
		_, err := collection.Insert(ctx, documents)
		if err != nil {
			require.NoError(b, err)
		}
	}

	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 20},
			{Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}}, NumCandidates: 20},
			{Field: "title", FTS: &FTSClause{Match: "go search"}, NumCandidates: 20},
		},
		TopK: 10,
		Reranker: testRerankerFunc(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
			return firstDistinctDocuments(batches, topK), nil
		}),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		{
			_, err := collection.MultiQuery(ctx, query)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

func testMultiQuerySchema() CollectionSchema {
	fts := NewFTSIndexParams()
	schema := NewCollectionSchema("hybrid",
		FieldSchema{Name: "title", DataType: DataTypeString, Nullable: true, Index: fts},
		FieldSchema{Name: "category", DataType: DataTypeString, Index: NewInvertIndexParams()},
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Index: NewInvertIndexParams()},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Nullable: true, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func testMultiQueryDocuments() []Document {
	return []Document{
		{PrimaryKey: "a", Fields: map[string]any{
			"title": "Go database", "category": "keep", "rating": int32(1),
			"embedding": VectorFP32{1, 0}, "sparse": SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}},
		}},
		{PrimaryKey: "b", Fields: map[string]any{
			"title": "Go Go search", "category": "keep", "rating": int32(2),
			"embedding": VectorFP32{0.8, 0}, "sparse": SparseVectorFP32{Indices: []uint32{2}, Values: []float32{3}},
		}},
		{PrimaryKey: "c", Fields: map[string]any{
			"title": "Go database search", "category": "drop", "rating": int32(3),
			"embedding": VectorFP32{2, 0}, "sparse": SparseVectorFP32{Indices: []uint32{2}, Values: []float32{5}},
		}},
		{PrimaryKey: "d", Fields: map[string]any{
			"title": nil, "category": "keep", "rating": int32(4),
			"embedding": VectorFP32{0.2, 0}, "sparse": nil,
		}},
	}
}

func assertHybridResults(t *testing.T, results []Document) {
	t.Helper()
	{
		got := documentKeys(results)
		require.Equal(t, []string{"b", "a"}, got)
	}
	require.True(t, results[0].Score == 42)
	require.True(t, results[1].Score == 41)
	require.Equal(t, map[string]any{"title": "Go Go search"}, results[0].Fields)
	require.Equal(t, map[string]any{"title": "Go database"}, results[1].Fields)
}

func firstDistinctDocuments(batches []RerankBatch, topK int) []Document {
	seen := make(map[uint64]struct{})
	result := make([]Document, 0, topK)
	for _, batch := range batches {
		for _, document := range batch.Documents {
			if _, found := seen[document.DocID]; found {
				continue
			}
			seen[document.DocID] = struct{}{}
			result = append(result, document)
			if len(result) == topK {
				return result
			}
		}
	}
	return result
}

func benchmarkNumber(value int) string {
	const digits = "0123456789abcdef"
	return string([]byte{digits[(value>>8)&15], digits[(value>>4)&15], digits[value&15]})
}

func TestOptimizeFTSCompactsDeletesAndReopens(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "optimize-fts")
	fts := FTSIndexParams{
		Tokenizer: "whitespace", Filters: []string{"lowercase", "stemmer"},
		ExtraParams: `{"stemmer_lang":"english"}`,
	}
	schema := NewCollectionSchema("optimize_fts",
		FieldSchema{Name: "title", DataType: DataTypeString, Index: fts},
		FieldSchema{
			Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2,
			Index: NewFlatIndexParams(MetricTypeIP),
		},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{
			{PrimaryKey: "a", Fields: map[string]any{"title": "Go searching", "embedding": VectorFP32{1, 0}}},
			{PrimaryKey: "b", Fields: map[string]any{"title": "Database search", "embedding": VectorFP32{0.7, 0}}},
			{PrimaryKey: "remove", Fields: map[string]any{"title": "Go removed", "embedding": VectorFP32{0.9, 0}}},
		})
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		_, err := collection.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"title": "Go optimized searching"}}})
		require.NoError(t, err)
	}
	{
		_, err := collection.Delete(ctx, []string{"remove"})
		require.NoError(t, err)
	}

	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "title", FTS: &FTSClause{Match: "optimized search"}, NumCandidates: 3},
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 3},
		},
		TopK: 2, Projection: Projection{OutputFields: []string{"title"}},
	}
	want, err := collection.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Len(t, want, 2)
	require.True(t, want[0].PrimaryKey == "a")

	before := collection.Stats()
	require.True(t, before.DeletedDocuments >= 2)
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	after := collection.Stats()
	require.True(t, after.DocumentCount == 2)
	require.True(t, after.DeletedDocuments == 0)
	require.True(t, after.MutableDocuments == 0)
	require.True(t, after.ImmutableSegments == 2)

	got, err := collection.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Equal(t, want, got)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	options := NewCollectionOptions()
	options.ReadOnly = true
	reopened, err := Open(ctx, path, options)
	require.NoError(t, err)

	defer reopened.Close()
	got, err = reopened.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Equal(t, want, got)
}

func TestOptimizeCompactsLiveDocumentsAndPrunesArtifacts(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "optimize")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)

	documents := []Document{
		testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "bravo", "high", 2, 2, []float32{2, 0}),
		testPublicDocument("c", "charlie", "low", 3, 3, []float32{3, 0}),
		testPublicDocument("d", "delta", "high", 4, 4, []float32{4, 0}),
		testPublicDocument("e", "echo", "low", 5, 5, []float32{5, 0}),
		testPublicDocument("f", "foxtrot", "high", 6, 6, []float32{6, 0}),
	}
	wantIDs := make(map[string]uint64, len(documents))
	for start := 0; start < len(documents); start += 2 {
		results, insertErr := collection.Insert(ctx, documents[start:start+2])
		require.NoError(t, insertErr)

		for index := range results {
			wantIDs[results[index].PrimaryKey] = results[index].DocID
		}
		{
			err := collection.Flush(ctx)
			require.NoError(t, err)
		}
	}
	initial := collection.store.Manifest()
	require.Len(t, initial.PersistedSegments, 3)

	unknown := filepath.Join(path, "segments", "application", "note.txt")
	{
		err := os.MkdirAll(filepath.Dir(unknown), 0o755)
		require.NoError(t, err)
	}
	{
		err := os.WriteFile(unknown, []byte("retain me"), 0o644)
		require.NoError(t, err)
	}

	outside := t.TempDir()
	escapeTarget := filepath.Join(outside, "data-external.seg")
	{
		err := os.WriteFile(escapeTarget, []byte("external"), 0o644)
		require.NoError(t, err)
	}

	escapeLink := filepath.Join(path, "segments", "escape")
	symlinkCreated := os.Symlink(outside, escapeLink) == nil

	before, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 6,
		Projection: Projection{OutputFields: []string{"title"}},
	})
	require.NoError(t, err)
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	optimized := collection.store.Manifest()
	require.True(t, optimized.Generation > initial.Generation)
	require.Len(t, optimized.PersistedSegments, 1)
	require.True(t, optimized.WritingSegmentStartDocID == 6)

	assertOptimizeArtifacts(t, path, 1)
	{
		content, err := os.ReadFile(unknown)
		require.NoError(t, err)
		require.True(t, string(content) == "retain me")
	}

	if symlinkCreated {
		{
			content, err := os.ReadFile(escapeTarget)
			require.NoError(t, err)
			require.True(t, string(content) == "external")
		}
	}
	after, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 6,
		Projection: Projection{OutputFields: []string{"title"}},
	})
	require.NoError(t, err)
	require.Equal(t, before, after)

	assertOptimizeDocumentIDs(t, ctx, collection, wantIDs)

	// A canonical collection is a manifest no-op, while prune remains safe to
	// retry for a process that stopped just after an earlier publication.
	generation := optimized.Generation
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 1})
		require.NoError(t, err)
	}
	{
		got := collection.store.Manifest().Generation
		require.Equal(t, generation, got)
	}

	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"rating": int32(10)}}})
	require.NoError(t, err)

	wantIDs["a"] = updated[0].DocID
	{
		_, err := collection.Delete(ctx, []string{"e"})
		require.NoError(t, err)
	}

	delete(wantIDs, "e")
	temporary := testPublicDocument("temporary", "temporary", "low", 7, 7, []float32{7, 0})
	inserted, err := collection.Insert(ctx, []Document{temporary})
	require.NoError(t, err)
	require.True(t, inserted[0].DocID == 7)
	{
		_, err := collection.Delete(ctx, []string{"temporary"})
		require.NoError(t, err)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	optimized = collection.store.Manifest()
	require.Len(t, optimized.PersistedSegments, 2)
	require.True(t, optimized.WritingSegmentStartDocID == 8)

	assertOptimizeArtifacts(t, path, 2)
	assertOptimizeDocumentIDs(t, ctx, collection, wantIDs)
	fetched, err := collection.Fetch(ctx, []string{"a", "e", "temporary"}, Projection{})
	require.NoError(t, err)
	require.NotNil(t, fetched[0])
	require.Equal(t, int32(10), fetched[0].Fields["rating"])
	require.Nil(t, fetched[1])
	require.Nil(t, fetched[2])

	next := testPublicDocument("next", "next", "low", 8, 8, []float32{8, 0})
	inserted, err = collection.Insert(ctx, []Document{next})
	require.NoError(t, err)
	require.True(t, inserted[0].DocID == 8)

	wantIDs["next"] = 8
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	assertOptimizeDocumentIDs(t, ctx, collection, wantIDs)
	require.Equal(t, uint64(len(wantIDs)), collection.Stats().DocumentCount)
}

func TestOptimizeFullyDeletedCollectionKeepsDocumentIDsMonotonic(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "optimize-empty")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)

	documents := []Document{
		testPublicDocument("a", "a", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "b", "high", 2, 2, []float32{2, 0}),
	}
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		_, err := collection.Delete(ctx, []string{"a", "b"})
		require.NoError(t, err)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{})
		require.NoError(t, err)
	}

	manifest := collection.store.Manifest()
	require.Len(t, manifest.PersistedSegments, 0)
	require.True(t, manifest.WritingSegmentStartDocID == 2)
	require.True(t, collection.Stats().DocumentCount == 0)

	assertOptimizeArtifacts(t, path, 0)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	inserted, err := collection.Insert(ctx, []Document{testPublicDocument("c", "c", "low", 3, 3, []float32{3, 0})})
	require.NoError(t, err)
	require.True(t, inserted[0].DocID == 2)
}

func TestOptimizeValidationAndRollback(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	{
		err := nilCollection.Optimize(ctx, OptimizeOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	path := filepath.Join(t.TempDir(), "optimize-errors")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		err := collection.Optimize(nil, OptimizeOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: -1})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.Optimize(canceled, OptimizeOptions{})
		require.ErrorIs(t, err, context.Canceled)
	}

	initialGeneration := collection.store.Manifest().Generation
	{
		err := collection.Optimize(ctx, OptimizeOptions{})
		require.NoError(t, err)
	}
	{
		got := collection.store.Manifest().Generation
		require.Equal(t, initialGeneration, got)
	}
	{
		_, err := collection.Insert(ctx, []Document{testPublicDocument("a", "a", "low", 1, 1, []float32{1, 0})})
		require.NoError(t, err)
	}

	initialGeneration = collection.store.Manifest().Generation
	versionLock := flock.New(filepath.Join(path, ".version.lock"))
	locked, err := versionLock.TryLock()
	require.NoError(t, err)
	require.True(t, locked)

	deadline, cancel := context.WithTimeout(ctx, 75*time.Millisecond)
	err = collection.Optimize(deadline, OptimizeOptions{Concurrency: 2})
	cancel()
	if !errors.Is(err, context.DeadlineExceeded) {
		_ = versionLock.Close()
	}
	require.ErrorIs(t, err, context.DeadlineExceeded)
	{
		err := versionLock.Close()
		require.NoError(t, err)
	}
	{
		got := collection.store.Manifest().Generation
		require.Equal(t, initialGeneration, got)
	}

	fetched, err := collection.Fetch(ctx, []string{"a"}, Projection{})
	require.NoError(t, err)
	require.NotNil(t, fetched[0])
	require.True(t, fetched[0].DocID == 0)
	{
		err := collection.Close()
		require.NoError(t, err)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{})
		require.ErrorIs(t, err, ErrFailedPrecondition)
	}

	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	collection, err = Open(ctx, path, readOnlyOptions)
	require.NoError(t, err)
	{
		err := collection.Optimize(ctx, OptimizeOptions{})
		require.ErrorIs(t, err, ErrPermissionDenied)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}
}

func assertOptimizeDocumentIDs(t *testing.T, ctx context.Context, collection *Collection, want map[string]uint64) {
	t.Helper()
	keys := make([]string, 0, len(want))
	for key := range want {
		keys = append(keys, key)
	}
	fetched, err := collection.Fetch(ctx, keys, Projection{})
	require.NoError(t, err)

	for index, key := range keys {
		require.NotNil(t, fetched[index])
		require.Equal(t, want[key], fetched[index].DocID)
	}
}

func assertOptimizeArtifacts(t *testing.T, path string, segments int) {
	t.Helper()
	patterns := map[string]int{
		filepath.Join(path, "segments", "*", "data-*.seg"): segments,
		filepath.Join(path, "wal", "*.wal"):                1,
		filepath.Join(path, "wal", "*.wal.lock"):           1,
		filepath.Join(path, "snapshots", "primary-*.snap"): 1,
		filepath.Join(path, "snapshots", "delete-*.snap"):  1,
	}
	for pattern, want := range patterns {
		matches, err := filepath.Glob(pattern)
		require.NoError(t, err)

		filtered := matches[:0]
		for _, name := range matches {
			parent, statErr := os.Lstat(filepath.Dir(name))
			require.NoError(t, statErr)

			if parent.Mode()&os.ModeSymlink == 0 {
				filtered = append(filtered, name)
			}
		}
		matches = filtered
		require.Len(t, matches, want)
	}
}

func TestRuntimeConfigDefaultsAndValidation(t *testing.T) {
	config := NewRuntimeConfig()
	{
		err := config.Validate()
		require.NoError(t, err)
	}

	defaultWorkers := min(runtime.GOMAXPROCS(0), MaxRuntimeConcurrency)
	require.Equal(t, defaultWorkers, config.QueryConcurrency)
	require.Equal(t, defaultWorkers, config.OptimizeConcurrency)
	require.Equal(t, LogLevelWarn, config.LogLevel)
	require.True(t, config.MemoryLimitBytes == 0)
	require.True(t, config.InvertToForwardScanRatio == 0.9)
	require.True(t, config.BruteForceByKeysRatio == 0.1)
	require.True(t, config.FTSBruteForceByKeysRatio == 0.05)

	for level, name := range map[LogLevel]string{
		LogLevelDebug: "DEBUG", LogLevelInfo: "INFO", LogLevelWarn: "WARN",
		LogLevelError: "ERROR", LogLevelFatal: "FATAL",
	} {
		require.True(t, level.Valid())
		require.Equal(t, name, level.String())
	}

	tests := []struct {
		name   string
		mutate func(*RuntimeConfig)
		want   error
	}{
		{name: "small memory", mutate: func(c *RuntimeConfig) { c.MemoryLimitBytes = MinRuntimeMemoryLimit - 1 }, want: ErrInvalidArgument},
		{name: "zero query concurrency", mutate: func(c *RuntimeConfig) { c.QueryConcurrency = 0 }, want: ErrInvalidArgument},
		{name: "large query concurrency", mutate: func(c *RuntimeConfig) { c.QueryConcurrency = MaxRuntimeConcurrency + 1 }, want: ErrInvalidArgument},
		{name: "zero optimize concurrency", mutate: func(c *RuntimeConfig) { c.OptimizeConcurrency = 0 }, want: ErrInvalidArgument},
		{name: "large optimize concurrency", mutate: func(c *RuntimeConfig) { c.OptimizeConcurrency = MaxRuntimeConcurrency + 1 }, want: ErrInvalidArgument},
		{name: "invalid log level", mutate: func(c *RuntimeConfig) { c.LogLevel = 99 }, want: ErrInvalidArgument},
		{name: "negative invert ratio", mutate: func(c *RuntimeConfig) { c.InvertToForwardScanRatio = -0.1 }, want: ErrInvalidArgument},
		{name: "large vector ratio", mutate: func(c *RuntimeConfig) { c.BruteForceByKeysRatio = 1.1 }, want: ErrInvalidArgument},
		{name: "NaN FTS ratio", mutate: func(c *RuntimeConfig) { c.FTSBruteForceByKeysRatio = float32(math.NaN()) }, want: ErrInvalidArgument},
		{name: "query binding", mutate: func(c *RuntimeConfig) { c.QueryThreadBinding = true }, want: ErrNotSupported},
		{name: "optimize binding", mutate: func(c *RuntimeConfig) { c.OptimizeThreadBinding = true }, want: ErrNotSupported},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			value := NewRuntimeConfig()
			testCase.mutate(&value)
			{
				err := value.Validate()
				require.ErrorIs(t, err, testCase.want)
			}
		})
	}
	config.MemoryLimitBytes = MinRuntimeMemoryLimit
	config.Logger = nil
	{
		err := config.Validate()
		require.NoError(t, err)
	}
}

func TestMemoryBudgetLimitsWaitsCancellationAndStats(t *testing.T) {
	budget := newMemoryBudget(10)
	release, err := budget.acquire(context.Background(), 7)
	require.NoError(t, err)

	waitContext, cancel := context.WithCancel(context.Background())
	waitResult := make(chan error, 1)
	go func() {
		_, waitErr := budget.acquire(waitContext, 4)
		waitResult <- waitErr
	}()
	waitForRuntimeCounter(t, func() uint64 {
		_, _, _, waiters := budget.stats()
		return waiters
	}, 1)
	cancel()
	{
		err := <-waitResult
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := budget.acquire(context.Background(), 11)
		require.ErrorIs(t, err, errRuntimeMemoryLimit)
	}

	release()
	releaseAll, err := budget.acquire(context.Background(), 10)
	require.NoError(t, err)

	limit, used, peak, waiters := budget.stats()
	require.True(t, limit == 10)
	require.True(t, used == 10)
	require.True(t, peak == 10)
	require.True(t, waiters == 0)

	releaseAll()
	_, used, peak, _ = budget.stats()
	require.True(t, used == 0)
	require.True(t, peak == 10)

	unlimited := newMemoryBudget(0)
	releaseUnlimited, err := unlimited.acquire(context.Background(), math.MaxUint32)
	require.NoError(t, err)
	{
		limit, used, peak, _ := unlimited.stats()
		require.True(t, limit == 0)
		require.Equal(t, uint64(math.MaxUint32), used)
		require.Equal(t, uint64(math.MaxUint32), peak)
	}

	releaseUnlimited()
}

func TestTaskLimiterBoundsQueuesAndCounts(t *testing.T) {
	limiter := newTaskLimiter(1)
	releaseFirst, err := limiter.acquire(context.Background())
	require.NoError(t, err)

	acquired := make(chan func(), 1)
	go func() {
		release, acquireErr := limiter.acquire(context.Background())
		if acquireErr == nil {
			acquired <- release
		}
	}()
	waitForRuntimeCounter(t, func() uint64 {
		_, _, queued, _ := limiter.stats()
		return queued
	}, 1)
	canceledContext, cancel := context.WithCancel(context.Background())
	canceled := make(chan error, 1)
	go func() {
		_, acquireErr := limiter.acquire(canceledContext)
		canceled <- acquireErr
	}()
	waitForRuntimeCounter(t, func() uint64 {
		_, _, queued, _ := limiter.stats()
		return queued
	}, 2)
	cancel()
	{
		err := <-canceled
		require.ErrorIs(t, err, context.Canceled)
	}

	waitForRuntimeCounter(t, func() uint64 {
		_, _, queued, _ := limiter.stats()
		return queued
	}, 1)
	releaseFirst()
	releaseSecond := <-acquired
	releaseSecond()
	active, peak, queued, completed := limiter.stats()
	require.True(t, active == 0)
	require.True(t, peak == 1)
	require.True(t, queued == 0)
	require.True(t, completed == 2)

	alreadyCanceled, cancelAlready := context.WithCancel(context.Background())
	cancelAlready()
	{
		_, err := limiter.acquire(alreadyCanceled)
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestRuntimeResourcesLoggingAdmissionAndCollectionStats(t *testing.T) {
	handler := &runtimeTestLogHandler{}
	config := NewRuntimeConfig()
	config.Logger = slog.New(handler)
	config.LogLevel = LogLevelDebug
	config.QueryConcurrency = 1
	config.OptimizeConcurrency = 2
	resources := newRuntimeResources(config)

	ctx := context.Background()
	schema := NewCollectionSchema("runtime", FieldSchema{
		Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2,
		Index: NewFlatIndexParams(MetricTypeIP),
	})
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "runtime"), schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	collection.runtime = resources
	{
		_, err := collection.Insert(ctx, []Document{
			{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP32{1, 0}}},
			{PrimaryKey: "b", Fields: map[string]any{"embedding": VectorFP32{0.8, 0}}},
			{PrimaryKey: "c", Fields: map[string]any{"embedding": VectorFP32{0.6, 0}}},
			{PrimaryKey: "d", Fields: map[string]any{"embedding": VectorFP32{0.4, 0}}},
		})
		require.NoError(t, err)
	}

	stats := collection.Stats()
	require.True(t, stats.DocumentCount == 4)
	require.True(t, stats.MutableDocuments == 4)
	require.True(t, stats.ImmutableSegments == 0)
	require.False(t, stats.StorageMemoryBytes == 0)
	{
		_, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 2})
		require.NoError(t, err)
	}

	runtimeStats := resources.stats()
	require.True(t, runtimeStats.CompletedQueries == 1)
	require.True(t, runtimeStats.PeakQueries == 1)
	require.True(t, runtimeStats.MemoryInUseBytes == 0)
	require.False(t, runtimeStats.PeakMemoryBytes == 0)
	{
		messages := handler.messages()
		require.Equal(t, []string{"operation started", "operation completed"}, messages)
	}
	require.True(t, collection.queryWorkers() == 1)
	require.True(t, collection.optimizeWorkers(0) == 2)
	require.True(t, collection.optimizeWorkers(9) == 2)
	require.True(t, collection.optimizeWorkers(1) == 1)
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}

	stats = collection.Stats()
	require.True(t, stats.ImmutableSegments == 1)
	require.True(t, stats.MutableDocuments == 0)
	require.False(t, stats.StorageMemoryBytes == 0)

	beforeDeleteMemory := stats.StorageMemoryBytes
	{
		_, err := collection.Delete(ctx, []string{"a"})
		require.NoError(t, err)
	}

	stats = collection.Stats()
	require.True(t, stats.DocumentCount == 3)
	require.True(t, stats.DeletedDocuments == 1)
	require.Equal(t, beforeDeleteMemory+8, stats.StorageMemoryBytes)
	{
		err := collection.Optimize(ctx, OptimizeOptions{})
		require.NoError(t, err)
	}

	runtimeStats = resources.stats()
	require.True(t, runtimeStats.CompletedOptimizeTasks == 1)
	require.True(t, runtimeStats.PeakOptimizeTasks == 1)
	require.True(t, runtimeStats.MemoryInUseBytes == 0)
	{
		messages := handler.messages()
		require.Equal(t, []string{
			"operation started", "operation completed", "operation started", "operation completed",
		}, messages)
	}

	tiny := NewRuntimeConfig()
	tiny.Logger = nil
	tiny.MemoryLimitBytes = 1
	collection.runtime = newRuntimeResources(tiny)
	{
		_, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1})
		require.ErrorIs(t, err, ErrResourceExhausted)
	}
}

func TestRuntimePlannerRatiosAndFTSCandidateSeek(t *testing.T) {
	{
		got := collectionDiskANNCacheCapacity(2*4096, 100)
		require.True(t, got == 2)
	}
	{
		got := collectionDiskANNCacheCapacity(4095, 100)
		require.True(t, got == 0)
	}
	{
		got := collectionDiskANNCacheCapacity(DefaultMaxBufferSize, 1)
		require.True(t, got == 1)
	}

	documents := testMultiQueryDocuments()
	for index := range documents {
		documents[index].DocID = uint64(index + 1)
	}
	plan, err := buildFilterPlan("category = 'keep'", testMultiQuerySchema())
	require.NoError(t, err)

	indexed, err := evaluateFilterDocuments(context.Background(), plan, documents, 0.9)
	require.NoError(t, err)

	forward, err := evaluateFilterDocuments(context.Background(), plan, documents, 0.5)
	require.NoError(t, err)
	require.True(t, indexed.usedIndex)
	require.False(t, forward.usedIndex)
	require.True(t, indexed.matched == 3)
	require.True(t, indexed.total == 4)
	require.True(t, indexed.useBruteForce(0.75))
	require.False(t, indexed.useBruteForce(0.74))

	field, found := testMultiQuerySchema().Field("title")
	require.True(t, found,
		"missing title field")

	runtime, err := buildCollectionFTSRuntime(context.Background(), field, documents, indexed.predicate)
	require.NoError(t, err)

	posting, err := searchCollectionFTS(context.Background(), runtime, &FTSClause{Match: "go"}, nil, 10, indexed.ordinals, false)
	require.NoError(t, err)

	candidate, err := searchCollectionFTS(context.Background(), runtime, &FTSClause{Match: "go"}, nil, 10, indexed.ordinals, true)
	require.NoError(t, err)
	require.Equal(t, posting, candidate)
}

func TestConfigureRuntimeOneShotSubprocess(t *testing.T) {
	if os.Getenv("ZVEC_RUNTIME_CONFIG_HELPER") == "1" {
		bad := NewRuntimeConfig()
		bad.QueryConcurrency = 0
		{
			err := ConfigureRuntime(bad)
			require.ErrorIs(t, err, ErrInvalidArgument)
		}

		SetDefaultJiebaDictDir("before-config")
		first := NewRuntimeConfig()
		first.Logger = nil
		first.MemoryLimitBytes = MinRuntimeMemoryLimit
		first.QueryConcurrency = 1
		first.OptimizeConcurrency = 2
		first.LogLevel = LogLevelInfo
		first.JiebaDictionaryDir = "configured"
		{
			err := ConfigureRuntime(first)
			require.NoError(t, err)
		}

		second := NewRuntimeConfig()
		second.Logger = nil
		second.QueryConcurrency = 7
		second.JiebaDictionaryDir = "ignored"
		{
			err := ConfigureRuntime(second)
			require.NoError(t, err)
		}

		got := CurrentRuntimeConfig()
		require.True(t, got.QueryConcurrency == 1)
		require.True(t, got.OptimizeConcurrency == 2)
		require.Equal(t, MinRuntimeMemoryLimit, got.MemoryLimitBytes)
		require.Equal(t, LogLevelInfo, got.LogLevel)
		require.True(t, DefaultJiebaDictDir() == "configured")
		{
			stats := CurrentRuntimeStats()
			require.Equal(t, MinRuntimeMemoryLimit, stats.MemoryLimitBytes)
		}

		return
	}
	command := exec.Command(os.Args[0], "-test.run=^TestConfigureRuntimeOneShotSubprocess$")
	command.Env = append(os.Environ(), "ZVEC_RUNTIME_CONFIG_HELPER=1")
	output, err := command.CombinedOutput()
	require.NoError(t, err, "runtime config subprocess output:\n%s", output)
}

func TestRuntimeConfigCompatibilityFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/runtime_config_58375ff.json")
	require.NoError(t, err)

	var fixture struct {
		BaselineCommit string             `json:"baseline_commit"`
		ConfigHeader   string             `json:"config_header_sha256"`
		ConfigSource   string             `json:"config_source_sha256"`
		OptionsHeader  string             `json:"options_header_sha256"`
		StatsHeader    string             `json:"stats_header_sha256"`
		Defaults       map[string]float64 `json:"planner_ratio_defaults"`
		LogLevels      map[string]int     `json:"log_levels"`
	}
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b")
	require.True(t, fixture.ConfigHeader == "e2fdabad1fca4b3ffd647081962c2869b4c376379fc1e5506f1e465c985b1758")
	require.True(t, fixture.ConfigSource == "04c9ea1d60b74dd3c5a1fb78bd61251bd11ab54acf2ada944e780f5800f3d929")
	require.True(t, fixture.OptionsHeader == "865c50a022754ad5101f9f40a03401e2832c5b008713c92d487ebf125334670d")
	require.True(t, fixture.StatsHeader == "791bb777751cb3f76ed79ec8c068a3575068361a717d246fb02f43027fc685af")
	require.Equal(t, map[string]float64{"invert_to_forward": 0.9, "vector_brute_force": 0.1, "fts_brute_force": 0.05}, fixture.Defaults)
	require.Equal(t, map[string]int{"debug": 0, "info": 1, "warn": 2, "error": 3, "fatal": 4}, fixture.LogLevels)
}

func FuzzRuntimeConfigValidation(f *testing.F) {
	f.Add(uint64(0), int32(1), int32(1), uint32(math.Float32bits(0.9)), uint32(math.Float32bits(0.1)))
	f.Add(uint64(MinRuntimeMemoryLimit), int32(4), int32(2), uint32(math.Float32bits(0)), uint32(math.Float32bits(1)))
	f.Fuzz(func(t *testing.T, memory uint64, queryWorkers, optimizeWorkers int32, invertBits, bruteBits uint32) {
		config := NewRuntimeConfig()
		config.Logger = nil
		config.MemoryLimitBytes = memory
		config.QueryConcurrency = int(queryWorkers)
		config.OptimizeConcurrency = int(optimizeWorkers)
		config.InvertToForwardScanRatio = math.Float32frombits(invertBits)
		config.BruteForceByKeysRatio = math.Float32frombits(bruteBits)
		err := config.Validate()
		if err == nil {
			require.True(t, config.QueryConcurrency > 0)
			require.True(t, config.QueryConcurrency <= MaxRuntimeConcurrency)
			require.True(t, config.OptimizeConcurrency > 0)
			require.True(t, config.OptimizeConcurrency <= MaxRuntimeConcurrency)
			require.False(t, config.MemoryLimitBytes != 0 && config.MemoryLimitBytes < MinRuntimeMemoryLimit)

			return
		}
		require.False(t, !errors.Is(err, ErrInvalidArgument) && !errors.Is(err, ErrNotSupported))
	})
}

func BenchmarkRuntimeAdmission(b *testing.B) {
	config := NewRuntimeConfig()
	config.Logger = nil
	resources := newRuntimeResources(config)
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		release, err := resources.begin(context.Background(), runtimeQueryTask, "benchmark", "", 1024)
		if err != nil {
			require.NoError(b, err)
		}

		release()
	}
}

type runtimeTestLogHandler struct {
	mu      sync.Mutex
	records []slog.Record
}

func (*runtimeTestLogHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *runtimeTestLogHandler) Handle(_ context.Context, record slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.records = append(h.records, record.Clone())
	return nil
}

func (h *runtimeTestLogHandler) WithAttrs([]slog.Attr) slog.Handler { return h }

func (h *runtimeTestLogHandler) WithGroup(string) slog.Handler { return h }

func (h *runtimeTestLogHandler) messages() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	messages := make([]string, len(h.records))
	for index := range h.records {
		messages[index] = h.records[index].Message
	}
	return messages
}

func waitForRuntimeCounter(t *testing.T, read func() uint64, want uint64) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if read() == want {
			return
		}
		runtime.Gosched()
	}
	require.Equal(t, want, read())
}
