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

package db

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestManifestRoundTrip(t *testing.T) {
	t.Parallel()

	original := sampleManifest(42)
	encoded, err := MarshalManifest(original)
	require.NoError(t, err)

	decoded, err := UnmarshalManifest(encoded)
	require.NoError(t, err)
	require.Equal(t, original, decoded)
}

func TestManifestCloneIsIndependent(t *testing.T) {
	t.Parallel()

	original := sampleManifest(1)
	clone := original.Clone()
	clone.Schema[0] = '['
	clone.PersistedSegments[0].Files[0] = "changed"
	clone.WritingSegment.Files[0] = "changed"
	clone.SegmentIndexSnapshots[0].Artifacts[0].File = "changed"
	require.False(t, json.Valid(clone.Schema),
		"schema clone shares storage")
	require.True(t, json.Valid(original.Schema),
		"schema clone shares storage")
	require.False(t, original.PersistedSegments[0].Files[0] == "changed",
		"persisted segment clone shares files")
	require.False(t, original.WritingSegment.Files[0] == "changed",
		"writing segment clone shares files")
	require.False(t, original.SegmentIndexSnapshots[0].Artifacts[0].File == "changed",
		"segment index snapshot clone shares artifacts")
}

func TestManifestValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		mutate   func(*Manifest)
		expected error
	}{
		{name: "legacy format", mutate: func(m *Manifest) { m.FormatVersion = 1 }, expected: ErrUnsupportedFormatVersion},
		{name: "future format", mutate: func(m *Manifest) { m.FormatVersion = 3 }, expected: ErrUnsupportedFormatVersion},
		{name: "zero generation", mutate: func(m *Manifest) { m.Generation = 0 }, expected: ErrManifestCorrupt},
		{name: "invalid schema", mutate: func(m *Manifest) { m.Schema = json.RawMessage(`{`) }, expected: ErrManifestCorrupt},
		{name: "null schema", mutate: func(m *Manifest) { m.Schema = json.RawMessage(`null`) }, expected: ErrManifestCorrupt},
		{name: "array schema", mutate: func(m *Manifest) { m.Schema = json.RawMessage(`[]`) }, expected: ErrManifestCorrupt},
		{name: "zero segment capacity", mutate: func(m *Manifest) { m.SegmentMaxDocuments = 0 }, expected: ErrManifestCorrupt},
		{name: "duplicate segment", mutate: func(m *Manifest) { m.PersistedSegments = append(m.PersistedSegments, m.PersistedSegments[0]) }, expected: ErrManifestCorrupt},
		{name: "writing already persisted", mutate: func(m *Manifest) { m.WritingSegment.ID = m.PersistedSegments[0].ID }, expected: ErrManifestCorrupt},
		{name: "next ID not advanced", mutate: func(m *Manifest) { m.NextSegmentID = m.WritingSegment.ID }, expected: ErrManifestCorrupt},
		{name: "empty segment range", mutate: func(m *Manifest) { m.WritingSegment.DocCount = 0 }, expected: ErrManifestCorrupt},
		{name: "descending range", mutate: func(m *Manifest) { m.WritingSegment.MinDocID = 100 }, expected: ErrManifestCorrupt},
		{name: "count exceeds range", mutate: func(m *Manifest) { m.WritingSegment.DocCount = 100 }, expected: ErrManifestCorrupt},
		{name: "absolute file", mutate: func(m *Manifest) { m.WritingSegment.Files = []string{"/data.seg"} }, expected: ErrManifestCorrupt},
		{name: "parent file", mutate: func(m *Manifest) { m.WritingSegment.Files = []string{"../data.seg"} }, expected: ErrManifestCorrupt},
		{name: "unclean file", mutate: func(m *Manifest) { m.WritingSegment.Files = []string{"segment/../data.seg"} }, expected: ErrManifestCorrupt},
		{name: "windows separator", mutate: func(m *Manifest) { m.WritingSegment.Files = []string{`segment\data.seg`} }, expected: ErrManifestCorrupt},
		{name: "duplicate file", mutate: func(m *Manifest) { m.WritingSegment.Files = []string{"data.seg", "data.seg"} }, expected: ErrManifestCorrupt},
		{name: "missing indexed segment", mutate: func(m *Manifest) { m.SegmentIndexSnapshots[0].SegmentID = 99 }, expected: ErrManifestCorrupt},
		{name: "duplicate segment index", mutate: func(m *Manifest) {
			m.SegmentIndexSnapshots = append(m.SegmentIndexSnapshots, m.SegmentIndexSnapshots[0])
		}, expected: ErrManifestCorrupt},
		{name: "segment index count mismatch", mutate: func(m *Manifest) { m.SegmentIndexSnapshots[0].DocumentCount++ }, expected: ErrManifestCorrupt},
		{name: "segment index min mismatch", mutate: func(m *Manifest) { m.SegmentIndexSnapshots[0].MinDocumentID++ }, expected: ErrManifestCorrupt},
		{name: "segment index max mismatch", mutate: func(m *Manifest) { m.SegmentIndexSnapshots[0].MaxDocumentID++ }, expected: ErrManifestCorrupt},
		{name: "short segment index hash", mutate: func(m *Manifest) { m.SegmentIndexSnapshots[0].SchemaSHA256 = "00" }, expected: ErrManifestCorrupt},
		{name: "duplicate segment artifact", mutate: func(m *Manifest) {
			m.SegmentIndexSnapshots[0].Artifacts = append(m.SegmentIndexSnapshots[0].Artifacts, m.SegmentIndexSnapshots[0].Artifacts[0])
		}, expected: ErrManifestCorrupt},
		{name: "legacy FTS artifact", mutate: func(m *Manifest) {
			m.SegmentIndexSnapshots[0].Artifacts[0].File = "indexes/legacy.zvi"
		}, expected: ErrManifestCorrupt},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			manifest := sampleManifest(1)
			testCase.mutate(&manifest)
			{
				err := manifest.Validate()
				require.ErrorIs(t, err, testCase.expected)
			}
		})
	}

	fullRange := Manifest{
		FormatVersion:       DiskFormatVersion,
		Generation:          1,
		Schema:              json.RawMessage(`{}`),
		SegmentMaxDocuments: 1,
		PersistedSegments: []SegmentMetadata{{
			ID: 1, MinDocID: 0, MaxDocID: math.MaxUint64, DocCount: math.MaxUint64,
		}},
		NextSegmentID: 2,
	}
	{
		err := fullRange.Validate()
		require.NoError(t, err)
	}
}

func TestManifestDetectsCorruptionAndTruncation(t *testing.T) {
	t.Parallel()

	encoded, err := MarshalManifest(sampleManifest(7))
	require.NoError(t, err)

	for length := 0; length < len(encoded); length++ {
		{
			_, err := UnmarshalManifest(encoded[:length])
			require.Error(t, err)
		}
	}

	tests := []struct {
		name     string
		mutate   func([]byte) []byte
		expected error
	}{
		{name: "magic", mutate: func(data []byte) []byte { data[0] ^= 0xff; return data }, expected: ErrManifestCorrupt},
		{name: "legacy format", mutate: func(data []byte) []byte { binary.LittleEndian.PutUint16(data[8:10], 1); return data }, expected: ErrUnsupportedFormatVersion},
		{name: "future format", mutate: func(data []byte) []byte { binary.LittleEndian.PutUint16(data[8:10], 3); return data }, expected: ErrUnsupportedFormatVersion},
		{name: "header size", mutate: func(data []byte) []byte { binary.LittleEndian.PutUint16(data[10:12], 31); return data }, expected: ErrManifestCorrupt},
		{name: "generation", mutate: func(data []byte) []byte { binary.LittleEndian.PutUint64(data[12:20], 8); return data }, expected: ErrManifestCorrupt},
		{name: "huge length", mutate: func(data []byte) []byte { binary.LittleEndian.PutUint64(data[20:28], math.MaxUint64); return data }, expected: ErrManifestCorrupt},
		{name: "checksum", mutate: func(data []byte) []byte { data[28] ^= 1; return data }, expected: ErrManifestCorrupt},
		{name: "payload", mutate: func(data []byte) []byte { data[len(data)-1] ^= 1; return data }, expected: ErrManifestCorrupt},
		{name: "trailing", mutate: func(data []byte) []byte { return append(data, 0) }, expected: ErrManifestCorrupt},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			corrupted := testCase.mutate(append([]byte(nil), encoded...))
			{
				_, err := UnmarshalManifest(corrupted)
				require.ErrorIs(t, err, testCase.expected)
			}
		})
	}
}

func TestManifestRejectsUnknownPayloadField(t *testing.T) {
	t.Parallel()

	encoded, err := manifestWithPayloadField("future_field", true)
	require.NoError(t, err)
	_, err = UnmarshalManifest(encoded)
	require.ErrorIs(t, err, ErrManifestCorrupt)
}

func TestManifestRejectsCollectionWideIndexSnapshot(t *testing.T) {
	t.Parallel()

	encoded, err := manifestWithPayloadField("index_snapshot", map[string]any{
		"schema_sha256":   strings.Repeat("a", 64),
		"document_count":  8,
		"max_document_id": 19,
		"artifacts": []any{map[string]any{
			"field": "embedding", "kind": "vector-2", "file": "indexes/embedding.zvi",
		}},
	})
	require.NoError(t, err)
	_, err = UnmarshalManifest(encoded)
	require.ErrorIs(t, err, ErrManifestCorrupt)
}

func TestManifestRequiresWritingSegmentStartDocumentID(t *testing.T) {
	t.Parallel()

	encoded, err := manifestWithoutPayloadField("writing_segment_start_doc_id")
	require.NoError(t, err)
	_, err = UnmarshalManifest(encoded)
	require.ErrorIs(t, err, ErrManifestCorrupt)
}

func TestManifestRejectsNullWritingSegmentStartDocumentID(t *testing.T) {
	t.Parallel()

	encoded, err := manifestWithPayloadField("writing_segment_start_doc_id", nil)
	require.NoError(t, err)
	_, err = UnmarshalManifest(encoded)
	require.ErrorIs(t, err, ErrManifestCorrupt)
}

func TestManifestAcceptsZeroWritingSegmentStartDocumentID(t *testing.T) {
	t.Parallel()

	manifest := sampleManifest(1)
	manifest.PersistedSegments = nil
	manifest.SegmentIndexSnapshots = nil
	manifest.WritingSegment.MinDocID = 0
	manifest.WritingSegment.MaxDocID = 0
	manifest.WritingSegment.DocCount = 0
	manifest.WritingSegmentStartDocID = 0
	encoded, err := MarshalManifest(manifest)
	require.NoError(t, err)
	decoded, err := UnmarshalManifest(encoded)
	require.NoError(t, err)
	require.Zero(t, decoded.WritingSegmentStartDocID)
}

func TestLifecycleManifestRejectsWritingSegmentBeforePersistedDocuments(t *testing.T) {
	t.Parallel()

	manifest := sampleManifest(1)
	manifest.WritingSegment.MinDocID = 0
	manifest.WritingSegment.MaxDocID = 0
	manifest.WritingSegment.DocCount = 0
	manifest.WritingSegmentStartDocID = 0
	err := validateLifecycleManifest(manifest)
	require.ErrorIs(t, err, ErrCollectionCorrupt)
}

func manifestWithPayloadField(name string, value any) ([]byte, error) {
	encoded, err := MarshalManifest(sampleManifest(1))
	if err != nil {
		return nil, err
	}
	var payload map[string]any
	if err := json.Unmarshal(encoded[manifestHeaderSize:], &payload); err != nil {
		return nil, err
	}
	payload[name] = value
	newPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	encoded = append(encoded[:manifestHeaderSize], newPayload...)
	binary.LittleEndian.PutUint64(encoded[20:28], uint64(len(newPayload)))
	binary.LittleEndian.PutUint32(encoded[28:32], ailego.CRC32C(newPayload))
	return encoded, nil
}

func manifestWithoutPayloadField(name string) ([]byte, error) {
	encoded, err := MarshalManifest(sampleManifest(1))
	if err != nil {
		return nil, err
	}
	var payload map[string]any
	if err := json.Unmarshal(encoded[manifestHeaderSize:], &payload); err != nil {
		return nil, err
	}
	delete(payload, name)
	newPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	encoded = append(encoded[:manifestHeaderSize], newPayload...)
	binary.LittleEndian.PutUint64(encoded[20:28], uint64(len(newPayload)))
	binary.LittleEndian.PutUint32(encoded[28:32], ailego.CRC32C(newPayload))
	return encoded, nil
}

func FuzzUnmarshalManifest(f *testing.F) {
	encoded, err := MarshalManifest(sampleManifest(1))
	require.NoError(f, err)

	f.Add(encoded)
	f.Add(encoded[:manifestHeaderSize])
	f.Add([]byte("not a manifest"))
	f.Fuzz(func(t *testing.T, data []byte) {
		manifest, err := UnmarshalManifest(data)
		if err == nil {
			{
				err := manifest.Validate()
				require.NoError(t, err)
			}
		}
	})
}

func sampleManifest(generation uint64) Manifest {
	return Manifest{
		FormatVersion:       DiskFormatVersion,
		Generation:          generation,
		Schema:              json.RawMessage(`{"name":"books","version":1}`),
		EnableMmap:          true,
		SegmentMaxDocuments: 100,
		PersistedSegments: []SegmentMetadata{{
			ID: 3, MinDocID: 10, MaxDocID: 19, DocCount: 8,
			Files: []string{"segments/3/data.seg", "segments/3/delete.snapshot"},
		}},
		WritingSegment: &SegmentMetadata{
			ID: 4, MinDocID: 20, MaxDocID: 21, DocCount: 2,
			Files: []string{"segments/4/data.wal"},
		},
		WritingSegmentStartDocID: 20,
		IDMapGeneration:          5,
		DeleteSnapshotGeneration: 6,
		NextSegmentID:            5,
		SegmentIndexSnapshots: []SegmentIndexSnapshotMetadata{{
			SegmentID: 3, SchemaSHA256: strings.Repeat("b", 64), DocumentCount: 8,
			MinDocumentID: 10, MaxDocumentID: 19,
			Artifacts: []IndexArtifactMetadata{{Field: "text", Kind: "fts", File: "indexes/segment-3-text.pebble"}},
		}},
	}
}
