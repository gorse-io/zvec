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

package core

import (
	"context"
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var ftsQueryTestDocuments = []string{
	"apple banana quick brown fox",
	"apple quick fox brown",
	"banana quick brown",
	"apple banana banana quick brown fox",
	"grape slow fox",
	"",
	"apple quick quick brown fox",
	"apple quick brown fox quick brown",
}

func TestFTSQueryIteratorTermPhraseAndBoolean(t *testing.T) {
	dictionary := buildFTSQueryTestDictionary(t)
	pipeline := newFTSStandardTestPipeline(t)
	tests := []struct {
		query string
		want  []uint32
	}{
		{"apple", []uint32{0, 1, 3, 6, 7}},
		{"apple OR grape", []uint32{0, 1, 3, 4, 6, 7}},
		{"apple AND banana", []uint32{0, 3}},
		{"apple NOT banana", []uint32{1, 6, 7}},
		{"apple -banana", []uint32{1, 6, 7}},
		{"+apple grape", []uint32{0, 1, 3, 6, 7}},
		{"-apple", []uint32{}},
		{`"quick brown"`, []uint32{0, 2, 3, 6, 7}},
		{`"quick fox"`, []uint32{1}},
		{`"banana banana"`, []uint32{3}},
		{`"quick quick brown"`, []uint32{6}},
		{`"quick brown" OR grape`, []uint32{0, 2, 3, 4, 6, 7}},
		{`"quick brown" AND apple`, []uint32{0, 3, 6, 7}},
		{`apple NOT "quick brown"`, []uint32{1}},
		{`"!!!"`, []uint32{}},
		{"(apple OR grape) AND fox", []uint32{0, 1, 3, 4, 6, 7}},
		{"apple OR missing", []uint32{0, 1, 3, 6, 7}},
		{"apple AND missing", []uint32{}},
		{"apple NOT missing", []uint32{0, 1, 3, 6, 7}},
		{"missing NOT apple", []uint32{}},
		{"apple apple", []uint32{0, 1, 3, 6, 7}},
	}
	for _, test := range tests {
		t.Run(test.query, func(t *testing.T) {
			node, err := ParseFTSQuery(context.Background(), test.query, pipeline, FTSDefaultOperatorOR)
			require.NoError(t, err)

			iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
			require.NoError(t, err)
			{
				got := collectFTSQueryDocuments(t, iterator)
				require.Equal(t, test.want, got)
			}
		})
	}
}

func TestFTSQueryIteratorAdvanceAndCost(t *testing.T) {
	dictionary := buildFTSQueryTestDictionary(t)
	pipeline := newFTSStandardTestPipeline(t)
	node, err := ParseFTSQuery(context.Background(), "apple OR grape", pipeline, FTSDefaultOperatorOR)
	require.NoError(t, err)

	iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
	require.NoError(t, err)
	require.True(t, iterator.Cost() == 6)
	require.False(t, iterator.Valid())
	require.True(t, iterator.DocumentID() == 0)
	require.True(t, iterator.Advance(context.Background(), 3))
	require.True(t, iterator.DocumentID() == 3)
	require.True(t, iterator.Advance(context.Background(), 3),
		"Advance did not retain current match")
	require.True(t, iterator.DocumentID() == 3,
		"Advance did not retain current match")
	require.True(t, iterator.Next(context.Background()))
	require.True(t, iterator.DocumentID() == 4)
	require.True(t, iterator.Advance(context.Background(), 7))
	require.True(t, iterator.DocumentID() == 7)
	require.False(t, iterator.Next(context.Background()))
	require.False(t, iterator.Valid())
	require.NoError(t, iterator.Err())
	require.False(t, iterator.Advance(context.Background(), 0),
		"exhausted iterator restarted")
}

func TestFTSQueryIteratorDeletionSnapshot(t *testing.T) {
	dictionary := buildFTSQueryTestDictionary(t)
	pipeline := newFTSStandardTestPipeline(t)
	node, err := ParseFTSQuery(context.Background(), `"quick brown"`, pipeline, FTSDefaultOperatorOR)
	require.NoError(t, err)

	deleted := ailego.NewBitmap(uint64(len(ftsQueryTestDocuments)))
	deleted.Set(0)
	deleted.Set(6)
	iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{DeletedDocuments: deleted})
	require.NoError(t, err)

	deleted.Clear(0)
	deleted.Set(2)
	{
		got, want := collectFTSQueryDocuments(t, iterator), []uint32{2, 3, 7}
		require.Equal(t, want, got)
	}

	invalid := ailego.NewBitmap(0)
	invalid.Set(1 << 26)
	{
		runtime.GC()
		var before runtime.MemStats
		runtime.ReadMemStats(&before)
		got, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{DeletedDocuments: invalid})
		var after runtime.MemStats
		runtime.ReadMemStats(&after)
		require.Nil(t, got)
		require.ErrorIs(t, err, ErrInvalidFTSQueryExecution)
		require.Less(t, after.TotalAlloc-before.TotalAlloc, uint64(1<<20),
			"out-of-domain deletion validation allocated a dense bitmap")
	}
}

func TestFTSQueryIteratorInvalidAndCancellation(t *testing.T) {
	dictionary := buildFTSQueryTestDictionary(t)
	pipeline := newFTSStandardTestPipeline(t)
	node, err := ParseFTSQuery(context.Background(), "apple", pipeline, FTSDefaultOperatorOR)
	require.NoError(t, err)
	{
		iterator, err := NewFTSQueryIterator(nil, dictionary, node, FTSQueryExecutionOptions{})
		require.Nil(t, iterator)
		require.ErrorIs(t, err, ErrInvalidFTSQueryExecution)
	}
	{
		iterator, err := NewFTSQueryIterator(context.Background(), nil, node, FTSQueryExecutionOptions{})
		require.Nil(t, iterator)
		require.ErrorIs(t, err, ErrInvalidFTSQueryExecution)
	}
	{
		iterator, err := NewFTSQueryIterator(context.Background(), dictionary, nil, FTSQueryExecutionOptions{})
		require.Nil(t, iterator)
		require.ErrorIs(t, err, ErrInvalidFTSQueryAST)
	}
	require.False(t, (*FTSQueryIterator)(nil).Next(context.Background()),
		"nil iterator behavior differs")
	require.False(t, (*FTSQueryIterator)(nil).Advance(context.Background(), 0),
		"nil iterator behavior differs")
	require.ErrorIs(t, (*FTSQueryIterator)(nil).Err(), ErrInvalidFTSQueryExecution,
		"nil iterator behavior differs")

	iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
	require.NoError(t, err)
	require.False(t, iterator.Next(nil))
	require.ErrorIs(t, iterator.Err(), ErrInvalidFTSQueryExecution)

	iterator, err = NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
	require.NoError(t, err)

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	require.False(t, iterator.Next(canceled))
	require.ErrorIs(t, iterator.Err(), context.Canceled)

	iterator, err = NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
	require.NoError(t, err)

	active, stop := context.WithCancel(context.Background())
	require.True(t, iterator.Next(active))

	stop()
	require.False(t, iterator.Next(active))
	require.ErrorIs(t, iterator.Err(), context.Canceled)
	require.False(t, iterator.Valid())
}

func TestFTSQueryIteratorASTAndDictionaryOwnership(t *testing.T) {
	dictionary := buildFTSQueryTestDictionary(t)
	node := &FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "apple"}
	iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
	require.NoError(t, err)

	node.Term = "missing"
	{
		got, want := collectFTSQueryDocuments(t, iterator), []uint32{0, 1, 3, 6, 7}
		require.Equal(t, want, got)
	}

	path := filepath.Join(t.TempDir(), "query.pebble")
	require.NoError(t, dictionary.Save(context.Background(), path))
	reopened, err := OpenFTSTermDictionary(context.Background(), path)
	require.NoError(t, err)

	query := &FTSPhraseQueryNode{Flags: defaultFTSQueryModifier(), Terms: []string{"banana", "banana"}}
	iterator, err = NewFTSQueryIterator(context.Background(), reopened, query, FTSQueryExecutionOptions{})
	require.NoError(t, err)

	{
		got, want := collectFTSQueryDocuments(t, iterator), []uint32{3}
		require.Equal(t, want, got)
	}
}

func TestFTSQueryIteratorConcurrentReaders(t *testing.T) {
	dictionary := buildFTSQueryTestDictionary(t)
	pipeline := newFTSStandardTestPipeline(t)
	node, err := ParseFTSQuery(context.Background(), `(apple OR grape) AND fox NOT banana`, pipeline, FTSDefaultOperatorOR)
	require.NoError(t, err)

	want := []uint32{1, 4, 6, 7}
	var wait sync.WaitGroup
	errorsChannel := make(chan error, 32)
	for worker := 0; worker < 32; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 50; iteration++ {
				iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
				if err != nil {
					errorsChannel <- err
					return
				}
				got := collectFTSQueryDocumentsError(iterator)
				if got.err != nil || !assert.Equal(t, want, got.documents) {
					errorsChannel <- fmt.Errorf("documents %#v: %v", got.documents, got.err)
					return
				}
			}
		}()
	}
	wait.Wait()
	close(errorsChannel)
	for err := range errorsChannel {
		require.NoError(t, err)
	}
}

func FuzzFTSQueryIterator(f *testing.F) {
	for _, seed := range []string{"apple", "apple OR grape", `"quick brown"`, "apple NOT banana", "+apple grape"} {
		f.Add(seed)
	}
	dictionary := buildFTSQueryTestDictionary(f)
	pipeline := newFTSStandardTestPipeline(f)
	f.Fuzz(func(t *testing.T, query string) {
		node, err := ParseFTSQuery(context.Background(), query, pipeline, FTSDefaultOperatorOR)
		if err != nil {
			return
		}
		iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
		require.NoError(t, err)

		var previous uint32
		first := true
		for iterator.Next(context.Background()) {
			documentID := iterator.DocumentID()
			require.True(t, int(documentID) < len(ftsQueryTestDocuments))
			require.False(t, !first && documentID <= previous)

			first, previous = false, documentID
		}
		{
			err := iterator.Err()
			require.NoError(t, err)
		}
	})
}

func BenchmarkFTSQueryIterator(b *testing.B) {
	documents := make([]string, 10_000)
	for index := range documents {
		documents[index] = fmt.Sprintf("common term-%03d phrase match", index%500)
	}
	dictionary := buildFTSQueryDictionaryFromDocuments(b, documents)
	pipeline := newFTSStandardTestPipeline(b)
	node, err := ParseFTSQuery(context.Background(), `common AND "phrase match"`, pipeline, FTSDefaultOperatorOR)
	if err != nil {
		require.NoError(b, err)
	}

	b.ReportAllocs()
	for b.Loop() {
		iterator, err := NewFTSQueryIterator(context.Background(), dictionary, node, FTSQueryExecutionOptions{})
		if err != nil {
			require.NoError(b, err)
		}

		for iterator.Next(context.Background()) {
		}
		{
			err := iterator.Err()
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

func buildFTSQueryTestDictionary(t testing.TB) *FTSTermDictionary {
	t.Helper()
	return buildFTSQueryDictionaryFromDocuments(t, ftsQueryTestDocuments)
}

func buildFTSQueryDictionaryFromDocuments(t testing.TB, documents []string) *FTSTermDictionary {
	t.Helper()
	builder := NewFTSFieldBuilder()
	for documentID, document := range documents {
		words := strings.Fields(document)
		tokens := make([]Token, len(words))
		for position, word := range words {
			tokens[position] = Token{Text: word, Position: uint32(position)}
		}
		{
			err := builder.AddDocument(context.Background(), uint32(documentID), tokens)
			require.NoError(t, err)
		}
	}
	dictionary, err := builder.Build(context.Background())
	require.NoError(t, err)

	return dictionary
}

func collectFTSQueryDocuments(t testing.TB, iterator *FTSQueryIterator) []uint32 {
	t.Helper()
	result := collectFTSQueryDocumentsError(iterator)
	require.NoError(t, result.err)

	return result.documents
}

type ftsQueryCollectionResult struct {
	documents []uint32
	err       error
}

func collectFTSQueryDocumentsError(iterator *FTSQueryIterator) ftsQueryCollectionResult {
	documents := make([]uint32, 0)
	for iterator.Next(context.Background()) {
		documents = append(documents, iterator.DocumentID())
	}
	return ftsQueryCollectionResult{documents: documents, err: iterator.Err()}
}
