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
	"errors"
	"slices"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMergeFTSTermDictionariesDenseRemapPositions(t *testing.T) {
	segment0 := buildFTSTestDictionary(t, [][]Token{
		{{Text: "apple", Position: 0}, {Text: "banana", Position: 1}},
		nil,
		{{Text: "apple", Position: 0}, {Text: "apple", Position: 1}},
	})
	segment1 := buildFTSTestDictionary(t, [][]Token{
		{{Text: "banana", Position: 0}, {Text: "carrot", Position: 1}},
		{{Text: "apple", Position: 0}, {Text: "banana", Position: 1}},
	})
	deleted0 := ailego.NewBitmap(3)
	deleted0.Set(1)
	deleted1 := ailego.NewBitmap(2)
	deleted1.Set(0)
	merged, err := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{
		{Dictionary: segment0, DeletedDocuments: deleted0},
		{Dictionary: segment1, DeletedDocuments: deleted1},
	})
	require.NoError(t, err)
	{
		got, want := merged.Stats(), (FTSSegmentStats{TotalDocuments: 3, TotalTokens: 6})
		require.Equal(t, want, got)
	}
	{
		got, want := merged.Terms(), []string{"apple", "banana"}
		require.Equal(t, want, got)
	}

	for documentID := uint32(0); documentID < 3; documentID++ {
		{
			length, ok := merged.DocumentLength(documentID)
			require.True(t, ok)
			require.True(t, length == 2)
		}
	}
	wantPostings := map[string][]FTSPosting{
		"apple": {
			{DocumentID: 0, TermFrequency: 1, DocumentLength: 2, Positions: []uint32{0}},
			{DocumentID: 1, TermFrequency: 2, DocumentLength: 2, Positions: []uint32{0, 1}},
			{DocumentID: 2, TermFrequency: 1, DocumentLength: 2, Positions: []uint32{0}},
		},
		"banana": {
			{DocumentID: 0, TermFrequency: 1, DocumentLength: 2, Positions: []uint32{1}},
			{DocumentID: 2, TermFrequency: 1, DocumentLength: 2, Positions: []uint32{1}},
		},
	}
	for term, want := range wantPostings {
		info, postings, found := merged.Lookup(term)
		require.True(t, found)
		require.Equal(t, uint32(len(want)), info.DocumentFrequency)
		require.Equal(t, want, collectFTSPostings(postings.Iterator()))
	}
	{
		info, _, found := merged.Lookup("carrot")
		require.False(t, found)
		require.Equal(t, FTSTermInfo{}, info)
	}

	appleInfo, _, _ := merged.Lookup("apple")
	require.True(t, appleInfo.MaximumTermFrequency == 2)

	pipeline := newFTSStandardTestPipeline(t)
	node, err := ParseFTSQuery(context.Background(), `"apple banana"`, pipeline, FTSDefaultOperatorOR)
	require.NoError(t, err)

	iterator, err := NewFTSQueryIterator(context.Background(), merged, node, FTSQueryExecutionOptions{})
	require.NoError(t, err)
	{
		got, want := collectFTSQueryDocuments(t, iterator), []uint32{0, 2}
		require.Equal(t, want, got)
	}

	deleted0.Clear(1)
	deleted1.Clear(0)
	require.Equal(t, []string{"apple", "banana"}, merged.Terms())
}

func TestMergeFTSTermDictionariesEmptyAllDeletedAndMaximumTF(t *testing.T) {
	empty, err := MergeFTSTermDictionaries(context.Background(), nil)
	require.NoError(t, err)
	require.Equal(t, FTSSegmentStats{}, empty.Stats())
	require.True(t, empty.TermCount() == 0)

	source := buildFTSTestDictionary(t, [][]Token{
		{{Text: "x", Position: 0}, {Text: "x", Position: 1}, {Text: "x", Position: 2}},
		{{Text: "x", Position: 0}},
	})
	deleteHighest := ailego.NewBitmap(2)
	deleteHighest.Set(0)
	merged, err := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{{Dictionary: source, DeletedDocuments: deleteHighest}})
	require.NoError(t, err)

	info, postings, found := merged.Lookup("x")
	require.True(t, found)
	require.True(t, info.MaximumTermFrequency == 1)
	require.Equal(t, []FTSPosting{
		{DocumentID: 0, TermFrequency: 1, DocumentLength: 1, Positions: []uint32{0}},
	}, collectFTSPostings(postings.Iterator()))

	deleteAll := ailego.NewBitmap(2)
	deleteAll.Set(0)
	deleteAll.Set(1)
	allDeleted, err := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{{Dictionary: source, DeletedDocuments: deleteAll}})
	require.NoError(t, err)
	require.Equal(t, FTSSegmentStats{}, allDeleted.Stats())
	require.True(t, allDeleted.TermCount() == 0)
}

func TestMergeFTSTermDictionariesValidationCancellationAndConcurrency(t *testing.T) {
	source := buildFTSTestDictionary(t, [][]Token{{{Text: "x", Position: 0}}})
	{
		merged, err := MergeFTSTermDictionaries(nil, nil)
		require.Nil(t, merged)
		require.ErrorIs(t, err, ErrInvalidFTSMerge)
	}
	{
		merged, err := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{{}})
		require.Nil(t, merged)
		require.ErrorIs(t, err, ErrInvalidFTSMerge)
	}

	inconsistent := buildFTSTestDictionary(t, [][]Token{{{Text: "x", Position: 0}}})
	inconsistent.stats.TotalTokens++
	{
		merged, err := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{{Dictionary: inconsistent}})
		require.Nil(t, merged)
		require.ErrorIs(t, err, ErrInvalidFTSMerge)
	}

	outside := ailego.NewBitmap(1)
	outside.Set(1)
	{
		merged, err := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{{Dictionary: source, DeletedDocuments: outside}})
		require.Nil(t, merged)
		require.ErrorIs(t, err, ErrInvalidFTSMerge)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		merged, err := MergeFTSTermDictionaries(canceled, []FTSSegmentView{{Dictionary: source}})
		require.Nil(t, merged)
		require.ErrorIs(t, err, context.Canceled)
	}

	largeDocuments := make([][]Token, 5000)
	for index := range largeDocuments {
		largeDocuments[index] = []Token{{Text: "x", Position: 0}}
	}
	large := buildFTSTestDictionary(t, largeDocuments)
	{
		merged, err := MergeFTSTermDictionaries(newCancelAfterChecks(3), []FTSSegmentView{{Dictionary: large}})
		require.Nil(t, merged)
		require.ErrorIs(t, err, context.Canceled)
	}

	var wait sync.WaitGroup
	errorsChannel := make(chan error, 12)
	for range 12 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			merged, mergeErr := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{{Dictionary: source}})
			if mergeErr != nil {
				errorsChannel <- mergeErr
				return
			}
			if info, _, found := merged.Lookup("x"); !found || info.DocumentFrequency != 1 {
				errorsChannel <- errors.New("unexpected concurrent merge")
			}
		}()
	}
	wait.Wait()
	close(errorsChannel)
	for err := range errorsChannel {
		assert.NoError(t, err)
	}
}

func FuzzMergeFTSTermDictionaries(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3, 4, 5}, uint64(0b001001))
	f.Add([]byte{}, uint64(0))
	f.Fuzz(func(t *testing.T, data []byte, deletionMask uint64) {
		if len(data) > 32 {
			data = data[:32]
		}
		documents := make([][]Token, len(data))
		for documentID, value := range data {
			tokenCount := int(value % 4)
			documents[documentID] = make([]Token, tokenCount)
			for position := range tokenCount {
				documents[documentID][position] = Token{
					Text: string(rune('a' + (int(value)+position)%4)), Position: uint32(position),
				}
			}
		}
		split := len(documents) / 2
		parts := [][][]Token{documents[:split], documents[split:]}
		views := make([]FTSSegmentView, len(parts))
		survivors := make([][]Token, 0, len(documents))
		global := 0
		for partIndex, part := range parts {
			views[partIndex].Dictionary = buildFTSTestDictionary(t, part)
			deleted := ailego.NewBitmap(uint64(len(part)))
			for local := range part {
				isDeleted := global < 64 && deletionMask&(uint64(1)<<global) != 0
				if isDeleted {
					deleted.Set(uint64(local))
				} else {
					survivors = append(survivors, part[local])
				}
				global++
			}
			views[partIndex].DeletedDocuments = deleted
		}
		merged, err := MergeFTSTermDictionaries(context.Background(), views)
		require.NoError(t, err)

		want := buildFTSTestDictionary(t, survivors)
		assertFTSDictionariesEqual(t, merged, want)
	})
}

func BenchmarkMergeFTSTermDictionaries(b *testing.B) {
	documents := make([][]Token, 5000)
	for documentID := range documents {
		documents[documentID] = []Token{
			{Text: "common", Position: 0},
			{Text: string(rune('a' + documentID%26)), Position: 1},
		}
	}
	left := buildFTSTestDictionary(b, documents[:2500])
	right := buildFTSTestDictionary(b, documents[2500:])
	views := []FTSSegmentView{{Dictionary: left}, {Dictionary: right}}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		{
			_, err := MergeFTSTermDictionaries(context.Background(), views)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

func assertFTSDictionariesEqual(t testing.TB, got, want *FTSTermDictionary) {
	t.Helper()
	if got == nil || want == nil {
		require.Equal(t, want, got)

		return
	}
	require.Equal(t, want.Stats(), got.Stats())
	require.True(t, slices.Equal(got.Terms(), want.Terms()))
	require.True(t, slices.Equal(got.documentLengths, want.documentLengths))

	for _, term := range want.Terms() {
		gotInfo, gotPostings, gotFound := got.Lookup(term)
		wantInfo, wantPostings, wantFound := want.Lookup(term)
		require.Equal(t, wantFound, gotFound)
		require.Equal(t, wantInfo, gotInfo)
		require.Equal(t, collectFTSPostings(wantPostings.Iterator()), collectFTSPostings(gotPostings.Iterator()))
	}
}
