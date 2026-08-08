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
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

type ftsPostingFixture struct {
	BaselineCommit      string `json:"baseline_commit"`
	PostingHeaderSHA256 string `json:"posting_header_sha256"`
	PostingSourceSHA256 string `json:"posting_source_sha256"`
	IndexerHeaderSHA256 string `json:"indexer_header_sha256"`
	IndexerSourceSHA256 string `json:"indexer_source_sha256"`
	ReducerHeaderSHA256 string `json:"reducer_header_sha256"`
	ReducerSourceSHA256 string `json:"reducer_source_sha256"`
	PhraseSourceSHA256  string `json:"phrase_source_sha256"`
	PositionDeltaHex    string `json:"position_delta_hex"`
	TotalDocuments      uint64 `json:"total_documents"`
	TotalTokens         uint64 `json:"total_tokens"`
	Documents           []struct {
		DocumentID uint32 `json:"document_id"`
		Tokens     []struct {
			Term     string `json:"term"`
			Position uint32 `json:"position"`
		} `json:"tokens"`
	} `json:"documents"`
	Terms []struct {
		Term                 string       `json:"term"`
		DocumentFrequency    uint32       `json:"document_frequency"`
		MaximumTermFrequency uint32       `json:"maximum_term_frequency"`
		Postings             []FTSPosting `json:"postings"`
	} `json:"terms"`
}

func loadFTSPostingFixture(t testing.TB) ftsPostingFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/fts_posting_58375ff.json")
	require.NoError(t, err)

	var fixture ftsPostingFixture
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}

	return fixture
}

func TestFTSTermDictionaryBaselineFixture(t *testing.T) {
	fixture := loadFTSPostingFixture(t)
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b")
	require.True(t, fixture.PostingHeaderSHA256 == "78b8e7e6af6ae7279eb7561fc4aec53b2f7811a209f3e32306f89eb9430f92b5")
	require.True(t, fixture.PostingSourceSHA256 == "ed99e2c626429926afc6633c1f4d3d6ae89ac32bb584860306b215302756180c")
	require.True(t, fixture.IndexerHeaderSHA256 == "83baa255dad8f86e18a9c49091bcbcf015d5c066d75e726332eb2dddf7a31056")
	require.True(t, fixture.IndexerSourceSHA256 == "dbe3bffb8fef7d15a4be09babd5c7d3706dcdde4343d1318a310ba24662d5cab")
	require.True(t, fixture.ReducerHeaderSHA256 == "b87f60888e230f39268dea6614d7aadca60f8945bc3bc0862f2fa026d1aeb43b")
	require.True(t, fixture.ReducerSourceSHA256 == "7bebfd9d410c598e2970c0d3b7331b9623e2b481b28466b2eae7543af7d58b2e")
	require.True(t, fixture.PhraseSourceSHA256 == "6316f87dab229ba02fbb588f0047f23a9b988aab584d0d87fb9987896dd565f7")

	positions, err := appendFTSPositionDeltas(context.Background(), nil, []uint32{0, 2, 130})
	require.NoError(t, err)
	{
		got := hex.EncodeToString(positions)
		require.Equal(t, fixture.PositionDeltaHex, got)
	}

	builder := NewFTSFieldBuilder()
	for _, document := range fixture.Documents {
		tokens := make([]Token, len(document.Tokens))
		for index, token := range document.Tokens {
			tokens[index] = Token{Text: token.Term, Position: token.Position}
		}
		{
			err := builder.AddDocument(context.Background(), document.DocumentID, tokens)
			require.NoError(t, err)
		}
	}
	dictionary, err := builder.Build(context.Background())
	require.NoError(t, err)
	{
		got, want := dictionary.Stats(), (FTSSegmentStats{TotalDocuments: fixture.TotalDocuments, TotalTokens: fixture.TotalTokens})
		require.Equal(t, want, got)
	}
	require.Equal(t, float64(5)/3, dictionary.Stats().AverageDocumentLength())
	require.Len(t, fixture.Terms, dictionary.TermCount())

	for _, term := range fixture.Terms {
		info, postingList, found := dictionary.Lookup(term.Term)
		require.True(t, found)
		require.Equal(t, FTSTermInfo{Term: term.Term, DocumentFrequency: term.DocumentFrequency, MaximumTermFrequency: term.MaximumTermFrequency}, info)
		{
			got := collectFTSPostings(postingList.Iterator())
			require.Equal(t, term.Postings, got)
		}
	}
	{
		_, _, found := dictionary.Lookup("missing")
		require.False(t, found,
			"missing term found")
	}
}

func TestFTSTermDictionaryPrefixAndSnapshot(t *testing.T) {
	builder := NewFTSFieldBuilder()
	{
		err := builder.AddDocument(context.Background(), 0, []Token{{Text: "banana", Position: 0}, {Text: "band", Position: 1}, {Text: "apple", Position: 2}})
		require.NoError(t, err)
	}

	first, err := builder.Build(context.Background())
	require.NoError(t, err)
	{
		got, want := first.Terms(), []string{"apple", "banana", "band"}
		require.Equal(t, want, got)
	}

	terms := first.Terms()
	terms[0] = "changed"
	require.True(t, first.Terms()[0] == "apple",
		"Terms aliases dictionary state")
	{
		got := first.Prefix("ban", 0)
		require.Equal(t, []FTSTermInfo{
			{Term: "banana", DocumentFrequency: 1, MaximumTermFrequency: 1},
			{Term: "band", DocumentFrequency: 1, MaximumTermFrequency: 1},
		}, got)
	}
	{
		got := first.Prefix("ban", 1)
		require.Len(t, got, 1)
		require.True(t, got[0].Term == "banana")
	}
	{
		got := first.Prefix("", -1)
		require.Len(t, got, 0)
	}
	{
		err := builder.AddDocument(context.Background(), 1, []Token{{Text: "apricot", Position: 0}})
		require.NoError(t, err)
	}

	second, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, first.Stats().TotalDocuments == 1)
	require.True(t, second.Stats().TotalDocuments == 2)
	require.True(t, first.TermCount() == 3)
	require.True(t, second.TermCount() == 4)
}

func TestFTSTermDictionaryAllEmptyDocuments(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{nil, nil})
	require.True(t, dictionary.TermCount() == 0)
	require.Equal(t, FTSSegmentStats{TotalDocuments: 2}, dictionary.Stats())
}

func TestFTSFieldBuilderInvalidInputAndCancellation(t *testing.T) {
	{
		err := (*FTSFieldBuilder)(nil).AddDocument(context.Background(), 0, nil)
		require.ErrorIs(t, err, ErrInvalidFTSDocument)
	}

	builder := NewFTSFieldBuilder()
	{
		err := builder.AddDocument(nil, 0, nil)
		require.ErrorIs(t, err, ErrInvalidFTSDocument)
	}
	{
		err := builder.AddDocument(context.Background(), 1, nil)
		require.ErrorIs(t, err, ErrInvalidFTSDocument)
	}
	{
		err := builder.AddDocument(context.Background(), 0, []Token{{Text: "a", Position: 2}, {Text: "b", Position: 1}})
		require.ErrorIs(t, err, ErrInvalidFTSDocument)
	}
	{
		err := builder.AddDocument(context.Background(), 0, nil)
		require.NoError(t, err)
	}
	{
		_, err := builder.Build(nil)
		require.ErrorIs(t, err, ErrInvalidFTSDictionary)
	}
	{
		_, err := (*FTSFieldBuilder)(nil).Build(context.Background())
		require.ErrorIs(t, err, ErrInvalidFTSDictionary)
	}

	tokens := make([]Token, 20000)
	for index := range tokens {
		tokens[index] = Token{Text: fmt.Sprintf("term-%05d", index), Position: uint32(index)}
	}
	midAdd := newCancelAfterChecks(3)
	{
		err := NewFTSFieldBuilder().AddDocument(midAdd, 0, tokens)
		require.ErrorIs(t, err, context.Canceled)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := NewFTSFieldBuilder().AddDocument(canceled, 0, nil)
		require.ErrorIs(t, err, context.Canceled)
	}

	buildBuilder := NewFTSFieldBuilder()
	{
		err := buildBuilder.AddDocument(context.Background(), 0, []Token{{Text: "alpha", Position: 0}})
		require.NoError(t, err)
	}
	{
		_, err := buildBuilder.Build(newCancelAfterChecks(4))
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestFTSTermDictionarySaveOpenCancellation(t *testing.T) {
	tokens := make([]Token, 4200)
	for index := range tokens {
		tokens[index] = Token{Text: fmt.Sprintf("term-%05d", index), Position: uint32(index)}
	}
	dictionary := buildFTSTestDictionary(t, [][]Token{tokens})
	{
		err := dictionary.Save(newCancelAfterChecks(3), filepath.Join(t.TempDir(), "canceled.pebble"))
		require.ErrorIs(t, err, context.Canceled)
	}

	path := filepath.Join(t.TempDir(), "dictionary.pebble")
	require.NoError(t, dictionary.Save(context.Background(), path))
	{
		_, err := OpenFTSTermDictionary(newCancelAfterChecks(3), path)
		require.ErrorIs(t, err, context.Canceled)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := OpenFTSTermDictionary(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}
}

func BenchmarkFTSTermDictionaryBuild(b *testing.B) {
	documents := make([][]Token, 1000)
	for documentID := range documents {
		documents[documentID] = make([]Token, 20)
		for position := range documents[documentID] {
			documents[documentID][position] = Token{Text: fmt.Sprintf("term-%03d", (documentID+position)%500), Position: uint32(position)}
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		builder := NewFTSFieldBuilder()
		for documentID, tokens := range documents {
			{
				err := builder.AddDocument(context.Background(), uint32(documentID), tokens)
				if err != nil {
					require.NoError(b, err)
				}
			}
		}
		{
			_, err := builder.Build(context.Background())
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

func buildFTSTestDictionary(t testing.TB, documents [][]Token) *FTSTermDictionary {
	t.Helper()
	builder := NewFTSFieldBuilder()
	for documentID, tokens := range documents {
		{
			err := builder.AddDocument(context.Background(), uint32(documentID), tokens)
			require.NoError(t, err)
		}
	}
	dictionary, err := builder.Build(context.Background())
	require.NoError(t, err)

	return dictionary
}

func TestFTSSegmentStatsAverageEmpty(t *testing.T) {
	{
		got := (FTSSegmentStats{}).AverageDocumentLength()
		require.True(t, got == 1)
	}
	{
		got := (FTSCorpusStats{}).AverageDocumentLength()
		require.True(t, got == 1)
	}
	{
		got := (FTSCorpusStats{}).Terms()
		require.Len(t, got, 0)
	}
	{
		got := (FTSCorpusStats{}).DocumentFrequency("x")
		require.True(t, got == 0)
	}
}

func TestAggregateFTSCorpusStats(t *testing.T) {
	segment0 := buildFTSTestDictionary(t, [][]Token{
		{{Text: "alpha", Position: 0}, {Text: "beta", Position: 1}},
		nil,
		{{Text: "alpha", Position: 0}, {Text: "alpha", Position: 1}, {Text: "only-deleted", Position: 2}},
	})
	segment1 := buildFTSTestDictionary(t, [][]Token{
		{{Text: "alpha", Position: 0}, {Text: "gamma", Position: 1}},
		{{Text: "beta", Position: 0}, {Text: "gamma", Position: 1}, {Text: "gamma", Position: 2}},
	})
	deleted0 := ailego.NewBitmap(3)
	deleted0.Set(2)
	deleted1 := ailego.NewBitmap(2)
	deleted1.Set(0)
	stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{
		{Dictionary: segment0, DeletedDocuments: deleted0},
		{Dictionary: segment1, DeletedDocuments: deleted1},
	})
	require.NoError(t, err)
	require.True(t, stats.TotalDocuments == 3)
	require.True(t, stats.TotalTokens == 5)
	require.Equal(t, float64(5)/3, stats.AverageDocumentLength())

	want := map[string]uint64{"alpha": 1, "beta": 2, "gamma": 1}
	{
		got := stats.DocumentFrequencies()
		require.Equal(t, want, got)
	}
	{
		got, wantTerms := stats.Terms(), []string{"alpha", "beta", "gamma"}
		require.Equal(t, wantTerms, got)
	}

	copy := stats.DocumentFrequencies()
	copy["alpha"] = 99
	require.True(t, stats.DocumentFrequency("alpha") == 1,
		"frequency map aliases stats")
}

func TestAggregateFTSCorpusStatsValidationAndCancellation(t *testing.T) {
	{
		_, err := AggregateFTSCorpusStats(nil, nil)
		require.ErrorIs(t, err, ErrInvalidFTSStats)
	}
	{
		_, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{}})
		require.ErrorIs(t, err, ErrInvalidFTSStats)
	}

	dictionary := buildFTSTestDictionary(t, [][]Token{{{Text: "alpha", Position: 0}}})
	deleted := ailego.NewBitmap(65)
	deleted.Set(64)
	{
		_, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary, DeletedDocuments: deleted}})
		require.ErrorIs(t, err, ErrInvalidFTSStats)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := AggregateFTSCorpusStats(canceled, []FTSSegmentView{{Dictionary: dictionary}})
		require.ErrorIs(t, err, context.Canceled)
	}

	largeDocuments := make([][]Token, 5000)
	for index := range largeDocuments {
		largeDocuments[index] = []Token{{Text: "alpha", Position: 0}}
	}
	large := buildFTSTestDictionary(t, largeDocuments)
	midway := newCancelAfterChecks(3)
	{
		_, err := AggregateFTSCorpusStats(midway, []FTSSegmentView{{Dictionary: large}})
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestFTSTermDictionaryLongSharedPrefix(t *testing.T) {
	prefix := strings.Repeat("x", 10000)
	dictionary := buildFTSTestDictionary(t, [][]Token{{
		{Text: prefix + "a", Position: 0},
		{Text: prefix + "b", Position: 1},
	}})
	path := filepath.Join(t.TempDir(), "long-prefix.pebble")
	require.NoError(t, dictionary.Save(context.Background(), path))
	reopened, err := OpenFTSTermDictionary(context.Background(), path)
	require.NoError(t, err)
	require.Equal(t, dictionary.Terms(), reopened.Terms())
}
