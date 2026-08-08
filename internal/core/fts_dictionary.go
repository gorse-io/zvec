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
	"fmt"
	"math"
	"sort"
	"strings"
)

var (
	// ErrInvalidFTSDocument identifies a non-sequential document or malformed
	// token stream supplied to FTSFieldBuilder.
	ErrInvalidFTSDocument = errors.New("core: invalid FTS document")
	// ErrInvalidFTSDictionary identifies an invalid in-memory dictionary input.
	ErrInvalidFTSDictionary = errors.New("core: invalid FTS dictionary")
	// ErrCorruptFTSDictionary identifies a malformed Pebble dictionary.
	ErrCorruptFTSDictionary = errors.New("core: corrupt FTS dictionary")
)

// FTSSegmentStats is the exact document/token summary used to calculate
// average document length. Empty documents count toward TotalDocuments.
type FTSSegmentStats struct {
	TotalDocuments uint64
	TotalTokens    uint64
}

// AverageDocumentLength returns total tokens divided by total documents, or
// one for an empty segment, matching the baseline scorer convention.
func (s FTSSegmentStats) AverageDocumentLength() float64 {
	if s.TotalDocuments == 0 {
		return 1
	}
	return float64(s.TotalTokens) / float64(s.TotalDocuments)
}

// FTSTermInfo is immutable dictionary metadata for one byte-ordered term.
type FTSTermInfo struct {
	Term                 string
	DocumentFrequency    uint32
	MaximumTermFrequency uint32
}

// FTSTermDictionary is an immutable term dictionary. Its compressed posting
// lists are persisted in independently chunked Pebble values.
type FTSTermDictionary struct {
	terms           []string
	postings        []*FTSPostingList
	maximumTF       []uint32
	documentLengths []uint32
	stats           FTSSegmentStats
}

// FTSFieldBuilder builds one segment-local dictionary. It is intentionally
// single-writer; dictionaries returned by Build are safe for concurrent use.
type FTSFieldBuilder struct {
	documentLengths []uint32
	totalTokens     uint64
	postings        map[string][]FTSPosting
}

// NewFTSFieldBuilder creates an empty segment-local FTS builder.
func NewFTSFieldBuilder() *FTSFieldBuilder {
	return &FTSFieldBuilder{postings: make(map[string][]FTSPosting)}
}

// AddDocument indexes one already-analyzed token stream. Document IDs must be
// dense and start at zero so they share the segment's forward-row domain.
func (b *FTSFieldBuilder) AddDocument(ctx context.Context, documentID uint32, tokens []Token) error {
	if ctx == nil {
		return fmt.Errorf("%w: nil context", ErrInvalidFTSDocument)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if b == nil {
		return fmt.Errorf("%w: nil builder", ErrInvalidFTSDocument)
	}
	if uint64(len(b.documentLengths)) >= math.MaxUint32 || documentID != uint32(len(b.documentLengths)) {
		return fmt.Errorf("%w: document ID %d, want %d", ErrInvalidFTSDocument, documentID, len(b.documentLengths))
	}
	if uint64(len(tokens)) > math.MaxUint32 || math.MaxUint64-b.totalTokens < uint64(len(tokens)) {
		return fmt.Errorf("%w: document token count overflows statistics", ErrInvalidFTSDocument)
	}
	for index := 1; index < len(tokens); index++ {
		if tokens[index-1].Position > tokens[index].Position {
			return fmt.Errorf("%w: token positions decrease at %d", ErrInvalidFTSDocument, index)
		}
	}
	positionsByTerm := make(map[string][]uint32)
	for index, token := range tokens {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		positionsByTerm[token.Text] = append(positionsByTerm[token.Text], token.Position)
	}
	documentLength := uint32(len(tokens))
	for term, positions := range positionsByTerm {
		ownedTerm := term
		if _, found := b.postings[term]; !found {
			ownedTerm = strings.Clone(term)
		}
		ownedPositions := append([]uint32(nil), positions...)
		b.postings[ownedTerm] = append(b.postings[ownedTerm], FTSPosting{
			DocumentID:     documentID,
			TermFrequency:  uint32(len(positions)),
			DocumentLength: documentLength,
			Positions:      ownedPositions,
		})
	}
	b.documentLengths = append(b.documentLengths, documentLength)
	b.totalTokens += uint64(documentLength)
	return nil
}

// Build seals a point-in-time immutable dictionary without consuming the
// builder; later AddDocument calls cannot mutate the returned snapshot.
func (b *FTSFieldBuilder) Build(ctx context.Context) (*FTSTermDictionary, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrInvalidFTSDictionary)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if b == nil {
		return nil, fmt.Errorf("%w: nil builder", ErrInvalidFTSDictionary)
	}
	terms := make([]string, 0, len(b.postings))
	for term := range b.postings {
		terms = append(terms, term)
	}
	sort.Strings(terms)
	dictionary := &FTSTermDictionary{
		terms:           make([]string, len(terms)),
		postings:        make([]*FTSPostingList, len(terms)),
		maximumTF:       make([]uint32, len(terms)),
		documentLengths: append([]uint32(nil), b.documentLengths...),
		stats: FTSSegmentStats{
			TotalDocuments: uint64(len(b.documentLengths)),
			TotalTokens:    b.totalTokens,
		},
	}
	for index, term := range terms {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		postingList, err := BuildFTSPostingList(ctx, b.postings[term])
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, ctxErr
			}
			return nil, fmt.Errorf("%w: term %q: %v", ErrInvalidFTSDictionary, term, err)
		}
		var maximumTF uint32
		for _, posting := range b.postings[term] {
			maximumTF = max(maximumTF, posting.TermFrequency)
		}
		dictionary.terms[index] = strings.Clone(term)
		dictionary.postings[index] = postingList
		dictionary.maximumTF[index] = maximumTF
	}
	return dictionary, nil
}

// Stats returns the immutable segment summary.
func (d *FTSTermDictionary) Stats() FTSSegmentStats {
	if d == nil {
		return FTSSegmentStats{}
	}
	return d.stats
}

// DocumentLength returns a segment-local document's token count.
func (d *FTSTermDictionary) DocumentLength(documentID uint32) (uint32, bool) {
	if d == nil || uint64(documentID) >= uint64(len(d.documentLengths)) {
		return 0, false
	}
	return d.documentLengths[documentID], true
}

// TermCount returns the number of unique terms.
func (d *FTSTermDictionary) TermCount() int {
	if d == nil {
		return 0
	}
	return len(d.terms)
}

// Terms returns an owned byte-lexicographically sorted term slice.
func (d *FTSTermDictionary) Terms() []string {
	if d == nil {
		return []string{}
	}
	return append([]string(nil), d.terms...)
}

// Lookup returns immutable term metadata and its posting list.
func (d *FTSTermDictionary) Lookup(term string) (FTSTermInfo, *FTSPostingList, bool) {
	if d == nil {
		return FTSTermInfo{}, nil, false
	}
	index := sort.SearchStrings(d.terms, term)
	if index == len(d.terms) || d.terms[index] != term {
		return FTSTermInfo{}, nil, false
	}
	return d.termInfo(index), d.postings[index], true
}

// Prefix returns up to limit terms beginning with prefix in byte-lexical
// order. A zero limit means no limit.
func (d *FTSTermDictionary) Prefix(prefix string, limit int) []FTSTermInfo {
	if d == nil || limit < 0 {
		return []FTSTermInfo{}
	}
	start := sort.Search(len(d.terms), func(index int) bool { return d.terms[index] >= prefix })
	result := make([]FTSTermInfo, 0)
	for index := start; index < len(d.terms) && strings.HasPrefix(d.terms[index], prefix); index++ {
		if limit > 0 && len(result) >= limit {
			break
		}
		result = append(result, d.termInfo(index))
	}
	return result
}

func (d *FTSTermDictionary) termInfo(index int) FTSTermInfo {
	return FTSTermInfo{
		Term:                 d.terms[index],
		DocumentFrequency:    d.postings[index].DocumentFrequency(),
		MaximumTermFrequency: d.maximumTF[index],
	}
}
