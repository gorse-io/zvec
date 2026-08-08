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
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"

	"github.com/gorse-io/zvec/internal/indexstore"
)

const (
	ftsPebbleFormat     = "zvec-fts-pebble/1"
	ftsPebbleChunkBytes = 4 << 10
	ftsPebbleBatchBytes = 4 << 20
)

var (
	ftsFormatKey = []byte("m/format")
	ftsStatsKey  = []byte("m/stats")
)

// Save writes the dictionary to a new immutable Pebble directory.
func (d *FTSTermDictionary) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return fmt.Errorf("%w: nil context", ErrInvalidFTSDictionary)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateFTSDictionaryForSave(ctx, d); err != nil {
		return err
	}
	if err := requireEmptyFTSDirectory(path); err != nil {
		return err
	}
	store, err := indexstore.Open(path, indexstore.Options{})
	if err != nil {
		return fmt.Errorf("core: create FTS Pebble store: %w", err)
	}
	complete := false
	defer func() {
		if !complete {
			_ = store.Destroy()
		}
	}()
	writer := newFTSStoreWriter(store)
	fail := func(err error) error {
		_ = writer.close()
		return err
	}
	if err := writer.set(ftsFormatKey, []byte(ftsPebbleFormat)); err != nil {
		return fail(err)
	}
	stats := make([]byte, 16)
	binary.LittleEndian.PutUint64(stats[0:8], d.stats.TotalDocuments)
	binary.LittleEndian.PutUint64(stats[8:16], d.stats.TotalTokens)
	if err := writer.set(ftsStatsKey, stats); err != nil {
		return fail(err)
	}
	if err := writeFTSDocumentLengths(ctx, writer, d.documentLengths); err != nil {
		return fail(err)
	}
	for ordinal, term := range d.terms {
		if ordinal&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return fail(err)
			}
		}
		termKey := make([]byte, 5)
		termKey[0] = 't'
		binary.BigEndian.PutUint32(termKey[1:], uint32(ordinal))
		termValue := make([]byte, 4, 4+len(term))
		binary.LittleEndian.PutUint32(termValue, d.maximumTF[ordinal])
		termValue = append(termValue, term...)
		if err := writer.set(termKey, termValue); err != nil {
			return fail(err)
		}
		if err := writeFTSPostingChunks(ctx, writer, uint32(ordinal), d.postings[ordinal].data); err != nil {
			return fail(err)
		}
	}
	if err := writer.close(); err != nil {
		return err
	}
	if err := store.Flush(); err != nil {
		return fmt.Errorf("core: flush FTS Pebble store: %w", err)
	}
	if err := store.Close(); err != nil {
		return fmt.Errorf("core: close FTS Pebble store: %w", err)
	}
	complete = true
	return nil
}

// OpenFTSTermDictionary opens and fully validates a Pebble-backed dictionary.
func OpenFTSTermDictionary(ctx context.Context, path string) (*FTSTermDictionary, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrCorruptFTSDictionary)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	store, err := indexstore.Open(path, indexstore.Options{ReadOnly: true})
	if err != nil {
		return nil, ftsDictionaryCorruption("open Pebble store", err)
	}
	defer store.Close()
	format, err := store.Get(ftsFormatKey)
	if err != nil || string(format) != ftsPebbleFormat {
		return nil, ftsDictionaryCorruption("invalid format marker", err)
	}
	statsBytes, err := store.Get(ftsStatsKey)
	if err != nil || len(statsBytes) != 16 {
		return nil, ftsDictionaryCorruption("invalid statistics", err)
	}
	stats := FTSSegmentStats{
		TotalDocuments: binary.LittleEndian.Uint64(statsBytes[0:8]),
		TotalTokens:    binary.LittleEndian.Uint64(statsBytes[8:16]),
	}
	if stats.TotalDocuments > math.MaxUint32 {
		return nil, ftsDictionaryCorruption("document count exceeds uint32", nil)
	}
	documentLengths, err := readFTSDocumentLengths(ctx, store)
	if err != nil {
		return nil, err
	}
	if uint64(len(documentLengths)) != stats.TotalDocuments {
		return nil, ftsDictionaryCorruption("document count does not match length chunks", nil)
	}
	var computedTokens uint64
	for _, length := range documentLengths {
		if math.MaxUint64-computedTokens < uint64(length) {
			return nil, ftsDictionaryCorruption("total token count overflows", nil)
		}
		computedTokens += uint64(length)
	}
	if computedTokens != stats.TotalTokens {
		return nil, ftsDictionaryCorruption("total token count mismatch", nil)
	}
	dictionary := &FTSTermDictionary{documentLengths: documentLengths, stats: stats}
	terms, err := store.NewPrefixIterator([]byte{'t'})
	if err != nil {
		return nil, ftsDictionaryCorruption("open term iterator", err)
	}
	ordinal := uint32(0)
	previousTerm := ""
	for valid := terms.First(); valid; valid = terms.Next() {
		if ordinal&1023 == 0 {
			if err := ctx.Err(); err != nil {
				_ = terms.Close()
				return nil, err
			}
		}
		key, value := terms.Key(), terms.Value()
		if len(key) != 5 || key[0] != 't' || binary.BigEndian.Uint32(key[1:]) != ordinal || len(value) < 4 {
			_ = terms.Close()
			return nil, ftsDictionaryCorruption("invalid or non-contiguous term key", nil)
		}
		maximumTF := binary.LittleEndian.Uint32(value[:4])
		term := string(append([]byte(nil), value[4:]...))
		if maximumTF == 0 || ordinal > 0 && previousTerm >= term {
			_ = terms.Close()
			return nil, ftsDictionaryCorruption("invalid term metadata or ordering", nil)
		}
		postingData, postingErr := readFTSPostingChunks(ctx, store, ordinal)
		if postingErr != nil {
			_ = terms.Close()
			return nil, postingErr
		}
		postingList, postingErr := openFTSPostingList(ctx, postingData, false)
		if postingErr != nil {
			_ = terms.Close()
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, ctxErr
			}
			return nil, ftsDictionaryCorruption(fmt.Sprintf("invalid posting for term %q", term), postingErr)
		}
		if postingList.DocumentFrequency() == 0 {
			_ = terms.Close()
			return nil, ftsDictionaryCorruption("term has an empty posting", nil)
		}
		var computedMaximumTF uint32
		iterator := postingList.Iterator()
		for iterator.Next() {
			documentID := iterator.DocumentID()
			if uint64(documentID) >= uint64(len(documentLengths)) || iterator.DocumentLength() != documentLengths[documentID] {
				_ = terms.Close()
				return nil, ftsDictionaryCorruption("posting has inconsistent document metadata", nil)
			}
			computedMaximumTF = max(computedMaximumTF, iterator.TermFrequency())
		}
		if computedMaximumTF != maximumTF {
			_ = terms.Close()
			return nil, ftsDictionaryCorruption("maximum term frequency mismatch", nil)
		}
		dictionary.terms = append(dictionary.terms, term)
		dictionary.postings = append(dictionary.postings, postingList)
		dictionary.maximumTF = append(dictionary.maximumTF, maximumTF)
		previousTerm = term
		ordinal++
	}
	if err := errors.Join(terms.Error(), terms.Close()); err != nil {
		return nil, ftsDictionaryCorruption("iterate terms", err)
	}
	if err := validateFTSPostingKeys(ctx, store, ordinal); err != nil {
		return nil, err
	}
	return dictionary, nil
}

func validateFTSDictionaryForSave(ctx context.Context, dictionary *FTSTermDictionary) error {
	if dictionary == nil || len(dictionary.terms) != len(dictionary.postings) ||
		len(dictionary.terms) != len(dictionary.maximumTF) ||
		uint64(len(dictionary.documentLengths)) != dictionary.stats.TotalDocuments {
		return fmt.Errorf("%w: inconsistent dictionary", ErrInvalidFTSDictionary)
	}
	if uint64(len(dictionary.terms)) > math.MaxUint32 || uint64(len(dictionary.documentLengths)) > math.MaxUint32 {
		return fmt.Errorf("%w: counts exceed uint32", ErrInvalidFTSDictionary)
	}
	var totalTokens uint64
	for _, length := range dictionary.documentLengths {
		if math.MaxUint64-totalTokens < uint64(length) {
			return fmt.Errorf("%w: total token count overflows", ErrInvalidFTSDictionary)
		}
		totalTokens += uint64(length)
	}
	if totalTokens != dictionary.stats.TotalTokens {
		return fmt.Errorf("%w: total token count mismatch", ErrInvalidFTSDictionary)
	}
	for index, term := range dictionary.terms {
		if index&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if index > 0 && dictionary.terms[index-1] >= term || dictionary.postings[index] == nil ||
			dictionary.postings[index].DocumentFrequency() == 0 || dictionary.maximumTF[index] == 0 {
			return fmt.Errorf("%w: invalid term metadata at %d", ErrInvalidFTSDictionary, index)
		}
	}
	return nil
}

func requireEmptyFTSDirectory(path string) error {
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidFTSDictionary)
	}
	entries, err := os.ReadDir(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("core: inspect FTS directory: %w", err)
	}
	if len(entries) != 0 {
		return fmt.Errorf("%w: FTS directory is not empty", ErrInvalidFTSDictionary)
	}
	return nil
}

type ftsStoreWriter struct {
	store *indexstore.Store
	batch *indexstore.Batch
	size  int
}

func newFTSStoreWriter(store *indexstore.Store) *ftsStoreWriter {
	return &ftsStoreWriter{store: store, batch: store.NewBatch()}
}

func (w *ftsStoreWriter) set(key, value []byte) error {
	if err := w.batch.Set(key, value); err != nil {
		return err
	}
	w.size += len(key) + len(value)
	if w.size < ftsPebbleBatchBytes {
		return nil
	}
	if err := errors.Join(w.batch.Commit(), w.batch.Close()); err != nil {
		return err
	}
	w.batch = w.store.NewBatch()
	w.size = 0
	return nil
}

func (w *ftsStoreWriter) close() error {
	if w == nil || w.batch == nil {
		return nil
	}
	batch := w.batch
	w.batch = nil
	return errors.Join(batch.Commit(), batch.Close())
}

func writeFTSDocumentLengths(ctx context.Context, writer *ftsStoreWriter, lengths []uint32) error {
	offset := 0
	chunk := uint32(0)
	for offset < len(lengths) || chunk == 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(offset+ftsPebbleChunkBytes/4, len(lengths))
		value := make([]byte, (end-offset)*4)
		for index, length := range lengths[offset:end] {
			binary.LittleEndian.PutUint32(value[index*4:], length)
		}
		key := make([]byte, 5)
		key[0] = 'd'
		binary.BigEndian.PutUint32(key[1:], chunk)
		if err := writer.set(key, value); err != nil {
			return err
		}
		offset = end
		chunk++
	}
	return nil
}

func readFTSDocumentLengths(ctx context.Context, store *indexstore.Store) ([]uint32, error) {
	iterator, err := store.NewPrefixIterator([]byte{'d'})
	if err != nil {
		return nil, ftsDictionaryCorruption("open document-length iterator", err)
	}
	var lengths []uint32
	chunk := uint32(0)
	for valid := iterator.First(); valid; valid = iterator.Next() {
		if err := ctx.Err(); err != nil {
			_ = iterator.Close()
			return nil, err
		}
		key, value := iterator.Key(), iterator.Value()
		if len(key) != 5 || key[0] != 'd' || binary.BigEndian.Uint32(key[1:]) != chunk ||
			len(value)%4 != 0 || len(value) > ftsPebbleChunkBytes || len(value) == 0 && chunk != 0 {
			_ = iterator.Close()
			return nil, ftsDictionaryCorruption("invalid document-length chunk", nil)
		}
		for offset := 0; offset < len(value); offset += 4 {
			lengths = append(lengths, binary.LittleEndian.Uint32(value[offset:]))
		}
		chunk++
	}
	if err := errors.Join(iterator.Error(), iterator.Close()); err != nil {
		return nil, ftsDictionaryCorruption("iterate document-length chunks", err)
	}
	if chunk == 0 {
		return nil, ftsDictionaryCorruption("missing document-length chunks", nil)
	}
	return lengths, nil
}

func writeFTSPostingChunks(ctx context.Context, writer *ftsStoreWriter, ordinal uint32, data []byte) error {
	for offset, chunk := 0, uint32(0); offset < len(data); chunk++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(offset+ftsPebbleChunkBytes, len(data))
		key := make([]byte, 9)
		key[0] = 'p'
		binary.BigEndian.PutUint32(key[1:5], ordinal)
		binary.BigEndian.PutUint32(key[5:9], chunk)
		if err := writer.set(key, data[offset:end]); err != nil {
			return err
		}
		offset = end
	}
	return nil
}

func readFTSPostingChunks(ctx context.Context, store *indexstore.Store, ordinal uint32) ([]byte, error) {
	prefix := make([]byte, 5)
	prefix[0] = 'p'
	binary.BigEndian.PutUint32(prefix[1:], ordinal)
	iterator, err := store.NewPrefixIterator(prefix)
	if err != nil {
		return nil, ftsDictionaryCorruption("open posting iterator", err)
	}
	var data []byte
	chunk := uint32(0)
	for valid := iterator.First(); valid; valid = iterator.Next() {
		if err := ctx.Err(); err != nil {
			_ = iterator.Close()
			return nil, err
		}
		key, value := iterator.Key(), iterator.Value()
		if len(key) != 9 || !bytes.Equal(key[:5], prefix) || binary.BigEndian.Uint32(key[5:]) != chunk ||
			len(value) == 0 || len(value) > ftsPebbleChunkBytes || uint64(len(data))+uint64(len(value)) > math.MaxUint32 {
			_ = iterator.Close()
			return nil, ftsDictionaryCorruption("invalid posting chunk", nil)
		}
		data = append(data, value...)
		chunk++
	}
	if err := errors.Join(iterator.Error(), iterator.Close()); err != nil {
		return nil, ftsDictionaryCorruption("iterate posting chunks", err)
	}
	if chunk == 0 {
		return nil, ftsDictionaryCorruption("missing posting chunks", nil)
	}
	return data, nil
}

func validateFTSPostingKeys(ctx context.Context, store *indexstore.Store, termCount uint32) error {
	iterator, err := store.NewPrefixIterator([]byte{'p'})
	if err != nil {
		return ftsDictionaryCorruption("open posting-key iterator", err)
	}
	ordinal, chunk := uint32(0), uint32(0)
	seen := false
	for valid := iterator.First(); valid; valid = iterator.Next() {
		if err := ctx.Err(); err != nil {
			_ = iterator.Close()
			return err
		}
		key := iterator.Key()
		if len(key) != 9 || key[0] != 'p' {
			_ = iterator.Close()
			return ftsDictionaryCorruption("invalid posting key", nil)
		}
		keyOrdinal, keyChunk := binary.BigEndian.Uint32(key[1:5]), binary.BigEndian.Uint32(key[5:9])
		if !seen {
			ordinal, chunk, seen = keyOrdinal, 0, true
		}
		if keyOrdinal != ordinal {
			if keyOrdinal != ordinal+1 || chunk == 0 {
				_ = iterator.Close()
				return ftsDictionaryCorruption("non-contiguous posting ordinals", nil)
			}
			ordinal, chunk = keyOrdinal, 0
		}
		if keyChunk != chunk {
			_ = iterator.Close()
			return ftsDictionaryCorruption("non-contiguous posting chunks", nil)
		}
		chunk++
	}
	if err := errors.Join(iterator.Error(), iterator.Close()); err != nil {
		return ftsDictionaryCorruption("iterate posting keys", err)
	}
	if termCount == 0 {
		if seen {
			return ftsDictionaryCorruption("postings exist without terms", nil)
		}
		return nil
	}
	if !seen || ordinal != termCount-1 || chunk == 0 {
		return ftsDictionaryCorruption("posting ordinals do not cover all terms", nil)
	}
	return nil
}

func ftsDictionaryCorruption(message string, err error) error {
	if err == nil {
		return fmt.Errorf("%w: %s", ErrCorruptFTSDictionary, message)
	}
	return fmt.Errorf("%w: %s: %v", ErrCorruptFTSDictionary, message, err)
}

func inspectFTSStoreKeys(path string) ([][]byte, error) {
	store, err := indexstore.Open(path, indexstore.Options{ReadOnly: true})
	if err != nil {
		return nil, err
	}
	defer store.Close()
	iterator, err := store.NewRangeIterator(nil, nil)
	if err != nil {
		return nil, err
	}
	var keys [][]byte
	for valid := iterator.First(); valid; valid = iterator.Next() {
		keys = append(keys, append([]byte(nil), iterator.Key()...))
	}
	return keys, errors.Join(iterator.Error(), iterator.Close())
}
