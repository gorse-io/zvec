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

package sql

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/bits"
	"os"
	"unicode/utf8"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/gorse-io/zvec/internal/indexstore"
)

const (
	invertedPebbleFormat     = "zvec-invert-pebble/1"
	invertedPebbleChunkBytes = 4 << 10
	invertedPebbleBatchBytes = 4 << 20
)

var (
	invertedFormatKey = []byte("m/format")
	invertedFieldKey  = []byte("m/field")

	// ErrCorruptInvertedIndex identifies an invalid Pebble-backed INVERT
	// snapshot. It is distinct from query and schema validation errors.
	ErrCorruptInvertedIndex = errors.New("sql: corrupt inverted index")
)

// Save writes a sealed index into a new Pebble directory. The directory is an
// immutable snapshot after Save returns.
func (i *InvertedIndex) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return errors.New("sql: nil inverted-index save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if i == nil {
		return errors.New("sql: nil inverted index")
	}
	if err := requireEmptyInvertedDirectory(path); err != nil {
		return err
	}

	i.mu.RLock()
	defer i.mu.RUnlock()
	if !i.sealed {
		return fmt.Errorf("sql: inverted index %q is not sealed", i.field.Name)
	}
	field, err := json.Marshal(i.field)
	if err != nil {
		return fmt.Errorf("sql: marshal inverted field: %w", err)
	}
	store, err := indexstore.Open(path, indexstore.Options{})
	if err != nil {
		return fmt.Errorf("sql: create inverted index store: %w", err)
	}
	complete := false
	defer func() {
		if !complete {
			_ = store.Destroy()
		}
	}()
	writer := newInvertedStoreWriter(store)
	fail := func(err error) error {
		_ = writer.close()
		return err
	}
	if err := writer.set(invertedFormatKey, []byte(invertedPebbleFormat)); err != nil {
		return fail(err)
	}
	if err := writer.set(invertedFieldKey, field); err != nil {
		return fail(err)
	}
	for kind, bitmap := range []*ailego.Bitmap{i.rows, i.nulls, i.nonNull} {
		if err := writeInvertedBitmap(ctx, writer, []byte{'b', byte(kind)}, bitmap); err != nil {
			return fail(err)
		}
	}
	for ordinal, key := range i.ordered {
		if ordinal&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return fail(err)
			}
		}
		termKey := make([]byte, 5)
		termKey[0] = 't'
		binary.BigEndian.PutUint32(termKey[1:], uint32(ordinal))
		if err := writer.set(termKey, encodeInvertedScalarKey(key)); err != nil {
			return fail(err)
		}
		if err := writeInvertedBitmap(ctx, writer, invertedPostingPrefix(uint32(ordinal)), i.postings[key]); err != nil {
			return fail(err)
		}
	}
	for _, length := range i.lengths {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		prefix := make([]byte, 5)
		prefix[0] = 'l'
		binary.BigEndian.PutUint32(prefix[1:], length)
		if err := writeInvertedBitmap(ctx, writer, prefix, i.arrayLength[length]); err != nil {
			return fail(err)
		}
	}
	if err := writer.close(); err != nil {
		return err
	}
	if err := store.Flush(); err != nil {
		return fmt.Errorf("sql: flush inverted index store: %w", err)
	}
	if err := store.Close(); err != nil {
		return fmt.Errorf("sql: close inverted index store: %w", err)
	}
	complete = true
	return nil
}

// OpenInvertedIndex opens and validates an immutable Pebble INVERT directory.
func OpenInvertedIndex(ctx context.Context, path string) (*InvertedIndex, error) {
	if ctx == nil {
		return nil, errors.New("sql: nil inverted-index open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	store, err := indexstore.Open(path, indexstore.Options{ReadOnly: true})
	if err != nil {
		return nil, invertedCorruption("open Pebble store", err)
	}
	defer store.Close()
	format, err := store.Get(invertedFormatKey)
	if err != nil || string(format) != invertedPebbleFormat {
		return nil, invertedCorruption("invalid format marker", err)
	}
	fieldBytes, err := store.Get(invertedFieldKey)
	if err != nil {
		return nil, invertedCorruption("read field metadata", err)
	}
	var field Field
	decoder := json.NewDecoder(bytes.NewReader(fieldBytes))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&field); err != nil {
		return nil, invertedCorruption("decode field metadata", err)
	}
	if err := ensureInvertedJSONEOF(decoder); err != nil {
		return nil, invertedCorruption("decode field metadata", err)
	}
	index, err := NewInvertedIndex(field)
	if err != nil {
		return nil, invertedCorruption("invalid field metadata", err)
	}
	bitmaps := []*ailego.Bitmap{index.rows, index.nulls, index.nonNull}
	for kind := range bitmaps {
		bitmaps[kind], err = readInvertedBitmap(ctx, store, []byte{'b', byte(kind)})
		if err != nil {
			return nil, err
		}
	}
	index.rows, index.nulls, index.nonNull = bitmaps[0], bitmaps[1], bitmaps[2]
	if !bitmapPartition(index.rows, index.nulls, index.nonNull) {
		return nil, invertedCorruption("NULL bitmaps do not partition rows", nil)
	}

	terms, err := store.NewPrefixIterator([]byte{'t'})
	if err != nil {
		return nil, invertedCorruption("open term iterator", err)
	}
	ordinal := uint32(0)
	var previous scalarKey
	havePrevious := false
	for valid := terms.First(); valid; valid = terms.Next() {
		if ordinal&1023 == 0 {
			if err := ctx.Err(); err != nil {
				_ = terms.Close()
				return nil, err
			}
		}
		key := terms.Key()
		if len(key) != 5 || key[0] != 't' || binary.BigEndian.Uint32(key[1:]) != ordinal {
			_ = terms.Close()
			return nil, invertedCorruption("non-contiguous term keys", nil)
		}
		scalar, decodeErr := decodeInvertedScalarKey(terms.Value(), field.Kind)
		if decodeErr != nil {
			_ = terms.Close()
			return nil, invertedCorruption("invalid term key", decodeErr)
		}
		if havePrevious {
			comparison, compareErr := compareValues(previous.value(), scalar.value())
			if compareErr != nil || comparison >= 0 {
				_ = terms.Close()
				return nil, invertedCorruption("terms are not strictly ordered", compareErr)
			}
		}
		posting, postingErr := readInvertedBitmap(ctx, store, invertedPostingPrefix(ordinal))
		if postingErr != nil {
			_ = terms.Close()
			return nil, postingErr
		}
		if posting.Count() == 0 || !bitmapSubset(posting, index.nonNull) {
			_ = terms.Close()
			return nil, invertedCorruption("posting is empty or outside the non-NULL row domain", nil)
		}
		index.postings[scalar] = posting
		index.ordered = append(index.ordered, scalar)
		previous, havePrevious = scalar, true
		ordinal++
	}
	if err := errors.Join(terms.Error(), terms.Close()); err != nil {
		return nil, invertedCorruption("iterate terms", err)
	}

	lengths, err := store.NewPrefixIterator([]byte{'l'})
	if err != nil {
		return nil, invertedCorruption("open array-length iterator", err)
	}
	var previousLength uint32
	for valid := lengths.First(); valid; {
		key := append([]byte(nil), lengths.Key()...)
		if len(key) != 9 || key[0] != 'l' {
			_ = lengths.Close()
			return nil, invertedCorruption("invalid array-length key", nil)
		}
		length := binary.BigEndian.Uint32(key[1:5])
		if len(index.lengths) > 0 && length <= previousLength {
			_ = lengths.Close()
			return nil, invertedCorruption("array lengths are not ordered", nil)
		}
		posting, postingErr := readInvertedBitmap(ctx, store, key[:5])
		if postingErr != nil {
			_ = lengths.Close()
			return nil, postingErr
		}
		if !field.Array || posting.Count() == 0 || !bitmapSubset(posting, index.nonNull) {
			_ = lengths.Close()
			return nil, invertedCorruption("invalid array-length posting", nil)
		}
		index.arrayLength[length] = posting
		index.lengths = append(index.lengths, length)
		previousLength = length
		// readInvertedBitmap used a separate iterator, so advance this iterator
		// beyond every chunk belonging to the length just consumed.
		for valid = lengths.Next(); valid && bytes.HasPrefix(lengths.Key(), key[:5]); valid = lengths.Next() {
		}
	}
	if err := errors.Join(lengths.Error(), lengths.Close()); err != nil {
		return nil, invertedCorruption("iterate array lengths", err)
	}
	index.sealed = true
	return index, nil
}

func requireEmptyInvertedDirectory(path string) error {
	if path == "" {
		return errors.New("sql: empty inverted-index path")
	}
	entries, err := os.ReadDir(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("sql: inspect inverted-index directory: %w", err)
	}
	if len(entries) != 0 {
		return errors.New("sql: inverted-index directory is not empty")
	}
	return nil
}

type invertedStoreWriter struct {
	store *indexstore.Store
	batch *indexstore.Batch
	size  int
}

func newInvertedStoreWriter(store *indexstore.Store) *invertedStoreWriter {
	return &invertedStoreWriter{store: store, batch: store.NewBatch()}
}

func (w *invertedStoreWriter) set(key, value []byte) error {
	if err := w.batch.Set(key, value); err != nil {
		return err
	}
	w.size += len(key) + len(value)
	if w.size < invertedPebbleBatchBytes {
		return nil
	}
	if err := errors.Join(w.batch.Commit(), w.batch.Close()); err != nil {
		return err
	}
	w.batch = w.store.NewBatch()
	w.size = 0
	return nil
}

func (w *invertedStoreWriter) close() error {
	if w == nil || w.batch == nil {
		return nil
	}
	batch := w.batch
	w.batch = nil
	return errors.Join(batch.Commit(), batch.Close())
}

func invertedPostingPrefix(ordinal uint32) []byte {
	prefix := make([]byte, 5)
	prefix[0] = 'p'
	binary.BigEndian.PutUint32(prefix[1:], ordinal)
	return prefix
}

func writeInvertedBitmap(ctx context.Context, writer *invertedStoreWriter, prefix []byte, bitmap *ailego.Bitmap) error {
	words := bitmap.Snapshot()
	wordOffset := 0
	chunk := uint32(0)
	for wordOffset < len(words) || chunk == 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		wordEnd := min(wordOffset+invertedPebbleChunkBytes/8, len(words))
		value := make([]byte, (wordEnd-wordOffset)*8)
		for index, word := range words[wordOffset:wordEnd] {
			binary.LittleEndian.PutUint64(value[index*8:], word)
		}
		key := make([]byte, len(prefix)+4)
		copy(key, prefix)
		binary.BigEndian.PutUint32(key[len(prefix):], chunk)
		if err := writer.set(key, value); err != nil {
			return err
		}
		wordOffset = wordEnd
		chunk++
	}
	return nil
}

func readInvertedBitmap(ctx context.Context, store *indexstore.Store, prefix []byte) (*ailego.Bitmap, error) {
	iterator, err := store.NewPrefixIterator(prefix)
	if err != nil {
		return nil, invertedCorruption("open bitmap iterator", err)
	}
	words := make([]uint64, 0)
	chunk := uint32(0)
	for valid := iterator.First(); valid; valid = iterator.Next() {
		if err := ctx.Err(); err != nil {
			_ = iterator.Close()
			return nil, err
		}
		key, value := iterator.Key(), iterator.Value()
		if len(key) != len(prefix)+4 || !bytes.Equal(key[:len(prefix)], prefix) ||
			binary.BigEndian.Uint32(key[len(prefix):]) != chunk || len(value)%8 != 0 || len(value) > invertedPebbleChunkBytes ||
			(len(value) == 0 && chunk != 0) {
			_ = iterator.Close()
			return nil, invertedCorruption("invalid bitmap chunk", nil)
		}
		for offset := 0; offset < len(value); offset += 8 {
			words = append(words, binary.LittleEndian.Uint64(value[offset:]))
		}
		chunk++
	}
	if err := errors.Join(iterator.Error(), iterator.Close()); err != nil {
		return nil, invertedCorruption("iterate bitmap chunks", err)
	}
	if chunk == 0 {
		return nil, invertedCorruption("missing bitmap chunks", nil)
	}
	return bitmapFromPersistedWords(words), nil
}

func encodeInvertedScalarKey(key scalarKey) []byte {
	encoded := []byte{byte(key.kind)}
	switch key.kind {
	case ValueBinary, ValueString:
		encoded = append(encoded, key.text...)
	case ValueBool:
		if key.boolean {
			encoded = append(encoded, 1)
		} else {
			encoded = append(encoded, 0)
		}
	case ValueInt32, ValueInt64:
		encoded = binary.LittleEndian.AppendUint64(encoded, uint64(key.signed))
	case ValueUint32, ValueUint64:
		encoded = binary.LittleEndian.AppendUint64(encoded, key.unsigned)
	case ValueFloat32, ValueFloat64:
		encoded = binary.LittleEndian.AppendUint64(encoded, key.bits)
	}
	return encoded
}

func decodeInvertedScalarKey(encoded []byte, kind ValueKind) (scalarKey, error) {
	if len(encoded) == 0 || ValueKind(encoded[0]) != kind {
		return scalarKey{}, errors.New("kind mismatch")
	}
	key := scalarKey{kind: kind}
	switch kind {
	case ValueBinary, ValueString:
		key.text = string(encoded[1:])
	case ValueBool:
		if len(encoded) != 2 || encoded[1] > 1 {
			return scalarKey{}, errors.New("invalid boolean")
		}
		key.boolean = encoded[1] == 1
	case ValueInt32, ValueInt64:
		if len(encoded) != 9 {
			return scalarKey{}, errors.New("invalid signed integer")
		}
		key.signed = int64(binary.LittleEndian.Uint64(encoded[1:]))
	case ValueUint32, ValueUint64:
		if len(encoded) != 9 {
			return scalarKey{}, errors.New("invalid unsigned integer")
		}
		key.unsigned = binary.LittleEndian.Uint64(encoded[1:])
	case ValueFloat32, ValueFloat64:
		if len(encoded) != 9 {
			return scalarKey{}, errors.New("invalid floating point value")
		}
		key.bits = binary.LittleEndian.Uint64(encoded[1:])
	default:
		return scalarKey{}, errors.New("invalid value kind")
	}
	if !validPersistedScalarKey(key) {
		return scalarKey{}, errors.New("invalid scalar value")
	}
	return key, nil
}

func validPersistedScalarKey(key scalarKey) bool {
	if !key.kind.valid() {
		return false
	}
	switch key.kind {
	case ValueString:
		return utf8.ValidString(key.text)
	case ValueInt32:
		return key.signed >= math.MinInt32 && key.signed <= math.MaxInt32
	case ValueUint32:
		return key.unsigned <= math.MaxUint32
	case ValueFloat32:
		value := math.Float64frombits(key.bits)
		return !math.IsNaN(value) && !math.IsInf(value, 0) && float64(float32(value)) == value
	case ValueFloat64:
		value := math.Float64frombits(key.bits)
		return !math.IsNaN(value) && !math.IsInf(value, 0)
	}
	return true
}

func bitmapFromPersistedWords(words []uint64) *ailego.Bitmap {
	bitmap := ailego.NewBitmap(uint64(len(words)) * 64)
	for wordIndex, word := range words {
		for word != 0 {
			bit := bits.TrailingZeros64(word)
			bitmap.Set(uint64(wordIndex*64 + bit))
			word &= word - 1
		}
	}
	return bitmap
}

func bitmapSubset(left, right *ailego.Bitmap) bool {
	leftWords, rightWords := left.Snapshot(), right.Snapshot()
	for index, word := range leftWords {
		if index >= len(rightWords) {
			if word != 0 {
				return false
			}
			continue
		}
		if word&^rightWords[index] != 0 {
			return false
		}
	}
	return true
}

func bitmapPartition(rows, nulls, nonNull *ailego.Bitmap) bool {
	rowWords, nullWords, nonNullWords := rows.Snapshot(), nulls.Snapshot(), nonNull.Snapshot()
	length := max(len(rowWords), len(nullWords), len(nonNullWords))
	for index := 0; index < length; index++ {
		var row, null, present uint64
		if index < len(rowWords) {
			row = rowWords[index]
		}
		if index < len(nullWords) {
			null = nullWords[index]
		}
		if index < len(nonNullWords) {
			present = nonNullWords[index]
		}
		if null&present != 0 || null|present != row {
			return false
		}
	}
	return true
}

func invertedCorruption(message string, err error) error {
	if err == nil {
		return fmt.Errorf("%w: %s", ErrCorruptInvertedIndex, message)
	}
	return fmt.Errorf("%w: %s: %v", ErrCorruptInvertedIndex, message, err)
}

func ensureInvertedJSONEOF(decoder *json.Decoder) error {
	var extra any
	err := decoder.Decode(&extra)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err == nil {
		return errors.New("multiple JSON values")
	}
	return err
}

func inspectInvertedStoreKeys(path string) ([][]byte, error) {
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
