// Copyright 2026-present the zvec-go project
//
// Licensed under the Apache License, Version 2.0 (the "License");
package sql

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gorse-io/zvec/internal/indexstore"
	"github.com/stretchr/testify/require"
)

func TestInvertedIndexPebbleRoundTripChunksAndCorruption(t *testing.T) {
	field := Field{Name: "tags", Kind: ValueString, Array: true, Nullable: true, Filterable: true, Indexed: true, RangeOptimized: true}
	index := mustInvertedIndex(t, field,
		mustNullValue(t, ValueString, true),
		mustArray(t, ValueString),
		mustArray(t, ValueString, StringValue("a"), StringValue("b")),
		mustArray(t, ValueString, StringValue("b"), StringValue("c")),
	)
	path := filepath.Join(t.TempDir(), "invert.pebble")
	require.NoError(t, index.Save(context.Background(), path))
	info, err := os.Stat(path)
	require.NoError(t, err)
	require.True(t, info.IsDir())

	reopened, err := OpenInvertedIndex(context.Background(), path)
	require.NoError(t, err)
	require.Equal(t, field, reopened.Field())
	require.Equal(t, uint64(4), reopened.RowCount())
	predicate, err := NewSetPredicate(PredicateContainAny, false, []Value{StringValue("a"), StringValue("c")})
	require.NoError(t, err)
	result, err := reopened.Search(predicate)
	require.NoError(t, err)
	require.Equal(t, []uint64{2, 3}, bitmapBits(result.Bitmap))
	length, err := NewComparisonPredicate(PredicateEQ, Uint32Value(0))
	require.NoError(t, err)
	result, err = reopened.SearchArrayLength(length)
	require.NoError(t, err)
	require.Equal(t, []uint64{1}, bitmapBits(result.Bitmap))

	current := filepath.Join(path, "ZVEC-INDEX")
	require.NoError(t, os.WriteFile(current, []byte("corrupt\n"), 0o600))
	_, err = OpenInvertedIndex(context.Background(), path)
	require.ErrorIs(t, err, ErrCorruptInvertedIndex)
}

func TestInvertedIndexPebbleUsesMultipleOrderedPostingKeys(t *testing.T) {
	field := Field{Name: "value", Kind: ValueInt64, Filterable: true, Indexed: true, RangeOptimized: true}
	index, err := NewInvertedIndex(field)
	require.NoError(t, err)
	for row := 0; row < 70_000; row++ {
		require.NoError(t, index.Add(uint64(row), Int64Value(int64(row%2))))
	}
	require.NoError(t, index.Seal())
	path := filepath.Join(t.TempDir(), "chunked.pebble")
	require.NoError(t, index.Save(context.Background(), path))
	keys, err := inspectInvertedStoreKeys(path)
	require.NoError(t, err)
	var postingKeys int
	for _, key := range keys {
		if len(key) > 0 && key[0] == 'p' {
			postingKeys++
		}
	}
	require.Greater(t, postingKeys, len(index.ordered), "postings were stored as one value per term")
}

func TestInvertedPebbleScalarKeyCodec(t *testing.T) {
	keys := []scalarKey{
		{kind: ValueBinary, text: string([]byte{0, 0xff})},
		{kind: ValueString, text: "hello"},
		{kind: ValueBool, boolean: true},
		{kind: ValueInt32, signed: math.MinInt32},
		{kind: ValueInt64, signed: math.MinInt64},
		{kind: ValueUint32, unsigned: math.MaxUint32},
		{kind: ValueUint64, unsigned: math.MaxUint64},
		{kind: ValueFloat32, bits: math.Float64bits(float64(float32(1.5)))},
		{kind: ValueFloat64, bits: math.Float64bits(-1.5)},
	}
	for _, key := range keys {
		decoded, err := decodeInvertedScalarKey(encodeInvertedScalarKey(key), key.kind)
		require.NoError(t, err)
		require.Equal(t, key, decoded)
	}

	require.False(t, validPersistedScalarKey(scalarKey{}))
	invalid := [][]byte{
		nil,
		{byte(ValueBool), 2},
		encodeInvertedScalarKey(scalarKey{kind: ValueInt32, signed: math.MaxInt64}),
		encodeInvertedScalarKey(scalarKey{kind: ValueUint32, unsigned: math.MaxUint64}),
		encodeInvertedScalarKey(scalarKey{kind: ValueFloat32, bits: math.Float64bits(1.1)}),
		encodeInvertedScalarKey(scalarKey{kind: ValueFloat64, bits: math.Float64bits(math.NaN())}),
	}
	kinds := []ValueKind{ValueString, ValueBool, ValueInt32, ValueUint32, ValueFloat32, ValueFloat64}
	for index := range invalid {
		_, err := decodeInvertedScalarKey(invalid[index], kinds[index])
		require.Error(t, err)
	}
}

func TestInvertedIndexPebbleRejectsInvalidInputsAndCorruption(t *testing.T) {
	field := Field{Name: "value", Kind: ValueInt64, Filterable: true, Indexed: true, RangeOptimized: true}
	index := mustInvertedIndex(t, field, Int64Value(1), Int64Value(2))
	require.Error(t, index.Save(nil, filepath.Join(t.TempDir(), "nil.pebble")))
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	require.ErrorIs(t, index.Save(canceled, filepath.Join(t.TempDir(), "canceled.pebble")), context.Canceled)
	var missing *InvertedIndex
	require.Error(t, missing.Save(context.Background(), filepath.Join(t.TempDir(), "missing.pebble")))
	require.Error(t, func() error { _, err := OpenInvertedIndex(nil, "missing"); return err }())
	require.ErrorIs(t, func() error { _, err := OpenInvertedIndex(canceled, "missing"); return err }(), context.Canceled)
	require.Error(t, index.Save(context.Background(), ""))
	nonEmpty := filepath.Join(t.TempDir(), "non-empty")
	require.NoError(t, os.Mkdir(nonEmpty, 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(nonEmpty, "entry"), []byte("x"), 0o600))
	require.Error(t, index.Save(context.Background(), nonEmpty))
	unsealed, err := NewInvertedIndex(field)
	require.NoError(t, err)
	require.Error(t, unsealed.Save(context.Background(), filepath.Join(t.TempDir(), "unsealed.pebble")))

	mutations := map[string]func(*indexstore.Store) error{
		"missing format": func(store *indexstore.Store) error { return store.Delete(invertedFormatKey) },
		"invalid field":  func(store *indexstore.Store) error { return store.Set(invertedFieldKey, []byte("{} {}")) },
		"missing rows": func(store *indexstore.Store) error {
			return store.Delete([]byte{'b', 0, 0, 0, 0, 0})
		},
		"invalid term": func(store *indexstore.Store) error {
			return store.Set([]byte{'t', 0, 0, 0, 0}, []byte{byte(ValueInt64)})
		},
		"invalid length key": func(store *indexstore.Store) error {
			key := make([]byte, 9)
			key[0] = 'l'
			binary.BigEndian.PutUint32(key[1:5], 1)
			return store.Set(key, []byte{1})
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "corrupt.pebble")
			require.NoError(t, index.Save(context.Background(), path))
			store, err := indexstore.Open(path, indexstore.Options{})
			require.NoError(t, err)
			require.NoError(t, mutate(store))
			require.NoError(t, store.Close())
			_, err = OpenInvertedIndex(context.Background(), path)
			require.ErrorIs(t, err, ErrCorruptInvertedIndex)
		})
	}
}
