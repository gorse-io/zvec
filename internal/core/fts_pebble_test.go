// Copyright 2026-present the zvec-go project
//
// Licensed under the Apache License, Version 2.0 (the "License");
package core

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

func TestFTSTermDictionaryPebbleRoundTripAndCorruption(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{
		{{Text: "", Position: 0}, {Text: "alpha", Position: 1}, {Text: "alpha", Position: 2}},
		nil,
		{{Text: "alphabet", Position: 0}, {Text: string([]byte{0xff, 'x'}), Position: 1}},
	})
	path := filepath.Join(t.TempDir(), "fts.pebble")
	require.NoError(t, dictionary.Save(context.Background(), path))
	info, err := os.Stat(path)
	require.NoError(t, err)
	require.True(t, info.IsDir())

	reopened, err := OpenFTSTermDictionary(context.Background(), path)
	require.NoError(t, err)
	assertFTSDictionariesEqual(t, reopened, dictionary)

	require.NoError(t, os.WriteFile(filepath.Join(path, "ZVEC-INDEX"), []byte("corrupt\n"), 0o600))
	_, err = OpenFTSTermDictionary(context.Background(), path)
	require.ErrorIs(t, err, ErrCorruptFTSDictionary)
}

func TestFTSTermDictionaryPebblePostingChunks(t *testing.T) {
	documents := make([][]Token, 10_000)
	for index := range documents {
		documents[index] = []Token{{Text: "common", Position: 0}, {Text: "common", Position: 1}}
	}
	dictionary := buildFTSTestDictionary(t, documents)
	path := filepath.Join(t.TempDir(), "chunked.pebble")
	require.NoError(t, dictionary.Save(context.Background(), path))
	keys, err := inspectFTSStoreKeys(path)
	require.NoError(t, err)
	var chunks int
	for _, key := range keys {
		if len(key) > 0 && key[0] == 'p' {
			chunks++
		}
	}
	require.Greater(t, chunks, dictionary.TermCount(), "posting lists were stored as one value per term")
}

func TestValidateFTSPostingKeysHonorsCancellation(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{{{Text: "alpha", Position: 0}}})
	path := filepath.Join(t.TempDir(), "cancel.pebble")
	require.NoError(t, dictionary.Save(context.Background(), path))
	store, err := indexstore.Open(path, indexstore.Options{ReadOnly: true})
	require.NoError(t, err)
	defer store.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	require.ErrorIs(t, validateFTSPostingKeys(ctx, store, 1), context.Canceled)
}

func TestFTSTermDictionaryPebbleRejectsInvalidInputsAndCorruption(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{{{Text: "alpha", Position: 0}}})
	require.ErrorIs(t, dictionary.Save(nil, filepath.Join(t.TempDir(), "nil.pebble")), ErrInvalidFTSDictionary)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	require.ErrorIs(t, dictionary.Save(canceled, filepath.Join(t.TempDir(), "canceled.pebble")), context.Canceled)
	var missing *FTSTermDictionary
	require.ErrorIs(t, missing.Save(context.Background(), filepath.Join(t.TempDir(), "missing.pebble")), ErrInvalidFTSDictionary)
	require.ErrorIs(t, func() error { _, err := OpenFTSTermDictionary(nil, "missing"); return err }(), ErrCorruptFTSDictionary)
	require.ErrorIs(t, func() error { _, err := OpenFTSTermDictionary(canceled, "missing"); return err }(), context.Canceled)
	require.ErrorIs(t, dictionary.Save(context.Background(), ""), ErrInvalidFTSDictionary)
	nonEmpty := filepath.Join(t.TempDir(), "non-empty")
	require.NoError(t, os.Mkdir(nonEmpty, 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(nonEmpty, "entry"), []byte("x"), 0o600))
	require.ErrorIs(t, dictionary.Save(context.Background(), nonEmpty), ErrInvalidFTSDictionary)

	inconsistent := *dictionary
	inconsistent.maximumTF = nil
	require.ErrorIs(t, inconsistent.Save(context.Background(), filepath.Join(t.TempDir(), "inconsistent.pebble")), ErrInvalidFTSDictionary)
	badStats := *dictionary
	badStats.stats.TotalTokens++
	require.ErrorIs(t, badStats.Save(context.Background(), filepath.Join(t.TempDir(), "stats.pebble")), ErrInvalidFTSDictionary)
	badMaximumTF := *dictionary
	badMaximumTF.maximumTF = []uint32{0}
	require.ErrorIs(t, badMaximumTF.Save(context.Background(), filepath.Join(t.TempDir(), "tf.pebble")), ErrInvalidFTSDictionary)

	mutations := map[string]func(*indexstore.Store) error{
		"missing format": func(store *indexstore.Store) error { return store.Delete(ftsFormatKey) },
		"short stats":    func(store *indexstore.Store) error { return store.Set(ftsStatsKey, []byte{1}) },
		"too many documents": func(store *indexstore.Store) error {
			stats := make([]byte, 16)
			binary.LittleEndian.PutUint64(stats, uint64(math.MaxUint32)+1)
			return store.Set(ftsStatsKey, stats)
		},
		"missing lengths": func(store *indexstore.Store) error {
			return store.Delete([]byte{'d', 0, 0, 0, 0})
		},
		"invalid term": func(store *indexstore.Store) error {
			return store.Set([]byte{'t', 0, 0, 0, 0}, []byte{1})
		},
		"orphan posting": func(store *indexstore.Store) error {
			return store.Set([]byte{'p', 0, 0, 0, 1, 0, 0, 0, 0}, []byte{1})
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "corrupt.pebble")
			require.NoError(t, dictionary.Save(context.Background(), path))
			store, err := indexstore.Open(path, indexstore.Options{})
			require.NoError(t, err)
			require.NoError(t, mutate(store))
			require.NoError(t, store.Close())
			_, err = OpenFTSTermDictionary(context.Background(), path)
			require.ErrorIs(t, err, ErrCorruptFTSDictionary)
		})
	}
}
