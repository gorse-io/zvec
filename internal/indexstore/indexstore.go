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

// Package indexstore provides the deliberately small Pebble surface used by
// immutable collection index artifacts.
package indexstore

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/cockroachdb/pebble/v2"
	"github.com/gorse-io/zvec/internal/ailego"
)

const (
	indexstoreMarkerName     = "ZVEC-INDEX"
	indexstoreMarkerContents = "zvec-indexstore-v1\n"
)

var (
	// ErrNotFound identifies a missing key.
	ErrNotFound = pebble.ErrNotFound
	// ErrClosed identifies an operation on a closed store, batch, or iterator.
	ErrClosed = errors.New("indexstore: closed")
	// ErrReadOnly identifies a mutation attempted through a read-only store.
	ErrReadOnly = errors.New("indexstore: read-only")
)

// Options controls how a Store is opened.
type Options struct {
	ReadOnly bool
}

// Store is a Pebble database rooted at one directory.
type Store struct {
	mu       sync.RWMutex
	db       *pebble.DB
	path     string
	readOnly bool
	closed   bool
}

// Open opens or creates a Pebble store. Existing symbolic-link paths are
// rejected before Pebble can follow them.
func Open(path string, options Options) (*Store, error) {
	if path == "" {
		return nil, errors.New("indexstore: empty path")
	}
	clean, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("indexstore: resolve path %q: %w", path, err)
	}
	clean = filepath.Clean(clean)
	empty := true
	if info, err := os.Lstat(clean); err == nil {
		if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
			return nil, fmt.Errorf("indexstore: path %q is not a directory", clean)
		}
		entries, err := os.ReadDir(clean)
		if err != nil {
			return nil, fmt.Errorf("indexstore: read path %q: %w", clean, err)
		}
		empty = len(entries) == 0
		if !empty {
			if err := validateIndexstoreMarker(clean); err != nil {
				return nil, err
			}
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("indexstore: inspect path %q: %w", clean, err)
	}
	db, err := pebble.Open(clean, &pebble.Options{
		ReadOnly:           options.ReadOnly,
		ErrorIfNotExists:   options.ReadOnly,
		FormatMajorVersion: pebble.FormatNewest,
	})
	if err != nil {
		return nil, fmt.Errorf("indexstore: open %q: %w", clean, err)
	}
	if empty {
		if options.ReadOnly {
			_ = db.Close()
			return nil, fmt.Errorf("indexstore: store %q has no zvec marker", clean)
		}
		if err := writeIndexstoreMarker(clean); err != nil {
			_ = db.Close()
			return nil, err
		}
	}
	return &Store{db: db, path: clean, readOnly: options.ReadOnly}, nil
}

// Get returns an owned value copy.
func (s *Store) Get(key []byte) ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed || s.db == nil {
		return nil, ErrClosed
	}
	value, closer, err := s.db.Get(key)
	if err != nil {
		return nil, translateError(err)
	}
	owned := append([]byte(nil), value...)
	if err := closer.Close(); err != nil {
		return nil, translateError(err)
	}
	return owned, nil
}

// Set stores one key and value durably.
func (s *Store) Set(key, value []byte) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.requireWritableLocked(); err != nil {
		return err
	}
	return translateError(s.db.Set(key, value, pebble.Sync))
}

// Delete durably removes one key.
func (s *Store) Delete(key []byte) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.requireWritableLocked(); err != nil {
		return err
	}
	return translateError(s.db.Delete(key, pebble.Sync))
}

// Flush writes the current memtable to an SSTable.
func (s *Store) Flush() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.requireWritableLocked(); err != nil {
		return err
	}
	return translateError(s.db.Flush())
}

// Compact compacts the half-open key range [start, end).
func (s *Store) Compact(start, end []byte) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.requireWritableLocked(); err != nil {
		return err
	}
	return translateError(s.db.Compact(context.Background(), start, end, false))
}

// Checkpoint creates a point-in-time Pebble directory at path.
func (s *Store) Checkpoint(path string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed || s.db == nil {
		return ErrClosed
	}
	if path == "" {
		return errors.New("indexstore: empty checkpoint path")
	}
	clean := filepath.Clean(path)
	if info, err := os.Lstat(clean); err == nil {
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("indexstore: checkpoint path %q is a symbolic link", clean)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("indexstore: inspect checkpoint path %q: %w", clean, err)
	}
	if err := s.db.Checkpoint(clean, pebble.WithFlushedWAL()); err != nil {
		return translateError(err)
	}
	return writeIndexstoreMarker(clean)
}

// NewBatch creates a write batch owned by the store.
func (s *Store) NewBatch() *Batch {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed || s.db == nil {
		return &Batch{closed: true}
	}
	return &Batch{store: s, batch: s.db.NewBatch()}
}

// NewPrefixIterator creates an iterator over all keys beginning with prefix.
func (s *Store) NewPrefixIterator(prefix []byte) (*Iterator, error) {
	return s.newIterator(prefix, prefixSuccessor(prefix))
}

// NewRangeIterator creates an iterator over the half-open range [lower, upper).
// A nil bound leaves that side of the range unbounded.
func (s *Store) NewRangeIterator(lower, upper []byte) (*Iterator, error) {
	return s.newIterator(lower, upper)
}

func (s *Store) newIterator(lower, upper []byte) (*Iterator, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed || s.db == nil {
		return nil, ErrClosed
	}
	iterator, err := s.db.NewIter(&pebble.IterOptions{
		LowerBound: append([]byte(nil), lower...),
		UpperBound: append([]byte(nil), upper...),
	})
	if err != nil {
		return nil, translateError(err)
	}
	return &Iterator{iterator: iterator}, nil
}

// Close closes the Pebble database. It is idempotent.
func (s *Store) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	db := s.db
	s.db = nil
	if db == nil {
		return nil
	}
	return translateError(db.Close())
}

// Destroy closes the database and removes its exact directory. It may be
// called only once and is unavailable for read-only stores.
func (s *Store) Destroy() error {
	if s == nil {
		return ErrClosed
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.db == nil {
		return ErrClosed
	}
	if s.readOnly {
		return ErrReadOnly
	}
	if !safeIndexstoreDestroyPath(s.path) {
		return fmt.Errorf("indexstore: refuse to destroy unsafe path %q", s.path)
	}
	s.closed = true
	db := s.db
	s.db = nil
	closeErr := db.Close()
	removeErr := os.RemoveAll(s.path)
	return errors.Join(translateError(closeErr), removeErr)
}

func (s *Store) requireWritableLocked() error {
	if s.closed || s.db == nil {
		return ErrClosed
	}
	if s.readOnly {
		return ErrReadOnly
	}
	return nil
}

// Batch is an atomic group of mutations.
type Batch struct {
	mu        sync.Mutex
	store     *Store
	batch     *pebble.Batch
	committed bool
	closed    bool
}

// Set adds a key assignment to the batch.
func (b *Batch) Set(key, value []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.requireOpenLocked(); err != nil {
		return err
	}
	return translateError(b.batch.Set(key, value, nil))
}

// Delete adds a key deletion to the batch.
func (b *Batch) Delete(key []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.requireOpenLocked(); err != nil {
		return err
	}
	return translateError(b.batch.Delete(key, nil))
}

// Commit applies the batch durably. A batch can be committed once.
func (b *Batch) Commit() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.requireOpenLocked(); err != nil {
		return err
	}
	b.store.mu.RLock()
	defer b.store.mu.RUnlock()
	if err := b.store.requireWritableLocked(); err != nil {
		return err
	}
	if err := translateError(b.batch.Commit(pebble.Sync)); err != nil {
		return err
	}
	b.committed = true
	return nil
}

// Close releases the batch. It is idempotent.
func (b *Batch) Close() error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	b.closed = true
	if b.batch == nil {
		return nil
	}
	err := b.batch.Close()
	b.batch = nil
	return translateError(err)
}

func (b *Batch) requireOpenLocked() error {
	if b == nil || b.closed || b.batch == nil || b.committed {
		return ErrClosed
	}
	return nil
}

// Iterator traverses one immutable Pebble view in byte-key order.
type Iterator struct {
	mu       sync.Mutex
	iterator *pebble.Iterator
	closed   bool
}

// First positions the iterator at its first key.
func (i *Iterator) First() bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	return !i.closed && i.iterator != nil && i.iterator.First()
}

// Next advances the iterator.
func (i *Iterator) Next() bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	return !i.closed && i.iterator != nil && i.iterator.Next()
}

// Key returns the current key. It remains valid until the next positioning
// operation or Close.
func (i *Iterator) Key() []byte {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.closed || i.iterator == nil || !i.iterator.Valid() {
		return nil
	}
	return i.iterator.Key()
}

// Value returns the current value. It remains valid until the next positioning
// operation or Close.
func (i *Iterator) Value() []byte {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.closed || i.iterator == nil || !i.iterator.Valid() {
		return nil
	}
	return i.iterator.Value()
}

// Error reports an iteration error.
func (i *Iterator) Error() error {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.closed || i.iterator == nil {
		return nil
	}
	return translateError(i.iterator.Error())
}

// Close releases the iterator. It is idempotent.
func (i *Iterator) Close() error {
	if i == nil {
		return nil
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.closed {
		return nil
	}
	i.closed = true
	if i.iterator == nil {
		return nil
	}
	err := i.iterator.Close()
	i.iterator = nil
	return translateError(err)
}

func prefixSuccessor(prefix []byte) []byte {
	successor := append([]byte(nil), prefix...)
	for index := len(successor) - 1; index >= 0; index-- {
		if successor[index] != 0xff {
			successor[index]++
			return successor[:index+1]
		}
	}
	return nil
}

func translateError(err error) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, pebble.ErrNotFound):
		return ErrNotFound
	case errors.Is(err, pebble.ErrClosed):
		return ErrClosed
	default:
		return err
	}
}

func validateIndexstoreMarker(path string) error {
	name := filepath.Join(path, indexstoreMarkerName)
	info, err := os.Lstat(name)
	if err != nil {
		return fmt.Errorf("indexstore: inspect zvec marker in %q: %w", path, err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() || info.Size() != int64(len(indexstoreMarkerContents)) {
		return fmt.Errorf("indexstore: invalid zvec marker in %q", path)
	}
	contents, err := os.ReadFile(name)
	if err != nil {
		return fmt.Errorf("indexstore: read zvec marker in %q: %w", path, err)
	}
	if !bytes.Equal(contents, []byte(indexstoreMarkerContents)) {
		return fmt.Errorf("indexstore: corrupt zvec marker in %q", path)
	}
	return nil
}

func writeIndexstoreMarker(path string) (err error) {
	file, err := os.CreateTemp(path, ".zvec-index-*")
	if err != nil {
		return fmt.Errorf("indexstore: create marker temporary file: %w", err)
	}
	temporary := file.Name()
	defer os.Remove(temporary)
	if err := file.Chmod(0o600); err != nil {
		_ = file.Close()
		return fmt.Errorf("indexstore: chmod marker temporary file: %w", err)
	}
	if _, err := file.Write([]byte(indexstoreMarkerContents)); err != nil {
		_ = file.Close()
		return fmt.Errorf("indexstore: write marker temporary file: %w", err)
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return fmt.Errorf("indexstore: sync marker temporary file: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("indexstore: close marker temporary file: %w", err)
	}
	if err := os.Rename(temporary, filepath.Join(path, indexstoreMarkerName)); err != nil {
		return fmt.Errorf("indexstore: publish zvec marker: %w", err)
	}
	if err := ailego.SyncDirectory(path); err != nil {
		return fmt.Errorf("indexstore: sync marker directory: %w", err)
	}
	return nil
}

func safeIndexstoreDestroyPath(path string) bool {
	if path == "" || !filepath.IsAbs(path) {
		return false
	}
	clean := filepath.Clean(path)
	volume := filepath.VolumeName(clean)
	root := volume + string(os.PathSeparator)
	return clean != root && clean != volume && filepath.Dir(clean) != clean
}
