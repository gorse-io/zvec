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

package ailego

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
)

const atomicWriteChunk = 1 << 20

// WriteFileAtomic durably replaces path with data. The temporary file lives in
// the destination directory, so publication is one atomic filesystem rename.
// The parent directory must already exist.
func WriteFileAtomic(ctx context.Context, path string, data []byte, perm fs.FileMode) (err error) {
	if ctx == nil {
		return errors.New("ailego: nil atomic write context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return errors.New("ailego: empty atomic write path")
	}

	dir := filepath.Dir(path)
	file, err := os.CreateTemp(dir, ".zvec-atomic-*.tmp")
	if err != nil {
		return fmt.Errorf("ailego: create atomic file: %w", err)
	}
	temp := file.Name()
	remove := true
	defer func() {
		if file != nil {
			closeErr := file.Close()
			if err == nil && closeErr != nil {
				err = closeErr
			}
		}
		if remove {
			_ = os.Remove(temp)
		}
	}()

	if err = file.Chmod(perm); err != nil {
		return fmt.Errorf("ailego: set atomic file mode: %w", err)
	}
	for len(data) > 0 {
		if err = ctx.Err(); err != nil {
			return err
		}
		chunk := min(len(data), atomicWriteChunk)
		if err = writeFull(file, data[:chunk]); err != nil {
			return fmt.Errorf("ailego: write atomic file: %w", err)
		}
		data = data[chunk:]
	}
	if err = file.Sync(); err != nil {
		return fmt.Errorf("ailego: sync atomic file: %w", err)
	}
	if err = file.Close(); err != nil {
		return fmt.Errorf("ailego: close atomic file: %w", err)
	}
	file = nil
	if err = ctx.Err(); err != nil {
		return err
	}
	if err = atomicReplaceFile(temp, path); err != nil {
		return fmt.Errorf("ailego: publish atomic file: %w", err)
	}
	remove = false
	if err = syncDirectory(dir); err != nil {
		return fmt.Errorf("ailego: sync atomic file directory: %w", err)
	}
	return nil
}

// SyncDirectory persists directory-entry changes where the platform supports
// directory fsync. It is a no-op on platforms without portable directory
// handles.
func SyncDirectory(path string) error {
	if path == "" {
		return errors.New("ailego: empty directory path")
	}
	return syncDirectory(path)
}

func writeFull(writer io.Writer, data []byte) error {
	for len(data) > 0 {
		written, err := writer.Write(data)
		if written > 0 {
			data = data[written:]
		}
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}
