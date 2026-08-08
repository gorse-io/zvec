# zvec

[![CI](https://github.com/gorse-io/zvec/actions/workflows/ci.yml/badge.svg)](https://github.com/gorse-io/zvec/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gorse-io/zvec/graph/badge.svg)](https://codecov.io/gh/gorse-io/zvec)
[![Go Reference](https://pkg.go.dev/badge/github.com/gorse-io/zvec.svg)](https://pkg.go.dev/github.com/gorse-io/zvec)
[![Go Version](https://img.shields.io/github/go-mod/go-version/gorse-io/zvec)](go.mod)
[![License](https://img.shields.io/github/license/gorse-io/zvec)](LICENSE)

zvec is a pure-Go reimplementation of [Alibaba zvec](https://github.com/alibaba/zvec),
providing an embedded vector database with durable local storage. It runs inside
your application without CGO, a separate database server, or prebuilt native
libraries.

> [!WARNING]
> zvec is under active development and is not ready for production use. Public
> APIs and on-disk formats may change before v1.0.

## Features

- Dense and sparse vector storage with exact and approximate nearest-neighbor search.
- Flat, HNSW, HNSW-RaBitQ, IVF, Vamana, and DiskANN indexes.
- L2, inner-product, cosine, and MIPS-L2 metrics with optional quantization and refinement.
- Scalar filtering, block-max WAND BM25 full-text search, grouping, and hybrid multi-query retrieval.
- Durable WAL-backed writes, crash recovery, segment-native incremental indexes, and atomic compaction.
- Pure Go on Linux, macOS, and Windows.

## Install

zvec requires Go 1.26 or later.

```bash
go get github.com/gorse-io/zvec
```

Then import it in your application:

```go
import "github.com/gorse-io/zvec"
```

## Vector storage tutorial

The following program creates a local collection, stores vectors with metadata,
and returns the two nearest documents.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/gorse-io/zvec"
)

func main() {
	ctx := context.Background()

	schema := zvec.NewCollectionSchema("articles",
		zvec.NewField("title", zvec.DataTypeString),
		zvec.NewField("category", zvec.DataTypeString),
		zvec.FieldSchema{
			Name:      "embedding",
			DataType:  zvec.DataTypeVectorFP32,
			Dimension: 3,
			Index:     zvec.NewFlatIndexParams(zvec.MetricTypeCosine),
		},
	)

	collection, err := zvec.CreateAndOpen(
		ctx,
		"./data/articles",
		schema,
		zvec.NewCollectionOptions(),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer collection.Close()

	_, err = collection.Insert(ctx, []zvec.Document{
		{
			PrimaryKey: "go",
			Fields: map[string]any{
				"title":     "The Go Programming Language",
				"category":  "programming",
				"embedding": zvec.VectorFP32{1.0, 0.1, 0.0},
			},
		},
		{
			PrimaryKey: "vector",
			Fields: map[string]any{
				"title":     "Vector Search Fundamentals",
				"category":  "search",
				"embedding": zvec.VectorFP32{0.9, 0.2, 0.1},
			},
		},
		{
			PrimaryKey: "sql",
			Fields: map[string]any{
				"title":     "Database Internals",
				"category":  "database",
				"embedding": zvec.VectorFP32{0.0, 0.2, 1.0},
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	results, err := collection.Query(ctx, zvec.VectorQuery{
		Field:       "embedding",
		DenseVector: zvec.VectorFP32{1.0, 0.0, 0.0},
		TopK:        2,
		Projection: zvec.Projection{
			OutputFields: []string{"title", "category"},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	for _, result := range results {
		fmt.Printf("%s: %s (score %.4f)\n",
			result.PrimaryKey,
			result.Fields["title"],
			result.Score,
		)
	}
}
```

The collection is persisted under `./data/articles`. Reopen it after restarting
your application with:

```go
collection, err := zvec.Open(
    context.Background(),
    "./data/articles",
    zvec.NewCollectionOptions(),
)
```

Use `Insert`, `Upsert`, `Update`, and `Delete` for document mutations. Call
`Flush` to publish an immutable segment and `Optimize` to compact stored data;
WAL-backed writes remain recoverable even without an explicit flush. `Query`
also accepts `PrimaryKey` as a vector target, a single `FTS` clause, or a
filter-only request with no target. `MultiQuery` fuses dense, sparse,
primary-key-vector, and FTS branches over one snapshot.

### Choosing an index

| Index | Best for |
| --- | --- |
| Flat | Exact search and small collections |
| HNSW | General-purpose low-latency ANN search |
| HNSW-RaBitQ | Memory-efficient graph search for larger vectors |
| IVF | Tunable approximate search with list probing |
| Vamana | Graph-based search with deterministic native persistence |
| DiskANN | Disk-backed graph search with bounded node caching |

Dense vectors support FP16 and FP32 storage, plus supported scalar
quantization options. Sparse vectors support exact Flat and HNSW
inner-product search. See the [Collection API](docs/collection-api.md) and
[vector query semantics](docs/vector-query.md) for filters, radius queries,
projections, ANN parameters, grouping, and refinement.

## Documentation

- [Collection API](docs/collection-api.md)
- [Vector query semantics](docs/vector-query.md)
- [Hybrid MultiQuery](docs/multi-query.md)
- [Segment-native indexes](docs/segment-native-indexes.md)
- [Runtime configuration](docs/runtime-config.md)
- [Native Go disk format](docs/disk-format.md)
- [VectorDBBench-compatible Go benchmark](cmd/vector-db-bench/README.md)

## Compatibility

The root `zvec` package is the public API. zvec uses native Go disk format v2
and does not read C++ zvec collection files. Version 1 collections are rejected;
there is no compatibility, migration, fallback, or dual-write path.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
