# Changelog

## [Unreleased]

### Changed
- `pinecone.init()` can now be used to set the pinecone logger's log level.

## [2.0.0] - 2020-10-04
### Added
- New major release!

### Changed
- `pinecone.create_index()` now requires a `dimension` parameter.
- The `pinecone.Index` interface has changed:
  - `Index.upsert`, `Index.query`, `Index.fetch`, and `Index.delete` now take different parameters and return different results.
  - `Index.info` has been removed. See `Index.describe_index_stats()` as an alternative.
  - The `Index()` constructor no longer validates index existence. This is instead done on all operations executed using the Index instance.


[Unreleased]: https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.1...HEAD
[2.0.1]: https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.0...v2.0.1
