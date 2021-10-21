# Changelog

## [2.0.2] - 2021-10-21

### Changed
- The python client `pinecone.config.OpenApiConfiguration` object now uses the certifi package's SSL CA bundle by default. This should fix HTTPS connection errors in certain environments depending on their default CA bundle, including some Google Colab notebooks. 
- A bug causing different index instances to share the same configuration object was fixed.
- Deprecated control via `pinecone.init()` of the pinecone logger's log level and removed the loguru dependency. To control log level now, use the standard library's logging module to manage the level of the "pinecone" logger or its children. 


## [2.0.1] - 2021-10-06
### Added
- New `timeout` parameter to the `pinecone.create_index()` and the `pinecone.delete_index()` call.
  - `timeout` allows you to set how many seconds you want to wait for `create_index()` and `delete_index()` to complete. If `None`, wait indefinitely; if `>=0`, time out after this many seconds; if `-1`, return immediately and do not wait. Defaults to `None`.

### Changed
- Updates the default openapi_config object to use the certifice ssl_ca_cert bundle.
- The python client `pinecone.config.OpenApiConfiguration` object now uses the certifi package's SSL CA bundle by default. This should fix HTTPS connection errors in certain environments depending on their default CA bundle, including some Google Colab notebooks. 

## [2.0.0] - 2020-10-04
### Added
- New major release!

### Changed
- `pinecone.create_index()` now requires a `dimension` parameter.
- The `pinecone.Index` interface has changed:
  - `Index.upsert`, `Index.query`, `Index.fetch`, and `Index.delete` now take different parameters and return different results.
  - `Index.info` has been removed. See `Index.describe_index_stats()` as an alternative.
  - The `Index()` constructor no longer validates index existence. This is instead done on all operations executed using the Index instance.


[2.0.2]: https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.2...v2.0.2
[2.0.1]: https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.0...v2.0.1
