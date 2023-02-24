# Changelog

## Unreleased Changes
None
## [2.2.0](https://github.com/pinecone-io/pinecone-python-client/compare/2.1.0...2.2.0)
- Support for Vector `sparse_values`
- Added function `upsert_from_dataframe()` which allows upserting a large dataset of vectors by providing a Pandas dataframe
- Added option to pass vectors to `upsert()` as a list of dictionaries
- Implemented GRPC retry by directly configuring the low-level `grpcio` behavior, instead of wrapping with an interceptor

## [2.1.0](https://github.com/pinecone-io/pinecone-python-client/compare/2.0.13...2.1.0)
- Fix "Connection Reset by peer" error after long idle periods 
- Add typing and explicit names for arguments in all client operations
- Add docstrings to all client operations
- Support batch upsert by passing `batch_size` to `upsert` method
- Improve gRPC query results parsing performance 


## [2.0.13](https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.12...v2.0.13)
- Added support for collections 
  - Users can manage collections using ``create_collection`` , ``describe_collection`` and ``delete_collection`` calls.
  - Users can specify additional ``source_collection`` parameter during index creation to create index from a collection
- The ```scale_index``` call is now deprecated in favor of ```configure_index``` , users can now modify both ``pod_type`` and ```replicas``` on existing indexes.
- Added support for vertical scaling. This can be done by changing ```pod_type ``` via the ```configure_index``` call or during index creation.
- Updated dependency requirements for grpc client.

## [2.0.12](https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.11...v2.0.12)

- Changed grpcio verison to be > 1.44.1
- Sanitized repo by removing leftover files from old versions.
- Added more info to ```describe_index_stats``` call. The call now gives a namespace wise vector count breakdown.

## [2.0.11](https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.10...v2.0.11)
### Changed
- Added support of querying by a single vector.
  - This is a step in deprecating batch queries.
- Added support of querying by vector id.
- Adds support to specify what metadata fields should be indexed on index creation using the ```index_metadata_config``` option.

## [2.0.10](https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.9...v2.0.10)
### Changed
- Added support for deleting vectors by metadata filter. The pinecone.Index.delete() api now accepts an additional filter= parameter which takes metadata filter expression equivalent to what query() supports.
  - Internally these requests are now sent as POST requests, though the previous DELETE api is still supported.
 
## [2.0.9](https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.8...v2.0.9)
### Changed
- Added [update](https://www.pinecone.io/docs/api/operation/update/) API. Allows updates to a vector and it's metadata.
- Added ```state``` to index status. Allows the user to check what state the index is in with the ```describe_index``` call. An index can be in one of the following states:
  - ```Initializing```: Index is getting ready
  - ```Ready```: Index is ready to receive requests
  - ```ScalingUp```: Index is adding replicas, it can still take requests.
  - ```ScalingDown```: Index is scaling down replicas, it can still take requests.
  - ```Terminating```: Index is in the stage of getting deleted.
  
## [2.0.8](https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.7...v2.0.8) - 2022-03-08

### Changed
- Added an ```index_fullnes``` metric in the ```describe_index_stats()``` response.
- Removed ```Sentry``` tracking and dependencies for client errors.

## [2.0.7](https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.6...v2.0.7) - 2022-03-01

### Changed
- Increased maximum length of ids to 512

## [2.0.6](https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.5...v2.0.6) - 2022-02-15

### Changed
- Changed the spec to add  ```pods``` and ```pod_type``` fields to ```create_index``` and ```describe_index```.
- ```pod_type``` is used to select between ```'s1'``` and ```'p1'``` pod types during index creation.
- The field ```pods``` means total number of pods the index will use, ```pods = shards*replicas```.

## [2.0.5](https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.4...v2.0.5) - 2022-01-17

### Changed

- Increased the max vector dimensionality to 20k.

## [2.0.4](https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.3...v2.0.4) - 2021-12-20

### Changed

- Public release of the gRPC flavor of client. The gRPC flavor comes with more dependencies but can give higher upsert speeds on multi node indexes. For more details on the gRPC client, please refer to the [installation](https://www.pinecone.io/docs/installation/) and [usage](https://www.pinecone.io/docs/performance-tuning/#using-the-grpc-client-to-get-higher-upsert-speeds) sections in the docs.
## [2.0.3](https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.2...v2.0.3) - 2021-10-31

### Changed

- Some type validations were moved to the backend for performance reasons. In these cases a 4xx ApiException will be returned instead of an ApiTypeError.

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

## 2.0.0 - 2020-10-04
### Added
- New major release!

### Changed
- `pinecone.create_index()` now requires a `dimension` parameter.
- The `pinecone.Index` interface has changed:
  - `Index.upsert`, `Index.query`, `Index.fetch`, and `Index.delete` now take different parameters and return different results.
  - `Index.info` has been removed. See `Index.describe_index_stats()` as an alternative.
  - The `Index()` constructor no longer validates index existence. This is instead done on all operations executed using the Index instance.

[2.0.2]: https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/pinecone-io/pinecone-python-client/compare/v2.0.0...v2.0.1
