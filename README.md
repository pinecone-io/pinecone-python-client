# Release Python client

Release the python client to Pypi, update documentation at https://docs.beta.pinecone.io/en/latest/index.html, and buid and push docker image to the beta environment repo.
This is automated by tagging a commit with `*-release`, which triggers the github workflow `release.yml`.

**Note**: make sure that the commit passed all tests before you proceed.

```
# Get current version:
make version

# Tag commit
git tag V0.1.2-release

# Push tag
git push origin V0.1.2-release
```

# Alpha release

Release the python client to test.pypi.org, update documentation at https://docs.alpha.pinecone.io/en/latest/index.html.
This is automated by tagging a commit with `*-alpha`, which triggers the github workflow `release.alpha.yml`.

```
# Get current version:
make version

# Tag commit
git tag V0.1.2-alpha

# Push tag
git push origin V0.1.2-alpha
```

# Python client development guide (`0.8.x`)

## Docs

The Python client reference documentation is auto-generated from Python docstrings using Sphinx with the ReadTheDocs theme. We use the [standard Sphinx format](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html). E.g.:

```
"""[Summary]

:param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
:type [ParamName]: [ParamType](, optional)
...
:raises [ErrorType]: [ErrorDescription]
...
:return: [ReturnDescription]
:rtype: [ReturnType]
"""
```

### Customizations

A custom CSS theming file is used, and it is currently dynamically loaded from the main website: https://optimistic-curran-b817a8.netlify.app/css/pinecone-docs.css

### Exclusions

Some legacy files have the `.exclude` extension. They are kept in the repository for reference only, and they are excluded during Sphinx compilation.

## Python client

The following files are the main entry points:

- `__init__.py` - Python client initialization and module imports.
- `manage.py` - for managing Pinecone services.
- `index.py` - for instantiating an `Index` object to interact with a live Pinecone index.
- `constants.py` - for managing constants.

### Managing Pinecone services

`manage.py` is a wrapper for `service.py` and `graph.py`. `service.py` and `graph.py` are kept in the client for backward compatibility. For historical reasons, a Pinecone index is also called a _Service_, whose specification is called a _Graph_, signifying the graphical nature of different components within a service.

`service.py` contains convenience functions for interacting with the _Controller API_. The returned values of those functions consist of the HTTP response and a progressbar.

### Interactive with a Pinecone index

`index.py` works with `grpc.py` to support `upsert`, `query`, `delete`, `fetch`, and `info` operations of the a Pinecone index. `grpc.py` is a generic module for generating gRPC requests and parsing gRPC response.

### API endpoints

- `api_action.py` corresponds to the `actions` APIs in the controller.
- `api_controller.py` corresponds to the `services` APIs in the controller.
- [**deprecated**] `api_hub.py` corresponds to the model hub APIs in the hub.
- [**deprecated**] `api_router.py` corresponds to the `routers` APIs in the controller.

### The Deprecated

- `cli.py` - Pyhton CLI for setting up the environment.
- `connector.py` - connects to gRPC, generates requests, parse responses.
- `hub.py` - convenience functions for creating and uploading Docker images to be used as preprocessors and postprocessors for an index.
- `router.py` - traffic routers for zero-downtime index switching.
