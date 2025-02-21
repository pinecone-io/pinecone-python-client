# Pinecone Python Client (DEPRECATED)

The official Pinecone python package has been renamed from `pinecone-client` to `pinecone`. To upgrade, please
**remove** `pinecone-client` from your project dependencies and then **add** the `pinecone` package to get the
latest updates.

> [!WARNING]
> Failure to remove `pinecone-client` before installing `pinecone` can lead to confusing interactions
> between the two packages.

```sh
pip uninstall pinecone-client
pip install pinecone
```

Or, if you are using grpc:

```sh
pip uninstall pinecone-client
pip install "pinecone[grpc]"
```

## Links

- `pinecone` on [PyPI](https://pypi.org/project/pinecone/)
- [Source on Github](https://github.com/pinecone-io/pinecone-python-client)
