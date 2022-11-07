# pinecone-client
The Pinecone python client

For more information, see the docs at https://www.pinecone.io/docs/

## Installation

Install a released version from pip:
```shell
pip install pinecone-client
```

Or the latest development version:
```shell
pip install git+https://git@github.com/pinecone-io/pinecone-python-client.git
```

Or a specific development version:
```shell
pip install git+https://git@github.com/pinecone-io/pinecone-python-client.git
pip install git+https://git@github.com/pinecone-io/pinecone-python-client.git@example-branch-name
pip install git+https://git@github.com/pinecone-io/pinecone-python-client.git@259deff
```

## Creating an index

The following example creates an index without a metadata
configuration. By default, Pinecone indexes all metadata.

```python

import pinecone


pinecone.init(api_key="YOUR_API_KEY",
              environment="us-west1-gcp")

pinecone.create_index("example-index", dimension=1024)
```

The following example creates an index that only indexes
the "color" metadata field. Queries against this index
cannot filter based on any other metadata field.

```python
metadata_config = {
    "indexed": ["color"]
}

pinecone.create_index("example-index-2", dimension=1024,
                      metadata_config=metadata_config)
```

## List indexes

The following example returns all indexes in your project.

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

active_indexes = pinecone.list_indexes()
```


