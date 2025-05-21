# Testing the Pinecone SDK

We have a lot of different types of tests in this repository. At a high level, they are structured like this:

```
tests
├── dependency
├── integration
├── perf
├── unit
├── unit_grpc
└── upgrade
```

- `dependency`: These tests are a set of very minimal end-to-end integration tests that ensure basic functionality works to upsert and query vectors from an index. These are rarely run locally; we use them in CI to confirm the client can be used when installed wtih a large matrix of different python versions and versions of key dependencies. See [`.github/workflows/testing-dependency.yaml`](https://github.com/pinecone-io/pinecone-python-client/blob/main/.github/workflows/testing-dependency.yamll) for more details on how these are run.

- `integration`: These are a large suite of end-to-end integration tests exercising most of the core functions of the product. They are slow and expensive to run, but they give the greatest confidence the SDK actually works end-to-end. See notes below on how to setup the required configuration and run individual tests if you are iterating on a bug or feature and want to get more rapid feedback than running the entire suite in CI will give you.

- `perf`: These tests are still being developed. But eventually, they will play an important roll in making sure we don't regress on client performance when building new features.

- `unit` and `unit_grpc`. These are what you would probably expect. Unit-testing makes up a relatively small portion of our testing because there's not that much business logic that makes sense to test in isolation. But it is ocassionally useful when doing some sort of data conversions with many edge cases (e.g. `VectorFactory`) or merging results (e.g. `QueryResultsAggregator`) to write some unit tests. If you have a situation where unit testing is appropriate, they are really great to work with because they are fast and don't have any external dependencies. In CI, these are run with the [`.github/workflows/testing-unit.yaml`](https://github.com/pinecone-io/pinecone-python-client/blob/main/.github/workflows/testing-unit.yaml) workflow.

- `upgrade`: These are also still being developed and if you are reading this guide you probably don't need to worry about them. The goal of these is to ensure we're not introducing breaking changes without realizing it.


## Running the ruff linter / formatter

These should automatically trigger if you have enabled pre-commit hooks with `poetry run pre-commit install`. But in case you want to trigger these yourself, you can run them like this:

```sh
poetry run ruff check --fix # lint rules
poetry run ruff format      # formatting
```

If you want to adjust the behavior of ruff, configurations are in `pyproject.toml`.

## Running the type checker

If you are adding new code, you should make an effort to annotate it with [type hints](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html).

You can run the type-checker to check for issues with:

```sh
poetry run mypy pinecone
```

## Automated tests

### Running unit tests

Unit-testing makes up a relatively small portion of our testing because there's not that much business logic that makes sense to test in isolation. But it is ocassionally useful when doing some sort of data conversions with many edge cases (e.g. `VectorFactory`) or merging results (e.g. `QueryResultsAggregator`) to write some unit tests.

Unit tests do not automatically read environment variables in the `.env` file because some of the tests relate to parsing values from environment variables and we don't want values in our `.env` file to impact how these tests execute.

To run them:

- For REST: `poetry run pytest tests/unit`
- For GRPC: `poetry run pytest tests/unit_grpc`

If you want to set an environment variable anyway, you can do it be prefacing the test command inline. E.g. `FOO='bar' poetry run pytest tests/unit`

### Running integration tests

Integration tests make real calls to Pinecone. They are slow but give the highest level of confidence the client is actually working end to end. **In general, only Pinecone employees should run these because the cost of the creating underlying resources can be quite significant, particularly if errors occur and some resources are not cleaned up properly.**

For these tests, you need to make sure you've set values inside of an `.env` file (see `.env.example` for more information). These will be read using `dotenv` when tests are run.

I never run all of these locally in one shot because it would take too long and is generally unnecessary; in CI, the tests are broken up across many different jobs so they can run in parallel and minimize the amount of retesting when a failure results in the entire build being re-run.

If I see one or a few tests broken in CI, I will run just those tests locally while iterating on the fix:

- Run the tests for a specific part of the SDK (example: index): `poetry run pytest tests/integration/control/resources/index`
- Run the tests in a single file: `poetry run pytest tests/integration/control/resources/index/test_create.py`
- Run a single test `poetry run pytest tests/integration/control/resources/index/test_list.py::TestListIndexes::test_list_indexes_includes_ready_indexes`

### Fixtures and other test configuration

Many values are read from environment variables (from `.env`) or set in CI workflows such as `.github/workflows/testing-integration.yaml`.

At the level of the testing framework, a lot of test fixtures as well as setup & cleanup tasks take place in special files called `conftest.py`. This file name has [special significance](https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files) to pytest and your fixtures won't be loaded if you mispell the name of the file, so be careful if you are setting up a new group of tests that need a `conftest.py` file.

Within a conftest file, a fixture can be defined like this with the `@pytest.fixture` decorator:

```python
@pytest.fixture()
def foo(request):
    return "FOO"
```

Then in the test file, you can refer to the fixture by name in the parameters to your test function:

```python
class MyExampleTest:
    def test_foo(self, foo):
        assert foo == "FOO"
```

This is a highly contrived example, but we use this technique to access test configuration controlled with environment variables and resources that have heavy setup & cleanup cost (e.g. spinning up indexes) that we want to manage in one place rather than duplicating those steps in many places throughout a codebase.

### Testing data plane: REST vs GRPC vs Asyncio

Integration tests for the data plane (i.e. `poetry run pytest tests/integration/data`) are reused for both the REST and GRPC client variants since the interfaces of these different client implementations are nearly identical (other than `async_req=True` responses). To toggle how they are run, set `USE_GRPC='true'` in your `.env` before running.

There are a relatively small number of tests which are not shared, usually related to futures when using GRPC with `async_req=True`. We use `@pytest.mark.skipif` to control whether these are run or not.

```python
class TestDeleteFuture:
    @pytest.mark.skipif(
        os.getenv("USE_GRPC") != "true", reason="PineconeGrpcFutures only returned from grpc client"
    )
    def test_delete_future(self, idx):
        # ... test implementation
```

Asyncio tests of the data plane are unfortunately separate because there are quite a few differences in how you interact with the asyncio client. So those tests are found in a different directory, `tests/integration/data_asyncio`

## Manual testing

### With an interactive REPL

You can access a python REPL that is preloaded with the virtualenv maintained by Poetry (including all dependencies declared in `pyproject.toml`), any changes you've made to the code in `pinecone/`, the environment variables set in your `.env` file, and a few useful variables and functions defined in [`scripts/repl.py`](https://github.com/pinecone-io/pinecone-python-client/blob/main/scripts/repl.py) :

```sh
$ poetry run repl

    Welcome to the custom Python REPL!
    Your initialization steps have been completed.

    Two Pinecone objects are available:
    - pc: Built using the PINECONE_API_KEY env var, if set
    - pcci: Built using the PINECONE_API_KEY_CI_TESTING env var, if set

    You can use the following functions to clean up the environment:
    - delete_all_indexes(pc)
    - delete_all_pod_indexes(pc)
    - delete_all_collections(pc)
    - delete_all_backups(pc)
    - cleanup_all(pc)

>>> pc.describe_index(name='jen')
{
    "name": "jen",
    "metric": "cosine",
    "host": "jen-dojoi3u.svc.preprod-aws-0.pinecone.io",
    "spec": {
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1"
        }
    },
    "status": {
        "ready": true,
        "state": "Ready"
    },
    "vector_type": "dense",
    "dimension": 2,
    "deletion_protection": "disabled",
    "tags": null
}
```

### Investigating module import performance

We don't have automated tests for this, but if you want to do some one-off testing to check on how efficiently the package can be imported and initialized, you can run code like this:

```sh
poetry run python3 -X importtime -c 'from pinecone import Pinecone; pc = Pinecone(api_key="foo")' 2> import_time.log
```

And then inspect the results with a visualization tool called tuna.

```sh
poetry run tuna import_time.log
```

This is a useful thing to do when you are introducing new classes or plugins to ensure you're not causing a performance regression on imports.

### Installing SDK WIP in another project on your machine

pip, poetry, and similar tools know how to install from local files. This can sometimes be useful to validate a change or bugfix.

If your local files look like this:

```
workspace
├── pinecone-python-client/
└── repro_project/
```

You should be able to test changes in your repro project by doing something like

```sh
cd repro_project

# With poetry
poetry add ../pinecone-python-client

# With pip3
pip3 install ../pinecone-python-client
```
