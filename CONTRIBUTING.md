# Contributing

## Developing locally with Poetry

[Poetry](https://python-poetry.org/) is a tool that combines [virtualenv](https://virtualenv.pypa.io/en/latest/) usage with dependency management, to provide a consistent experience for project maintainers and contributors who need to develop the pinecone-python-client
as a library.

A common need when making changes to the Pinecone client is to test your changes against existing Python code or Jupyter Notebooks that `pip install` the Pinecone Python client as a library.

Developers want to be able to see their changes to the library immediately reflected in their main application code, as well as to track all changes they make in git, so that they can be contributed back in the form of a pull request.

The Pinecone Python client therefore supports Poetry as its primary means of enabling a consistent local development experience. This guide will walk you through the setup process so that you can:
1. Make local changes to the Pinecone Python client that are separated from your system's Python installation
2. Make local changes to the Pinecone Python client that are immediately reflected in other local code that imports the pinecone client
3. Track all your local changes to the Pinecone Python client so that you can contribute your fixes and feature additions back via GitHub pull requests

### Step 1. Fork the Pinecone python client repository

On the [GitHub repository page](https://github.com/pinecone-io/pinecone-python-client) page, click the fork button at the top of the screen and create a personal fork of the repository:

![Create a GitHub fork of the Pinecone Python client](./pdoc/pinecone-python-client-fork.png)

It will take a few seconds for your fork to be ready. When it's ready, **clone your fork** of the Pinecone python client repository to your machine.

Change directory into the repository, as we'll be setting up a virtualenv from within the root of the repository.

### Step 1. Install Poetry

Visit [the Poetry site](https://python-poetry.org/) for installation instructions.
To use the [Poetry `shell` command](https://python-poetry.org/docs/cli#shell), install the [`shell` plugin](https://github.com/python-poetry/poetry-plugin-shell).

### Step 2. Install dependencies

Run `poetry install -E grpc -E asyncio` from the root of the project. These extra groups for `grpc` and `asyncio` are optional but required to do development on those optional parts of the SDK.

### Step 3. Activate the Poetry virtual environment and verify success

Run `poetry shell` from the root of the project. At this point, you now have a virtualenv set up in this directory, which you can verify by running:

`poetry env info`

You should see something similar to the following output:

```bash
Virtualenv
Python:         3.9.16
Implementation: CPython
Path:           /home/youruser/.cache/pypoetry/virtualenvs/pinecone-fWu70vbC-py3.9
Executable:     /home/youruser/.cache/pypoetry/virtualenvs/pinecone-fWu70vbC-py3.9/bin/python
Valid:          True

System
Platform:   linux
OS:         posix
Python:     3.9.16
Path:       /home/linuxbrew/.linuxbrew/opt/python@3.9
```
If you want to extract only the path to your new virtualenv, you can run `poetry env info --path`

### Step 4. Enable pre-commit hooks.

Run `poetry run pre-commit install` to enable checks to run when you commit so you don't have to find out during your CI run that minor lint issues need to be addressed.

## Common tasks

### Testing: Unit tests

Unit tests do not automatically read environment variables in the `.env` file because some of the tests relate to parsing values from environment variables (`PINECONE_API_KEY`, `PINECONE_ADDITIONAL_HEADERS`, etc) and we don't want values in our `.env` file to impact how these tests execute.

- Unit tests (REST): `poetry run pytest tests/unit`
- Unit tests (GRPC): `poetry run pytest tests/unit_grpc`


### Testing: Integration tests

Integration tests make real calls to Pinecone. They are slow but give the highest level of confidence the client is actually working end to end.

For these tests, you need to make sure you've set values inside of an `.env` file (see `.env.example` for more information). These will be read using `dotenv` when tests are run.

I never run all of these locally in one shot because it would take too long and is generally unnecessary; in CI, the tests are broken up across many different jobs so they can run in parallel. If I see one or a few tests broken in CI, I will run just those tests locally while iterating on the fix.

- Run the tests for a specific part of the SDK (example: index): `poetry run pytest tests/integration/control/resources/index`
- Run the tests in a single file: `poetry run pytest tests/integration/control/resources/index/test_create.py`
- Run a single test `poetry run pytest tests/integration/control/resources/index/test_list.py::TestListIndexes::test_list_indexes_includes_ready_indexes`

### Testing: Module import performance

We don't have automated tests for this, but if you want to check on how efficiently the package can be imported and initialized, you can run code like this:

```sh
poetry run python3 -X importtime -c 'from pinecone import Pinecone; pc = Pinecone(api_key="foo")' 2> import_time.log
```

And then inspect the results with a visualization tool called tuna.

```sh
poetry run tuna import_time.log
```


### Running the ruff linter / formatter

These should automatically trigger if you have enabled pre-commit hooks with `poetry run pre-commit install`. But in case you want to trigger these yourself, you can run them like this:

```sh
poetry run ruff check --fix # lint rules
poetry run ruff format      # formatting
```

If you want to adjust the behavior of ruff, configurations are in `pyproject.toml`.

### Running the type checker

```sh
poetry run mypy pinecone
```

### Consuming API version upgrades and updating generated portions of the client

These instructions can only be followed by Pinecone employees with access to our private APIs repository.

Prerequisites:
- You must be an employee with access to private Pinecone repositories
- You must have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running. Our code generation script uses a dockerized version of the OpenAPI CLI.
- You must have initialized the git submodules under codegen

```sh
git submodule
```

To regenerate the generated portions of the client with the latest version of the API specifications, you need to have Docker Desktop running on your local machine.

Then you run the build script by passing a version, like this:

```sh
./codegen/build-oas.sh 2025-07
```

Commit the generated files which should be mainly placed under `pinecone/core`. Running the type check with `poetry run mypy pinecone` will usually surface breaking changes as a result of things being renamed or modified.

## Installing development versions

If you want to explore a potential code change, investigate
a bug, or just want to try unreleased features, you can also install
specific git shas.

Some example commands:

```shell
pip3 install git+https://git@github.com/pinecone-io/pinecone-python-client.git
pip3 install git+https://git@github.com/pinecone-io/pinecone-python-client.git@example-branch-name
pip3 install git+https://git@github.com/pinecone-io/pinecone-python-client.git@44fc7ed

poetry add git+https://github.com/pinecone-io/pinecone-python-client.git@44fc7ed
```
