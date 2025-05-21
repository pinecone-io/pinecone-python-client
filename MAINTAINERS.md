# Maintainers

This guide is aimed primarily at Pinecone employees working on maintaining and developing the python SDK.

## Setup

### 1. Clone the repo

```sh
git clone git@github.com:pinecone-io/pinecone-python-client.git
```

### 2. Install Poetry

Visit [the Poetry site](https://python-poetry.org/docs/#installation) for installation instructions.

### 3. Install dependencies

Run this from the root of the project.

```sh
poetry install -E grpc -E asyncio
```

These extra groups for `grpc` and `asyncio` are optional but required to do development on those optional parts of the SDK.

### 4. Enable pre-commit hooks

Run `poetry run pre-commit install` to enable checks to run when you commit so you don't have to find out during your CI run that minor lint issues need to be addressed.

### 5. Setup environment variables

Some tests require environment variables to be set in order to run.

```sh
cp .env.example .env
```

After copying the template, you will need to fill in your secrets. `.env` is in `.gitignore`, so there's no concern about accidentally committing your secrets.

### Testing

There is a lot to say about testing the Python SDK. See the [testing guide](./docs/maintainers/testing-guide.md).

### Consuming API version upgrades and updating generated portions of the client

These instructions can only be followed by Pinecone employees with access to our private APIs repository.

Prerequisites:
- You must be an employee with access to private Pinecone repositories
- You must have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running. Our code generation script uses a dockerized version of the OpenAPI CLI.
- You must have initialized the git submodules under codegen

First create a prerelease branch to hold work for the upcoming release. For example, for 2025-04 release I worked off of this branch:

```
git checkout main
git pull
git checkout release-candidate/2025-04
git push origin release-candidate/2025-04
```

The release-candidate branch is where we will integrate all changes for an upcoming release which may include work from many different PRs and commits.

Next, to regenerate, I make a second branch to hold my changes

```sh
git checkout jhamon/regen-2025-04
```

Then you run the build script by passing a version, like this:

```sh
./codegen/build-oas.sh 2025-07
```

For grpc updates, it's a similar story:

```sh
./codegen/build-grpc.sh 2025-07
```

Commit the generated files which should be mainly placed under `pinecone/core`. Commit the sha changes in the git submodule at `codegen/apis`.

Running the type check with `poetry run mypy pinecone` will usually surface breaking changes as a result of things being renamed or modified.

Push your branch (`git push origin jhamon/regen-2025-04` in this example) and open a PR against the RC branch (in this example `release-candidate/2025-04`). This will allow the full PR test suite to kick off and help you discover what other changes you need to make.
