# Pinecone SDK Docs

Sphinx sources for the published SDK reference. This file is in `conf.py`'s
`exclude_patterns`, so it never renders — it's notes for whoever builds the docs.

## Building

Every target runs from this directory, and `uv` is the only supported runner
(there is no poetry setup here). The Makefile wraps `uv run --extra docs
sphinx-build`, so the `docs` extra installs itself on first use:

```bash
cd docs
make html
```

Output lands in `docs/_build/html/index.html`.

Other targets:

| Target | What it does |
| --- | --- |
| `make html` | Full HTML build |
| `make doctest` | Runs the doctests in `pinecone/` docstrings |
| `make coverage` | Reports undocumented objects |
| `make clean` | Removes `_build/` |

## Warnings are errors

`SPHINXOPTS` defaults to `-W` (`--fail-on-warning`), so any warning fails the
build. CI runs the same flag, which means a warning caught locally is a warning
that would have blocked the merge. It reports every warning before failing, so
one run gives you the whole list.

To iterate without the gate, override it: `make html SPHINXOPTS=""`.

If the `--extra docs` resolution is slow or you've already synced the
environment, skip it the way CI does:

```bash
make html SPHINXBUILD="uv run --no-sync sphinx-build"
```

## Doctests

`make doctest` executes the `>>>` examples in the docstrings under `pinecone/`.
Nothing in `docs/**.md` or `docs/**.rst` is executed, so prose-guide snippets are
not covered by it. `conf.py`'s `doctest_global_setup` mocks the HTTP layer and
pre-binds `pc` and `admin`, which is why the examples can make API calls without
credentials.

`autodoc_mock_imports` covers `pinecone._grpc` and `pandas`, so the gRPC client
and the DataFrame helpers are stubs during a build. Examples that need either
one for real won't run under doctest.

## PR previews

Each same-repo PR gets a rendered copy of its docs. After CI's "Sphinx docs
build" job passes, `.github/workflows/docs-preview.yml` deploys the HTML to the
`gh-pages` branch under `pr-preview/pr-<N>/` and posts the URL as a sticky PR
comment. Closing the PR removes the folder. The Pages site is private, so only
people with read access to the repo can open it. Fork and Dependabot PRs get no
preview.

If the deploy fails, download the `docs-html` artifact from the CI run's
summary page instead (kept 7 days).

## Cross-reference checking

`nitpicky` is off in `conf.py` for now, so an unresolved cross-reference does
not fail the `-W` build yet. Check them with:

```bash
uv run --no-sync sphinx-build -E -n -b html . _build/nit -q
```

`conf.py` resolves most targets for you. `msgspec`, `pandas` and the Python
stdlib link through intersphinx. A bare name such as `NotFoundError` or a
`pinecone.X` alias links to the one documented object it names. A name two
documented objects share stays unresolved: qualify it, for example
`` :class:`~pinecone.models.enums.EmbedModel` ``. Private and internal targets
(any `_`-prefixed segment, the HTTP clients, type variables) are ignored. A
public name never goes in `nitpick_ignore`; give it a reference entry instead.
Once the remaining docstring warnings are fixed, set `nitpicky = True` so CI
enforces this.
