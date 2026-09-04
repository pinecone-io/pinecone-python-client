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

## Published URLs must keep resolving

The 10.0 restructure moved pages, and every external link written against the
old layout started 404ing. `conf.py`'s `redirects` table brought those URLs
back; `docs/published-urls.txt` and `scripts/check_docs_urls.py` stop it
recurring.

`published-urls.txt` is the inventory of URLs the site serves. After each build
CI checks that every one of them still exists in `_build/html`, and fails if
one has vanished. It is an offline file-existence test: deterministic, no
network, same answer for the same commit.

Deleting or renaming a page is still fine — add an entry to `redirects` in
`conf.py` and sphinx-reredirects writes a forwarding stub at the old path,
which the check accepts in place of the original page.

When you add a page, CI fails until the inventory catches up:

```bash
make html
uv run python scripts/check_docs_urls.py --update
```

Then commit `published-urls.txt`. `--update` also folds in anything the live
sitemap lists that the file is missing; pass `--no-fetch` to skip that.

`--update` only ever adds. Running it after deleting a page won't make the
check pass — the URL stays in the inventory, which is the point. Retiring a URL
for real means deleting its line by hand, and that's only correct for one that
was never publicly reachable.
