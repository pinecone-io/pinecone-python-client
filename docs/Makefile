SPHINXOPTS    ?= -W
SPHINXBUILD   = uv run --extra docs sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean doctest coverage

help:
	@$(SPHINXBUILD) -M help $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS)

html:
	$(SPHINXBUILD) -M html $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)

doctest:
	$(SPHINXBUILD) -M doctest $(SOURCEDIR) $(BUILDDIR)

coverage:
	$(SPHINXBUILD) -M coverage $(SOURCEDIR) $(BUILDDIR)
