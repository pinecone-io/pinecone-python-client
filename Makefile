.PHONY: image develop tests tag-and-push docs version package upload license
mkfile_path := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PYPI_USERNAME ?= __token__

image:
	MODULE=pinecone ../scripts/build.sh ./

develop:
	poetry install -E grpc

tests:
	@echo "Installing dependencies..."
	poetry install
	@echo "Running tests..."
	# skipping flake8 for now
	poetry run pytest --cov=pinecone --timeout=120 tests/unit

version:
	poetry version

package:
	poetry build

upload:
	poetry publish --verbose --username ${PYPI_USERNAME} --password ${PYPI_PASSWORD}
	
license:
	# Add license header using https://github.com/google/addlicense.
	# If the license header already exists in a file, re-running this command has no effect.
	pushd ${mkfile_path}/pinecone && \
		docker run --rm -it -v ${mkfile_path}/pinecone:/src ghcr.io/google/addlicense:latest -f ./license_header.txt *.py */*.py */*/*.py */*/*/*.py */*/*/*/*.py */*/*/*/*/*.py */*/*/*/*/*/*.py; \
		popd

set-production:
	echo "production" > pinecone/__environment__

set-development:
	echo "" > pinecone/__environment__
