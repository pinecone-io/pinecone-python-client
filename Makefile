.PHONY: image develop tests tag-and-push docs version package upload upload-spruce license
mkfile_path := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PYPI_USERNAME ?= __token__

image:
	MODULE=pinecone ../scripts/build.sh ./

develop:
	uv sync --extra grpc

test-unit:
	@echo "Running tests..."
	uv run pytest --cov=pinecone --timeout=120 tests/unit -s -vv

test-integration:
	@echo "Running integration tests..."
	PINECONE_ENVIRONMENT="us-east4-gcp" SPEC='{"serverless": {"cloud": "aws", "region": "us-east-1" }}' DIMENSION=2 METRIC='cosine' GITHUB_BUILD_NUMBER='local' uv run pytest tests/integration

test-grpc-unit:
	@echo "Running tests..."
	uv run pytest --cov=pinecone --timeout=120 tests/unit_grpc

type-check:
	uv run mypy pinecone --exclude pinecone/core

generate-oas:
	./codegen/build-oas.sh "2024-07"

version:
	@python -c "import re; print(re.search(r'version = \"([^\"]+)\"', open('pyproject.toml').read()).group(1))"

package:
	uv build

upload:
	uv publish --username ${PYPI_USERNAME} --password ${PYPI_PASSWORD}
