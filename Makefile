.PHONY: image develop tests tag-and-push docs version package upload upload-spruce license
mkfile_path := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PYPI_USERNAME ?= __token__

image:
	MODULE=pinecone ../scripts/build.sh ./

develop:
	poetry install -E grpc

test-unit:
	@echo "Running tests..."
	poetry run pytest --cov=pinecone --timeout=120 tests/unit -s -vv

test-integration:
	@echo "Running integration tests..."
	PINECONE_ENVIRONMENT="us-east4-gcp" SPEC='{"serverless": {"cloud": "aws", "region": "us-east-1" }}' DIMENSION=2 METRIC='cosine' GITHUB_BUILD_NUMBER='local' poetry run pytest tests/integration

test-grpc-unit:
	@echo "Running tests..."
	poetry run pytest --cov=pinecone --timeout=120 tests/unit_grpc

make type-check:
	poetry run mypy pinecone --exclude pinecone/core

make generate-oas:
	./codegen/build-oas.sh "2024-07"

version:
	poetry version

package:
	poetry build

upload:
	poetry publish --verbose --username ${PYPI_USERNAME} --password ${PYPI_PASSWORD}
