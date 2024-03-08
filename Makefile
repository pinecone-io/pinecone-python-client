SHELL = /bin/bash
mkfile_path := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

test-unit:
	poetry run pytest --cov=pinecone --timeout=120 tests/unit

test-grpc-unit:
	poetry run pytest --cov=pinecone --timeout=120 tests/unit_grpc

make type-check:
	poetry run mypy pinecone --exclude pinecone/core

gcloud-auth:
	docker run -ti --name gcloud-config gcr.io/google.com/cloudsdktool/google-cloud-cli gcloud auth login

clean:
	rm -rf downloads
	rm -rf gen

download:
	mkdir downloads
	docker run --rm \
		--volumes-from gcloud-config \
		-v ${mkfile_path}:/workspace \
		-w /workspace \
		gcr.io/google.com/cloudsdktool/google-cloud-cli \
		gcloud storage cp gs://api-codegen/_latest downloads --recursive

regenerate: clean download
	# Remove problematic query param
	jq 'walk(if type == "array" then map(if type == "object" then select(.name != "filter") else . end) else . end)' downloads/_latest/openapi/pinecone_api.json > tmp.json
	mv tmp.json downloads/_latest/openapi/pinecone_api.json
	
	# Generate new openapi rest client
	rm -rf pinecone/core
	mkdir -p pinecone/core
	docker run --rm -v ${mkfile_path}:/workspace openapitools/openapi-generator-cli:v5.2.0 generate \
		--input-spec /workspace/downloads/_latest/openapi/pinecone_api.json \
		--config /workspace/codegen/openapi/openapi-generator-args.python.json \
		--generator-name python \
		--template-dir /workspace/codegen/openapi/templates5.2.0 \
		--output /workspace/gen/python
	cp -r ${mkfile_path}gen/python/pinecone/core/* ${mkfile_path}pinecone/core/
	
	# Update grpc client
	mkdir -p pinecone/core/grpc/protos
	cp -r ${mkfile_path}downloads/_latest/python/pinecone/data/v1/* ${mkfile_path}pinecone/core/grpc/protos
	sed -i '' 's|pinecone\.data\.v1|pinecone.core.grpc.protos|g' pinecone/core/grpc/protos/vector_service_pb2_grpc.py