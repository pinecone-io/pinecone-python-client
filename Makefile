SHELL = /bin/bash
mkfile_path := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

gcloud-auth:
	gcloud auth login
	gcloud config set project development-pinecone

clean:
	rm -rf downloads
	rm -rf gen

regenerate: clean
	mkdir downloads
	gcloud storage cp gs://api-codegen/_latest downloads --recursive
	
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
	${mkfile_path}codegen/grpc.sh