#!/bin/bash

version='2024-04'

rm -rf build
mkdir build

update_apis_repo() {
	echo "Updating apis repo"
	pushd codegen/apis
		git fetch
		git pull
		just build
	popd
}

verify_spec_version() {
	local version=$1
	echo "Verifying spec version $version exists in apis repo"
	if [ -z "$version" ]; then
		echo "Version is required"
		exit 1
	fi

	if [ ! -d "codegen/apis/static/$version" ]; then
		echo "Version $version does not exist in static dir"
		exit 1
	fi
}

verify_spec_file_exists() {
	local oas_file=$1
	if [ ! -f "$oas_file" ]; then
		echo "Spec file does not exist at $oas_file"
		exit 1
	fi
}

generate_client() {
	local oas_file=$1
	local template_dir="codegen/python-oas-templates/templates5.2.0"

	if [ ! -f "$oas_file" ]; then
		echo "Spec file does not exist at $oas_file"
		exit 1
	fi
	if [ ! -d "$template_dir" ]; then
		echo "Template directory does not exist at $template_dir"
		exit 1
	fi

	# Cleanup previous build files
	echo "Cleaning up previous build files"
	rm -rf build

	docker run --rm -v $(pwd):/workspace openapitools/openapi-generator-cli:v5.2.0 generate \
		--input-spec "/workspace/$oas_file" \
		--generator-name python \
		--config /workspace/codegen/openapi-config.json \
		--output /workspace/build \
		--template-dir "/workspace/$template_dir"
}

update_apis_repo
verify_spec_version $version

# Remove old generated files
rm -rf pinecone/core/client
mkdir -p pinecone/core/client

# Generate data plane client
data_oas="codegen/apis/static/$version/data_$version.oas.yaml"
verify_spec_file_exists $data_oas
generate_client $data_oas
cp -r build/pinecone/core/client pinecone/core

# Generate control plane client
control_oas="codegen/apis/static/$version/control_$version.oas.yaml"
verify_spec_file_exists $control_oas
generate_client $control_oas

# The openapi generator isn't really built to handle multiple spec files, 
# so we have to manually merge the generated outputs from data plane and
# control plane.

# 1. Concat data and control models/__init__.py files
cat pinecone/core/client/models/__init__.py >> build/pinecone/core/client/models/__init__.py
# 2. Copy control models files and combined __init__ file into pinecone
cp -r build/pinecone/core/client/models/* pinecone/core/client/models
# 3. Concat data and control apis/__init__.py files
cat pinecone/core/client/apis/__init__.py >> build/pinecone/core/client/apis/__init__.py
# 4. Copy control apis files and combined __init__ file into pinecone
cp -r build/pinecone/core/client/apis/* pinecone/core/client/apis

# Format generated files
poetry run black pinecone/core
