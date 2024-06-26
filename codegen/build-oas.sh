#!/bin/bash

set -eux -o pipefail

version='2024-07'

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

	if [ ! -d "codegen/apis/_build/$version" ]; then
		echo "Version $version does not exist in apis build dir"
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
	local openapi_generator_config=$2
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
		--config "/workspace/$openapi_generator_config" \
		--output /workspace/build \
		--template-dir "/workspace/$template_dir"
}

extract_shared_classes() {
	target_directory="pinecone/core/shared"
	rm -rf $target_directory
	mkdir -p $target_directory

	modules=("control" "data")

	# Define the list of shared source files
	sharedFiles=(
		"api_client"
		"configuration" 
		"exceptions" 
		"model_utils" 
		"rest"
	)

	source_directory="pinecone/core/${modules[0]}/client"

	# Loop through each file we want to share and copy it to the target directory
	for file in "${sharedFiles[@]}"; do
		cp "$source_directory/$file.py" "$target_directory"
	done

	# Cleanup shared files in each module
	for module in "${modules[@]}"; do
		source_directory="pinecone/core/$module/client"
		for file in "${sharedFiles[@]}"; do
			rm "$source_directory/$file.py"
		done
	done

	# Remove the docstring headers that aren't really correct in the 
	# context of this new shared package structure
	find "$target_directory" -name "*.py" -print0 | xargs -0 -I {} sh -c 'sed -i "" "/^\"\"\"/,/^\"\"\"/d" "{}"'

	echo "All shared files have been copied to $target_directory."

	# Adjust import paths in every file
	find pinecone/core -name "*.py" | while IFS= read -r file; do
		sed -i '' 's/from \.\.model_utils/from pinecone\.core\.shared\.model_utils/g' "$file"
		sed -i '' 's/from pinecone\.core\.control\.client import rest/from pinecone\.core\.shared import rest/g' "$file"

		for module in "${modules[@]}"; do
			for sharedFile in "${sharedFiles[@]}"; do
				sed -i '' "s/from pinecone\.core\.$module\.client\.$sharedFile/from pinecone\.core\.shared\.$sharedFile/g" "$file"
			done
		done
	done
}

update_apis_repo
verify_spec_version $version

# Generate data plane client
data_oas="codegen/apis/_build/$version/data_$version.oas.yaml"
data_config="codegen/openapi-config.data.json"
verify_spec_file_exists $data_oas
generate_client $data_oas $data_config
rm -rf pinecone/core/data
cp -r build/pinecone/core/data pinecone/core/data

# Generate control plane client
control_oas="codegen/apis/_build/$version/control_$version.oas.yaml"
control_config="codegen/openapi-config.control.json"
verify_spec_file_exists $control_oas
generate_client $control_oas $control_config
rm -rf pinecone/core/control
cp -r build/pinecone/core/control pinecone/core/control

# Even though we want to generate multiple packages, we
# don't want to duplicate every exception and utility class.
# So we do a bit of surgery to combine the shared files.
extract_shared_classes

# Format generated files
poetry run black pinecone/core
