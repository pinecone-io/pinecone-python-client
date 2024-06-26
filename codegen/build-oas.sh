#!/bin/bash

set -eux -o pipefail

version='2024-07'
modules=("control" "data")

destination="pinecone/core"

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

	verify_directory_exists "codegen/apis/_build/${version}"
}

verify_file_exists() {
	local filename=$1
	if [ ! -f "$filename" ]; then
		echo "File does not exist at $filename"
		exit 1
	fi
}

verify_directory_exists() {
	local directory=$1
	if [ ! -d "$directory" ]; then
		echo "Directory does not exist at $directory"
		exit 1
	fi
}

generate_client() {
	local module_name=$1

	oas_file="codegen/apis/_build/${version}/${module_name}_${version}.oas.yaml"
	openapi_generator_config="codegen/openapi-config.${module_name}.json"
	template_dir="codegen/python-oas-templates/templates5.2.0"
	
	verify_file_exists $oas_file
	verify_file_exists $openapi_generator_config
	verify_directory_exists $template_dir

	# Cleanup previous build files
	echo "Cleaning up previous build files"
	rm -rf build

	# Generate client module
	docker run --rm -v $(pwd):/workspace openapitools/openapi-generator-cli:v5.2.0 generate \
		--input-spec "/workspace/$oas_file" \
		--generator-name python \
		--config "/workspace/$openapi_generator_config" \
		--output /workspace/build \
		--template-dir "/workspace/$template_dir"

	# Copy the generated module to the correct location
	rm -rf "${destination}/${module_name}"
	cp -r "build/pinecone/core/${module_name}" "${destination}/${module_name}"
}

extract_shared_classes() {
	target_directory="${destination}/shared"
	rm -rf $target_directory
	mkdir -p $target_directory

	# Define the list of shared source files
	sharedFiles=(
		"api_client"
		"configuration" 
		"exceptions" 
		"model_utils" 
		"rest"
	)

	source_directory="${destination}/${modules[0]}/client"

	# Loop through each file we want to share and copy it to the target directory
	for file in "${sharedFiles[@]}"; do
		cp "${source_directory}/${file}.py" "$target_directory"
	done

	# Cleanup shared files in each module
	for module in "${modules[@]}"; do
		source_directory="${destination}/${module}/client"
		for file in "${sharedFiles[@]}"; do
			rm "${source_directory}/${file}.py"
		done
	done

	# Remove the docstring headers that aren't really correct in the 
	# context of this new shared package structure
	find "$target_directory" -name "*.py" -print0 | xargs -0 -I {} sh -c 'sed -i "" "/^\"\"\"/,/^\"\"\"/d" "{}"'

	echo "All shared files have been copied to $target_directory."

	# Adjust import paths in every file
	find "${destination}" -name "*.py" | while IFS= read -r file; do
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

for module in "${modules[@]}"; do
	generate_client $module
done

# Even though we want to generate multiple packages, we
# don't want to duplicate every exception and utility class.
# So we do a bit of surgery to combine the shared files.
extract_shared_classes

# Format generated files
poetry run black "${destination}"
