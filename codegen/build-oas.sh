#!/bin/bash

set -ex -o pipefail

# Functions to get version and branch for each module
get_module_version() {
	local module=$1
	case "$module" in
		db_control) echo "2026-01.alpha" ;;
		db_data) echo "2026-01.alpha" ;;
		inference) echo "2025-10" ;;
		oauth) echo "2025-10" ;;
		admin) echo "2025-10" ;;
		*) echo "Unknown module: $module" >&2; exit 1 ;;
	esac
}

get_module_branch() {
	local module=$1
	# All modules use the fts branch since it contains both alpha and stable specs
	echo "jhamon/fts"
}

destination="pinecone/core/openapi"
modules=("db_control" "db_data" "inference" "oauth" "admin")
py_module_name="core"
template_dir="codegen/python-oas-templates/templates5.2.0"

build_dir="build"

# Track the current branch to avoid unnecessary rebuilds
current_apis_branch=""

checkout_and_build_apis() {
	local branch=$1

	if [ "$current_apis_branch" == "$branch" ]; then
		echo "Already on branch $branch, skipping checkout"
		return
	fi

	echo "Checking out and building apis repo on branch: $branch"
	pushd codegen/apis
		git fetch origin
		git checkout "$branch"
		git pull origin "$branch"
		just clean
		just build
	popd

	current_apis_branch="$branch"
}

update_templates_repo() {
	echo "Updating templates repo"
	pushd codegen/python-oas-templates
		git fetch
		git checkout jhamon/core-sdk
		git pull
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
	local version=$2

	oas_file="codegen/apis/_build/${version}/${module_name}_${version}.oas.yaml"
	package_name="pinecone.${py_module_name}.openapi.${module_name}"

	verify_file_exists $oas_file
	verify_directory_exists $template_dir

	# Cleanup previous build files
	echo "Cleaning up previous build files"
	rm -rf "${build_dir}"

	# Generate client module
	docker run --rm -v $(pwd):/workspace openapitools/openapi-generator-cli:v5.2.0 generate \
		--input-spec "/workspace/$oas_file" \
		--generator-name python \
		--additional-properties=packageName=$package_name,pythonAttrNoneIfUnset=true,exceptionsPackageName=pinecone.openapi_support.exceptions \
		--output "/workspace/${build_dir}" \
		--template-dir "/workspace/$template_dir"

	# Hack to prevent coercion of strings into datetimes within "object" types while still
	# allowing datetime parsing for fields that are explicitly typed as datetime
	find "${build_dir}" -name "*.py" | while IFS= read -r file; do
		sed -i '' "s/bool, date, datetime, dict, float, int, list, str, none_type/bool, dict, float, int, list, str, none_type/g" "$file"
	done

	# Fix invalid dict type annotations in return types and casts
	# Replace {str: (bool, dict, float, int, list, str, none_type)} with Dict[str, Any]
	find "${build_dir}" -name "*.py" | while IFS= read -r file; do
		# Need to escape the braces and parentheses for sed
		sed -i '' 's/{str: (bool, dict, float, int, list, str, none_type)}/Dict[str, Any]/g' "$file"
	done

	# Remove globals() assignments from TYPE_CHECKING blocks
	# These should only be in lazy_import() functions, not in TYPE_CHECKING blocks
	find "${build_dir}" -name "*.py" | while IFS= read -r file; do
		python3 <<PYTHON_SCRIPT
import sys

with open('$file', 'r') as f:
    lines = f.readlines()

in_type_checking = False
output_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if 'if TYPE_CHECKING:' in line:
        in_type_checking = True
        output_lines.append(line)
        i += 1
        # Skip blank line after 'if TYPE_CHECKING:' if present
        if i < len(lines) and lines[i].strip() == '':
            i += 1
        # Process lines until we hit a blank line or 'def lazy_import'
        while i < len(lines):
            next_line = lines[i]
            stripped = next_line.strip()
            if stripped == '' or stripped.startswith('def lazy_import'):
                in_type_checking = False
                break
            # Only include lines that are imports, not globals() assignments
            if not stripped.startswith('globals('):
                output_lines.append(next_line)
            i += 1
        continue
    output_lines.append(line)
    i += 1

with open('$file', 'w') as f:
    f.writelines(output_lines)
PYTHON_SCRIPT
	done

	# Remove unused type: ignore[misc] comments from __new__ methods
	# The explicit type annotation is sufficient for mypy
	find "${build_dir}" -name "*.py" | while IFS= read -r file; do
		sed -i '' 's/instance: T = super().__new__(cls, \*args, \*\*kwargs)  # type: ignore\[misc\]/instance: T = super().__new__(cls, *args, **kwargs)/g' "$file"
	done

	# Fix ApplyResult import - move from TYPE_CHECKING to runtime import
	# ApplyResult is used in cast() calls which need it at runtime
	find "${build_dir}" -name "*_api.py" | while IFS= read -r file; do
		python3 <<PYTHON_SCRIPT
with open('$file', 'r') as f:
    lines = f.readlines()

# Check if ApplyResult is imported under TYPE_CHECKING
apply_result_in_type_checking = False
apply_result_line_idx = -1
typing_import_idx = -1
type_checking_start_idx = -1
output_lines = []
i = 0

while i < len(lines):
    line = lines[i]

    # Find typing import line
    if 'from typing import' in line and typing_import_idx == -1:
        typing_import_idx = len(output_lines)
        output_lines.append(line)
        i += 1
        continue

    # Check for TYPE_CHECKING block
    if 'if TYPE_CHECKING:' in line:
        type_checking_start_idx = len(output_lines)
        output_lines.append(line)
        i += 1
        # Check next line for ApplyResult import
        if i < len(lines) and 'from multiprocessing.pool import ApplyResult' in lines[i]:
            apply_result_in_type_checking = True
            apply_result_line_idx = i
            i += 1  # Skip the ApplyResult import line
            # Skip blank line if present
            if i < len(lines) and lines[i].strip() == '':
                i += 1
            # Check if TYPE_CHECKING block is now empty
            if i < len(lines):
                next_line = lines[i]
                # If next line is not indented, the TYPE_CHECKING block is empty
                if next_line.strip() and not (next_line.startswith(' ') or next_line.startswith('\t')):
                    # Remove the empty TYPE_CHECKING block
                    output_lines.pop()  # Remove 'if TYPE_CHECKING:'
                    type_checking_start_idx = -1
            continue

    output_lines.append(line)
    i += 1

# If we found ApplyResult under TYPE_CHECKING, add it after typing import
if apply_result_in_type_checking and typing_import_idx != -1:
    # Check if it's not already imported at module level
    module_start = ''.join(output_lines[:typing_import_idx+10])
    if 'from multiprocessing.pool import ApplyResult' not in module_start:
        output_lines.insert(typing_import_idx + 1, 'from multiprocessing.pool import ApplyResult\n')

with open('$file', 'w') as f:
    f.writelines(output_lines)
PYTHON_SCRIPT
	done

	# Copy the generated module to the correct location
	rm -rf "${destination}/${module_name}"
	mkdir -p "${destination}"
	cp -r "build/pinecone/$py_module_name/openapi/${module_name}" "${destination}/${module_name}"
	echo "API_VERSION = '${version}'" >> "${destination}/${module_name}/__init__.py"
}

remove_shared_classes() {
	# Define the list of shared source files
	sharedFiles=(
		"api_client"
		"configuration"
		"exceptions"
		"model_utils"
		"rest"
	)

	source_directory="${destination}/${modules[0]}"

	# Cleanup shared files in each module
	for module in "${modules[@]}"; do
		source_directory="${destination}/${module}"
		for file in "${sharedFiles[@]}"; do
			rm "${source_directory}/${file}.py"
		done
	done
}

# Generated Python code attempts to internally map OpenAPI fields that begin
# with "_" to a non-underscored alternative. Along with a polymorphic object,
# this causes collisions and headaches. We massage the generated models to
# maintain the original field names from the OpenAPI spec and circumvent
# the remapping behavior as this is simpler for now than creating a fully
# custom java generator class.
clean_oas_underscore_manipulation() {
	temp_file="$(mktemp)"

	db_data_destination="${destination}/db_data"

	# Only run if the files exist (they may not in alpha spec)
	if [ -f "${db_data_destination}/model/upsert_record.py" ]; then
		sed -i '' \
		-e "s/'id'/'_id'/g" \
		-e 's/self.id/self._id/g' \
		-e 's/id (/_id (/g' \
		-e 's/= id/= _id/g' \
		-e 's/id,/_id,/g' \
		-e "s/'vector\'/'_vector'/g" \
		-e "s/'embed\'/'_embed'/g" \
		-e 's/vector (/_vector (/g' \
		-e 's/embed (/_embed (/g' \
		"${db_data_destination}/model/upsert_record.py"
	fi

	if [ -f "${db_data_destination}/model/hit.py" ]; then
		sed -i '' \
		-e "s/'id'/'_id'/g" \
		-e "s/'score'/'_score'/g" \
		-e 's/ id, score,/ _id, _score,/g' \
		-e 's/id (/_id (/g' \
		-e 's/score (/_score (/g' \
		-e 's/self.id/self._id/g' \
		-e 's/self.score/self._score/g' \
		-e 's/= id/= _id/g' \
		-e 's/= score/= _score/g' \
		"${db_data_destination}/model/hit.py"
	fi
}

# Update templates repo (same for all modules)
update_templates_repo

# Create destination directory if it doesn't exist
mkdir -p "${destination}"

# Track which branches and versions have been processed
processed_branches=""
processed_versions=""

for module in "${modules[@]}"; do
	branch=$(get_module_branch "$module")
	version=$(get_module_version "$module")

	# Checkout and build if we haven't processed this branch yet
	case "$processed_branches" in
		*"$branch"*) ;;  # Already processed
		*)
			checkout_and_build_apis "$branch"
			processed_branches="$processed_branches $branch"
			;;
	esac

	# Verify spec version exists if we haven't already
	case "$processed_versions" in
		*"$version"*) ;;  # Already verified
		*)
			verify_spec_version "$version"
			processed_versions="$processed_versions $version"
			;;
	esac

	# Generate client for this module
	generate_client "$module" "$version"
done

clean_oas_underscore_manipulation

# Write api_version.py using db_data version (for gRPC compatibility)
# The db_data version is used since gRPC is data plane only
db_data_version=$(get_module_version "db_data")
version_file="pinecone/openapi_support/api_version.py"
echo "# This file is generated by codegen/build-oas.sh" > $version_file
echo "# Do not edit this file manually." >> $version_file
echo "# For REST APIs, use the API_VERSION from each module's __init__.py" >> $version_file
echo "# This version is used by gRPC clients (data plane only)" >> $version_file
echo "" >> $version_file

echo "API_VERSION = '${db_data_version}'" >> $version_file
echo "APIS_REPO_SHA = '$(git rev-parse :codegen/apis)'" >> $version_file

# Even though we want to generate multiple packages, we
# don't want to duplicate every exception and utility class.
# So we do a bit of surgery to find these shared files
# elsewhere, in the pinecone.openapi_support package.
remove_shared_classes

# Format generated files
uv run ruff format "${destination}"

rm -rf "$build_dir"
