.PHONY: image develop tests tag-and-push docs version package upload license
mkfile_path := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PYPI_USERNAME ?= __token__

image:
	MODULE=pinecone ../scripts/build.sh ./

develop:
	pip3 install -e .[grpc]

tests:
	# skipping flake8 for now
	pip3 install --upgrade --quiet tox && tox -p 4 -e py36,py37,py38,py39

docs:
	echo skipping temporarily...
	# pip3 install --upgrade --quiet tox && tox -e docs

version:
	python3 setup.py --version

package:
	pip3 install -U wheel && python3 setup.py sdist bdist_wheel

upload:
	pip3 install --upgrade --quiet twine && \
	twine upload \
		--verbose \
		--non-interactive \
		--username ${PYPI_USERNAME} \
		--password ${PYPI_PASSWORD} \
		dist/* 

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

gen-openapi:
	docker run --rm -v "${mkfile_path}:/local" openapitools/openapi-generator-cli:v5.2.0 generate --input-spec /local/specs/pinecone_api.json --config /local/codegen-src/openapi-generator-args.python.json --generator-name python --template-dir /local/codegen-src/templates --output /local/openapi-gen
	cp -r ${mkfile_path}/openapi-gen/pinecone/core/client/ ${mkfile_path}/pinecone/core/client/
	rm -r ${mkfile_path}/openapi-gen
