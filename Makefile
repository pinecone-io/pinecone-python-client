.PHONY: image develop tests tag-and-push docs version package upload license

PYPI_USERNAME ?= __token__

image:
	MODULE=pinecone ../scripts/build.sh ./

develop:
	pip3 install -e .

tests:
	# skipping flake8 for now
	pip3 install --upgrade --quiet tox && tox -e py38

docs:
	pip3 install --upgrade --quiet tox && tox -e docs

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
	pushd pinecone && \
		addlicense -f ../license_header.txt *.py */*.py */*/*.py */*/*/*.py */*/*/*/*.py */*/*/*/*/*.py */*/*/*/*/*/*.py && \
		popd

set-production:
	echo "production" > pinecone/__environment__

set-development:
	echo "" > pinecone/__environment__