import os
import yaml
from string import Template
import httpx

def extract_path(spec):
    if len(spec.items()) != 1:
        raise ValueError('spec.yaml must contain exactly one path')
    for key, _ in spec.items():
        return key
    
def extract_http_method(spec, url_path):
    if len(spec[url_path].items()) != 1:
        raise ValueError('spec.yaml must contain exactly one REST operation')
    for key, _ in spec[url_path].items():
        return key
    
def extract_operationId(spec, url_path, http_method):
    if 'operationId' not in spec[url_path][http_method]:
        raise ValueError('operationId must be present in spec.yaml')
    return spec[url_path][http_method]['operationId']

def format_url(url_path):
    return "f'" + url_path + "'"

def build_validations(propName, propConfig, required_props, validations):
    if propConfig['type'] == 'string' and 'minLength' in propConfig:
        if propName in required_props:
            validations.append((f'if len({propName}) < {propConfig["minLength"]}', f'raise PineconeApiValueError("{propName} must be at least {propConfig["minLength"]} characters long")'))
        else:
            validations.append((f'if {propName} is not None and len({propName}) < {propConfig["minLength"]}', f'raise PineconeApiValueError("{propName} must be at least {propConfig["minLength"]} characters long")'))

    if propConfig['type'] == 'string' and 'maxLength' in propConfig:
        if propName in required_props:
            validations.append((f'if len({propName}) > {propConfig["maxLength"]}', f'raise PineconeApiValueError("{propName} must be at most {propConfig["maxLength"]} characters long")'))
        else:
            validations.append((f'if {propName} is not None and len({propName}) > {propConfig["maxLength"]}', f'raise PineconeApiValueError("{propName} must be at most {propConfig["maxLength"]} characters long")'))

    if propConfig['type'] == 'integer' and 'minimum' in propConfig:
        if propName in required_props:
            validations.append((f'if {propName} < {propConfig["minimum"]}', f'raise PineconeApiValueError("{propName} must be at least {propConfig["minimum"]}")'))
        else:
            validations.append((f'if {propName} is not None and {propName} < {propConfig["minimum"]}', f'raise PineconeApiValueError("{propName} must be at least {propConfig["minimum"]}")'))

    if propConfig['type'] == 'integer' and 'maximum' in propConfig:
        if propName in required_props:
            validations.append((f'if {propName} > {propConfig["maximum"]}', f'raise PineconeApiValueError("{propName} must be at most {propConfig["maximum"]}")'))
        else:
            validations.append((f'if {propName} is not None and {propName} > {propConfig["maximum"]}', f'raise PineconeApiValueError("{propName} must be at most {propConfig["maximum"]}")'))

    if propConfig['type'] == 'string' and 'enum' in propConfig:
        if propName in required_props:
            validations.append((f'if {propName} not in {propConfig["enum"]}', f'raise PineconeApiValueError("{propName} must be one of {propConfig["enum"]}")'))
        else:
            validations.append((f'if {propName} is not None and {propName} not in {propConfig["enum"]}', f'raise PineconeApiValueError("{propName} must be one of {propConfig["enum"]}")'))

type_translation = {
    'string': 'str',
    'integer': 'int',
    'boolean': 'bool',
    'array': 'List',
    'object': 'Dict',
    'number': 'float'
}

def build_type_checks(propName, propConfig, required_props, type_checks):
    configType = propConfig['type']
    pythonType = type_translation[configType]

    if propName in required_props:
        type_checks.append((f'if not isinstance({propName}, {pythonType})', f'raise PineconeApiTypeError("{propName} must be a {pythonType}")'))
    else:
        type_checks.append((f'if {propName} is not None and not isinstance({propName}, {pythonType})', f'raise PineconeApiTypeError("{propName} must be a {pythonType}")'))

def generate_method_from_spec(spec):
    url_path = extract_path(spec)
    http_method = extract_http_method(spec, url_path)
    method_name = extract_operationId(spec, url_path, http_method)

    # Parameters can be in path, query, header, or body when the request is ultimately made.
    # All of these variables will need to be aggregated into method kwargs and then passed
    # to the request method.
    required_method_kwargs = []
    optional_method_kwargs = []

    if 'parameters' in spec[url_path][http_method]:
        parameters = spec[url_path][http_method]['parameters']
        for param in parameters:
            if param['required']:
                required_method_kwargs.append((param['name'], param['schema']['type']))
            else:
                optional_method_kwargs.append((param['name'], param['schema']['type']))
    else:
        parameters = []

    formatted_url = format_url(url_path)

    validations = []
    type_checks = []
    request_body_params = []
    if 'requestBody' in spec[url_path][http_method]:
        required_props = spec[url_path][http_method]['requestBody']['content']['application/json']['schema']['required']
        props = spec[url_path][http_method]['requestBody']['content']['application/json']['schema']['properties']

        for propName, propConfig in props.items():
            request_body_params.append(propName)
            if propName in required_props:
                required_method_kwargs.append((propName, propConfig['type']))
            else:
                optional_method_kwargs.append((propName, propConfig['type']))
            build_validations(propName, propConfig, required_props, validations)
            build_type_checks(propName, propConfig, required_props, type_checks)

    formatted_validations = '\n'.join([f'    {condition}:\n        {action}' for condition, action in validations])
    formatted_type_checks = '\n'.join([f'    {condition}:\n        {action}' for condition, action in type_checks])

    required_method_params = [f'{name}: {type_translation[type]}' for name, type in required_method_kwargs]
    optional_method_params = [f'{name}: Optional[{type_translation[type]}] = None' for name, type in optional_method_kwargs]
    method_params = ', '.join(['http_client: httpx.Client'] + required_method_params + optional_method_params)

    responses_dict = {}
    responses = spec[url_path][http_method]['responses']
    for key, value in responses.items():
        responses_dict[key] = value['description']
    expected_responses = ', \n'.join([f'    "{key}": "{value}"' for key, value in responses_dict.items()])

    preamble = [
        el for el in [
            formatted_type_checks, 
            formatted_validations, 
            '\n    request_body = {' + ', '.join([f'"{name}": {name}' for name in request_body_params]) + '}'
        ] if el != ''
    ]
    print(len(preamble))

    template = Template('''from pinecone.exceptions import PineconeApiException, PineconeApiTypeError, PineconeApiValueError
from typing import List, Dict, Optional
import httpx

EXPECTED_RESPONSES = {
$expected_responses
}

def generated_$method_name($method_parameters):
$preamble
    if request_body == {}:
        r = http_client.request('$http_method', $path)
    else: 
        r = http_client.request('$http_method', $path, json=request_body)

    if r.status_code < 300:
        return r.json()
    elif str(r.status_code) in EXPECTED_RESPONSES:
        raise PineconeApiException(f'HTTP Status: {r.status_code}. Message: {EXPECTED_RESPONSES[str(r.status_code)]} ')
    else:
        raise Exception(f'Unknown Error: {r.status_code} {r.text}')
    
''')

    return "generated_" + method_name, template.substitute(
        method_name=method_name,
        http_method=http_method,
        path=formatted_url,
        method_parameters=method_params,
        preamble="\n".join(preamble),
        expected_responses=expected_responses
    )


import hashlib
import shutil
from datetime import datetime

def calculate_file_hash(filename):
    """Compute the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filename, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def save_hashes_to_file(file_paths, sums_file_path):
    """Calculate and save hashes of multiple files to a sums file."""
    with open(sums_file_path, 'w') as sums_file:
        sums = {}
        for path in file_paths:
            sums[path] = calculate_file_hash(path)
        
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        metadata = {
            'files': sums,
            'generated_at': formatted_time,
            'generated_by': os.environ['USER'] or 'unknown'
        }
        yaml.dump(metadata, sums_file)

def verify_hashes(hashes_file):
    """Verify hashes of multiple files against a sums file."""
    with open(hashes_file, 'r') as hf:
        metadata = yaml.safe_load(hf)
        for path, hash in metadata['files'].values():
            if hash != calculate_file_hash(path):
                raise ValueError(f'Hash mismatch for file {path}')


def main():
    features = [
        'pinecone/features/create_index',
        'pinecone/features/describe_index',
        'pinecone/features/list_indexes',
        'pinecone/features/delete_index',
        'pinecone/features/configure_index',
    ]

    for feature in features:
        # Clean up old generated files
        build_folder = 'generated'
        if os.path.exists(f'{feature}/{build_folder}'):
            shutil.rmtree(f'{feature}/{build_folder}')
        os.mkdir(f'{feature}/{build_folder}')

        with open(f'{feature}/spec.yaml', 'r') as file:
            spec = yaml.safe_load(file)
            method_name, endpointpy = generate_method_from_spec(spec)
            initpy = f'from .endpoint import {method_name}\n\n'
 
        with open(f'{feature}/{build_folder}/endpoint.py', 'w') as f:
            f.write(endpointpy)
        with open(f'{feature}/{build_folder}/__init__.py', 'w') as f:
            f.write(initpy)

        # Save the hash of the generated files
        files = [
            f'{feature}/spec.yaml', 
            f'{feature}/{build_folder}/endpoint.py', 
            f'{feature}/{build_folder}/__init__.py'
        ]
        save_hashes_to_file(files, f'{feature}/{build_folder}/hashes.yaml')
        



if __name__ == '__main__':
    main()