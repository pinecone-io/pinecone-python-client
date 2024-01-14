import random
import string
import os
import pinecone
import time
import math

def read_env_var(name):
    value = os.environ.get(name)
    if value is None:
        raise Exception('Environment variable {} is not set'.format(name))
    return value

def random_string(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))

def write_gh_output(name, value):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'{name}={value}', file=fh)

def main():
    api_key = read_env_var('PINECONE_API_KEY')
    environment = read_env_var('PINECONE_ENVIRONMENT')
    collection_name_prefix = read_env_var('COLLECTION_NAME_PREFIX')
    index_name = read_env_var('INDEX_NAME')

    print(f'Beginning test with environment {environment} and index {index_name}')

    pinecone.init(
        api_key=api_key,
        environment=environment
    )

    if index_name not in pinecone.list_indexes():
        raise Exception(f'Index {index_name} does not exist. Cannot create collection.')

    collection_name = collection_name_prefix + '-' + random_string(10)
    write_gh_output('collection_name', collection_name)
    print(f'Creating collection {collection_name}...')
    
    pinecone.create_collection(
        name=collection_name,
        source=index_name
    )

    print(f'Waiting for collection {collection_name} to be ready...')
    collection_ready = False
    max_wait = 120
    while not collection_ready and max_wait >= 0:
        try:
            collection_description = pinecone.describe_collection(collection_name)
            print(f'Collection description: {collection_description}')
            if collection_description['status'] == 'Ready':
                collection_ready = True
            else:
                print(f'Collection {collection_name} not ready yet. Waiting...')
                max_wait -= 10
                time.sleep(10)
        except Exception as e:
            print(f'Error while polling for collection {collection_name} status: {e}')
            time.sleep(10)

    if not collection_ready:
        raise Exception(f'Collection {collection_name} not ready after waiting for {max_wait} seconds.')
        
if __name__ == '__main__':
    main()