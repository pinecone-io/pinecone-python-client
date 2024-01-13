import random
import string
import os
import time
import math
from pinecone import Pinecone

def read_env_var(name):
    value = os.environ.get(name)
    if value is None:
        raise Exception('Environment variable {} is not set'.format(name))
    return value

def random_embedding_values(dimension=2):
    return [random.random() for _ in range(dimension)]

def random_string(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))

def main():
    api_key = read_env_var('PINECONE_API_KEY')
    environment = read_env_var('PINECONE_ENVIRONMENT')
    index_name = read_env_var('INDEX_NAME')
    vectors_to_upsert = int(read_env_var('VECTORS_TO_UPSERT'))

    print(f'Beginning test with index {index_name}')

    pc = Pinecone(api_key=api_key)

    print(f'Checking if index {index_name} exists...')
    if index_name not in pc.list_indexes().names():
        print(f'Index {index_name} does not exist. Creating...')
        raise Exception('Index does not exist. Found indexes: {}'.format(pc.list_indexes().names()))

    print(f'Checking index description...')
    description = pc.describe_index(index_name)
    print(f'Index description: {description}')
    
    assert description.name == index_name
    assert description.dimension > 0
    assert description.metric != None
    assert description.spec.serverless == None
    assert description.spec.pod != None
    assert description.spec.pod.environment == environment

if __name__ == '__main__':
    main()