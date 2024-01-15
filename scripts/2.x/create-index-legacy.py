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

def random_embedding_values(dimension=2):
    return [random.random() for _ in range(dimension)]

def write_gh_output(name, value):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'{name}={value}', file=fh)

def main():
    api_key = read_env_var('PINECONE_API_KEY')
    environment = read_env_var('PINECONE_ENVIRONMENT')
    index_name_prefix = read_env_var('INDEX_NAME_PREFIX')
    dimension = int(read_env_var('DIMENSION'))
    metric = read_env_var('METRIC')
    vectors_to_upsert = int(read_env_var('VECTORS_TO_UPSERT'))

    index_name = index_name_prefix + '-' + random_string(10)
    write_gh_output('index_name', index_name)

    print(f'Beginning test with environment {environment} and index {index_name}')

    pinecone.init(
        api_key=api_key,
        environment=environment
    )

    if index_name not in pinecone.list_indexes():
        print(f'Creating index {index_name}...')
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric
        )
        print(f'Waiting for index {index_name} to be ready...')
        time.sleep(60)
        print(f'Done waiting.')
    else:
        print(f'Index {index_name} already exists. Skipping create.')

    description = pinecone.describe_index(index_name)
    print(f'Index description: {description}')

    print(f'Beginning upsert of {vectors_to_upsert} vectors to index {index_name}...')
    batch_size = 10
    num_batches = math.floor(vectors_to_upsert / batch_size)
    index = pinecone.Index(index_name)
    for _ in range(num_batches):
        vector = random_embedding_values(dimension)
        vecs = [{'id': random_string(10), 'values': vector} for i in range(batch_size)]
        index.upsert(vectors=vecs)
    print(f'Done upserting.')

    print(f'Beginning query of index {index_name}...')
    index.query(vector=random_embedding_values(dimension), top_k=10)
    print(f'Done querying.')

if __name__ == '__main__':
    main()