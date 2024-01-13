import random
import string
import os
import pinecone
import time

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
    dimension = int(read_env_var('DIMENSION'))
    metric = read_env_var('METRIC')

    print(f'Beginning test with environment {environment} and index {index_name}')

    pinecone.init(
        api_key=api_key,
        environment=environment
    )

    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists')
        pinecone.delete_index(index_name)
    
    pinecone.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric
    )

    print(f'Waiting for index {index_name} to be ready...')
    time.sleep(60)
    print(f'Done waiting.')

    description = pinecone.describe_index(index_name)
    print(f'Index description: {description}')

    print(f'Beginning upsert of 1000 vectors to index {index_name}...')
    index = pinecone.Index(name=index_name)
    for _ in range(100):
        vector = random_embedding_values(dimension)
        vecs = [{'id': random_string(10), 'values': vector} for i in range(10)]
        index.upsert(vectors=[vecs])
    print(f'Done upserting.')

    print(f'Beginning query of index {index_name}...')
    index.query(vector=random_embedding_values(dimension))
    print(f'Done querying.')

if __name__ == '__main__':
    main()