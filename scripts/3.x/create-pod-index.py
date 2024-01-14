import os
import random
import string
import time
from pinecone import Pinecone

def read_env_var(name):
    value = os.environ.get(name)
    if value is None:
        raise 'Environment variable {} is not set'.format(name)
    return value

def random_string(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))

def random_embedding_values(dimension=2):
    return [random.random() for _ in range(dimension)]

def write_gh_output(name, value):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'{name}={value}', file=fh)

def main():
    environment = read_env_var('PINECONE_ENVIRONMENT')
    pc = Pinecone(api_key=read_env_var('PINECONE_API_KEY'))
    environment = read_env_var('PINECONE_ENVIRONMENT')
    vectors_to_upsert = int(read_env_var('VECTORS_TO_UPSERT'))

    index_name = read_env_var('INDEX_NAME_PREFIX') + random_string(10)
    pc.create_index(
        name=index_name,
        metric=read_env_var('METRIC'),
        dimension=int(read_env_var('DIMENSION')),
        spec={
            'pod': {
                'environment': environment,
                'source_collection': collection_name,
                'pod_type': 'p1'
            }
        }
    )
    write_gh_output('index_name', index_name)
    time.sleep(60) # More waiting, just in case

    description = pc.describe_index(index_name)
    print(f'Index description: {description}')
    assert description.name == index_name

    assert description.status.state == 'Ready'
    assert description.status.ready == True
    
    idx = pc.Index(name=index_name)
    stats = idx.describe_index_stats()
    print(f'Index stats: {stats}')

    print(f'Beginning upsert of {vectors_to_upsert} vectors to index {index_name}...')
    batch_size = 10
    num_batches = math.floor(vectors_to_upsert / batch_size)
    for _ in range(num_batches):
        vector = random_embedding_values(dimension)
        vecs = [{'id': random_string(10), 'values': vector} for i in range(batch_size)]
        idx.upsert(vectors=vecs)
    print(f'Done upserting.')

    assert stats.dimension == int(read_env_var('DIMENSION'))
    assert stats.metric == read_env_var('METRIC')
    assert stats.total_vector_count == vectors_to_upsert

if __name__ == '__main__':
    main()

