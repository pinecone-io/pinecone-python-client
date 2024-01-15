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

def write_gh_output(name, value):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'{name}={value}', file=fh)

def main():
    environment = read_env_var('PINECONE_ENVIRONMENT')
    pc = Pinecone(api_key=read_env_var('PINECONE_API_KEY'))
    environment = read_env_var('PINECONE_ENVIRONMENT')
    collection_name = read_env_var('COLLECTION_NAME_PREFIX')

    index_name = read_env_var('INDEX_NAME_PREFIX') + '-from-' + collection_name

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
    time.sleep(60)

    description = pc.describe_index(index_name)
    print(f'Index description: {description}')
    assert description.name == index_name

    assert description.status.state == 'Ready'
    assert description.status.ready == True
    
    idx = pc.Index(name=index_name)
    stats = idx.describe_index_stats()
    print(f'Index stats: {stats}')

    assert stats.dimension == int(read_env_var('DIMENSION'))
    assert stats.metric == read_env_var('METRIC')
    assert stats.total_vector_count == read_env_var('TOTAL_VECTOR_COUNT') 

if __name__ == '__main__':
    main()

