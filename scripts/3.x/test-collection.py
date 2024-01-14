import random
import string
import os
from pinecone import Pinecone

def read_env_var(name):
    value = os.environ.get(name)
    if value is None:
        raise Exception('Environment variable {} is not set'.format(name))
    return value

def random_string(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))

def main():
    api_key = read_env_var('PINECONE_API_KEY')
    environment = read_env_var('PINECONE_ENVIRONMENT')
    collection_name = read_env_var('COLLECTION_NAME')
    
    print(f'Beginning test with environment {environment} and collection {collection_name}')

    pc = Pinecone(api_key=api_key)

    print(f'Checking if collection {collection_name} exists...')
    if collection_name not in pc.list_collections().names():
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
    assert description.host != None
    assert len(description.host) > 0
    assert description.status != None
    assert description.status.state == 'Ready'
    assert description.status.ready == True

    print(f'Index description looks correct.')

    host_url = description.host

    idx = pc.Index(name=index_name)

    stats = idx.describe_index_stats()
    print(f'Index stats: {stats}')




if __name__ == '__main__':
    main()