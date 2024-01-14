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

    print(f'Checking collection description...')
    description = pc.describe_collection(collection_name)
    print(f'Collection description: {description}')
    
    assert description.name == collection_name
    assert description.status.state == 'Ready'
    assert description.status.ready == True



if __name__ == '__main__':
    main()