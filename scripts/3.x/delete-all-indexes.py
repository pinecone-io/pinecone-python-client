import random
import string
import os
from pinecone import Pinecone

def read_env_var(name):
    value = os.environ.get(name)
    if value is None:
        raise Exception('Environment variable {} is not set'.format(name))
    return value

def main():
    api_key = read_env_var('PINECONE_API_KEY')
   
    pc = Pinecone(api_key=api_key)

    indexes = pc.list_indexes()
    print(f'Found {len(indexes)} indexes.')
    print(f'Indexes: {indexes}')

    for index in indexes:
        print(f'Deleting index {index.name}...')
        pc.delete_index(index.name)


if __name__ == '__main__':
    main()