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

    collections = pc.list_collections()
    print(f'Found {len(collections)} collections.')
    print(f'Collections: {collections}')

    for collection in collections:
        print(f'Deleting collection {collection.name}...')
        pc.delete_collection(collection.name)


if __name__ == '__main__':
    main()