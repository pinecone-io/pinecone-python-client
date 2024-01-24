import os
from pinecone import Pinecone

def read_env_var(name):
    value = os.environ.get(name)
    if value is None:
        raise Exception('Environment variable {} is not set'.format(name))
    return value

def main():
    pc = Pinecone(api_key=read_env_var('PINECONE_API_KEY'))

    collections = pc.list_collections().names()
    for collection in collections:
        if collection != "":
            pc.delete_collection(collection)

if __name__ == '__main__':
    main()

