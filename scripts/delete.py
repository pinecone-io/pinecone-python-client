import os
from pinecone import Pinecone

def read_env_var(name):
    value = os.environ.get(name)
    if value is None:
        raise 'Environment variable {} is not set'.format(name)
    return value

def main():
    pc = Pinecone(api_key=read_env_var('PINECONE_API_KEY'))
    to_delete = read_env_var('INDEX_NAME')
    pc.delete_index(name=to_delete)
    print('Index deleted: ' + to_delete)

if __name__ == '__main__':
    main()