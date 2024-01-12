import os
import random
import string
from pinecone.grpc import PineconeGRPC

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

DIMENSION = 1536 # common for openai embeddings

def create_index_if_not_exists(pc, index_name):
    if index_name not in pc.list_indexes().names():
        print(f'Index {index_name} does not exist, creating it')
        pc.create_index(
            name=index_name,
            metric='cosine',
            dimension=DIMENSION,
            spec={
                'serverless': {
                    'cloud': read_env_var('CLOUD'),
                    'region': read_env_var('REGION'),
                }
            }
        )

upserted_ids = set()

def main():
    pc = PineconeGRPC(api_key=read_env_var('PINECONE_API_KEY'))
    index_name = read_env_var('INDEX_NAME')
    iterations = int(read_env_var('ITERATIONS'))

    create_index_if_not_exists(pc, index_name)

    index = pc.Index(name=index_name)
    for i in range(iterations):
        try:
            # Upsert some vectors
            items_to_upsert = random.randint(1, 100)
            vector_list = [
                {
                    'id': random_string(10), 
                    'values': random_embedding_values(DIMENSION),
                    'metadata': {
                        'genre': random.choice(['action', 'comedy', 'drama']),
                        'runtime': random.randint(60, 120)
                    }
                } for x in range(items_to_upsert)
            ]
            index.upsert(vectors=vector_list)
            print('Upserted {} vectors'.format(items_to_upsert))

            for v in vector_list:
                upserted_ids.add(v['id'])

            # Fetch some vectors
            ids_to_fetch = random.sample(upserted_ids, k=random.randint(1, 20))
            print('Fetching {} vectors'.format(len(ids_to_fetch)))
            fetched_vectors = index.fetch(ids=ids_to_fetch)

            # Query some vectors
            print('Querying 10 times')
            for i in range(10):
                # Query by vector values
                query_vector = random_embedding_values(DIMENSION)
                query_results = index.query(vector=query_vector, top_k=10)
            
            # Delete some vectors
            print('Deleting some vectors')
            id_to_delete = random.sample(upserted_ids, k=random.randint(1, 10))
            index.delete(ids=id_to_delete)
        except Exception as e:
            print('Exception: {}'.format(e))

if __name__ == '__main__':
    main()

