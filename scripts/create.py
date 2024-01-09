import os
import random
import string
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
    pc = Pinecone(api_key=read_env_var('PINECONE_API_KEY'))
    index_name = read_env_var('NAME_PREFIX') + random_string(20)
    pc.create_index(
        name=index_name,
        metric=read_env_var('METRIC'),
        dimension=int(read_env_var('DIMENSION')),
        spec={
            'serverless': {
                'cloud': read_env_var('CLOUD'),
                'region': read_env_var('REGION'),
            }
        }
    )
    write_gh_output('index_name', index_name)

if __name__ == '__main__':
    main()

