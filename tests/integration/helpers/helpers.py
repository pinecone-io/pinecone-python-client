import re
import os
import random
import string

def random_string(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))

def generate_index_name(test_name: str) -> str:
    buildNumber = os.getenv('GITHUB_BUILD_NUMBER', None)
    
    if test_name.startswith('test_'):
        test_name = test_name[5:]

    # Trim name length to save space for other info in name
    test_name = test_name[:35]

    # Remove trailing underscore, if any
    if test_name.endswith('_'):
        test_name = test_name[:-1]

    name_parts = [buildNumber, test_name, random_string(45)]
    index_name = '-'.join([x for x in name_parts if x is not None])
    
    # Remove invalid characters
    replace_with_hyphen = re.compile(r'[\[\(_,\s]')
    index_name = re.sub(replace_with_hyphen, '-', index_name)
    replace_with_empty = re.compile(r'[\]\)\.]')
    index_name = re.sub(replace_with_empty, '', index_name)

    max_length = 45
    index_name = index_name[:max_length]

    return index_name.lower()

def get_environment_var(name: str) -> str:
    val = os.getenv(name)
    if (val is None):
        raise Exception('Expected environment variable '  + name + ' is not set')
    else:
        return val
