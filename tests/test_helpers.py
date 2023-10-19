import random

def generate_random_string(length: int):
    letters = "abcdefghijklmnopqrstuvwxyz0123456789"
    return ''.join(random.choice(letters) for i in range(length))

def generate_index_name():
    return f"python-index-{generate_random_string(10)}"


# generate records 
# generate sparse values