import yaml
import hashlib

def calculate_file_hash(filename):
    """Compute the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filename, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def verify_hashes(hashes_file):
    """Verify hashes of multiple files against a sums file."""
    with open(hashes_file, 'r') as hf:
        metadata = yaml.safe_load(hf)
        for path, hash in metadata['files'].items():
            if hash != calculate_file_hash(path):
                raise ValueError(f'Hash mismatch for file {path}')