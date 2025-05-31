import os
import logging
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


def mask(value):
    """Mask the value in Github Actions logs"""
    print(f"::add-mask::{value}")


def main():
    secret = os.getenv("SECRET")
    encryption_key = os.getenv("ENCRYPTION_KEY")

    if secret is None:
        raise Exception("SECRET is not set")
    if encryption_key is None:
        raise Exception("ENCRYPTION_KEY is not set")

    mask(secret)
    mask(encryption_key)

    cipher_suite = Fernet(encryption_key.encode())
    encrypted_secret = cipher_suite.encrypt(secret.encode()).decode()

    output_file = os.environ.get("GITHUB_OUTPUT", None)
    if output_file is None:
        logger.error("GITHUB_OUTPUT is not set, cannot write to output file")
    else:
        with open(output_file, "a") as f:
            f.write(f"encrypted_secret={encrypted_secret}\n")


if __name__ == "__main__":
    main()
