import os
import logging
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


def main():
    encrypted_secret = os.getenv("ENCRYPTED_SECRET")
    encryption_key = os.getenv("FERNET_ENCRYPTION_KEY")

    if encrypted_secret is None:
        raise Exception("ENCRYPTED_SECRET is not set")
    if encryption_key is None:
        raise Exception("FERNET_ENCRYPTION_KEY is not set")

    cipher_suite = Fernet(encryption_key.encode())
    decrypted_secret = cipher_suite.decrypt(encrypted_secret.encode()).decode()

    output_file = os.environ.get("GITHUB_OUTPUT", None)
    if output_file is None:
        logger.error("GITHUB_OUTPUT is not set, cannot write to output file")
    else:
        with open(output_file, "a") as f:
            f.write(f"decrypted_secret={decrypted_secret}\n")


if __name__ == "__main__":
    main()
