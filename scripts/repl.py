import dotenv
import code
from pinecone import Pinecone
import logging


def main():
    # You can add any setup code here, such as:
    # - Setting environment variables
    # - Importing commonly used modules
    # - Setting up logging
    # - Loading configuration files

    dotenv.load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )

    # Start the interactive REPL
    banner = """
    Welcome to the custom Python REPL!
    Your initialization steps have been completed.
    """

    # Create a custom namespace with any pre-loaded variables
    namespace = {
        "__name__": "__main__",
        "__doc__": None,
        "pc": Pinecone(),
        # Add any other variables you want to have available in the REPL
    }

    # Start the interactive console
    code.interact(banner=banner, local=namespace)


if __name__ == "__main__":
    main()
