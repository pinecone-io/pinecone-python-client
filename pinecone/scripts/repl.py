import code
import logging
from pinecone.utils.repr_overrides import setup_readline_history


def setup_logging():
    # Create a custom formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create and configure the console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    return root_logger


def main():
    # Set up logging
    logger = setup_logging()
    logger.info("Initializing environment...")

    # Set up readline history
    setup_readline_history()

    # You can add any setup code here, such as:
    # - Setting environment variables
    # - Importing commonly used modules
    # - Loading configuration files

    # Start the interactive REPL
    banner = """
    Welcome to the custom Python REPL!
    Your initialization steps have been completed.
    """

    # Create a custom namespace with any pre-loaded variables
    namespace = {
        "__name__": "__main__",
        "__doc__": None,
        "logger": logger,  # Make logger available in REPL
        # Add any other variables you want to have available in the REPL
    }

    # Start the interactive console
    code.interact(banner=banner, local=namespace)


if __name__ == "__main__":
    main()
