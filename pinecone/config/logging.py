import logging

PARENT_LOGGER_NAME = "pinecone"
DEFAULT_PARENT_LOGGER_LEVEL = "ERROR"

_logger = logging.getLogger(__name__)
_parent_logger = logging.getLogger(PARENT_LOGGER_NAME)
_parent_logger.setLevel(DEFAULT_PARENT_LOGGER_LEVEL)
