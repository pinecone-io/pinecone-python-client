from pinecone.core.openapi.shared.exceptions import PineconeException


class PineconeProtocolError(PineconeException):
    """Raised when something unexpected happens mid-request/response."""


class PineconeConfigurationError(PineconeException):
    """Raised when a configuration error occurs."""


class ListConversionException(PineconeException, TypeError):
    def __init__(self, message):
        super().__init__(message)
