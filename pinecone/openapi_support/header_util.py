from typing import List, Dict, Any
from .serializer import Serializer
from .api_client_utils import parameters_to_tuples


class HeaderUtil:
    @staticmethod
    def select_header_content_type(content_types: List[str]) -> str:
        """Returns `Content-Type` based on an array of content_types provided.

        :param content_types: List of content-types.
        :return: Content-Type (e.g. application/json).
        """
        if not content_types:
            return "application/json"

        content_types = [x.lower() for x in content_types]

        if "application/json" in content_types or "*/*" in content_types:
            return "application/json"
        else:
            return content_types[0]

    @staticmethod
    def select_header_accept(accepts):
        """Returns `Accept` based on an array of accepts provided.

        :param accepts: List of headers.
        :return: Accept (e.g. application/json).
        """
        if not accepts:
            return

        accepts = [x.lower() for x in accepts]

        if "application/json" in accepts:
            return "application/json"
        else:
            return ", ".join(accepts)

    @staticmethod
    def process_header_params(default_headers, header_params, collection_formats):
        header_params = header_params or {}
        header_params.update(default_headers)
        if header_params:
            sanitized_header_params: Dict[str, Any] = Serializer.sanitize_for_serialization(
                header_params
            )
            processed_header_params: Dict[str, Any] = dict(
                parameters_to_tuples(sanitized_header_params, collection_formats)
            )
        return processed_header_params
