from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi


class CreateIndexForModel:
    """
    The `CreateIndexForModel` class adds functionality to the Pinecone SDK to allow creating indexes from
    from specific embedding models.

    :param config: A `pinecone.config.Config` object, configured and built in the Pinecone class.
    :type config: `pinecone.config.Config`, required
    """

    def __init__(self, manage_indexes_api: ManageIndexesApi):
        self.db_control_api = manage_indexes_api

    def _get_status(self, name: str):
        api_instance = self.db_control_api
        response = api_instance.describe_index(name)
        return response["status"]

    def _is_index_ready(self, name: str) -> bool:
        status = self._get_status(name)
        ready = status["ready"]
        return ready
