from pinecone.specs import database as db_specs


class Database(db_specs.DatabaseSpec):
    """The index as a database."""

    def __init__(self, name: str,dimension: int,index_type: str='approximated',metric: str='cosine',replicas: int=1,shards: int=1, engine_config: {}=None):
        """"""
        super().__init__(name,dimension,index_type,metric,replicas,shards,engine_config)

