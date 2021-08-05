from pinecone.legacy.specs import Spec

import argparse


class DatabaseSpec(Spec):

    @property
    def name(self) -> str:
        return self._name

    def __init__(self, name: str, dimension: int, index_type: str = 'approximated', metric: str = 'cosine',
                 replicas: int = 1, shards: int = 1, index_config:dict = None):
        self._name = name
        self.index_type = index_type
        self.metric = metric
        self.dimension = dimension
        self.shards = shards
        self.replicas = replicas
        self.index_config = index_config

    @classmethod
    def from_obj(cls, obj: dict) -> "Spec":
        spec = obj['spec']
        metadata = obj['metadata']
        new_index = cls(metadata['name'], index_type=spec['index_type'], dimension=spec['dimension'],
                        replicas=spec['replicas'], shards=spec['shards'], index_config=spec['index_config'])
        return new_index

    def to_obj(self) -> dict:
        spec = {
            "index_type": self.index_type,
            "metric": self.metric,
            "dimension": self.dimension,
            "shards": self.shards,
            "replicas": self.replicas,
            "index_config": self.index_config
        }
        return {
            'version': 'pinecone/beta',
            'kind': 'Database',
            'metadata': {"name": self._name},
            'spec': spec
        }
