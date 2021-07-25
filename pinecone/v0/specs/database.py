from pinecone.v0.specs import Spec
from pinecone.v0.functions.index.namespaced import NamespacedIndex

import argparse


class DatabaseSpec(Spec):

    @property
    def name(self) -> str:
        return self._name

    def __init__(self, name: str, dimension: int, index_type: str = 'approximate',  replicas: int = 1, **config):
        self._name = name
        self.index = NamespacedIndex(index_type, **config)
        self.replicas = replicas
        self.dimension = dimension

    @classmethod
    def from_obj(cls, obj: dict) -> "Spec":
        spec = obj['spec']
        metadata = obj['metadata']
        new_index = cls(metadata['name'], spec['dimension'], replicas=spec['replicas'])
        parser = argparse.ArgumentParser()
        NamespacedIndex.add_args(parser)

        new_index.index = NamespacedIndex.from_args(parser.parse_args(spec.get('index', spec.get('engine'))))
        return new_index

    def to_obj(self) -> dict:
        spec = {
            "index": self.index.to_args(),
            "replicas": self.replicas,
            "dimension": self.dimension
        }
        return {
            'version': 'pinecone/beta',
            'kind': 'Database',
            'metadata': {"name": self._name},
            'spec': spec
        }
