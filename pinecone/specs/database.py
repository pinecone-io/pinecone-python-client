from pinecone.specs import Spec

class DatabaseSpec(Spec):

    @property
    def name(self) -> str:
        return self._name

    def __init__(self, name: str, dimension: int, index_type: str = 'approximated', metric: str = 'cosine',
                 replicas: int = 1, shards: int = 1, index_config: dict = None):
        self._name = name
        self.index_type = index_type
        self.metric = metric
        self.dimension = dimension
        self.shards = shards
        self.replicas = replicas
        self.index_config = self.get_config(index_config, index_type) if index_config else {}

    def get_config(self, index_config: dict, index_type: str):
        if index_type == 'approximated':
            kbits = index_config['kbits'] if 'kbits' in index_config else 512
            hybrid = index_config['hybrid'] if 'hybrid' in index_config else False
            deduplication = index_config['deduplication'] if 'deduplication' in index_config else False
            return {
                'kbits': kbits, 'hybrid': hybrid, 'deduplication': deduplication
            }
        if index_type == 'hnsw':
            ef_construction = index_config['ef_construction'] if 'ef_construction'  in index_config else 500
            ef = index_config['ef'] if 'ef' in index_config else 250
            M = index_config['M'] if 'M' in index_config else 12
            max_elements = index_config['max_elements'] if 'max_elements' in index_config else 5000000
            return {
                'ef_construction': ef_construction,
                'ef': ef,
                'M': M,
                'max_elements': max_elements
            }

    @classmethod
    def from_obj(cls, obj: dict) -> "DatabaseSpec":
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