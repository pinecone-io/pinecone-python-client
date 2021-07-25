#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

import json
from pathlib import Path

import pickle
from pydoc import locate
import numpy as np
from pinecone import logger
from pinecone.legacy.functions.index import Index
from pinecone.protos import core_pb2
from pinecone.utils import dump_strings, dump_numpy, load_numpy
from pinecone.legacy.utils.constants import STATS_KEY_NS_PREFIX

class NamespacedIndex(Index):
    """
    This index allows indexing of vectors in namespaces using exact or approximated index, such that
    each namespace will is handled by a unique sub-index.
    """

    SUPPORTED_INDEXES = {'approximated': 'pinecone_engine.functions.index.vector.ananas.AnanasIndex',
                         'exact': 'pinecone_engine.functions.index.vector.exact.ExactIndex',
                         'hnsw': 'pinecone_engine.functions.index.vector.hnsw.HNSWIndex'}
    SUPPORTED_INDEX_IMAGES = {'approximated': 'pinecone/engine/approximated',
                               'exact': 'pinecone/engine/exact',
                               'hnsw': 'pinecone/engine/hnsw'}

    @property
    def image(self):
        return self.SUPPORTED_INDEX_IMAGES[self.config['index_type']]

    def __init__(self, index_type="approximated", *args, **config):
        """
        NamespacedIndex Constructor
        :param index_type: type of index to use for each namespace.
                            The supported indexs are 'approximated' and 'exact':
                            'approximated' -  This index allows a fast approximated search based on common metrics
                                              It's based on Pinecone's proprietary vector index
                            'exact' -  This index allows an accurate vector search based on common metrics
                                       It's based on an exhaustive search, thus might be slow
        :param metric: type of metric used in the vector index -
                        'cosine' - cosine similarity
                        'dotproduct' - dot-product
                        'euclidean' - euclidean distance
        :return:
        """
        super().__init__(*args, **config)
        self.config['index_type'] = index_type
        self.index_class = None
        self.indexes = {}

    def setup(self):
        super().setup()
        self.index_class = self.get_index_class()
        self.export_metadata()

    def cleanup(self):
        for index in self.indexes.values():
            index.cleanup()

    def get_index_class(self):
        index_type = self.config['index_type']
        if index_type not in NamespacedIndex.SUPPORTED_INDEXES:
            raise ValueError(f'{index_type} is not a supported index type')
        return locate(NamespacedIndex.SUPPORTED_INDEXES[index_type])

    def export_metadata(self):
        logger.opt(raw=True).info(json.dumps({"recordType": 'namespacedIndex.metadata',
                                              "name": self.name, **self.config,
                                              "namespaces": list(self.indexes.keys())}) + "\n")

    def has_query_overrides(self, msg: 'core_pb2.Request'):
        return len(msg.query.top_k_overrides) > 0 and len(msg.query.namespace_overrides) > 0

    def delete_namespace(self, namespace: str):
        if namespace not in self.indexes:
            return
        index = self.indexes[namespace]
        index.cleanup()
        self.indexes.pop(namespace)


    def handle_index_msg(self, msg: 'core_pb2.Request') -> 'core_pb2.Request':
        namespace = msg.namespace
        if namespace not in self.indexes:
            index_name = f'{self.name}-{self.id}-{namespace}'
            new_index = self.index_class(name=index_name, **self.config)
            new_index.setup()
            self.indexes[namespace] = new_index
            self.export_metadata()
        return self.indexes[namespace].handle_msg(msg)

    def handle_query_msg(self, msg: 'core_pb2.Request') -> 'core_pb2.Request':
        top_k_overrides = msg.query.top_k_overrides
        namespace_overrides = msg.query.namespace_overrides
        data = load_numpy(msg.query.data)
        if len(top_k_overrides) != data.shape[0] or len(namespace_overrides) != data.shape[0]:
            raise ValueError("Must supply override top_k and namespace values for each query")
        for top_k, namespace, query in zip(top_k_overrides, namespace_overrides, data):
            if namespace not in self.indexes:
                raise ValueError(f"Namespace {namespace} does not exist")
            msg.namespace = namespace
            msg.query.top_k = top_k
            msg.query.data.CopyFrom(dump_numpy(np.array([query])))
            msg = self.indexes[namespace].handle_msg(msg)
        return msg

    def handle_msg(self, msg: 'core_pb2.Request') -> 'core_pb2.Request':
        req_type = msg.WhichOneof('body')
        if req_type == 'index':
            msg = self.handle_index_msg(msg)
        elif req_type == 'list' and msg.list.resource_type == 'namespaces':
            msg.list.items.CopyFrom(dump_strings(list(self.indexes.keys())))
        elif req_type == 'delete' and msg.delete.delete_all:
            self.delete_namespace(msg.namespace)
        elif req_type == 'query' and self.has_query_overrides(msg):
            msg = self.handle_query_msg(msg)
        else:
            namespace = msg.namespace
            if namespace in self.indexes:
                msg = self.indexes[namespace].handle_msg(msg)
        return msg

    def size(self) -> int:
        return sum(index.size() for index in self.indexes.values())

    def get_stats(self):
        return {'size': self.size(), **{f'{STATS_KEY_NS_PREFIX}{namespace}': self.indexes[namespace].size() for namespace in self.indexes}}

    @property
    def threadsafe(self) -> bool:
        return self.config['index_type'] == 'approximated'

    def save(self, path: Path):
        namespaces_top_level_dir = path.joinpath('namespaces')
        namespaces_top_level_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving ephemeral snapshot to {}", str(namespaces_top_level_dir))
        metadata_path = namespaces_top_level_dir.joinpath("engine.meta")
        with metadata_path.open('wb') as metadata_file:
            pickle.dump(self.config, metadata_file)
            pickle.dump(list(self.indexes.keys()), metadata_file)

        for namespace in self.indexes:
            namespace_dir = namespaces_top_level_dir.joinpath(namespace)
            namespace_dir.mkdir(parents=True, exist_ok=True)
            index = self.indexes[namespace]
            logger.info("Saving ephemeral snapshot for namespace: {} to {}", namespace, str(namespace_dir))
            index.save(namespace_dir)
            logger.info("Done")

    def load(self, path: Path):
        namespaces_top_level_dir = path.joinpath('namespaces')
        metadata_path = namespaces_top_level_dir.joinpath("engine.meta")
        logger.info("Loading ephemeral snapshots from {}", str(namespaces_top_level_dir))

        if not namespaces_top_level_dir.is_dir():
            raise ValueError(f"{str(namespaces_top_level_dir)} is not a directory!")
        if not metadata_path.is_file():
            logger.info("Couldn't find metadata {} or bin {}. Skipping load.", str(metadata_path))
            return

        with open(metadata_path, "rb") as metadata_file:
            self.config = pickle.load(metadata_file)
            namespaces = pickle.load(metadata_file)

        def load_namespace(namespace: str):
            logger.info("Loading namespace {}", namespace)
            index_name = f'{self.name}-{self.id}-{namespace}'
            new_index: Index = self.index_class(name=index_name, **self.config)
            new_index.setup()
            new_index.load(namespaces_top_level_dir.joinpath(namespace))
            self.indexes[namespace] = new_index
            logger.info("Done loading namespace {}", namespace)

        for namespace in namespaces:
            load_namespace(namespace)
