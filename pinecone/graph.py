#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from collections import defaultdict
# import graphviz
# import networkx as nx

import pinecone
from pinecone.legacy.specs import service as service_specs
from pinecone.legacy.functions.index import namespaced
from pinecone.legacy.functions.ranker import aggregator
from pinecone.utils.sentry import sentry_decorator as sentry
from pinecone.legacy.utils.constants import NodeType
from typing import NamedTuple, Optional

__all__ = ["Graph", "IndexGraph"]


class Graph(service_specs.Service):
    """The graphical representation of a service."""

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

    @sentry
    def dump(self, format: str = "yaml") -> str:
        """Dumps the graph as yaml or json string.

        :param format: one of {"yaml", "json"}, defaults to "yaml".
        :type format: str, optional
        :rtype: str
        """
        """Dumps current graph as yaml or json."""
        if format == "yaml":
            return self.to_yaml()
        elif format == "json":
            return self.to_json()
        else:
            raise NotImplementedError("Format {format} not supported.".format(format=format))

    # @sentry
    # def to_graph(self) -> nx.MultiDiGraph:
    #     """Exports to a `networkx <https://networkx.org/>`_ graph.
    #
    #     Construct a multi directed graph.
    #     Add incoming traffic and outgoing traffic nodes to signify the traffic flow.
    #
    #     :return: [description]
    #     :rtype: nx.MultiDiGraph
    #     """
    #
    #     def _get_node_name(fn: "pinecone.function.Function"):
    #         return fn.name
    #
    #     def _get_edge_name(path_name: str):
    #         return "[{}]".format(path_name)
    #
    #     graph = nx.MultiDiGraph()
    #
    #     for path_name, steps in self.paths.items():
    #         node_names = [_get_node_name(self.functions[ss]) for ss in steps]
    #         for start_node, end_node in zip(["incoming traffic", *node_names], [*node_names, "outgoing traffic"]):
    #             graph.add_edge(
    #                 start_node,
    #                 end_node,
    #                 label=_get_edge_name(path_name),
    #                 key=_get_edge_name(path_name),
    #             )
    #     return graph

    @sentry
    def view(self):
        """Visualizes the graph in an iPython notebook.

        .. note::
            This method requires the `graphviz <https://graphviz.org/download/>`_ package
            installed on your operating system.
        """
        raise NotImplementedError("this method has been removed")
        # multigraph = self.to_graph()
        # union_labels = defaultdict(list)
        # for start_node, end_node, label in multigraph.edges.data("label"):
        #     if label:
        #         union_labels[(start_node, end_node)].append(label)
        #
        # graph = nx.DiGraph()
        # for start_node, end_node, path in multigraph.edges:
        #     labels = union_labels.get((start_node, end_node)) or []
        #     graph.add_edge(start_node, end_node, label="".join(labels))
        #
        # return graphviz.Source(nx.nx_pydot.to_pydot(graph))


class IndexConfig(NamedTuple):
    """Index configuration options.

    :param index_type: type of index, one of {"approximated", "exact"}, defaults to "approximated".
        The "approximated" index uses fast approximate search algorithms developed by Pinecone.
        The "exact" index uses accurate exact search algorithms.
        It performs exhaustive searches and thus it is usually slower than the "approximated" index.
    :type index_type: str, optional
    :param metric: type of metric used in the vector index, one of {"cosine", "dotproduct", "euclidean"}, defaults to "cosine".
        Use "cosine" for cosine similarity,
        "dotproduct" for dot-product,
        and "euclidean" for euclidean distance.
    :type metric: str, optional
    :param shards: the number of shards for the index, defaults to 1.
        As a general guideline, use 1 shard per 1 GB of data.
    :type shards: int, optional
    :param replicas: the number of replicas, defaults to 1.
        Use at least 2 replicas if you need high availability (99.99% uptime) for querying.
        For additional throughput (QPS) your service needs to support, provision additional replicas.
    :type replicas: int, optional
    :param gateway_replicas: number of replicas of both the gateway and the aggregator.
    :type gateway_replicas: int
    :param index_args: advanced arguments for the index instance in the graph.
    :type index_args: dict
    """
    index_type: str = "approximated"
    metric: str = "cosine"
    shards: int = 1
    replicas: int = 1
    gateway_replicas: int = 1
    node_type: str = NodeType.STANDARD.value
    index_args: Optional[dict] = None

    @classmethod
    def _from_dict(cls, kwargs: dict):
        cls_kwargs = {kk: vv for kk, vv in kwargs.items() if kk in cls._fields}
        return cls(**cls_kwargs)

    @classmethod
    def _from_graph(cls, graph: Graph):
        """Reconstruct configs from a Graph."""
        config = {}
        config["gateway_replicas"] = graph.gateway_replicas
        index = graph.functions.get("index")
        if index:
            config["replicas"] = index.replicas
            config["shards"] = index.shards
            config["node_type"] = index._node_type.value
            config["index_type"] = index.config.pop("index_type")
            config["metric"] = index.config.pop("metric")
            config["index_args"] = {**(index.config or {})}
        return cls(**config)


class IndexGraph(Graph):
    """The graphical representation of an index service.
    """

    _default_index = namespaced.NamespacedIndex
    _default_aggregator = aggregator.Aggregator

    @sentry
    def __init__(self, **kwargs):
        """See :class:`IndexConfig` for details about the parameters."""
        config = IndexConfig._from_dict(kwargs)
        index_type = config.index_type
        metric = config.metric
        shards = config.shards
        replicas = config.replicas
        gateway_replicas = config.gateway_replicas
        node_type = NodeType(config.node_type)
        index_args = config.index_args or {}

        super().__init__(gateway_replicas=gateway_replicas)

        # The index and the aggregator must be specified for an index service
        self.index = self._default_index(
            index_type,
            metric=metric,
            replicas=replicas,
            shards=shards,
            name="index",
            node_type=node_type,
            **index_args
        )
        self.aggregator = self._default_aggregator(num_shards=shards, name="aggregator", replicas=gateway_replicas)
        self._update_graph()

    def _update_graph(self):
        write_functions = [self.index, self.aggregator]
        read_functions = [self.index, self.aggregator]
        self.paths["write"] = [fn.name for fn in write_functions if fn]
        self.paths["read"] = [fn.name for fn in read_functions if fn]
        self.functions = {fn.name: fn for fn in [*write_functions, *read_functions] if fn}
