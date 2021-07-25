#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from pinecone.legacy.functions import Function
from pinecone.legacy.specs import Spec
from pinecone.legacy.utils import validate_dns_name, constants

from collections import defaultdict
from pydoc import locate
import argparse




class Service (Spec):
    """
    Class that represents a functional pipeline of vector search as a graph.
    """

    def __init__(self, name: str = None, gateway_replicas: int = 1):
        """
        :param gateway_port: Port to run the gRPC gateway on
        :param native: Whether to run with native python processes or kubernetes
        """
        self._name = name
        self.paths = defaultdict(list)
        self.paths['read'] = []
        self.paths['write'] = []
        self.functions = {}
        self.write_paths = {'write'}
        self.gateway_replicas = gateway_replicas

    @property
    def name(self) -> str:
        """Name of the service."""
        return self._name

    @name.setter
    def name(self, value: str):
        validate_dns_name(value)
        self._name = value

    def set_path_as_write(self, path_name: str):
        """
        Set a given path as a "write" path, used for stateful services
        :param path_name:
        :return:
        """
        self.write_paths.add(path_name)

    def unset_path_as_write(self, path_name: str):
        """
        Set a given path as not a "write" path, used for stateful functions
        :param path_name:
        :return:
        """
        self.write_paths.remove(path_name)

    def append_to_path(self, function: Function, path: str):
        """Appends a function to a given path.

        :param function: a Pinecone Function
        :type function: Function
        :param path: path name
        :type path: str
        """
        self.functions[function.name] = function
        self.paths[path].append(function.name)

    def append_to_all_paths(self, function: Function):
        """Appends a function to every path.

        :param function: a Pinecone Function
        :type function: Function
        """
        self.functions[function.name] = function
        self.paths = {k: [*v, function.name] for k, v in self.paths.items()}

    def remove_from_all_paths(self, name: str):
        """Appends a function from every path.

        :param name: name of Pinecone Function to remove
        :type name: str
        """
        if name in self.functions:
            del self.functions[name]
        self.paths = {k: [func for func in v if func != name] for k, v in self.paths.items()}

    def validate(self):
        """Validates the service.

        Perform the following validations:

        - There should be no orphaned nodes; that is, every node should belong to at least one path.
        - There should be no missing nodes; that is, any node name specified in one of the paths
          should have be in the list of functions.
        - Make sure that the paths have no cycles.
        - Service name and node names should be DNS compatible
        """
        node_names = [name for name in self.functions.keys()]
        all_steps = [ss for path in self.paths.values() for ss in path]
        orphaned_nodes = set(node_names) - set(all_steps)
        missing_nodes = set(all_steps) - set(node_names)

        for name in node_names:
            validate_dns_name(name)

        if orphaned_nodes:
            raise ValueError("Orphaned nodes (every node must belong to at least one path): {}".format(orphaned_nodes))
        if missing_nodes:
            raise ValueError("Missing nodes (at least one path has missing nodes): {}".format(missing_nodes))

        # Check for cycles
        for path_name, steps in self.paths.items():
            if len(steps) > len(set(steps)):
                raise ValueError("The following path contains cycles: ({}, {})".format(path_name, steps))

        aggregator = self.functions.get(constants.AGGREGATOR_NAME, None)
        if aggregator and aggregator.replicas != self.gateway_replicas:
            raise ValueError("The number of aggregator replicas must be the same as gateway replicas.")

    def to_obj(self) -> dict:
        """Serializes specs."""
        metadata = {'name': self.name}
        spec = {
            'functions': [function.to_dict() for function in self.functions.values()],
            'paths': [{
                'name': name,
                'steps': steps
            } for name, steps in self.paths.items()],
            'write_paths': list(self.write_paths),
            'gateway_replicas': self.gateway_replicas
        }
        return {
            'version': 'pinecone/v1alpha1',
            'kind': 'Service',
            'metadata': metadata,
            'spec': spec
        }

    @classmethod
    def from_obj(cls, state_dict: dict) -> "Service":
        """Returns an instance of a Pinecone service from a graph.

        :return: :class:`Service`
        :rtype: :class:`Service`
        """
        assert state_dict['kind'] == 'Service'

        metadata = state_dict['metadata']
        spec = state_dict['spec']
        app = cls(metadata['name'], gateway_replicas=spec['gateway_replicas'])

        parser = argparse.ArgumentParser()
        Function.add_args(parser)
        app.functions = {
            function['name']: locate(function['class']).from_args(parser.parse_args(function['args']))
            for function in spec['functions']
        }
        app.paths = {path['name']: path['steps'] for path in spec['paths']}
        app.write_paths = set(spec['write_paths'])
        return app
