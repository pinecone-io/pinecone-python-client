#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from typing import List, NamedTuple, Tuple

from pinecone import logger

from pinecone.api_controller import ControllerAPI
from pinecone.constants import Config
from pinecone.graph import Graph
from pinecone.utils.sentry import sentry_decorator as sentry
from pinecone.utils.progressbar import ProgressBar

__all__ = ["describe", "deploy", "stop", "ls", "ServiceMeta"]


class ServiceMeta(NamedTuple):
    """Metadata of a service."""

    name: str
    graph: Graph
    status: dict


def _get_controller_api():
    return ControllerAPI(host=Config.CONTROLLER_HOST, api_key=Config.API_KEY)


@sentry
def describe(service_name: str) -> ServiceMeta:
    """Returns the metadata of a service.

    :param service_name: name of the service
    :type service_name: str
    :return: :class:`ServiceMeta`
    """
    api = _get_controller_api()
    service_json = api.get_service(service_name)
    graph = Graph.from_json(service_json) if service_json else None
    return ServiceMeta(name=graph.name, graph=graph, status=api.get_status(service_name) or {})


@sentry
def deploy(service_name: str, graph: Graph, wait: bool = True, **kwargs) -> Tuple[dict, ProgressBar]:
    """Create a new Pinecone service from the graph.

    :param service_name: name of the service
    :type service_name: str
    :param graph: the graphical representation fo the service
    :type graph: :class:`pinecone.graph.Graph`
    :param wait: wait for the service to deploy. Defaults to ``True``
    :type wait: bool
    """
    graph.validate()

    # copy graph
    graph_ = Graph.from_json(graph.to_json())
    graph_.name = service_name

    api = _get_controller_api()

    if service_name in api.list_services():
        raise RuntimeError(
            "A service with the name '{}' already exists. Please deploy your service with a different name.".format(
                service_name
            )
        )
    else:
        response = api.deploy(graph_.to_json())

    # Wait for service to deploy
    status = api.get_status(service_name)
    total_deployments = len(status.get("waiting") or []) + len(status.get("crashed") or [])

    def get_remaining():
        """Get the number of pods that still need to be deployed."""
        status = api.get_status(service_name)
        logger.info("Deployment status: waiting={}, crashed={}".format(status.get("waiting"), status.get("crashed")))
        remaining_deployments = len(status.get("waiting") or []) + len(status.get("crashed") or [])
        return remaining_deployments

    pbar = ProgressBar(total=total_deployments, get_remaining_fn=get_remaining)
    if wait:
        pbar.watch()
    return response, pbar


@sentry
def stop(service_name: str, wait: bool = True, **kwargs) -> Tuple[dict, ProgressBar]:
    """Stops a service.

    :param service_name: name of the service
    :type service_name: str
    :param wait: wait for the service to deploy. Defaults to ``True``
    :type wait: bool
    """
    api = _get_controller_api()
    response = api.stop(service_name)

    # Wait for service to stop
    def get_remaining():
        return 1 * (service_name in api.list_services())

    pbar = ProgressBar(total=1, get_remaining_fn=get_remaining)
    if wait:
        pbar.watch()
    return response, pbar


@sentry
def ls() -> List[str]:
    """Returns all services names."""
    api = _get_controller_api()
    return api.list_services()
