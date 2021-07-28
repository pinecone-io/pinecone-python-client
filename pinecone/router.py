#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from typing import List, NamedTuple
from pinecone.legacy.specs import traffic_router as traffic_router_spec

from pinecone.utils.sentry import sentry_decorator as sentry
from pinecone.api_router import RouterAPI
from pinecone.api_controller import ControllerAPI
from pinecone.constants import Config

__all__ = ["describe", "deploy", "stop", "ls", "update_active_service", "update_services", "RouterMeta"]


class RouterMeta(NamedTuple):
    """Metadata of a traffic router."""

    name: str
    active_service: str
    services: List[str]
    status: dict


def _get_router_api():
    return RouterAPI(host=Config.CONTROLLER_HOST, api_key=Config.API_KEY)


def _get_controller_api():
    return ControllerAPI(host=Config.CONTROLLER_HOST, api_key=Config.API_KEY)


@sentry
def ls():
    """Returns the names of all traffic routers."""
    api = _get_router_api()
    return api.list()


@sentry
def deploy(router_name: str, services: List[str], active_service: str = None):
    """Creates a traffic router.

    By default the first service will be set as active unless otherwise specified.

    :param router_name: name of the traffic router
    :type router_name: str
    :param services: the list of services that the traffic router can route traffic to.
        By default, the first service will be set as the active service.
    :type services: List[str]
    :param active_service: defaults to None. if specified, the given service will be the active service for the traffic router.
    :type active_service: str, optional
    """
    if active_service and active_service not in services:
        raise RuntimeError("{active_service} must be one of the services {services}".format(**locals()))

    active_service = active_service or services[0]

    controller_api = _get_controller_api()
    router_api = _get_router_api()

    user_services = controller_api.list_services()
    if set(services) - set(user_services):
        raise RuntimeError("These services are not yet available: {0}".format(list(set(services) - set(user_services))))

    router_ = traffic_router_spec.TrafficRouter(router_name, services, active_service)
    return router_api.deploy(router_.to_json())


@sentry
def update_active_service(router_name: str, active_service: str):
    """Updates the active service of a router.

    :param router_name: name of the traffic router
    :type router_name: str
    :param active_service: name of the active service. It has to be a service already in a traffic router.
    :type active_service: str
    """
    api = _get_router_api()
    router_json = api.get_router(router_name)
    router_ = traffic_router_spec.TrafficRouter.from_json(router_json)
    if active_service not in router_.all_services:
        raise RuntimeError("Service {0} is not in router {1}".format(active_service, router_.name))

    router_.active_service = active_service
    return api.deploy(router_.to_json())


@sentry
def update_services(router_name: str, services: List[str]):
    """Updates the list of services available in the traffic router.

    :param router_name: name of the traffic router
    :type router_name: str
    :param services: the list of services that the traffic router can route traffic to.
    :type services: List[str]
    """
    controller_api = _get_controller_api()
    router_api = _get_router_api()

    user_services = controller_api.list_services()
    if set(services) - set(user_services):
        raise RuntimeError("These services are not yet available: {0}".format(list(set(services) - set(user_services))))

    router_json = router_api.get_router(router_name)
    router_ = traffic_router_spec.TrafficRouter.from_json(router_json)
    if router_.active_service not in services:
        raise RuntimeError(
            "The new services must include the current active service {0}.".format(router_.active_service)
        )

    router_.all_services = services
    return router_api.deploy(router_.to_json())


@sentry
def stop(router_name: str):
    """Stops a router.

    :param router_name: the name of the router
    """
    api = _get_router_api()
    return api.stop(router_name)


@sentry
def describe(router_name: str):
    """Returns the metadata of a router.

    :param router_name: the name of the router
    """
    api = _get_router_api()
    router_json = api.get_router(router_name)
    router_ = traffic_router_spec.TrafficRouter.from_json(router_json)
    return RouterMeta(
        name=router_.name,
        active_service=router_.active_service,
        services=router_.all_services,
        status=api.get_status(router_name) or {},
    )
