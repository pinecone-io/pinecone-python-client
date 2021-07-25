#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#
from contextlib import contextmanager
from pathlib import Path
import inspect

from pinecone.protos import core_pb2, vector_column_service_pb2
from pinecone.legacy.utils import constants
import numpy as np
import hashlib
import os
import re
import traceback
import lz4.frame
from typing import List, Union
from pinecone import logger

DNS_COMPATIBLE_REGEX = re.compile("^[a-z0-9]([a-z0-9]|[-])+[a-z0-9]$")


def load_numpy(proto_arr: 'core_pb2.NdArray') -> 'np.ndarray':
    """
    Load numpy array from protobuf
    :param proto_arr:
    :return:
    """
    if len(proto_arr.shape) == 0:
        return np.array([])
    if proto_arr.compressed:
        numpy_arr = np.frombuffer(lz4.frame.decompress(proto_arr.buffer), dtype=proto_arr.dtype)
    else:
        numpy_arr = np.frombuffer(proto_arr.buffer, dtype=proto_arr.dtype)
    return numpy_arr.reshape(proto_arr.shape)


def dump_numpy(np_array: 'np.ndarray', compressed: bool = False) -> 'core_pb2.NdArray':
    """
    Dump numpy array to core_pb2.NdArray
    """
    return _dump_numpy(np_array, core_pb2.NdArray(), compressed=compressed)


def dump_numpy_public(np_array: 'np.ndarray', compressed: bool = False) -> 'vector_column_service_pb2.NdArray':
    """
    Dump numpy array to vector_column_service_pb2.NdArray
    """
    return _dump_numpy(np_array, vector_column_service_pb2.NdArray(), compressed=compressed)


def _dump_numpy(np_array: 'np.ndarray', protobuf_arr, compressed: bool = False):
    protobuf_arr.dtype = str(np_array.dtype)
    protobuf_arr.shape.extend(np_array.shape)
    if compressed:
        protobuf_arr.buffer = lz4.frame.compress(np_array.tobytes())
        protobuf_arr.compressed = True
    else:
        protobuf_arr.buffer = np_array.tobytes()
    return protobuf_arr


def dump_strings_public(strs: List[str], compressed: bool = False) -> 'vector_column_service_pb2.NdArray':
    return dump_numpy_public(np.array(strs, dtype='S'), compressed=compressed)


def dump_strings(strs: List[str], compressed: bool = False) -> 'core_pb2.NdArray':
    return dump_numpy(np.array(strs, dtype='S'), compressed=compressed)


def load_strings(proto_arr: 'core_pb2.NdArray') -> List[str]:
    return [str(item, 'utf-8') for item in load_numpy(proto_arr)]



HUB_CLASSES = set()


def hubify(cls):
    global HUB_CLASSES
    HUB_CLASSES.add(cls)
    return cls


def load_hub_service(module):
    global HUB_CLASSES
    subclasses = [getattr(module, d) for d in dir(module)
                  if inspect.isclass(getattr(module, d)) and getattr(module, d) in HUB_CLASSES]
    if len(subclasses) > 1:
        logger.error(f'Found more than one hub service, unable to load: {subclasses}.')
    elif len(subclasses) == 0:
        logger.error(f'No implementations of {HUB_CLASSES} found.')
    else:
        return subclasses[0]


def module_name(module):
    return module.__class__.__module__ + '.' + module.__class__.__name__


def get_version():
    return Path(__file__).parent.parent.joinpath('__version__').read_text().strip()


def get_environment():
    return Path(__file__).parent.parent.joinpath('__environment__').read_text().strip()


def clear_pb_repeated(field):
    while len(field) > 0:
        field.pop()


def get_native_port(name: str):
    return 6000 + int(hashlib.sha1(name.encode()).hexdigest(), 16) % 20000


def shard_name(name: str, shard_id: int):
    if shard_id > 0:
        return name + '-s' + str(shard_id)
    return name


def replica_from_shard_name(name: str, replica_id: int = None):
    if replica_id is not None:
        return name + '-' + str(replica_id)
    return name


def replica_name(name: str, shard_id: int, replica_id: int = None):  # shard == 0 is router
    svc_name = shard_name(name, shard_id)
    return replica_from_shard_name(svc_name, replica_id)


def replica_kube_hostname(name: str, shard_id: int, replica_id: int):
    if replica_id is not None:
        return replica_name(name, shard_id, replica_id) + '.' + shard_name(name, shard_id)
    else:
        return shard_name(name, shard_id)


def replica_kube_hostname_from_shard(name: str, replica_id: int):
    return replica_from_shard_name(name, replica_id) + '.' + name


def parse_hostname(function_name: str, hostname: str):
    hostname_list = hostname.replace(function_name, '').split('-')
    shard_id = None
    replica_id = None

    if 'deployment' in hostname:
        return shard_id, replica_id

    for item in hostname_list[1:]:
        if item[0] == 's':
            try:
                shard_id = int(item[1:])
            except ValueError:
                pass
        else:
            try:
                replica_id = int(item)
            except ValueError:
                pass
    return shard_id, replica_id


def open_or_create(path: Union[str, Path], truncate: int = None):
    if os.path.exists(path):
        file = open(path, 'r+b', 1000000)
    else:
        file = open(path, 'w+b', 1000000)
    if truncate:
        file.truncate(truncate)
    return file


def get_hostname():
    return os.environ.get("HOSTNAME")


def get_container_memory_usage():
    with open('/sys/fs/cgroup/memory/memory.stat') as stat_file:
        rss_line = stat_file.readlines()[1].strip()
        return int(rss_line.split()[1])


def get_container_cpu_limit():
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as quota_file:
            with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as period_file:
                quota = float(quota_file.read())
                period = float(period_file.read())
                return quota/period
    except:
        logger.warning('Unable to get container CPU limit, defaulting to 1')
        return 1


def validate_dns_name(name):
    if not DNS_COMPATIBLE_REGEX.match(name):
        raise ValueError("{} is invalid - service names and node names must consist of lower case "
                         "alphanumeric characters or '-', start with an alphabetic character, and end with an "
                         "alphanumeric character (e.g. 'my-name', or 'abc-123')".format(name))


def get_current_namespace():
    try:
        return open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read().strip()
    except FileNotFoundError:
        return os.environ.get("KUBERNETES_NAMESPACE", None)  # for injecting during tests


def index_state() -> constants.IndexState:
    if constants.MEMORY_LIMIT_BYTES == 0:
        return constants.IndexState.READY
    mem_usage = get_container_memory_usage()
    if mem_usage > constants.MEMORY_LIMIT_BYTES:
        return constants.IndexState.FULL
    elif mem_usage > constants.SHARD_SCALE_BYTES:
        return constants.IndexState.PENDING
    else:
        return constants.IndexState.READY


def format_error_msg(msg: core_pb2.Request, e: Exception, function_name: str):
    msg.status.code = core_pb2.Status.StatusCode.ERROR
    details = core_pb2.Status.Details(function=function_name,
                                      function_id=get_hostname(),
                                      exception=str(e),
                                      traceback=traceback.format_exc())
    details.time.GetCurrentTime()
    logger.error(details)
    msg.status.details.append(details)


@contextmanager
def success_fail_cm(success_observers=None, error_observers=None, finally_observers=None):
    try:
        yield
    except Exception as e:
        if error_observers:
            for handler in error_observers:
                handler(e)
        raise e
    else:
        if success_observers:
            for handler in success_observers:
                handler()
    finally:
        if finally_observers:
            for handler in finally_observers:
                handler()
