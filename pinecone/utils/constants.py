#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

import os
import enum

RECV_TIMEOUT = 0.5
SEND_TIMEOUT = 0.5

STORAGE_CLASS_NAME = os.environ.get('STORAGE_CLASS_NAME', "standard")
GCP_REGION = os.environ.get('GCP_REGION')

KSA_NAME = 'pc-service'

SNAPSHOTTER_NAME = 'snapshotter'
INDEX_BUILDER_NAME = 'index-builder'

MAX_CLIENTS = 10000
MAX_RETRY_MSG = 100
DEFAULT_TIMEOUT = 2
GATEWAY_NAME = 'gatewayrouter'
AGGREGATOR_NAME = 'aggregator'
MAX_MSG_SIZE = 128 * 1024 * 1024
MAX_MSGS_PER_CONNECTION = 100
MAX_SOCKS_OPEN = 1000
KUBE_SYNC_TTL = 5  # interval to sync with the kube control plane, reloading DNS and shard config

STATS_KEY_NS_PREFIX = 'ns:'

HASH_RING_PARTITIONS = int(os.environ.get("HASH_RING_PARTITIONS", 200))

ZMQ_CONTROL_PORT = os.environ.get('ZMQ_CONTROL_PORT', 5559)
ZMQ_PORT_IN = os.environ.get('ZMQ_PORT_IN', 5557)
ZMQ_LOG_PORT = os.environ.get('ZMQ_LOG_PORT', 5558)
ZMQ_SERVICE_METRICS_PORT = os.environ.get('ZMQ_SERVICE_METRICS_PORT', 5560)
PC_CONTROLLER_PORT = os.environ.get('PC_CONTROLLER_PORT', 8083)
SNAPSHOTTER_PORT = os.environ.get('SNAPSHOTTER_PORT', 8093)
SERVICE_GATEWAY_PORT = os.environ.get('SERVICE_GATEWAY_PORT', 5007)
SERVICE_GATEWAY_METRICS_PORT = os.environ.get('SERVICE_GATEWAY_METRICS_PORT', 5008)
PERSISTENT_VOLUME_MOUNT = '/data'
ENV_VARS = [ZMQ_CONTROL_PORT, ZMQ_PORT_IN, ZMQ_LOG_PORT, PC_CONTROLLER_PORT]
ZMQ_SECONDARY_PORT = 5559
MAX_MSGS_PENDING = 1000

SHARD_CONFIG_PATH = '/etc/config'
NATIVE_SHARD_PATH = '/tmp/config'

MAX_ID_LENGTH = int(os.environ.get("PINECONE_MAX_ID_LENGTH", default="64"))
MEMORY_UTILIZATION_LIMIT = float(os.environ.get("MEMORY_UTILIZATION_LIMIT", default="0.9"))
COMPACT_WAL_THRESHOLD = float(os.environ.get("COMPACT_WAL_THRESHOLD", default="0.4"))

KAFKA_STATISTICS_INTERVAL_MS = int(os.environ.get("KAFKA_STATISTICS_INTERVAL_MS", default="60000"))
DD_STATSD_SAMPLE_RATE = float(os.environ.get("DD_STATSD_SAMPLE_RATE", default="0.1"))


NS_PER_SECOND = 1_000_000_000

def get_container_memory_limit():
    MEMORY_LIMIT_FILE = '/sys/fs/cgroup/memory/memory.limit_in_bytes'
    if not os.path.exists(MEMORY_LIMIT_FILE):
        return 0  # no limit
    with open(MEMORY_LIMIT_FILE) as stat_file:
        return int(stat_file.read())


MEMORY_LIMIT_BYTES = MEMORY_UTILIZATION_LIMIT * get_container_memory_limit()

SHARD_SCALE_THRESHOLD = float(os.environ.get("SHARD_SCALE_THRESHOLD", default="0.8"))
SHARD_SCALE_BYTES = MEMORY_LIMIT_BYTES * SHARD_SCALE_THRESHOLD

CPU_UNIT = 500
MEMORY_UNIT = 2000
DISK_UNIT = 10


class NodeType(str, enum.Enum):
    STANDARD = 'STANDARD'
    COMPUTE = 'COMPUTE'
    MEMORY = 'MEMORY'
    STANDARD2X = 'STANDARD2X'
    COMPUTE2X = 'COMPUTE2X'
    MEMORY2X = 'MEMORY2X'
    STANDARD4X = 'STANDARD4X'
    COMPUTE4X = 'COMPUTE4X'
    MEMORY4X = 'MEMORY4X'


NODE_TYPE_RESOURCES = {NodeType.STANDARD: {'memory': MEMORY_UNIT, 'cpu': CPU_UNIT, 'disk': DISK_UNIT},
                       NodeType.STANDARD2X: {'memory': 2 * MEMORY_UNIT, 'cpu': 2 * CPU_UNIT, 'disk': 2 * DISK_UNIT},
                       NodeType.STANDARD4X: {'memory': 4 * MEMORY_UNIT, 'cpu': 4 * CPU_UNIT, 'disk': 4 * DISK_UNIT},
                       NodeType.COMPUTE: {'memory': MEMORY_UNIT, 'cpu': 2 * CPU_UNIT, 'disk': DISK_UNIT},
                       NodeType.COMPUTE2X: {'memory': 2 * MEMORY_UNIT, 'cpu': 4 * CPU_UNIT, 'disk': DISK_UNIT * 2},
                       NodeType.COMPUTE4X: {'memory': 4 * MEMORY_UNIT, 'cpu': 8 * CPU_UNIT, 'disk': DISK_UNIT * 4},
                       NodeType.MEMORY: {'memory': 2 * MEMORY_UNIT, 'cpu': CPU_UNIT, 'disk': DISK_UNIT * 2},
                       NodeType.MEMORY2X: {'memory': 4 * MEMORY_UNIT, 'cpu': 2 * CPU_UNIT, 'disk': DISK_UNIT * 4},
                       NodeType.MEMORY4X: {'memory': 8 * MEMORY_UNIT, 'cpu': 4 * CPU_UNIT, 'disk': DISK_UNIT * 8}}


class IndexState(enum.Enum):
    READY = 1  # healthy
    PENDING = 2  # buffering writes during rebalancing
    FULL = 3  # cannot accept more writes

