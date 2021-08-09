#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

import os
import enum

MAX_MSG_SIZE = 128 * 1024 * 1024

MAX_ID_LENGTH = int(os.environ.get("PINECONE_MAX_ID_LENGTH", default="64"))

REQUEST_ID: str = "request_id"


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
