from typing import Optional, Union
from google.protobuf import json_format
from google.protobuf.message import Message

import uuid

from pinecone.core.openapi.db_data.models import (
    Vector as _Vector,
    Usage,
    ScoredVector,
    SparseValues,
    QueryResponse,
    FetchResponse,
    IndexDescription as DescribeIndexStatsResponse,
    UpsertResponse,
    NamespaceSummary,
    NamespaceDescription,
    ListNamespacesResponse,
    Pagination,
)
import pinecone.core.grpc.protos.db_data_2025_04_pb2 as pc_pb2

from google.protobuf.struct_pb2 import Struct


def _generate_request_id() -> str:
    return str(uuid.uuid4())


def dict_to_proto_struct(d: Optional[dict]) -> "Struct":
    if not d:
        d = {}
    s = Struct()
    s.update(d)
    return s


def parse_sparse_values(sparse_values: pc_pb2.SparseValues):
    return SparseValues(
        indices=list(sparse_values.indices),
        values=list(sparse_values.values),
    )


def parse_fetch_response(response: pc_pb2.FetchResponse):
    vd = {}

    for vec_id, vec in response.vectors.items():
        vd[vec_id] = _Vector(
            id=vec.id,
            values=list(vec.values),
            sparse_values=parse_sparse_values(vec.sparse_values),
            metadata=json_format.MessageToDict(vec.metadata),
            _check_type=False,
        )

    return FetchResponse(
        vectors=vd,
        namespace=response.namespace,
        usage=parse_usage(response.usage),
        _check_type=False,
    )


def parse_usage(usage: pc_pb2.Usage):
    return Usage(read_units=usage.read_units, _check_type=False)


def parse_upsert_response(response: Message, _check_type: bool = False):
    json_response = json_format.MessageToDict(response)
    upserted_count = json_response.get("upsertedCount", 0)
    return UpsertResponse(upserted_count=int(upserted_count))


def parse_update_response(response: Union[dict, Message], _check_type: bool = False):
    return {}


def parse_delete_response(response: Union[dict, Message], _check_type: bool = False):
    return {}


def parse_query_response(response: pc_pb2.QueryResponse, _check_type: bool = False):
    matches = []
    for item in response.matches:
        sc = ScoredVector(
            id=item.id,
            score=item.score,
            values=list(item.values),
            sparse_values=parse_sparse_values(item.sparse_values),
            metadata=json_format.MessageToDict(item.metadata),
            _check_type=_check_type,
        )
        matches.append(sc)

    return QueryResponse(
        namespace=response.namespace,
        matches=matches,
        usage=parse_usage(response.usage),
        _check_type=_check_type,
    )


def parse_stats_response(response: dict):
    fullness = response.get("indexFullness", 0.0)
    total_vector_count = response.get("totalVectorCount", 0)
    dimension = response.get("dimension", 0)
    summaries = response.get("namespaces", {})
    namespace_summaries = {}
    for key in summaries:
        vc = summaries[key].get("vectorCount", 0)
        namespace_summaries[key] = NamespaceSummary(vector_count=vc)
    return DescribeIndexStatsResponse(
        namespaces=namespace_summaries,
        dimension=dimension,
        index_fullness=fullness,
        total_vector_count=total_vector_count,
        _check_type=False,
    )


def parse_namespace_description(response: Message) -> NamespaceDescription:
    json_response = json_format.MessageToDict(response)
    return NamespaceDescription(
        name=json_response.get("name", ""),
        record_count=json_response.get("recordCount", 0),
        _check_type=False,
    )


def parse_list_namespaces_response(response: Message) -> ListNamespacesResponse:
    json_response = json_format.MessageToDict(response)
    
    namespaces = []
    for ns in json_response.get("namespaces", []):
        namespaces.append(NamespaceDescription(
            name=ns.get("name", ""),
            record_count=ns.get("recordCount", 0),
            _check_type=False,
        ))
    
    pagination = None
    if "pagination" in json_response and json_response["pagination"]:
        pagination = Pagination(
            next=json_response["pagination"].get("next", ""),
            _check_type=False,
        )
    
    return ListNamespacesResponse(
        namespaces=namespaces,
        pagination=pagination,
        _check_type=False,
    )
