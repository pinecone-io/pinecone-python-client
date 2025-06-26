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
    IndexDescription as DescribeIndexStatsResponse,
    UpsertResponse,
    NamespaceSummary,
    NamespaceDescription,
    ListNamespacesResponse,
    Pagination,
)
from pinecone.db_data.dataclasses import FetchResponse

from google.protobuf.struct_pb2 import Struct


def _generate_request_id() -> str:
    return str(uuid.uuid4())


def dict_to_proto_struct(d: Optional[dict]) -> "Struct":
    if not d:
        d = {}
    s = Struct()
    s.update(d)
    return s


def parse_sparse_values(sparse_values: dict):
    return (
        SparseValues(indices=sparse_values["indices"], values=sparse_values["values"])
        if sparse_values
        else SparseValues(indices=[], values=[])
    )


def parse_fetch_response(response: Message):
    json_response = json_format.MessageToDict(response)

    vd = {}
    vectors = json_response.get("vectors", {})
    namespace = json_response.get("namespace", "")

    for id, vec in vectors.items():
        vd[id] = _Vector(
            id=vec["id"],
            values=vec.get("values", None),
            sparse_values=parse_sparse_values(vec.get("sparseValues", None)),
            metadata=vec.get("metadata", None),
            _check_type=False,
        )

    return FetchResponse(
        vectors=vd, namespace=namespace, usage=parse_usage(json_response.get("usage", {}))
    )


def parse_usage(usage: dict):
    return Usage(read_units=int(usage.get("readUnits", 0)))


def parse_upsert_response(response: Message, _check_type: bool = False):
    json_response = json_format.MessageToDict(response)
    upserted_count = json_response.get("upsertedCount", 0)
    return UpsertResponse(upserted_count=int(upserted_count))


def parse_update_response(response: Union[dict, Message], _check_type: bool = False):
    return {}


def parse_delete_response(response: Union[dict, Message], _check_type: bool = False):
    return {}


def parse_query_response(response: Union[dict, Message], _check_type: bool = False):
    if isinstance(response, Message):
        json_response = json_format.MessageToDict(response)
    else:
        json_response = response

    matches = []
    for item in json_response.get("matches", []):
        sc = ScoredVector(
            id=item["id"],
            score=item.get("score", 0.0),
            values=item.get("values", []),
            sparse_values=parse_sparse_values(item.get("sparseValues")),
            metadata=item.get("metadata", None),
            _check_type=_check_type,
        )
        matches.append(sc)

    # Due to OpenAPI model classes / actual parsing cost, we want to avoid
    # creating empty `Usage` objects and then passing them into QueryResponse
    # when they are not actually present in the response from the server.
    args = {
        "namespace": json_response.get("namespace", ""),
        "matches": matches,
        "_check_type": _check_type,
    }
    usage = json_response.get("usage")
    if usage:
        args["usage"] = parse_usage(usage)
    return QueryResponse(**args)


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
