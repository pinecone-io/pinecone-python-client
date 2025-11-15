from typing import Optional, Union, Dict
from google.protobuf import json_format
from google.protobuf.message import Message

import uuid

from pinecone.core.openapi.db_data.models import (
    Vector as _Vector,
    Usage,
    ScoredVector,
    SparseValues,
    IndexDescription as DescribeIndexStatsResponse,
    NamespaceSummary,
    NamespaceDescription,
    NamespaceDescriptionIndexedFields,
    ListNamespacesResponse,
    Pagination as OpenApiPagination,
)
from pinecone.db_data.dataclasses import (
    FetchResponse,
    FetchByMetadataResponse,
    Vector,
    Pagination,
    QueryResponse,
    UpsertResponse,
)

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


def parse_fetch_response(response: Message, initial_metadata: Optional[Dict[str, str]] = None):
    json_response = json_format.MessageToDict(response)

    vd = {}
    vectors = json_response.get("vectors", {})
    namespace = json_response.get("namespace", "")

    for id, vec in vectors.items():
        # Convert to Vector dataclass
        sparse_vals = vec.get("sparseValues")
        parsed_sparse = None
        if sparse_vals:
            from pinecone.db_data.dataclasses import SparseValues

            parsed_sparse = SparseValues(
                indices=sparse_vals.get("indices", []), values=sparse_vals.get("values", [])
            )
        vd[id] = Vector(
            id=vec["id"],
            values=vec.get("values") or [],
            sparse_values=parsed_sparse,
            metadata=vec.get("metadata", None),
        )

    # Extract response info from initial metadata
    from pinecone.utils.response_info import extract_response_info

    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)

    fetch_response = FetchResponse(
        vectors=vd,
        namespace=namespace,
        usage=parse_usage(json_response.get("usage", {})),
        _response_info=response_info,
    )
    return fetch_response


def parse_fetch_by_metadata_response(
    response: Message, initial_metadata: Optional[Dict[str, str]] = None
):
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

    pagination = None
    if json_response.get("pagination") and json_response["pagination"].get("next"):
        pagination = Pagination(next=json_response["pagination"]["next"])

    # Extract response info from initial metadata
    from pinecone.utils.response_info import extract_response_info

    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)

    fetch_by_metadata_response = FetchByMetadataResponse(
        vectors=vd,
        namespace=namespace,
        usage=parse_usage(json_response.get("usage", {})),
        pagination=pagination,
        _response_info=response_info,
    )
    return fetch_by_metadata_response


def parse_usage(usage: dict):
    return Usage(read_units=int(usage.get("readUnits", 0)))


def parse_upsert_response(
    response: Message, _check_type: bool = False, initial_metadata: Optional[Dict[str, str]] = None
):
    from pinecone.utils.response_info import extract_response_info

    json_response = json_format.MessageToDict(response)
    upserted_count = json_response.get("upsertedCount", 0)

    # Extract response info from initial metadata
    # For gRPC, LSN headers are in initial_metadata
    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)

    return UpsertResponse(upserted_count=int(upserted_count), _response_info=response_info)


def parse_update_response(
    response: Union[dict, Message],
    _check_type: bool = False,
    initial_metadata: Optional[Dict[str, str]] = None,
):
    from pinecone.db_data.dataclasses import UpdateResponse
    from pinecone.utils.response_info import extract_response_info
    from google.protobuf import json_format

    # Extract response info from initial metadata
    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)

    # Extract matched_records from response
    matched_records = None
    if isinstance(response, Message):
        # GRPC response - convert to dict to extract matched_records
        json_response = json_format.MessageToDict(response)
        matched_records = json_response.get("matchedRecords") or json_response.get(
            "matched_records"
        )
    elif isinstance(response, dict):
        # Dict response - extract directly
        matched_records = response.get("matchedRecords") or response.get("matched_records")

    return UpdateResponse(matched_records=matched_records, _response_info=response_info)


def parse_delete_response(
    response: Union[dict, Message],
    _check_type: bool = False,
    initial_metadata: Optional[Dict[str, str]] = None,
):
    from pinecone.utils.response_info import extract_response_info

    # Extract response info from initial metadata
    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)

    result = {"_response_info": response_info}
    return result


def parse_query_response(
    response: Union[dict, Message],
    _check_type: bool = False,
    initial_metadata: Optional[Dict[str, str]] = None,
):
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
    args = {"namespace": json_response.get("namespace", ""), "matches": matches}
    usage = json_response.get("usage")
    if usage:
        args["usage"] = parse_usage(usage)

    # Extract response info from initial metadata
    # For gRPC, LSN headers are in initial_metadata
    from pinecone.utils.response_info import extract_response_info

    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)

    query_response = QueryResponse(**args, _response_info=response_info)
    return query_response


def parse_stats_response(response: dict):
    fullness = response.get("indexFullness", 0.0)
    total_vector_count = response.get("totalVectorCount", 0)
    # For sparse indexes, dimension is not present, so use None instead of 0
    dimension = response.get("dimension") if "dimension" in response else None
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


def parse_namespace_description(
    response: Message, initial_metadata: Optional[Dict[str, str]] = None
) -> NamespaceDescription:
    from pinecone.utils.response_info import extract_response_info

    json_response = json_format.MessageToDict(response)

    # Extract indexed_fields if present
    indexed_fields = None
    if "indexedFields" in json_response and json_response["indexedFields"]:
        indexed_fields_data = json_response["indexedFields"]
        if "fields" in indexed_fields_data:
            indexed_fields = NamespaceDescriptionIndexedFields(
                fields=indexed_fields_data.get("fields", []), _check_type=False
            )

    namespace_desc = NamespaceDescription(
        name=json_response.get("name", ""),
        record_count=json_response.get("recordCount", 0),
        indexed_fields=indexed_fields,
        _check_type=False,
    )

    # Attach _response_info as an attribute (NamespaceDescription is an OpenAPI model)
    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)
    namespace_desc._response_info = response_info

    return namespace_desc


def parse_list_namespaces_response(response: Message) -> ListNamespacesResponse:
    json_response = json_format.MessageToDict(response)

    namespaces = []
    for ns in json_response.get("namespaces", []):
        # Extract indexed_fields if present
        indexed_fields = None
        if "indexedFields" in ns and ns["indexedFields"]:
            indexed_fields_data = ns["indexedFields"]
            if "fields" in indexed_fields_data:
                indexed_fields = NamespaceDescriptionIndexedFields(
                    fields=indexed_fields_data.get("fields", []), _check_type=False
                )

        namespaces.append(
            NamespaceDescription(
                name=ns.get("name", ""),
                record_count=ns.get("recordCount", 0),
                indexed_fields=indexed_fields,
                _check_type=False,
            )
        )

    pagination = None
    if "pagination" in json_response and json_response["pagination"]:
        pagination = OpenApiPagination(
            next=json_response["pagination"].get("next", ""), _check_type=False
        )

    total_count = json_response.get("totalCount")
    return ListNamespacesResponse(
        namespaces=namespaces, pagination=pagination, total_count=total_count, _check_type=False
    )
