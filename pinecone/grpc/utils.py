from __future__ import annotations

from typing import Any, TYPE_CHECKING
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
    UpdateResponse,
)

from google.protobuf.struct_pb2 import Struct

if TYPE_CHECKING:
    from pinecone.core.grpc.protos.db_data_2025_10_pb2 import (
        FetchResponse as ProtoFetchResponse,
        FetchByMetadataResponse as ProtoFetchByMetadataResponse,
        QueryResponse as ProtoQueryResponse,
        UpsertResponse as ProtoUpsertResponse,
        UpdateResponse as ProtoUpdateResponse,
        NamespaceDescription as ProtoNamespaceDescription,
        ListNamespacesResponse as ProtoListNamespacesResponse,
        DescribeIndexStatsResponse as ProtoDescribeIndexStatsResponse,
        Usage as ProtoUsage,
    )


def _generate_request_id() -> str:
    return str(uuid.uuid4())


def dict_to_proto_struct(d: dict | None) -> "Struct":
    if not d:
        d = {}
    s = Struct()
    s.update(d)
    return s


def _struct_to_dict(struct: "Struct") -> dict[str, Any]:
    """Convert a protobuf Struct to dict by directly accessing fields.

    This optimized version is ~2x faster than json_format.MessageToDict
    by avoiding JSON serialization/deserialization overhead.

    Args:
        struct: A protobuf Struct message.

    Returns:
        Dictionary representation of the Struct.
    """

    result: dict[str, Any] = {}
    for key, value in struct.fields.items():
        # Directly access the Value fields based on which one is set
        if value.HasField("null_value"):
            result[key] = None
        elif value.HasField("number_value"):
            result[key] = value.number_value
        elif value.HasField("string_value"):
            result[key] = value.string_value
        elif value.HasField("bool_value"):
            result[key] = value.bool_value
        elif value.HasField("struct_value"):
            result[key] = _struct_to_dict(value.struct_value)
        elif value.HasField("list_value"):
            # Convert ListValue to Python list
            list_result: list[Any] = []
            for item in value.list_value.values:
                if item.HasField("null_value"):
                    list_result.append(None)
                elif item.HasField("number_value"):
                    list_result.append(item.number_value)
                elif item.HasField("string_value"):
                    list_result.append(item.string_value)
                elif item.HasField("bool_value"):
                    list_result.append(item.bool_value)
                elif item.HasField("struct_value"):
                    list_result.append(_struct_to_dict(item.struct_value))
                elif item.HasField("list_value"):
                    # Nested lists
                    nested_list: list[Any] = []
                    for nested_item in item.list_value.values:
                        if nested_item.HasField("number_value"):
                            nested_list.append(nested_item.number_value)
                        elif nested_item.HasField("string_value"):
                            nested_list.append(nested_item.string_value)
                        elif nested_item.HasField("bool_value"):
                            nested_list.append(nested_item.bool_value)
                    list_result.append(nested_list)
            result[key] = list_result
    return result


def parse_sparse_values(sparse_values: dict | None) -> SparseValues:
    from typing import cast

    result = (
        SparseValues(indices=sparse_values["indices"], values=sparse_values["values"])
        if sparse_values
        else SparseValues(indices=[], values=[])
    )
    return cast(SparseValues, result)


def parse_fetch_response(
    response: "ProtoFetchResponse", initial_metadata: dict[str, str] | None = None
) -> FetchResponse:
    """Parse a FetchResponse protobuf message directly without MessageToDict conversion.

    This optimized version directly accesses protobuf fields for better performance.
    """
    # Extract response info from initial metadata
    from pinecone.utils.response_info import extract_response_info
    from pinecone.db_data.dataclasses import SparseValues

    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)

    # Directly access protobuf fields instead of converting entire message to dict
    vectors = response.vectors
    vd = {}
    # namespace is a required string field, so it will always have a value (default empty string)
    namespace = response.namespace

    # Iterate over vectors map directly
    for vec_id, vec in vectors.items():
        # Convert vector.values (RepeatedScalarFieldContainer) to list
        values = list(vec.values) if vec.values else []

        # Handle sparse_values if present (check if field is set and not empty)
        parsed_sparse = None
        if vec.HasField("sparse_values") and vec.sparse_values:
            parsed_sparse = SparseValues(
                indices=list(vec.sparse_values.indices), values=list(vec.sparse_values.values)
            )

        # Convert metadata Struct to dict only when needed using optimized conversion
        metadata_dict = None
        if vec.HasField("metadata") and vec.metadata:
            metadata_dict = _struct_to_dict(vec.metadata)

        vd[vec_id] = Vector(
            id=vec.id, values=values, sparse_values=parsed_sparse, metadata=metadata_dict
        )

    # Parse usage if present (usage is optional, so check HasField)
    usage = None
    if response.HasField("usage") and response.usage:
        usage = parse_usage(response.usage)

    fetch_response = FetchResponse(
        vectors=vd, namespace=namespace, usage=usage, _response_info=response_info
    )
    return fetch_response


def parse_fetch_by_metadata_response(
    response: "ProtoFetchByMetadataResponse", initial_metadata: dict[str, str] | None = None
) -> FetchByMetadataResponse:
    """Parse a FetchByMetadataResponse protobuf message directly without MessageToDict conversion.

    This optimized version directly accesses protobuf fields for better performance.
    """
    # Extract response info from initial metadata
    from pinecone.utils.response_info import extract_response_info
    from pinecone.db_data.dataclasses import SparseValues

    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)

    # Directly access protobuf fields instead of converting entire message to dict
    vectors = response.vectors
    vd = {}
    # namespace is a required string field, so it will always have a value (default empty string)
    namespace = response.namespace

    # Iterate over vectors map directly
    for vec_id, vec in vectors.items():
        # Convert vector.values (RepeatedScalarFieldContainer) to list
        values = list(vec.values) if vec.values else None

        # Handle sparse_values if present - optimize by creating SparseValues directly
        parsed_sparse = None
        if vec.HasField("sparse_values") and vec.sparse_values:
            parsed_sparse = SparseValues(
                indices=list(vec.sparse_values.indices), values=list(vec.sparse_values.values)
            )

        # Convert metadata Struct to dict only when needed using optimized conversion
        metadata_dict = None
        if vec.HasField("metadata") and vec.metadata:
            metadata_dict = _struct_to_dict(vec.metadata)

        vd[vec_id] = _Vector(
            id=vec.id,
            values=values,
            sparse_values=parsed_sparse,
            metadata=metadata_dict,
            _check_type=False,
        )

    # Parse pagination if present
    pagination = None
    if response.HasField("pagination") and response.pagination:
        pagination = Pagination(next=response.pagination.next)

    # Parse usage if present
    usage = None
    if response.HasField("usage") and response.usage:
        usage = parse_usage(response.usage)

    fetch_by_metadata_response = FetchByMetadataResponse(
        vectors=vd,
        namespace=namespace,
        usage=usage,
        pagination=pagination,
        _response_info=response_info,
    )
    return fetch_by_metadata_response


def parse_usage(usage: "ProtoUsage") -> Usage:
    """Parse a Usage protobuf message to Usage model.

    This optimized version directly accesses protobuf fields for better performance.

    Args:
        usage: ProtoUsage protobuf message.

    Returns:
        Usage dataclass instance.
    """
    from typing import cast

    return cast(Usage, Usage(read_units=usage.read_units))


def parse_upsert_response(
    response: "ProtoUpsertResponse",
    _check_type: bool = False,
    initial_metadata: dict[str, str] | None = None,
) -> UpsertResponse:
    """Parse an UpsertResponse protobuf message directly without MessageToDict conversion.

    This optimized version directly accesses protobuf fields for better performance.
    """
    from pinecone.utils.response_info import extract_response_info

    # Extract response info from initial metadata
    # For gRPC, LSN headers are in initial_metadata
    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)

    # Directly access upserted_count field (required field in proto3, always has a value)
    upserted_count = response.upserted_count

    return UpsertResponse(upserted_count=int(upserted_count), _response_info=response_info)


def parse_update_response(
    response: dict | "ProtoUpdateResponse",
    _check_type: bool = False,
    initial_metadata: dict[str, str] | None = None,
) -> UpdateResponse:
    """Parse an UpdateResponse protobuf message directly without MessageToDict conversion.

    This optimized version directly accesses protobuf fields for better performance.
    For dict responses (REST API), falls back to the original dict-based parsing.
    """
    from pinecone.utils.response_info import extract_response_info

    # Extract response info from initial metadata
    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)

    # Extract matched_records from response
    matched_records = None
    if isinstance(response, Message) and not isinstance(response, dict):
        # Optimized path: directly access protobuf field
        matched_records = response.matched_records if response.HasField("matched_records") else None
    elif isinstance(response, dict):
        # Fallback for dict responses (REST API)
        matched_records = response.get("matchedRecords") or response.get("matched_records")

    return UpdateResponse(matched_records=matched_records, _response_info=response_info)


def parse_delete_response(
    response: dict | Message,
    _check_type: bool = False,
    initial_metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    from pinecone.utils.response_info import extract_response_info

    # Extract response info from initial metadata
    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)

    result: dict[str, Any] = {"_response_info": response_info}
    return result


def query_response_to_dict(response: "ProtoQueryResponse") -> dict[str, Any]:
    """Convert a QueryResponse protobuf message to dict format for aggregator.

    This optimized version directly accesses protobuf fields instead of using MessageToDict.
    Only converts the fields needed by the aggregator.
    """
    result: dict[str, Any] = {"namespace": response.namespace, "matches": []}

    # Convert matches
    for match in response.matches:
        match_dict: dict[str, Any] = {"id": match.id, "score": match.score}

        # Convert values if present
        if match.values:
            match_dict["values"] = list(match.values)

        # Convert sparse_values if present
        if match.HasField("sparse_values") and match.sparse_values:
            match_dict["sparseValues"] = {
                "indices": list(match.sparse_values.indices),
                "values": list(match.sparse_values.values),
            }

        # Convert metadata if present using optimized conversion
        if match.HasField("metadata") and match.metadata:
            match_dict["metadata"] = _struct_to_dict(match.metadata)

        result["matches"].append(match_dict)

    # Convert usage if present
    if response.HasField("usage") and response.usage:
        result["usage"] = {"readUnits": response.usage.read_units}

    return result


def parse_query_response(
    response: "ProtoQueryResponse",
    _check_type: bool = False,
    initial_metadata: dict[str, str] | None = None,
) -> QueryResponse:
    """Parse a QueryResponse protobuf message directly without MessageToDict conversion.

    This optimized version directly accesses protobuf fields for better performance.

    Args:
        response: ProtoQueryResponse protobuf message.
        _check_type: Whether to check types when creating ScoredVector objects.
        initial_metadata: Initial metadata from gRPC call containing response info.

    Returns:
        QueryResponse dataclass instance.
    """
    # Extract response info from initial metadata
    from pinecone.utils.response_info import extract_response_info

    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)

    # Directly access protobuf fields
    # Pre-allocate matches list with known size for better performance
    matches_proto = response.matches
    matches: list[ScoredVector] = [None] * len(matches_proto) if matches_proto else []  # type: ignore[list-item]
    # namespace is a required string field, so it will always have a value (default empty string)
    namespace = response.namespace

    # Iterate over matches directly
    for idx, match in enumerate(matches_proto):
        # Convert match.values (RepeatedScalarFieldContainer) to list
        # Optimize: only convert if values exist, avoid creating empty list unnecessarily
        values = list(match.values) if match.values else []

        # Handle sparse_values if present (check if field is set and not empty)
        parsed_sparse = None
        if match.HasField("sparse_values") and match.sparse_values:
            parsed_sparse = SparseValues(
                indices=list(match.sparse_values.indices), values=list(match.sparse_values.values)
            )

        # Convert metadata Struct to dict only when needed using optimized conversion
        metadata_dict = None
        if match.HasField("metadata") and match.metadata:
            metadata_dict = _struct_to_dict(match.metadata)

        matches[idx] = ScoredVector(
            id=match.id,
            score=match.score,
            values=values,
            sparse_values=parsed_sparse,
            metadata=metadata_dict,
            _check_type=_check_type,
        )

    # Parse usage if present (usage is optional, so check HasField)
    usage = None
    if response.HasField("usage") and response.usage:
        usage = parse_usage(response.usage)

    query_response = QueryResponse(
        namespace=namespace, matches=matches, usage=usage, _response_info=response_info
    )
    return query_response


def parse_stats_response(
    response: dict | "ProtoDescribeIndexStatsResponse",
) -> "DescribeIndexStatsResponse":
    """Parse a DescribeIndexStatsResponse protobuf message directly without MessageToDict conversion.

    This optimized version directly accesses protobuf fields for better performance.
    For dict responses (REST API), falls back to the original dict-based parsing.
    """
    from typing import cast

    if isinstance(response, Message) and not isinstance(response, dict):
        # Optimized path: directly access protobuf fields
        # For sparse indexes, dimension is not present, so use None instead of 0
        dimension = None
        if response.HasField("dimension"):
            dimension = response.dimension

        # Extract index_fullness (required float field)
        index_fullness = response.index_fullness

        # Extract total_vector_count (required int field)
        total_vector_count = response.total_vector_count

        # Extract namespaces map - pre-allocate dict with known size
        namespaces_proto = response.namespaces
        namespace_summaries = {}
        for ns_name, ns_summary in namespaces_proto.items():
            namespace_summaries[ns_name] = NamespaceSummary(vector_count=ns_summary.vector_count)

        result = DescribeIndexStatsResponse(
            namespaces=namespace_summaries,
            dimension=dimension,
            index_fullness=index_fullness,
            total_vector_count=total_vector_count,
            _check_type=False,
        )
        return cast(DescribeIndexStatsResponse, result)
    else:
        # Fallback for dict responses (REST API)
        fullness = response.get("indexFullness", 0.0)
        total_vector_count = response.get("totalVectorCount", 0)
        # For sparse indexes, dimension is not present, so use None instead of 0
        dimension = response.get("dimension") if "dimension" in response else None
        summaries = response.get("namespaces", {})
        namespace_summaries = {}
        for key in summaries:
            vc = summaries[key].get("vectorCount", 0)
            namespace_summaries[key] = NamespaceSummary(vector_count=vc)

        result = DescribeIndexStatsResponse(
            namespaces=namespace_summaries,
            dimension=dimension,
            index_fullness=fullness,
            total_vector_count=total_vector_count,
            _check_type=False,
        )
        return cast(DescribeIndexStatsResponse, result)


def parse_namespace_description(
    response: "ProtoNamespaceDescription", initial_metadata: dict[str, str] | None = None
) -> NamespaceDescription:
    """Parse a NamespaceDescription protobuf message directly without MessageToDict conversion.

    This optimized version directly accesses protobuf fields for better performance.
    """
    from pinecone.utils.response_info import extract_response_info

    # Directly access protobuf fields
    name = response.name
    record_count = response.record_count

    # Extract indexed_fields if present
    indexed_fields = None
    if response.HasField("indexed_fields") and response.indexed_fields:
        # Access indexed_fields.fields directly (RepeatedScalarFieldContainer)
        fields_list = list(response.indexed_fields.fields) if response.indexed_fields.fields else []
        if fields_list:
            indexed_fields = NamespaceDescriptionIndexedFields(
                fields=fields_list, _check_type=False
            )

    namespace_desc = NamespaceDescription(
        name=name, record_count=record_count, indexed_fields=indexed_fields, _check_type=False
    )

    # Attach _response_info as an attribute (NamespaceDescription is an OpenAPI model)
    metadata = initial_metadata or {}
    response_info = extract_response_info(metadata)
    namespace_desc._response_info = response_info

    from typing import cast

    return cast(NamespaceDescription, namespace_desc)


def parse_list_namespaces_response(
    response: "ProtoListNamespacesResponse",
) -> ListNamespacesResponse:
    """Parse a ListNamespacesResponse protobuf message directly without MessageToDict conversion.

    This optimized version directly accesses protobuf fields for better performance.
    """
    # Directly iterate over namespaces
    # Pre-allocate namespaces list with known size for better performance
    namespaces_proto = response.namespaces
    namespaces = [None] * len(namespaces_proto) if namespaces_proto else []
    for idx, ns in enumerate(namespaces_proto):
        # Extract indexed_fields if present
        indexed_fields = None
        if ns.HasField("indexed_fields") and ns.indexed_fields:
            # Access indexed_fields.fields directly (RepeatedScalarFieldContainer)
            fields_list = list(ns.indexed_fields.fields) if ns.indexed_fields.fields else []
            if fields_list:
                indexed_fields = NamespaceDescriptionIndexedFields(
                    fields=fields_list, _check_type=False
                )

        namespaces[idx] = NamespaceDescription(
            name=ns.name,
            record_count=ns.record_count,
            indexed_fields=indexed_fields,
            _check_type=False,
        )

    # Parse pagination if present
    pagination = None
    if response.HasField("pagination") and response.pagination:
        pagination = OpenApiPagination(next=response.pagination.next, _check_type=False)

    # Parse total_count (int field in proto3, always has a value, default 0)
    # If 0, treat as None to match original behavior
    total_count = response.total_count if response.total_count else None

    from typing import cast

    result = ListNamespacesResponse(
        namespaces=namespaces, pagination=pagination, total_count=total_count, _check_type=False
    )
    return cast(ListNamespacesResponse, result)
