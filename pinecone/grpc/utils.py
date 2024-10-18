from typing import Optional, Union
import uuid

from pinecone.core.openapi.data.models import (
    Vector as _Vector,
    Usage,
    ScoredVector,
    SparseValues,
    FetchResponse,
    QueryResponse,
    DescribeIndexStatsResponse,
    NamespaceSummary,
)
from pinecone.core.grpc.protos.vector_service_pb2 import SparseValues as GRPCSparseValues
from .sparse_vector import SparseVectorTypedDict

from google.protobuf.struct_pb2 import Struct


def _generate_request_id() -> str:
    return str(uuid.uuid4())


def normalize_endpoint(endpoint: str) -> str:
    grpc_host = endpoint.replace("https://", "")
    if ":" not in grpc_host:
        grpc_host = f"{grpc_host}:443"
    return grpc_host


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


def parse_fetch_response(response: dict):
    vd = {}
    vectors = response.get("vectors", {})
    namespace = response.get("namespace", "")

    for id, vec in vectors.items():
        vd[id] = _Vector(
            id=vec["id"],
            values=vec["values"],
            sparse_values=parse_sparse_values(vec.get("sparseValues")),
            metadata=vec.get("metadata", None),
            _check_type=False,
        )

    return FetchResponse(
        vectors=vd,
        namespace=namespace,
        usage=parse_usage(response.get("usage", {})),
        _check_type=False,
    )


def parse_usage(usage: dict):
    return Usage(read_units=int(usage.get("readUnits", 0)))


def parse_query_response(response: dict, _check_type: bool = False):
    matches = []
    for item in response.get("matches", []):
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
        "namespace": response.get("namespace", ""),
        "matches": matches,
        "_check_type": _check_type,
    }
    usage = response.get("usage")
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


def parse_sparse_values_arg(
    sparse_values: Optional[Union[GRPCSparseValues, SparseVectorTypedDict]],
) -> Optional[GRPCSparseValues]:
    if sparse_values is None:
        return None

    if isinstance(sparse_values, GRPCSparseValues):
        return sparse_values

    if (
        not isinstance(sparse_values, dict)
        or "indices" not in sparse_values
        or "values" not in sparse_values
    ):
        raise ValueError(
            "Invalid sparse values argument. Expected a dict of: {'indices': List[int], 'values': List[float]}."
            f"Received: {sparse_values}"
        )

    return GRPCSparseValues(indices=sparse_values["indices"], values=sparse_values["values"])
