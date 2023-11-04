import uuid

from google.protobuf.struct_pb2 import Struct

from pinecone.core.client.models import (
    Vector as _Vector,
    ScoredVector,
    SparseValues,
    FetchResponse,
    SingleQueryResults,
    QueryResponse,
    DescribeIndexStatsResponse,
    NamespaceSummary,
)

from typing import NamedTuple, Optional

class QueryResponseKwargs(NamedTuple):
    check_type: bool
    namespace: Optional[str]
    matches: Optional[list]
    results: Optional[list]

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


def parse_fetch_response(response: dict):
    vd = {}
    vectors = response.get("vectors")
    if not vectors:
        return None
    for id, vec in vectors.items():
        v_obj = _Vector(
            id=vec["id"],
            values=vec["values"],
            sparse_values=parse_sparse_values(vec.get("sparseValues")),
            metadata=vec.get("metadata", None),
            _check_type=False,
        )
        vd[id] = v_obj
    namespace = response.get("namespace", "")
    return FetchResponse(vectors=vd, namespace=namespace, _check_type=False)


def parse_query_response(response: dict, unary_query: bool, _check_type: bool = False):
    res = []

    # TODO: consider deleting this deprecated case
    for match in response.get("results", []):
        namespace = match.get("namespace", "")
        m = []
        if "matches" in match:
            for item in match["matches"]:
                sc = ScoredVector(
                    id=item["id"],
                    score=item.get("score", 0.0),
                    values=item.get("values", []),
                    sparse_values=parse_sparse_values(item.get("sparseValues")),
                    metadata=item.get("metadata", {}),
                )
                m.append(sc)
        res.append(SingleQueryResults(matches=m, namespace=namespace))

    m = []
    for item in response.get("matches", []):
        sc = ScoredVector(
            id=item["id"],
            score=item.get("score", 0.0),
            values=item.get("values", []),
            sparse_values=parse_sparse_values(item.get("sparseValues")),
            metadata=item.get("metadata", {}),
            _check_type=_check_type,
        )
        m.append(sc)

    if unary_query:
        namespace = response.get("namespace", "")
        matches = m
        results = []
    else:
        namespace = ""
        matches = []
        results = res

    kw = QueryResponseKwargs(_check_type, namespace, matches, results)
    kw_dict = kw._asdict()
    kw_dict["_check_type"] = kw.check_type
    return QueryResponse(**kw._asdict())


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
