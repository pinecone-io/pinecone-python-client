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

    kwargs = {"_check_type": _check_type}
    if unary_query:
        kwargs["namespace"] = response.get("namespace", "")
        kwargs["matches"] = m
    else:
        kwargs["results"] = res
    return QueryResponse(**kwargs)


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
