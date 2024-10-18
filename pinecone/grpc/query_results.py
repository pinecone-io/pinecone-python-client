from typing import TypedDict, List, Dict, Any


class ScoredVectorTypedDict(TypedDict):
    id: str
    score: float
    values: List[float]
    metadata: dict


class QueryResultsTypedDict(TypedDict):
    matches: List[ScoredVectorTypedDict]
    namespace: str
    usage: Dict[str, Any]
