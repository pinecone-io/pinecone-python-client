from typing import List, Tuple
import json
import heapq
from pinecone.core.openapi.data.models import QueryResponse, Usage

from dataclasses import dataclass, asdict


@dataclass
class ScoredVectorWithNamespace:
    namespace: str
    score: float
    id: str
    values: List[float]
    sparse_values: dict
    metadata: dict

    def __init__(self, aggregate_results_heap_tuple: Tuple[float, int, dict, str]):
        json_vector = aggregate_results_heap_tuple[2]
        self.namespace = aggregate_results_heap_tuple[3]
        self.id = json_vector.get("id")
        self.score = json_vector.get("score")
        self.values = json_vector.get("values")
        self.sparse_values = json_vector.get("sparse_values", None)
        self.metadata = json_vector.get("metadata", None)

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"'{key}' not found in ScoredVectorWithNamespace")

    def __repr__(self):
        return json.dumps(self._truncate(asdict(self)), indent=4)

    def __json__(self):
        return self._truncate(asdict(self))

    def _truncate(self, obj, max_items=2):
        """
        Recursively traverse and truncate lists that exceed max_items length.
        Only display the "... X more" message if at least 2 elements are hidden.
        """
        if obj is None:
            return None  # Skip None values
        elif isinstance(obj, list):
            filtered_list = [self._truncate(i, max_items) for i in obj if i is not None]
            if len(filtered_list) > max_items:
                # Show the truncation message only if more than 1 item is hidden
                remaining_items = len(filtered_list) - max_items
                if remaining_items > 1:
                    return filtered_list[:max_items] + [f"... {remaining_items} more"]
                else:
                    # If only 1 item remains, show it
                    return filtered_list
            return filtered_list
        elif isinstance(obj, dict):
            # Recursively process dictionaries, omitting None values
            return {k: self._truncate(v, max_items) for k, v in obj.items() if v is not None}
        return obj


@dataclass
class CompositeQueryResults:
    usage: Usage
    matches: List[ScoredVectorWithNamespace]

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"'{key}' not found in CompositeQueryResults")

    def __repr__(self):
        return json.dumps(
            {
                "usage": self.usage.to_dict(),
                "matches": [match.__json__() for match in self.matches],
            },
            indent=4,
        )


class QueryResultsAggregator:
    def __init__(self, top_k: int):
        self.top_k = top_k
        self.usage_read_units = 0
        self.heap = []
        self.insertion_counter = 0
        self.read = False

    def add_results(self, results: QueryResponse):
        if self.read:
            raise ValueError("Results have already been read. Cannot add more results.")

        self.usage_read_units += results.get("usage", {}).get("readUnits", 0)
        ns = results.get("namespace")
        for match in results.get("matches", []):
            self.insertion_counter += 1
            score = match.get("score")
            if len(self.heap) < self.top_k:
                heapq.heappush(self.heap, (-score, -self.insertion_counter, match, ns))
            else:
                heapq.heappushpop(self.heap, (-score, -self.insertion_counter, match, ns))

    def get_results(self) -> CompositeQueryResults:
        if self.read:
            # This is mainly just to sanity check in test cases which get quite confusing
            # if you read results twice due to the heap being emptied each time you read
            # results into an ordered list.
            raise ValueError("Results have already been read. Cannot read again.")
        self.read = True

        return CompositeQueryResults(
            usage=Usage(read_units=self.usage_read_units),
            matches=[
                ScoredVectorWithNamespace(heapq.heappop(self.heap)) for _ in range(len(self.heap))
            ][::-1],
        )
