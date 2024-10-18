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


class QueryResultsAggregationEmptyResultsError(Exception):
    def __init__(self, namespace: str):
        super().__init__(
            f"Cannot infer metric type from empty query results. Query result for namespace '{namespace}' is empty. Have you spelled the namespace name correctly?"
        )


class QueryResultsAggregregatorNotEnoughResultsError(Exception):
    def __init__(self, top_k: int, num_results: int):
        super().__init__(
            f"Cannot interpret results without at least two matches. In order to aggregate results from multiple queries, top_k must be greater than 1 in order to correctly infer the similarity metric from scores. Expected at least {top_k} results but got {num_results}."
        )


class QueryResultsAggregatorInvalidTopKError(Exception):
    def __init__(self, top_k: int):
        super().__init__(f"Invalid top_k value {top_k}. top_k must be a positive integer.")


class QueryResultsAggregator:
    def __init__(self, top_k: int):
        if top_k < 1:
            raise QueryResultsAggregatorInvalidTopKError(top_k)
        self.top_k = top_k
        self.usage_read_units = 0
        self.heap = []
        self.insertion_counter = 0
        self.is_dotproduct = None
        self.read = False

    def __is_dotproduct_index(self, matches):
        # The interpretation of the score depends on the similar metric used.
        # Unlike other index types, in indexes configured for dotproduct,
        # a higher score is better. We have to infer this is the case by inspecting
        # the order of the scores in the results.
        for i in range(1, len(matches)):
            if matches[i].get("score") > matches[i - 1].get("score"):  # Found an increase
                return False
        return True

    def add_results(self, results: QueryResponse):
        if self.read:
            # This is mainly just to sanity check in test cases which get quite confusing
            # if you read results twice due to the heap being emptied when constructing
            # the ordered results.
            raise ValueError("Results have already been read. Cannot add more results.")

        matches = results.get("matches", [])
        ns = results.get("namespace")
        self.usage_read_units += results.get("usage", {}).get("readUnits", 0)

        if self.is_dotproduct is None:
            if len(matches) == 0:
                raise QueryResultsAggregationEmptyResultsError(ns)
            if len(matches) == 1:
                raise QueryResultsAggregregatorNotEnoughResultsError(self.top_k, len(matches))
            self.is_dotproduct = self.__is_dotproduct_index(matches)

        print("is_dotproduct:", self.is_dotproduct)
        if self.is_dotproduct:
            raise NotImplementedError("Dotproduct indexes are not yet supported.")
        else:
            for match in matches:
                self.insertion_counter += 1
                score = match.get("score")
                if len(self.heap) < self.top_k:
                    heapq.heappush(self.heap, (-score, -self.insertion_counter, match, ns))
                else:
                    heapq.heappushpop(self.heap, (-score, -self.insertion_counter, match, ns))

    def get_results(self) -> CompositeQueryResults:
        if self.read:
            return self.final_results
        self.read = True

        self.final_results = CompositeQueryResults(
            usage=Usage(read_units=self.usage_read_units),
            matches=[
                ScoredVectorWithNamespace(heapq.heappop(self.heap)) for _ in range(len(self.heap))
            ][::-1],
        )
        return self.final_results
