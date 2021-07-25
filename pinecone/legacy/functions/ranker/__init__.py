#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from abc import abstractmethod
from collections import defaultdict
from typing import Iterable

import numpy as np
from pinecone.legacy.functions import Function
from pinecone.protos import core_pb2
from pinecone.utils import load_numpy, dump_numpy


class Ranker(Function):
    """
    Encapsulates aggregating and ranking logic in the execution plane
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msgs = dict()
        self.shard_counts = defaultdict(set)

    @abstractmethod
    def score(self, q: np.ndarray, vectors: np.ndarray, prev_scores: np.ndarray) -> np.ndarray:
        """
        :param q:
        :param vectors:
        :param prev_scores: Existing scores if present
        :return:
        """
        raise NotImplementedError

    def top_k_idx(self, scores: np.ndarray, k: int) -> Iterable[int]:
        topk_idx = np.argpartition(scores, -k)[-k:]
        sorted_topk_idx = topk_idx[scores[topk_idx].argsort()[::-1]]
        return sorted_topk_idx

    @staticmethod
    def merge_fetch_msg(orig: 'core_pb2.Request', msg: 'core_pb2.Request') -> 'core_pb2.Request':
        for i, vector in enumerate(msg.fetch.vectors):
            if sum(vector.shape) > 0:
                orig.fetch.vectors[i].CopyFrom(vector)
        return orig

    @staticmethod
    def merge_nd_arrays(orig: 'core_pb2.NdArray', new: 'core_pb2.NdArray'):
        orig_data = load_numpy(orig)
        new_data = load_numpy(new)
        if len(new_data) == 0:
            pass
        elif len(orig_data) == 0:
            orig.CopyFrom(dump_numpy(new_data))
        else:
            orig.CopyFrom(dump_numpy(
                np.concatenate((orig_data, new_data))))

    @staticmethod
    def merge_query_msg(orig: 'core_pb2.Request', msg: 'core_pb2.Request') -> 'core_pb2.Request':
        for orig_matches, new_matches in zip(orig.query.matches, msg.query.matches):
            Ranker.merge_nd_arrays(orig_matches.ids, new_matches.ids)
            Ranker.merge_nd_arrays(orig_matches.scores, new_matches.scores)
            if orig.query.include_data:
                Ranker.merge_nd_arrays(orig_matches.data, new_matches.data)

        orig.routes.append(msg.routes.pop(-1))
        return orig

    @staticmethod
    def merge_info_msg(orig: 'core_pb2.Request', msg: 'core_pb2.Request') -> 'core_pb2.Request':
        orig.info.index_size += msg.info.index_size
        return orig

    @staticmethod
    def merge_delete_msg(orig: 'core_pb2.Request', msg: 'core_pb2.Request') -> 'core_pb2.Request':
        orig.delete.ids.extend(msg.delete.ids)
        return orig

    @staticmethod
    def merge_index_msg(orig: 'core_pb2.Request', msg: 'core_pb2.Request') -> 'core_pb2.Request':
        orig.index.ids.extend(msg.index.ids)
        return orig

    @staticmethod
    def merge_list_msg(orig: 'core_pb2.Request', msg: 'core_pb2.Request') -> 'core_pb2.Request':
        if orig.list.resource_type == 'ids':
            Ranker.merge_nd_arrays(orig.list.items, msg.list.items)
        return orig

    def rank_query_msg(self, query: 'core_pb2.QueryRequest'):
        if len(query.top_k_overrides) > 0:
            top_ks = [query.top_k] * len(query.matches)
        else:
            top_ks = query.top_k_overrides
        queries_vectors = load_numpy(query.data)
        for matches, q, k in zip(query.matches, queries_vectors, top_ks):
            q = q.reshape(1, -1)
            ids = load_numpy(matches.ids)
            if len(ids) == 0:  # nothing to rank
                continue
            embeddings = load_numpy(matches.data)
            scores = load_numpy(matches.scores)
            scores = self.score(q, embeddings, scores)
            k = min(k, len(scores))  # for k>n case
            topk_idx = self.top_k_idx(scores, k)
            sorted_ids = ids[topk_idx]
            sorted_scores = scores[topk_idx]
            if len(embeddings) > 0 and query.include_data:
                sorted_embeddings = dump_numpy(embeddings[topk_idx, :])
                matches.CopyFrom(core_pb2.ScoredResults(
                    ids=dump_numpy(sorted_ids), scores=dump_numpy(sorted_scores), data=sorted_embeddings))
            else:
                matches.CopyFrom(core_pb2.ScoredResults(
                    ids=dump_numpy(sorted_ids), scores=dump_numpy(sorted_scores)))

    def merge_msg(self, msg: 'core_pb2.Request'):
        request_type = msg.WhichOneof('body')
        if self.msgs.get(msg.request_id) is None:
            self.msgs[msg.request_id] = msg
        elif request_type == 'query':
            self.msgs[msg.request_id] = Ranker.merge_query_msg(
                self.msgs[msg.request_id], msg)
        elif request_type == 'fetch':
            self.msgs[msg.request_id] = Ranker.merge_fetch_msg(
                self.msgs[msg.request_id], msg)
        elif request_type == 'info':
            self.msgs[msg.request_id] = Ranker.merge_info_msg(
                self.msgs[msg.request_id], msg)
        elif request_type == 'index':
            self.msgs[msg.request_id] = Ranker.merge_index_msg(
                self.msgs[msg.request_id], msg)
        elif request_type == 'delete':
            self.msgs[msg.request_id] = Ranker.merge_delete_msg(
                self.msgs[msg.request_id], msg)
        elif request_type == 'list':
            self.msgs[msg.request_id] = Ranker.merge_list_msg(
                self.msgs[msg.request_id], msg)

    def handle_msg(self, msg: 'core_pb2.Request') -> 'core_pb2.Request':
        if msg.shard_num in self.shard_counts[msg.request_id]:
            return
        self.shard_counts[msg.request_id].add(msg.shard_num)
        self.merge_msg(msg)
        if len(self.shard_counts[msg.request_id]) == msg.num_shards:
            result = self.msgs[msg.request_id]
            del self.msgs[msg.request_id]
            del self.shard_counts[msg.request_id]
            if msg.WhichOneof('body') == 'query':
                self.rank_query_msg(result.query)
            return result
