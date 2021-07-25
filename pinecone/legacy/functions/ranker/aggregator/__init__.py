#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from typing import Optional

import numpy as np

from pinecone.legacy.functions.ranker import Ranker


class Aggregator(Ranker):

    def score(self, q: np.ndarray, vectors: np.ndarray, prev_scores: np.ndarray) -> np.ndarray:
        return prev_scores

    @property
    def volume_request(self) -> Optional[int]:
        return 0

    @property
    def image(self):
        return 'pinecone/nuts'
