"""Test to verify LSN header behavior for sparse vs dense indices.

This test verifies that sparse indices may not return x-pinecone-max-indexed-lsn
headers in query responses, which explains why LSN polling fails for sparse indices.
"""

import logging
from tests.integration.helpers import embedding_values, random_string
from tests.integration.helpers.helpers import get_query_response
from tests.integration.helpers.lsn_utils import extract_lsn_reconciled, extract_lsn_committed

logger = logging.getLogger(__name__)


def test_verify_lsn_headers_dense_vs_sparse(idx, sparse_idx):
    """Verify that dense indices return LSN headers but sparse indices may not.

    This test helps verify the hypothesis that sparse indices don't return
    x-pinecone-max-indexed-lsn headers in query responses.
    """
    test_namespace = random_string(10)

    # Upsert to dense index
    logger.info("Upserting to dense index...")
    dense_upsert = idx.upsert(vectors=[("dense-1", embedding_values(2))], namespace=test_namespace)
    dense_committed_lsn = extract_lsn_committed(dense_upsert._response_info.get("raw_headers", {}))
    logger.info(f"Dense index upsert - committed LSN: {dense_committed_lsn}")
    logger.info(
        f"Dense index upsert - all headers: {list(dense_upsert._response_info.get('raw_headers', {}).keys())}"
    )

    # Query dense index
    logger.info("Querying dense index...")
    dense_query = get_query_response(idx, test_namespace, dimension=2)
    dense_reconciled_lsn = extract_lsn_reconciled(dense_query._response_info.get("raw_headers", {}))
    logger.info(f"Dense index query - reconciled LSN: {dense_reconciled_lsn}")
    logger.info(
        f"Dense index query - all headers: {list(dense_query._response_info.get('raw_headers', {}).keys())}"
    )

    # Upsert to sparse index
    logger.info("Upserting to sparse index...")
    from pinecone import Vector, SparseValues

    sparse_upsert = sparse_idx.upsert(
        vectors=[
            Vector(id="sparse-1", sparse_values=SparseValues(indices=[0, 1], values=[0.5, 0.5]))
        ],
        namespace=test_namespace,
    )
    sparse_committed_lsn = extract_lsn_committed(
        sparse_upsert._response_info.get("raw_headers", {})
    )
    logger.info(f"Sparse index upsert - committed LSN: {sparse_committed_lsn}")
    logger.info(
        f"Sparse index upsert - all headers: {list(sparse_upsert._response_info.get('raw_headers', {}).keys())}"
    )

    # Query sparse index
    logger.info("Querying sparse index...")
    sparse_query = get_query_response(sparse_idx, test_namespace, dimension=None)
    sparse_reconciled_lsn = extract_lsn_reconciled(
        sparse_query._response_info.get("raw_headers", {})
    )
    logger.info(f"Sparse index query - reconciled LSN: {sparse_reconciled_lsn}")
    logger.info(
        f"Sparse index query - all headers: {list(sparse_query._response_info.get('raw_headers', {}).keys())}"
    )

    # Assertions
    assert dense_committed_lsn is not None, "Dense index should return committed LSN in upsert"
    assert dense_reconciled_lsn is not None, "Dense index should return reconciled LSN in query"

    assert sparse_committed_lsn is not None, "Sparse index should return committed LSN in upsert"

    # This is the key assertion - sparse indices may not return reconciled LSN
    if sparse_reconciled_lsn is None:
        logger.warning(
            "Sparse index does not return x-pinecone-max-indexed-lsn header in query response. "
            "This explains why LSN polling fails for sparse indices."
        )
    else:
        logger.info("Sparse index does return reconciled LSN header (unexpected)")
