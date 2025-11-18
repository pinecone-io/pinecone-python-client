from tests.integration.helpers import embedding_values, poll_until_lsn_reconciled
from pinecone import Vector
import logging

logger = logging.getLogger(__name__)


def setup_data(idx, target_namespace, wait):
    # Upsert without metadata
    logger.info(
        "Upserting 3 vectors as tuples to namespace '%s' without metadata", target_namespace
    )
    upsert1 = idx.upsert(
        vectors=[
            ("1", embedding_values(2)),
            ("2", embedding_values(2)),
            ("3", embedding_values(2)),
        ],
        namespace=target_namespace,
    )

    # Upsert with metadata
    logger.info(
        "Upserting 3 vectors as Vector objects to namespace '%s' with metadata", target_namespace
    )
    upsert2 = idx.upsert(
        vectors=[
            Vector(
                id="4", values=embedding_values(2), metadata={"genre": "action", "runtime": 120}
            ),
            Vector(id="5", values=embedding_values(2), metadata={"genre": "comedy", "runtime": 90}),
            Vector(
                id="6", values=embedding_values(2), metadata={"genre": "romance", "runtime": 240}
            ),
        ],
        namespace=target_namespace,
    )

    # Upsert with dict
    logger.info("Upserting 3 vectors as dicts to namespace '%s'", target_namespace)
    upsert3 = idx.upsert(
        vectors=[
            {"id": "7", "values": embedding_values(2)},
            {"id": "8", "values": embedding_values(2)},
            {"id": "9", "values": embedding_values(2)},
        ],
        namespace=target_namespace,
    )

    poll_until_lsn_reconciled(idx, upsert1._response_info, namespace=target_namespace)
    poll_until_lsn_reconciled(idx, upsert2._response_info, namespace=target_namespace)
    poll_until_lsn_reconciled(idx, upsert3._response_info, namespace=target_namespace)
