"""Fixtures for FTS data plane integration tests."""

import pytest
import uuid
import logging
import dotenv
from pinecone import Pinecone, TextField, IntegerField, DenseVectorField
from tests.integration.helpers import (
    delete_indexes_from_run,
    index_tags,
    embedding_values,
    poll_until_lsn_reconciled,
)

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

RUN_ID = str(uuid.uuid4())

FTS_INDEX_DIMENSION = 8


@pytest.fixture(scope="module")
def pc():
    """Create a Pinecone client."""
    return Pinecone()


@pytest.fixture(scope="module")
def fts_tags(request):
    """Generate tags for FTS test indexes."""
    return index_tags(request, RUN_ID)


@pytest.fixture(scope="module")
def fts_index_name():
    """Generate a unique index name for FTS tests."""
    return f"fts-data-{str(uuid.uuid4())[:8]}"


@pytest.fixture(scope="module")
def fts_index_host(pc: Pinecone, fts_index_name: str, fts_tags: dict):
    """Create an FTS-enabled index with schema and return its host.

    This creates an index with:
    - title: full-text searchable text field
    - description: full-text searchable text field
    - category: filterable text field
    - year: filterable integer field
    - embedding: dense vector field
    """
    schema = {
        "title": TextField(full_text_searchable=True),
        "description": TextField(full_text_searchable=True),
        "category": TextField(filterable=True),
        "year": IntegerField(filterable=True),
        "embedding": DenseVectorField(dimension=FTS_INDEX_DIMENSION, metric="cosine"),
    }

    logger.info(f"Creating FTS index {fts_index_name} with schema")
    pc.db.index.create(name=fts_index_name, schema=schema, tags=fts_tags)

    description = pc.db.index.describe(name=fts_index_name)
    host = description.host
    logger.info(f"FTS index {fts_index_name} created with host: {host}")

    yield host

    logger.info(f"Deleting FTS index {fts_index_name}")
    try:
        pc.db.index.delete(name=fts_index_name)
    except Exception as e:
        logger.warning(f"Failed to delete FTS index {fts_index_name}: {e}")


@pytest.fixture(scope="module")
def fts_index(pc: Pinecone, fts_index_name: str, fts_index_host: str):
    """Get an Index client for the FTS index."""
    return pc.Index(name=fts_index_name, host=fts_index_host)


@pytest.fixture(scope="module")
def seeded_fts_namespace(fts_index, fts_index_name):
    """Seed the FTS index with test documents and return the namespace.

    Returns the namespace that contains the seeded documents.
    """
    namespace = f"test-{str(uuid.uuid4())[:8]}"

    documents = [
        {
            "_id": "movie-1",
            "title": "Return of the Pink Panther",
            "description": "Inspector Clouseau investigates a diamond heist.",
            "category": "comedy",
            "year": 1975,
            "embedding": embedding_values(FTS_INDEX_DIMENSION),
        },
        {
            "_id": "movie-2",
            "title": "The Pink Panther Strikes Again",
            "description": "Clouseau's former boss tries to eliminate him.",
            "category": "comedy",
            "year": 1976,
            "embedding": embedding_values(FTS_INDEX_DIMENSION),
        },
        {
            "_id": "movie-3",
            "title": "Revenge of the Pink Panther",
            "description": "Clouseau is believed dead and investigates his own murder.",
            "category": "comedy",
            "year": 1978,
            "embedding": embedding_values(FTS_INDEX_DIMENSION),
        },
        {
            "_id": "movie-4",
            "title": "The Matrix",
            "description": "A hacker discovers the true nature of reality.",
            "category": "scifi",
            "year": 1999,
            "embedding": embedding_values(FTS_INDEX_DIMENSION),
        },
        {
            "_id": "movie-5",
            "title": "Blade Runner",
            "description": "A blade runner must pursue and terminate rogue replicants.",
            "category": "scifi",
            "year": 1982,
            "embedding": embedding_values(FTS_INDEX_DIMENSION),
        },
    ]

    logger.info(f"Upserting {len(documents)} documents to namespace {namespace}")
    upsert_response = fts_index.upsert_documents(namespace=namespace, documents=documents)

    poll_until_lsn_reconciled(fts_index, upsert_response._response_info, namespace=namespace)

    logger.info(f"Seeded namespace {namespace} with {len(documents)} documents")
    return namespace


def pytest_sessionfinish(session, exitstatus):
    """Clean up indexes created during the test session."""
    logger.info("Running final cleanup after FTS data plane tests...")
    pc = Pinecone()
    delete_indexes_from_run(pc, RUN_ID)
