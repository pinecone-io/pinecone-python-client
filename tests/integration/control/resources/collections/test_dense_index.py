import time
from pinecone import PodSpec
from ....helpers import generate_index_name, generate_collection_name
import logging
from .helpers import attempt_cleanup_collection, attempt_cleanup_index, random_vector

logger = logging.getLogger(__name__)


class TestCollectionsHappyPath:
    def test_dense_index_to_collection_to_index(self, pc, pod_environment, index_tags):
        # Create a pod index
        index_name = generate_index_name("pod-index")
        dimension = 10
        metric = "cosine"
        pod_index = pc.db.index.create(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=PodSpec(environment=pod_environment),
            tags=index_tags,
        )

        # Insert some vectors into the pod index
        idx = pc.Index(host=pod_index.host)
        num_vectors = 10
        namespaces = ["", "test-ns1", "test-ns2"]
        for namespace in namespaces:
            vectors = [(str(i), random_vector(dimension)) for i in range(num_vectors)]
            idx.upsert(vectors=vectors, namespace=namespace)

        # Wait for the vectors to be available
        all_vectors_available = False
        max_wait = 180
        time_waited = 0
        while not all_vectors_available and time_waited < max_wait:
            all_vectors_available = True
            desc = idx.describe_index_stats()
            for namespace in namespaces:
                if (
                    desc.namespaces.get(namespace, None) is None
                    or desc.namespaces[namespace]["vector_count"] != num_vectors
                ):
                    logger.debug(f"Waiting for vectors to be available in namespace {namespace}...")
                    all_vectors_available = False
                    break
            for namespace in namespaces:
                for i in range(num_vectors):
                    try:
                        idx.fetch(ids=[str(i)], namespace=namespace)
                    except Exception:
                        logger.debug(
                            f"Waiting for vector {i} to be available in namespace {namespace}..."
                        )
                        all_vectors_available = False
                        break
            if not all_vectors_available:
                time.sleep(5)
                time_waited += 5
        if not all_vectors_available:
            raise Exception(f"Vectors were not available after {max_wait} seconds")

        # Create a collection from the pod index
        collection_name = generate_collection_name("coll1")
        pc.db.collection.create(name=collection_name, source=index_name)
        collection_desc = pc.db.collection.describe(name=collection_name)
        logger.debug(f"Collection desc: {collection_desc}")
        assert collection_desc["name"] == collection_name
        assert collection_desc["environment"] == pod_environment
        assert collection_desc["status"] is not None

        # Wait for the collection to be ready
        time_waited = 0
        max_wait = 120
        collection_ready = collection_desc["status"]
        while collection_ready.lower() != "ready" and time_waited < max_wait:
            logger.debug(
                f"Waiting for collection {collection_name} to be ready. Waited {time_waited} seconds..."
            )
            desc = pc.db.collection.describe(name=collection_name)
            logger.debug(f"Collection desc: {desc}")
            collection_ready = desc["status"]
            if collection_ready.lower() != "ready":
                time.sleep(10)
                time_waited += 10
        if collection_ready.lower() != "ready":
            raise Exception(f"Collection {collection_name} is not ready after {max_wait} seconds")

        # Verify the collection was created
        assert collection_name in pc.db.collection.list().names()

        # Verify the collection has the correct info
        collection_desc = pc.db.collection.describe(name=collection_name)
        logger.debug(f"Collection desc: {collection_desc}")
        assert collection_desc["name"] == collection_name
        assert collection_desc["environment"] == pod_environment
        assert collection_desc["status"] == "Ready"
        assert collection_desc["dimension"] == dimension
        assert collection_desc["vector_count"] == len(namespaces) * num_vectors
        assert collection_desc["size"] is not None
        assert collection_desc["size"] > 0

        # Create new index from collection
        index_name2 = generate_index_name("index-from-collection-" + collection_name)
        print(f"Creating index {index_name} from collection {collection_name}...")
        new_index = pc.db.index.create(
            name=index_name2,
            dimension=dimension,
            metric=metric,
            spec=PodSpec(environment=pod_environment, source_collection=collection_name),
            tags=index_tags,
        )
        logger.debug(f"Created index {index_name2} from collection {collection_name}: {new_index}")

        # Wait for the index to be ready
        max_wait = 120
        time_waited = 0
        index_ready = False
        while not index_ready and time_waited < max_wait:
            logger.debug(
                f"Waiting for index {index_name} to be ready. Waited {time_waited} seconds..."
            )
            desc = pc.db.index.describe(name=index_name)
            logger.debug(f"Index {index_name} status: {desc['status']}")
            index_ready = desc["status"]["ready"] == True
            if not index_ready:
                time.sleep(10)
                time_waited += 10
        if not index_ready:
            raise Exception(f"Index {index_name} is not ready after {max_wait} seconds")

        new_index_desc = pc.db.index.describe(name=index_name)
        logger.debug(f"New index desc: {new_index_desc}")
        assert new_index_desc["name"] == index_name
        assert new_index_desc["status"]["ready"] == True

        new_idx = pc.Index(name=index_name)

        # Verify stats reflect the vectors present in the collection
        stats = new_idx.describe_index_stats()
        logger.debug(f"New index stats: {stats}")
        assert stats.total_vector_count == len(namespaces) * num_vectors

        # Verify the vectors from the collection can be fetched
        for namespace in namespaces:
            results = new_idx.fetch(ids=[v[0] for v in vectors], namespace=namespace)
            logger.debug(f"Results for namespace {namespace}: {results}")
            assert len(results.vectors) != 0

        # Verify the vectors from the collection can be queried by id
        for namespace in namespaces:
            for i in range(num_vectors):
                results = new_idx.query(top_k=3, id=str(i), namespace=namespace)
                logger.debug(
                    f"Query results for namespace {namespace} and id {i} in index {index_name2}: {results}"
                )
                assert len(results.matches) == 3

                # Compapre with results from original index
                original_results = idx.query(top_k=3, id=str(i), namespace=namespace)
                logger.debug(
                    f"Original query results for namespace {namespace} and id {i} in index {index_name}: {original_results}"
                )
                assert len(original_results.matches) == 3
                assert original_results.matches[0].id == results.matches[0].id
                assert original_results.matches[1].id == results.matches[1].id
                assert original_results.matches[2].id == results.matches[2].id

        # Cleanup
        attempt_cleanup_collection(pc, collection_name)
        attempt_cleanup_index(pc, index_name)
        attempt_cleanup_index(pc, index_name2)
