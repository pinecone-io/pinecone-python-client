from ..helpers import poll_fetch_for_ids_in_namespace
from pinecone import Vector
from .utils import embedding_values

def setup_data(idx, target_namespace, wait):
    # Upsert without metadata
    idx.upsert(vectors=[
            ('1', embedding_values(2)), 
            ('2', embedding_values(2)),
            ('3', embedding_values(2))
        ], 
        namespace=target_namespace
    )

    # Upsert with metadata
    idx.upsert(vectors=[
            Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
            Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
            Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        ], 
        namespace=target_namespace
    )

    # Upsert with dict
    idx.upsert(vectors=[
            {'id': '7', 'values': embedding_values(2)},
            {'id': '8', 'values': embedding_values(2)},
            {'id': '9', 'values': embedding_values(2)}
        ], 
        namespace=target_namespace
    )

    if wait:
        poll_fetch_for_ids_in_namespace(idx, ids=['1', '2', '3', '4', '5', '6', '7', '8', '9'], namespace=target_namespace)
