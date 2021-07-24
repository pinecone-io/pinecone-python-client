from pinecone.graph import IndexGraph


def test_validate_replicas():
    index = IndexGraph()
    index.gateway_replicas = 2

    try:
        index.validate()
        assert False
    except ValueError as e:
        assert 'aggregator replicas must be the same as gateway replicas' in str(e)


if __name__ == "__main__":
    test_validate_replicas()
