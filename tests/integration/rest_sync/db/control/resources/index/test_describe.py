from pinecone import IndexModel


class TestDescribeIndex:
    def test_describe_index_when_ready(self, pc, ready_sl_index, create_index_params):
        description = pc.db.index.describe(name=ready_sl_index)

        assert isinstance(description, IndexModel)
        assert description.name == ready_sl_index
        assert description.dimension == create_index_params["dimension"]
        assert description.metric == create_index_params["metric"]
        assert (
            description.spec.serverless["cloud"]
            == create_index_params["spec"]["serverless"]["cloud"]
        )
        assert (
            description.spec.serverless["region"]
            == create_index_params["spec"]["serverless"]["region"]
        )

        assert isinstance(description.host, str)
        assert description.host != ""
        assert ready_sl_index in description.host

        assert description.status.state == "Ready"
        assert description.status.ready == True

    def test_describe_index_when_not_ready(self, pc, notready_sl_index, create_index_params):
        description = pc.db.index.describe(name=notready_sl_index)

        assert isinstance(description, IndexModel)
        assert description.name == notready_sl_index
        assert description.dimension == create_index_params["dimension"]
        assert description.metric == create_index_params["metric"]
        assert (
            description.spec.serverless["cloud"]
            == create_index_params["spec"]["serverless"]["cloud"]
        )
        assert (
            description.spec.serverless["region"]
            == create_index_params["spec"]["serverless"]["region"]
        )

        assert isinstance(description.host, str)
        assert description.host != ""
        assert notready_sl_index in description.host
