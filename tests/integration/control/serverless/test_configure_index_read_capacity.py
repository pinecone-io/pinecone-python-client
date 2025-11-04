class TestConfigureIndexReadCapacity:
    def test_configure_serverless_index_read_capacity_ondemand(self, client, ready_sl_index):
        """Test configuring a serverless index to use OnDemand read capacity."""
        # Configure to OnDemand (should be idempotent if already OnDemand)
        client.configure_index(name=ready_sl_index, read_capacity={"mode": "OnDemand"})

        # Verify the configuration was applied
        desc = client.describe_index(name=ready_sl_index)
        assert hasattr(desc.spec.serverless, "read_capacity")
        assert desc.spec.serverless.read_capacity.mode == "OnDemand"

    def test_configure_serverless_index_read_capacity_dedicated(self, client, ready_sl_index):
        """Test configuring a serverless index to use Dedicated read capacity."""
        # Configure to Dedicated
        client.configure_index(
            name=ready_sl_index,
            read_capacity={
                "mode": "Dedicated",
                "dedicated": {"node_type": "t1", "scaling": "Manual"},
            },
        )

        # Verify the configuration was applied
        desc = client.describe_index(name=ready_sl_index)
        assert hasattr(desc.spec.serverless, "read_capacity")
        assert desc.spec.serverless.read_capacity.mode == "Dedicated"
        assert desc.spec.serverless.read_capacity.dedicated.node_type == "t1"
        assert desc.spec.serverless.read_capacity.dedicated.scaling == "Manual"

    def test_configure_serverless_index_read_capacity_dedicated_with_manual(
        self, client, ready_sl_index
    ):
        """Test configuring a serverless index to use Dedicated read capacity with manual scaling."""
        # Configure to Dedicated with manual scaling configuration
        client.configure_index(
            name=ready_sl_index,
            read_capacity={
                "mode": "Dedicated",
                "dedicated": {
                    "node_type": "t1",
                    "scaling": "Manual",
                    "manual": {"shards": 1, "replicas": 1},
                },
            },
        )

        # Verify the configuration was applied
        desc = client.describe_index(name=ready_sl_index)
        assert hasattr(desc.spec.serverless, "read_capacity")
        assert desc.spec.serverless.read_capacity.mode == "Dedicated"
        assert desc.spec.serverless.read_capacity.dedicated.node_type == "t1"
        assert desc.spec.serverless.read_capacity.dedicated.scaling == "Manual"
        assert desc.spec.serverless.read_capacity.dedicated.manual.shards == 1
        assert desc.spec.serverless.read_capacity.dedicated.manual.replicas == 1

    def test_configure_serverless_index_read_capacity_from_ondemand_to_dedicated(
        self, client, ready_sl_index
    ):
        """Test changing read capacity from OnDemand to Dedicated."""
        # First configure to OnDemand
        client.configure_index(name=ready_sl_index, read_capacity={"mode": "OnDemand"})
        desc = client.describe_index(name=ready_sl_index)
        assert desc.spec.serverless.read_capacity.mode == "OnDemand"

        # Then change to Dedicated
        client.configure_index(
            name=ready_sl_index,
            read_capacity={
                "mode": "Dedicated",
                "dedicated": {"node_type": "t1", "scaling": "Manual"},
            },
        )
        desc = client.describe_index(name=ready_sl_index)
        assert desc.spec.serverless.read_capacity.mode == "Dedicated"
        assert desc.spec.serverless.read_capacity.dedicated.node_type == "t1"
