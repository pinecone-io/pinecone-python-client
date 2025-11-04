import pytest
from pinecone import PineconeAsyncio


@pytest.mark.asyncio
class TestConfigureIndexReadCapacity:
    async def test_configure_serverless_index_read_capacity_ondemand(self, ready_sl_index):
        """Test configuring a serverless index to use OnDemand read capacity."""
        pc = PineconeAsyncio()

        # Configure to OnDemand (should be idempotent if already OnDemand)
        await pc.configure_index(name=ready_sl_index, read_capacity={"mode": "OnDemand"})

        # Verify the configuration was applied
        desc = await pc.describe_index(name=ready_sl_index)
        assert hasattr(desc.spec.serverless, "read_capacity")
        assert desc.spec.serverless.read_capacity.mode == "OnDemand"
        await pc.close()

    async def test_configure_serverless_index_read_capacity_dedicated(self, ready_sl_index):
        """Test configuring a serverless index to use Dedicated read capacity."""
        pc = PineconeAsyncio()

        # Configure to Dedicated
        await pc.configure_index(
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
        desc = await pc.describe_index(name=ready_sl_index)
        assert hasattr(desc.spec.serverless, "read_capacity")
        assert desc.spec.serverless.read_capacity.mode == "Dedicated"
        assert desc.spec.serverless.read_capacity.dedicated.node_type == "t1"
        assert desc.spec.serverless.read_capacity.dedicated.scaling == "Manual"
        await pc.close()

    async def test_configure_serverless_index_read_capacity_dedicated_with_manual(
        self, ready_sl_index
    ):
        """Test configuring a serverless index to use Dedicated read capacity with manual scaling."""
        pc = PineconeAsyncio()

        # Configure to Dedicated with manual scaling configuration
        await pc.configure_index(
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
        desc = await pc.describe_index(name=ready_sl_index)
        assert hasattr(desc.spec.serverless, "read_capacity")
        assert desc.spec.serverless.read_capacity.mode == "Dedicated"
        assert desc.spec.serverless.read_capacity.dedicated.node_type == "t1"
        assert desc.spec.serverless.read_capacity.dedicated.scaling == "Manual"
        assert desc.spec.serverless.read_capacity.dedicated.manual.shards == 1
        assert desc.spec.serverless.read_capacity.dedicated.manual.replicas == 1
        await pc.close()

    async def test_configure_serverless_index_read_capacity_from_ondemand_to_dedicated(
        self, ready_sl_index
    ):
        """Test changing read capacity from OnDemand to Dedicated."""
        pc = PineconeAsyncio()

        # First configure to OnDemand
        await pc.configure_index(name=ready_sl_index, read_capacity={"mode": "OnDemand"})
        desc = await pc.describe_index(name=ready_sl_index)
        assert desc.spec.serverless.read_capacity.mode == "OnDemand"

        # Then change to Dedicated
        await pc.configure_index(
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
        desc = await pc.describe_index(name=ready_sl_index)
        assert desc.spec.serverless.read_capacity.mode == "Dedicated"
        assert desc.spec.serverless.read_capacity.dedicated.node_type == "t1"
        await pc.close()
