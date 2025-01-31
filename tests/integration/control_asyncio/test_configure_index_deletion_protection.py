import pytest
from pinecone import DeletionProtection, PineconeAsyncio


@pytest.mark.asyncio
class TestDeletionProtection:
    @pytest.mark.parametrize(
        "dp_enabled,dp_disabled",
        [("enabled", "disabled"), (DeletionProtection.ENABLED, DeletionProtection.DISABLED)],
    )
    async def test_deletion_protection(self, create_sl_index_params, dp_enabled, dp_disabled):
        pc = PineconeAsyncio()
        name = create_sl_index_params["name"]
        await pc.create_index(**create_sl_index_params, deletion_protection=dp_enabled)
        desc = await pc.describe_index(name)
        assert desc.deletion_protection == "enabled"

        with pytest.raises(Exception) as e:
            await pc.delete_index(name)
        assert "Deletion protection is enabled for this index" in str(e.value)

        await pc.configure_index(name, deletion_protection=dp_disabled)
        desc = await pc.describe_index(name)
        assert desc.deletion_protection == "disabled"

        await pc.delete_index(name)
        await pc.close()

    @pytest.mark.parametrize("deletion_protection", ["invalid"])
    async def test_deletion_protection_invalid_options(
        self, create_sl_index_params, deletion_protection
    ):
        pc = PineconeAsyncio()
        with pytest.raises(Exception) as e:
            await pc.create_index(**create_sl_index_params, deletion_protection=deletion_protection)
        assert "deletion_protection must be either 'enabled' or 'disabled'" in str(e.value)
        await pc.close()

    @pytest.mark.parametrize("deletion_protection", ["invalid"])
    async def test_configure_deletion_protection_invalid_options(
        self, create_sl_index_params, deletion_protection
    ):
        pc = PineconeAsyncio()
        with pytest.raises(Exception) as e:
            await pc.create_index(**create_sl_index_params, deletion_protection=deletion_protection)
        assert "deletion_protection must be either 'enabled' or 'disabled'" in str(e.value)
        await pc.close()
