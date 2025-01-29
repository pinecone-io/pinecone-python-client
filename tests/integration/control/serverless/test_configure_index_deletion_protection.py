import pytest
from pinecone import DeletionProtection


class TestDeletionProtection:
    @pytest.mark.parametrize(
        "dp_enabled,dp_disabled",
        [("enabled", "disabled"), (DeletionProtection.ENABLED, DeletionProtection.DISABLED)],
    )
    def test_deletion_protection(self, client, create_sl_index_params, dp_enabled, dp_disabled):
        name = create_sl_index_params["name"]
        client.create_index(**create_sl_index_params, deletion_protection=dp_enabled)
        desc = client.describe_index(name)
        assert desc.deletion_protection == "enabled"

        with pytest.raises(Exception) as e:
            client.delete_index(name)
        assert "Deletion protection is enabled for this index" in str(e.value)

        client.configure_index(name, deletion_protection=dp_disabled)
        desc = client.describe_index(name)
        assert desc.deletion_protection == "disabled"

        client.delete_index(name)

    @pytest.mark.parametrize("deletion_protection", ["invalid"])
    def test_deletion_protection_invalid_options(
        self, client, create_sl_index_params, deletion_protection
    ):
        with pytest.raises(Exception) as e:
            client.create_index(**create_sl_index_params, deletion_protection=deletion_protection)
        assert "deletion_protection must be either 'enabled' or 'disabled'" in str(e.value)

    @pytest.mark.parametrize("deletion_protection", ["invalid"])
    def test_configure_deletion_protection_invalid_options(
        self, client, create_sl_index_params, deletion_protection
    ):
        with pytest.raises(Exception) as e:
            client.create_index(**create_sl_index_params, deletion_protection=deletion_protection)
        assert "deletion_protection must be either 'enabled' or 'disabled'" in str(e.value)
