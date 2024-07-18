import pytest

class TestDeletionProtection:
    def test_deletion_protection(self, client, create_sl_index_params):
        name = create_sl_index_params["name"]
        client.create_index(**create_sl_index_params, deletion_protection=True)
        desc = client.describe_index(name)
        assert desc.deletion_protection == "enabled"
        
        with pytest.raises(Exception) as e:
            client.delete_index(name)
        assert "Deletion protection is enabled for this index" in str(e.value)

        client.configure_index(name, deletion_protection=False)
        desc = client.describe_index(name)
        assert desc.deletion_protection == "disabled"

        client.delete_index(name)