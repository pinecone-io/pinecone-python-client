from pinecone.core.openapi.db_control.models import DeletionProtection


def test_simple_model_instantiation():
    dp = DeletionProtection(value="enabled")
    assert dp.value == "enabled"

    dp2 = DeletionProtection(value="disabled")
    assert dp2.value == "disabled"

    dp3 = DeletionProtection("enabled")
    assert dp3.value == "enabled"
