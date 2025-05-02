from tests.integration.helpers import random_string


class TestHasIndex:
    def test_index_exists_success(self, pc, create_sl_index_params):
        name = create_sl_index_params["name"]
        pc.db.index.create(**create_sl_index_params)
        has_index = pc.db.index.has(name)
        assert has_index == True

    def test_index_does_not_exist(self, pc):
        name = random_string(8)
        has_index = pc.db.index.has(name)
        assert has_index == False

    def test_has_index_with_null_index_name(self, pc):
        has_index = pc.db.index.has("")
        assert has_index == False
