import pytest


@pytest.mark.skip(reason="slow")
class TestCreateIndexPods:
    def test_create_with_optional_tags(self, client, create_index_params):
        index_name = create_index_params["name"]
        tags = {"foo": "FOO", "bar": "BAR"}
        create_index_params["tags"] = tags

        client.create_index(**create_index_params)

        desc = client.describe_index(name=index_name)
        assert desc.tags.to_dict() == tags
