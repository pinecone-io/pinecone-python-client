class TestCreateIndexPods:
    def test_create_with_optional_tags(self, client, create_index_params):
        index_name = create_index_params["name"]
        tags = {"foo": "FOO", "bar": "BAR"}
        create_index_params["tags"] = create_index_params

        client.create_index(**create_index_params)

        desc = client.describe_index(name=index_name)
        assert desc.tags.to_dict() == tags
