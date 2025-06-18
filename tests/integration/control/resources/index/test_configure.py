class TestConfigureIndexTags:
    def test_add_index_tags(self, pc, ready_sl_index):
        starting_tags = pc.db.index.describe(name=ready_sl_index).tags
        assert "foo" not in starting_tags
        assert "bar" not in starting_tags

        pc.db.index.configure(name=ready_sl_index, tags={"foo": "FOO", "bar": "BAR"})

        found_tags = pc.db.index.describe(name=ready_sl_index).tags.to_dict()
        assert found_tags is not None
        assert found_tags["foo"] == "FOO"
        assert found_tags["bar"] == "BAR"

    def test_remove_tags_by_setting_empty_value_for_key(self, pc, ready_sl_index):
        pc.db.index.configure(name=ready_sl_index, tags={"foo": "FOO", "bar": "BAR"})
        pc.db.index.configure(name=ready_sl_index, tags={})
        found_tags = pc.db.index.describe(name=ready_sl_index).tags.to_dict()
        assert found_tags is not None
        assert found_tags.get("foo", None) == "FOO", "foo should not be removed"
        assert found_tags.get("bar", None) == "BAR", "bar should not be removed"

        pc.db.index.configure(name=ready_sl_index, tags={"foo": ""})
        found_tags2 = pc.db.index.describe(name=ready_sl_index).tags.to_dict()
        assert found_tags2 is not None
        assert found_tags2.get("foo", None) is None, "foo should be removed"
        assert found_tags2.get("bar", None) == "BAR", "bar should not be removed"

    def test_merge_new_tags_with_existing_tags(self, pc, ready_sl_index):
        pc.db.index.configure(name=ready_sl_index, tags={"foo": "FOO", "bar": "BAR"})
        pc.db.index.configure(name=ready_sl_index, tags={"baz": "BAZ"})
        found_tags = pc.db.index.describe(name=ready_sl_index).tags.to_dict()
        assert found_tags is not None
        assert found_tags.get("foo", None) == "FOO", "foo should not be removed"
        assert found_tags.get("bar", None) == "BAR", "bar should not be removed"
        assert found_tags.get("baz", None) == "BAZ", "baz should be added"

    def test_remove_multiple_tags(self, pc, ready_sl_index):
        pc.db.index.configure(name=ready_sl_index, tags={"foo": "FOO", "bar": "BAR"})
        pc.db.index.configure(name=ready_sl_index, tags={"foo": "", "bar": ""})
        found_tags = pc.db.index.describe(name=ready_sl_index).tags.to_dict()
        assert found_tags is not None
        assert found_tags.get("foo", None) is None, "foo should be removed"
        assert found_tags.get("bar", None) is None, "bar should be removed"

    def test_configure_index_embed(self, pc, create_sl_index_params):
        name = create_sl_index_params["name"]
        create_sl_index_params["dimension"] = 1024
        pc.db.index.create(**create_sl_index_params)
        desc = pc.db.index.describe_index(name)
        assert desc.embed is None

        embed_config = {
            "model": "multilingual-e5-large",
            "field_map": {"text": "chunk_text"},
        }
        pc.db.index.configure(name, embed=embed_config)

        desc = pc.db.index.describe_index(name)
        assert desc.embed.model == "multilingual-e5-large"
        assert desc.embed.field_map == {"text": "chunk_text"}
        assert desc.embed.read_parameters == {"input_type": "query", "truncate": "END"}
        assert desc.embed.write_parameters == {
            "input_type": "passage",
            "truncate": "END",
        }
        assert desc.embed.vector_type == "dense"
        assert desc.embed.dimension == 1024
        assert desc.embed.metric == "cosine"

        pc.db.index.delete_index(name)
