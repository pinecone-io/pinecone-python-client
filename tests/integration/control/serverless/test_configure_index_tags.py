import pytest


class TestIndexTags:
    def test_index_tags_none_by_default(self, client, ready_sl_index):
        client.describe_index(name=ready_sl_index)
        assert client.describe_index(name=ready_sl_index).tags is None

    def test_add_index_tags(self, client, ready_sl_index):
        client.configure_index(name=ready_sl_index, tags={"foo": "FOO", "bar": "BAR"})
        assert client.describe_index(name=ready_sl_index).tags.to_dict() == {
            "foo": "FOO",
            "bar": "BAR",
        }

    def test_remove_tags_by_setting_empty_value_for_key(self, client, ready_sl_index):
        client.configure_index(name=ready_sl_index, tags={"foo": "FOO", "bar": "BAR"})
        client.configure_index(name=ready_sl_index, tags={})
        assert client.describe_index(name=ready_sl_index).tags.to_dict() == {
            "foo": "FOO",
            "bar": "BAR",
        }

        client.configure_index(name=ready_sl_index, tags={"foo": ""})
        assert client.describe_index(name=ready_sl_index).tags.to_dict() == {"bar": "BAR"}

    def test_merge_new_tags_with_existing_tags(self, client, ready_sl_index):
        client.configure_index(name=ready_sl_index, tags={"foo": "FOO", "bar": "BAR"})
        client.configure_index(name=ready_sl_index, tags={"baz": "BAZ"})
        assert client.describe_index(name=ready_sl_index).tags.to_dict() == {
            "foo": "FOO",
            "bar": "BAR",
            "baz": "BAZ",
        }

    @pytest.mark.skip(reason="Backend bug filed")
    def test_remove_all_tags(self, client, ready_sl_index):
        client.configure_index(name=ready_sl_index, tags={"foo": "FOO", "bar": "BAR"})
        client.configure_index(name=ready_sl_index, tags={"foo": "", "bar": ""})
        assert client.describe_index(name=ready_sl_index).tags is None
