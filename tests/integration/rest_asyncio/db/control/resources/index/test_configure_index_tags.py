import pytest
from pinecone import PineconeAsyncio


@pytest.mark.asyncio
class TestIndexTags:
    async def test_add_index_tags(self, ready_sl_index):
        pc = PineconeAsyncio()

        await pc.configure_index(name=ready_sl_index, tags={"foo": "FOO", "bar": "BAR"})
        desc = await pc.describe_index(name=ready_sl_index)
        assert desc.tags.to_dict()["foo"] == "FOO"
        assert desc.tags.to_dict()["bar"] == "BAR"
        await pc.close()

    async def test_remove_tags_by_setting_empty_value_for_key(self, ready_sl_index):
        pc = PineconeAsyncio()

        await pc.configure_index(name=ready_sl_index, tags={"foo": "FOO", "bar": "BAR"})
        await pc.configure_index(name=ready_sl_index, tags={})

        desc = await pc.describe_index(name=ready_sl_index)
        assert desc.tags.to_dict()["foo"] == "FOO"
        assert desc.tags.to_dict()["bar"] == "BAR"

        await pc.configure_index(name=ready_sl_index, tags={"foo": ""})
        desc2 = await pc.describe_index(name=ready_sl_index)
        assert desc2.tags.to_dict()["bar"] == "BAR"
        assert "foo" not in desc2.tags.to_dict()
        await pc.close()

    async def test_merge_new_tags_with_existing_tags(self, ready_sl_index):
        pc = PineconeAsyncio()

        await pc.configure_index(name=ready_sl_index, tags={"foo": "FOO", "bar": "BAR"})
        await pc.configure_index(name=ready_sl_index, tags={"baz": "BAZ"})
        desc = await pc.describe_index(name=ready_sl_index)
        assert desc.tags.to_dict()["foo"] == "FOO"
        assert desc.tags.to_dict()["bar"] == "BAR"
        assert desc.tags.to_dict()["baz"] == "BAZ"
        await pc.close()

    @pytest.mark.skip(reason="Backend bug filed")
    async def test_remove_all_tags(self, ready_sl_index):
        pc = PineconeAsyncio()
        await pc.configure_index(name=ready_sl_index, tags={"foo": "FOO", "bar": "BAR"})
        await pc.configure_index(name=ready_sl_index, tags={"foo": "", "bar": ""})
        desc = await pc.describe_index(name=ready_sl_index)
        assert "foo" not in desc.tags.to_dict()
        assert "bar" not in desc.tags.to_dict()
        await pc.close()
