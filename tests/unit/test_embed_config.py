from __future__ import annotations

from pinecone.models.indexes.specs import EmbedConfig


class TestEmbedConfigToDict:
    def test_to_dict_all_fields(self) -> None:
        config = EmbedConfig(
            model="m",
            field_map={"text": "t"},
            metric="cosine",
            read_parameters={"k": 1},
            write_parameters={"j": 2},
        )
        result = config.to_dict()
        assert result == {
            "model": "m",
            "field_map": {"text": "t"},
            "metric": "cosine",
            "read_parameters": {"k": 1},
            "write_parameters": {"j": 2},
        }

    def test_to_dict_defaults_empty_dicts(self) -> None:
        config = EmbedConfig(model="m", field_map={"text": "t"})
        result = config.to_dict()
        assert result["read_parameters"] == {}
        assert result["write_parameters"] == {}

    def test_to_dict_no_metric(self) -> None:
        config = EmbedConfig(model="m", field_map={"text": "t"})
        result = config.to_dict()
        assert "metric" not in result

    def test_to_dict_with_metric(self) -> None:
        config = EmbedConfig(model="m", field_map={"text": "t"}, metric="dotproduct")
        result = config.to_dict()
        assert result["metric"] == "dotproduct"

    def test_embed_config_read_parameters_none_by_default(self) -> None:
        config = EmbedConfig(model="m", field_map={})
        assert config.read_parameters is None

    def test_embed_config_write_parameters_none_by_default(self) -> None:
        config = EmbedConfig(model="m", field_map={})
        assert config.write_parameters is None
