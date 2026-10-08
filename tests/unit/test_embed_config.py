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

    def test_to_dict_includes_dimension(self) -> None:
        config = EmbedConfig(model="m", field_map={"text": "t"}, dimension=1024)
        assert config.to_dict()["dimension"] == 1024

    def test_to_dict_omits_unset_dimension(self) -> None:
        config = EmbedConfig(model="m", field_map={"text": "t"})
        assert "dimension" not in config.to_dict()

    def test_to_dict_covers_every_declared_field(self) -> None:
        config = EmbedConfig(
            model="m",
            field_map={"text": "t"},
            dimension=1024,
            metric="cosine",
            read_parameters={"input_type": "query"},
            write_parameters={"input_type": "passage"},
        )
        assert set(config.to_dict()) == set(EmbedConfig.__struct_fields__)

    def test_to_dict_round_trips_a_fully_populated_config(self) -> None:
        config = EmbedConfig(
            model="multilingual-e5-large",
            field_map={"text": "chunk_text"},
            dimension=1024,
            metric="cosine",
            read_parameters={"input_type": "query"},
            write_parameters={"input_type": "passage"},
        )
        assert EmbedConfig(**config.to_dict()) == config

    def test_round_trip_preserves_dimension_without_optional_parameters(self) -> None:
        config = EmbedConfig(model="m", field_map={"text": "t"}, dimension=512)
        assert EmbedConfig(**config.to_dict()).dimension == 512
