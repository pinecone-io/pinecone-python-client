class TestConfigureIndexEmbed:
    def test_convert_index_to_integrated(self, client, create_sl_index_params):
        name = create_sl_index_params["name"]
        create_sl_index_params["dimension"] = 1024
        client.create_index(**create_sl_index_params)
        desc = client.describe_index(name)
        assert desc.embed is None

        embed_config = {
            "model": "multilingual-e5-large",
            "field_map": {"text": "chunk_text"},
        }
        client.configure_index(name, embed=embed_config)

        desc = client.describe_index(name)
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

        client.delete_index(name)
