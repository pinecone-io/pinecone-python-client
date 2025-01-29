from pinecone import IndexEmbed, EmbedModel, Metric


def test_initialization_required_fields():
    embed = IndexEmbed(model="test-model", field_map={"text": "my_text_field"})

    assert embed.model == "test-model"
    assert embed.field_map == {"text": "my_text_field"}


def test_initialization_with_optional_fields():
    embed = IndexEmbed(
        model="test-model",
        field_map={"text": "my_text_field"},
        metric="cosine",
        read_parameters={"param1": "value1"},
        write_parameters={"param2": "value2"},
    )

    assert embed.model == "test-model"
    assert embed.field_map == {"text": "my_text_field"}
    assert embed.metric == "cosine"
    assert embed.read_parameters == {"param1": "value1"}
    assert embed.write_parameters == {"param2": "value2"}


def test_as_dict_method():
    embed = IndexEmbed(
        model="test-model",
        field_map={"text": "my_text_field"},
        metric="cosine",
        read_parameters={"param1": "value1"},
        write_parameters={"param2": "value2"},
    )
    embed_dict = embed.as_dict()

    expected_dict = {
        "model": "test-model",
        "field_map": {"text": "my_text_field"},
        "metric": "cosine",
        "read_parameters": {"param1": "value1"},
        "write_parameters": {"param2": "value2"},
    }

    assert embed_dict == expected_dict


def test_when_passed_enums():
    embed = IndexEmbed(
        model=EmbedModel.Multilingual_E5_Large,
        field_map={"text": "my_text_field"},
        metric=Metric.COSINE,
    )

    assert embed.model == EmbedModel.Multilingual_E5_Large.value
    assert embed.field_map == {"text": "my_text_field"}
    assert embed.metric == Metric.COSINE.value
